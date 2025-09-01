import os
import csv
import argparse
from pathlib import Path
from PIL import Image, ImageOps
import torchvision.transforms as T

def preprocess_image(img, size=128, mode='crop', fill_color=(0,0,0)):
	"""
	Prétraite une PIL.Image img vers un carré (size x size).
	mode:
	 - 'crop' : scale to cover -> center crop (zoom)
	 - 'pad'  : scale to fit -> pad symétriquement (letterbox)
	 - 'stretch': direct resize (déformation)
	"""
	w, h = img.size
	if mode == 'stretch':
		return img.resize((size, size), Image.LANCZOS)
	if mode == 'crop':
		# scale so the smaller side >= size -> then center crop
		scale = max(size / w, size / h)
		new_w = int(round(w * scale))
		new_h = int(round(h * scale))
		img_resized = img.resize((new_w, new_h), Image.LANCZOS)
		left = (new_w - size) // 2
		top = (new_h - size) // 2
		return img_resized.crop((left, top, left + size, top + size))
	if mode == 'pad':
		# scale so the larger side <= size -> then pad to square
		scale = min(size / w, size / h)
		new_w = int(round(w * scale))
		new_h = int(round(h * scale))
		img_resized = img.resize((new_w, new_h), Image.LANCZOS)
		dx = (size - new_w)
		dy = (size - new_h)
		padding = (dx // 2, dy // 2, dx - dx // 2, dy - dy // 2)
		return ImageOps.expand(img_resized, border=padding, fill=fill_color)
	raise ValueError(f"Unknown mode: {mode}")

def is_image_file(p: Path):
	try:
		s = p.suffix.lower()
		return s in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
	except Exception:
		return False

def preprocess_dataset(input_root, output_root, size=128, mode='crop', overwrite=False, save_format='JPEG', quality=90):
	"""
	Parcours input_root/<class>/*.jpg et sauve sous output_root/<class>/img.jpg après preprocess.
	Retourne la liste des tuples (relpath, label, orig_w, orig_h).
	"""
	input_root = Path(input_root)
	output_root = Path(output_root)
	output_root.mkdir(parents=True, exist_ok=True)
	records = []
	for class_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
		label = class_dir.name
		out_class_dir = output_root / label
		out_class_dir.mkdir(parents=True, exist_ok=True)
		for img_path in sorted(class_dir.iterdir()):
			if not is_image_file(img_path):
				continue
			out_path = out_class_dir / img_path.name
			if out_path.exists() and not overwrite:
				# still record metadata
				try:
					with Image.open(img_path) as im:
						orig_w, orig_h = im.size
				except Exception:
					continue
				records.append((str(out_path.relative_to(output_root)), label, orig_w, orig_h))
				continue
			try:
				with Image.open(img_path) as im:
					im = im.convert('RGB')
					orig_w, orig_h = im.size
					out_im = preprocess_image(im, size=size, mode=mode)
					# ensure extension matches save_format
					out_path = out_class_dir / (img_path.stem + ('.jpg' if save_format.upper() in ('JPG','JPEG') else img_path.suffix))
					out_im.save(out_path, format=save_format, quality=quality)
					records.append((str(out_path.relative_to(output_root)), label, orig_w, orig_h))
			except Exception as e:
				# skip corrupted file
				print(f"Warning: failed processing {img_path}: {e}")
				continue
	return records

def write_csv(records, output_root, csv_name='labels.csv'):
	output_root = Path(output_root)
	csv_path = output_root / csv_name
	with open(csv_path, 'w', newline='', encoding='utf-8') as f:
		w = csv.writer(f)
		w.writerow(['filepath', 'label', 'orig_width', 'orig_height'])
		for row in records:
			w.writerow(row)
	return csv_path

# --- Remplacement des anciennes fonctions d'augmentation manuelle ---
def get_train_transforms(size=128):
	"""
	Return a torchvision.transforms.Compose for on-the-fly training augmentation.
	Examples: RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor, Normalize.
	"""
	return T.Compose([
		# random crop with scale & ratio jitter (acts like zoom + crop)
		T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
		# random horizontal flip
		T.RandomHorizontalFlip(p=0.5),
		# small random rotations
		T.RandomRotation(degrees=15),
		# color variations
		T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
		# convert to tensor and normalize
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

def get_eval_transforms(size=128):
	"""
	Return a torchvision.transforms.Compose for evaluation / validation.
	Deterministic: resize -> center crop -> ToTensor -> Normalize.
	"""
	return T.Compose([
		T.Resize((size, size)),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])

# Optional helper to create a DataLoader using ImageFolder and the transforms above.
def make_dataloader_from_preprocessed(root_dir, batch_size=32, size=128, train=True, num_workers=4, shuffle=True):
	"""
	Simple helper: root_dir should be an ImageFolder-style preprocessed dataset.
	If train=True applies training augmentation, otherwise evaluation transforms.
	"""
	transforms = get_train_transforms(size) if train else get_eval_transforms(size)
	ds = __import__('torchvision.datasets', fromlist=['ImageFolder']).ImageFolder(root_dir, transform=transforms)
	from torch.utils.data import DataLoader
	loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle if train else False, num_workers=num_workers, pin_memory=True)
	return loader, ds.classes

def parse_args():
	p = argparse.ArgumentParser(description="Preprocess dataset folders into fixed-size images for LeNet")
	p.add_argument("--input", "-i", required=True, help="Root folder containing class subfolders (./datasets/...)")
	p.add_argument("--output", "-o", required=True, help="Output root for preprocessed images (ImageFolder layout)")
	p.add_argument("--size", type=int, default=128, help="Output square size (default 128)")
	p.add_argument("--mode", choices=['crop','pad','stretch'], default='crop', help="Preprocess mode (crop=zoom, pad=letterbox, stretch=resize)")
	p.add_argument("--overwrite", action='store_true', help="Overwrite existing processed images")
	p.add_argument("--format", default='JPEG', choices=['JPEG','PNG'], help="Save format for processed images")
	p.add_argument("--quality", type=int, default=90, help="JPEG quality (if JPEG)")
	return p.parse_args()

def main():
	args = parse_args()
	records = preprocess_dataset(args.input, args.output, size=args.size, mode=args.mode, overwrite=args.overwrite, save_format=args.format, quality=args.quality)
	csv_path = write_csv(records, args.output)
	print(f"Processed {len(records)} images. Labels CSV: {csv_path}")

if __name__ == "__main__":
	main()