import os
import csv
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import cv2

def is_image_file(p: Path):
	try:
		s = p.suffix.lower()
		return s in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
	except Exception:
		return False

def cutout_and_paste_black(pil_img, size=128, seg_thresh=100, min_area=300, margin=8):
	"""
	Segment main object and paste centered on black square of given size.
	Returns PIL.Image (RGB).
	"""
	# Convert to numpy BGR for OpenCV
	img_rgb = np.array(pil_img.convert("RGB"))
	img_bgr = img_rgb[..., ::-1].copy()
	h, w = img_bgr.shape[:2]

	# Grayscale + threshold
	gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
	_, th = cv2.threshold(gray, seg_thresh, 255, cv2.THRESH_BINARY)

	# Morphological cleanup to remove speckle
	k = max(3, int(round(min(h, w) * 0.01)))
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
	th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
	th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

	# Find contours and pick largest
	contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	if not contours:
		# fallback: center-resize onto black
		canvas = Image.new("RGB", (size, size), (0,0,0))
		resized = pil_img.resize((size, size), Image.LANCZOS)
		canvas.paste(resized, (0,0))
		return canvas

	areas = [cv2.contourArea(c) for c in contours]
	max_idx = int(np.argmax(areas))
	if areas[max_idx] < min_area:
		# fallback
		canvas = Image.new("RGB", (size, size), (0,0,0))
		resized = pil_img.resize((size, size), Image.LANCZOS)
		canvas.paste(resized, (0,0))
		return canvas

	cnt = contours[max_idx]
	x, y, bw, bh = cv2.boundingRect(cnt)

	# apply margin and clamp
	x0 = max(0, x - margin)
	y0 = max(0, y - margin)
	x1 = min(w, x + bw + margin)
	y1 = min(h, y + bh + margin)

	crop_bgr = img_bgr[y0:y1, x0:x1]
	crop_rgb = crop_bgr[..., ::-1]
	crop_pil = Image.fromarray(crop_rgb)

	# Resize preserving aspect ratio so the object fits within size
	cw, ch = crop_pil.size
	scale = min(size / cw, size / ch)
	new_w = max(1, int(round(cw * scale)))
	new_h = max(1, int(round(ch * scale)))
	resized = crop_pil.resize((new_w, new_h), Image.LANCZOS)

	# Paste centered on black canvas
	canvas = Image.new("RGB", (size, size), (0,0,0))
	offset_x = (size - new_w) // 2
	offset_y = (size - new_h) // 2
	canvas.paste(resized, (offset_x, offset_y))
	return canvas

def process_dataset_cutout(input_root, output_root, size=128, seg_thresh=100, min_area=300, margin=8, overwrite=False, save_format='JPEG', quality=90):
	input_root = Path(input_root)
	output_root = Path(output_root)
	output_root.mkdir(parents=True, exist_ok=True)
	records = []
	for class_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
		label = class_dir.name
		out_class = output_root / label
		out_class.mkdir(parents=True, exist_ok=True)
		for img_path in sorted(class_dir.iterdir()):
			if not is_image_file(img_path):
				continue
			out_path = out_class / img_path.name
			if out_path.exists() and not overwrite:
				# still record metadata from original
				try:
					with Image.open(img_path) as im:
						ow, oh = im.size
				except Exception:
					continue
				records.append((str(out_path.relative_to(output_root)), label, ow, oh))
				continue
			try:
				with Image.open(img_path) as im:
					im = im.convert('RGB')
					ow, oh = im.size
					out_im = cutout_and_paste_black(im, size=size, seg_thresh=seg_thresh, min_area=min_area, margin=margin)
					ext = '.jpg' if save_format.upper() in ('JPG','JPEG') else img_path.suffix
					out_file = out_class / (img_path.stem + ext)
					out_im.save(out_file, format=save_format, quality=quality)
					records.append((str(out_file.relative_to(output_root)), label, ow, oh))
			except Exception as e:
				print(f"Warning: failed {img_path}: {e}")
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

def interactive_process(input_folder, output_root, classes_root, size=128, overwrite=False, save_format='JPEG', quality=90, delete_after=True):
	"""
	Interactive loop:
	 - shows each image in input_folder (flat)
	 - user draws ROI with cv2.selectROI
	 - user chooses class from classes_root subfolders (or creates new)
	 - crop is pasted onto black canvas size x size and saved under output_root/<class>/
	 - if delete_after True, delete source image after successful save
	"""
	input_folder = Path(input_folder)
	output_root = Path(output_root)
	classes_root = Path(classes_root) if classes_root else None

	# gather class names from classes_root (folder names)
	class_names = []
	if classes_root and classes_root.exists():
		class_names = [p.name for p in sorted(classes_root.iterdir()) if p.is_dir()]
	print(f"Detected classes ({len(class_names)}): {class_names}")

	# ensure output root
	output_root.mkdir(parents=True, exist_ok=True)

	# collect image files (flat)
	img_files = [p for p in sorted(input_folder.iterdir()) if is_image_file(p)]
	if not img_files:
		print("No image files found in", input_folder)
		return []

	records = []
	for img_path in img_files:
		print(f"\nProcessing: {img_path.name} ({img_path})")
		# load with cv2 for interactive display
		cv_img = cv2.imread(str(img_path))
		if cv_img is None:
			print("  Failed to load with cv2, skipping.")
			continue
		# show image and let user select ROI
		win = "Select ROI - draw box then ENTER/SPACE, or press c to cancel"
		cv2.namedWindow(win, cv2.WINDOW_NORMAL)
		cv2.imshow(win, cv_img)
		cv2.waitKey(1)
		r = cv2.selectROI(win, cv_img, showCrosshair=True, fromCenter=False)
		cv2.destroyWindow(win)
		x, y, w, h = r
		if w == 0 or h == 0:
			# cancelled or no selection
			choice = input("No ROI selected. [s]kip / [a]uto cutout / [q]uit: ").strip().lower()
			if choice == 'q':
				print("Exiting interactive session.")
				break
			if choice == 'a':
				# use automatic cutout on entire image
				try:
					with Image.open(img_path) as im:
						im = im.convert('RGB')
						out_im = cutout_and_paste_black(im, size=size)
				except Exception as e:
					print("  Auto cutout failed:", e)
					continue
			else:
				print("Skipping image.")
				continue
		else:
			# crop selected region and create PIL.Image
			crop_bgr = cv_img[int(y):int(y+h), int(x):int(x+w)]
			crop_rgb = crop_bgr[..., ::-1]
			crop_pil = Image.fromarray(crop_rgb)
			# paste on black canvas centered, preserving aspect ratio
			cw, ch = crop_pil.size
			scale = min(size / cw, size / ch)
			new_w = max(1, int(round(cw * scale)))
			new_h = max(1, int(round(ch * scale)))
			resized = crop_pil.resize((new_w, new_h), Image.LANCZOS)
			canvas = Image.new("RGB", (size, size), (0,0,0))
			offset_x = (size - new_w) // 2
			offset_y = (size - new_h) // 2
			canvas.paste(resized, (offset_x, offset_y))
			out_im = canvas

		# choose class
		if not class_names:
			print("No classes found in classes_root; will ask for class name to create.")
		while True:
			if class_names:
				print("Choose class index or type new class name:")
				for i, name in enumerate(class_names):
					print(f"  [{i}] {name}")
				ch = input(f"Class (0-{len(class_names)-1}) or new name: ").strip()
			else:
				ch = input("Enter class name to create: ").strip()
			if ch == '':
				print("Empty input, please enter a class.")
				continue
			# index?
			if ch.isdigit() and class_names:
				ix = int(ch)
				if 0 <= ix < len(class_names):
					class_name = class_names[ix]
					break
				else:
					print("Index out of range.")
					continue
			# new name
			class_name = ch
			# create class folder if not exists (both in classes_root for reference and in output_root)
			if classes_root:
				(Path(classes_root) / class_name).mkdir(parents=True, exist_ok=True)
			if class_name not in class_names:
				class_names.append(class_name)
			break

		# save image to output_root/class_name with same stem (avoid collision)
		out_dir = output_root / class_name
		out_dir.mkdir(parents=True, exist_ok=True)
		out_ext = '.jpg' if save_format.upper() in ('JPG','JPEG') else img_path.suffix
		out_file = out_dir / (img_path.stem + out_ext)
		suffix_idx = 1
		while out_file.exists() and not overwrite:
			out_file = out_dir / f"{img_path.stem}_{suffix_idx}{out_ext}"
			suffix_idx += 1
		try:
			out_im.save(out_file, format=save_format, quality=quality)
			print(f"Saved to {out_file}")
			records.append((str(out_file.relative_to(output_root)), class_name, int(img_path.stat().st_size), 0))
			if delete_after:
				try:
					os.remove(img_path)
					print("Deleted source", img_path)
				except Exception as e:
					print("Failed to delete source:", e)
		except Exception as e:
			print("Failed to save:", e)
			continue

	# write labels CSV
	if records:
		csv_path = write_csv(records, output_root)
		print("Wrote labels CSV:", csv_path)
	return records

def parse_args():
	p = argparse.ArgumentParser(description="Cutout boats and save 128x128 on black background")
	p.add_argument("--input", "-i", required=True, help="Root folder with images (flat) or class subfolders")
	p.add_argument("--output", "-o", required=True, help="Output root for cutout images (ImageFolder layout)")
	p.add_argument("--size", type=int, default=128, help="Output square size (default 128)")
	p.add_argument("--seg-thresh", type=int, default=100, help="Threshold for segmentation (0-255) used in automatic mode")
	p.add_argument("--min-area", type=int, default=300, help="Minimum contour area to keep (automatic mode)")
	p.add_argument("--margin", type=int, default=8, help="Margin around bounding box (px)")
	p.add_argument("--overwrite", action='store_true', help="Overwrite existing outputs")
	p.add_argument("--format", default='JPEG', choices=['JPEG','PNG'], help="Save format")
	p.add_argument("--quality", type=int, default=90, help="JPEG quality")
	# new interactive options
	p.add_argument("--interactive", action="store_true", help="Run interactive manual crop mode")
	p.add_argument("--classes-root", default="", help="Path containing class subfolders (used to list/select classes); optional")
	p.add_argument("--delete-after", action="store_true", help="Delete source image after successful processing (interactive mode)")
	return p.parse_args()

def main():
	args = parse_args()
	# if interactive, run interactive_process on a flat folder
	if args.interactive:
		records = interactive_process(args.input, args.output, args.classes_root, size=args.size, overwrite=args.overwrite, save_format=args.format, quality=args.quality, delete_after=args.delete_after)
		# CSV already written by interactive_process
		return

	# otherwise fallback to batch automatic processing (existing behavior)
	records = process_dataset_cutout(args.input, args.output, size=args.size, seg_thresh=args.seg_thresh, min_area=args.min_area, margin=args.margin, overwrite=args.overwrite, save_format=args.format, quality=args.quality)
	csv_path = write_csv(records, args.output)
	print(f"Processed {len(records)} images. Labels CSV: {csv_path}")

if __name__ == "__main__":
	main()
