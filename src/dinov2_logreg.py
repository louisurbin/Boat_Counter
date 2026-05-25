#!/usr/bin/env python3
"""
Extraction d'embeddings DINOv2 + entraînement d'un classifieur LogisticRegression.

Usage:
    python src/dinov2_logreg.py --data ./datasets --out_dir ./models

Le script :
 - charge les images via torchvision.datasets.ImageFolder
 - extrait des embeddings avec DINOv2 (torch.hub)
 - entraîne une LogisticRegression (scikit-learn)
 - affiche accuracy, classification_report et matrice de confusion
 - sauvegarde les embeddings et le classifieur dans `out_dir`

Conçu pour tourner sur CPU.
"""
import argparse
import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import random
import matplotlib.pyplot as plt


def get_args():
    p = argparse.ArgumentParser(description="DINOv2 embeddings -> LogisticRegression (CPU)")
    p.add_argument("--data", default="./datasets", help="Path to dataset root (ImageFolder structure)")
    p.add_argument("--out_dir", default="./models", help="Directory to save embeddings and model")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for feature extraction (CPU)")
    p.add_argument("--num_workers", type=int, default=0, help="Dataloader workers (0 for Windows/CPU)")
    p.add_argument("--eval", action="store_true", help="Run interactive evaluation visualization on validation images")
    p.add_argument("--eval_dir", default="./datasets/validation", help="Directory with validation images (ImageFolder) or class subfolders")
    p.add_argument("--model_path", default=None, help="Path to saved model joblib (overrides default in out_dir)")
    p.add_argument("--eval-only", action="store_true", help="Only run evaluation on --eval_dir using saved model; do not (re)train")
    return p.parse_args()


def build_dataloader(data_root, batch_size=8, num_workers=0, exclude_classes=None):
    # Small preprocessing transforms to mitigate exposure issues:
    class CLAHETransform:
        """Apply CLAHE on the L channel in LAB color space."""
        def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
            self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

        def __call__(self, pil_img: Image.Image):
            img = np.array(pil_img)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            L, a, b = cv2.split(lab)
            L = self.clahe.apply(L)
            lab = cv2.merge((L, a, b))
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return Image.fromarray(rgb)

    class GammaTransform:
        """Simple gamma correction (gamma < 1 brightens; >1 darkens)."""
        def __init__(self, gamma=0.9):
            self.gamma = float(gamma)

        def __call__(self, pil_img: Image.Image):
            arr = np.array(pil_img).astype(np.float32) / 255.0
            arr = np.clip(arr ** self.gamma, 0.0, 1.0)
            return Image.fromarray((arr * 255).astype('uint8'))
    

    # Transforms recommended: CLAHE -> Gamma -> Resize -> ToTensor -> Normalize(ImageNet)
    # Use Resize((224,224)) instead of CenterCrop to preserve the full boat crop.
    tf = transforms.Compose([
        CLAHETransform(clipLimit=2.0, tileGridSize=(8, 8)),
        GammaTransform(gamma=0.9),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load full dataset then filter out unwanted classes (e.g., 'arriere','avant','validation')
    ds_full = ImageFolder(root=str(data_root), transform=None)

    if exclude_classes is None:
        exclude_classes = set(['arriere', 'avant', 'validation'])
    else:
        exclude_classes = set(exclude_classes)

    # Build mapping old_idx -> new_idx for classes to keep
    new_class_names = []
    old_to_new = {}
    for old_idx, cname in enumerate(ds_full.classes):
        if cname in exclude_classes:
            continue
        old_to_new[old_idx] = len(new_class_names)
        new_class_names.append(cname)

    # Filter samples and remap labels
    filtered_samples = [(p, old_to_new[idx]) for (p, idx) in ds_full.samples if idx in old_to_new]

    class FilteredFolder(Dataset):
        def __init__(self, samples, transform, loader):
            self.samples = samples
            self.transform = transform
            self.loader = loader

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target

    ds = FilteredFolder(filtered_samples, transform=tf, loader=ds_full.loader)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # attach classes attribute for compatibility
    ds.classes = new_class_names
    return ds, loader


def get_transform():
    """Return the preprocessing transform used for training/evaluation."""
    class CLAHETransform:
        def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
            self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

        def __call__(self, pil_img: Image.Image):
            img = np.array(pil_img)
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            L, a, b = cv2.split(lab)
            L = self.clahe.apply(L)
            lab = cv2.merge((L, a, b))
            rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            return Image.fromarray(rgb)

    class GammaTransform:
        def __init__(self, gamma=0.9):
            self.gamma = float(gamma)

        def __call__(self, pil_img: Image.Image):
            arr = np.array(pil_img).astype(np.float32) / 255.0
            arr = np.clip(arr ** self.gamma, 0.0, 1.0)
            return Image.fromarray((arr * 255).astype('uint8'))

    return transforms.Compose([
        CLAHETransform(clipLimit=2.0, tileGridSize=(8, 8)),
        GammaTransform(gamma=0.9),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_dinov2_model(device):
    print("Chargement du modèle DINOv2 (torch.hub) — cela télécharge le modèle si besoin...")
    try:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement de DINOv2 via torch.hub: {e}\nVérifie la connexion Internet ou installe le modèle manuellement.")
    model.eval()
    model.to(device)
    return model


def load_classifier(joblib_path):
    if not Path(joblib_path).exists():
        raise FileNotFoundError(f"Classifier not found: {joblib_path}")
    data = joblib.load(joblib_path)
    clf = data.get('clf') if isinstance(data, dict) else data
    classes = data.get('classes') if isinstance(data, dict) else None
    return clf, classes


def evaluate_on_evaldir(clf, classes, dinomodel, eval_dir, device, batch_size=8, num_workers=0):
    """Evaluate a saved classifier on images under `eval_dir` and show sample predictions."""
    eval_root = Path(eval_dir)
    if not eval_root.exists():
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")

    tf = get_transform()
    # grid size: 5x7 (max images to display)
    rows, cols = 5, 7
    max_cells = rows * cols
    # Determine whether eval_dir has class subfolders (ImageFolder) or is a flat list of images
    has_subdirs = any((eval_root / p).is_dir() for p in os.listdir(eval_root)) if any(eval_root.iterdir()) else False

    if has_subdirs:
        eval_ds = ImageFolder(str(eval_root), transform=tf)
        eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        X_eval, y_eval = extract_embeddings(dinomodel, eval_loader, device)

        # Predict
        y_pred = clf.predict(X_eval)
        print("Eval accuracy:", accuracy_score(y_eval, y_pred))
        target_names = classes if classes is not None else eval_ds.classes
        print("\nClassification report (eval):\n", classification_report(y_eval, y_pred, target_names=target_names))
        print("Confusion matrix (eval):\n", confusion_matrix(y_eval, y_pred))

        # show samples from labeled eval dataset (sample up to grid capacity)
        total_items = len(eval_ds)
        sample_count = min(max_cells, total_items)
        sample_idxs = random.sample(range(total_items), sample_count)
        sample_paths = [eval_ds.samples[i][0] for i in sample_idxs]
    else:
        # Flat folder of images (no classes). Collect image files.
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        image_paths = [p for p in sorted(eval_root.iterdir()) if p.suffix.lower() in exts and p.is_file()]
        if len(image_paths) == 0:
            raise FileNotFoundError(f"No images found in {eval_dir}")
        sample_paths = random.sample(image_paths, min(max_cells, len(image_paths)))
        target_names = classes  # may be None

    # Display samples in a 5x7 grid (max 35 images) with predictions above images
    # if we have more sample_paths than grid cells, randomly pick max_cells
    if len(sample_paths) > max_cells:
        sample_paths = random.sample(sample_paths, max_cells)
    n = len(sample_paths)
    fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows))
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        if i < n:
            path = sample_paths[i]
            img = Image.open(path).convert('RGB')
            ax.imshow(img)
            # prepare tensor and compute embedding
            img_t = tf(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = dinomodel(img_t)
                if isinstance(emb, (list, tuple)):
                    emb = emb[0]
                if isinstance(emb, dict):
                    emb = next(iter(emb.values()))
                if emb.dim() == 4:
                    emb = emb.mean(dim=(2, 3))
                emb_np = emb.reshape(emb.size(0), -1).cpu().numpy()
            pred = clf.predict(emb_np)[0]
            proba = None
            if hasattr(clf, 'predict_proba'):
                try:
                    proba = clf.predict_proba(emb_np)[0].max()
                except Exception:
                    proba = None
            pred_label = target_names[pred] if (target_names is not None and pred < len(target_names)) else str(pred)
            title = f"pred: {pred_label}"
            if proba is not None:
                title += f" ({proba:.2f})"
            ax.set_title(title)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


def extract_embeddings(model, dataloader, device):
    X_list = []
    y_list = []
    total = len(dataloader.dataset)
    processed = 0
    print(f"Extraction des embeddings pour {total} images (CPU). Batch size={dataloader.batch_size}")
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            out = model(imgs)

            # Robust extraction of embeddings from various output formats
            if isinstance(out, torch.Tensor):
                emb = out
            elif isinstance(out, (list, tuple)):
                emb = out[0]
            elif isinstance(out, dict):
                # try some common keys
                for k in ("x", "feat", "features", "last_hidden_state"):
                    if k in out:
                        emb = out[k]
                        break
                else:
                    emb = next(iter(out.values()))
            else:
                raise RuntimeError(f"Format de sortie du modèle inattendu: {type(out)}")

            # If spatial map, global average pooling
            if emb.dim() == 4:
                emb = emb.mean(dim=(2, 3))

            emb = emb.reshape(emb.size(0), -1).cpu().numpy()
            X_list.append(emb)
            y_list.append(labels.numpy())
            processed += imgs.size(0)
            print(f"  processed {processed}/{total} images", end="\r")
    print("\nExtraction terminée.")
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def main():
    args = get_args()
    data_root = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If requested, run evaluation only (use saved classifier + DINOv2 to embed eval images)
    if getattr(args, 'eval-only', False) or getattr(args, 'eval_only', False):
        model_path = args.model_path if args.model_path is not None else out_dir / 'logreg_dinov2.joblib'
        try:
            clf, saved_classes = load_classifier(model_path)
        except Exception as e:
            print(f"Impossible de charger le classifieur: {e}")
            return
        device = torch.device('cpu')
        dinomodel = load_dinov2_model(device)
        evaluate_on_evaldir(clf, saved_classes, dinomodel, args.eval_dir, device,
                    batch_size=args.batch_size, num_workers=args.num_workers)
        return

    # build dataloader
    ds, loader = build_dataloader(data_root, batch_size=args.batch_size, num_workers=args.num_workers)
    class_names = ds.classes
    print(f"Classes trouvées: {class_names}")

    device = torch.device('cpu')
    model = load_dinov2_model(device)

    X, y = extract_embeddings(model, loader, device)
    print(f"Embeddings shape: {X.shape}, labels shape: {y.shape}")

    # save embeddings for reuse
    np.savez_compressed(out_dir / 'embeddings.npz', X=X, y=y, classes=class_names)
    print(f"Embeddings sauvegardés: {out_dir / 'embeddings.npz'}")

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Split: train={X_train.shape[0]} samples, test={X_test.shape[0]} samples")

    # train logistic regression
    print("Entraînement du classifieur LogisticRegression...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # evaluations
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    print("Train accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test accuracy:", accuracy_score(y_test, y_test_pred))
    print("\nClassification report (test):\n", classification_report(y_test, y_test_pred, target_names=class_names))
    print("Confusion matrix (test):\n", confusion_matrix(y_test, y_test_pred))

    # save classifier
    joblib.dump({'clf': clf, 'classes': class_names}, out_dir / 'logreg_dinov2.joblib')
    print(f"Classifieur sauvegardé: {out_dir / 'logreg_dinov2.joblib'}")


if __name__ == '__main__':
    main()
