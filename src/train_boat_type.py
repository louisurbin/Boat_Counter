#!/usr/bin/env python3
"""
Entraînement du classifieur de type de bateau avec DINOv2 + LogisticRegression.

Usage:
    python src/train_boat_type.py --data ./datasets --out_dir ./models

Le script :
 - charge les images via torchvision.datasets.ImageFolder
 - extrait des embeddings avec DINOv2 (torch.hub)
 - entraîne une LogisticRegression (scikit-learn)
 - affiche accuracy, classification_report et matrice de confusion
 - sauvegarde le classifieur dans `out_dir`

Conçu pour tourner sur CPU.
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

from transforms import transform as default_transform


def get_args():
    """Parse les arguments de la ligne de commande."""
    p = argparse.ArgumentParser(description="DINOv2 embeddings -> LogisticRegression (CPU)")
    p.add_argument("--data", default="../datasets", help="Path to dataset root (ImageFolder structure)")
    p.add_argument("--out_dir", default="../../models", help="Directory to save the model")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for feature extraction (CPU)")
    p.add_argument("--num_workers", type=int, default=0, help="Dataloader workers (0 for Windows/CPU)")
    return p.parse_args()


def build_dataloader(data_root, batch_size=8, num_workers=0, exclude_classes=None):
    """
    Construit un DataLoader à partir du dossier dataset, en excluant certaines classes.
    
    Args:
        data_root: Chemin vers le dossier racine du dataset (structure ImageFolder)
        batch_size: Taille des batches
        num_workers: Nombre de workers pour le DataLoader
        exclude_classes: Liste des classes à exclure (par défaut: ['arriere', 'avant', 'none'])
    
    Returns:
        tuple: (dataset, dataloader)
    """
    # Charger le dataset complet puis filtrer les classes indésirables
    ds_full = ImageFolder(root=str(data_root), transform=None)

    if exclude_classes is None:
        exclude_classes = set(['arriere', 'avant', 'none'])
    else:
        exclude_classes = set(exclude_classes)

    # Construire la mapping old_idx -> new_idx pour les classes à garder
    new_class_names = []
    old_to_new = {}
    for old_idx, cname in enumerate(ds_full.classes):
        if cname in exclude_classes:
            continue
        old_to_new[old_idx] = len(new_class_names)
        new_class_names.append(cname)

    # Filtrer les échantillons et remapper les labels
    filtered_samples = [(p, old_to_new[idx]) for (p, idx) in ds_full.samples if idx in old_to_new]

    class FilteredFolder(Dataset):
        """Dataset personnalisé pour charger uniquement les classes sélectionnées."""
        def __init__(self, samples, transform, loader):
            """Initialise le dataset avec les échantillons filtrés."""
            self.samples = samples
            self.transform = transform
            self.loader = loader

        def __len__(self):
            """Retourne le nombre d'échantillons."""
            return len(self.samples)

        def __getitem__(self, index):
            """Retourne un échantillon à l'index donné."""
            path, target = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample, target

    ds = FilteredFolder(filtered_samples, transform=default_transform, loader=ds_full.loader)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # Attacher l'attribut classes pour la compatibilité
    ds.classes = new_class_names
    return ds, loader


def load_dinov2_model(device):
    """Charge le modèle DINOv2 depuis torch.hub."""
    print("Chargement du modèle DINOv2 (torch.hub) — cela télécharge le modèle si besoin...")
    try:
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement de DINOv2 via torch.hub: {e}\nVérifie la connexion Internet ou installe le modèle manuellement.")
    model.eval()
    model.to(device)
    return model


def extract_embeddings(model, dataloader, device):
    """
    Extrait les embeddings pour toutes les images du DataLoader.
    
    Args:
        model: Modèle DINOv2
        dataloader: DataLoader contenant les images
        device: Appareil (cpu/cuda)
    
    Returns:
        tuple: (X, y) où X est le tableau des embeddings et y le tableau des labels
    """
    X_list = []
    y_list = []
    total = len(dataloader.dataset)
    processed = 0
    print(f"Extraction des embeddings pour {total} images (CPU). Batch size={dataloader.batch_size}")
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            out = model(imgs)

            # Extraction robuste des embeddings à partir de divers formats de sortie
            if isinstance(out, torch.Tensor):
                emb = out
            elif isinstance(out, (list, tuple)):
                emb = out[0]
            elif isinstance(out, dict):
                # Essayer quelques clés courantes
                for k in ("x", "feat", "features", "last_hidden_state"):
                    if k in out:
                        emb = out[k]
                        break
                else:
                    emb = next(iter(out.values()))
            else:
                raise RuntimeError(f"Format de sortie du modèle inattendu: {type(out)}")

            # Si carte spatiale, pooling moyenne globale
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
    """Fonction principale du script d'entraînement."""
    args = get_args()
    data_root = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Construire le dataloader
    ds, loader = build_dataloader(data_root, batch_size=args.batch_size, num_workers=args.num_workers)
    class_names = ds.classes
    print(f"Classes trouvées: {class_names}")

    device = torch.device('cpu')
    model = load_dinov2_model(device)

    X, y = extract_embeddings(model, loader, device)
    print(f"Embeddings shape: {X.shape}, labels shape: {y.shape}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Split: train={X_train.shape[0]} samples, test={X_test.shape[0]} samples")

    # Entraînement du classifieur LogisticRegression
    print("Entraînement du classifieur LogisticRegression...")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Évaluations
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    print("Train accuracy:", accuracy_score(y_train, y_train_pred))
    print("Test accuracy:", accuracy_score(y_test, y_test_pred))
    print("\nClassification report (test):\n", classification_report(y_test, y_test_pred, target_names=class_names))
    print("Confusion matrix (test):\n", confusion_matrix(y_test, y_test_pred))

    # Sauvegarder le classifieur
    joblib.dump({'clf': clf, 'classes': class_names}, out_dir / 'boat_type_dinov2.joblib')
    print(f"Classifieur sauvegardé: {out_dir / 'boat_type_dinov2.joblib'}")


if __name__ == '__main__':
    main()
