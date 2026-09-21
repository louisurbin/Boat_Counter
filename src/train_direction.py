#!/usr/bin/env python3
"""
Entraînement d'un classifieur de direction (avant/arriere/none) avec DINOv2 + LogisticRegression.

Usage:
    python src/train_direction.py --data ./datasets --out_dir ./models

Le script :
 - charge UNIQUEMENT les images des dossiers 'avant/', 'arriere/' et 'none/' (ignore les autres sous-dossiers)
   'none' correspond à des faux positifs (bruit), équivalent de la classe 'noise' pour le type de bateau
 - extrait des embeddings avec DINOv2 (torch.hub)
 - entraîne une LogisticRegression (scikit-learn)
 - sauvegarde le classifieur dans out_dir/direction_dinov2.joblib

Conçu pour tourner sur CPU.
"""

import argparse
import os
import warnings
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Désactiver les warnings xFormers
warnings.filterwarnings("ignore", message="xFormers is not available")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

from transforms import transform as default_transform


def load_dinov2_model(device):
    """Charge le modèle DINOv2 depuis torch.hub."""
    print("Chargement du modèle DINOv2 (torch.hub)...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    model.eval()
    model.to(device)
    return model


def main():
    """Fonction principale du script d'entraînement de la détection de direction."""
    parser = argparse.ArgumentParser(
        description="Entraînement classifieur de direction (DINOv2 + LogisticRegression)"
    )
    parser.add_argument(
        "--data", 
        default="../../datasets", 
        help="Chemin vers le dossier parent contenant avant/, arriere/ et none/"
    )
    parser.add_argument(
        "--out_dir", 
        default="../../models", 
        help="Dossier de sortie pour le modèle"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Taille des batches pour l'extraction des embeddings"
    )
    parser.add_argument(
        "--num_workers",
        type=int, 
        default=0,
        help="Nombre de workers pour DataLoader (0 pour Windows/CPU)"
    )
    args = parser.parse_args()

    # Préparation des dossiers
    data_root = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vérification du dataset (recherche avant/ et arriere/ dans data_root, none/ optionnel)
    avant_dir = data_root / "avant"
    arriere_dir = data_root / "arriere"
    none_dir = data_root / "none"
    
    if not avant_dir.exists() or not arriere_dir.exists():
        raise FileNotFoundError(
            f"Le dataset doit contenir les dossiers 'avant' et 'arriere' dans {data_root}. "
            f"Trouvé: avant={avant_dir.exists()}, arriere={arriere_dir.exists()}"
        )
    
    # Compter les images
    num_avant = len(list(avant_dir.glob("*.jpg"))) + len(list(avant_dir.glob("*.png")))
    num_arriere = len(list(arriere_dir.glob("*.jpg"))) + len(list(arriere_dir.glob("*.png")))
    num_none = len(list(none_dir.glob("*.jpg"))) + len(list(none_dir.glob("*.png"))) if none_dir.exists() else 0
    print(f"Dataset: {num_avant} images 'avant', {num_arriere} images 'arriere', {num_none} images 'none' (total: {num_avant + num_arriere + num_none})")

    # Créer un dataset personnalisé qui ne contient que avant/, arriere/ et none/
    # (ImageFolder chargerait tous les sous-dossiers, ce qu'on ne veut pas)
    
    class DirectionDataset(Dataset):
        """Dataset personnalisé pour charger uniquement les images des dossiers avant/, arriere/ et none/."""
        def __init__(self, data_root, transform=None):
            """Initialise le dataset avec les images de direction."""
            self.transform = transform
            self.samples = []
            
            # Charger uniquement les images de avant/ (label 0), arriere/ (label 1) et none/ (label 2)
            for class_name, label in [("avant", 0), ("arriere", 1), ("none", 2)]:
                class_dir = data_root / class_name
                if not class_dir.exists():
                    continue
                
                # Trouver toutes les images dans ce dossier
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for img_path in class_dir.glob(ext):
                        self.samples.append((str(img_path), label))
            
            self.classes = ["avant", "arriere", "none"]
        
        def __len__(self):
            """Retourne le nombre d'échantillons."""
            return len(self.samples)
        
        def __getitem__(self, idx):
            """Retourne un échantillon à l'index donné."""
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label
    
    dataset = DirectionDataset(data_root, transform=default_transform)
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )

    print(f"Classes trouvées: {dataset.classes}")

    # Chargement de DINOv2
    device = torch.device("cpu")
    model = load_dinov2_model(device)

    # Extraction des embeddings
    X_list = []
    y_list = []
    total = len(dataset)
    
    print(f"Extraction des embeddings pour {total} images (CPU)...")
    
    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device)
            out = model(imgs)
            
            # Gestion des différents formats de sortie de DINOv2
            if isinstance(out, torch.Tensor):
                emb = out
            elif isinstance(out, (list, tuple)):
                emb = out[0]
            elif isinstance(out, dict):
                # Essayer des clés courantes
                for k in ("x", "feat", "features", "last_hidden_state"):
                    if k in out:
                        emb = out[k]
                        break
                else:
                    emb = next(iter(out.values()))
            else:
                raise RuntimeError(f"Format de sortie inattendu: {type(out)}")
            
            # Global Average Pooling si on a une carte spatiale
            if emb.dim() == 4:
                emb = emb.mean(dim=(2, 3))
            
            # Aplatir et passer en numpy
            emb = emb.reshape(emb.size(0), -1).cpu().numpy()
            X_list.append(emb)
            y_list.append(labels.numpy())
            
            print(f"  Traité batch {batch_idx + 1}/{len(loader)} ({min((batch_idx + 1) * args.batch_size, total)}/{total})", end="\r")
    
    print("\nExtraction terminée.")
    
    # Concatenation
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    print(f"Embeddings shape: {X.shape}, labels shape: {y.shape}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y, 
        random_state=42
    )
    print(f"Split: train={X_train.shape[0]}, test={X_test.shape[0]}")

    # Entraînement du classifieur
    print("\nEntraînement du classifieur LogisticRegression...")
    clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)

    # Évaluation
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    
    print(f"\nPrécision entraînement: {train_acc:.4f}")
    print(f"Précision test: {test_acc:.4f}")
    print("\nRapport de classification:")
    print(classification_report(y_test, clf.predict(X_test), target_names=dataset.classes))

    # Sauvegarde du classifieur
    model_path = out_dir / 'direction_dinov2.joblib'
    joblib.dump({'clf': clf, 'classes': dataset.classes}, model_path)
    print(f"\nClassifieur de direction sauvegardé: {model_path}")


if __name__ == "__main__":
    main()
