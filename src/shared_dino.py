"""
Module partagé pour le traitement DINOv2.
Centralise le chargement du modèle DINOv2 et des classifieurs (direction + type de bateau)
pour éviter la duplication des calculs d'embeddings.

Utilisation :
    from shared_dino import (
        get_device,
        load_dinomodel,
        load_direction_clf,
        load_boat_type_clf,
        process_id_folder,
    )
"""

import os
import warnings
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import joblib

# Désactiver les warnings xFormers
warnings.filterwarnings("ignore", message="xFormers is not available")

from transforms import transform

# Seuils de probabilité
NONE_PROB_THRESHOLD = 0.5  # Pour direction ('none' = faux positif)
NOISE_PROB_THRESHOLD = 0.5  # Pour type de bateau ('noise' = faux positif)

# --- Variables globales pour le cache des modèles ---
_dinomodel = None
_direction_clf = None
_direction_classes = None
_boat_type_clf = None
_boat_type_classes = None


def get_device():
    """Retourne le device optimal (GPU si disponible, sinon CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dinomodel():
    """Charge le modèle DINOv2 (mis en cache pour éviter de recharger)."""
    global _dinomodel
    if _dinomodel is None:
        print("[shared_dino] Chargement de DINOv2...")
        device = get_device()
        _dinomodel = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        _dinomodel.eval()
        _dinomodel.to(device)
        print(f"[shared_dino] DINOv2 chargé sur {device}")
    return _dinomodel


def get_models_dir():
    """Retourne le chemin vers le dossier models (relatif au projet)."""
    # Le dossier models est à la racine du projet (au même niveau que src/)
    script_dir = Path(__file__).parent  # src/
    models_dir = script_dir.parent / "models"  # Boat_Counter/models/
    return models_dir


def load_direction_clf():
    """Charge le classifieur de direction (mis en cache)."""
    global _direction_clf, _direction_classes
    if _direction_clf is None:
        models_dir = get_models_dir()
        model_path = models_dir / "direction_dinov2.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"[shared_dino] Modèle de direction non trouvé: {model_path}."
            )
        data = joblib.load(model_path)
        _direction_clf = data['clf']
        _direction_classes = data['classes']
        print(f"[shared_dino] Classifieur de direction chargé (classes: {_direction_classes})")
    return _direction_clf, _direction_classes


def load_boat_type_clf():
    """Charge le classifieur de type de bateau (mis en cache)."""
    global _boat_type_clf, _boat_type_classes
    if _boat_type_clf is None:
        models_dir = get_models_dir()
        model_path = models_dir / "boat_type_dinov2.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"[shared_dino] Modèle de type de bateau non trouvé: {model_path}."
            )
        data = joblib.load(model_path)
        _boat_type_clf = data.get('clf') if isinstance(data, dict) else data
        _boat_type_classes = data.get('classes') if isinstance(data, dict) else None
        print(f"[shared_dino] Classifieur de type de bateau chargé (classes: {_boat_type_classes})")
    return _boat_type_clf, _boat_type_classes


def load_images_from_folder(folder_path):
    """
    Charge toutes les images d'un dossier et applique les transformations.
    
    Args:
        folder_path (str): Chemin vers le dossier contenant les images.
    
    Returns:
        torch.Tensor: Batch d'images transformées (shape: [N, 3, 224, 224])
        list: Liste des noms de fichiers valides.
    """
    if not os.path.exists(folder_path):
        return None, []
    
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]
    
    if not image_files:
        return None, []
    
    images = []
    valid_files = []
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(transform(img))
            valid_files.append(img_file)
        except Exception as e:
            print(f"[shared_dino] Erreur lors du chargement de {img_path}: {e}")
            continue
    
    if not images:
        return None, []
    
    return torch.stack(images), valid_files


@torch.no_grad()
def extract_embeddings(dinomodel, batch):
    """
    Extrait les embeddings DINOv2 à partir d'un batch d'images.
    
    Args:
        dinomodel: Modèle DINOv2 chargé.
        batch (torch.Tensor): Batch d'images (shape: [N, 3, 224, 224]).
    
    Returns:
        numpy.ndarray: Embeddings (shape: [N, embedding_dim]).
    """
    device = next(dinomodel.parameters()).device
    batch = batch.to(device)
    
    out = dinomodel(batch)
    
    # Gestion des différents formats de sortie
    if isinstance(out, torch.Tensor):
        emb = out
    elif isinstance(out, (list, tuple)):
        emb = out[0]
    elif isinstance(out, dict):
        emb = next(iter(out.values()))
    else:
        raise RuntimeError(f"[shared_dino] Format de sortie inattendu: {type(out)}")
    
    if emb.dim() == 4:
        emb = emb.mean(dim=(2, 3))
    
    emb = emb.reshape(emb.size(0), -1).cpu().numpy()
    return emb


def classify_direction(embeddings, clf, classes):
    """
    Classifie la direction à partir des embeddings.
    
    Args:
        embeddings (numpy.ndarray): Embeddings DINOv2 (shape: [N, embedding_dim]).
        clf: Classifieur de direction.
        classes: Liste des classes de direction.
    
    Returns:
        tuple: (direction_class, direction_prob)
            direction_class: int (0=avant, 1=arriere, 2=none)
            direction_prob: float (probabilité de la classe prédite)
    """
    none_idx = classes.index("none") if "none" in classes else -1
    
    # Prédiction avec le classifieur
    probs = clf.predict_proba(embeddings)  # Shape: (N, C)
    
    # Filtrage des crops 'none'
    if none_idx >= 0:
        non_none_mask = probs[:, none_idx] < NONE_PROB_THRESHOLD
        if non_none_mask.sum() == 0:
            # Toutes les images sont 'none' → l'ID est un faux positif
            mean_prob = np.mean(probs, axis=0)
            return none_idx, float(mean_prob[none_idx])
        probs = probs[non_none_mask]
    
    # Moyenne sur les images restantes
    mean_prob = np.mean(probs, axis=0)
    if none_idx >= 0:
        mean_prob[none_idx] = 0.0
    
    final_class = int(np.argmax(mean_prob))
    final_prob = float(mean_prob[final_class])
    
    return final_class, final_prob


def classify_boat_type(embeddings, clf, classes):
    """
    Classifie le type de bateau à partir des embeddings.
    
    Args:
        embeddings (numpy.ndarray): Embeddings DINOv2 (shape: [N, embedding_dim]).
        clf: Classifieur de type de bateau.
        classes: Liste des classes de type de bateau.
    
    Returns:
        tuple: (boat_type_class, boat_type_prob)
            boat_type_class: int (index de la classe) ou -1 si vide
            boat_type_prob: torch.Tensor (probabilités moyennes)
    """
    if embeddings.size == 0:
        num_classes = len(classes) if classes is not None else 0
        mean_prob = np.zeros(num_classes)
        return -1, mean_prob
    
    # Prédiction avec le classifieur
    probs = clf.predict_proba(embeddings)  # Shape: (N, C)
    
    # Filtrage des crops 'noise'
    noise_class_idx = classes.index("noise") if (classes is not None and "noise" in classes) else -1
    
    if noise_class_idx >= 0:
        non_noise_mask = probs[:, noise_class_idx] < NOISE_PROB_THRESHOLD
        if non_noise_mask.sum() == 0:
            # Toutes les images sont du bruit
            mean_prob = np.mean(probs, axis=0)
            return noise_class_idx, mean_prob
        probs = probs[non_noise_mask]
    
    # Moyenne sur les images restantes
    mean_prob = np.mean(probs, axis=0)
    if noise_class_idx >= 0:
        mean_prob[noise_class_idx] = 0.0
    
    predicted_class = int(np.argmax(mean_prob))
    return predicted_class, mean_prob


@torch.no_grad()
def process_id_folder(folder_path):
    """
    Traite un dossier d'ID : charge les images, calcule les embeddings DINO,
    et classifie à la fois la direction et le type de bateau.
    
    Args:
        folder_path (str): Chemin vers le dossier contenant les crops d'un ID.
    
    Returns:
        dict: {
            'direction_class': int (0=avant, 1=arriere, 2=none),
            'direction_prob': float,
            'boat_type_class': int ou -1,
            'boat_type_prob': numpy.ndarray,
            'num_images': int,
        }
    """
    # Charger les modèles (une seule fois grâce au cache)
    dinomodel = load_dinomodel()
    direction_clf, direction_classes = load_direction_clf()
    boat_type_clf, boat_type_classes = load_boat_type_clf()
    
    # Charger les images
    batch, valid_files = load_images_from_folder(folder_path)
    
    if batch is None or len(valid_files) == 0:
        num_classes_dir = len(direction_classes) if direction_classes else 0
        num_classes_type = len(boat_type_classes) if boat_type_classes else 0
        return {
            'direction_class': 2 if "none" in direction_classes else 0,
            'direction_prob': 0.0,
            'boat_type_class': -1,
            'boat_type_prob': np.zeros(num_classes_type),
            'num_images': 0,
        }
    
    # UN SEUL forward DINO pour toutes les images
    embeddings = extract_embeddings(dinomodel, batch)
    
    # Classifier direction et type
    direction_class, direction_prob = classify_direction(embeddings, direction_clf, direction_classes)
    boat_type_class, boat_type_prob = classify_boat_type(embeddings, boat_type_clf, boat_type_classes)
    
    return {
        'direction_class': direction_class,
        'direction_prob': direction_prob,
        'boat_type_class': boat_type_class,
        'boat_type_prob': boat_type_prob,
        'num_images': len(valid_files),
    }


def write_crossings_file(folder_path, gate_label, direction_class, direction_prob, frame_idx, seconds, 
                         boat_type_class, boat_type_prob, boat_type_classes, min_prob_threshold=0.5):
    """
    Écrit le fichier crossings.txt pour un ID.
    
    Format :
        Ligne 1: {gate_label}\t{direction}\t{prob}\t{frame_idx}\t{seconds}
        Ligne 2: {boat_type} {prob} OU "noise" OU "empty"
    
    Args:
        folder_path: Chemin vers le dossier de l'ID.
        gate_label: Label de la gate.
        direction_class: Classe de direction (0=avant, 1=arriere, 2=none).
        direction_prob: Probabilité de la direction.
        frame_idx: Index de la première frame.
        seconds: Timestamp en secondes.
        boat_type_class: Classe de type de bateau.
        boat_type_prob: Probabilités du type (array).
        boat_type_classes: Liste des classes de type de bateau.
        min_prob_threshold: Seuil minimal de probabilité pour accepter un type.
    """
    crossings_path = os.path.join(folder_path, 'crossings.txt')
    
    # Convertir direction_class au format attendu par LGC
    # 0=avant → -1, 1=arriere → 1, 2=none → 0
    if direction_class == 2:
        direction_out = 0
    elif direction_class == 1:
        direction_out = 1
    else:
        direction_out = -1
    
    # Ligne 1: Direction
    line1 = f"{gate_label}\t{direction_out}\t{direction_prob:.4f}\t{frame_idx}\t{int(seconds)}\n"
    
    # Ligne 2: Type de bateau
    if boat_type_class < 0:
        line2 = "empty\n"
    else:
        # Vérifier si c'est 'noise' ou probabilité insuffisante
        is_noise_class = (boat_type_class < len(boat_type_prob) and 
                          boat_type_class < len(boat_type_classes) and 
                          boat_type_classes[boat_type_class].lower() == "noise")
        if is_noise_class or (boat_type_prob[boat_type_class] < min_prob_threshold):
            line2 = "noise\n"
        else:
            boat_type_name = (boat_type_classes[boat_type_class] 
                             if boat_type_class < len(boat_type_classes) 
                             else str(boat_type_class))
            line2 = f"{boat_type_name} {boat_type_prob[boat_type_class]:.4f}\n"
    
    with open(crossings_path, 'w', encoding='utf-8') as f:
        f.write(line1)
        f.write(line2)


@torch.no_grad()
def process_all_ids(ids_root, first_crops, real_fps=1/3, batch_size=8):
    """
    Traite les IDs par batchs pour réduire l'overhead CUDA/CPU.
    Chaque batch = 1 forward DINO pour batch_size IDs (défaut: 8).
    
    Args:
        ids_root (str): Chemin vers le dossier racine des extractions (ex: temp/extractions).
        first_crops (dict): Dictionnaire {(gate_label, oid): first_frame_idx}.
        real_fps (float): FPS réel pour convertir frame_idx en secondes.
        batch_size (int): Nombre d'IDs à traiter par batch (défaut: 8).
    
    Returns:
        dict: Statistiques (nombre d'IDs traités, temps d'exécution, etc.).
    """
    dinomodel = load_dinomodel()
    direction_clf, direction_classes = load_direction_clf()
    boat_type_clf, boat_type_classes = load_boat_type_clf()
    
    stats = {'total_ids': 0, 'successful_ids': 0, 'total_images': 0}
    id_list = sorted(first_crops.items())
    
    # Traiter par batchs
    for i in range(0, len(id_list), batch_size):
        batch_ids = id_list[i:i + batch_size]
        all_batches = []  # Liste des tensors d'images pour chaque ID du batch
        batch_meta = []   # Métadonnées: (oid_dir, gate_label, frame_idx, num_images)
        
        # Collecter les images de tous les IDs du batch
        for (gate_label, oid), frame_idx in batch_ids:
            oid_dir = os.path.join(ids_root, f"{gate_label}_id_{oid}")
            
            if not os.path.isdir(oid_dir):
                continue
            
            batch, valid_files = load_images_from_folder(oid_dir)
            
            if batch is None or len(valid_files) == 0:
                # Créer un crossings.txt vide pour cet ID
                seconds = frame_idx / real_fps
                write_crossings_file(
                    oid_dir, gate_label, 2, 0.0, frame_idx, seconds,
                    -1, np.array([]), boat_type_classes, 0.5
                )
                stats['total_ids'] += 1
                continue
            
            all_batches.append(batch)
            batch_meta.append((oid_dir, gate_label, frame_idx, len(valid_files)))
            stats['total_images'] += len(valid_files)
        
        if not all_batches:
            continue
        
        # Concaténer tous les crops du batch en UN SEUL grand tensor
        global_batch = torch.cat(all_batches, dim=0)  # Shape: [total_crops, 3, 224, 224]
        
        # UN SEUL forward DINO pour TOUT le batch
        embeddings = extract_embeddings(dinomodel, global_batch)  # Shape: [total_crops, emb_dim]
        
        # Découper les embeddings par ID et classer
        start_idx = 0
        for oid_dir, gate_label, frame_idx, num_images in batch_meta:
            end_idx = start_idx + num_images
            id_embeddings = embeddings[start_idx:end_idx]
            
            # Classifier direction et type
            direction_class, direction_prob = classify_direction(
                id_embeddings, direction_clf, direction_classes
            )
            boat_type_class, boat_type_prob = classify_boat_type(
                id_embeddings, boat_type_clf, boat_type_classes
            )
            
            # Écrire crossings.txt
            seconds = frame_idx / real_fps
            write_crossings_file(
                oid_dir, gate_label, direction_class, direction_prob,
                frame_idx, seconds, boat_type_class, boat_type_prob,
                boat_type_classes, 0.5
            )
            
            start_idx = end_idx
            stats['total_ids'] += 1
            stats['successful_ids'] += 1
    
    return stats
