import cv2
import numpy as np
import os
import shutil

# =============================================================================
# CONSTANTES
# =============================================================================

# Paramètres de détection
MIN_AREA = 100                 # Aire minimale (px^2) pour considérer un contour comme valide
REAL_FPS = 1/5                 # Fréquence réelle (1 image toutes les 5 secondes)
NMS_IOU = 0.3                  # Seuil IoU pour la suppression des non-maxima (fusion doublons détection)

# Paramètres de tracking 
MAX_DISTANCE = 150              # Distance max (px) pour associer une détection à un track (None = pas de limite)
NEW_IOU = 0.5                  # Seuil IoU pour éviter de créer un ID sur un objet existant
MIN_TRACK_LENGTH = 3            # Durée de vie min pour valider trajectoire et intervalle entre vérifs de crossings

# Paramètre de crop
MIN_SIDE_CROP = 30              # Taille min (px) d'un côté pour sauvegarder un crop

# =============================================================================
# FONCTIONS UTILITAIRES 
# =============================================================================

def bbox_area(b):
    """Calcule l'aire d'une boîte (x1, y1, x2, y2)."""
    width = max(0, int(b[2]) - int(b[0]))
    height = max(0, int(b[3]) - int(b[1]))
    return width * height

def bbox_iou(a, b):
    """Calcule l'Intersection over Union (IoU) entre deux boîtes."""
    # Coordonnées de l'intersection
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih
    if inter == 0:
        return 0.0

    union = bbox_area(a) + bbox_area(b) - inter
    return float(inter) / float(union) if union > 0 else 0.0

def nms(boxes, scores=None):
    """
    Non-Maximum Suppression (NMS) pour supprimer les boîtes chevauchantes.
    boxes: liste de (x1, y1, x2, y2)
    scores: liste de scores (optionnel, sinon utilise l'aire)
    """
    if not boxes:
        return []

    boxes_np = np.array(boxes, dtype=np.float32)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 2]
    y2 = boxes_np[:, 3]

    # Convertir les boîtes au format (x, y, w, h) pour cv2.dnn.NMSBoxes
    boxes_cv = np.zeros((len(boxes), 4), dtype=np.float32)
    boxes_cv[:, 0] = x1
    boxes_cv[:, 1] = y1
    boxes_cv[:, 2] = x2 - x1
    boxes_cv[:, 3] = y2 - y1

    # Si aucun score fourni, on priorise les plus grandes boîtes
    if scores is None:
        scores = (x2 - x1) * (y2 - y1)

    # Utiliser cv2.dnn.NMSBoxes
    indices = cv2.dnn.NMSBoxes(boxes_cv.tolist(), scores, NMS_IOU, 0.0)

    # Convertir les indices en liste
    if len(indices) > 0:
        return indices.flatten().tolist()
    else:
        return []

def detect_rectangles(frame):
    """
    Détecte les rectangles blancs sur fond noir (image binaire).
    """
    if frame.shape[2] == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame

    contours, _ = cv2.findContours(gray_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    scores = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        box = (int(x), int(y), int(x + w), int(y + h))

        candidates.append(box)
        scores.append(area)

    # Suppression des doublons (NMS)
    keep_idx = nms(candidates, scores=scores)
    return [candidates[i] for i in keep_idx]

def check_crossing(p1, p2, prev_pt, curr_pt):
    """
    Vérifie si le segment [prev_pt, curr_pt] croise le segment [p1, p2].
    Retourne (True, sign) ou (False, 0).
    """
    # Vecteur ligne
    line_vec = np.array(p2) - np.array(p1)
    # Normale
    normal = np.array([-line_vec[1], line_vec[0]])
    norm_val = np.linalg.norm(normal)
    if norm_val == 0: return False, 0
    normal = normal / norm_val

    # Projections
    vec_prev = np.array(prev_pt) - np.array(p1)
    vec_curr = np.array(curr_pt) - np.array(p1)

    prod_prev = np.dot(normal, vec_prev)
    prod_curr = np.dot(normal, vec_curr)

    # Changement de signe = traversée 
    if prod_prev * prod_curr < 0:
        # Convention: monte = +1 (y diminue)
        sign = 1 if curr_pt[1] < prev_pt[1] else -1
        return True, sign
    return False, 0

def save_crops(color_src, frame_annotations, crossings_per_id, base_temp, completed):
    """
    Extrait et sauvegarde les crops des objets qui ont croisé une ligne dans la vidéo couleur.
    """
    cap_color = cv2.VideoCapture(color_src)

    if not cap_color.isOpened():
        print("Attention: Vidéo couleur non disponible pour les crops.")
        return

    print("Extraction des vignettes...")
    c_idx = 0
    save_counts = {}

    while True:
        ret, frame = cap_color.read()
        if not ret: break

        if c_idx < len(frame_annotations):
            anns = frame_annotations[c_idx]
            h, w = frame.shape[:2]

            for oid, bbox in anns.items():
                # On ne garde que les objets qui ont croisé une ligne et dont la trajectoire est suffisamment longue
                if oid not in crossings_per_id or oid not in completed:
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if (x2-x1) > MIN_SIDE_CROP and (y2-y1)> MIN_SIDE_CROP: # Taille min du crop
                    crop = frame[y1:y2, x1:x2]
                    oid_dir = os.path.join(base_temp, str(oid))
                    os.makedirs(oid_dir, exist_ok=True)
                    cnt = save_counts.get(oid, 0)
                    fname = f"{c_idx:06d}_{cnt}.jpg"
                    cv2.imwrite(os.path.join(oid_dir, fname), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    save_counts[oid] = cnt + 1
            
        c_idx += 1
    cap_color.release()
    
    # Nettoyage des objets sans crops sauvegardés
    for oid in list(completed):
        if save_counts.get(oid, 0) == 0:
            completed.pop(oid, None)
            crossings_per_id.pop(oid, None)

            oid_dir = os.path.join(base_temp, str(oid))
            if os.path.exists(oid_dir):
                shutil.rmtree(oid_dir)

def export_crossings_data(crossings_per_id, base_temp, completed):
    """
    Exporte les données de franchissement de lignes.
    """
    print("Export des données...")
    for oid, crossings in crossings_per_id.items():
        # Vérifier si la trajectoire est suffisamment longue
        if oid in completed:
            oid_dir = os.path.join(base_temp, str(oid))
            os.makedirs(oid_dir, exist_ok=True)

            with open(os.path.join(oid_dir, "crossings.txt"), "w", encoding="utf-8") as f:
                for (label, sign, f_idx) in crossings:
                    # Calcul du temps réel basé sur REAL_FPS
                    seconds = f_idx / REAL_FPS
                    direction = "+1" if sign > 0 else "-1"
                    f.write(f"{label}\t{direction}\t{f_idx}\t{int(seconds)}\n")