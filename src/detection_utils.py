import cv2
import numpy as np

# =============================================================================
# Constantes
# =============================================================================

# Paramètres de détection
MIN_AREA = 100                 # Aire minimale (px^2) pour considérer un contour comme valide
NMS_IOU = 0.3                  # Seuil IoU pour la suppression des non-maxima (fusion doublons)

# =============================================================================
# Fonctions utilitaires
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
