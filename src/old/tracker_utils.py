import numpy as np
import os
import cv2
import math
import json

# -----------------------------
# CROSSINGS
# -----------------------------
def load_lines(lines_json_path):
    """Charge les lignes depuis JSON et calcule leurs vecteurs normaux."""
    lines_info = []
    normals = []
    if not lines_json_path:
        return lines_info, normals

    with open(lines_json_path, "r", encoding="utf-8") as f:
        lines_info = json.load(f).get("lines", [])

    for line in lines_info:
        p1 = np.array(line["p1"], dtype=np.float32)
        p2 = np.array(line["p2"], dtype=np.float32)
        line_vec = p2 - p1
        normal = np.array([-line_vec[1], line_vec[0]])
        normal /= (np.linalg.norm(normal)+1e-8)
        normals.append((p1, normal, line["label"]))
    return lines_info, normals


def update_crossings(objects, prev_centroids, normals, frame_idx, real_fps, crossings_per_id):
    """Met à jour les crossings pour cette frame et renvoie les dicts mis à jour."""
    for oid, bbox in objects.items():
        cx = (bbox[0]+bbox[2])/2
        cy = (bbox[1]+bbox[3])/2
        prev = prev_centroids.get(oid)
        if prev:
            for (p1, normal, label) in normals:
                prod_prev = np.dot(np.array(prev)-p1, normal)
                prod_now = np.dot(np.array([cx, cy])-p1, normal)
                if prod_prev * prod_now < 0:
                    time_sec = frame_idx * real_fps
                    crossings_per_id.setdefault(oid, []).append({"line": label, "time": time_sec})
        prev_centroids[oid] = (cx, cy)
    return crossings_per_id, prev_centroids

# -----------------------------
# CROPS COULEUR
# -----------------------------
def save_color_crops(color_frame, objects, base_temp, save_counts, created_dirs):
    """Sauvegarde les crops de chaque objet dans le dossier correspondant."""
    for oid, bbox in objects.items():
        x1, y1, x2, y2 = map(int, bbox)
        folder = os.path.join(base_temp, str(oid))
        if oid not in created_dirs:
            os.makedirs(folder, exist_ok=True)
            created_dirs.add(oid)
        idx = save_counts.get(oid, 0)
        cv2.imwrite(os.path.join(folder, f"{idx:05d}.png"), color_frame[y1:y2, x1:x2])
        save_counts[oid] = idx+1
    return save_counts, created_dirs

# -----------------------------
# SPLIT MASK WATERSHED
# -----------------------------
def split_mask_watershed(bw, bbox, min_area=100, thresh_rel=0.5, frame_h=None, bottom_margin=20):
    """Découpe un gros blob en plusieurs objets via Watershed."""
    x1, y1, x2, y2 = bbox
    if frame_h and y2 >= frame_h - bottom_margin:
        return []

    roi = bw[y1:y2, x1:x2]
    if roi.size == 0:
        return []

    dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return []

    _, sure_fg = cv2.threshold(dist, thresh_rel * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    k = max(3, int(round(math.sqrt(min_area))))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    sure_bg = cv2.dilate(roi, kernel, iterations=1)
    unknown = cv2.subtract(sure_bg, sure_fg)

    num_markers, markers = cv2.connectedComponents(sure_fg)
    if num_markers <= 1:
        return []

    roi_color = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    markers = markers + 1
    markers[unknown == 255] = 0
    cv2.watershed(roi_color, markers)

    boxes = []
    for lab in range(2, markers.max() + 1):
        mask = (markers == lab).astype('uint8') * 255
        conts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in conts:
            a = cv2.contourArea(c)
            if a < min_area:
                continue
            xx, yy, ww, hh = cv2.boundingRect(c)
            boxes.append((int(x1 + xx), int(y1 + yy), int(x1 + xx + ww), int(y1 + yy + hh)))
    return boxes
