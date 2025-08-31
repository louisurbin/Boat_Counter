import cv2
import json
import numpy as np
from scipy.spatial.distance import cdist

# Charger les lignes depuis JSON
with open("./temp/TLC00000_extrait_lines.json", "r") as f:
    data = json.load(f)
lines = []
for line in data["lines"]:
    p1 = tuple(line["p1"])
    p2 = tuple(line["p2"])
    label = line["label"]
    lines.append({"p1": p1, "p2": p2, "label": label})

# Fonction pour savoir de quel côté d’une ligne est un point
def point_side(p1, p2, point):
    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)
    x, y = map(int, point)
    return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

# Tracker très simple par centroides
track_history = {}      # {id: [(cx,cy), ...]}
track_positions = {}    # {id: (cx,cy)}
crossed = {}            # {id: set(labels)}
next_id = 0
max_distance = 80  # pixels

# Compteurs par ligne
line_counts = {line["label"]: 0 for line in lines}

# Background subtractor
cap = cv2.VideoCapture("./temp/masked_video.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    mask = fgbg.apply(frame)
    mask = cv2.medianBlur(mask, 5)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    # Trouver blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    current_centroids = []
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:  # ignorer bruit
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w//2, y + h//2
            current_centroids.append((cx, cy))
            boxes.append((x, y, w, h))

    current_centroids = np.array(current_centroids)

    # Association avec anciens tracks
    if len(track_positions) == 0:
        for centroid in current_centroids:
            track_positions[next_id] = centroid
            track_history[next_id] = [centroid]
            crossed[next_id] = set()
            next_id += 1
    else:
        old_ids = list(track_positions.keys())
        old_centroids = np.array([track_positions[i] for i in old_ids])

        if len(current_centroids) > 0:
            D = cdist(old_centroids, current_centroids)
            row_ind, col_ind = np.where(D < max_distance)

            matched_old = set()
            matched_new = set()
            for r_idx, c_idx in zip(row_ind, col_ind):
                tid = old_ids[r_idx]
                track_positions[tid] = current_centroids[c_idx]
                track_history[tid].append(current_centroids[c_idx])
                matched_old.add(tid)
                matched_new.add(c_idx)

            # Nouveaux objets
            for i, centroid in enumerate(current_centroids):
                if i not in matched_new:
                    track_positions[next_id] = centroid
                    track_history[next_id] = [centroid]
                    crossed[next_id] = set()
                    next_id += 1

    # Dessin
    for tid, centroid in track_positions.items():
        cx, cy = centroid
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"ID {tid}", (cx+10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Vérifier franchissements
        path = track_history[tid]
        if len(path) >= 2:
            prev, curr = path[-2], path[-1]
            for line in lines:
                if line["label"] in crossed[tid]:
                    continue
                side_prev = point_side(line["p1"], line["p2"], prev)
                side_curr = point_side(line["p1"], line["p2"], curr)
                if side_prev * side_curr < 0:
                    line_counts[line["label"]] += 1
                    crossed[tid].add(line["label"])
                    print(f"Track {tid} crossed line {line['label']}")

        # Trajectoire
        if len(path) > 1:
            pts = np.array(path, np.int32).reshape((-1,1,2))
            cv2.polylines(frame, [pts], False, (0,165,255), 2)

    # Dessiner lignes
    for line in lines:
        cv2.line(frame, line["p1"], line["p2"], (255,0,0), 2)
        cv2.putText(frame, f"{line['label']} ({line_counts[line['label']]})",
                    (50, 50+30*list(line_counts.keys()).index(line["label"])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Motion + Tracking", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
