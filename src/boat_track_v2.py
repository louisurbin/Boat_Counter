from ultralytics import YOLO
import cv2
import json
import numpy as np
from sort import Sort  # Assure-toi d'avoir sort.py dans src/
import os

# Load the model
model = YOLO("yolo11n.pt")

# Load lines from JSON
with open("./temp/TLC00010_extrait_lines.json", "r") as f:
    data = json.load(f)
lines = []
for line in data["lines"]:
    p1 = tuple(line["p1"])
    p2 = tuple(line["p2"])
    label = line["label"]
    lines.append({"p1": p1, "p2": p2, "label": label})

def point_side(p1, p2, point):
    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)
    x, y = map(int, point)
    return (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.1)

track_history = {}  # {track_id: list of (cx, cy)}
crossed = {}        # {track_id: set([crossed_line_labels])}
line_counts = {line["label"]: 0 for line in lines}

# Run YOLO predictions
results = model.predict(
    source="./temp/masked_video.mp4",
    conf=0.1,  # seuil abaissé pour petits objets
    classes=[8],  # only boats
    stream=True,
    #imgsz=1024
)

for r in results:
    frame = r.orig_img.copy()
    dets = []
    for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
        # Pas de filtrage par taille ni par confiance
        x1, y1, x2, y2 = box.tolist()
        dets.append([x1, y1, x2, y2, float(conf)])
    dets = np.array(dets)
    # Update tracker
    dets = np.array(dets)
    if dets.shape[0] == 0:
        dets = np.empty((0, 5))
    tracks = tracker.update(dets)
    # tracks: [x1, y1, x2, y2, id]

    # Draw lines
    for line in lines:
        cv2.line(frame, line["p1"], line["p2"], (255,0,0), 2)

    for track in tracks:
        x1, y1, x2, y2, tid = track
        tid = int(tid)
        cx, cy = int((x1 + x2)//2), int((y1 + y2)//2)
        if tid not in track_history:
            track_history[tid] = []
            crossed[tid] = set()
        track_history[tid].append((cx, cy))

        # Draw bounding box and label with confidence
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        # Affiche la confiance sur chaque box
        conf = None
        # Recherche la confiance correspondante (optionnel, car on a déjà la liste dets)
        for d in dets:
            if int(d[0]) == int(x1) and int(d[1]) == int(y1) and int(d[2]) == int(x2) and int(d[3]) == int(y2):
                conf = d[4]
                break
        label = f"boat {tid}"
        if conf is not None:
            label += f" ({conf:.2f})"
        cv2.putText(frame, label, (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Check line crossings
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
                    print(f"Boat {tid} crossed line {line['label']}")
                    cv2.putText(frame,
                                f"Count {line['label']}: {line_counts[line['label']]}",
                                (50, 50 + 30 * list(line_counts.keys()).index(line["label"])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0,0,255),
                                2)

    # Draw trajectories
    for tid, path in track_history.items():
        if len(path) > 1:
            pts = np.array(path, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, (0, 165, 255), 2)  # orange

    cv2.imshow("SORT Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

import matplotlib.pyplot as plt

labels = list(line_counts.keys())
counts = [line_counts[label] for label in labels]

plt.figure(figsize=(8,5))
plt.bar(labels, counts, color='blue')
plt.xlabel("Lines")
plt.ylabel("Number of boats")
plt.title("Number of boats that crossed each line")
plt.show()