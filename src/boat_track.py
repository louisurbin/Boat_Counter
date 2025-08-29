from ultralytics import YOLO
import cv2
import json
import numpy as np
from scipy.spatial.distance import cdist

# Load the model
model = YOLO("yolo11n.pt")

# Load lines from JSON
with open("./temp/TLC00000_extrait_lines.json", "r") as f:
    data = json.load(f)
lines = []
for line in data["lines"]:
    p1 = tuple(line["p1"])
    p2 = tuple(line["p2"])
    label = line["label"]
    lines.append({"p1": p1, "p2": p2, "label": label})

# Helper function to check which side of a line a point is on
def point_side(p1, p2, point):
    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)
    x, y = map(int, point)
    return (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)

# Centroid-based tracker
track_history = {}  # {track_id: list of (cx, cy)}
track_positions = {}  # {track_id: current centroid}
next_id = 0
max_distance = 90  # pixels

# Line crossing counters
line_counts = {line["label"]: 0 for line in lines}
crossed = {}  # {track_id: set([crossed_line_labels])}

# Run YOLO predictions
results = model.predict(
    source="./temp/masked_video.mp4",
    conf=0.2,
    classes=[8],  # only boats
    stream=True
)

for r in results:
    frame = r.orig_img.copy()
    current_centroids = []

    for line in lines:
        cv2.line(frame, line["p1"], line["p2"], (255,0,0), 2)

    # Compute current centroids
    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        current_centroids.append((cx, cy))

    current_centroids = np.array(current_centroids)

    # Associate centroids with existing tracks
    if len(track_positions) == 0:
        # Initialize tracks
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

            # New objects
            for i, centroid in enumerate(current_centroids):
                if i not in matched_new:
                    track_positions[next_id] = centroid
                    track_history[next_id] = [centroid]
                    crossed[next_id] = set()
                    next_id += 1

    # Draw tracks and bounding boxes
    # Build a reverse mapping: centroid -> track_id
    centroid_to_id = {tuple(pos): tid for tid, pos in track_positions.items()}
    for idx, box in enumerate(r.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.tolist())
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        tid = centroid_to_id.get((cx, cy))
        if tid is None:
            continue
        conf = float(r.boxes.conf[idx])

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        label = f"boat {tid} ({conf:.2f})"
        cv2.putText(frame, label, (x1, y1-10),
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

    cv2.imshow("Centroid Tracker", frame)
    if cv2.getWindowProperty("Centroid Tracker", cv2.WND_PROP_VISIBLE) < 1:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

import matplotlib.pyplot as plt

# After the main loop (after cv2.destroyAllWindows())
# line_counts already contains the number of boats per line

labels = list(line_counts.keys())
counts = [line_counts[label] for label in labels]

plt.figure(figsize=(8,5))
plt.bar(labels, counts, color='blue')
plt.xlabel("Lines")
plt.ylabel("Number of boats")
plt.title("Number of boats that crossed each line")
plt.show()
