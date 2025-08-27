import cv2
import numpy as np
import json
from ultralytics import YOLO
import os


class CentroidTracker():
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = dict()
        self.disappeared = dict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - inputCentroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects

# Load YOLO model
yolo_model = YOLO('yolo11n.pt')

# Load lines from JSON
with open('./temp/TLC00010_extrait_lines.json', 'r') as f:
    data = json.load(f)
    lines = data.get('lines', [])

# Initialize CentroidTracker
ct = CentroidTracker(maxDisappeared=2)
track_history = {}
crossed = {}
line_counts = {line["label"]: 0 for line in lines}

def point_side(p1, p2, point):
    x1, y1 = map(int, p1)
    x2, y2 = map(int, p2)
    x, y = map(int, point)
    return (x2 - x1)*(y - y1) - (y2 - y1)*(x - x1)

results = yolo_model.predict(
    source="./temp/masked_video.mp4",
    conf=0.12,
    classes=[8],
    stream=True,
    imgsz=1024
)

for r in results:
    frame = r.orig_img.copy()
    rects = []
    for box in r.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        rects.append((x1, y1, x2, y2))
    objects = ct.update(rects)

    # Draw lines
    for line in lines:
        cv2.line(frame, tuple(line["p1"]), tuple(line["p2"]), (255,0,0), 2)

    # Draw and track
    for objectID, centroid in objects.items():
        if objectID not in track_history:
            track_history[objectID] = []
            crossed[objectID] = set()
        track_history[objectID].append(tuple(centroid))
        cv2.circle(frame, tuple(centroid), 4, (0,255,0), -1)
        cv2.putText(frame, f"ID {objectID}", (centroid[0]-10, centroid[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        # Draw trajectory
        if len(track_history[objectID]) > 1:
            pts = np.array(track_history[objectID], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, (0, 165, 255), 2)
        # Check line crossings
        path = track_history[objectID]
        if len(path) >= 2:
            prev, curr = path[-2], path[-1]
            for line in lines:
                if line["label"] in crossed[objectID]:
                    continue
                side_prev = point_side(line["p1"], line["p2"], prev)
                side_curr = point_side(line["p1"], line["p2"], curr)
                if side_prev * side_curr < 0:
                    line_counts[line["label"]] += 1
                    crossed[objectID].add(line["label"])
                    print(f"Object {objectID} crossed line {line['label']}")
                    cv2.putText(frame,
                                f"Count {line['label']}: {line_counts[line['label']]}",
                                (50, 50 + 30 * list(line_counts.keys()).index(line["label"])),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9,
                                (0,0,255),
                                2)
    cv2.imshow("Centroid Tracker", frame)
    key = cv2.waitKey(1)
    # Stoppe si la fenêtre est fermée ou si 'q' est pressé
    if cv2.getWindowProperty("Centroid Tracker", cv2.WND_PROP_VISIBLE) < 1 or (key & 0xFF == ord('q')):
        break
cv2.destroyAllWindows()
