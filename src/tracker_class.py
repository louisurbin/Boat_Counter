import cv2
import numpy as np
import math
from scipy.optimize import linear_sum_assignment
from tracker_utils import bbox_area, bbox_iou, MAX_DISAPPEARED, MAX_DISTANCE, NEW_IOU, MIN_TRACK_LENGTH, MIN_AREA

class SortTracker:
    """Tracker simple basé sur les centroïdes et une prédiction de vitesse linéaire."""
    def __init__(self):
        self.next_object_id = 0
        self.objects = {}       # id -> bbox
        self.centroids = {}     # id -> (cx, cy)
        self.velocities = {}    # id -> (vx, vy)
        self.disappeared = {}   # id -> frames since last seen
        self.trajectories = {}  # id -> list of (frame_idx, bbox, centroid)
        self.completed = {}     # id -> list of (frame_idx, bbox, centroid) (finished tracks)

    def register(self, bbox, frame_idx=None):
        # Vérifications de base (taille)
        if bbox_area(bbox) < MIN_AREA:
            return

        # Vérifier chevauchement avec objets existants
        for ob in self.objects.values():
            if bbox_iou(bbox, ob) > NEW_IOU:
                return

        # Enregistrement
        oid = self.next_object_id
        self.objects[oid] = bbox
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        self.centroids[oid] = (cx, cy)
        self.velocities[oid] = (0.0, 0.0)
        self.disappeared[oid] = 0
        self.trajectories[oid] = []

        if frame_idx is not None:
            self.trajectories[oid].append((frame_idx, bbox, (cx, cy)))

        self.next_object_id += 1

    def deregister(self, object_id):
        # Sauvegarder la trajectoire avant suppression
        if object_id in self.trajectories:
            # Valider la trajectoire seulement si elle a au moins MIN_TRACK_LENGTH points
            if len(self.trajectories[object_id]) >= MIN_TRACK_LENGTH:
                self.completed[object_id] = self.trajectories.pop(object_id)
            else:
                self.trajectories.pop(object_id)

        self.objects.pop(object_id, None)
        self.centroids.pop(object_id, None)
        self.velocities.pop(object_id, None)
        self.disappeared.pop(object_id, None)

    def update(self, detections, frame_idx=None):
        # 1. Gestion des disparitions si aucune détection
        if len(detections) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > MAX_DISAPPEARED:
                    self.deregister(oid)
            return dict(self.objects)

        # 2. Préparation des centroïdes d'entrée
        input_centroids = []
        for (x1, y1, x2, y2) in detections:
            input_centroids.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))

        # 3. Si vide, tout enregistrer
        if len(self.objects) == 0:
            for bbox in detections:
                self.register(bbox, frame_idx)
            return dict(self.objects)

        # 4. Matrice de coût (Distance Euclidienne avec prédiction de position)
        object_ids = list(self.objects.keys())
        object_centroids = []
        for oid in object_ids:
            cx, cy = self.centroids[oid]
            vx, vy = self.velocities[oid]
            object_centroids.append((cx + vx, cy + vy))

        D = np.zeros((len(object_centroids), len(input_centroids)), dtype=np.float32)
        for i, (ocx, ocy) in enumerate(object_centroids):
            for j, (icx, icy) in enumerate(input_centroids):
                D[i, j] = math.hypot(ocx - icx, ocy - icy)

        # 5. Association
        rows, cols = linear_sum_assignment(D)

        # 6. Mise à jour des objets assignés
        assigned_rows = set()
        assigned_cols = set()

        for r, c in zip(rows, cols):
            if MAX_DISTANCE is not None and D[r, c] > MAX_DISTANCE:
                continue

            assigned_rows.add(r)
            assigned_cols.add(c)
            oid = object_ids[r]

            # Mise à jour position et vitesse (lissage)
            new_cx, new_cy = input_centroids[c]
            prev_cx, prev_cy = self.centroids[oid]

            alpha = 0.6
            vx = alpha * (new_cx - prev_cx) + (1 - alpha) * self.velocities[oid][0]
            vy = alpha * (new_cy - prev_cy) + (1 - alpha) * self.velocities[oid][1]

            self.velocities[oid] = (vx, vy)
            self.centroids[oid] = (new_cx, new_cy)
            self.objects[oid] = detections[c]
            self.disappeared[oid] = 0

            if frame_idx is not None:
                self.trajectories[oid].append((frame_idx, detections[c], (new_cx, new_cy)))

        # 7. Gestion des non-assignés (disparition)
        for r in range(len(object_ids)):
            if r not in assigned_rows:
                oid = object_ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > MAX_DISAPPEARED:
                    self.deregister(oid)

        # 8. Enregistrement des nouvelles détections
        for c in range(len(detections)):
            if c not in assigned_cols:
                self.register(detections[c], frame_idx)

        return dict(self.objects)

    def finalize(self):
        for oid in list(self.trajectories.keys()):
            # Valider la trajectoire seulement si elle a au moins MIN_TRACK_LENGTH points
            if len(self.trajectories[oid]) >= MIN_TRACK_LENGTH:
                self.completed[oid] = self.trajectories.pop(oid)
            else:
                self.trajectories.pop(oid)

        self.objects.clear()
        self.centroids.clear()
        self.velocities.clear()
        self.disappeared.clear()

    def filter_trajectories(self):
        valid = set()
        for container in [self.completed, self.trajectories]:
            for oid, traj in container.items():
                if len(traj) >= MIN_TRACK_LENGTH:
                    valid.add(oid)
        return valid

    def annotate(self, frame, objects=None):
        if objects is None: objects = self.objects
        out = frame.copy()
        for oid, bbox in objects.items():
            x1, y1, x2, y2 = bbox
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, str(oid), (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return out