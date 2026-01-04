import cv2
import numpy as np
import argparse
import os
import json
import sys
from tracker_class import SortTracker
from tracker_utils import detect_rectangles, save_crops, export_crossings_data, open_capture, check_crossing


def parse_args():
    p = argparse.ArgumentParser(description="Track white rectangles on black background (integrated main)")
    p.add_argument("--video", "-v", default="0", help="Path to video file or '0' for webcam")
    p.add_argument("--save", "-s", help="Path to save annotated output (optional)")
    p.add_argument("--color_video", "-c", default=None, help="Path to real color video to extract crops from (optional). If omitted uses --video")
    p.add_argument("--lines_json", default=None, help="Path to lines.json file (required for crossings)")
    return p.parse_args()

def main():
    args = parse_args()

    # 1. Initialisation Vidéo
    cap = open_capture(args.video)
    if not cap.isOpened():
        sys.exit(f"Erreur: Impossible d'ouvrir {args.video}")

    # Chargement des lignes
    lines_info = []
    if args.lines_json and os.path.exists(args.lines_json):
        with open(args.lines_json, "r", encoding="utf-8") as f:
            lines_info = json.load(f).get("lines", [])

    # 2. Initialisation Tracker
    tracker = SortTracker()

    # Dossiers de sortie
    base_temp = os.path.abspath(os.path.join(".", "temp", "extractions"))
    os.makedirs(base_temp, exist_ok=True)

    # Writers
    fps_video = cap.get(cv2.CAP_PROP_FPS) 
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    writer_traj = None
    if args.save:
        traj_path = os.path.join(os.path.dirname(args.save), "trajectories.mp4")
        writer_traj = cv2.VideoWriter(traj_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_video, frame_size)

    # Variables d'état
    frame_idx = 0
    frame_annotations = [] # frame_idx -> {oid: bbox}
    prev_centroids = {}    # oid -> (cx, cy)
    crossings_per_id = {}  # oid -> list of (line_label, sign, frame_idx)

    print(f"Démarrage du tracking sur {args.video}...")

    # 3. Boucle Principale (Tracking)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Détection
        detections = detect_rectangles(frame)

        # Mise à jour Tracker
        objects = tracker.update(detections, frame_idx=frame_idx)
        frame_annotations.append(dict(objects))

        # Détection des franchissements de ligne
        if lines_info:
            for oid, bbox in objects.items():
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                curr_pt = (cx, cy)

                if oid in prev_centroids:
                    prev_pt = prev_centroids[oid]
                    for line in lines_info:
                        crossed, sign = check_crossing(line["p1"], line["p2"], prev_pt, curr_pt)
                        if crossed:
                            crossings_per_id.setdefault(oid, []).append((line["label"], sign, frame_idx))

                prev_centroids[oid] = curr_pt

        # Rendu Vidéo Trajectoires
        if writer_traj:
            vis = frame.copy()
            # Dessin trajectoires
            for oid, traj in tracker.trajectories.items():
                if len(traj) > 1:
                    pts = np.array([t[2] for t in traj], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis, [pts], False, (0, 255, 255), 2)
            # Dessin boîtes avec IDs
            for oid, bbox in objects.items():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, str(oid), (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            writer_traj.write(vis)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"Frame {frame_idx}...", end='\r')

    cap.release()
    if writer_traj: writer_traj.release()
    tracker.finalize()
    print(f"\nTracking terminé. {len(tracker.completed)} objets suivis.")

    # 4. Extraction des Crops (sur vidéo couleur)
    color_src = args.color_video if args.color_video else args.video
    save_crops(color_src, frame_annotations, crossings_per_id, base_temp)

    # 5. Export des données (Crossings)
    export_crossings_data(crossings_per_id, base_temp)

if __name__ == "__main__":
    main()