import cv2
import numpy as np
import argparse
import os
import json
import sys
from tracker_class import SortTracker
from tracker_utils import detect_rectangles, save_crops, export_crossings_data, check_crossing, MIN_TRACK_LENGTH


def parse_args():
    p = argparse.ArgumentParser(description="Track white rectangles on black background (integrated main)")
    p.add_argument("--video", "-v", default="0", help="Path to video file")
    p.add_argument("--save", "-s", help="Path to save annotated output (optional)")
    p.add_argument("--color_video", "-c", default=None, help="Path to real color video to extract crops from (optional). If omitted uses --video")
    p.add_argument("--lines_json", default=None, help="Path to lines.json file (required for crossings)")
    return p.parse_args()

def main():
    args = parse_args()

    # 1. Initialisation Vidéo
    cap = cv2.VideoCapture(args.video)
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
    centroid_history = {}    # oid -> list of (cx, cy)
    crossings_per_id = {}  # oid -> list of (line_label, sign, frame_idx)
    last_checked_idx = {}      # oid -> int
    crossed_lines_per_id = {}  # oid -> set(line_label)

    print(f"Démarrage du tracking sur {args.video}...")

    # 3. Boucle Principale (Tracking)
    while True:
        ret, gray_frame = cap.read()
        if not ret:
            break

        # Détection
        detections = detect_rectangles(gray_frame)

        # Mise à jour Tracker
        objects = tracker.update(detections, frame_idx=frame_idx)
        frame_annotations.append(dict(objects))

        # Détection des franchissements de ligne tous les MIN_TRACK_LENGTH points
        if lines_info:
            for oid, bbox in objects.items():
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                curr_pt = (cx, cy)

                centroid_history.setdefault(oid, []).append(curr_pt)
                last_checked_idx.setdefault(oid, 0)
                crossed_lines_per_id.setdefault(oid, set())

                hist = centroid_history[oid]
                i0 = last_checked_idx[oid]

                # Assez de points pour tester un segment
                if len(hist) - i0 >= MIN_TRACK_LENGTH:
                    p_start = hist[i0]
                    p_end   = hist[i0 + MIN_TRACK_LENGTH - 1]
                    
                    for line in lines_info:
                        label = line["label"]

                        if label in crossed_lines_per_id[oid]:
                            continue

                        crossed, sign = check_crossing(line["p1"], line["p2"], p_start, p_end)

                        if crossed:
                                crossings_per_id.setdefault(oid, []).append((label, sign, frame_idx))
                                crossed_lines_per_id[oid].add(label)

                    # Avancer la fenêtre 
                    last_checked_idx[oid] += MIN_TRACK_LENGTH - 1

        # Rendu Vidéo Trajectoires
        if writer_traj:
            vis = gray_frame.copy()
            # Dessin trajectoires
            for oid, traj in tracker.trajectories.items():
                n = len(traj)
                if n < MIN_TRACK_LENGTH:
                    continue

                pts = []
                i = 0
                while i + MIN_TRACK_LENGTH - 1 < n:
                    p_start = traj[i][2]
                    p_end = traj[i + MIN_TRACK_LENGTH - 1][2]
                    pts.extend([p_start, p_end])
                    i += MIN_TRACK_LENGTH - 1
                
                if pts:
                    pts_np = np.array(pts, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(vis, [pts_np], False, (0, 255, 255), 2)

            # Dessin boîtes et IDs pour chaque frame
            for oid, bbox in objects.items():
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, str(oid), (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Écriture dans la vidéo
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
    save_crops(color_src, frame_annotations, crossings_per_id, base_temp, tracker.completed)

    # 5. Export des données (Crossings)
    export_crossings_data(crossings_per_id, base_temp, tracker.completed)

if __name__ == "__main__":
    main()