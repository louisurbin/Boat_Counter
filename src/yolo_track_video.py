import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

def main():
    import argparse
    parser = argparse.ArgumentParser(description="YOLOv11 + tracker intégré sur une vidéo.")
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", required=True, help="Output video path")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--conf-threshold", type=float, default=0.2, help="YOLO confidence threshold")
    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="Tracker type (bytetrack, botsort, etc.)")
    args = parser.parse_args()

    # Charger le modèle YOLO11n (nano)
    model = YOLO("yolo11n.pt") 

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        print(f"Failed to open input video: {args.input}", file=sys.stderr)
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height), True)
    if not writer.isOpened():
        print(f"Failed to open output video: {args.output}", file=sys.stderr)
        sys.exit(1)

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLOv11 inference avec tracking intégré (frame par frame, pas stream=True !)
        results = model.track(
            frame,
            persist=True,
            conf=args.conf_threshold,
            tracker=args.tracker,
            device=args.device,
            verbose=False
        )

        # Dessiner les résultats sur la frame
        if results and hasattr(results[0], "plot"):
            out_frame = results[0].plot()
        else:
            out_frame = frame

        writer.write(out_frame)
        frame_count += 1

    cap.release()
    writer.release()
    print(f"Processed {frame_count} frames.")
    print(f"Saved tracked video to: {args.output}")

if __name__ == "__main__":
    main()
