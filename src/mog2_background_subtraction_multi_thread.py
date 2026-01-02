import argparse
from pathlib import Path
import sys
import cv2
import math
import numpy as np
import threading
import queue

# PARAMÈTRES PAR DÉFAUT
BINARIZE_THRESHOLD = 100
VAR_THRESHOLD = 50
HISTORY = 1500
MIN_AREA = 500
MAX_QUEUE = 100  # Taille maximale des queues

def parse_args():
    p = argparse.ArgumentParser(description="Apply MOG2 background subtraction to a video.")
    p.add_argument("--input", "-i", type=str, required=True, help="Input video path.")
    p.add_argument("--output", "-o", type=str, required=True, help="Output video path for the mask.")
    p.add_argument("--mask", "-m", type=str, help="Path to the mask image to apply before MOG2.")
    p.add_argument("--history", type=int, default=HISTORY, help=f"MOG2 history. default={HISTORY}")
    p.add_argument("--var-threshold", dest="var_threshold", type=float, default=VAR_THRESHOLD, help=f"MOG2 varThreshold. default={VAR_THRESHOLD}")
    p.add_argument("--binarize-threshold", dest="binarize_threshold", type=int, default=BINARIZE_THRESHOLD, help=f"Threshold (0-255) to binarize MOG2 mask. default={BINARIZE_THRESHOLD}")
    p.add_argument("--min-area", type=int, default=MIN_AREA, help=f"Minimum area to keep a cluster. default={MIN_AREA}")
    p.add_argument("--no-shadows", action="store_true", help="Disable shadow detection in MOG2.")
    p.add_argument("--no-morph", action="store_true", help="Disable morphological cleanup of the mask.")
    return p.parse_args()

def analyze_clusters(mask, min_area):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    valid = stats[1:, cv2.CC_STAT_AREA] >= min_area
    valid_stats = stats[1:][valid]

    result = np.zeros_like(mask)
    for x, y, w, h, _ in valid_stats:
        result[y:y+h, x:x+w] = 255
    return result

def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"Input video not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"Failed to open video: {in_path}", file=sys.stderr)
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 1:
        print(f"Cannot read FPS from video: {in_path}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    # Load mask if provided
    mask = None
    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load mask: {args.mask}", file=sys.stderr)
            sys.exit(1)
        if mask.shape != (height, width):
            print(f"Mask dimensions do not match video dimensions: {mask.shape} vs {(height, width)}", file=sys.stderr)
            sys.exit(1)

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=args.history,
        varThreshold=args.var_threshold,
        detectShadows=not args.no_shadows
    )

    ext = out_path.suffix.lower()
    fourcc = cv2.VideoWriter_fourcc(*"XVID") if ext == ".avi" else cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height), False)
    if not writer.isOpened():
        alt_fourcc = cv2.VideoWriter_fourcc(*"XVID") if fourcc != cv2.VideoWriter_fourcc(*"XVID") else cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), alt_fourcc, fps, (width, height), False)
        if not writer.isOpened():
            print(f"Failed to open VideoWriter for: {out_path}", file=sys.stderr)
            cap.release()
            sys.exit(1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # Queues pour pipeline multi-thread
    frame_queue = queue.Queue(maxsize=MAX_QUEUE)
    mask_queue = queue.Queue(maxsize=MAX_QUEUE)

    # Thread lecture
    def reader():
        while True:
            ok, frame = cap.read()
            if not ok:
                frame_queue.put(None)
                break
            frame_queue.put(frame)

    # Thread traitement
    def processor():
        while True:
            frame = frame_queue.get()
            if frame is None:
                mask_queue.put(None)
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if mask is not None:
                gray_frame = cv2.bitwise_and(gray_frame, gray_frame, mask=mask)

            fgmask = subtractor.apply(gray_frame)
            if not args.no_morph:
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=1)

            _, fgmask_bin = cv2.threshold(fgmask, args.binarize_threshold, 255, cv2.THRESH_BINARY)
            rect_mask = analyze_clusters(fgmask_bin, args.min_area)

            mask_queue.put(rect_mask)

    # Thread écriture
    def writer_thread():
        while True:
            rect_mask = mask_queue.get()
            if rect_mask is None:
                break
            writer.write(rect_mask)

    # Lancement des threads
    t_reader = threading.Thread(target=reader)
    t_processor = threading.Thread(target=processor)
    t_writer = threading.Thread(target=writer_thread)

    t_reader.start()
    t_processor.start()
    t_writer.start()

    t_reader.join()
    t_processor.join()
    t_writer.join()

    cap.release()
    writer.release()
    print(f"MOG2 mask video saved at: {out_path}")

if __name__ == "__main__":
    main()
