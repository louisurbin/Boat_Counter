import argparse
from pathlib import Path
import sys
import cv2
import math
import numpy as np

### history=1500, varThreshold=35.0 are the important parameters for MOG2 ###
### cv2.threshold is used to binarize the mask before connectedComponents ###
### min_area=50 is used to filter small components in analyze_clusters ###
### example usage: python mog2_background_subtraction.py -i input.mp4 -o output.mp4 -d --no-shadows --no-morph ###

def parse_args():
    p = argparse.ArgumentParser(description="Apply MOG2 background subtraction to a video.")
    p.add_argument("--input", "-i", type=str, required=True, help="Input video path.")
    p.add_argument("--output", "-o", type=str, required=True, help="Output video path for the mask.")
    p.add_argument("--display", "-d", action="store_true", help="Display frames while processing.")
    p.add_argument("--history", type=int, default=1500, help="MOG2 history.")
    p.add_argument("--var-threshold", dest="var_threshold", type=float, default=35.0, help="MOG2 varThreshold.")
    p.add_argument("--no-shadows", action="store_true", help="Disable shadow detection in MOG2.")
    p.add_argument("--no-morph", action="store_true", help="Disable morphological cleanup of the mask.")
    return p.parse_args()

def analyze_clusters(mask, min_area=50):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result_mask = np.zeros_like(mask)
    for label in range(1, num_labels):
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cv2.rectangle(result_mask, (x, y), (x + width, y + height), 255, -1)
    return result_mask, stats, centroids, num_labels

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

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or (isinstance(fps, float) and math.isnan(fps)):
        fps = 30.0

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=args.history,
        varThreshold=args.var_threshold,
        detectShadows=not args.no_shadows
    )

    ext = out_path.suffix.lower()
    if ext in (".mp4", ".m4v", ".mov"):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    elif ext == ".avi":
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height), True)
    if not writer.isOpened():
        alt_fourcc = cv2.VideoWriter_fourcc(*"XVID") if fourcc != cv2.VideoWriter_fourcc(*"XVID") else cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), alt_fourcc, fps, (width, height), True)
        if not writer.isOpened():
            print(f"Failed to open VideoWriter for: {out_path}", file=sys.stderr)
            cap.release()
            sys.exit(1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        fgmask = subtractor.apply(frame)

        if not args.no_morph:
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Binarisation pour garantir un masque binaire (important pour connectedComponents)
        _, fgmask_bin = cv2.threshold(fgmask, 100, 255, cv2.THRESH_BINARY)

        rect_mask, stats, centroids, num_labels = analyze_clusters(fgmask_bin, min_area=50)

        mask_bgr = cv2.cvtColor(rect_mask, cv2.COLOR_GRAY2BGR)
        writer.write(mask_bgr)

        if args.display:
            cv2.imshow("MOG2 Mask", fgmask)
            cv2.imshow("Rectangles", rect_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    #print(f"Processed {frame_count} frames.")
    print(f"MOG2 mask video saved at: {out_path}")

if __name__ == "__main__":
    main()
