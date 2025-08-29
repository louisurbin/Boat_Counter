import argparse
from pathlib import Path
import sys
import cv2

def parse_args():
    p = argparse.ArgumentParser(description="Apply MOG2 background subtraction to a video.")
    p.add_argument("--input", "-i", type=str, required=True, help="Input video path.")
    p.add_argument("--output", "-o", type=str, required=True, help="Output video path for the mask.")
    # Added options used later in the script
    p.add_argument("--display", "-d", action="store_true", help="Display frames while processing.")
    p.add_argument("--history", type=int, default=500, help="MOG2 history.")
    p.add_argument("--var-threshold", dest="var_threshold", type=float, default=16.0, help="MOG2 varThreshold.")
    p.add_argument("--no-shadows", action="store_true", help="Disable shadow detection in MOG2.")
    p.add_argument("--no-morph", action="store_true", help="Disable morphological cleanup of the mask.")
    return p.parse_args()

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
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # MOG2 subtractor
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=args.history,
        varThreshold=args.var_threshold,
        detectShadows=not args.no_shadows
    )

    # VideoWriter expects 3-channel; convert mask to BGR for saving
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height), True)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        fgmask = subtractor.apply(frame)

        # Optional cleanup: remove noise and fill small holes
        if not args.no_morph:
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Convert to BGR for saving
        mask_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        writer.write(mask_bgr)

        if args.display:
            cv2.imshow("MOG2 Mask", fgmask)
            # Optional side-by-side view
            # vis = cv2.hconcat([frame, mask_bgr])
            # cv2.imshow("Input | Mask", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    if args.display:
        cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames.")
    print(f"Saved mask video to: {out_path}")

if __name__ == "__main__":
    main()
