import cv2
import sys
from pathlib import Path

def apply_mask_to_video(input_video_path, mask_video_path, output_video_path):
    cap = cv2.VideoCapture(str(input_video_path))
    mask_cap = cv2.VideoCapture(str(mask_video_path))

    if not cap.isOpened():
        print(f"Failed to open input video: {input_video_path}", file=sys.stderr)
        sys.exit(1)
    if not mask_cap.isOpened():
        print(f"Failed to open mask video: {mask_video_path}", file=sys.stderr)
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height), True)
    if not writer.isOpened():
        print(f"Failed to open output video: {output_video_path}", file=sys.stderr)
        sys.exit(1)

    while True:
        ok1, frame = cap.read()
        ok2, mask_frame = mask_cap.read()
        if not ok1 or not ok2:
            break

        # Assume mask is single channel or 3-channel grayscale
        if mask_frame.ndim == 3 and mask_frame.shape[2] == 3:
            mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask_frame

        # Binarize mask just in case
        _, mask_bin = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)

        # Apply mask: keep color where mask is white, else black
        masked = cv2.bitwise_and(frame, frame, mask=mask_bin)

        writer.write(masked)

    cap.release()
    mask_cap.release()
    writer.release()
    print(f"Saved masked video to: {output_video_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Apply a mask video to a color video (same duration/size).")
    parser.add_argument("--input", "-i", required=True, help="Input color video path")
    parser.add_argument("--mask", "-m", required=True, help="Input mask video path (black/white)")
    parser.add_argument("--output", "-o", required=True, help="Output masked video path")
    args = parser.parse_args()
    apply_mask_to_video(args.input, args.mask, args.output)
