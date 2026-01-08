import argparse
from pathlib import Path
import sys
import cv2
import math
import numpy as np

# PARAMÈTRES
VAR_THRESHOLD = 30                          # paramètre de variance
HISTORY = 50                                # nombre de frames retenues par MOG2 pour estimer le fond
MIN_AREA = 10**2                            # aire minimale pour considérer un cluster comme valide  
MORPH_KERNEL_OPEN = 7                       # taille max du bruit blanc à supprimer
MORPH_KERNEL_CLOSE = 50                     # taille max des trous noirs à combler 

def parse_args():
    p = argparse.ArgumentParser(description="Apply MOG2 background subtraction to a video.")
    p.add_argument("--input", "-i", type=str, required=True, help="Input video path.")
    p.add_argument("--output", "-o", type=str, required=True, help="Output video path for the mask.")
    p.add_argument("--mask", "-m", type=str, help="Path to the mask image to apply before MOG2.")
    return p.parse_args()

def analyze_clusters(mask, result):
    """
    Fill bounding boxes of connected components above MIN_AREA directly in 'result'.
    Reuses the same buffer to avoid allocations.
    """
    result.fill(0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            result[y:y+h, x:x+w] = 255

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
    if not fps or math.isnan(fps) or fps <= 1:
        print(f"Cannot read FPS from video: {in_path}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    # Load the mask if provided
    mask = None
    mask_zero = None
    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Failed to load mask: {args.mask}", file=sys.stderr)
            sys.exit(1)
        if mask.shape != (height, width):
            print(f"Mask dimensions do not match video dimensions: {mask.shape} vs {(height, width)}", file=sys.stderr)
            sys.exit(1)
        mask_zero = (mask == 0)  # precompute once

        # Compute ROI from mask
        ys, xs = np.where(mask != 0)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        roi_slice = (slice(y0, y1), slice(x0, x1))
    else:
        roi_slice = (slice(0, height), slice(0, width))

    # MOG2 background subtractor
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=HISTORY,
        varThreshold=VAR_THRESHOLD,
        detectShadows=True  # shadows treated as objects
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

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_OPEN, MORPH_KERNEL_OPEN))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_CLOSE, MORPH_KERNEL_CLOSE))
    
    frame_count = 0

    # buffer reused for analyze_clusters to avoid allocation
    rect_mask = np.zeros((height, width), dtype=np.uint8)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply static mask (in-place)
        if mask_zero is not None:
            gray_frame[mask_zero] = 0

        # Apply MOG2 only on ROI
        fgmask = np.zeros((height, width), dtype=np.uint8)
        roi = gray_frame[roi_slice]
        fg_roi = subtractor.apply(roi)
        fgmask[roi_slice] = fg_roi

        # Morphological cleanup
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # Note: no threshold needed, shadows included as objects
        analyze_clusters(fgmask, rect_mask)
        writer.write(rect_mask)

    cap.release()
    writer.release()
    print(f"MOG2 mask video saved at: {out_path}")

if __name__ == "__main__":
    main()
