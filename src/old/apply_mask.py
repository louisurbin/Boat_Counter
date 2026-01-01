import cv2
import numpy as np
import os
import argparse

def apply_mask_to_video(video_path, mask_path, out_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Read mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Cannot read mask: {mask_path}")

    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    # Resize mask if needed
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_bool = mask > 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Apply mask: keep only masked region, set outside to black
        masked = frame.copy()
        masked[~mask_bool] = 0
        out.write(masked)

    cap.release()
    out.release()
    print(f"Masked video saved at: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply mask to all frames of a video.")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("mask", help="Path to mask image (PNG)")
    parser.add_argument("--out", default="./temp/masked_video.mp4", help="Output video path")
    args = parser.parse_args()
    apply_mask_to_video(args.video, args.mask, args.out)