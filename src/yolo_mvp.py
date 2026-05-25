import argparse
import tempfile
import os
import cv2
from pathlib import Path
from ultralytics import YOLO
import shutil


def create_masked_subsampled_video(input_path, seconds_interval=3, side_mask_ratio=0.15):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sample_interval = max(1, int(round(fps * seconds_interval)))

    # create temp output file
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # write only sampled frames; choose a low FPS (1/frame every seconds_interval)
    out_fps = 1.0 / seconds_interval
    writer = cv2.VideoWriter(tmp_path, fourcc, out_fps, (width, height))

    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % sample_interval == 0:
            # apply side masks
            mask = frame.copy()
            w = frame.shape[1]
            left = int(w * side_mask_ratio)
            right = w - left
            cv2.rectangle(mask, (0, 0), (left, frame.shape[0]), (0, 0, 0), -1)
            cv2.rectangle(mask, (right, 0), (w, frame.shape[0]), (0, 0, 0), -1)
            writer.write(mask)
            written += 1

        frame_idx += 1

    writer.release()
    cap.release()

    if written == 0:
        os.remove(tmp_path)
        raise RuntimeError("No frames written to temporary video (check interval/fps)")

    return tmp_path


def create_frame_subsampled_video(input_path, frame_step=2, side_mask_ratio=0.0):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # create temp output file
    tmp = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    tmp_path = tmp.name
    tmp.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # keep same fps but drop frames by writing only every frame_step
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    frame_idx = 0
    written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_step == 0:
            if side_mask_ratio and side_mask_ratio > 0:
                # apply side masks
                mask = frame.copy()
                w = frame.shape[1]
                left = int(w * side_mask_ratio)
                right = w - left
                cv2.rectangle(mask, (0, 0), (left, frame.shape[0]), (0, 0, 0), -1)
                cv2.rectangle(mask, (right, 0), (w, frame.shape[0]), (0, 0, 0), -1)
                writer.write(mask)
            else:
                writer.write(frame)
            written += 1

        frame_idx += 1

    writer.release()
    cap.release()

    if written == 0:
        os.remove(tmp_path)
        raise RuntimeError("No frames written to temporary video (check frame_step)")

    return tmp_path


def main():
    parser = argparse.ArgumentParser(description="MVP YOLO + ByteTrack tracker for boats (CPU-friendly).")
    parser.add_argument("--input", "-i", required=True, help="Input video file")
    parser.add_argument("--output", "-o", default=None, help="Optional output path for tracked video")
    parser.add_argument("--model", default="yolo26n.pt", help="YOLO model (default: yolov26n.pt)")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config (default: bytetrack.yaml)")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu or cuda)")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--mask", action="store_true", help="Preprocess: apply side-mask + subsample 1 frame every N seconds before tracking")
    parser.add_argument("--seconds-interval", type=float, default=3.0, help="Seconds between frames when --mask is used (default 3s)")
    parser.add_argument("--side-mask-ratio", type=float, default=0.15, help="Fraction of image width masked on each side (default 0.15)")
    parser.add_argument("--frame-step", type=int, default=0, help="Keep 1 frame every N frames (e.g. 2 => keep 1 of 2). When >0 this overrides --seconds-interval for sampling.")
    args = parser.parse_args()

    src_path = Path(args.input)
    if not src_path.exists():
        raise SystemExit(f"Input not found: {src_path}")

    to_process = str(src_path)
    tmp_path = None
    try:
        # Preprocessing: frame-step subsampling (keep 1 frame every N) is applied independently of --mask
        if args.frame_step and args.frame_step > 1:
            print(f"Preprocessing: creating subsampled video (1 frame every {args.frame_step})...")
            # apply side mask only if --mask is provided
            mask_ratio = args.side_mask_ratio if args.mask else 0.0
            tmp_path = create_frame_subsampled_video(to_process, frame_step=args.frame_step, side_mask_ratio=mask_ratio)
            to_process = tmp_path
            print(f"Preprocessed video saved to: {to_process}")
        elif args.mask:
            print("Preprocessing: creating masked & subsampled video (this may take a moment)...")
            tmp_path = create_masked_subsampled_video(to_process, seconds_interval=args.seconds_interval, side_mask_ratio=args.side_mask_ratio)
            to_process = tmp_path
            print(f"Preprocessed video saved to: {to_process}")

        model = YOLO(args.model)

        track_kwargs = {
            "source": to_process,
            "tracker": args.tracker,
            "device": args.device,
            "persist": True,
            "conf": args.conf,
            "classes": [8],  # keep only class id 8 (boat)
        }

        if args.output:
            # Ensure output directory exists and tell ultralytics where to save
            out_path = Path(args.output)
            out_dir = out_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            track_kwargs["save"] = True
            track_kwargs["save_dir"] = str(out_dir)

        print("Running tracker (this runs ultralytics .track)...")
        results = model.track(**track_kwargs)

        print("Done tracking.")
        if args.output:
            # Try to find the most recent video file produced by Ultralytics in the save_dir and move it
            exts = [".mp4", ".avi", ".mkv", ".mov"]
            candidates = []
            for ext in exts:
                candidates.extend(list(Path(out_dir).rglob(f"*{ext}")))

            if not candidates:
                print(f"Warning: no video files found in {out_dir}. Check Ultralytics outputs manually.")
            else:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                dest = Path(args.output)
                try:
                    shutil.move(str(latest), str(dest))
                    print(f"Moved {latest} -> {dest}")
                except Exception as e:
                    print(f"Failed to move {latest} to {dest}: {e}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    main()
