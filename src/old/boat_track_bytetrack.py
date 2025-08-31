import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def draw_tracks(frame, boxes_xyxy, ids, confs=None, class_names=None, cls_ids=None, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = map(int, box)
        tid = int(ids[i]) if ids is not None and ids[i] is not None else -1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_parts = []
        if cls_ids is not None:
            cid = int(cls_ids[i])
            name = class_names.get(cid, str(cid)) if isinstance(class_names, dict) else str(cid)
            label_parts.append(name)
        if confs is not None:
            label_parts.append(f"{float(confs[i]):.2f}")
        if tid != -1:
            label_parts.append(f"ID {tid}")
        label = " ".join(label_parts)
        if label:
            y_text = max(20, y1 - 10)
            cv2.putText(frame, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return frame


def resolve_default_model_path(default_name: str = "yolo11n.pt") -> str:
    # Try CWD first, then repo root relative to this file
    if os.path.exists(default_name):
        return default_name
    here = Path(__file__).resolve()
    root = here.parents[1]  # repo root assumed two levels up from src
    candidate = root / default_name
    return str(candidate) if candidate.exists() else default_name


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 + ByteTrack boat tracking (low-FPS tuned)")
    parser.add_argument("--source", type=str, default="./temp/masked_video.mp4", help="Video source path")
    parser.add_argument("--model", type=str, default=resolve_default_model_path(), help="YOLOv11 model path")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.12, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--classes", type=int, nargs="*", default=[8], help="Class IDs to filter (COCO boat=8)")
    parser.add_argument("--tracker", type=str, default=str(Path(__file__).with_name("trackers").joinpath("bytetrack_lowfps.yaml")), help="Tracker YAML config")
    parser.add_argument("--save", type=str, default="./temp/bytetrack_output.mp4", help="Path to save the output video")
    parser.add_argument("--view", action="store_true", help="Show live visualization window")
    parser.add_argument("--no-save", dest="save_video", action="store_false", help="Disable saving video output")
    parser.set_defaults(save_video=True)
    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)

    # Prepare video writer if needed
    vw = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    results = model.track(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        classes=args.classes,
        tracker=args.tracker,
        stream=True,
        persist=True,
        verbose=False,
    )

    window_name = "YOLOv11 ByteTrack"

    for r in results:
        frame = r.orig_img.copy()

        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r, "boxes") and r.boxes is not None and r.boxes.xyxy is not None else np.empty((0, 4))
        ids = r.boxes.id.cpu().numpy() if hasattr(r.boxes, "id") and r.boxes.id is not None else None
        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes, "conf") and r.boxes.conf is not None else None
        cls_ids = r.boxes.cls.cpu().numpy() if hasattr(r.boxes, "cls") and r.boxes.cls is not None else None

        frame = draw_tracks(frame, boxes, ids, confs, getattr(r, "names", getattr(model.model, "names", None)), cls_ids)

        # Init writer lazily with real frame size and fps from source if possible
        if vw is None and args.save_video:
            h, w = frame.shape[:2]
            # Try to read FPS from source video
            cap = getattr(r, "vid_cap", None)
            fps = None
            if cap is not None:
                try:
                    fps = cap.get(cv2.CAP_PROP_FPS)
                except Exception:
                    fps = None
            if not fps or fps <= 0:
                fps = 10  # sensible default for low-FPS inputs
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
            vw = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

        if args.save_video and vw is not None:
            vw.write(frame)

        if args.view:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    if vw is not None:
        vw.release()
    if args.view:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
