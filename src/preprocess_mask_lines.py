import cv2
import numpy as np
import os
import argparse
import json


def create_mask_and_lines(video_path, out_dir="temp", window_name="Mask and Lines Editor"):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Unable to read the first frame of the video: {video_path}")

    h, w = frame.shape[:2]
    poly_pts = []
    lines = []  # list of tuples: (p1, p2, label)
    temp_line = []
    polygon_closed = False

    display = frame.copy()

    def draw():
        nonlocal display
        display = frame.copy()
        polygon_color = (255, 0, 0)  # blue for polygon / water zone
        line_color = (0, 0, 255)     # red for counting lines

        # overlay mask preview if polygon closed or has points
        if poly_pts:
            pts = np.array(poly_pts, np.int32).reshape((-1, 1, 2))
            if polygon_closed:
                overlay = display.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
                cv2.polylines(display, [pts], isClosed=True, color=polygon_color, thickness=2)
            else:
                cv2.polylines(display, [pts], isClosed=False, color=line_color, thickness=2)
            for p in poly_pts:
                cv2.circle(display, tuple(p), 4, polygon_color, -1)

         # draw lines
        for idx, (p1, p2, label) in enumerate(lines):
            cv2.line(display, tuple(p1), tuple(p2), line_color, 2)
            if label:
                cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                cv2.putText(display, label, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

            
    # temp line preview (one point placed)
        if temp_line:
            cv2.circle(display, tuple(temp_line[0]), 4, (255, 0, 0), -1)

    def on_mouse(event, x, y, flags, param):
        nonlocal poly_pts, temp_line, polygon_closed, line_mode, lines
        if event == cv2.EVENT_LBUTTONDOWN:
            if line_mode:
                # line-creation mode: collect two points
                temp_line.append((x, y))
                if len(temp_line) == 2:
                    label = input("Label for this line (direction code, enter empty to skip): ").strip()
                    lines.append((temp_line[0], temp_line[1], label))
                    temp_line.clear()
            else:
                # polygon mode: add polygon point
                if not polygon_closed:
                    poly_pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # right click closes polygon if enough points
            if not polygon_closed and len(poly_pts) >= 3:
                polygon_closed = True

    # Initialize temp_line as empty list but treat it as "line-mode" toggle via 'l' key
    temp_line = []

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    draw()
    print("Instructions:")
    print(" - Left click: add polygon point (when polygon open) or line point (in line mode).")
    print(" - Right click: close polygon (when >=3 points).")
    print(" - l : enter line mode (next two left clicks create a line).")
    print(" - z : undo last polygon point or last line.")
    print(" - r : reset everything.")
    print(" - s : save mask and lines to temp/.")
    print(" - q or ESC : quit without saving.")

    line_mode = False

    while True:
        draw()
        mode_text = "LINE MODE" if line_mode else "POLY MODE"
        cv2.putText(display, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            poly_pts = []
            lines = []
            temp_line = []
            polygon_closed = False
            line_mode = False
            draw()
        elif key == ord('l'):
            # toggle line mode
            line_mode = not line_mode
            temp_line = []
            if line_mode:
                print("Line mode: click two points to create a line (then input label in console).")
        elif key == ord('z'):
            # undo
            if line_mode and temp_line:
                temp_line.pop()
            elif line_mode and not temp_line and lines:
                lines.pop()
            elif not line_mode and not polygon_closed and poly_pts:
                poly_pts.pop()
            elif not line_mode and polygon_closed:
                # reopen polygon (undo close) if desired
                polygon_closed = False
        elif key == ord('s'):
            # save mask and lines
            mask = np.zeros((h, w), dtype=np.uint8)
            if poly_pts:
                # fill polygon (interior = water -> 255)
                cv2.fillPoly(mask, [np.array(poly_pts, np.int32)], 255)
            base = os.path.splitext(os.path.basename(video_path))[0]
            mask_path = os.path.join(out_dir, f"{base}_mask.png")
            json_path = os.path.join(out_dir, f"{base}_lines.json")
            cv2.imwrite(mask_path, mask)
            out_lines = []
            for idx, (p1, p2, label) in enumerate(lines):
                out_lines.append({"id": idx, "p1": list(p1), "p2": list(p2), "label": label})
            meta = {"video": os.path.basename(video_path), "image_size": [w, h], "lines": out_lines}
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"Saved mask -> {mask_path}")
            print(f"Saved lines -> {json_path}")

    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create water mask and counting lines from first frame.")
    parser.add_argument("video", help="Path to video in data/")
    parser.add_argument("--out", default="temp", help="Output directory for mask and lines (default: temp/)")
    args = parser.parse_args()
    create_mask_and_lines(args.video, args.out)