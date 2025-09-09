'''
Instructions principales :
 - Tracer le polygone (zone d'eau) avec des clics gauches
 - Fermer le polygone avec un clic droit
 - Passer en mode ligne avec la touche 'l'
 - Placer deux points (clic gauche), puis entrer le nom de la ligne directement dans la fenêtre (taper le nom puis Entrée)
 - Touche 's' pour enregistrer le masque et les lignes
 - Touche 'echap' ou 'q' pour quitter
'''

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

            
    # temp line preview (un ou deux points placés)
        if temp_line:
            cv2.circle(display, tuple(temp_line[0]), 4, (255, 0, 0), -1)
            if len(temp_line) >= 2:
                cv2.circle(display, tuple(temp_line[1]), 4, (255, 0, 0), -1)
                # Affiche le trait rouge dès que les deux points sont placés
                cv2.line(display, tuple(temp_line[0]), tuple(temp_line[1]), (0, 0, 255), 2)

    label_input_mode = False
    label_text = ""
    date_input_mode = False
    date_text = ""
    date_mode_active = False
    def on_mouse(event, x, y, flags, param):
        nonlocal poly_pts, temp_line, polygon_closed, line_mode, lines, label_input_mode, label_text
        if event == cv2.EVENT_LBUTTONDOWN:
            if line_mode and not label_input_mode:
                temp_line.append((x, y))
                if len(temp_line) == 2:
                    label_input_mode = True
                    label_text = ""
            else:
                if not polygon_closed:
                    poly_pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
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
    print(" - d : enter date input mode (MM/DD HH:MM:SS) to set video start time.")
    print(" - s : save mask and lines to temp/.")
    print(" - ESC : quit without saving.")

    line_mode = False

    while True:
        draw()
        # show mode (DATE MODE if date_mode_active) at top-left
        mode_text = "DATE MODE" if date_mode_active else ("LINE MODE" if line_mode else "POLY MODE")
        cv2.putText(display, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if label_input_mode:
            # Afficher la zone de saisie du label (1/3 de la largeur de l'image)
            overlay = display.copy()
            box_w = max(50, int(w / 3))
            x2 = min(10 + box_w, w - 10)
            cv2.rectangle(overlay, (10, 50), (x2, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
            cv2.putText(display, f"Nom de la ligne : {label_text}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if date_input_mode:
            # Afficher la zone de saisie semi-transparente comme pour le label, mais pour la date
            overlay = display.copy()
            # box width = 1/3 of frame width (same as label box)
            box_w = max(50, int(w / 3))
            x2 = min(10 + box_w, w - 10)
            cv2.rectangle(overlay, (10, 50), (x2, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
            # show fixed format hint and current typed text, place date_text after prefix
            prefix = "MM/DD HH:MM:SS :"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.7
            thickness = 2
            cv2.putText(display, prefix, (15, 80), font, scale, (0, 255, 255), thickness)
            if date_text:
                (pw, ph), _ = cv2.getTextSize(prefix, font, scale, thickness)
                x_date = 15 + pw + 8
                cv2.putText(display, date_text, (x_date, 80), font, scale, (0, 255, 255), thickness)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20)
        # normalize key
        if key == -1:
            k = None
        else:
            k = key & 0xFF

        # date input handling (use k)
        if date_input_mode:
            if k is not None:
                if k == 27:  # Esc: cancel
                    date_input_mode = False
                    date_text = ""
                elif k == 13 or k == 10:  # Enter: accept and close date box but keep DATE MODE active
                    date_input_mode = False
                    date_text = date_text.strip()
                    date_mode_active = True
                elif k == 8:  # Backspace
                    date_text = date_text[:-1]
                else:
                    # accept digits and a few separators
                    if 32 <= k <= 126:
                        ch = chr(k)
                        if ch.isdigit() or ch in ['/', ':', ' ', '-']:
                            date_text += ch
            # ensure line mode disabled while editing date
            line_mode = False
            continue

        # label input handling (use k)
        if label_input_mode:
            if k is not None:
                if k == 13 or k == 10:  # Enter
                    lines.append((temp_line[0], temp_line[1], label_text.strip()))
                    temp_line.clear()
                    label_input_mode = False
                    label_text = ""
                elif k == 27:  # Esc cancel
                    temp_line.clear()
                    label_input_mode = False
                    label_text = ""
                elif k == 8:  # Backspace
                    label_text = label_text[:-1]
                elif 32 <= k <= 126:
                    label_text += chr(k)
            continue

        # general keys (use k)
        if k is None:
            continue
        if k == ord('d'):
            # toggle persistent date mode; opening it also enables the input box
            date_mode_active = not date_mode_active
            date_input_mode = date_mode_active
            if date_mode_active:
                date_text = ""
                line_mode = False
            continue
        elif k == 27:
            break
        elif k == ord('r'):
            poly_pts = []
            lines = []
            temp_line = []
            polygon_closed = False
            line_mode = False
            draw()
        elif k == ord('l'):
            line_mode = not line_mode
            temp_line = []
        elif k == ord('z'):
            if line_mode and temp_line:
                temp_line.pop()
            elif line_mode and not temp_line and lines:
                lines.pop()
            elif not line_mode and not polygon_closed and poly_pts:
                poly_pts.pop()
            elif not line_mode and polygon_closed:
                polygon_closed = False
        elif k == ord('s'):
            mask = np.zeros((h, w), dtype=np.uint8)
            if poly_pts:
                cv2.fillPoly(mask, [np.array(poly_pts, np.int32)], 255)
            base = os.path.splitext(os.path.basename(video_path))[0]
            mask_path = os.path.join(out_dir, f"{base}_mask.png")
            json_path = os.path.join(out_dir, f"{base}_lines_date.json")
            cv2.imwrite(mask_path, mask)
            out_lines = []
            for idx, (p1, p2, label) in enumerate(lines):
                out_lines.append({"id": idx, "p1": list(p1), "p2": list(p2), "label": label})
            meta = {"video": os.path.basename(video_path), "image_size": [w, h], "lines": out_lines}
            # include start_time if provided via GUI 'd' input
            if date_text:
                meta['start_time'] = date_text
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"Saved mask -> {mask_path}")
            print(f"Saved lines -> {json_path}")
            # start_time handled via 'd' GUI input (date_text)
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create water mask and counting lines from first frame.")
    parser.add_argument("video", help="Path to video in data/")
    parser.add_argument("--out", default="temp", help="Output directory for mask and lines (default: temp/)")
    args = parser.parse_args()
    create_mask_and_lines(args.video, args.out)