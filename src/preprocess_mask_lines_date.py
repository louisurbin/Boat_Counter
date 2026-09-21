import cv2
import numpy as np
import os
import argparse
import json

def get_mask_lines_date_paths(video_path, out_dir):
    """Génère les chemins pour le masque et les lignes."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    mask_path = os.path.join(out_dir, f"{base}_mask.png")
    lines_path = os.path.join(out_dir, f"{base}_lines_date.json")
    return mask_path, lines_path

def create_mask_lines_date(video_path, out_dir="temp", window_name="Mask, Lines and Date Editor"):
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

    # Créer une fenêtre OpenCV avec une taille appropriée
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, w, h)

    def draw():
        nonlocal display
        display = frame.copy()
        polygon_color = (255, 0, 0)  # Bleu pour le polygone / zone d'eau
        line_color = (0, 0, 255)     # Rouge pour les lignes de comptage

        # Aperçu du masque si le polygone est fermé ou a des points
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

        # Dessiner les lignes
        for p1, p2, label in lines:
            cv2.line(display, tuple(p1), tuple(p2), line_color, 2)
            if label:
                cx, cy = (p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2
                cv2.putText(display, label, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

    # Aperçu de la ligne en cours
        if temp_line:
            cv2.circle(display, tuple(temp_line[0]), 4, (255, 0, 0), -1)
            if len(temp_line) >= 2:
                cv2.circle(display, tuple(temp_line[1]), 4, (255, 0, 0), -1)
                # Afficher le trait rouge dès que les deux points sont placés
                cv2.line(display, tuple(temp_line[0]), tuple(temp_line[1]), (0, 0, 255), 2)

    label_input_mode = False
    label_text = ""
    date_input_mode = False
    date_text = ""
    date_mode_active = False
    interval_input_mode = False
    interval_text = ""
    real_fps = 1/3  # Valeur par défaut
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

    # Initialiser temp_line comme liste vide (le mode ligne s'active via la touche 'l')
    temp_line = []

    cv2.setMouseCallback(window_name, on_mouse)

    draw()
    print("Instructions:")
    print(" - Clic gauche : ajouter un point au polygone (ou un point de ligne en mode ligne).")
    print(" - Clic droit : fermer le polygone (>= 3 points).")
    print(" - l : mode ligne (les deux prochains clics gauches créent une ligne).")
    print(" - z : annuler le dernier point du polygone ou la dernière ligne.")
    print(" - r : tout réinitialiser.")
    print(" - d : saisir la date de début (MM/DD HH:MM:SS).")
    print(" - f : saisir le frametime (secondes, ex: 3 pour 1 image/3s).")
    print(" - s : sauvegarder le masque, les lignes et la date.")
    print(" - ESC : quitter sans sauvegarder.")

    line_mode = False

    while True:
        draw()
        # Afficher le mode en haut à gauche
        mode_text = "DATE MODE" if date_mode_active else ("LINE MODE" if line_mode else "POLY MODE")
        cv2.putText(display, mode_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if label_input_mode or date_input_mode or interval_input_mode:
            # Zone de saisie semi-transparente (superposée pour tous les modes)
            overlay = display.copy()
            box_w = max(50, int(w / 3))
            x2 = min(10 + box_w, w - 10)
            cv2.rectangle(overlay, (10, 50), (x2, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, display, 0.5, 0, display)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.7
            thickness = 2
            
        if label_input_mode:
            cv2.putText(display, f"Nom de la ligne : {label_text}", (15, 80), font, 0.8, (0, 255, 255), 2)
        if date_input_mode:
            # Afficher le format fixe et le texte saisi
            prefix = "MM/DD HH:MM:SS :"
            cv2.putText(display, prefix, (15, 80), font, scale, (0, 255, 255), thickness)
            if date_text:
                (pw, ph), _ = cv2.getTextSize(prefix, font, scale, thickness)
                x_date = 15 + pw + 8
                cv2.putText(display, date_text, (x_date, 80), font, scale, (0, 255, 255), thickness)
        if interval_input_mode:
            prefix = "Frametime :"
            cv2.putText(display, prefix, (15, 80), font, scale, (0, 255, 255), thickness)
            if interval_text:
                (pw, ph), _ = cv2.getTextSize(prefix, font, scale, thickness)
                x_interval = 15 + pw + 8
                cv2.putText(display, interval_text, (x_interval, 80), font, scale, (0, 255, 255), thickness)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20)
        # Normaliser la touche
        if key == -1:
            k = None
        else:
            k = key & 0xFF

        # Gestion de la saisie de la date
        if date_input_mode:
            if k is not None:
                if k == 27:  # Échap : annuler
                    date_input_mode = False
                    date_text = ""
                elif k == 13 or k == 10:  # Entrée : valider et garder le mode DATE actif
                    date_input_mode = False
                    date_text = date_text.strip()
                    date_mode_active = True
                elif k == 8:  # Retour arrière
                    date_text = date_text[:-1]
                else:
                    # Accepter les chiffres et quelques séparateurs
                    if 32 <= k <= 126:
                        ch = chr(k)
                        if ch.isdigit() or ch in ['/', ':', ' ', '-']:
                            date_text += ch
            # Désactiver le mode ligne pendant l'édition de la date
            line_mode = False
            continue

        # Gestion de la saisie de l'intervalle
        if interval_input_mode:
            if k is not None:
                if k == 27:  # Échap : annuler
                    interval_input_mode = False
                    interval_text = ""
                elif k == 13 or k == 10:  # Entrée : valider
                    interval_input_mode = False
                    try:
                        interval_seconds = float(interval_text.strip())
                        if interval_seconds > 0:
                            real_fps = 1.0 / interval_seconds
                        else:
                            print("Erreur: l'intervalle doit être > 0. Utilisation de la valeur par défaut (1/3).")
                            real_fps = 1/3
                    except ValueError:
                        print("Erreur: intervalle invalide. Utilisation de la valeur par défaut (1/3).")
                        real_fps = 1/3
                elif k == 8:  # Retour arrière
                    interval_text = interval_text[:-1]
                else:
                    # Accepter les chiffres et le point décimal
                    if 32 <= k <= 126:
                        ch = chr(k)
                        if ch.isdigit() or ch == '.':
                            interval_text += ch
            # Désactiver le mode ligne pendant l'édition de l'intervalle
            line_mode = False
            continue

        # Gestion de la saisie du label
        if label_input_mode:
            if k is not None:
                if k == 13 or k == 10:  # Entrée
                    lines.append((temp_line[0], temp_line[1], label_text.strip()))
                    temp_line.clear()
                    label_input_mode = False
                    label_text = ""
                elif k == 27:  # Échap : annuler
                    temp_line.clear()
                    label_input_mode = False
                    label_text = ""
                elif k == 8:  # Retour arrière
                    label_text = label_text[:-1]
                elif 32 <= k <= 126:
                    label_text += chr(k)
            continue

        # Touches générales
        if k is None:
            continue
        if k == ord('d'):
            # Basculer le mode date persistant; l'ouvre active aussi la saisie
            date_mode_active = not date_mode_active
            date_input_mode = date_mode_active
            if date_mode_active:
                date_text = ""
                line_mode = False
            continue
        if k == ord('f'):
            # Basculer le mode frametime
            interval_input_mode = not interval_input_mode
            if interval_input_mode:
                interval_text = ""
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
            meta = {"video": os.path.basename(video_path), "image_size": [w, h], "real_fps": real_fps, "lines": out_lines}
            # Inclure start_time si fourni via la saisie 'd'
            if date_text:
                meta['start_time'] = date_text
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"Masque sauvegardé -> {mask_path}")
            print(f"Lignes et date sauvegardées -> {json_path}")
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Création du masque, des lignes de comptage et de la date à partir de la première frame.")
    parser.add_argument("video", help="Chemin vers la vidéo")
    parser.add_argument("--out", default="temp", help="Dossier de sortie (défaut: temp/)")
    args = parser.parse_args()
    create_mask_lines_date(args.video, args.out)