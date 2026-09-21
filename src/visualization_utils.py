import cv2
import json
import os
import math

def visualize_line_crossings(video_path, lines_json_path, id_dir, show=True):
    """Visualise les franchissements de lignes sur la première frame de la vidéo, avec flèches et compteurs.

    show : True  -> affiche une fenêtre et attend une touche (mode manuel)
           False -> ne dessine et n'affiche rien (mode automatique / traitement par lot) :
                    seuls les compteurs sont calculés et retournés.
    Retourne {label: {"up": n, "down": n}}.
    """
    # Charger les lignes
    with open(lines_json_path, "r", encoding="utf-8") as f:
        lines = json.load(f).get("lines", [])

    # Initialiser les compteurs
    line_counts = {line["label"]: {"up": 0, "down": 0} for line in lines}

    # Agréger les crossings, en ignorant les IDs classifiés comme noise/empty
    skip_classes = {"noise", "empty"}
    if id_dir and os.path.exists(id_dir):
        for entry in os.scandir(id_dir):
            if not entry.is_dir():
                continue
            txt_path = os.path.join(entry.path, "crossings.txt")
            if not os.path.isfile(txt_path):
                continue

            try:
                with open(txt_path, "r", encoding="utf-8") as cf:
                    all_lines = [ln.strip() for ln in cf if ln.strip()]

                # Vérifier la ligne 2 (type de bateau) : ignorer si noise/empty
                if len(all_lines) >= 2:
                    boat_class = all_lines[1].split()[0] if all_lines[1].split() else ""
                    if boat_class.lower() in skip_classes:
                        continue

                for ln in all_lines:
                    parts = ln.split()
                    if len(parts) < 2:
                        continue

                    label, sign_str = parts[0], parts[1]

                    try:
                        sign = int(sign_str)
                    except ValueError:
                        sign = 1 if sign_str.startswith("+") else -1 if sign_str.startswith("-") else 0

                    counts = line_counts.setdefault(label, {"up": 0, "down": 0})
                    if sign > 0:
                        counts["up"] += 1
                    elif sign < 0:
                        counts["down"] += 1

            except Exception as e:
                print(f"Warning reading {txt_path}: {e}")

    elif id_dir:
        print(f"Dossier des crossings introuvable: {id_dir}")

    if not show:
        return line_counts

    # Lire la première frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Impossible de lire la première frame pour la visualisation.")
        return line_counts

    h, w = frame.shape[:2]

    # Constantes de dessin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    margin = 6
    arrow_len = 30
    offset = max(20, int(min(w, h) * 0.05))

    # Dessiner les lignes, flèches et compteurs
    for l in lines:
        p1x, p1y = l["p1"]
        p2x, p2y = l["p2"]
        label = l["label"]

        cv2.line(frame, (p1x, p1y), (p2x, p2y), (0, 0, 255), 2)

        # Calculer la normale
        vx = p2x - p1x
        vy = p2y - p1y
        nx = -vy
        ny = vx
        norm = math.hypot(nx, ny)
        if norm == 0:
            continue
        nx /= norm
        ny /= norm

        midx = (p1x + p2x) // 2
        midy = (p1y + p2y) // 2

        counts = line_counts.get(label, {"up": 0, "down": 0})

        # Flèche descendante (bleu)
        dcx = int(midx - ny * offset)
        dcy = int(midy + nx * offset)
        d_start = (int(dcx - nx * arrow_len), int(dcy - ny * arrow_len))
        d_end = (int(dcx + nx * arrow_len), int(dcy + ny * arrow_len))
        cv2.arrowedLine(frame, d_start, d_end, (255, 0, 0), 2, tipLength=0.3)

        down_text = str(counts["down"])
        (tw, th), _ = cv2.getTextSize(down_text, font, font_scale, thickness)
        tx = max(0, min(w - tw, (d_start[0] + d_end[0]) // 2 - tw // 2))
        ty = max(th, min(h, max(d_start[1], d_end[1]) + margin + th))
        cv2.putText(frame, down_text, (tx, ty), font, font_scale, (255, 0, 0), thickness)

        # Flèche montante (verte)
        ucx = int(midx + ny * offset)
        ucy = int(midy - nx * offset)
        u_start = (int(ucx + nx * arrow_len), int(ucy + ny * arrow_len))
        u_end = (int(ucx - nx * arrow_len), int(ucy - ny * arrow_len))
        cv2.arrowedLine(frame, u_start, u_end, (0, 255, 0), 2, tipLength=0.3)

        up_text = str(counts["up"])
        (tw, th), _ = cv2.getTextSize(up_text, font, font_scale, thickness)
        tx = max(0, min(w - tw, (u_start[0] + u_end[0]) // 2 - tw // 2))
        ty = max(th, min(h, max(u_start[1], u_end[1]) + margin + th))
        cv2.putText(frame, up_text, (tx, ty), font, font_scale, (0, 255, 0), thickness)

    cv2.imshow("Crossings Visualization", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return line_counts
