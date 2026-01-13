import cv2
import os 
import json
import sys
import argparse
from tracker_utils import detect_rectangles
from ResNet_direction import get_direction

# Paramètres
GATE_WIDTH = 100  # Largeur de la zone de détection (gate) autour de la ligne
GATE_TIMEOUT = 5  # Temps en nombre de frames pour considérer qu'un objet a quitté la zone    
MIN_SIDE_CROP = 20  # Taille minimale d'un côté du crop pour le sauvegarder
REAL_FPS = 1/5                 # Fréquence réelle (1 image toutes les 5 secondes)


def create_gates_from_lines(lines_path):
    """
    À chaque ligne associe une gate rectangulaire autour de la ligne.
    Renvoie un dictionnaire {line_label: gate_rectangle}.
    """
    with open(lines_path, "r") as file:
        data = json.load(file)

    gates = {}
    for line in data.get("lines", []):
        p1, p2 = line.get("p1"), line.get("p2")
        if not p1 or not p2:
            continue

        x1, y1, x2, y2 = float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1])
        dx, dy = x2 - x1, y2 - y1
        length = (dx ** 2 + dy ** 2) ** 0.5

        if length == 0:  # Ce cas ne devrait pas se produire, mais on le gère quand même
            gate = (
                x1 - GATE_WIDTH / 2,
                y1 - GATE_WIDTH / 2,
                x1 + GATE_WIDTH / 2,
                y1 + GATE_WIDTH / 2,
            )
        else:
            nx, ny = -dy / length, dx / length
            gate_width_half = GATE_WIDTH / 2
            x1_new, y1_new = x1 + nx * gate_width_half, y1 + ny * gate_width_half
            x2_new, y2_new = x2 + nx * gate_width_half, y2 + ny * gate_width_half
            x3_new, y3_new = x2 - nx * gate_width_half, y2 - ny * gate_width_half
            x4_new, y4_new = x1 - nx * gate_width_half, y1 - ny * gate_width_half
            min_x = min(x1_new, x2_new, x3_new, x4_new)
            max_x = max(x1_new, x2_new, x3_new, x4_new)
            min_y = min(y1_new, y2_new, y3_new, y4_new)
            max_y = max(y1_new, y2_new, y3_new, y4_new)
            gate = (min_x, min_y, max_x, max_y)

        label = line.get("label")
        gates[label] = gate if label is not None else "unknown"

    return gates


def does_rectangle_hit_gate(rect_detected, rect_gate):  
    
    if rect_detected is None or rect_gate is None:
        return False

    x1, y1, x2, y2 = (float(v) for v in rect_detected)
    gx1, gy1, gx2, gy2 = (float(v) for v in rect_gate)

    x_min, x_max = sorted((x1, x2))   # On s'assure que x_min < x_max, car les coins de rect_detected peuvent être dans n'importe quel ordre a priori
    y_min, y_max = sorted((y1, y2))
    gx_min, gx_max = sorted((gx1, gx2))
    gy_min, gy_max = sorted((gy1, gy2))

    return not (x_max < gx_min or gx_max < x_min or y_max < gy_min or gy_max < y_min)


# detect_rectangles() est dans tracker_utils.py et renvoie une liste de rectangles detectés dans une frame donnée

def extract_and_save_crops(color_src, frame_annotations, temp_dir):  

    cap_color = cv2.VideoCapture(color_src)

    if not cap_color.isOpened():
        print("Attention: Vidéo couleur non disponible pour les crops.")
        return
    
    extra_root = os.path.join(temp_dir, "extractions")
    os.makedirs(extra_root, exist_ok=True)
    
    frame_idx = 0
    save_counts = {}
    first_crops = {}

    while True:
        ret, frame = cap_color.read()
        if not ret: break

        detections = frame_annotations[frame_idx] if frame_idx < len(frame_annotations) else []
        if detections:
            h, w = frame.shape[:2]
            for gate_label, oid, bbox in detections:
                x1, y1, x2, y2 = (int(round(v)) for v in bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if (x2 - x1) <= MIN_SIDE_CROP or (y2 - y1) <= MIN_SIDE_CROP:
                    continue

                crop = frame[y1:y2, x1:x2]
                oid_dir = os.path.join(extra_root, f"{gate_label}_id_{oid}")
                os.makedirs(oid_dir, exist_ok=True)

                save_key = (gate_label, oid)
                if save_key not in first_crops:
                    first_crops[save_key] = frame_idx

                cnt = save_counts.get(oid, 0)
                fname = f"{frame_idx}_{cnt}.jpg"
                cv2.imwrite(os.path.join(oid_dir, fname), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                save_counts[save_key] = cnt + 1

        frame_idx += 1
    cap_color.release()

    return first_crops


def write_crossings_files(first_crops, temp_dir):

    extra_root = os.path.join(temp_dir, "extractions")

    for (gate_label, oid), frame_idx in sorted(first_crops.items()):
        
        oid_dir = os.path.join(extra_root, f"{gate_label}_id_{oid}")

        seconds = frame_idx / REAL_FPS   ### Il faudra ajouter ici l'appel aux 2 ResNet (ou 1 + id_to_boat.py) pour obtenir le sens du bateau et la classe du bateau avec sa proba ###
        direction = get_direction(oid_dir)  # On prédit la direction du bateau à l'aide des images extraites

        with open(os.path.join(oid_dir, "crossings.txt"), "w", encoding="utf-8") as f:
            f.write(f"{gate_label}\t{direction}\t{frame_idx}\t{int(seconds)}\n")


def parse_args():
    p = argparse.ArgumentParser(description="Line Gate Capture algorithm")
    p.add_argument("--video", "-v", default="0", help="Path to video file")
    p.add_argument("--color_video", "-c", default=None, help="Path to real color video to extract crops from (optional). If omitted uses --video")
    p.add_argument("--lines_json", default=None, help="Path to lines.json file (required for crossings)")
    return p.parse_args()


def main():
    args = parse_args()
    temp_dir = "./temp"
    lines_path = args.lines_json
    if not lines_path or not os.path.exists(lines_path):
        sys.exit("Erreur: fichier *_lines.json introuvable.")

    gates = create_gates_from_lines(lines_path)
    if not gates:
        sys.exit("Erreur: aucune ligne pour définir des gates.")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Erreur: Impossible d'ouvrir {args.video}")


    frame_annotations = []  # liste [(gate, id, bbox), ... ] par frame
    gates_states = { 
        label: {'next_id': 0, 'active_id': None, 'frames_since_contact': GATE_TIMEOUT}
        for label in gates
    }

    while True:
        ret, gray_frame = cap.read()
        if not ret:
            break

        # Détection
        detections = detect_rectangles(gray_frame)
        current_annotations = []

        for gate_label, gate in gates.items():
            hits = [rect for rect in detections if does_rectangle_hit_gate(rect, gate)]
            state = gates_states[gate_label]
                    
            if hits:
                if state['active_id'] is None or state['frames_since_contact'] >= GATE_TIMEOUT:
                    state['active_id'] = state['next_id']
                    state['next_id'] += 1
                state['frames_since_contact'] = 0
                for rect in hits:
                    current_annotations.append((gate_label, state['active_id'], rect))
            else:
                if state['active_id'] is not None:
                    state['frames_since_contact'] += 1
                    if state['frames_since_contact'] >= GATE_TIMEOUT:
                        state['active_id'] = None
                        state['frames_since_contact'] = GATE_TIMEOUT

        frame_annotations.append(current_annotations)

    cap.release()

    color_src = args.color_video if args.color_video else args.video
    first_crops = extract_and_save_crops(color_src, frame_annotations, temp_dir)
    write_crossings_files(first_crops, temp_dir)

    ## puis ici etape d'export des données dans les fichiers crossings (pour calcul date) ##

    print("LGC terminé.")

if __name__ == "__main__":
    main()


    
 


