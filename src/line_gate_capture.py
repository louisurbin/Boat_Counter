### Lors de la generalisation à plusieurs lignes, il faudra aussi gerer la date de premiere intersection stocké dans un fichier .txt 
# qui sera ensuite utiliser pour former all_crossings à l'aide de all_crossings_generator.
# il faudra aussi gerer où est ce qu'on stocke le sens et la classe de l'id (surement un fichier crossings.txt sous la meme forme que celui en sortie du tracker)
# Pour tout cela on pourra reutiliser add_dates_to_crossings.py, all_crossings_generator.py et puis res_net pour la direction et la classe (on fera 2 fonction differente en fonction d'un argument d'entrée)
###

import cv2
import os 
import json
import sys
import argparse
from preprocess_mask_lines_date import get_lines_date_path
from tracker_utils import detect_rectangles

# Paramètres
GATE_WIDTH = 100  # Largeur de la zone de détection autour de la ligne
GATE_TIMEOUT = 5  # Temps en nombre de frames pour considérer qu'un objet a quitté la zone    
MIN_SIDE_CROP = 20  # Taille minimale d'un côté du crop pour le sauvegarder

# On commence par former le recctangle englobant la ligne qu'on lit dans le .json   ### on traitera plus tard le cas avec plusieurs lignes et donc plusieurs rectangles ###
def create_gate_from_line(line_path):
    """
    Crée une zone de détection (rectangle) autour d'une ligne donnée.
    """
    with open(line_path, 'r') as file:
        data = json.load(file)
    line_coords = (data['lines'][0]['p1'][0], data['lines'][0]['p1'][1],    ### a changer si plusieurs lignes ###
                   data['lines'][0]['p2'][0], data['lines'][0]['p2'][1])

    x1, y1, x2, y2 = line_coords
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2)**0.5
    if length == 0:      # Ce cas ne devrait pas se produire, mais on le gère quand même
        return (x1 - GATE_WIDTH // 2, y1 - GATE_WIDTH // 2,
                x1 + GATE_WIDTH // 2, y1 + GATE_WIDTH // 2)
    nx = -dy / length
    ny = dx / length
    gate_width_half = GATE_WIDTH / 2       
    # voir si on ne peut pas faire ce calcul matriciellement pour plusieurs lignes en même temps et plus rapidement
    x1_new = x1 + nx * gate_width_half
    y1_new = y1 + ny * gate_width_half
    x2_new = x2 + nx * gate_width_half
    y2_new = y2 + ny * gate_width_half
    x3_new = x2 - nx * gate_width_half
    y3_new = y2 - ny * gate_width_half
    x4_new = x1 - nx * gate_width_half
    y4_new = y1 - ny * gate_width_half
    min_x = min(x1_new, x2_new, x3_new, x4_new)
    max_x = max(x1_new, x2_new, x3_new, x4_new)
    min_y = min(y1_new, y2_new, y3_new, y4_new)
    max_y = max(y1_new, y2_new, y3_new, y4_new)
    gate = (min_x, min_y, max_x, max_y)
    return gate


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

def extract_and_save_crops(color_src, frame_annotations, temp_dir):  # frame_annotations : liste d'ensemble de rectangles détectés par frame

    cap_color = cv2.VideoCapture(color_src)

    if not cap_color.isOpened():
        print("Attention: Vidéo couleur non disponible pour les crops.")
        return
    
    extra_root = os.path.join(temp_dir, "extractions")
    os.makedirs(extra_root, exist_ok=True)
    
    frame_idx = 0
    save_counts = {}

    while True:
        ret, frame = cap_color.read()
        if not ret: break

        detections = frame_annotations[frame_idx] if frame_idx < len(frame_annotations) else []
        if detections:
            h, w = frame.shape[:2]
            for oid, bbox in detections:
                x1, y1, x2, y2 = (int(round(v)) for v in bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if (x2 - x1) <= MIN_SIDE_CROP or (y2 - y1) <= MIN_SIDE_CROP:
                    continue

                crop = frame[y1:y2, x1:x2]
                oid_dir = os.path.join(extra_root, f"id_{oid:04d}")
                os.makedirs(oid_dir, exist_ok=True)

                cnt = save_counts.get(oid, 0)
                fname = f"{frame_idx:06d}_{cnt}.jpg"
                cv2.imwrite(os.path.join(oid_dir, fname), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                save_counts[oid] = cnt + 1

        frame_idx += 1
    cap_color.release()



# frame_annotations est directement formé dans le main à l'aide de detect_rectangles à chaque frame lue de la vidéo prétraitée par MOG2 et de la logique expliquée dans le main (timeout etc...)

def parse_args():
    p = argparse.ArgumentParser(description="Line gate capture algorithm")
    p.add_argument("--video", "-v", default="0", help="Path to video file")
    p.add_argument("--color_video", "-c", default=None, help="Path to real color video to extract crops from (optional). If omitted uses --video")
    #p.add_argument("--lines_json", default=None, help="Path to lines.json file (required for crossings)")
    return p.parse_args()


def main():
    args = parse_args()
    temp_dir = "./temp"
    line_path = get_lines_date_path(temp_dir)
    if not line_path or not os.path.exists(line_path):
        sys.exit("Erreur: fichier *_lines.json introuvable.")

    gate = create_gate_from_line(line_path)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Erreur: Impossible d'ouvrir {args.video}")


    frame_annotations = []  # liste [(id, bbox), ... ] par frame
    next_id = 1
    active_id = None
    frames_since_contact = GATE_TIMEOUT

    while True:
        ret, gray_frame = cap.read()
        if not ret:
            break

        # Détection
        detections = detect_rectangles(gray_frame)
        hits = [rect for rect in detections if does_rectangle_hit_gate(rect, gate)]
                
        if hits:
            if active_id is None or frames_since_contact >= GATE_TIMEOUT:
                active_id = next_id
                next_id += 1
            frames_since_contact = 0
            frame_annotations.append([(active_id, rect) for rect in hits])
        else:
            if active_id is not None:
                frames_since_contact += 1
                if frames_since_contact >= GATE_TIMEOUT:
                    active_id = None
                    frames_since_contact = GATE_TIMEOUT
            frame_annotations.append([])

    cap.release()

    color_src = args.color_video if args.color_video else args.video
    extract_and_save_crops(color_src, frame_annotations, temp_dir)

    ## puis ici etape d'export des données dans les fichiers crossings (pour calcul date) ##

    print("LGC terminé.")

if __name__ == "__main__":
    main()

    # puis dans la boucle : frame_annotations.append(dict(objects)), l'idée est qu'à chaque id soit asso un ensemble de bboxs

    # a present il faut parcourir frame à frame la video qui a subit mog2, ie en sortie de mog2_background subtraction. pour chaque frame, on detecte les rectangles (a l'aide de la fonction detect_rectangles), puis on verifie si ils intersectent avec la gate.
    # Des qu'on detecte un rectangle qui intersecte la gate, on lui cree un repertoire id dans ./temp/extractions comme on le fait avec le tracker.
    # Ensuite tous les autres rectangles qui intersectent la gate dans les Gate_TimeOut frames(ou seconde) suivante sont aussisauvergadé sous le meme id.
    # Une fois timeout, il faut considerer que l'objet a quitté la zone, et donc si un rectangle intersecte de nouveau la gate, on cree un nouveau repertoire id dans ./temp/extractions.
    # On pourra donc avoir une variable dans notre boucle du genre : id_exists = False si pas encore de detection et True si un id crée, puis au bout de timeout frames, revient à False.
    # Il faut preciser ce qu'on entend par sauvegarder les rectangles dans les dossiers id. En fait il s'agit d'extraire et enregister le crop dans le dossier correspondant. cela se fait via la fonction extract_and_save_crops() 
    
 


