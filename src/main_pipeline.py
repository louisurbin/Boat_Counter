import argparse
import os
import sys
import subprocess
from preprocess_mask_lines import create_mask_and_lines
from apply_mask import apply_mask_to_video


### IMPORTANT : -> Les paramètres MOG2 sont modifiables dans 'mog2_background_subtraction.py' -> VAR_THRESHOLD est le plus important ###
###             -> Les paramètres de suivi sont modifiables dans 'sort_tracker.py' -> MIN_TRACK_AREA et MAX_DISTANCE sont importants ###
###             -> Supprimer ./temp/extractions/ avant de relancer le pipeline (sinon pb pour compter) ###
###             -> Dans les fichiers de crossings.txt, +1 signifie que le bateau "monte", -1 signifie qu'il "descend" ###
###             -> Lors du tracé des lignes, toujours tracer la ligne dans le sens "bas-gauche" vers "haut-droit" (pour cohérence des normales) ###
###             -> Si la video n'est pas en 1 image toutes les 5 secondes, modifier 'fps' au début de 'sort_tracker.py' ###

### Exemple d'utilisation : python3 ./src/main_pipeline.py ./data/input_video.mp4 --out output_directory ###


# Helper to get mask and lines paths from video and output dir
def get_mask_and_lines_paths(video_path, out_dir):
    base = os.path.splitext(os.path.basename(video_path))[0]
    mask_path = os.path.join(out_dir, f"{base}_mask.png")
    lines_path = os.path.join(out_dir, f"{base}_lines_date.json")
    return mask_path, lines_path

def main(video_path, output_dir):
    # Step 1: Create mask and lines interactively
    print("Step 1: Creating mask and lines...")
    create_mask_and_lines(video_path, output_dir)
    mask_path, lines_path = get_mask_and_lines_paths(video_path, output_dir)
    if not os.path.exists(mask_path):
        print(f"Mask not found: {mask_path}")
        sys.exit(1)

    # Step 2: Apply mask to video
    print("Step 2: Applying mask to video...")
    masked_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_masked.mp4")
    apply_mask_to_video(video_path, mask_path, masked_video_path)

    # Step 3: Apply MOG2 background subtraction via script
    print("Step 3: Applying MOG2 background subtraction...")
    mog2_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_mog2.mp4")
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "mog2_background_subtraction.py"), "-i", masked_video_path, "-o", mog2_output_path]
    subprocess.run(cmd, check=True)

    # Step 4: Apply sort_tracker to count crossings
    print("Step 4: Applying sort_tracker...")
    tracked_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_tracked.mp4")
    cmd_sort = [sys.executable, os.path.join(os.path.dirname(__file__), "sort_tracker.py"),
               "--video", mog2_output_path,
               "--lines_json", lines_path,
               "--save", tracked_video_path,
               "--color", os.path.abspath(video_path)]
    subprocess.run(cmd_sort, check=True)

    # Generate per-id crossings with datetime and aggregated all_crossings in ./temp
    print("Step 5: Visualizing crossings...")
    crossings_dir = os.path.abspath(os.path.join(".", "temp", "extractions"))
    visualize_line_crossings(video_path, lines_path, crossings_dir)

    # --- NEW: aggregate all per-id crossings and write summary to output_dir ---
    summary = {}  # label -> {'up':int,'down':int,'per_id':{id:[signs]}}
    if os.path.exists(crossings_dir):
        for entry in os.listdir(crossings_dir):
            txt_path = os.path.join(crossings_dir, entry, "crossings.txt")
            if not os.path.isfile(txt_path):
                continue
            try:
                with open(txt_path, "r", encoding="utf-8") as cf:
                    for ln in cf:
                        ln = ln.strip()
                        if not ln:
                            continue
                        parts = ln.split()
                        if len(parts) < 2:
                            continue
                        label = parts[0]
                        sign_str = parts[1]
                        try:
                            sign = int(sign_str)
                        except Exception:
                            sign = 1 if sign_str.startswith("+") else (-1 if sign_str.startswith("-") else 0)
                        rec = summary.setdefault(label, {"up": 0, "down": 0, "per_id": {}})
                        # Contrainte actuelle : +1 -> "up" (vert), +1 -> "down" (bleu)
                        if sign < 0:
                            rec["down"] += 1
                        elif sign > 0:
                            rec["up"] += 1
                        # store per-id detail
                        oid_key = str(entry)
                        rec["per_id"].setdefault(oid_key, []).append(sign)
            except Exception as e:
                print(f"Warning reading {txt_path}: {e}")

    # write summary file in output_dir
    try:
        os.makedirs(output_dir, exist_ok=True)
        # include source video base name in summary filename
        video_base = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(output_dir, f"{video_base}_all_crossings.txt")
        with open(out_path, "w", encoding="utf-8") as fo:
            fo.write("line\tup\tdown\ttotal\n")
            for label, rec in sorted(summary.items()):
                up = rec.get("up", 0)
                down = rec.get("down", 0)
                total = up + down
                fo.write(f"{label}\t{up}\t{down}\t{total}\n")
            fo.write("\n# Details per id\n")
            for label, rec in sorted(summary.items()):
                fo.write(f"\n[{label}]\n")
                per_items = rec.get("per_id", {})
                def sort_key_item(kv):
                    k = kv[0]
                    try:
                        return (0, int(k))
                    except Exception:
                        return (1, k)
                for oid, signs in sorted(per_items.items(), key=sort_key_item):
                    fo.write(f"{oid}\t{', '.join(str(s) for s in signs)}\n")
        # summary written (no debug printing)
    except Exception as e:
        print(f"Warning: could not write summary file: {e}")
    # --- end new block ---

    # After summary file created, update per-id crossings and existing all_crossings files in temp
    try:
        cmd_dates = [sys.executable, os.path.join(os.path.dirname(__file__), "add_dates_to_crossings.py"),
                     "--extractions", os.path.join(".", "temp", "extractions"),
                     "--temp", os.path.join(".", "temp")]
        subprocess.run(cmd_dates, check=True)
    except Exception as e:
        print(f"Warning: add_dates_to_crossings failed: {e}")

    # Clean up non-essential temp files
    try:
        for fname in os.listdir(output_dir):
            fpath = os.path.join(output_dir, fname)
            if not os.path.isfile(fpath):
                continue
            # keep trajectories.mp4
            if fname == "trajectories.mp4":
                continue
            low = fname.lower()
            # candidate mask files to remove: contain '_mask', '_mog2', '_tracked' or end with '_masked.mp4'
            if (
                "_mask" in low
                or "_mog2" in low
                or "_tracked" in low
                or low.endswith("_masked.mp4")
            ):
                try:
                    os.remove(fpath)
                    print(f"Deleted: {fpath}")
                except Exception as e:
                    print(f"Warning: Could not delete file {fpath}: {e}")
    except Exception as e:
        print(f"Warning during cleanup of masks in {output_dir}: {e}")

def visualize_line_crossings(video_path, lines_json_path, crossings_dir):
    import cv2
    import numpy as np
    import json
    import os
    # Read first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Could not read first frame for visualization.")
        return
    # Load lines
    with open(lines_json_path, "r", encoding="utf-8") as f:
        lines = json.load(f).get("lines", [])
    # Initialize counts per line (will be filled by reading crossings files)
    line_counts = {line["label"]: {"up": 0, "down": 0} for line in lines}
    # Aggregate crossings from per-id files if provided
    files_read = 0
    if crossings_dir:
        if not os.path.exists(crossings_dir):
            print(f"Crossings directory not found: {crossings_dir}")
        else:
            for entry in os.listdir(crossings_dir):
                txt_path = os.path.join(crossings_dir, entry, "crossings.txt")
                if not os.path.isfile(txt_path):
                    continue
                try:
                    with open(txt_path, "r", encoding="utf-8") as cf:
                        files_read += 1
                        for ln in cf:
                            ln = ln.strip()
                            if not ln:
                                continue
                            parts = ln.split()
                            if len(parts) < 2:
                                continue
                            label = parts[0]
                            sign_str = parts[1]
                            try:
                                sign = int(sign_str)
                            except Exception:
                                sign = 1 if sign_str.startswith("+") else (-1 if sign_str.startswith("-") else 0)
                            if label not in line_counts:
                                line_counts.setdefault(label, {"up": 0, "down": 0})
                            # Convention : +1 incrémente la flèche verte ("up"), -1 incrémente la flèche bleue ("down")
                            if sign < 0:
                                line_counts[label]["down"] += 1
                            elif sign > 0:
                                line_counts[label]["up"] += 1
                except Exception as e:
                    print(f"Warning reading {txt_path}: {e}")
            # aggregation computed (no debug printing)
    else:
        print("No crossings_dir provided, using zero counts.")

    # Draw lines, arrows and counts using aggregated line_counts
    for l in lines:
        p1 = tuple(l["p1"])
        p2 = tuple(l["p2"])
        label = l["label"]
        cv2.line(frame, p1, p2, (0,0,255), 2)
        # Compute normal vector (bottom-top)
        line_vec = np.array(p2, dtype=float) - np.array(p1, dtype=float)
        normal = np.array([-line_vec[1], line_vec[0]], dtype=float)
        normal = normal / (np.linalg.norm(normal) + 1e-8)
        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
        # Offset to separate arrows (relative)
        h, w = frame.shape[:2]
        offset = max(20, int(min(w, h) * 0.05))
        # Arrow down (bottom-top) shifted + offset (blue = "down")
        down_center = (int(mid[0] - normal[1]*offset), int(mid[1] + normal[0]*offset))
        arrow_start = (int(down_center[0] - normal[0]*30), int(down_center[1] - normal[1]*30))
        arrow_end = (int(down_center[0] + normal[0]*30), int(down_center[1] + normal[1]*30))
        cv2.arrowedLine(frame, arrow_start, arrow_end, (255,0,0), 2, tipLength=0.3)
        # Place the 'down' count under the blue arrow
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        margin = 6
        down_text = f"{line_counts.get(label, {}).get('down', 0)}"
        (tw, th), baseline = cv2.getTextSize(down_text, font, font_scale, thickness)
        down_mid_x = (arrow_start[0] + arrow_end[0]) // 2
        down_bottom_y = max(arrow_start[1], arrow_end[1])
        down_text_x = int(down_mid_x - tw / 2)
        down_text_y = int(down_bottom_y + margin + th)
        down_text_x = max(0, min(w - tw, down_text_x))
        down_text_y = max(th, min(h, down_text_y))
        cv2.putText(frame, down_text, (down_text_x, down_text_y), font, font_scale, (255,0,0), thickness)
        # Arrow up (top-bottom) shifted - offset (green = "up")
        up_center2 = (int(mid[0] + normal[1]*offset), int(mid[1] - normal[0]*offset))
        arrow_start2 = (int(up_center2[0] + normal[0]*30), int(up_center2[1] + normal[1]*30))
        arrow_end2 = (int(up_center2[0] - normal[0]*30), int(up_center2[1] - normal[1]*30))
        cv2.arrowedLine(frame, arrow_start2, arrow_end2, (0,255,0), 2, tipLength=0.3)
        # Place the 'up' count under the green arrow
        up_text2 = f"{line_counts.get(label, {}).get('up', 0)}"
        (uw2, uh2), ubaseline2 = cv2.getTextSize(up_text2, font, font_scale, thickness)
        up_mid_x2 = (arrow_start2[0] + arrow_end2[0]) // 2
        up_bottom_y2 = max(arrow_start2[1], arrow_end2[1])
        up_text_x2 = int(up_mid_x2 - uw2 / 2)
        up_text_y2 = int(up_bottom_y2 + margin + uh2)
        up_text_x2 = max(0, min(w - uw2, up_text_x2))
        up_text_y2 = max(uh2, min(h, up_text_y2))
        cv2.putText(frame, up_text2, (up_text_x2, up_text_y2), font, font_scale, (0,255,0), thickness)
    cv2.imshow("Crossings Visualization", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def _signed_distance_to_line(pt, p1, p2):
    """
    Retourne la distance signée du point pt à la ligne définie par p1->p2.
    Sign convention: positive si pt est du côté de la normale (rot90 de la direction p1->p2).
    pt, p1, p2: tuples ou listes (x,y)
    """
    import numpy as np
    p = np.array(pt, dtype=float)
    a = np.array(p1, dtype=float)
    b = np.array(p2, dtype=float)
    v = b - a
    # normale orthogonale
    normal = np.array([-v[1], v[0]], dtype=float)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-8:
        return 0.0
    normal_unit = normal / n_norm
    return float(np.dot(p - a, normal_unit))

def signed_distance_to_line(pt, p1, p2):
	"""
	Retourne la distance signée du point pt à la ligne (p1->p2).
	pt, p1, p2: (x,y) ou listes/np.array
	Convention: normale = (-v_y, v_x). Positive = côté 'up'.
	"""
	import numpy as np
	p = np.array(pt, dtype=float)
	a = np.array(p1, dtype=float)
	b = np.array(p2, dtype=float)
	v = b - a
	normal = np.array([-v[1], v[0]], dtype=float)
	norm = np.linalg.norm(normal)
	if norm < 1e-8:
		return 0.0
	normal_unit = normal / norm
	return float(np.dot(p - a, normal_unit))

def detect_crossing(prev_pt, curr_pt, p1, p2, min_abs_delta=2.0, check_segment_intersection=True):
	"""
	Detecte si le déplacement prev_pt->curr_pt traverse la ligne p1->p2.
	Retour: (crossed: bool, direction: "up"/"down"/None, intersection_point or None)
	- min_abs_delta: seuil minimal sur |d_curr - d_prev| pour éviter comptage bruité.
	- check_segment_intersection: si True, vérifie que les segments se coupent réellement.
	Usage: copier dans sort_tracker.update() et appeler avant d'incrémenter compteurs.
	"""
	import numpy as np
	A = np.array(prev_pt, dtype=float)
	B = np.array(curr_pt, dtype=float)
	C = np.array(p1, dtype=float)
	D = np.array(p2, dtype=float)

	d_prev = signed_distance_to_line(A, C, D)
	d_curr = signed_distance_to_line(B, C, D)

	# quick reject by small delta
	if abs(d_curr - d_prev) < min_abs_delta:
		return False, None, None

	if d_prev * d_curr >= 0:
		# same side -> no crossing
		return False, None, None

	# optional: check segment intersection for robustness
	intersection = None
	if check_segment_intersection:
		BA = B - A
		DC = D - C
		den = BA[0]*DC[1] - BA[1]*DC[0]
		if abs(den) >= 1e-8:
			t = ((C[0]-A[0])*DC[1] - (C[1]-A[1])*DC[0]) / den
			u = ((C[0]-A[0])*BA[1] - (C[1]-A[1])*BA[0]) / (-den)
			if 0 <= t <= 1 and 0 <= u <= 1:
				intersection = A + t * BA
			else:
				# numeric or timing issue: still consider crossing if sign changed but no segment intersection
				intersection = None
	# direction: convention choisie ici -> d_curr > d_prev => "up"
	direction = "up" if d_curr > d_prev else "down"
	return True, direction, (None if intersection is None else (float(intersection[0]), float(intersection[1])))

def check_crossing(prev_pt, curr_pt, p1, p2):
    """
    Détecte si le segment prev_pt->curr_pt traverse la ligne p1->p2.
    Retour: (crossed: bool, direction: "up"/"down"/None)
    - crossed True si signes des distances sont opposés.
    - direction déterminée par le signe de distance actuelle (convention: positive => "up").
    Exemple d'intégration (dans sort_tracker) :
      d_prev = _signed_distance_to_line(prev_centroid, p1, p2)
      d_curr = _signed_distance_to_line(curr_centroid, p1, p2)
      crossed, direction = check_crossing(prev_centroid, curr_centroid, p1, p2)
    """
    d_prev = _signed_distance_to_line(prev_pt, p1, p2)
    d_curr = _signed_distance_to_line(curr_pt, p1, p2)
    # traversée si signes différents et variation non nulle
    if d_prev * d_curr < 0:
        direction = "up" if d_curr > d_prev else "down"
        # Alternativement, on peut utiliser d_curr>0 => "up" else "down" selon convention
        return True, direction
    return False, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main pipeline: preprocess mask/lines, apply mask, MOG2 background subtraction.")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--out", default="temp", help="Output directory")
    args = parser.parse_args()
    main(args.video, args.out)
