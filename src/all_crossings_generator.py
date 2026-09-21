import os
from add_dates_to_crossings import add_dates


def generate_all_crossings(video_path, id_dir, output_dir, temp_dir="temp"):
    """
    Génère un fichier all_crossings.txt à partir des fichiers crossings.txt individuels.

    Format attendu dans crossings.txt:
        Ligne 1: gate_label    direction    direction_prob    frame_idx    timestamp
        Ligne 2: boat_type    boat_type_prob

    Où:
        - direction: -1 (avant), 1 (arriere) ou 0 (none)
        - direction_prob: probabilité de la direction (0.0-1.0)
        - boat_type: type de bateau (ex: sailboat, motorboat)
        - boat_type_prob: probabilité du type de bateau (0.0-1.0)
    """
    # Structure: { label: { "avant": int, "arriere": int, "none": int, "per_id": { oid: { infos } } } }
    summary = {}

    if not os.path.exists(id_dir):
        print(f"id_dir not found: {id_dir}")
        return

    for oid in os.listdir(id_dir):
        txt_path = os.path.join(id_dir, oid, "crossings.txt")
        if not os.path.isfile(txt_path):
            continue

        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
                if len(lines) < 2:
                    continue

                # Ligne 1: gate_label direction frame_idx timestamp direction_prob
                first_parts = lines[0].split()
                if len(first_parts) < 5:
                    continue
                
                gate_label = first_parts[0]
                try:
                    direction = int(first_parts[1])  # -1=avant, 1=arriere, 0=none
                    direction_prob = float(first_parts[2])
                    frame_idx = int(first_parts[3])
                    timestamp = int(first_parts[4])
                except (ValueError, IndexError) as e:
                    continue

                # Ligne 2: boat_type boat_type_prob (ou mot seul: noise, empty)
                second_parts = lines[1].split()
                boat_class = second_parts[0] if second_parts else "unknown"

                # Skip des faux positifs et erreurs :
                # - empty : toujours skippé (dossier sans images valides)
                # - noise : toujours skippé (faux positif du type, quel que soit la direction)
                #   Les IDs restants (vrais bateaux) sont affichés, même si direction=none
                if boat_class.lower() in ("noise", "empty"):
                    continue

                if len(second_parts) < 2:
                    boat_type_prob = 0.0
                else:
                    try:
                        boat_type_prob = float(second_parts[1])
                    except ValueError:
                        boat_type_prob = 0.0

                # Initialiser la structure pour cette gate
                rec = summary.setdefault(
                    gate_label,
                    {"avant": 0, "arriere": 0, "none": 0, "per_id": {}}
                )

                # Compter la direction (-1=avant, 1=arriere, 0=none)
                if direction == -1:
                    rec["avant"] += 1
                elif direction == 1:
                    rec["arriere"] += 1
                else:
                    rec["none"] += 1

                # Stocker les infos par ID
                per_id = rec["per_id"].setdefault(
                    oid,
                    {
                        "direction": direction,
                        "boat_class": boat_class,
                        "direction_prob": direction_prob,
                        "boat_type_prob": boat_type_prob,
                        "frame_idx": frame_idx,
                        "timestamp": timestamp,
                    }
                )

        except Exception as e:
            print(f"Warning reading {txt_path}: {e}")

    # Écriture du résumé
    os.makedirs(output_dir, exist_ok=True)
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f"{video_base}_all_crossings.txt")

    with open(out_path, "w", encoding="utf-8") as fo:

        # Global
        fo.write("=== Global ===\n")
        fo.write("line\tavant\tarriere\tnone\ttotal\n")
        for label, rec in sorted(summary.items()):
            avant, arriere, none = rec["avant"], rec["arriere"], rec["none"]
            fo.write(f"{label}\t{avant}\t{arriere}\t{none}\t{avant + arriere + none}\n")

        # Par type de bateau
        all_classes = set()
        for label, rec in summary.items():
            for info in rec["per_id"].values():
                all_classes.add(info["boat_class"])
        
        for boat_class in sorted(all_classes):
            fo.write(f"\n=== {boat_class} ===\n")
            fo.write("line\tavant\tarriere\tnone\ttotal\n")

            class_summary = {}

            for label, rec in summary.items():
                for oid, info in rec["per_id"].items():
                    if info["boat_class"] == boat_class:
                        cs = class_summary.setdefault(label, {"avant": 0, "arriere": 0, "none": 0})
                        if info["direction"] == -1:
                            cs["avant"] += 1
                        elif info["direction"] == 1:
                            cs["arriere"] += 1
                        else:
                            cs["none"] += 1

            for label, cs in sorted(class_summary.items()):
                fo.write(f"{label}\t{cs['avant']}\t{cs['arriere']}\t{cs['none']}\t{cs['avant'] + cs['arriere'] + cs['none']}\n")

        # Détails par ID
        fo.write("\n=== Details per ID ===\n")
        for label, rec in sorted(summary.items()):
            fo.write(f"\n[{label}]\n")
            # Trier par oid (extraction du numéro si format est "line_id_X")
            def extract_id_num(oid_str):
                try:
                    # Essayer de convertir directement en int
                    return int(oid_str)
                except ValueError:
                    # Sinon extraire le dernier nombre après '_'
                    if '_' in oid_str:
                        last_part = oid_str.split('_')[-1]
                        try:
                            return int(last_part)
                        except ValueError:
                            return 0
                    return 0
            
            for oid, info in sorted(rec["per_id"].items(), key=lambda x: extract_id_num(x[0])):
                if info["direction"] == -1:
                    direction_str = "avant"
                elif info["direction"] == 1:
                    direction_str = "arriere"
                else:
                    direction_str = "none"
                fo.write(
                    f"{oid}\t{direction_str}\t{info['boat_class']}\t{info['direction_prob']:.4f}\t{info['boat_type_prob']:.4f}\tframe_idx={info['frame_idx']}, ts={info['timestamp']}\n"
                )

    # Ajout des dates aux crossings
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    lines_date_path = os.path.join(temp_dir, f"{video_base}_lines_date.json")
    add_dates(extractions_dir=id_dir, lines_date_path=lines_date_path, output_dir=output_dir)
