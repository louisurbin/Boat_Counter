import argparse
import json
import os
import sys
import shutil
import subprocess
from preprocess_mask_lines_date import create_mask_lines_date, get_mask_lines_date_paths
from all_crossings_generator import generate_all_crossings
from visualization_utils import visualize_line_crossings

### Exemple : python3 ./src/main_pipeline.py --in ./videos/input_video.mp4 --out ./output ###
### Sans aucune interaction : ajouter --mask m.png --lines_json l.json [--start_time "06/22 05:00:04"] --no_display ###

def use_existing_mask_and_lines(video_path, temp_dir, mask_src, lines_src, start_time=None):
    """
    Mode automatique : copie un masque + un fichier lignes/date existants dans temp/
    (mêmes noms que ceux produits par l'éditeur), en remplaçant start_time si fourni.
    """
    mask_path, lines_path = get_mask_lines_date_paths(video_path, temp_dir)
    shutil.copyfile(mask_src, mask_path)
    with open(lines_src, "r", encoding="utf-8") as f:
        meta = json.load(f)
    meta["video"] = os.path.basename(video_path)
    if start_time:
        meta["start_time"] = start_time
    with open(lines_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main(video_path, output_dir, temp_dir, mask_src=None, lines_src=None, start_time=None, show=True):
    """
    Orchestre le pipeline complet (mode LGC).

    mask_src / lines_src : masque et fichier lignes/date déjà prêts -> l'éditeur interactif n'est pas ouvert.
    start_time           : "MM/DD HH:MM:SS", remplace celui du fichier lignes/date (avec mask_src/lines_src).
    show                 : False -> aucune fenêtre ni image de fin (seuls les compteurs sont calculés).
    Retourne les compteurs {ligne: {"up": n, "down": n}}.
    """
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Repartir d'un dossier d'extractions vide : évite de mélanger les crops d'une exécution
    # précédente interrompue avec ceux de cette vidéo.
    stale_extractions = os.path.join(temp_dir, "extractions")
    if os.path.exists(stale_extractions):
        shutil.rmtree(stale_extractions)

    # Step 1: Création du masque et des lignes (intermédiaire -> temp/)
    print("Step 1: Creating mask and lines...")
    if mask_src and lines_src:
        use_existing_mask_and_lines(video_path, temp_dir, mask_src, lines_src, start_time)
    else:
        create_mask_lines_date(video_path, temp_dir)
    mask_path, lines_path = get_mask_lines_date_paths(video_path, temp_dir)
    if not os.path.exists(mask_path):
        print(f"Masque introuvable: {mask_path}")
        sys.exit(1)

    # Step 2: Soustraction de fond MOG2 (intermédiaire -> temp/)
    print("Step 2: Applying MOG2 background subtraction...")
    mog2_output_path = os.path.join(temp_dir, f"{video_base}_mog2.mp4")
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "mog2_background_subtraction.py"),
           "-i", video_path, "-o", mog2_output_path, "-m", mask_path]
    subprocess.run(cmd, check=True)

    # Step 3: LineGateCapture (extractions -> temp/extractions/)
    print("Step 3: Line Gate Capture...")
    cmd_lgc = [sys.executable, os.path.join(os.path.dirname(__file__), "line_gate_capture.py"),
               "--video", mog2_output_path,
               "--color_video", os.path.abspath(video_path),
               "--lines_json", lines_path,
               "--temp", temp_dir]
    subprocess.run(cmd_lgc, check=True)

    # Step 4: Classification des types de bateaux et direction (déjà fait dans LGC)
    # La classification direction + type est maintenant intégrée dans line_gate_capture.py
    # via shared_dino.process_all_ids() - UN SEUL forward DINO par ID au lieu de deux.
    print("Step 4: Boat type and direction classification (integrated in LGC)")

    # Définir id_dir pour les étapes suivantes
    id_dir = os.path.abspath(os.path.join(temp_dir, "extractions"))

    # Step 5: Génération du résumé all_crossings (final -> output/)
    print("Step 5: Generating all_crossings summary...")
    generate_all_crossings(video_path, id_dir, output_dir, temp_dir)

    # Step 6: Visualisation des crossings (après filtrage noise)
    print("Step 6: Visualizing crossings...")
    counts = visualize_line_crossings(video_path, lines_path, id_dir, show=show)

    # Step 7: Nettoyage — déplacer extractions vers output/, supprimer les intermédiaires de temp/
    print("Step 7: Cleaning up...")
    # Renommer temp/extractions/ en output/extractions_<video>/
    extractions_new = os.path.abspath(os.path.join(output_dir, f"extractions_{video_base}"))
    if os.path.exists(id_dir):
        if os.path.exists(extractions_new):
            shutil.rmtree(extractions_new)
        os.rename(id_dir, extractions_new)
        print(f"Moved: {id_dir} -> {extractions_new}")

    # Supprimer les fichiers intermédiaires de temp/
    try:
        for fname in os.listdir(temp_dir):
            fpath = os.path.join(temp_dir, fname)
            if not os.path.isfile(fpath):
                continue
            low = fname.lower()
            if (
                "_mask" in low
                or "_mog2" in low
                or "_lines_date" in low
            ):
                try:
                    os.remove(fpath)
                    print(f"Deleted: {fpath}")
                except Exception as e:
                    print(f"Warning: Could not delete file {fpath}: {e}")
    except Exception as e:
        print(f"Warning during cleanup in {temp_dir}: {e}")

    print("Pipeline terminé.")
    return counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main pipeline: preprocess mask/lines/date, MOG2 background subtraction, LGC, classification.")
    parser.add_argument("--video", "--in", "-i", dest="video", help="Path to input video")
    parser.add_argument("--out", "-o", default="output", help="Output directory for final results")
    parser.add_argument("--temp", default="temp", help="Temp directory for intermediate files")
    parser.add_argument("--mask", default=None, help="Masque existant à réutiliser (avec --lines_json) : n'ouvre pas l'éditeur")
    parser.add_argument("--lines_json", default=None, help="Fichier lignes/date existant à réutiliser (avec --mask)")
    parser.add_argument("--start_time", default=None, help="Date de début 'MM/DD HH:MM:SS' (remplace celle de --lines_json)")
    parser.add_argument("--no_display", action="store_true", help="N'ouvre aucune fenêtre à la fin (pas d'image de fin)")
    args = parser.parse_args()
    if bool(args.mask) != bool(args.lines_json):
        parser.error("--mask et --lines_json doivent être fournis ensemble")
    main(args.video, args.out, args.temp, args.mask, args.lines_json, args.start_time, show=not args.no_display)
