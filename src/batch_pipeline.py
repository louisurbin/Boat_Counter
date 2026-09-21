"""
Traitement par lot : lance le pipeline sur toutes les vidéos du dossier videos/.

Structure attendue (un sous-dossier = une scène = même masque + mêmes lignes) :
    videos/
        archeveche/   *.mp4
        birhakeim/    *.mp4
        philippe/     *.mp4
        st_louis/     *.mp4

Déroulement :
  Phase 1 - Configuration des scènes (seule partie interactive, à faire au début)
      Si scenes/<dossier>/mask.png et scenes/<dossier>/lines_date.json n'existent pas,
      l'éditeur s'ouvre UNE fois sur la 1re vidéo du dossier (ordre alphabétique) :
      masque, lignes, et date de début de CETTE vidéo (touche 'd'), puis 's' et ESC.
  Phase 2 - Traitement (aucune fenêtre, aucune interaction)
      Chaque vidéo du dossier est traitée avec le même masque et les mêmes lignes ;
      la date de début est décalée de +1 jour par vidéo (1re vidéo = date saisie).
      Résultats dans output/<dossier>/.

Une vidéo déjà traitée (extractions_<vidéo>/ et <vidéo>_all_crossings.txt présents
dans output/<dossier>/) est ignorée, sauf avec --force : on peut donc relancer la
commande après une interruption.

Exemples :
    python3 ./src/batch_pipeline.py
    python3 ./src/batch_pipeline.py --only st_louis
    python3 ./src/batch_pipeline.py --videos ./videos --out ./output --force
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import timedelta

from add_dates_to_crossings import parse_start_time
from main_pipeline import main as run_pipeline
from preprocess_mask_lines_date import create_mask_lines_date, get_mask_lines_date_paths

VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv")
DATE_FMT = "%m/%d %H:%M:%S"


# Utilitaires

def natural_key(text):
    """Tri naturel : video_2 avant video_10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


def list_videos(folder):
    names = [f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXT)]
    return [os.path.join(folder, f) for f in sorted(names, key=natural_key)]


def video_base(video_path):
    return os.path.splitext(os.path.basename(video_path))[0]


def is_done(video_path, out_dir):
    base = video_base(video_path)
    return (os.path.isdir(os.path.join(out_dir, f"extractions_{base}"))
            and os.path.isfile(os.path.join(out_dir, f"{base}_all_crossings.txt")))


def shift_start_time(start_time, days):
    """'06/22 05:00:04' + 2 jours -> '06/24 05:00:04' (gère les changements de mois)."""
    return (parse_start_time(start_time) + timedelta(days=days)).strftime(DATE_FMT)


def check_scene(lines_path):
    """Vérifie le fichier lignes/date ; retourne (start_time, [labels]) ou lève ValueError."""
    with open(lines_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    lines = meta.get("lines") or []
    if not lines:
        raise ValueError("aucune ligne de comptage dans le fichier lignes/date")
    labels = [l.get("label") or "" for l in lines]
    for label in labels:
        # Le nom de ligne est relu par split() dans all_crossings : pas de vide ni d'espace
        if not label or any(c.isspace() for c in label):
            raise ValueError(f"nom de ligne invalide ({label!r}) : ni vide, ni espace")
    start = meta.get("start_time")
    if not start:
        raise ValueError("date de début absente (touche 'd' dans l'éditeur)")
    parse_start_time(start)  # lève ValueError si le format est invalide
    return start, labels


# Phase 1 : configuration d'une scène

def ensure_scene_config(scene_dir, first_video):
    """
    Retourne (mask_path, lines_path, start_time, labels).
    Ouvre l'éditeur sur first_video si la configuration de la scène n'existe pas encore.
    """
    mask_path = os.path.join(scene_dir, "mask.png")
    lines_path = os.path.join(scene_dir, "lines_date.json")

    if not (os.path.isfile(mask_path) and os.path.isfile(lines_path)):
        os.makedirs(scene_dir, exist_ok=True)
        print(f"  Éditeur : masque (clics, clic droit pour fermer), ligne(s) (touche l), "
              f"date de début DE CETTE VIDÉO (touche d), puis s (sauvegarder) et ESC (fermer).")
        create_mask_lines_date(first_video, scene_dir)

        saved_mask, saved_lines = get_mask_lines_date_paths(first_video, scene_dir)
        if not (os.path.isfile(saved_mask) and os.path.isfile(saved_lines)):
            raise RuntimeError("rien n'a été sauvegardé dans l'éditeur (touche 's' oubliée ?)")
        try:
            check_scene(saved_lines)
        except Exception:
            os.remove(saved_mask)   # pour que l'éditeur se rouvre au prochain lancement
            os.remove(saved_lines)
            raise
        os.replace(saved_mask, mask_path)
        os.replace(saved_lines, lines_path)

    start_time, labels = check_scene(lines_path)
    return mask_path, lines_path, start_time, labels


# Programme principal

def main():
    p = argparse.ArgumentParser(description="Lance le pipeline sur toutes les vidéos de videos/<scène>/.")
    p.add_argument("--videos", default="videos", help="Dossier racine contenant un sous-dossier par scène")
    p.add_argument("--out", default="output", help="Dossier racine des résultats (un sous-dossier par scène)")
    p.add_argument("--temp", default="temp", help="Dossier des fichiers intermédiaires")
    p.add_argument("--scenes", default="scenes", help="Dossier où sont mémorisés masque + lignes de chaque scène")
    p.add_argument("--only", nargs="+", metavar="SCENE", help="Ne traiter que ces sous-dossiers")
    p.add_argument("--force", action="store_true", help="Retraiter aussi les vidéos déjà traitées")
    args = p.parse_args()

    if not os.path.isdir(args.videos):
        sys.exit(f"Dossier introuvable: {args.videos}")

    scene_names = sorted(d for d in os.listdir(args.videos)
                         if os.path.isdir(os.path.join(args.videos, d)))
    if args.only:
        unknown = [s for s in args.only if s not in scene_names]
        if unknown:
            sys.exit(f"Sous-dossier(s) inconnu(s): {unknown}. Disponibles: {scene_names}")
        scene_names = [s for s in scene_names if s in args.only]

    # Phase 1 : configuration de toutes les scènes (interactif, une seule fois)
    jobs = []   # (scène, out_dir, mask, lines, start_time, labels, [(index, vidéo)])
    print("=== Phase 1 : configuration des scènes ===")
    for scene in scene_names:
        videos = list_videos(os.path.join(args.videos, scene))
        out_dir = os.path.join(args.out, scene)
        if not videos:
            print(f"[{scene}] aucune vidéo, ignoré.")
            continue
        todo = [(i, v) for i, v in enumerate(videos) if args.force or not is_done(v, out_dir)]
        if not todo:
            print(f"[{scene}] {len(videos)} vidéo(s) déjà traitée(s), ignoré (--force pour refaire).")
            continue
        try:
            mask, lines, start_time, labels = ensure_scene_config(os.path.join(args.scenes, scene), videos[0])
        except Exception as e:
            print(f"[{scene}] configuration impossible : {e} -> scène ignorée.")
            continue
        print(f"[{scene}] OK : {len(todo)}/{len(videos)} vidéo(s) à traiter, "
              f"lignes {labels}, début de la 1re vidéo : {start_time}")
        jobs.append((scene, out_dir, mask, lines, start_time, labels, todo))

    # Phase 2 : traitement sans interaction
    total = sum(len(j[-1]) for j in jobs)
    print(f"\n=== Phase 2 : traitement de {total} vidéo(s), sans interaction ===")
    recap = []
    n = 0
    for scene, out_dir, mask, lines, first_start, labels, todo in jobs:
        for idx, video in todo:
            n += 1
            start_time = shift_start_time(first_start, idx)
            base = video_base(video)
            print(f"\n[{n}/{total}] {scene} / {base}  (début : {start_time})")
            t0 = time.time()
            try:
                counts = run_pipeline(video, out_dir, args.temp,
                                      mask_src=mask, lines_src=lines,
                                      start_time=start_time, show=False)
                status = "OK"
                detail = "  ".join(f"{l}: montants={counts.get(l, {}).get('up', 0)} "
                                   f"descendants={counts.get(l, {}).get('down', 0)}" for l in labels)
            except (Exception, SystemExit):   # SystemExit : main_pipeline utilise sys.exit(1)
                traceback.print_exc()
                status, detail = "ERREUR", "voir le message ci-dessus"
            recap.append((scene, base, start_time, status, detail, (time.time() - t0) / 60))

    # Récapitulatif
    print("\n=== Récapitulatif ===")
    for scene, base, start_time, status, detail, minutes in recap:
        print(f"{status:<6} {scene}/{base}  début={start_time}  {detail}  ({minutes:.1f} min)")
    errors = sum(1 for r in recap if r[3] != "OK")
    print(f"\n{len(recap) - errors} OK, {errors} en erreur.")
    if errors:
        print("Relancer la même commande retraite uniquement les vidéos en erreur.")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
