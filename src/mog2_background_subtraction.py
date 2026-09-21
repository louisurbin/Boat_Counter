import argparse
from pathlib import Path
import sys
import cv2
import math
import numpy as np

# Paramètres
VAR_THRESHOLD = 20                          # Paramètre de variance
HISTORY = 100                               # Nombre de frames retenues par MOG2 pour estimer le fond
MIN_AREA = 10**2                             # Aire minimale pour considérer un cluster comme valide
MORPH_KERNEL_OPEN = 7                       # Taille max du bruit blanc à supprimer
MORPH_KERNEL_CLOSE = 50                     # Taille max des trous noirs à combler

def parse_args():
    p = argparse.ArgumentParser(description="Applique la soustraction de fond MOG2 à une vidéo.")
    p.add_argument("--input", "-i", type=str, required=True, help="Chemin de la vidéo d'entrée.")
    p.add_argument("--output", "-o", type=str, required=True, help="Chemin de la vidéo de sortie (masque).")
    p.add_argument("--mask", "-m", type=str, help="Chemin du masque à appliquer avant MOG2.")
    return p.parse_args()

def analyze_clusters(mask, result):
    """
    Remplit les bounding boxes des composantes connexes au-dessus de MIN_AREA directement dans 'result'.
    Réutilise le même buffer pour éviter les allocations.
    """
    result.fill(0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) >= MIN_AREA:
            x, y, w, h = cv2.boundingRect(cnt)
            result[y:y+h, x:x+w] = 255

def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"Input video not found: {in_path}", file=sys.stderr)
        sys.exit(1)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"Failed to open video: {in_path}", file=sys.stderr)
        sys.exit(1)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 1:
        print(f"Cannot read FPS from video: {in_path}", file=sys.stderr)
        cap.release()
        sys.exit(1)

    # Charger le masque si fourni
    mask = None
    mask_zero = None
    if args.mask:
        mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Impossible de charger le masque: {args.mask}", file=sys.stderr)
            sys.exit(1)
        if mask.shape != (height, width):
            print(f"Warning: Les dimensions du masque ne correspondent pas à la vidéo: {mask.shape} vs {(height, width)}. Redimensionnement...", file=sys.stderr)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_zero = (mask == 0)  # Précalculé une fois

        # Calculer la ROI à partir du masque
        ys, xs = np.where(mask != 0)
        y0, y1 = ys.min(), ys.max() + 1
        x0, x1 = xs.min(), xs.max() + 1
        roi_slice = (slice(y0, y1), slice(x0, x1))
    else:
        roi_slice = (slice(0, height), slice(0, width))

    # Soustracteur de fond MOG2
    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=HISTORY,
        varThreshold=VAR_THRESHOLD,
        detectShadows=True  # Les ombres sont traitées comme des objets
    )

    ext = out_path.suffix.lower()
    fourcc = cv2.VideoWriter_fourcc(*"XVID") if ext == ".avi" else cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height), False)
    if not writer.isOpened():
        alt_fourcc = cv2.VideoWriter_fourcc(*"XVID") if fourcc != cv2.VideoWriter_fourcc(*"XVID") else cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), alt_fourcc, fps, (width, height), False)
        if not writer.isOpened():
            print(f"Failed to open VideoWriter for: {out_path}", file=sys.stderr)
            cap.release()
            sys.exit(1)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_OPEN, MORPH_KERNEL_OPEN))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_CLOSE, MORPH_KERNEL_CLOSE))
    
    frame_count = 0

    # Buffer réutilisé pour analyze_clusters (évite les allocations)
    rect_mask = np.zeros((height, width), dtype=np.uint8)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Appliquer le masque statique (in-place)
        if mask_zero is not None:
            gray_frame[mask_zero] = 0

        # Appliquer MOG2 seulement sur la ROI
        fgmask = np.zeros((height, width), dtype=np.uint8)
        roi = gray_frame[roi_slice]
        fg_roi = subtractor.apply(roi)
        fgmask[roi_slice] = fg_roi

        # Nettoyage morphologique
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel_close, iterations=1)

        # Pas de seuillage nécessaire, les ombres sont incluses comme objets
        analyze_clusters(fgmask, rect_mask)

        # Forcer la première frame en noir (sinon blanche par défaut dans MOG2)
        if frame_count == 1:
            rect_mask.fill(0)

        writer.write(rect_mask)

    cap.release()
    writer.release()
    print(f"Vidéo masque MOG2 sauvegardée: {out_path}")

if __name__ == "__main__":
    main()
