
import cv2
import numpy as np
from ultralytics import YOLO

try:
    from botsort import BoTSORT
except ImportError:
    raise ImportError("Le module botsort n'est pas installé. Installez-le avec 'pip install botsort' ou suivez la doc officielle.")

# Paramètres principaux
VIDEO_PATH = "./temp/masked_video.mp4"
MODEL_PATH = "yolo11n.pt"
CONFIDENCE = 0.12
IMG_SIZE = 1024

# Chargement du modèle YOLO
yolo_model = YOLO(MODEL_PATH)

# Accès direct au YAML/classes
class_names = yolo_model.model.names  # dict {class_id: name}
print("Classes du modèle:", class_names)

# Initialisation du tracker BotSORT
tracker = BoTSORT()

# Prédiction et tracking simplifiés
results = yolo_model.predict(
    source=VIDEO_PATH,
    conf=CONFIDENCE,
    stream=True,
    imgsz=IMG_SIZE
)

for r in results:
    frame = r.orig_img.copy()
    # Pour chaque détection, on affiche la classe
    for i, box in enumerate(r.boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.tolist())
        cls_id = int(r.boxes.cls[i]) if hasattr(r.boxes, 'cls') else -1
        conf = float(r.boxes.conf[i]) if hasattr(r.boxes, 'conf') else 1.0
        name = class_names.get(cls_id, str(cls_id))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Préparation pour BotSORT (x1,y1,x2,y2,score)
    rects = [list(map(int, box.tolist())) for box in r.boxes.xyxy]
    scores = [float(r.boxes.conf[i]) if hasattr(r.boxes, 'conf') else 1.0 for i in range(len(rects))]
    dets = np.array([rects[i] + [scores[i]] for i in range(len(rects))], dtype=float) if len(rects) > 0 else np.empty((0,5))
    tracks = tracker.update(dets, frame)

    # Affichage des tracks (ID et bbox)
    if isinstance(tracks, (list, np.ndarray)):
        for tr in tracks:
            if isinstance(tr, dict):
                tid = tr.get('track_id', tr.get('id', None))
                bbox = tr.get('bbox', None)
            else:
                tid = int(tr[4]) if len(tr) > 4 else None
                bbox = tr[:4]
            if tid is not None and bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, f"ID {tid}", (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 1)

    cv2.imshow("BotSORT Tracker", frame)
    key = cv2.waitKey(1)
    if cv2.getWindowProperty("BotSORT Tracker", cv2.WND_PROP_VISIBLE) < 1 or (key & 0xFF == ord('q')):
        break
cv2.destroyAllWindows()