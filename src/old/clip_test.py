import os
import torch
import shutil
import clip
from PIL import Image
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "./datasets"
ERROR_DIR = "./datasets/errors"
CLASSES = ["avant", "arriere"]

PROMPTS = {
    "avant": "a boat approaching the camera on the water",
    "arriere": "a boat seen from behind, waves visible in the water",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Créer le dossier d'erreurs
os.makedirs(ERROR_DIR, exist_ok=True)
for cls in CLASSES:
    os.makedirs(os.path.join(ERROR_DIR, cls), exist_ok=True)

# -----------------------------n
# LOAD MODEL
# -----------------------------
model, preprocess = clip.load("ViT-B/32", device=DEVICE)
model.eval()

# Encode text ONCE
text_tokens = clip.tokenize([PROMPTS[c] for c in CLASSES]).to(DEVICE)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# -----------------------------
# EVALUATION
# -----------------------------
correct = 0
total = 0

results = []

for label_idx, label in enumerate(CLASSES):
    folder = os.path.join(DATA_DIR, label)

    for img_name in tqdm(os.listdir(folder), desc=f"Evaluating {label}"):
        img_path = os.path.join(folder, img_name)

        image = preprocess(Image.open(img_path).convert("RGB")) \
                    .unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            logits = (image_features @ text_features.T).softmax(dim=-1)
            pred = logits.argmax(dim=-1).item()

        results.append((label, CLASSES[pred], logits.cpu().numpy()))

        if pred == label_idx:
            correct += 1
        else:
            shutil.copy(img_path, os.path.join(ERROR_DIR, label, img_name))
        total += 1

# -----------------------------
# RESULTS
# -----------------------------
accuracy = correct / total
print(f"\nCLIP Zero-Shot Accuracy: {accuracy:.3f}")

### Conclusion : Not good enough for our use case, cannot recognize boats from behind.