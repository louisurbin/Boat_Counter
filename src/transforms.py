"""
Transformations d'images partagées entre les classifieurs (direction et type de bateau).
Identiques à celles utilisées pendant l'entraînement DINOv2 + LogisticRegression.
"""

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


class CLAHETransform:
    """Applique CLAHE sur le canal L de l'espace LAB."""
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def __call__(self, pil_img):
        img = np.array(pil_img)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        L, a, b = cv2.split(lab)
        L = self.clahe.apply(L)
        lab = cv2.merge((L, a, b))
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)


class GammaTransform:
    """Correction gamma."""
    def __init__(self, gamma=0.9):
        self.gamma = float(gamma)

    def __call__(self, pil_img):
        arr = np.array(pil_img).astype(np.float32) / 255.0
        arr = np.clip(arr ** self.gamma, 0.0, 1.0)
        return Image.fromarray((arr * 255).astype('uint8'))


transform = transforms.Compose([
    CLAHETransform(clipLimit=2.0, tileGridSize=(8, 8)),
    GammaTransform(gamma=0.9),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
