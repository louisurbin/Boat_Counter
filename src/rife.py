import os
import subprocess

# ---- PARAMÈTRES ----
input_video = "./temp/low_fps_video.mp4"   # ta vidéo originale
output_video = "./temp/high_fps_video.mp4" # vidéo interpolée
exp = 4  # facteur d'augmentation (2 = double fps, 4 = quadruple fps)

# ---- LANCEMENT RIFE ----
cmd = [
    "python3", "inference_video.py",
    "--exp=" + str(exp),
    "--video=" + input_video,
    "--output=" + output_video
]

print("Execution:", " ".join(cmd))
subprocess.run(cmd, check=True)

print(f"Vidéo interpolée sauvegardée dans {output_video}")
