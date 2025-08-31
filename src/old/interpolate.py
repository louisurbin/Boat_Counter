import subprocess
from fractions import Fraction

input_path = "./data/TLC00010_extrait.m4v"
output_path = "./data/TLC00010_extrait_2xFPS.m4v"

# Récupérer le framerate d'origine
probe_cmd = [
    "ffprobe", "-v", "error", "-select_streams", "v:0",
    "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1", input_path
]
result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)

# Convertir la fraction renvoyée par ffprobe en float
orig_fps = float(Fraction(result.stdout.strip()))
target_fps = orig_fps * 2

# Doubler les FPS avec minterpolate
ffmpeg_cmd = [
    "ffmpeg", "-i", input_path,
    "-vf", f"minterpolate=mi_mode=mci:mc_mode=aobmc:vsbmc=1:fps={target_fps}",
    "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
    "-y", output_path
]
subprocess.run(ffmpeg_cmd, check=True)
print(f"Vidéo exportée avec {target_fps} FPS : {output_path}")
