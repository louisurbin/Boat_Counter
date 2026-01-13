import argparse
import os
import sys
import subprocess
from preprocess_mask_lines_date import create_mask_lines_date, get_mask_lines_date_paths
from all_crossings_generator import generate_all_crossings
from visualization_utils import visualize_line_crossings

### Exemple d'utilisation : python3 ./src/main_pipeline.py --in ./data/input_video.mp4 --out output_directory --mode lgc ###

def main(video_path, output_dir, mode):
    """
    Orchestre le pipeline complet.
    """
    # Step 1: Create mask and lines interactively
    print("Step 1: Creating mask and lines...")
    create_mask_lines_date(video_path, output_dir)
    mask_path, lines_path = get_mask_lines_date_paths(video_path, output_dir)
    if not os.path.exists(mask_path):
        print(f"Mask not found: {mask_path}")
        sys.exit(1)

    # Step 2: Apply MOG2 background subtraction
    print("Step 2: Applying MOG2 background subtraction...")
    mog2_output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_mog2.mp4")
    cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "mog2_background_subtraction.py"), "-i", video_path, "-o", mog2_output_path, "-m", mask_path]
    subprocess.run(cmd, check=True)

    # Step 3: Tracking or LineGateCapture
    if mode == "tracker":
        print("Step 3: Tracking...")
        tracked_video_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_tracked.mp4")
        cmd_sort = [sys.executable, os.path.join(os.path.dirname(__file__), "tracker_main.py"),
                "--video", mog2_output_path,
                "--lines_json", lines_path,
                "--save", tracked_video_path,
                "--color_video", os.path.abspath(video_path)]
        subprocess.run(cmd_sort, check=True)
    elif mode == "lgc":
        print("Step 3: Line Gate Capture...")
        cmd_lgc = [sys.executable, os.path.join(os.path.dirname(__file__), "line_gate_capture.py"),
                   "--video", mog2_output_path,
                   "--color_video", os.path.abspath(video_path),
                   "--lines_json", lines_path]
        subprocess.run(cmd_lgc, check=True)
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)

    # Step 4: Generate per-id crossings with datetime and aggregated all_crossings in ./temp
    print("Step 4: Visualizing crossings...")
    id_dir = os.path.abspath(os.path.join(".", "temp", "extractions"))
    visualize_line_crossings(video_path, lines_path, id_dir)

    # Step 5: Call id_to_boat.py to add boat type to _all_crossings.txt 
    print("Step 5: Adding boat type to crossings...")
    try:
        cmd_boat = [sys.executable, os.path.join(os.path.dirname(__file__), "id_to_boat.py")]
        subprocess.run(cmd_boat, check=True)
    except Exception as e:
        print(f"Warning: id_to_boat.py failed: {e}")

    # Generate aggregated all_crossings summary
    generate_all_crossings(video_path, id_dir, output_dir)

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
            # candidate mask files to remove: contain '_mask', '_mog2', '_tracked'
            if (
                "_mask" in low
                #or "_mog2" in low
                #or "_tracked" in low
            ):
                try:
                    os.remove(fpath)
                    print(f"Deleted: {fpath}")
                except Exception as e:
                    print(f"Warning: Could not delete file {fpath}: {e}")
    except Exception as e:
        print(f"Warning during cleanup of masks in {output_dir}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main pipeline: preprocess mask/lines/date, MOG2 background subtraction, tracking, classification.")
    parser.add_argument("--in", help="Path to input video")
    parser.add_argument("--out", default="temp", help="Output directory")
    parser.add_argument("--mode", choices=["tracker", "lgc"], default="tracker", help="Processing mode")
    args = parser.parse_args()
    main(args.video, args.out, args.mode)
