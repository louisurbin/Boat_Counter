import os
from ResNet import get_boat_type_name
from add_dates_to_crossings import add_dates


def generate_all_crossings(video_path, crossings_dir, output_dir, num_classes=6):
    summary = {} # structure: { label: { "up": int, "down": int, "per_id": { oid: { "signs": [int], "boat_class": str, "mean_prob": float } } } }

    if not os.path.exists(crossings_dir):
        print(f"Crossings dir not found: {crossings_dir}")
        return

    for oid in os.listdir(crossings_dir):
        txt_path = os.path.join(crossings_dir, oid, "crossings.txt")
        if not os.path.isfile(txt_path):
            continue

        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
                if len(lines) < 2:
                    continue

            # ---- last line = boat_class + mean_prob ----
            last_parts = lines[-1].split()
            if len(last_parts) < 2:
                continue

            boat_class = last_parts[0]
            if boat_class.lower() == "noise":
                continue  # skip noise 
            try:
                mean_prob = float(last_parts[1])
            except ValueError:
                mean_prob = 0.0

            # ---- crossings lines ----
            for ln in lines[:-1]:
                parts = ln.split()
                if len(parts) < 2:
                    continue

                label, sign_str = parts[0], parts[1]
                try:
                    sign = int(sign_str)
                except Exception:
                    sign = 1 if sign_str.startswith("+") else (-1 if sign_str.startswith("-") else 0)

                if sign == 0:
                    continue

                rec = summary.setdefault(
                    label,
                    {"up": 0, "down": 0, "per_id": {}}
                )

                if sign > 0:
                    rec["up"] += 1
                else:
                    rec["down"] += 1

                per_id = rec["per_id"].setdefault(
                    oid,
                    {
                        "signs": [],
                        "boat_class": boat_class,
                        "mean_prob": mean_prob,
                    }
                )
                per_id["signs"].append(sign)

        except Exception as e:
            print(f"Warning reading {txt_path}: {e}")

    # ========= WRITE SUMMARY =========
    os.makedirs(output_dir, exist_ok=True)
    video_base = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f"{video_base}_all_crossings.txt")

    with open(out_path, "w", encoding="utf-8") as fo:

        # ----- Global -----
        fo.write("=== Global ===\n")
        fo.write("line\tup\tdown\ttotal\n")
        for label, rec in sorted(summary.items()):
            up, down = rec["up"], rec["down"]
            fo.write(f"{label}\t{up}\t{down}\t{up + down}\n")

        # ----- Per class -----
        for class_id in range(num_classes):
            class_name = get_boat_type_name(class_id)
            fo.write(f"\n=== {class_name} ===\n")
            fo.write("line\tup\tdown\ttotal\n")

            class_summary = {}

            for label, rec in summary.items():
                for info in rec["per_id"].values():
                    if info["boat_class"].lower() == "noise":
                        continue  # skip noise
                    if info["boat_class"] != class_name:
                        continue
                    for sign in info["signs"]:
                        cs = class_summary.setdefault(label, {"up": 0, "down": 0})
                        if sign > 0:
                            cs["up"] += 1
                        else:
                            cs["down"] += 1

            for label, cs in sorted(class_summary.items()):
                fo.write(f"{label}\t{cs['up']}\t{cs['down']}\t{cs['up'] + cs['down']}\n")

        # ----- Details per ID -----
        fo.write("\n=== Details per ID ===\n")
        for label, rec in sorted(summary.items()):
            fo.write(f"\n[{label}]\n")
            for oid, info in sorted(rec["per_id"].items(), key=lambda x: int(x[0])):
                if info["boat_class"].lower() == "noise":
                    continue  # skip noise 
                signs_str = ", ".join(str(s) for s in info["signs"])
                fo.write(
                    f"{oid}\t{signs_str}\t{info['boat_class']}\t{info['mean_prob']:.4f}\n"
                )

    # ===== ADD DATES TO CROSSINGS =====
    add_dates()