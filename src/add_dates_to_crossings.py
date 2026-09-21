import os
import sys
import json
from datetime import datetime, timedelta

def parse_start_time(s):
    fmts = ['%m/%d %H:%M:%S', '%m/%d/%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']
    for fmt in fmts:
        try:
            dt = datetime.strptime(s, fmt)
            if '%Y' not in fmt:
                dt = dt.replace(year=datetime.now().year)
            return dt
        except Exception:
            continue
    raise ValueError(f"Could not parse start_time: {s}")

def is_float_tok(tok):
    try:
        float(tok)
        return True
    except Exception:
        return False


def looks_like_date(tok):
    tok = tok.strip()
    return ('/' in tok or ':' in tok) and any(ch.isdigit() for ch in tok)


def add_dates(extractions_dir='temp/extractions', lines_date_path=None, output_dir=None):
    """
    Parcourt tous les crossings.txt et ajoute une colonne date calculée
    à partir de start_time dans le fichier lines_date_path et du nombre de secondes.
    Met à jour aussi les sections Details per ID dans *_all_crossings.txt.

    Args:
        extractions_dir: Dossier contenant les sous-dossiers avec crossings.txt
        lines_date_path: Chemin vers le fichier *_lines_date.json (dans temp/)
        output_dir: Dossier contenant les fichiers *_all_crossings.txt (dans output/)
    """
    if not lines_date_path:
        print('Error: lines_date_path is required', file=sys.stderr)
        return 1
    if not os.path.exists(lines_date_path):
        print('Error: lines_date file not found:', lines_date_path, file=sys.stderr)
        return 1
    with open(lines_date_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    start_s = meta.get('start_time')
    if not start_s:
        print('Error: start_time missing in', lines_date_path, file=sys.stderr)
        return 1
    try:
        start_dt = parse_start_time(start_s)
    except Exception as e:
        print('Error parsing start_time:', e, file=sys.stderr)
        return 1

    extra_root = os.path.abspath(extractions_dir)
    if not os.path.isdir(extra_root):
        print('Error: extractions dir not found:', extra_root, file=sys.stderr)
        return 1

    rows_all = []

    for name in sorted(os.listdir(extra_root), key=lambda x: int(x) if x.isdigit() else x):
        sub = os.path.join(extra_root, name)
        if not os.path.isdir(sub):
            continue
        crossings_path = os.path.join(sub, 'crossings.txt')
        if not os.path.exists(crossings_path):
            continue

        out_lines = []
        changed = False

        with open(crossings_path, 'r', encoding='utf-8') as f:
            for line in f:
                sline = line.rstrip('\n')
                if not sline.strip():
                    out_lines.append(sline)
                    continue
                parts = sline.split('\t')

                last = parts[-1].strip()
                seconds = None
                date_str = None

                if is_float_tok(last):
                    seconds = float(last)
                    dt = start_dt + timedelta(seconds=seconds)
                    date_str = dt.strftime('%m/%d %H:%M:%S')
                    out_lines.append(sline + '\t' + date_str)
                    changed = True
                elif looks_like_date(last):
                    date_str = last
                    if len(parts) >= 2 and is_float_tok(parts[-2].strip()):
                        seconds = float(parts[-2].strip())
                    else:
                        for tok in reversed(parts[:-1]):
                            if is_float_tok(tok.strip()):
                                seconds = float(tok.strip())
                                break
                    out_lines.append(sline)
                else:
                    for tok in reversed(parts):
                        if is_float_tok(tok.strip()):
                            seconds = float(tok.strip())
                            break
                    if seconds is None:
                        out_lines.append(sline)
                    else:
                        dt = start_dt + timedelta(seconds=seconds)
                        date_str = dt.strftime('%m/%d %H:%M:%S')
                        out_lines.append(sline + '\t' + date_str)
                        changed = True

                if seconds is not None:
                    label = parts[0] if len(parts) >= 1 else ''
                    sens = parts[1] if len(parts) >= 2 else ''
                    rows_all.append((name, label, sens, str(seconds), date_str))

        if changed:
            with open(crossings_path, 'w', encoding='utf-8') as f:
                for ol in out_lines:
                    f.write(ol + '\n')

    # Construire la map des dates pour la mise à jour des all_crossings
    dates_map = {}
    for oid, label, sens, sec, date in rows_all:
        key = (label.strip().lower(), str(oid).strip())
        dates_map.setdefault(key, [])
        if date not in dates_map[key]:
            dates_map[key].append(date)

    # Extraire le nom de la video depuis lines_date_path
    video_name = os.path.basename(lines_date_path).replace('_lines_date.json', '')
    target_filename = f"{video_name}_all_crossings.txt"

    # Chercher UNIQUEMENT le fichier {video_name}_all_crossings.txt
    candidates = set()
    search_dirs = [output_dir] if output_dir else ['output', os.path.dirname(lines_date_path)]
    for base in search_dirs:
        if not base or not os.path.isdir(base):
            continue
        target_path = os.path.join(base, target_filename)
        if os.path.exists(target_path):
            candidates.add(os.path.abspath(target_path))
            break

    for path in sorted(candidates):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            out_lines = []
            in_details = False
            current_label = None

            for raw in lines:
                ln = raw.rstrip('\n')
                stripped = ln.strip()

                if stripped.startswith('=== Details per ID ==='):
                    in_details = True
                    out_lines.append(ln)
                    continue
                if in_details and stripped.startswith('[') and stripped.endswith(']'):
                    current_label = stripped[1:-1]
                    out_lines.append(ln)
                    continue
                if in_details and '\t' in ln:
                    parts = ln.split('\t')
                    oid = parts[0].strip()
                    key = (current_label.strip().lower(), oid)
                    dates = dates_map.get(key, [])
                    if dates:
                        last_part = parts[-1].strip()
                        if looks_like_date(last_part):
                            # Remplacer la date existante
                            parts[-1] = dates[0]
                            new_ln = '\t'.join(parts)
                        else:
                            # Ajouter la date
                            new_ln = ln + '\t' + dates[0]
                        out_lines.append(new_ln)
                    else:
                        out_lines.append(ln)
                    continue
                out_lines.append(ln)

            if out_lines != [l.rstrip('\n') for l in lines]:
                with open(path, 'w', encoding='utf-8') as f:
                    for ol in out_lines:
                        f.write(ol + '\n')

        except Exception as e:
            print(f'Warning updating {path}: {e}', file=sys.stderr)

    return 0
