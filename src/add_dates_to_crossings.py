import os
import sys
import json
from datetime import datetime, timedelta


def find_lines_json(temp_dir):
    if not os.path.isdir(temp_dir):
        return None
    # prefer *_lines_date.json, fallback to *_lines.json
    for fn in os.listdir(temp_dir):
        if fn.endswith('_lines_date.json'):
            return os.path.join(temp_dir, fn)
    for fn in os.listdir(temp_dir):
        if fn.endswith('_lines.json'):
            return os.path.join(temp_dir, fn)
    return None


def parse_start_time(s):
    # try formats, if no year add current year
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


def main(extractions_dir='temp/extractions', temp_dir='temp'):
    lines_json = find_lines_json(temp_dir)
    if not lines_json:
        print('Error: no *_lines.json found in', temp_dir, file=sys.stderr)
        return 1
    with open(lines_json, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    start_s = meta.get('start_time')
    if not start_s:
        print('Error: start_time missing in', lines_json, file=sys.stderr)
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

    updated = 0
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
                # determine if last token is seconds or already a date; find seconds token if needed
                def is_float_tok(tok):
                    try:
                        float(tok)
                        return True
                    except Exception:
                        return False
                def looks_like_date(tok):
                    tok = tok.strip()
                    return ('/' in tok or ':' in tok) and any(ch.isdigit() for ch in tok)

                last = parts[-1].strip()
                seconds = None
                date_str = None
                if is_float_tok(last):
                    seconds = float(last)
                    # compute date and will append
                    dt = start_dt + timedelta(seconds=seconds)
                    date_str = dt.strftime('%m/%d %H:%M:%S')
                    new_line = sline + '\t' + date_str
                    out_lines.append(new_line)
                    changed = True
                elif looks_like_date(last):
                    # already has date in last column; try to find seconds in previous token
                    date_str = last
                    if len(parts) >= 2 and is_float_tok(parts[-2].strip()):
                        seconds = float(parts[-2].strip())
                    else:
                        # scan from right for a numeric token
                        for tok in reversed(parts[:-1]):
                            t = tok.strip()
                            if is_float_tok(t):
                                seconds = float(t)
                                break
                    # keep original line (do not append date twice)
                    out_lines.append(sline)
                else:
                    # last token is neither float nor date: search for numeric token from right
                    for tok in reversed(parts):
                        t = tok.strip()
                        if is_float_tok(t):
                            seconds = float(t)
                            break
                    if seconds is None:
                        # cannot find seconds, leave unchanged
                        out_lines.append(sline)
                    else:
                        dt = start_dt + timedelta(seconds=seconds)
                        date_str = dt.strftime('%m/%d %H:%M:%S')
                        new_line = sline + '\t' + date_str
                        out_lines.append(new_line)
                        changed = True

                if seconds is None:
                    # cannot aggregate without seconds; skip recording
                    continue
                # record for aggregation: (id, label, sens, seconds, date)
                label = parts[0] if len(parts) >= 1 else ''
                sens = parts[1] if len(parts) >= 2 else ''
                rows_all.append((name, label, sens, str(seconds), date_str))
        if changed:
            with open(crossings_path, 'w', encoding='utf-8') as f:
                for ol in out_lines:
                    f.write(ol + '\n')
            # updated crossings file (silent)
            updated += 1
    # write aggregated all_crossings using collected rows_all
    # Build mapping (label, id) -> list of dates
    dates_map = {}
    for oid, label, sens, sec, date in rows_all:
        lab = (label or '').strip().lower()
        oid_s = str(oid).strip()
        key = (lab, oid_s)
        dates_map.setdefault(key, []).append(date)

    # Find existing *_all_crossings* files in temp_dir and update their Details per id sections
    touched = 0
    # collect candidate all_crossings files from temp_dir and project root recursively
    candidates = set()
    for base in [temp_dir, '.']:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for fn in files:
                if '_all_crossings' in fn:
                    candidates.add(os.path.join(root, fn))

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
                 # detect start of details section
                 if stripped.startswith('# Details'):
                     in_details = True
                     out_lines.append(ln)
                     continue
                 # section header [label]
                 if in_details and stripped.startswith('[') and stripped.endswith(']'):
                     current_label = stripped[1:-1]
                     out_lines.append(ln)
                     continue
                 # in details, lines with tabs are id\tsigns or similar -> append dates for (current_label, id)
                 if in_details and '\t' in ln:
                     parts = ln.split('\t')
                     oid = parts[0].strip()
                     lab_norm = (current_label or '').strip().lower()
                     key = (lab_norm, oid)
                     dates = dates_map.get(key, [])
                     if dates:
                         new_ln = ln + '\t' + ', '.join(dates)
                     else:
                         new_ln = ln
                     out_lines.append(new_ln)
                     continue
                 out_lines.append(ln)

             # overwrite file only if changed
             if out_lines != [l.rstrip('\n') for l in lines]:
                 with open(path, 'w', encoding='utf-8') as f:
                     for ol in out_lines:
                         f.write(ol + '\n')
                 # updated existing all_crossings file (silent)
                 touched += 1
         except Exception as e:
             print(f'Warning updating {path}: {e}', file=sys.stderr)

    # done (silent)
    return 0

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Append datetime to each crossings.txt based on start_time in lines.json')
    p.add_argument('--extractions', default=os.path.join('.', 'temp', 'extractions'))
    p.add_argument('--temp', default=os.path.join('.', 'temp'))
    args = p.parse_args()
    sys.exit(main(extractions_dir=args.extractions, temp_dir=args.temp))
