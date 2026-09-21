"""
export_weekly_tables.py
Convertit les fichiers output de comptage fluvial en tableaux hebdomadaires (format txt).

Usage :
    python3 ./src/export_weekly_tables.py [annee]
    python3 ./src/export_weekly_tables.py --in ./output --out ./export 2024
    python3 ./src/export_weekly_tables.py --none-mode proportionnel 2024

- Scanne --in/<dossier_scene>/*_all_crossings.txt  (défaut : ./output)
- Écrit   --out/<dossier_scene>/comptages_<scene>_semaine_<NN>_<annee>[_<none-mode>].txt  (défaut : ./export)
  (le suffixe _<none-mode> n'est ajouté que si --none-mode != ignore)
- Nom de fichier attendu (extrait) : ..._<mois>_<jour>_<jourSemaine>...
  ex: St-Louis_aval_juin_22_lundi_005_all_crossings.txt
- Sortie : un fichier par semaine ISO, 14 tableaux (avalant puis montant associé,
  pour TOUS NAVIRES puis chaque classe)
- Colonnes : plages horaires 5h-6h ... 21h-22h ; lignes : jours de la semaine.

Traitement des bateaux dont la direction est "none" (--none-mode) :
    ignore         (défaut) les none ne sont pas comptés.
    majoritaire    les none prennent la direction majoritaire du fichier all_crossings
                   (parmi ses bateaux avant/arriere). Égalité ou aucun bateau fiable :
                   les none de ce fichier sont ignorés (avertissement).
    proportionnel  les none sont répartis entre avant/arriere selon la proportion des
                   bateaux bien détectés du fichier : round(N_none * p_avant) none
                   deviennent avant, les autres arriere. Le choix des bateaux concernés
                   est aléatoire mais reproductible (graine interne fixe). Aucun bateau
                   fiable : les none de ce fichier sont ignorés (avertissement).
    scene_none     les none prennent la direction imposée par la scène (cf. ci-dessous).
    scene_tous     TOUS les bateaux prennent la direction imposée par la scène
                   (les directions du fichier all_crossings ne sont plus utilisées).

Directions imposées par scène (scene_none / scene_tous), d'après le nom du dossier de scène :
    philippe, archeveche, birhakeim, st_michel : toujours "avant".
    st-louis : "avant" de hh:00 à hh:35, "arriere" de hh:35 à hh:60 (minute < 35 -> avant).
Les statistiques (majoritaire / proportionnel) sont calculées séparément pour chaque
fichier all_crossings, sur les bateaux situés dans la plage horaire étudiée.
"""

import argparse
import random
import re
import sys
from pathlib import Path
from datetime import date

CLASSES = ["administration", "marchandise", "passager_gros", "passager_petit", "plaisance", "restaurant"]
DIRS = [("avant", "AVALANT (avant)"), ("arriere", "MONTANT (arriere)")]
DAYS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MONTHS_FR = {
    "janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12,
}
FIRST_HOUR = 5   # inclus
LAST_HOUR = 22   # exclus -> plages 5-6h ... 21-22h
HOURS = list(range(FIRST_HOUR, LAST_HOUR))

# --- Gestion des directions "none" -------------------------------------------
NONE_MODES = ["ignore", "majoritaire", "proportionnel", "scene_none", "scene_tous"]
NONE_MODE_LABELS = {
    "ignore": "none ignorés",
    "majoritaire": "none -> direction majoritaire du fichier all_crossings",
    "proportionnel": "none répartis selon la proportion avant/arriere du fichier all_crossings (tirage seedé)",
    "scene_none": "none -> direction imposée par la scène",
    "scene_tous": "tous les bateaux -> direction imposée par la scène (directions détectées ignorées)",
}
SCENE_MODES = ("scene_none", "scene_tous")

# Le nom du dossier de scène est normalisé (minuscules, sans accents ni ponctuation)
# puis cherché par inclusion : "St-Louis_aval" -> "stlouisaval" contient "stlouis".
SCENE_ALIASES = {
    "philippe": ["philippe"],
    "archeveche": ["archeveche"],
    "birhakeim": ["birhakeim"],
    "stlouis": ["stlouis", "saintlouis"],
    "stmichel": ["stmichel", "saintmichel"],
}
STLOUIS_SWITCH_MINUTE = 35   # St-Louis : minute < 35 -> avant ; minute >= 35 -> arriere

DETAIL_RE = re.compile(
    r"^line_id_\d+\t(avant|arriere|none)\t(\S+)\t[\d.]+\t[\d.]+\t"
    r"frame_idx=\d+, ts=\d+\t(\d{2})/(\d{2}) (\d{2}):(\d{2}):\d{2}"
)

UNACCENT = str.maketrans("éèêëàâäîïôöûüç", "eeeeaaaiioouuc")


def strip_accents(s):
    return s.translate(UNACCENT)


def iso_week(d):
    """Numéro de semaine ISO (lundi = premier jour)."""
    return d.isocalendar()[1]


def date_from_filename(filename, year):
    m = re.search(r"([a-zéèêëàâäîïôöûüç]+)_(\d{1,2})_([a-zéèêëàâäîïôöûüç]+)", filename.lower())
    if not m:
        raise ValueError(f"Nom de fichier non reconnu (mois/jour introuvables) : {filename}")
    month = MONTHS_FR.get(strip_accents(m.group(1)))
    if not month:
        raise ValueError(f"Mois inconnu '{m.group(1)}' dans : {filename}")
    d = date(year, month, int(m.group(2)))
    day_name = DAYS_FR[d.weekday()]
    if day_name != strip_accents(m.group(3)):
        raise ValueError(f"Incohérence jour/date dans : {filename} ({day_name} attendu, '{m.group(3)}' lu)")
    return d


def parse_file(path, year):
    """Retourne (date, boats, ignored) ; boats = liste de (direction, classe, heure, minute) dans la plage étudiée."""
    d = date_from_filename(path.name, year)
    boats = []
    ignored = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        m = DETAIL_RE.match(line)
        if not m:
            continue
        direction, classe = m.group(1), m.group(2)
        hour, minute = int(m.group(5)), int(m.group(6))
        if hour < FIRST_HOUR or hour >= LAST_HOUR:
            ignored += 1
            continue
        boats.append((direction, classe, hour, minute))
    return d, boats, ignored


def scene_key(scene_name):
    """Clé de scène ('philippe', 'archeveche', 'birhakeim', 'stlouis', 'stmichel') déduite du nom de dossier, ou None."""
    norm = re.sub(r"[^a-z0-9]", "", strip_accents(scene_name.lower()))
    for key, aliases in SCENE_ALIASES.items():
        if any(a in norm for a in aliases):
            return key
    return None


def scene_direction(scene, minute):
    """Direction imposée par la scène pour un bateau passant à la minute donnée de l'heure."""
    if scene == "stlouis":
        return "avant" if minute < STLOUIS_SWITCH_MINUTE else "arriere"
    return "avant"   # philippe, archeveche, birhakeim, stmichel : toujours avant


def resolve_directions(boats, mode, scene, rng):
    """Retourne (dirs, warning) ; dirs[i] ∈ {avant, arriere, None} selon le mode de traitement des none."""
    raw = [b[0] for b in boats]

    if mode == "scene_tous":
        return [scene_direction(scene, b[3]) for b in boats], ""

    dirs = [r if r != "none" else None for r in raw]
    none_idx = [i for i, r in enumerate(raw) if r == "none"]
    if mode == "ignore" or not none_idx:
        return dirs, ""

    n_av, n_ar = raw.count("avant"), raw.count("arriere")
    warning = ""

    if mode == "scene_none":
        for i in none_idx:
            dirs[i] = scene_direction(scene, boats[i][3])

    elif mode == "majoritaire":
        if n_av + n_ar == 0:
            warning = "aucun bateau avant/arriere fiable dans ce fichier"
        elif n_av == n_ar:
            warning = f"égalité avant/arriere ({n_av}/{n_ar}) dans ce fichier"
        else:
            major = "avant" if n_av > n_ar else "arriere"
            for i in none_idx:
                dirs[i] = major

    elif mode == "proportionnel":
        if n_av + n_ar == 0:
            warning = "aucun bateau avant/arriere fiable dans ce fichier"
        else:
            k = round(len(none_idx) * n_av / (n_av + n_ar))   # nombre de none -> avant
            # k plus petites clés parmi des clés uniformes i.i.d. = sous-ensemble uniforme de taille k.
            # On n'utilise que rng.random(), dont la séquence est garantie stable entre versions de Python
            # (contrairement à rng.sample()) -> résultats reproductibles quelle que soit la version.
            order = sorted(none_idx, key=lambda i: rng.random())
            to_avant = set(order[:k])
            for i in none_idx:
                dirs[i] = "avant" if i in to_avant else "arriere"

    return dirs, (f"none ignorés : {warning}" if warning else "")


def aggregate(boats, dirs):
    """Agrège en data = {direction: {classe: {heure: count}}} ; les dirs None sont exclus."""
    data = {"avant": {}, "arriere": {}}
    for (_, classe, hour, _), direction in zip(boats, dirs):
        if direction is None:
            continue
        cell = data[direction].setdefault(classe, {})
        cell[hour] = cell.get(hour, 0) + 1
    return data


def summarize(boats, dirs, mode):
    """Résumé pour le log : devenir des none (et directions contredites en scene_tous)."""
    none_dirs = [d for b, d in zip(boats, dirs) if b[0] == "none"]
    s = (f"none={len(none_dirs)} -> {none_dirs.count('avant')} avant, "
         f"{none_dirs.count('arriere')} arriere, {none_dirs.count(None)} non comptés")
    if mode == "scene_tous":
        diff = sum(1 for b, d in zip(boats, dirs) if b[0] != "none" and b[0] != d)
        s += f" ; directions détectées contredites par la règle de scène : {diff}"
    return s


def render_table(title, cell_fn, days):
    header = ["Jour"] + [f"{h}-{h+1}h" for h in HOURS] + ["Total"]
    rows = []
    for d in days:
        rows.append([d["label"]] + [str(cell_fn(d, h)) for h in HOURS]
                    + [str(sum(cell_fn(d, h) for h in HOURS))])
    rows.append(["Total"]
                + [str(sum(cell_fn(d, h) for d in days)) for h in HOURS]
                + [str(sum(cell_fn(d, h) for d in days for h in HOURS))])
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    pad = lambda r: " | ".join(
        c.ljust(widths[i]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(r)
    )
    line = "-" * (sum(widths) + 3 * (len(widths) - 1))
    return "\n".join([title, line, pad(header), line]
                     + [pad(r) for r in rows] + [line])


def build_week_output(week_data, week_num, year, none_mode="ignore"):
    """week_data : {date: data}. Avalant puis montant associé, pour chaque catégorie."""
    days = sorted(
        ({"d": d, "label": f"{DAYS_FR[d.weekday()]} {d.day:02d}/{d.month:02d}"} for d in week_data),
        key=lambda x: x["d"].weekday(),
    )

    dir_label = dict(DIRS)
    tables = []

    def add(direction_key, title_suffix, cell_fn):
        tables.append(render_table(f"=== {dir_label[direction_key]} - {title_suffix} ===", cell_fn, days))

    # TOUS NAVIRES : avalant puis montant
    for key, _ in DIRS:
        add(key, "TOUS NAVIRES",
            lambda d, h, key=key: sum(
                per_hour.get(h, 0) for per_hour in week_data[d["d"]][key].values()
            ))

    # Par classe : avalant puis le montant associé juste en dessous
    for cls in CLASSES:
        for key, _ in DIRS:
            add(key, cls.upper(),
                lambda d, h, key=key, cls=cls: week_data[d["d"]][key].get(cls, {}).get(h, 0))

    header = (
        f"# Comptages fluvial - Semaine {week_num} ({year})\n"
        f"# Jours couverts : {', '.join(d['label'] for d in days)}\n"
        f"# Plage horaire étudiée : {FIRST_HOUR}h - {LAST_HOUR}h\n"
    )
    if none_mode != "ignore":   # défaut : en-tête identique à l'ancienne version
        header += f"# Directions none (--none-mode {none_mode}) : {NONE_MODE_LABELS[none_mode]}\n"
    header += "\n"
    return header + "\n\n".join(tables) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Exporte des tableaux hebdomadaires depuis les *_all_crossings.txt")
    ap.add_argument("annee", nargs="?", type=int, default=None,
                    help="Année des données (défaut: année courante)")
    ap.add_argument("--in", dest="in_dir", default="./output",
                    help="Dossier racine des résultats (un sous-dossier par scène) [défaut: ./output]")
    ap.add_argument("--out", dest="out_dir", default="./export",
                    help="Dossier racine de l'export (un sous-dossier par scène) [défaut: ./export]")
    ap.add_argument("--none-mode", choices=NONE_MODES, default="ignore",
                    help="Traitement des bateaux de direction 'none' : "
                         "ignore (défaut) | majoritaire | proportionnel | scene_none | scene_tous "
                         "(voir la docstring du script)")
    args = ap.parse_args()

    year = args.annee if args.annee is not None else date.today().year
    root = Path(args.in_dir)
    if not root.is_dir():
        sys.exit(f"Dossier introuvable : {root.resolve()}")

    out_root = Path(args.out_dir)

    scene_dirs = [d for d in root.iterdir() if d.is_dir()]
    if not scene_dirs:
        sys.exit(f"Aucun sous-dossier de scène dans {root.resolve()}")

    for scene_dir in scene_dirs:
        files = sorted(scene_dir.glob("*_all_crossings.txt"))
        if not files:
            print(f"-- {scene_dir.name} : aucun fichier *_all_crossings.txt, ignoré")
            continue

        weeks = {}  # week_num -> {date: data}
        scene = scene_key(scene_dir.name)
        if args.none_mode in SCENE_MODES and scene is None:
            print(f"-- {scene_dir.name} : scène non reconnue pour --none-mode {args.none_mode} "
                  f"(noms attendus : {', '.join(SCENE_ALIASES)}), ignorée")
            continue
        rule = f" ; scène reconnue : {scene}" if args.none_mode in SCENE_MODES else ""
        print(f"-- Scène : {scene_dir.name}  [none-mode : {args.none_mode}{rule}]")
        for f in files:
            try:
                d, boats, ignored = parse_file(f, year)
            except ValueError as e:
                print(f"   ERREUR {f.name} : {e}")
                continue
            wk = iso_week(d)
            weeks.setdefault(wk, {})
            if d in weeks[wk]:
                print(f"   ERREUR {f.name} : date déjà vue ({d}), fichier ignoré")
                continue
            # RNG propre à (scène, date du fichier) : résultat reproductible et indépendant des autres fichiers
            rng = random.Random(f"{scene_dir.name}|{d.isoformat()}")
            dirs, warning = resolve_directions(boats, args.none_mode, scene, rng)
            weeks[wk][d] = aggregate(boats, dirs)
            print(f"   OK {f.name} -> semaine {wk} "
                  f"(hors plage {FIRST_HOUR}-{LAST_HOUR}h : {ignored} détections ignorées)")
            if args.none_mode != "ignore":
                print(f"      {summarize(boats, dirs, args.none_mode)}")
            if warning:
                print(f"      ATTENTION : {warning}")

        suffix = "" if args.none_mode == "ignore" else f"_{args.none_mode}"
        for wk, week_data in sorted(weeks.items()):
            out_scene = out_root / scene_dir.name
            out_scene.mkdir(parents=True, exist_ok=True)
            out_path = out_scene / f"comptages_{scene_dir.name}_semaine_{wk:02d}_{year}{suffix}.txt"
            out_path.write_text(build_week_output(week_data, wk, year, args.none_mode), encoding="utf-8")
            print(f"   Écrit : {out_path} ({len(week_data)} jour(s), 14 tableaux)")


if __name__ == "__main__":
    main()
