# Boat Counter

Comptage et classification des bateaux à partir de vidéos de surveillance fluviale.
Le pipeline applique une soustraction de fond MOG2, détecte les franchissements de
lignes virtuelles (Line Gate Capture), puis classe chaque passage en direction
(avant/arrière) et en type de bateau à l'aide de DINOv2 + LogisticRegression.

## Structure du projet

```
Boat_Counter/
├── src/                            Code source
│   ├── main_pipeline.py             Point d'entrée du pipeline (une vidéo)
│   ├── batch_pipeline.py            Traitement par lot de tout le dossier videos/
│   ├── preprocess_mask_lines_date.py  Éditeur interactif (masque + lignes + date)
│   ├── mog2_background_subtraction.py  Soustraction de fond MOG2
│   ├── line_gate_capture.py        Détection des franchissements (LGC)
│   ├── detection_utils.py          Détection de rectangles + NMS
│   ├── shared_dino.py              Classification direction + type (DINOv2 fusionné)
│   ├── all_crossings_generator.py  Génération du résumé all_crossings.txt
│   ├── add_dates_to_crossings.py   Ajout des dates aux crossings
│   ├── visualization_utils.py      Visualisation des franchissements
│   ├── transforms.py               Transformations d'images partagées (CLAHE, gamma)
│   ├── export_weekly_tables.py     Export tableaux hebdomadaires (txt) depuis output/
│   ├── train_boat_type.py          Entraînement classifieur type de bateau
│   └── train_direction.py          Entraînement classifieur direction
├── models/                         Modèles entraînés (.joblib)
│   ├── direction_dinov2.joblib     Classifieur de direction
│   └── boat_type_dinov2.joblib     Classifieur de type de bateau
├── datasets/                       Images d'entraînement (non suivi git)
├── scenes/                         Masque + lignes + date de début de chaque scène (traitement par lot)
│   └── <scène>/mask.png, lines_date.json
├── videos/                         Vidéos d'entrée (non suivi git)
│   └── <scène>/*.mp4               Un sous-dossier par scène (même caméra, même cadrage)
├── output/                         Résultats finaux (non suivi git)
│   ├── extractions_<video>/        Crops + crossings.txt par ID
│   └── <video>_all_crossings.txt   Résumé global
├── export/                         Tableaux d'export (non suivi git, généré depuis output/)
│   └── <scène>/comptages_<scène>_semaine_<NN>_<annee>[_<none-mode>].txt
├── temp/                           Fichiers intermédiaires (non suivi git)
│   ├── <video>_mask.png            Masque de la zone d'eau
│   ├── <video>_lines_date.json     Lignes de comptage + date de début
│   ├── <video>_mog2.mp4            Vidéo masque MOG2
│   └── extractions/                Crops en cours de traitement
├── .gitignore
└── README.md
```

## Prérequis

- Python 3.10+
- OpenCV (`cv2`)
- PyTorch + torchvision
- scikit-learn
- joblib
- NumPy
- PIL (Pillow)

Installation des dépendances :

```bash
pip install opencv-python torch torchvision scikit-learn joblib numpy pillow
```

## Utilisation

### Lancer le pipeline complet

```bash
python3 ./src/main_pipeline.py --in ./videos/video.mp4 --out ./output
```

Arguments :

| Argument | Raccourci | Défaut | Description |
|---|---|---|---|
| `--video` | `--in`, `-i` | (requis) | Chemin de la vidéo d'entrée |
| `--out` | `-o` | `output` | Dossier des résultats finaux |
| `--temp` | — | `temp` | Dossier des fichiers intermédiaires |
| `--mask` | — | — | Masque existant à réutiliser (avec `--lines_json`) : l'éditeur n'est pas ouvert |
| `--lines_json` | — | — | Fichier lignes/date existant à réutiliser (avec `--mask`) |
| `--start_time` | — | — | Date de début `MM/DD HH:MM:SS` (remplace celle de `--lines_json`) |
| `--no_display` | — | — | Aucune fenêtre ni image de fin (les comptes restent dans `<vidéo>_all_crossings.txt`) |

### Traitement par lot (toutes les vidéos de `videos/`)

```bash
python3 ./src/batch_pipeline.py
```

Organisation attendue : un sous-dossier de `videos/` par scène (même cadrage = même masque et mêmes lignes),
par exemple `videos/archeveche/`, `videos/birhakeim/`, `videos/philippe/`, `videos/st_louis/`.

1. **Phase 1 (interactive, une seule fois par scène)** — Si `scenes/<scène>/` n'existe pas encore, l'éditeur
   s'ouvre sur la 1re vidéo du dossier (ordre alphabétique) : masque, lignes, et **date de début de cette vidéo**
   (touche `d`), puis `s` et `ESC`. Le masque et les lignes sont mémorisés dans `scenes/<scène>/`. Les scènes
   déjà configurées ne demandent rien.
2. **Phase 2 (sans aucune interaction)** — Chaque vidéo est traitée avec le masque et les lignes de sa scène ;
   la date de début est décalée de **+1 jour par vidéo** (1re vidéo = date saisie, 2e = +1 jour, etc.).
   Aucune fenêtre n'est ouverte et aucune image de fin n'est produite.

Résultats dans `output/<scène>/` (`<vidéo>_all_crossings.txt`, `extractions_<vidéo>/`).
Un récapitulatif (OK / ERREUR, nombre de bateaux montants/descendants) est affiché à la fin dans la console.

| Option | Description |
|---|---|
| `--only st_louis philippe` | Ne traiter que ces sous-dossiers |
| `--force` | Retraiter aussi les vidéos déjà traitées (par défaut elles sont ignorées : on peut relancer après une interruption) |
| `--videos`, `--out`, `--temp`, `--scenes` | Changer les dossiers (défauts : `videos`, `output`, `temp`, `scenes`) |

> Attention : le décalage de date suit l'ordre alphabétique des vidéos. Si un jour manque dans un dossier,
> les dates suivantes seront décalées — la date de début de chaque vidéo est affichée dans la console pour vérification.
> Pour refaire le masque d'une scène, supprimer `scenes/<scène>/`.

### Exporter les tableaux hebdomadaires

Une fois le pipeline terminé, on génère des tableaux hebdomadaires (un fichier txt
par semaine ISO et par scène) depuis les `*_all_crossings.txt` :

```bash
python3 ./src/export_weekly_tables.py 2026
python3 ./src/export_weekly_tables.py --none-mode majoritaire 2026
python3 ./src/export_weekly_tables.py --none-mode proportionnel 2026
python3 ./src/export_weekly_tables.py --none-mode scene_tous 2026
```

Résultats dans `export/<scène>/comptages_<scène>_semaine_<NN>_<annee>[_<none-mode>].txt` : 14 tableaux par
semaine (avalant puis montant associé, pour TOUS NAVIRES puis chaque classe), colonnes
5h-6h ... 21h-22h, lignes = jours de la semaine. Le suffixe `_<none-mode>` n'est présent
que si `--none-mode` est différent de `ignore` (pour ne pas écraser le résultat par défaut).

| Argument | Défaut | Description |
|---|---|---|
| `annee` | année courante | Année des données |
| `--in` | `./output` | Dossier racine des résultats, un sous-dossier par scène |
| `--out` | `./export` | Dossier racine de l'export, un sous-dossier par scène |
| `--none-mode` | `ignore` | Traitement des bateaux de direction `none` (voir ci-dessous) |

#### Traitement des directions `none` (`--none-mode`)

Un bateau `none` est un bateau dont l'algorithme n'a pas pu déterminer la direction. Par
défaut (`ignore`), il n'est pas compté — c'est le comportement historique, et le fichier
produit est alors identique octet pour octet à la version précédente.

| Mode | Effet |
|---|---|
| `ignore` (défaut) | Les `none` ne sont pas comptés. |
| `majoritaire` | Les `none` d'un fichier prennent la direction majoritaire de ce fichier (comparaison du nombre de `avant` / `arriere`). Égalité ou aucun bateau fiable : les `none` de ce fichier sont ignorés (avertissement dans le log). |
| `proportionnel` | Les `none` d'un fichier sont répartis entre avant et arrière selon la proportion `avant / (avant + arriere)` des bateaux bien détectés de ce fichier (`round(N_none × p_avant)` deviennent `avant`). Tirage aléatoire reproductible (graine interne fixe). Aucun bateau fiable : `none` ignorés (avertissement). |
| `scene_none` | Les `none` prennent la direction imposée par la scène (cf. ci-dessous). |
| `scene_tous` | **Tous** les bateaux prennent la direction imposée par la scène : la direction lue dans `all_crossings` n'est plus utilisée. |

Les statistiques de `majoritaire` et `proportionnel` sont calculées **séparément pour chaque
fichier** `all_crossings`, sur les bateaux de la plage 5h–22h.

**Règles par scène** (`scene_none` et `scene_tous`), déduites du nom du dossier de scène
(normalisé : minuscules, sans accents ni ponctuation, cherché par inclusion) :

| Scène | Alias reconnus | Direction imposée |
|---|---|---|
| `philippe` | `philippe` | toujours **avant** |
| `archeveche` | `archeveche` | toujours **avant** |
| `birkaheim` | `birkaheim` | toujours **avant** |
| `stlouis` | `stlouis`, `saintlouis` | **avant** si minute < 35, **arrière** si minute ≥ 35 (pour chaque heure) |

En mode `scene_none` ou `scene_tous`, une scène non reconnue est **ignorée** avec un message.
La liste des alias est éditable dans le dictionnaire `SCENE_ALIASES` en haut du script.

> Les modes `majoritaire`, `proportionnel` et `scene_*` sont des **hypothèses de
> modélisation**, pas des mesures. Les tableaux produits dans des modes différents ne sont
> pas comparables sans précaution : l'en-tête et le suffixe du fichier indiquent le mode
> utilisé. Le log décrit, pour chaque fichier, le devenir des `none` (et, en `scene_tous`,
> le nombre de directions détectées contredites par la règle de scène).

### Étapes du pipeline

1. **Création du masque et des lignes** — Une fenêtre interactive s'ouvre sur la
   première frame de la vidéo. Dessinez le polygone de la zone d'eau (clics gauches,
   puis clic droit pour fermer), placez les lignes de comptage (touche `l` puis deux
   clics gauches), saisissez la date de début (touche `d`), puis sauvegardez (touche
   `s`). Voir [Éditeur interactif](#éditeur-interactif).

2. **Soustraction de fond MOG2** — Applique MOG2 sur la vidéo en restreignant au
   masque. Produit une vidéo binaire (objets en blanc sur fond noir) dans `temp/`.

3. **Line Gate Capture (LGC)** — Détecte les rectangles (bateaux) dans la vidéo
   binaire et surveille les franchissements des gates. Extrait les crops depuis la
   vidéo couleur d'origine. **Classifie la direction (avant/arrière/none) ET le type
   de bateau en un seul passage DINOv2**. La classe `none` correspond à un faux positif,
   comme `noise` pour le type.

4. **Génération du résumé** — Agrège tous les crossings dans un fichier
   `<video>_all_crossings.txt` (comptes globaux, par type de bateau, détails par ID).

5. **Visualisation** — Affiche la première frame avec les lignes, flèches et compteurs de
   franchissements (fenêtre à fermer par une touche), sauf avec `--no_display` : aucune fenêtre ni image.

6. **Nettoyage** — Déplace `temp/extractions/` vers `output/extractions_<video>/`
   et supprime les fichiers intermédiaires de `temp/`. Au démarrage, un éventuel `temp/extractions/`
   résiduel (exécution interrompue) est supprimé pour ne pas polluer la vidéo en cours.

### Lancer une étape individuellement

Chaque script peut être lancé séparément :

```bash
# Éditeur de masque/lignes
python3 ./src/preprocess_mask_lines_date.py ./videos/video.mp4 --out ./temp

# Soustraction de fond MOG2
python3 ./src/mog2_background_subtraction.py -i ./videos/video.mp4 -o ./temp/video_mog2.mp4 -m ./temp/video_mask.png

# Line Gate Capture (inclut classification direction + type)
python3 ./src/line_gate_capture.py -v ./temp/video_mog2.mp4 -c ./videos/video.mp4 --lines_json ./temp/video_lines_date.json --temp ./temp
```

## Éditeur interactif

L'éditeur s'ouvre lors de l'étape 1 du pipeline. Contrôles :

| Touche / Clic | Action |
|---|---|
| Clic gauche | Ajouter un point au polygone (ou un point de ligne en mode ligne) |
| Clic droit | Fermer le polygone (≥ 3 points) |
| `l` | Activer/désactiver le mode ligne |
| `z` | Annuler le dernier point (polygone ou ligne) |
| `r` | Tout réinitialiser |
| `d` | Saisir la date de début (format `MM/DD HH:MM:SS`) |
| `f` | Saisir le frametime (en secondes, ex: `3` pour 1 image/3s) |
| `s` | Sauvegarder le masque, les lignes, la date et l'intervalle |
| `ESC` | Quitter sans sauvegarder |

## Format des fichiers de sortie

### `crossings.txt` (par ID)

```
gate_label	direction	direction_prob	frame_idx	timestamp	date
boat_type	boat_type_prob
```

- **Ligne 1** : `gate_label` (nom de la ligne), `direction` (-1=avant, 1=arrière, 0=none),
  `direction_prob` (0.0-1.0), `frame_idx`, `timestamp` (secondes), `date` (ajoutée par
  `add_dates_to_crossings.py`)
- **Ligne 2** : `boat_type` (ou `noise`, `empty`) et sa probabilité

### `<video>_all_crossings.txt`

Trois sections :

1. **Global** — Comptes par ligne (avant/arrière/none/total)
2. **Par type de bateau** — Comptes par type et par ligne (avant/arrière/none/total)
3. **Détails par ID** — Chaque passage avec direction, type, probabilités, frame,
   timestamp et date

## Modèles

Deux modèles DINOv2 + LogisticRegression sont utilisés :

| Fichier | Rôle | Classes |
|---|---|---|
| `models/direction_dinov2.joblib` | Direction | `avant`, `arriere`, `none` |
| `models/boat_type_dinov2.joblib` | Type de bateau | `administration`, `marchandise`, `passager_gros`, `passager_petit`, `plaisance`, `restaurant`, `noise` |

DINOv2 (`dinov2_vits14`) est chargé via `torch.hub` au runtime. Les classifieurs
LogisticRegression sont stockés au format `.joblib`.

### Ré-entraîner les modèles

Si vous souhaitez ré-entraîner les classifieurs (par exemple avec de nouvelles données ou
pour ajuster les classes), utilisez les scripts dans `src/` :

```bash
# Entraîner le classifieur de type de bateau
# Structure attendue de datasets/ :
#   datasets/
#   ├── administration/
#   ├── marchandise/
#   ├── passager_gros/
#   ├── passager_petit/
#   ├── plaisance/
#   ├── restaurant/
#   ├── noise/ (faux positifs : eau, crops trop zoomés, fragments sans information)

python3 ./src/train_boat_type.py \
    --data ./datasets \
    --out_dir ./models

# Entraîner le classifieur de direction
# Structure attendue de datasets/ :
#   datasets/
#   ├── avant/
#   ├── arriere/
#   └── none/ (optionnel, faux positifs / bruit)

python3 ./src/train_direction.py \
    --data ./datasets \
    --out_dir ./models
```

**Arguments communs** :
- `--data` : Chemin vers le dossier racine du dataset (structure ImageFolder)
- `--out_dir` : Dossier de sortie pour les modèles entraînés
- `--batch_size` : Taille des batches pour l'extraction des embeddings (défaut: 8)
- `--num_workers` : Nombre de workers pour le DataLoader (défaut: 0 pour Windows/CPU)

## Organisation des dossiers

| Dossier | Rôle | Git |
|---|---|---|
| `src/` | Code source | Suivi |
| `models/` | Modèles `.joblib` | Suivi |
| `datasets/` | Images d'entraînement | Ignoré |
| `videos/` | Vidéos d'entrée | Ignoré |
| `output/` | Résultats finaux | Ignoré |
| `temp/` | Fichiers intermédiaires | Ignoré |
