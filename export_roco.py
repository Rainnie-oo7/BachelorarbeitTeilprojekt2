# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from tqdm import tqdm
import shutil

# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "/home/b/Dokumente/ROCO-checkpoints_merged10/merged_roco.csv"

ROCO_ROOT = Path("/home/b/PycharmProjects/ba2roco/data")

OUTPUT_DIR = Path(
    "/home/b/Dokumente/ROCO-checkpoints_merged10/images"
)

FINAL_CLASSES = [
    'ct',
    'ct_kombimodalitaet_spect+ct_pet+ct',
    'us',
    "mrt_body",
    "mrt_hirn",
    "xray",
    "xray_fluoroskopie_angiographie"
]

# ============================================================
# SPLITS
# ============================================================

IMAGE_DIRS = [

    ROCO_ROOT / "train" / "radiology" / "images",
    ROCO_ROOT / "test" / "radiology" / "images",
    ROCO_ROOT / "val" / "radiology" / "images",
    ROCO_ROOT / "validation" / "radiology" / "images",
]

# ============================================================
# CSV LADEN
# ============================================================

df = pd.read_csv(CSV_PATH)

print("CSV geladen:", len(df))

# ============================================================
# FILTER
# ============================================================

df = df[
    df["final_label"].isin(FINAL_CLASSES)
]

print("Nach Klassenfilter:", len(df))

# ============================================================
# DEBUG
# ============================================================

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print(df.columns.tolist())

print(df[["pmc_id", "final_label"]].head())

# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# HELPER
# ============================================================

def find_image(roco_id):

    filename_jpg = f"{roco_id}.jpg"
    filename_png = f"{roco_id}.png"

    for d in IMAGE_DIRS:

        if not d.exists():
            continue

        p = d / filename_jpg

        if p.exists():
            return p

        p = d / filename_png

        if p.exists():
            return p

    return None

# ============================================================
# EXPORT
# ============================================================

saved = 0
failed = 0

for idx, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Exportiere Bilder"
):

    # ========================================================
    # WICHTIG:
    # ROCO-ID steht bei dir in pmc_id
    # ========================================================

    roco_id = row.get("pmc_id")

    if pd.isna(roco_id):
        failed += 1
        continue

    roco_id = str(roco_id).strip()

    # ========================================================
    # LABEL
    # ========================================================

    label = str(
        row.get("final_label", "unknown")
    ).strip()

    # ========================================================
    # Bild suchen
    # ========================================================

    image_path = find_image(roco_id)

    if image_path is None:

        failed += 1

        print(f"\nNICHT GEFUNDEN: {roco_id}")

        continue

    # ========================================================
    # Klassenordner
    # ========================================================

    class_dir = OUTPUT_DIR / label

    class_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # Dateiname
    # ========================================================

    row_id = row.get("row_id", idx)

    ext = image_path.suffix

    filename = f"{roco_id}_{row_id}{ext}"

    out_path = class_dir / filename

    # ========================================================
    # COPY
    # ========================================================

    try:

        shutil.copy2(
            str(image_path),
            str(out_path)
        )

        saved += 1

    except Exception as e:

        failed += 1

        print("\nFEHLER:")
        print(image_path)
        print(e)

# ============================================================
# DONE
# ============================================================

print("\n======================")
print("FERTIG")
print("======================")

print("Gespeichert:", saved)
print("Fehlgeschlagen:", failed)