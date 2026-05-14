# -*- coding: utf-8 -*-

"""
Iteratives konfliktfreies Zusammenfuehren (widerspruchsfreie Zuordnung) (paarweise dinjunkt) von
ROCO- oder PMC-Checkpoints.

Automatisches Finden aller Logs
Automatisches Finden der neuesten CSV
Konflikterkennung
Nur echte Neuitems
Disjunkte Klassen
Iteratives Auffuellen
Bildkopie beachten
Statistik
PMC oder ROCO Modus

PMC:
python mk_nonoverlapping_class_sets_pmc.py \
    --mode pmc \
    --root_dir /home/b/Dokumente \
    --output_dir /home/b/PycharmProjects/ba1pmc/merged_pmc

ROCO:
python mk_nonoverlapping_class_sets.py \
    --mode roco \
    --root_dir /home/b/Dokumente \
    --output_dir /home/b/PycharmProjects/ba2roco/merged_roco/

"""

from __future__ import annotations

import os
import os.path as osp
import re
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from datasets import Dataset, concatenate_datasets
import pandas as pd
from tqdm import tqdm
from PIL import Image
import io
import numpy as np

CLASSES = [
    "ct",
    "ct_kombimodalitaet_spect+ct_pet+ct",
    "us",
    "mrt_body",
    "mrt_hirn",
    "xray",
    "xray_fluoroskopie_angiographie",
]


# ============================================================
# FIND LOG DIRS
# ============================================================

def find_log_dirs(root_dir: Path, mode: str):

    found = []

    if mode == "roco":
        pattern = r"ROCO-checkpoints_log(\d+)"

    elif mode == "pmc":
        pattern = r"PMC-checkpoints_log(\d+)"

    else:
        raise ValueError(mode)

    for p in root_dir.iterdir():

        if not p.is_dir():
            continue

        m = re.match(pattern, p.name)

        if m:
            found.append((int(m.group(1)), p))

    found.sort(key=lambda x: x[0])

    return [x[1] for x in found]


# ============================================================
# FIND NEWEST CSV
# ============================================================

def find_latest_csv(log_dir: Path, mode: str):

    round_dir = log_dir / "round_checkpoints"

    if not round_dir.exists():
        return None

    # ========================================================
    # ROCO
    # ========================================================

    if mode == "roco":

        p = round_dir / "output_roco.csv"

        if p.exists():
            return p

        return None

    # ========================================================
    # PMC
    # ========================================================

    elif mode == "pmc":

        csvs = list(round_dir.glob("round_*.csv"))

        if not csvs:
            return None

        def extract_num(p):

            m = re.search(r"round_(\d+)\.csv", p.name)

            if not m:
                return -1

            return int(m.group(1))

        csvs.sort(key=extract_num)

        return csvs[-1]

    else:
        raise ValueError(mode)


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(p: Path):

    try:
        df = pd.read_csv(p)

        print(f"Geladen: {p}")
        print(df["final_label"].value_counts())

        return df

    except Exception as e:

        print(f"FEHLER bei {p}")
        print(e)

        return None


# ============================================================
# BUILD ITEM ID
# ============================================================

def build_item_id(row, mode):

    # ========================================================
    # ROCO
    # ========================================================

    if mode == "roco":

        for key in [
            "roco_id",
            "image_id",
            "image_path",
            "id",
        ]:
            if key in row and pd.notna(row[key]):
                return str(row[key])

    # ========================================================
    # PMC
    # ========================================================

    elif mode == "pmc":

        for key in [
            "row_id",
            "pmc_id",
            "image_path",
            "id",
        ]:
            if key in row and pd.notna(row[key]):
                return str(row[key])

    # ========================================================
    # FALLBACK
    # ========================================================

    caption = str(row.get("caption", ""))

    return caption.strip()


# ============================================================
# BUILD IMAGE PATH
# ============================================================

def get_image_path(row):

    for key in [
        "image_path",
        "saved_image_path",
        "exported_image_path",
    ]:

        if key in row and pd.notna(row[key]):

            return str(row[key])

    return None


# ============================================================
# MERGE
# ============================================================

def iterative_merge(dfs, mode, max_per_class):

    # id -> labels
    id_to_labels = defaultdict(set)

    # id -> first row
    id_to_row = {}

    print("\nAnalysiere Konflikte...")

    # ========================================================
    # PASS 1
    # ========================================================

    for df in dfs:

        for _, row in tqdm(df.iterrows(), total=len(df)):

            label = row["final_label"]

            if label not in CLASSES:
                continue

            item_id = build_item_id(row, mode)

            id_to_labels[item_id].add(label)

            if item_id not in id_to_row:
                id_to_row[item_id] = row

    # ========================================================
    # CONFLICTS
    # ========================================================

    conflicts = {
        k
        for k, v in id_to_labels.items()
        if len(v) > 1
    }

    print(f"\nKonflikte gefunden: {len(conflicts)}")

    # ========================================================
    # BUILD CLEAN DATASET
    # ========================================================

    print("\nErzeuge konfliktfreien Datensatz...")

    final_rows = []

    used_ids = set()

    per_class_counter = defaultdict(int)

    for df in dfs:

        for _, row in tqdm(df.iterrows(), total=len(df)):

            label = row["final_label"]

            if label not in CLASSES:
                continue

            item_id = build_item_id(row, mode)

            # Konflikt?
            if item_id in conflicts:
                continue

            # Bereits vorhanden?
            if item_id in used_ids:
                continue

            # Klasse voll?
            if per_class_counter[label] >= max_per_class:
                continue

            used_ids.add(item_id)

            final_rows.append(row)

            per_class_counter[label] += 1

    final_df = pd.DataFrame(final_rows)

    return final_df, conflicts



# ============================================================
# ARROW LOAD
# ============================================================

def find_arrow_files(dataset_root: Path):

    return sorted(
        dataset_root.glob("data-*.arrow")
    )


def load_arrow_shards(arrow_files):

    datasets_list = []

    for fp in tqdm(
        arrow_files,
        desc="Lade Arrow-Shards"
    ):

        ds = Dataset.from_file(str(fp))

        datasets_list.append(ds)

    if len(datasets_list) == 1:
        return datasets_list[0]

    return concatenate_datasets(datasets_list)


# ============================================================
# GET IMAGE FROM ROW
# ============================================================

def get_image_from_row(row):

    if row is None:
        return None

    # ========================================================
    # bytes
    # ========================================================

    if "jpg" in row and row["jpg"] is not None:

        try:

            return Image.open(
                io.BytesIO(row["jpg"])
            ).convert("RGB")

        except Exception:
            return None

    return None


# ============================================================
# EXPORT
# ============================================================
# ============================================================
# COPY IMAGES AUS BESTEHENDEN images_after_rounds
# ============================================================

def copy_images(df, output_dir, root_dir):

    img_root = output_dir / "images"

    img_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # ALLE images_after_rounds ORDNER FINDEN
    # ========================================================

    image_dirs = []

    for log_dir in find_log_dirs(root_dir, "pmc"):

        p = (
            log_dir
            / "round_checkpoints"
            / "images_after_rounds"
        )

        if p.exists():

            image_dirs.append(p)

    print("\nGefundene images_after_rounds:")

    for p in image_dirs:
        print(p)

    # ========================================================
    # DATEIINDEX AUFBAUEN
    # ========================================================

    print("\nBaue Bildindex...")

    file_index = {}

    for base in image_dirs:

        for img_path in tqdm(
            list(base.rglob("*.jpg")),
            desc=f"Indexiere {base.name}"
        ):

            file_index[img_path.name] = img_path

    print(f"\nIndexierte Bilder: {len(file_index)}")

    # ========================================================
    # EXPORT
    # ========================================================

    copied = 0
    failed = 0
    missing = 0

    print("\nKopiere Bilder...")

    for _, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):

        label = row.get(
            "final_label",
            "unknown"
        )

        pmc_id = str(
            row.get("pmc_id", "unknown")
        )

        row_id = str(
            row.get("row_id", "unknown")
        )

        fname = f"{pmc_id}_{row_id}.jpg"

        src = file_index.get(fname)

        if src is None:

            missing += 1

            if missing <= 25:
                print(f"\nNICHT GEFUNDEN: {fname}")

            continue

        class_dir = img_root / label

        class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        dst = class_dir / fname

        try:

            shutil.copy2(src, dst)

            copied += 1

        except Exception as e:

            failed += 1

            print("\nCOPY FEHLER:")
            print(src)
            print(e)

    print("\n======================")
    print("PMC EXPORT FERTIG")
    print("======================")

    print("Gespeichert:", copied)
    print("Nicht gefunden:", missing)
    print("Fehlgeschlagen:", failed)


# ============================================================
# STATS
# ============================================================

def print_stats(df, conflicts):

    print("\n==============================")
    print("FINALE KLASSENVERTEILUNG")
    print("==============================\n")

    vc = df["final_label"].value_counts()

    for c in CLASSES:

        print(f"{c}: {vc.get(c, 0)}")

    print(f"\nKonflikte entfernt: {len(conflicts)}")
    print(f"Gesamt final: {len(df)}")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["roco", "pmc"]
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True
    )

    parser.add_argument(
        "--max_per_class",
        type=int,
        default=5000
    )

    args = parser.parse_args()

    root_dir = Path(args.root_dir)

    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # FIND LOGS
    # ========================================================

    log_dirs = find_log_dirs(
        root_dir,
        args.mode
    )

    print("\nGefundene Logs:")

    for p in log_dirs:
        print(p)

    # ========================================================
    # FIND CSVS
    # ========================================================

    csvs = []

    for log_dir in log_dirs:

        csv_path = find_latest_csv(
            log_dir,
            args.mode
        )

        if csv_path is None:
            continue

        csvs.append(csv_path)

    print("\nGefundene CSVs:")

    for p in csvs:
        print(p)

    # ========================================================
    # LOAD
    # ========================================================

    dfs = []

    for p in csvs:

        df = load_csv(p)

        if df is not None:
            dfs.append(df)

    if not dfs:
        print("Keine CSVs geladen.")
        return

    # ========================================================
    # MERGE
    # ========================================================

    final_df, conflicts = iterative_merge(
        dfs,
        args.mode,
        args.max_per_class
    )

    # ========================================================
    # SAVE CSV
    # ========================================================

    out_csv = output_dir / f"merged_{args.mode}.csv"

    final_df.to_csv(out_csv, index=False)

    print(f"\nCSV gespeichert:")
    print(out_csv)

    # ========================================================
    # COPY IMAGES
    # ========================================================

    copy_images(
        final_df,
        output_dir,
        root_dir
        )

    # ========================================================
    # STATS
    # ========================================================

    print_stats(
        final_df,
        conflicts
    )

    print("\nFERTIG.")


# ============================================================

if __name__ == "__main__":
    main()