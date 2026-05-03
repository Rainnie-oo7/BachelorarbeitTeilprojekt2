# -*- coding: utf-8 -*-

import os
from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from collections import Counter
import random
import os.path as osp

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = osp.normpath(osp.join(osp.dirname(__file__), "dataset3"))
BASE_DIR = Path(BASE_DIR)

PCAM_DIR = BASE_DIR / "PatchCamelyon_PCam_histologie"
DOCF_DIR = BASE_DIR / "DocFigure_chart"

# ============================================================
# 1) PCAM H5 INSPEKTION
# ============================================================

def inspect_pcam():
    print("\n" + "="*60)
    print("PCam (.h5) Analyse")
    print("="*60)

    h5_files = list(PCAM_DIR.glob("*.h5"))

    for h5_path in h5_files:
        print(f"\nDatei: {h5_path.name}")

        with h5py.File(h5_path, 'r') as f:
            print("Keys:", list(f.keys()))

            for key in f.keys():
                data = f[key]
                print(f"\nKey: {key}")
                print("Shape:", data.shape)
                print("Dtype:", data.dtype)

                # Beispiel ansehen
                if len(data.shape) >= 3:
                    sample = data[0]
                    print("Sample min/max:", sample.min(), sample.max())

            # Typische Struktur:
            # x: Bilder
            # y: Labels

            if "x" in f and "y" in f:
                x = f["x"]
                y = f["y"]

                print("\nDataset Übersicht:")
                print("Anzahl Bilder:", x.shape[0])
                print("Bildgröße:", x.shape[1:])

                # Labels prüfen
                labels = y[:1000].flatten()
                print("Label-Verteilung (Sample):", Counter(labels))


# ============================================================
# 2) DOCFIGURE ANNOTATION PARSEN
# ============================================================

def parse_docfigure():
    print("\n" + "="*60)
    print("DocFigure Annotation Analyse")
    print("="*60)

    ann_dir = DOCF_DIR / "DocFigure_annotation" / "annotation"
    img_dir = DOCF_DIR / "DocFigure_image" / "images"

    train_file = ann_dir / "train.txt"
    test_file = ann_dir / "test.txt"

    def parse_file(txt_path):
        data = []

        with open(txt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Format:
                # filename.png, Label
                parts = line.split(",")

                if len(parts) < 2:
                    continue

                fname = parts[0].strip()
                label = parts[1].strip()

                data.append((fname, label))

        return data

    train_data = parse_file(train_file)
    test_data = parse_file(test_file)

    print(f"\nTrain Samples: {len(train_data)}")
    print(f"Test Samples: {len(test_data)}")

    # Labelverteilung
    all_labels = [l for _, l in train_data + test_data]
    label_counts = Counter(all_labels)

    print("\nLabel-Verteilung:")
    for k, v in label_counts.most_common():
        print(f"{k}: {v}")

    # Docfigure hat auch Labels, die nicht gebraucht werden
    EXCLUDE = {"3D objects", "Medical images", "Mask", "Natural images"}

    filtered = [(f, l) for f, l in train_data if l not in EXCLUDE]

    print(f"\nNach Filter:")
    print("Samples:", len(filtered))

    filtered_labels = Counter([l for _, l in filtered])
    print("\nGefilterte Labels:")
    for k, v in filtered_labels.most_common():
        print(f"{k}: {v}")

    # Beispielbilder prüfen
    print("\nBeispielbilder laden:")
    for fname, label in random.sample(filtered, min(5, len(filtered))):
        img_path = img_dir / fname

        if img_path.exists():
            try:
                img = Image.open(img_path)
                print(f"{fname} | {label} | size={img.size} mode={img.mode}")
            except Exception as e:
                print("Fehler bei:", fname, e)
        else:
            print("Nicht gefunden:", fname)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    inspect_pcam()
    parse_docfigure()