# -*- coding: utf-8 -*-

"""
ROCO Klassifikation mit:
- Rules
- CNN1
- CNN2 (CT vs Xray Spezialmodell)
- CNN3 Filter
- OCR Multipanel
- Tierfilter
- Agreement Filter
- Balanced Sampling

ROCO Struktur:

dataset_root/
    train/
        radiology/
            captions.txt
            images/
        non-radiology/
            captions.txt
            images/
    validation/
    test/

Beispiel:

python roco_pipeline.py \
  --dataset_root /home/user/ROCO \
  --output_csv /home/user/output.csv \
  --cnn_path /home/user/cnn1.pth \
  --cnn2_path /home/user/cnn2.pth \
  --cnn3_path /home/user/cnn3.pth
"""

from __future__ import annotations

import os
import re
import io
import json
import random
import string
import argparse

from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from PIL import Image

import easyocr

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

DEVICE = "cpu"

FINAL_CLASSES = [
    'ct',
    'ct_kombimodalitaet_spect+ct_pet+ct',
    'us',
    "mrt_body",
    "mrt_hirn",
    "xray",
    "xray_fluoroskopie_angiographie"
]

# WICHTIG:
# Reihenfolge MUSS identisch zum CNN Training sein
CNN_CLASS_NAMES = [
    'ct',
    'ct_kombimodalitaet_spect+ct_pet+ct',
    'us',
    "mrt_body",
    "mrt_hirn",
    "xray",
    "xray_fluoroskopie_angiographie"
]

CNN2_CLASS_NAMES = [
    "ct",
    "xray"
]

CNN3_CLASS_NAMES = [
    "histologie",
    "haut",
    "chart",
    "endoskopie",
    "mikroskopie",
    "chirurgie",
]


# ============================================================
# OCR
# ============================================================

ocrreader = easyocr.Reader(
    ['en'],
    gpu=False
)


# ============================================================
# TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ============================================================
# CNNs
# ============================================================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x


class ThirdCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),

            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        return self.net(x)


# ============================================================
# ROCO LOADER
# ============================================================

def parse_roco_captions_file(
    file_path: Path,
    split_name: str,
    domain_name: str,
    images_dir: Path
):

    rows = []

    with file_path.open("r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.rstrip("\n")

            if not line.strip():
                continue

            parts = line.split("\t", 1)

            if len(parts) != 2:
                continue

            roco_id = parts[0].strip()
            caption = parts[1].strip()

            image_path = images_dir / f"{roco_id}.jpg"

            if not image_path.exists():
                continue

            rows.append({
                "split": split_name,
                "domain": domain_name,
                "id": roco_id,
                "caption": caption,
                "image_path": str(image_path),
            })

    return rows


def load_roco_dataset(base_dir: Path):

    splits = ["train", "test", "validation"]
    domains = ["radiology", "non-radiology"]

    all_rows = []

    for split in splits:

        for domain in domains:

            subset_dir = base_dir / split / domain

            captions_file = subset_dir / "captions.txt"
            images_dir = subset_dir / "images"

            if not captions_file.exists():
                continue

            if not images_dir.exists():
                continue

            print(f"[INFO] Lade: {captions_file}")

            rows = parse_roco_captions_file(
                captions_file,
                split,
                domain,
                images_dir
            )

            all_rows.extend(rows)

    return all_rows


# ============================================================
# IMAGE
# ============================================================

def get_image_from_record(record):

    path = record.get("image_path")

    if path is None:
        return None

    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


# ============================================================
# NORMALIZE
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_for_rules(text):

    text = normalize_text(text).lower()

    text = re.sub(r"[_/\\\-]+", " ", text)

    return text


# ============================================================
# RULES
# ============================================================

CT_HYBRID_RULES = [
    r"\bpet\s*/\s*ct\b",
    r"\bpet-ct\b",
    r"\bspect\s*/\s*ct\b",
    r"\bspect-ct\b",
    r"\bfused pet-ct\b",
    r"\bfused spect-ct\b",
]

CT_RULES = [
    r"\bct\b",
    r"\bcomputed tomography\b",
    r"\bmdct\b",
    r"\bhrct\b",
]

US_RULES = [
    r"\bultrasound\b",
    r"\bsonography\b",
    r"\bultrasonography\b",
    r"\bdoppler\b",
]

XRAY_RULES = [
    r"\bx-ray\b",
    r"\bxray\b",
    r"\bradiograph\b",
]

XRAY_ANGIO_RULES = [
    r"\bangiography\b",
    r"\bfluoroscopy\b",
    r"\bdsa\b",
]

MRI_BRAIN_RULES = [
    r"\bbrain mri\b",
    r"\bcranial mri\b",
    r"\bflair\b",
    r"\bt1\b",
    r"\bt2\b",
]

MRI_BODY_RULES = [
    r"\bprostate mri\b",
    r"\bpelvic mri\b",
    r"\babdominal mri\b",
]

RULES = [
    ("ct_kombimodalitaet_spect+ct_pet+ct", CT_HYBRID_RULES),
    ("xray_fluoroskopie_angiographie", XRAY_ANGIO_RULES),
    ("us", US_RULES),
    ("mrt_hirn", MRI_BRAIN_RULES),
    ("mrt_body", MRI_BODY_RULES),
    ("ct", CT_RULES),
    ("xray", XRAY_RULES),
]


# ============================================================
# RULE CLASSIFY
# ============================================================

def rule_based_classify(text):

    t = normalize_for_rules(text)

    hits = defaultdict(int)

    for label, patterns in RULES:

        for p in patterns:

            try:
                if re.search(p, t):
                    hits[label] += 1
            except:
                pass

    if not hits:
        return "unknown"

    best = sorted(
        hits.items(),
        key=lambda x: x[1],
        reverse=True
    )[0][0]

    return best


# ============================================================
# OCR
# ============================================================

def run_ocr_pil(image):

    if image is None:
        return ""

    try:

        results = ocrreader.readtext(
            np.array(image),
            detail=0
        )

        return " ".join(results)

    except:
        return ""


# ============================================================
# CNN PREDICT
# ============================================================

def predict_with_cnn(
    model,
    image,
    transform,
    device,
    class_names
):

    if image is None:
        return [], {}

    try:

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():

            logits = model(img_tensor)

            probs = F.softmax(logits, dim=1)[0]

        probs = probs.cpu().numpy()

        top_indices = probs.argsort()[-3:][::-1]

        top3 = []

        for idx in top_indices:

            label = class_names[idx]
            prob = float(probs[idx])

            top3.append((label, prob))

        full_scores = {
            class_names[i]: float(probs[i])
            for i in range(len(class_names))
        }

        return top3, full_scores

    except:

        return [], {}


# ============================================================
# TOP
# ============================================================

def get_top_prediction(score_dict):

    if not score_dict:
        return "unknown", 0.0, 0.0

    sorted_items = sorted(
        score_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top1_label, top1_score = sorted_items[0]

    if len(sorted_items) > 1:
        top2_score = sorted_items[1][1]
    else:
        top2_score = 0.0

    margin = top1_score - top2_score

    return top1_label, top1_score, margin


# ============================================================
# MODEL CONTEXT
# ============================================================

class ModelContext:

    def __init__(
        self,
        cnn_model,
        cnn2_model,
        cnn3_model,
        transform,
        device
    ):

        self.cnn = cnn_model
        self.cnn2 = cnn2_model
        self.cnn3 = cnn3_model

        self.transform = transform
        self.device = device


# ============================================================
# PROCESS
# ============================================================

def process_single_record(
    r,
    ctx
):

    text = normalize_text(r["caption"])

    image = get_image_from_record(r)

    ocr_text = run_ocr_pil(image)

    combined_text = text + " " + ocr_text

    # ========================================================
    # RULES
    # ========================================================

    rule_pred = rule_based_classify(combined_text)

    # ========================================================
    # CNN1
    # ========================================================

    cnn_top3, cnn_scores = predict_with_cnn(
        ctx.cnn,
        image,
        ctx.transform,
        ctx.device,
        CNN_CLASS_NAMES
    )

    cnn_pred, cnn_conf, cnn_margin = get_top_prediction(
        cnn_scores
    )

    # ========================================================
    # CNN2
    # ========================================================

    if rule_pred == "ct":

        cnn2_top3, cnn2_scores = predict_with_cnn(
            ctx.cnn2,
            image,
            ctx.transform,
            ctx.device,
            CNN2_CLASS_NAMES
        )

        cnn2_pred, cnn2_conf, cnn2_margin = get_top_prediction(
            cnn2_scores
        )

        if cnn2_pred != "ct":

            r["is_filtered"] = True
            r["filter_reason"] = "cnn2_reject"

            return r

    # ========================================================
    # CNN3
    # ========================================================

    cnn3_top3, cnn3_scores = predict_with_cnn(
        ctx.cnn3,
        image,
        ctx.transform,
        ctx.device,
        CNN3_CLASS_NAMES
    )

    cnn3_pred, cnn3_conf, _ = get_top_prediction(
        cnn3_scores
    )

    if cnn3_conf > 0.90:

        r["is_filtered"] = True
        r["filter_reason"] = "cnn3"

        return r

    # ========================================================
    # AGREEMENT
    # ========================================================

    if rule_pred != cnn_pred:

        r["is_filtered"] = True
        r["filter_reason"] = "rule_cnn_disagreement"

        return r

    # ========================================================
    # SAVE
    # ========================================================

    r["ocr_text"] = ocr_text

    r["rule_pred"] = rule_pred

    r["cnn_pred"] = cnn_pred
    r["cnn_conf"] = cnn_conf

    r["cnn_top3"] = cnn_top3

    r["cnn3_pred"] = cnn3_pred
    r["cnn3_conf"] = cnn3_conf

    r["final_label"] = cnn_pred
    r["final_conf"] = cnn_conf

    r["is_filtered"] = False

    return r


# ============================================================
# PROCESS BATCH
# ============================================================

def process_batch(
    records,
    ctx
):

    processed = []

    with ThreadPoolExecutor(max_workers=8) as ex:

        futures = [
            ex.submit(
                process_single_record,
                r,
                ctx
            )
            for r in records
        ]

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing"
        ):

            try:

                out = fut.result()

                if out is None:
                    continue

                if out.get("is_filtered"):
                    continue

                processed.append(out)

            except Exception as e:

                print(e)

    return processed


# ============================================================
# SAVE IMAGES
# ============================================================

def save_selected_images(
    records,
    output_dir
):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    saved = 0

    for r in records:

        img = get_image_from_record(r)

        if img is None:
            continue

        label = r.get(
            "final_label",
            "unknown"
        )

        class_dir = output_dir / label

        class_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        fname = f"{r['id']}.jpg"

        path = class_dir / fname

        try:

            img.save(path)

            saved += 1

        except:
            pass

    print(f"Saved images: {saved}")


# ============================================================
# MAIN
# ============================================================

def classify_dataset(
    dataset_root: Path,
    output_csv: Path,
    cnn_path: str,
    cnn2_path: str,
    cnn3_path: str,
    per_class: int
):

    print("[INFO] Lade ROCO...")

    records_raw = load_roco_dataset(dataset_root)

    print(f"[INFO] Samples: {len(records_raw)}")

    random.shuffle(records_raw)

    # ========================================================
    # EARLY BALANCING
    # ========================================================

    buckets = defaultdict(list)

    for r in tqdm(records_raw):

        text = normalize_text(r["caption"])

        label = rule_based_classify(text)

        if label not in FINAL_CLASSES:
            continue

        if len(buckets[label]) >= per_class:
            continue

        r["rule_pred"] = label

        buckets[label].append(r)

        done = all(
            len(buckets[c]) >= per_class
            for c in FINAL_CLASSES
        )

        if done:
            break

    presample = []

    for c in FINAL_CLASSES:
        presample.extend(buckets[c])

    print(f"[INFO] Presample: {len(presample)}")

    # ========================================================
    # CNN LOAD
    # ========================================================

    print("[INFO] Lade CNNs...")

    cnn_model = SimpleCNN(
        num_classes=len(CNN_CLASS_NAMES)
    )

    cnn_model.load_state_dict(
        torch.load(cnn_path, map_location=DEVICE)
    )

    cnn_model.to(DEVICE).eval()

    cnn2_model = SimpleCNN(
        num_classes=len(CNN2_CLASS_NAMES)
    )

    cnn2_model.load_state_dict(
        torch.load(cnn2_path, map_location=DEVICE)
    )

    cnn2_model.to(DEVICE).eval()

    cnn3_model = ThirdCNN(
        num_classes=len(CNN3_CLASS_NAMES)
    )

    cnn3_model.load_state_dict(
        torch.load(cnn3_path, map_location=DEVICE)
    )

    cnn3_model.to(DEVICE).eval()

    ctx = ModelContext(
        cnn_model,
        cnn2_model,
        cnn3_model,
        transform,
        DEVICE
    )

    # ========================================================
    # PROCESS
    # ========================================================

    records = process_batch(
        presample,
        ctx
    )

    print(f"[INFO] Final: {len(records)}")

    # ========================================================
    # CSV
    # ========================================================

    df = pd.DataFrame(records)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_csv,
        index=False,
        encoding="utf-8"
    )

    print(output_csv)

    # ========================================================
    # IMAGES
    # ========================================================

    save_selected_images(
        records,
        output_csv.parent / "images"
    )

    print(df["final_label"].value_counts())


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True
    )

    parser.add_argument(
        "--cnn_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--cnn2_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--cnn3_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--per_class",
        type=int,
        default=5000
    )

    return parser.parse_args()


# ============================================================
# ENTRY
# ============================================================

def main():

    args = parse_args()

    classify_dataset(
        dataset_root=Path(args.dataset_root),
        output_csv=Path(args.output_csv),

        cnn_path=args.cnn_path,
        cnn2_path=args.cnn2_path,
        cnn3_path=args.cnn3_path,

        per_class=args.per_class
    )


if __name__ == "__main__":
    main()