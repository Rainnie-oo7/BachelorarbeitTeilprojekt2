#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from typing import List, Dict, Optional, Tuple
import re

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import BertTokenizer, BertModel

import os
import os.path as osp
# =========================================================
# Konfiguration
# =========================================================

OPERATING_PATH  = osp.normpath(osp.join(osp.dirname(__file__), "data"))
BASE_DIR = Path(OPERATING_PATH)
LOCAL_BIOMEDBERT_PATH = Path("/home/b/Dokumente/biomedbert")

OUTPUT_TXT = Path("caption_classes_rule_plus_biomedbert.txt")

BATCH_SIZE = 32
MAX_LENGTH = 128
DEBUG_LIMIT = 200  # z.B. None
DEVICE = "cpu"

# Wenn True: Falls BERT-Fallback keinen guten Abstand liefert -> Unklar
USE_UNCLEAR_LABEL = True
UNCLEAR_LABEL = "Unklar"

# Mindestabstand zwischen Top-1 und Top-2 Similarity, damit BERT-Fallback akzeptiert wird
MIN_MARGIN = 0.015


# =========================================================
# Klassenbeschreibungen für BERT-Fallback
# =========================================================

CLASS_TEXTS: Dict[str, str] = {

    "xray": (
        "X-ray radiography, radiograph, plain radiograph, chest x-ray, abdominal x-ray, "
        "skeletal radiograph, projection radiography, conventional x-ray, AP view, PA view, "
        "lateral view, portable x-ray, panoramic radiograph, dorsoplantar projection, radiographic examination, "
        "Fluoroscopy with x-ray guidance, fluoroscopic x-ray image, fluoroscopic guidance, "
        "real-time x-ray imaging, c-arm x-ray fluoroscopy, contrast fluoroscopy, interventional fluoroscopy, "
        "fluoroscopic radiograph, x-ray guided procedure."
    ),
    "xray_fluoroskopie_angiographie": (
        "Angioplasty procedure, balloon angioplasty, percutaneous transluminal angioplasty, PTA, PTCA, "
        "coronary angioplasty, vascular intervention, interventional catheter procedure, stent placement, "
        "angioplasty balloon inflation, percutaneous coronary intervention."
    ),
    "us": (
        "Ultrasound imaging, sonography, ultrasonography, echography, echocardiography, B-mode, "
        "doppler ultrasound, duplex sonography, endoscopic ultrasound, EUS, B-mode ultrasound, "
        "color doppler, power doppler, sonographic examination, ultrasonographic image."
    ),
    "MRI": (
        "Magnetic resonance imaging, MRI scan, MR image, T1-weighted MRI, T2-weighted MRI, FLAIR, DWI, "
        "diffusion-weighted imaging, gadolinium-enhanced MRI, contrast-enhanced MRI, fMRI, SShDWI, dSShDWI, DCE, TSE, 4xSOVS, P5S1, inphase, T1+C."

    ),
    "CT": (
        "Computed tomography, CT scan, computed tomographic image, axial CT, coronal CT, sagittal CT, "
        "contrast-enhanced CT, non-contrast CT, helical CT, multidetector CT, MDCT, HRCT."
    ),
    "ct_kombimodalitaet_spect+ct_pet+ct": (
        "PET/CT, PET-CT, fused PET-CT image, hybrid PET/CT imaging, "
        "co-registered PET and CT, combined positron emission tomography and computed tomography, "
        "Positron emission tomography, PET scan, FDG PET, PET imaging, 18F-FDG, tracer uptake, "
        "metabolic imaging, positron-emission tomography."
    ),
}


# =========================================================
# Regelbasierte Klassifikation
# =========================================================

def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


# Reihenfolge = Priorität
RULES: List[Tuple[str, List[str]]] = [
    (
        "KombiModalitaet(CT,PET)",
        [
            r"\bpet\s*/\s*ct\b",
            r"\bpet-ct\b",
            r"\bhybrid pet\s*/\s*ct\b",
            r"\bfused pet-ct\b",
            r"\bco-registered pet and ct\b",
            r"\bcombined positron emission tomography and computed tomography\b",
        ],
    ),
    (
        "MRI",
        [
            r"\bmri\b",
            r"\bmr image\b",
            r"\bmr images\b",
            r"\bmr scan\b",
            r"\bmagnetic resonance\b",
            r"\bt1[- ]weighted\b",
            r"\bt2[- ]weighted\b",
            r"\bflair\b",
            r"\bdwi\b",
            r"\bdiffusion-weighted\b",
            r"\bfmri\b",
        ],
    ),
    (
        "CT",
        [
            r"\bct\b",
            r"\bct scan\b",
            r"\bcomputed tomography\b",
            r"\bcomputed tomograph\w*\b",
            r"\baxial ct\b",
            r"\bcoronal ct\b",
            r"\bsagittal ct\b",
            r"\bcontrast-enhanced ct\b",
            r"\bnon-contrast ct\b",
            r"\bhelical ct\b",
            r"\bmdct\b",
            r"\bhrct\b",
        ],
    ),
    (
        "Ultraschall",
        [
            r"\bultrasound\b",
            r"\bsonograph\w*\b",
            r"\bultrasonograph\w*\b",
            r"\bechograph\w*\b",
            r"\bechocardiograph\w*\b",
            r"\bdoppler\b",
            r"\bduplex sonograph\w*\b",
            r"\bendoscopic ultrasound\b",
            r"\beus\b",
            r"\bb-mode ultrasound\b",
            r"\bcolor doppler\b",
            r"\bpower doppler\b",
        ],
    ),
    (
        "Fluoroskopie+Xray",
        [
            r"\bfluoroscopy\b",
            r"\bfluoroscopic\b",
            r"\bc-arm\b",
            r"\bx-ray guided\b",
            r"\bfluoroscopic guidance\b",
            r"\breal-time x-ray\b",
        ],
    ),
    (
        "PET",
        [
            r"\bpet\b",
            r"\bpet scan\b",
            r"\bfdg pet\b",
            r"\bpet imaging\b",
            r"\b18f-fdg\b",
            r"\bpositron emission tomography\b",
            r"\btracer uptake\b",
        ],
    ),
    (
        "X-ray",
        [
            r"\bx-ray\b",
            r"\bxray\b",
            r"\bradiograph\b",
            r"\bradiographic\b",
            r"\bprojection radiography\b",
            r"\bap view\b",
            r"\bpa view\b",
            r"\blateral view\b",
            r"\bportable x-ray\b",
            r"\bpanoramic radiograph\b",
            r"\bdorsoplantar projection\b",
            r"\bprojection\b",
        ],
    ),
    (
        "Angioplastie",
        [
            r"\bangioplast\w*\b",
            r"\bballoon angioplasty\b",
            r"\bpercutaneous transluminal angioplasty\b",
            r"\bpta\b",
            r"\bptca\b",
            r"\bcoronary angioplasty\b",
            r"\bstent placement\b",
            r"\bpercutaneous coronary intervention\b",
            r"\bpci\b",
        ],
    ),
]


def rule_based_classify(caption: str) -> Optional[str]:
    text = normalize_text(caption)

    for label, patterns in RULES:
        if contains_any(text, patterns):
            return label

    return None


# =========================================================
# Captions laden
# =========================================================

def parse_captions_file(file_path: Path, split_name: str, domain_name: str, images_dir: Path) -> pd.DataFrame:
    rows = []

    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                print(f"[WARN] Unerwartetes Format in {file_path} Zeile {line_number}: {line!r}")
                continue

            roco_id = parts[0].strip()
            caption = parts[1].strip()

            rows.append({
                "split": split_name,
                "domain": domain_name,
                "id": roco_id,
                "caption": caption,
                "image_path": str(images_dir / f"{roco_id}.jpg"),
            })

    return pd.DataFrame(rows)


def load_all_captions(base_dir: Path) -> pd.DataFrame:
    splits = ["train", "test", "validation"]
    domains = ["radiology", "non-radiology"]
    dfs = []

    for split in splits:
        for domain in domains:
            subset_dir = base_dir / split / domain
            captions_file = subset_dir / "captions.txt"
            images_dir = subset_dir / "images"

            if not captions_file.exists():
                print(f"[WARN] Datei fehlt: {captions_file}")
                continue
            if not images_dir.exists():
                print(f"[WARN] Bilderordner fehlt: {images_dir}")
                continue

            print(f"[INFO] Lese Captions aus: {captions_file}")
            dfs.append(parse_captions_file(captions_file, split, domain, images_dir))

    if not dfs:
        return pd.DataFrame(columns=["split", "domain", "id", "caption", "image_path"])

    return pd.concat(dfs, ignore_index=True)


# =========================================================
# BERT-Fallback
# =========================================================

def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer: BertTokenizer,
    model: BertModel,
    batch_size: int = 32,
    max_length: int = 128,
    device: str = "cpu",
) -> torch.Tensor:
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded)

        pooled = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        all_embeddings.append(pooled.cpu())

        end = min(start + batch_size, len(texts))
        print(f"[INFO] Encoded {end}/{len(texts)} Texte")

    return torch.cat(all_embeddings, dim=0)


def classify_by_similarity_with_margin(
    caption_embeddings: torch.Tensor,
    class_embeddings: torch.Tensor,
    class_names: List[str],
    min_margin: float = 0.015,
    unclear_label: Optional[str] = None,
) -> List[str]:
    similarity_matrix = caption_embeddings @ class_embeddings.T
    labels = []

    for row in similarity_matrix:
        values, indices = torch.topk(row, k=min(2, len(class_names)))
        top1_idx = indices[0].item()
        top1_val = values[0].item()

        if len(values) > 1:
            top2_val = values[1].item()
            margin = top1_val - top2_val
        else:
            margin = 999.0

        if unclear_label is not None and margin < min_margin:
            labels.append(unclear_label)
        else:
            labels.append(class_names[top1_idx])

    return labels


# =========================================================
# Ausgabe
# =========================================================

def write_txt_output(df: pd.DataFrame, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(f"{row['id']}\t{row['predicted_class']}\n")


# =========================================================
# Hauptprogramm
# =========================================================

def main():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Lade Daten aus: {BASE_DIR}")

    df = load_all_captions(BASE_DIR)

    if df.empty:
        print("[ERROR] Keine Captions gefunden.")
        return

    if DEBUG_LIMIT is not None:
        df = df.head(DEBUG_LIMIT).copy()
        print(f"[INFO] DEBUG_LIMIT aktiv: {len(df)} Zeilen")

    print(f"[INFO] Gesamtzahl Captions: {len(df)}")

    if not LOCAL_BIOMEDBERT_PATH.exists():
        raise FileNotFoundError(f"Modellpfad nicht gefunden: {LOCAL_BIOMEDBERT_PATH}")

    print(f"[INFO] Lade lokales BiomedBERT aus: {LOCAL_BIOMEDBERT_PATH}")
    tokenizer = BertTokenizer.from_pretrained(str(LOCAL_BIOMEDBERT_PATH), local_files_only=True)
    model = BertModel.from_pretrained(str(LOCAL_BIOMEDBERT_PATH), local_files_only=True)
    model.to(DEVICE)
    model.eval()

    # 1) Regeln zuerst
    df["rule_class"] = df["caption"].apply(rule_based_classify)

    num_rule_hits = df["rule_class"].notna().sum()
    num_fallback = df["rule_class"].isna().sum()

    print(f"[INFO] Durch Regeln klassifiziert: {num_rule_hits}")
    print(f"[INFO] BERT-Fallback noetig fuer: {num_fallback}")

    # 2) Fallback nur fuer unklare Captions
    if num_fallback > 0:
        class_names = list(CLASS_TEXTS.keys())
        class_texts = list(CLASS_TEXTS.values())

        print("[INFO] Encode Klassenbeschreibungen fuer Fallback ...")
        class_embeddings = encode_texts(
            texts=class_texts,
            tokenizer=tokenizer,
            model=model,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            device=DEVICE,
        )

        fallback_mask = df["rule_class"].isna()
        fallback_df = df.loc[fallback_mask].copy()

        print("[INFO] Encode Fallback-Captions ...")
        caption_embeddings = encode_texts(
            texts=fallback_df["caption"].fillna("").astype(str).tolist(),
            tokenizer=tokenizer,
            model=model,
            batch_size=BATCH_SIZE,
            max_length=MAX_LENGTH,
            device=DEVICE,
        )

        fallback_labels = classify_by_similarity_with_margin(
            caption_embeddings=caption_embeddings,
            class_embeddings=class_embeddings,
            class_names=class_names,
            min_margin=MIN_MARGIN,
            unclear_label=UNCLEAR_LABEL if USE_UNCLEAR_LABEL else None,
        )

        df["predicted_class"] = df["rule_class"]
        df.loc[fallback_mask, "predicted_class"] = fallback_labels
    else:
        df["predicted_class"] = df["rule_class"]

    # Falls durch irgendeinen Sonderfall noch leer:
    if USE_UNCLEAR_LABEL:
        df["predicted_class"] = df["predicted_class"].fillna(UNCLEAR_LABEL)

    print("\n[INFO] Erste 30 Ergebnisse:")
    print(df[["id", "caption", "rule_class", "predicted_class"]].head(30).to_string(index=False))

    print(f"\n[INFO] Schreibe Ausgabedatei: {OUTPUT_TXT}")
    write_txt_output(df, OUTPUT_TXT)

    print("[INFO] Fertig.")
    print(f"[INFO] Datei: {OUTPUT_TXT.resolve()}")


if __name__ == "__main__":
    main()