# -*- coding: utf-8 -*-

"""
evaluate_all_fusions.py

Automatische Evaluation aller gespeicherten LoRA-Fusionen
für CLIP Retrieval (Image-Text-Retrieval).

Unterstützt:
- ROCO
- PMC
- Einzel-LoRAs
- Fusionierte LoRAs

Metriken:
- Recall@1
- Recall@5
- Recall@10
- MRR

Rekonstruiert automatisch exakt denselben
Train/Val/Test Split wie im LoRA-Training. durch stable_hash() und gleichem Seed 42.

Dadurch:
- kein Data Leakage
- identische Testdaten
- wissenschaftlich saubere Recall@K Evaluation

BEISPIEL

python trainLoRA/evaluate_all_fusions.py \
    --roco_lora_root /home/b/PycharmProjects/ba2roco/LoRAs \
    --pmc_lora_root /home/b/PycharmProjects/ba1pmc/LoRAs \
    --roco_csv /home/b/Dokumente/ROCO-checkpoints_merged10/merged_roco.csv \
    --pmc_csv /home/b/Dokumente/PMC-checkpoints_merged/merged_pmc.csv \
    --output_csv /home/b/PycharmProjects/ba2roco/trainLoRA/fusion_eval.csv

    find /home/b/Dokumente/ROCO-fusions -name "adapter_model.bin"

und:

find /home/b/Dokumente/PMC-fusions -name "adapter_model.bin"
"""
from __future__ import annotations

import os
import os.path as osp
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from transformers import (
    CLIPProcessor,
    CLIPModel
)
from peft import get_peft_model, LoraConfig
from peft import PeftModel


# ============================================================
# CONFIG
# ============================================================

PMC_CLASS_CAPS = {

    "ct": 1000,
    "us": 1000,
    "mrt_body": 1000,
    "mrt_hirn": 1000,
    "xray": 1000,
    "xray_fluoroskopie_angiographie": 1000,
    "ct_kombimodalitaet_spect+ct_pet+ct": 752,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# STABLE HASH
# ============================================================

def stable_hash(x):

    return int(
        hashlib.md5(
            str(x).encode()
        ).hexdigest(),
        16
    )


# ============================================================
# DATASET
# ============================================================

class RetrievalDataset(Dataset):

    def __init__(self, records):

        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        r = self.records[idx]

        image = Image.open(
            r["image"]
        ).convert("RGB")

        return image, r["caption"], r["sample_id"]


# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):

    images = [x[0] for x in batch]
    captions = [x[1] for x in batch]
    sample_ids = [x[2] for x in batch]

    return images, captions, sample_ids


# ============================================================
# LOAD ROCO
# ============================================================

def load_roco_records(dataset_root):

    dataset_root = Path(dataset_root)

    csv_path = dataset_root / "merged_roco.csv"

    image_root = dataset_root / "images"

    df = pd.read_csv(csv_path)

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        try:

            image_id = str(row["pmc_id"])

            caption = str(row["caption"])

            label = str(row["final_label"])

            class_dir = image_root / label

            image_candidates = []

            for ext in [
                "jpg",
                "jpeg",
                "png",
                "webp"
            ]:

                image_candidates.extend(
                    list(
                        class_dir.glob(
                            f"{image_id}*.{ext}"
                        )
                    )
                )

            if len(image_candidates) == 0:
                continue

            records.append({
                "image": str(image_candidates[0]),
                "caption": caption,
                "label": label,
                "sample_id": image_id
            })

        except:
            pass

    return records


# ============================================================
# LOAD PMC
# ============================================================

def load_pmc_records(dataset_root):

    dataset_root = Path(dataset_root)

    csv_path = dataset_root / "merged_pmc.csv"

    image_root = dataset_root / "images"

    df = pd.read_csv(csv_path)

    records = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        try:

            image_id = str(row["pmc_id"])

            caption = str(row["caption"])

            label = str(row["final_label"])

            class_dir = image_root / label

            image_candidates = []

            for ext in [
                "jpg",
                "jpeg",
                "png",
                "webp"
            ]:

                image_candidates.extend(
                    list(
                        class_dir.glob(
                            f"{image_id}*.{ext}"
                        )
                    )
                )

            if len(image_candidates) == 0:
                continue

            records.append({
                "image": str(image_candidates[0]),
                "caption": caption,
                "label": label,
                "sample_id": image_id
            })

        except:
            pass

    return records


# ============================================================
# RECONSTRUCT CAPS
# ============================================================

def cap_samples_per_class(records):

    grouped = defaultdict(list)

    for r in records:
        grouped[r["label"]].append(r)

    for label in grouped:

        grouped[label] = sorted(
            grouped[label],
            key=lambda x: stable_hash(
                x["image"] + x["caption"]
            )
        )

    sampled = []

    for label, samples in grouped.items():

        cap = PMC_CLASS_CAPS.get(
            label,
            len(samples)
        )

        selected = samples[:cap]

        sampled.extend(selected)

    return sampled


# ============================================================
# RECONSTRUCT SPLIT
# ============================================================

def split_dataset(
        records,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42):

    rng = np.random.RandomState(seed)

    grouped = defaultdict(list)

    for r in records:
        grouped[r["label"]].append(r)

    test_records = []

    for label, samples in grouped.items():

        rng.shuffle(samples)

        n = len(samples)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        test = samples[n_train + n_val:]

        test_records.extend(test)

    return test_records


# ============================================================
# EMBEDDINGS
# ============================================================

@torch.no_grad()
def compute_embeddings(
        model,
        processor,
        loader):

    all_img = []
    all_txt = []
    all_ids = []

    for images, captions, sample_ids in tqdm(loader):

        img_inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        txt_inputs = processor(
            text=captions,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)

        outputs = model.base_model(

            input_ids=txt_inputs["input_ids"],
            attention_mask=txt_inputs["attention_mask"],
            pixel_values=img_inputs["pixel_values"],
            return_dict=True
        )

        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        image_features = F.normalize(
            image_features,
            dim=-1
        )

        text_features = F.normalize(
            text_features,
            dim=-1
        )

        all_img.append(image_features.cpu())
        all_txt.append(text_features.cpu())
        all_ids.extend(sample_ids)

    return (
        torch.cat(all_img),
        torch.cat(all_txt),
        all_ids
    )


# ============================================================
# METRICS
# ============================================================

def retrieval_metrics(
        image_embeds,
        text_embeds,
        sample_ids):

    sims = image_embeds @ text_embeds.T

    recalls = {
        "R@1": 0,
        "R@5": 0,
        "R@10": 0
    }

    mrr = 0.0

    n = len(sample_ids)

    for i in range(n):

        ranking = torch.argsort(
            sims[i],
            descending=True
        )
        # ========================================================
        # TEXT -> IMAGE
        # ========================================================

        t2i_recalls = {
            "T2I_R@1": 0,
            "T2I_R@5": 0,
            "T2I_R@10": 0
        }

        t2i_mrr = 0.0

        for i in range(n):

            ranking = torch.argsort(
                sims[:, i],
                descending=True
            )

            rank = None

            for r, idx in enumerate(ranking):

                if sample_ids[idx] == sample_ids[i]:
                    rank = r
                    break

            if rank is None:
                continue

            if rank < 1:
                t2i_recalls["T2I_R@1"] += 1

            if rank < 5:
                t2i_recalls["T2I_R@5"] += 1

            if rank < 10:
                t2i_recalls["T2I_R@10"] += 1

            t2i_mrr += 1.0 / (rank + 1)

        for k in t2i_recalls:
            t2i_recalls[k] /= n

        t2i_mrr /= n

        rank = None

        for r, idx in enumerate(ranking):

            if sample_ids[idx] == sample_ids[i]:
                rank = r
                break

        if rank is None:
            continue

        if rank < 1:
            recalls["R@1"] += 1

        if rank < 5:
            recalls["R@5"] += 1

        if rank < 10:
            recalls["R@10"] += 1

        mrr += 1.0 / (rank + 1)

    for k in recalls:
        recalls[k] /= n

    mrr /= n

    return {
        "Recall@1": round(recalls["R@1"], 4),
        "Recall@5": round(recalls["R@5"], 4),
        "Recall@10": round(recalls["R@10"], 4),
        "MRR": round(mrr, 4),

        # Image -> Text

        "I2T_R@1": round(recalls["R@1"], 4),
        "I2T_R@5": round(recalls["R@5"], 4),
        "I2T_R@10": round(recalls["R@10"], 4),
        "I2T_MRR": round(mrr, 4),

        # Text -> Image

        "T2I_R@1": round(t2i_recalls["T2I_R@1"], 4),
        "T2I_R@5": round(t2i_recalls["T2I_R@5"], 4),
        "T2I_R@10": round(t2i_recalls["T2I_R@10"], 4),
        "T2I_MRR": round(t2i_mrr, 4),
    }


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(lora_path):

    base = CLIPModel.from_pretrained(
        "/home/b/Dokumente/local_models/clip-vit-base-patch32"
    )

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none"
    )

    model = get_peft_model(base, config)

    weights_path_bin = osp.join(
        lora_path,
        "adapter_model.bin"
    )

    weights_path_safe = osp.join(
        lora_path,
        "adapter_model.safetensors"
    )

    if osp.exists(weights_path_safe):

        from safetensors.torch import load_file

        state_dict = load_file(weights_path_safe)

    else:

        state_dict = torch.load(
            weights_path_bin,
            map_location="cpu"
        )

    model.load_state_dict(
        state_dict,
        strict=False
    )

    processor = CLIPProcessor.from_pretrained(
        "/home/b/Dokumente/local_models/clip-vit-base-patch32"
    )

    model.to(DEVICE)
    model.eval()

    return model, processor


# ============================================================
# FIND ALL LORAS
# ============================================================

def find_all_loras(root):

    found = []

    for dirpath, _, filenames in os.walk(root):

        if (
            "adapter_model.bin" in filenames
            or
            "adapter_model.safetensors" in filenames
        ):
            found.append(dirpath)

    return sorted(found)


# ============================================================
# EVAL DATASET
# ============================================================

def evaluate_dataset(
        dataset_name,
        dataset_root,
        lora_root):

    print("\n==============================")
    print(dataset_name)
    print("==============================")

    if dataset_name == "roco":
        records = load_roco_records(dataset_root)

    else:
        records = load_pmc_records(dataset_root)

    # exakt wie Training

    records = cap_samples_per_class(records)

    # exakt wie Training

    test_records = split_dataset(records)

    print("TEST SAMPLES:", len(test_records))

    dataset = RetrievalDataset(test_records)

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    loras = find_all_loras(lora_root)

    results = []
    print("\nFOUND LORAS:")
    for x in loras:
        print(x)

    print("COUNT:", len(loras))
    for lora_path in loras:

        print("\nEVAL:", lora_path)

        try:

            model, processor = load_model(
                lora_path
            )

            image_embeds, text_embeds, sample_ids = \
                compute_embeddings(
                    model,
                    processor,
                    loader
                )

            metrics = retrieval_metrics(
                image_embeds,
                text_embeds,
                sample_ids
            )

            result = {
                "dataset": dataset_name,
                "lora": lora_path,
                **metrics
            }

            print(result)

            results.append(result)


        except Exception as e:

            import traceback

            print("\n================================")

            print("FAILED")

            print(lora_path)

            print("================================")

            traceback.print_exc()

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    all_results = []

    all_results.extend(
        evaluate_dataset(
            dataset_name="roco",
            dataset_root="/home/b/Dokumente/ROCO-checkpoints_merged10",
            lora_root="/home/b/PycharmProjects/ba2roco/LoRAs"
        )
    )

    all_results.extend(
        evaluate_dataset(
            dataset_name="pmc",
            dataset_root="/home/b/Dokumente/PMC-checkpoints_merged",
            lora_root="/home/b/PycharmProjects/ba1pmc/LoRAs"
        )
    )

    df = pd.DataFrame(all_results)

    print("\n================================")
    print("ALL RESULTS")
    print("================================")

    print(all_results)

    if len(all_results) == 0:
        print("\nKEINE ERFOLGREICHE EVALUATION.")

        return

    df = pd.DataFrame(all_results)

    print("\nDATAFRAME COLUMNS:")
    print(df.columns.tolist())

    if "Recall@1" not in df.columns:
        print("\nRecall@1 NICHT GEFUNDEN")
        print(df.head())

        return

    df = df.sort_values(
        by="Recall@1",
        ascending=False
    )

    output_csv = "/home/b/PycharmProjects/ba2roco/trainLoRA/fusion_eval.csv"

    df.to_csv(
        output_csv,
        index=False
    )

    print("\nGESPEICHERT:")
    print(output_csv)

if __name__ == "__main__":
    main()