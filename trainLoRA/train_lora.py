# -*- coding: utf-8 -*-

"""
train_lora_fusion.py

Bachelorarbeit:
Vergleich mehrerer LoRA-Fusionsstrategien für medizinisches
Image-Text-Retrieval mit CLIP und BLIP.

UNTERSTÜTZTE MODELLE
--------------------
- CLIP (openai/clip-vit-base-patch32)
- BLIP Retrieval (Salesforce/blip-itm-base)

UNTERSTÜTZTE DATASETS
---------------------
- PMC-15M, 1.7M Subset
- ROCO v1

Klassenweise LoRA-Trainings
Gemeinsames LoRA
Fullfinetuning
Basismodellvergleich
Contrastive Retrieval Loss
Recall@K / MRR
LoRA Fusion:
    - Additive
    - NormMerge
    - CosineMerge
    - Hadamard
    - Weighted Average
    - SVD Merge
    - Layer-wise Merge

Speicherort d. Modelle (Unter Ubuntu)
/home/user/Dokumente/local_models/
│
├── clip-vit-base-patch32/
├── blip-retrieval-base/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── preprocessor_config.json
│   └── ...
├── blip-image-captioning-base/
├── biomedblip-base/
└── biomedclip/
    ├── open_clip_pytorch_model.bin
    └── tokenizer.txt
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import os.path as osp
import argparse
import itertools
import json
from pathlib import Path
import hashlib
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from transformers import (
    CLIPProcessor,
    CLIPModel,

    BlipProcessor,
    BlipForImageTextRetrieval,

    AutoProcessor,
    AutoModel
)

import open_clip

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)

# ============================================================
# LOKALE MODELLE
# ============================================================

PATH = osp.abspath(osp.join(osp.dirname(__file__), "../../../Dokumente"))
BASE_DIR = Path(PATH)

LOCAL_MODEL_ROOT = BASE_DIR / "local_models"

LOCAL_CLIP_PATH = LOCAL_MODEL_ROOT / "clip-vit-base-patch32"

LOCAL_BLIP_RETRIEVAL_PATH = \
    LOCAL_MODEL_ROOT / "blip-itm-base-coco"

LOCAL_BLIP_CAPTION_PATH = \
    LOCAL_MODEL_ROOT / "blip-image-captioning-base"

LOCAL_BIOMEDCLIP_PATH = \
    LOCAL_MODEL_ROOT / "biomedclip"
# ============================================================
# GLOBALS
# ============================================================

PMC_CLASSES = [
    'ct',
    'ct_kombimodalitaet_spect+ct_pet+ct',
    'us',
    "mrt_body",
    "mrt_hirn",
    "xray",
    "xray_fluoroskopie_angiographie"
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# ARGPARSE
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "clip",
            "blip",
            "biomedclip",
        ],
        required=True
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["pmc", "roco"],
        required=True
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True
    )

    # --------------------------------------------------------
    # TRAIN MODE
    # --------------------------------------------------------

    parser.add_argument(
        "--train_mode",
        type=str,
        choices=[
            "class_lora",
            "shared_lora",
            "fullfinetune",
            "baseline"
        ],
        required=True
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Größe der größten Klasse nach proportionalem Nested-Downsampling"
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)

    # --------------------------------------------------------
    # LORA
    # --------------------------------------------------------

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs"
    )

    # --------------------------------------------------------
    # FUSION
    # --------------------------------------------------------

    parser.add_argument(
        "--fusion_method",
        type=str,
        default=None,
        choices=[
            "additive",
            "norm",
            "cosine",
            "hadamard",
            "weighted",
            "svd",
            "layerwise"
        ]
    )

    parser.add_argument(
        "--fusion_paths",
        nargs="+",
        default=None
    )

    return parser.parse_args()

# ============================================================
# DATASET
# ============================================================

class ImageTextDataset(Dataset):

    def __init__(
        self,
        records,
        processor,
        model_type="clip"
    ):

        self.records = records
        self.processor = processor
        self.model_type = model_type

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        r = self.records[idx]

        image = Image.open(r["image"]).convert("RGB")
        caption = r["caption"]
        label = r["label"]

        return {
            "image": image,
            "caption": caption,
            "label": label,
            "sample_id": r["sample_id"]
        }

# ============================================================
# PMC LOADER
# ============================================================

def load_pmc_dataset(dataset_root):

    dataset_root = Path(dataset_root)

    csv_path = dataset_root / "round_014.csv"

    image_root = dataset_root / "images_after_rounds"

    df = pd.read_csv(csv_path)

    records = []

    print("Lade PMC Dataset...")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        try:

            caption = str(row["caption"])

            label = str(row["final_label"])

            pmc_id = str(row["pmc_id"])

            class_dir = image_root / label

            image_candidates = []

            for ext in ["jpg", "jpeg", "png", "webp"]:

                image_candidates.extend(
                    list(class_dir.glob(f"{pmc_id}*.{ext}"))
                )

            if len(image_candidates) == 0:
                continue

            image_path = image_candidates[0]

            records.append({
                "image": str(image_path),
                "caption": caption,
                "label": label,
                "sample_id": pmc_id
            })

        except Exception as e:

            print("Fehler:", e)

    print("Geladene Samples:", len(records))

    return records

# ============================================================
# ROCO LOADER
# ============================================================

def load_roco_dataset(dataset_root):

    dataset_root = Path(dataset_root)

    merged_csv = dataset_root / "merged_roco.csv"

    image_root = dataset_root / "images"

    df = pd.read_csv(merged_csv)

    records = []

    print("Lade ROCO Dataset...")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        try:

            image_id = str(row["pmc_id"])

            caption = str(row["caption"])

            label = str(row["final_label"])

            # # --------------------------------------------
            # # Gegenvergleich (erstmal) nur 2 Klassen
            # # --------------------------------------------
            #
            # if label not in [
            #     "mrt_hirn",
            #     "xray"
            # ]:
            #     continue

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

            image_path = image_candidates[0]

            records.append({
                "image": str(image_path),
                "caption": caption,
                "label": label,
                "sample_id": image_id
            })

        except Exception as e:

            print("ROCO Fehler:", e)

    print("ROCO Samples:", len(records))

    return records

# ============================================================
# DATASET DISPATCHER
# ============================================================

def load_dataset_by_name(name, dataset_root):

    if name == "pmc":
        return load_pmc_dataset(dataset_root)

    elif name == "roco":
        return load_roco_dataset(dataset_root)

    else:
        raise ValueError(name)

# ============================================================
# MODEL LOADER
# ============================================================
def load_model_and_processor(model_type, model_name=None):

    # ========================================================
    # CLIP
    # ========================================================

    if model_type == "clip":

        model_path = LOCAL_CLIP_PATH

        processor = CLIPProcessor.from_pretrained(
            model_path
        )

        model = CLIPModel.from_pretrained(
            model_path
        )

        target_modules = [
            "q_proj",
            "v_proj"
        ]

    # ========================================================
    # BLIP
    # ========================================================

    elif model_type == "blip":

        model_path = LOCAL_BLIP_RETRIEVAL_PATH

        processor = BlipProcessor.from_pretrained(
            model_path
        )

        model = BlipForImageTextRetrieval.from_pretrained(
            model_path
        )

        target_modules = [
            "query",
            "value"
        ]


    # ========================================================
    # BIOMEDCLIP
    # ========================================================

    elif model_type == "biomedclip":

        model_name = (
            "hf-hub:microsoft/"
            "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )

        model, _, preprocess = \
            open_clip.create_model_and_transforms(
                model_name
            )

        tokenizer = open_clip.get_tokenizer(
            model_name
        )

        processor = {
            "image_processor": preprocess,
            "tokenizer": tokenizer
        }

        target_modules = [
            "q_proj",
            "v_proj"
        ]
    else:
        raise ValueError(model_type)
    return model, processor, target_modules

# ============================================================
# FREEZE VISION ENCODER
# ============================================================


def freeze_vision_encoder(model):

    possible_names = [
        "vision_model",
        "vision_encoder",
        "visual"
    ]

    found = False

    for name in possible_names:

        if hasattr(model, name):

            print(f"Freeze: {name}")

            module = getattr(model, name)

            for p in module.parameters():
                p.requires_grad = False

            found = True

    if not found:
        print("WARNUNG: Kein Vision Encoder gefunden")

# ============================================================
# APPLY LORA
# ============================================================

def apply_lora(
    model,
    target_modules,
    args
):

    config = LoraConfig(
        task_type=None,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none"
    )

    freeze_vision_encoder(model)

    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    return model

# ============================================================
# COLLATE FN
# ============================================================

def build_collate_fn(processor, model_type):

    def collate(batch):
        labels = [x["label"] for x in batch]
        images = [x["image"] for x in batch]
        captions = [x["caption"] for x in batch]
        sample_ids = [x["sample_id"] for x in batch]
        # ====================================================
        # CLIP
        # ====================================================

        if model_type == "clip":

            enc = processor(
                text=captions,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

        # ====================================================
        # BIOMEDCLIP
        # ====================================================

        elif model_type == "biomedclip":

            image_tensors = torch.stack([
                processor["image_processor"](img)
                for img in images
            ])

            text_tokens = processor["tokenizer"](
                captions
            )

            enc = {
                "pixel_values": image_tensors,
                "text_tokens": text_tokens
            }

        # ====================================================
        # BLIP
        # ====================================================

        elif model_type in ["blip"]:

            enc = processor(
                images=images,
                text=captions,
                return_tensors="pt",
                padding=True,
                truncation=True
            )

        else:
            raise ValueError(model_type)
        enc["labels_raw"] = labels
        enc["sample_ids"] = sample_ids
        return enc

    return collate

# ============================================================
# CONTRASTIVE LOSS
# ============================================================

def contrastive_loss(image_embeds, text_embeds, temperature=0.07):

    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    logits = torch.matmul(
        image_embeds,
        text_embeds.T
    )

    logits = logits / temperature

    targets = torch.arange(
        logits.size(0),
        device=logits.device
    )

    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.T, targets)

    loss = (loss_i + loss_t) / 2

    return loss

# ============================================================
# GET EMBEDDINGS
# ============================================================

def get_embeddings(model, batch, model_type):

    # ========================================================
    # CLIP
    # ========================================================

    if model_type == "clip":

        outputs = model(**batch)

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

    # ========================================================
    # BIOMEDCLIP
    # ========================================================

    elif model_type == "biomedclip":

        image_embeds = model.encode_image(
            batch["pixel_values"]
        )

        text_embeds = model.encode_text(
            batch["text_tokens"]
        )

    # ========================================================
    # BLIP
    # ========================================================

    elif model_type == "blip":

        outputs = model(
            input_ids=batch["input_ids"],
            pixel_values=batch["pixel_values"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
            use_itm_head=False
        )

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

    else:
        raise ValueError(model_type)

    return image_embeds, text_embeds

# ============================================================
# TRAIN
# ============================================================
# ============================================================
# MODALITY CONTRASTIVE LOSS
# ============================================================

def class_weighted_contrastive_loss(
    image_embeds,
    text_embeds,
    labels,
    target_label,
    temperature=0.07,
    positive_weight=3.0
):

    image_embeds = F.normalize(
        image_embeds,
        dim=-1
    )

    text_embeds = F.normalize(
        text_embeds,
        dim=-1
    )

    logits = (
        image_embeds @ text_embeds.T
    ) / temperature

    targets = torch.arange(
        logits.size(0),
        device=logits.device
    )

    loss_i_all = F.cross_entropy(
        logits,
        targets,
        reduction="none"
    )

    weights = torch.ones_like(
        loss_i_all
    )

    for i, label in enumerate(labels):

        if label == target_label:
            weights[i] = positive_weight

    loss_i = (
        loss_i_all * weights
    ).mean()

    loss_t_all = F.cross_entropy(
        logits.T,
        targets,
        reduction="none"
    )

    loss_t = (
        loss_t_all * weights
    ).mean()

    return (loss_i + loss_t) / 2

def train_model(
    model,
    loader,
    args,
    model_type,
    output_dir,
    target_label=None
):

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4
    )

    model.train()

    for epoch in range(args.epochs):

        total_loss = 0.0

        pbar = tqdm(loader)

        for batch in pbar:

            new_batch = {}

            for k, v in batch.items():

                if isinstance(v, torch.Tensor):
                    new_batch[k] = v.to(DEVICE)

                else:
                    new_batch[k] = v

            batch = new_batch

            image_embeds, text_embeds = get_embeddings(
                model,
                batch,
                model_type
            )

            if target_label is None:

                loss = contrastive_loss(
                    image_embeds,
                    text_embeds
                )

            else:

                loss = class_weighted_contrastive_loss(
                    image_embeds,
                    text_embeds,
                    batch["labels_raw"],
                    target_label
                )

                if loss is None:
                    continue

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            pbar.set_description(
                f"Epoch {epoch} Loss {loss.item():.4f}"
            )

        avg_loss = total_loss / len(loader)

        print(f"Epoch {epoch}: {avg_loss:.4f}")

    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(output_dir)

# ============================================================
# RETRIEVAL EVAL
# ============================================================

@torch.no_grad()
def evaluate_retrieval(
    model,
    loader,
    model_type
):
    image_ids_all = []
    text_ids_all = []
    model.eval()

    image_embeds_all = []
    text_embeds_all = []

    for batch in tqdm(loader):

        new_batch = {}

        for k, v in batch.items():

            if isinstance(v, torch.Tensor):
                new_batch[k] = v.to(DEVICE)

            else:
                new_batch[k] = v

        batch = new_batch

        image_embeds, text_embeds = get_embeddings(
            model,
            batch,
            model_type
        )

        image_embeds_all.append(image_embeds.cpu())
        text_embeds_all.append(text_embeds.cpu())
        image_ids_all.extend(batch["sample_ids"])
        text_ids_all.extend(batch["sample_ids"])

    image_embeds_all = torch.cat(image_embeds_all)
    text_embeds_all = torch.cat(text_embeds_all)

    image_embeds_all = F.normalize(
        image_embeds_all,
        dim=-1
    )

    text_embeds_all = F.normalize(
        text_embeds_all,
        dim=-1
    )

    sims = image_embeds_all @ text_embeds_all.T

    recalls = {
        "R@1": 0,
        "R@5": 0,
        "R@10": 0
    }

    mrr = 0.0

    for i in range(len(sims)):

        ranking = torch.argsort(
            sims[i],
            descending=True
        )

        matches = []

        for rank_idx, text_idx in enumerate(ranking):

            if image_ids_all[i] == text_ids_all[text_idx]:
                matches.append(rank_idx)

        matches = torch.tensor(matches)

        if len(matches) == 0:
            continue

        rank = matches[0].item()

        if rank < 1:
            recalls["R@1"] += 1

        if rank < 5:
            recalls["R@5"] += 1

        if rank < 10:
            recalls["R@10"] += 1

        mrr += 1.0 / (rank + 1)

    n = len(sims)

    for k in recalls:
        recalls[k] /= n

    mrr /= n

    print("\n===== RETRIEVAL =====")

    for k, v in recalls.items():
        print(k, round(v, 4))

    print("MRR", round(mrr, 4))

# ============================================================
# LOAD LORA STATE
# ============================================================

def load_lora_state_dict(path):

    path = Path(path)

    weights_path = path / "adapter_model.bin"

    state = torch.load(
        weights_path,
        map_location="cpu"
    )

    return state

# ============================================================
# ADDITIVE FUSION
# ============================================================

def additive_fusion(states):

    merged = {}

    keys = states[0].keys()

    for k in keys:

        merged[k] = sum(
            s[k]
            for s in states
        )

    return merged

# ============================================================
# WEIGHTED FUSION
# ============================================================

def weighted_fusion(states, weights=None):

    if weights is None:

        weights = [
            1 / len(states)
            for _ in states
        ]

    merged = {}

    keys = states[0].keys()

    for k in keys:

        merged[k] = sum(
            w * s[k]
            for w, s in zip(weights, states)
        )

    return merged

# ============================================================
# HADAMARD FUSION
# ============================================================

def hadamard_fusion(states):

    merged = {}

    keys = states[0].keys()

    for k in keys:

        x = states[0][k]

        for s in states[1:]:

            x = x * s[k]

        merged[k] = x

    return merged

# ============================================================
# NORM MERGE
# ============================================================

def norm_merge(states):

    merged = {}

    keys = states[0].keys()

    for k in keys:

        tensors = []

        for s in states:

            t = s[k]

            norm = torch.norm(t)

            if norm > 0:
                t = t / norm

            tensors.append(t)

        merged[k] = sum(tensors)

    return merged

# ============================================================
# COSINE MERGE
# ============================================================

def cosine_merge(states):

    """
    Vereinfachte CosineMerge Version
    """

    merged = {}

    keys = states[0].keys()

    for k in keys:

        base = states[0][k]

        acc = torch.zeros_like(base)

        for s in states:

            t = s[k]

            sim = F.cosine_similarity(
                base.flatten(),
                t.flatten(),
                dim=0
            )

            acc += sim * t

        merged[k] = acc

    return merged

# ============================================================
# SVD MERGE
# ============================================================

def svd_merge(states, rank=8):

    merged = {}

    keys = states[0].keys()

    for k in keys:

        avg = sum(
            s[k]
            for s in states
        ) / len(states)

        shape = avg.shape

        if avg.ndim < 2:
            merged[k] = avg
            continue

        U, S, V = torch.svd(avg)

        S[rank:] = 0

        merged[k] = (
            U @ torch.diag(S) @ V.T
        ).reshape(shape)

    return merged

# ============================================================
# LAYERWISE MERGE
# ============================================================

def layerwise_merge(states, alpha=0.5):

    """
    Vereinfachte Layer-wise Fusion
    """

    merged = {}

    keys = states[0].keys()

    for k in keys:

        current = states[0][k]

        for s in states[1:]:

            current = (
                alpha * current
                + (1 - alpha) * s[k]
            )

        merged[k] = current

    return merged

# ============================================================
# SAVE MERGED LORA
# ============================================================

def save_merged_lora(merged_state, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    torch.save(
        merged_state,
        osp.join(output_dir, "adapter_model.bin")
    )

# ============================================================
# FUSION DISPATCHER
# ============================================================

def run_fusion(args):

    states = [
        load_lora_state_dict(p)
        for p in args.fusion_paths
    ]

    if args.fusion_method == "additive":
        merged = additive_fusion(states)

    elif args.fusion_method == "weighted":
        merged = weighted_fusion(states)

    elif args.fusion_method == "hadamard":
        merged = hadamard_fusion(states)

    elif args.fusion_method == "norm":
        merged = norm_merge(states)

    elif args.fusion_method == "cosine":
        merged = cosine_merge(states)

    elif args.fusion_method == "svd":
        merged = svd_merge(states)

    elif args.fusion_method == "layerwise":
        merged = layerwise_merge(states)

    else:
        raise ValueError(args.fusion_method)

    fusion_name = "_".join([
        Path(x).name
        for x in args.fusion_paths
    ])

    output_dir = osp.join(
        args.output_dir,
        f"fusion_{args.fusion_method}_{fusion_name}"
    )

    save_merged_lora(
        merged,
        output_dir
    )

    print("Fusion gespeichert:", output_dir)

# ============================================================
# CLASS SPLIT
# ============================================================

def split_by_class(records):

    grouped = defaultdict(list)

    for r in records:
        grouped[r["label"]].append(r)

    return grouped
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
# NESTED PROPORTIONAL DOWNSAMPLING
# ============================================================

def proportional_nested_downsample(
    records,
    max_per_class=5000
):

    grouped = defaultdict(list)

    # --------------------------------------------------------
    # GROUP BY CLASS
    # --------------------------------------------------------

    for r in records:

        grouped[r["label"]].append(r)

    # --------------------------------------------------------
    # DETERMINISTIC SORTING
    # --------------------------------------------------------

    for label in grouped:

        grouped[label] = sorted(
            grouped[label],
            key=lambda x: stable_hash(
                x["image"] + x["caption"]
            )
        )

    # --------------------------------------------------------
    # LARGEST CLASS
    # --------------------------------------------------------

    largest_class_size = max(
        len(v)
        for v in grouped.values()
    )

    print("\n========================================")
    print("ORIGINAL DISTRIBUTION")
    print("========================================")

    for label, samples in grouped.items():

        print(
            f"{label:<40}"
            f"{len(samples)}"
        )

    print("\nLargest class:", largest_class_size)

    # --------------------------------------------------------
    # SCALE FACTOR
    # --------------------------------------------------------

    scale = max_per_class / largest_class_size

    print("Scale:", round(scale, 6))

    # --------------------------------------------------------
    # BUILD NESTED SUBSET
    # --------------------------------------------------------

    sampled_records = []

    print("\n========================================")
    print("NESTED SUBSET")
    print("========================================")

    for label, samples in grouped.items():

        original_n = len(samples)

        target_n = int(original_n * scale)

        target_n = max(target_n, 1)

        target_n = min(target_n, original_n)

        # --------------------------------------------
        # WICHTIG:
        # samples[:target_n]
        #
        # Dadurch gilt:
        #
        # 1250 ⊂ 2500 ⊂ 5000
        # --------------------------------------------

        selected = samples[:target_n]

        sampled_records.extend(selected)

        print(
            f"{label:<40}"
            f"{original_n:>8} -> {target_n}"
        )

    print("\nFinal Samples:", len(sampled_records))

    return sampled_records

# ============================================================
# TRAIN / VAL / TEST SPLIT
# ============================================================

def split_dataset(
    records,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
):

    rng = np.random.RandomState(seed)

    grouped = defaultdict(list)

    for r in records:
        grouped[r["label"]].append(r)

    train_records = []
    val_records = []
    test_records = []

    print("\n========================================")
    print("DATASET SPLITS")
    print("========================================")

    for label, samples in grouped.items():

        rng.shuffle(samples)

        n = len(samples)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train = samples[:n_train]
        val = samples[n_train:n_train+n_val]
        test = samples[n_train+n_val:]

        train_records.extend(train)
        val_records.extend(val)
        test_records.extend(test)

        print(
            f"{label:<40}"
            f"train={len(train):<6}"
            f"val={len(val):<6}"
            f"test={len(test)}"
        )

    print("\nTOTAL:")
    print("train:", len(train_records))
    print("val:", len(val_records))
    print("test:", len(test_records))

    return train_records, val_records, test_records

# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # FUSION ONLY
    # --------------------------------------------------------

    if args.fusion_method is not None:

        run_fusion(args)

        return

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    records = load_dataset_by_name(
        args.dataset,
        args.dataset_root
    )

    # ============================================================
    # PROPORTIONAL NESTED DOWNSAMPLING
    # ============================================================

    if args.max_per_class is not None:
        records = proportional_nested_downsample(
            records,
            max_per_class=args.max_per_class
        )

    print("Samples:", len(records))

    train_records, val_records, test_records = split_dataset(
        records
    )
    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model, processor, target_modules = \
        load_model_and_processor(
            args.model,
            args.model_name
        )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    if args.train_mode == "baseline":
        train_dataset = ImageTextDataset(
            train_records,
            processor,
            args.model
        )

        val_dataset = ImageTextDataset(
            val_records,
            processor,
            args.model
        )

        test_dataset = ImageTextDataset(
            test_records,
            processor,
            args.model
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=build_collate_fn(
                processor,
                args.model
            )
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=build_collate_fn(
                processor,
                args.model
            )
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=build_collate_fn(
                processor,
                args.model
            )
        )

        evaluate_retrieval(
            model.to(DEVICE),
            test_loader,
            args.model
        )

        return

    # --------------------------------------------------------
    # FULL FINETUNING
    # --------------------------------------------------------

    if args.train_mode == "fullfinetune":
        train_dataset = ImageTextDataset(
            train_records,
            processor,
            args.model
        )

        val_dataset = ImageTextDataset(
            val_records,
            processor,
            args.model
        )

        test_dataset = ImageTextDataset(
            test_records,
            processor,
            args.model
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=build_collate_fn(
                processor,
                args.model
            )
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=build_collate_fn(
                processor,
                args.model
            )
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=build_collate_fn(
                processor,
                args.model
            )
        )

        output_dir = osp.join(
            args.output_dir,
            "fullfinetune"
        )

        train_model(
            model,
            train_loader,
            args,
            args.model,
            output_dir
        )

        print("\nVALIDATION")

        evaluate_retrieval(
            model,
            val_loader,
            args.model
        )

        print("\nTEST")

        evaluate_retrieval(
            model,
            test_loader,
            args.model
        )

        return

    # --------------------------------------------------------
    # APPLY LORA
    # --------------------------------------------------------

    model = apply_lora(
        model,
        target_modules,
        args
    )

    # --------------------------------------------------------
    # SHARED LORA
    # --------------------------------------------------------

    if args.train_mode == "shared_lora":
        train_dataset = ImageTextDataset(
            train_records,
            processor,
            args.model
        )

        val_dataset = ImageTextDataset(
            val_records,
            processor,
            args.model
        )

        test_dataset = ImageTextDataset(
            test_records,
            processor,
            args.model
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=build_collate_fn(
                processor,
                args.model
            )
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=build_collate_fn(
                processor,
                args.model
            )
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=build_collate_fn(
                processor,
                args.model
            )
        )

        output_dir = osp.join(
            args.output_dir,
            "shared_lora"
        )

        train_model(
            model,
            train_loader,
            args,
            args.model,
            output_dir
        )

        print("\nVALIDATION")

        evaluate_retrieval(
            model,
            val_loader,
            args.model
        )

        print("\nTEST")

        evaluate_retrieval(
            model,
            test_loader,
            args.model
        )

        return

    # --------------------------------------------------------
    # CLASS LORA
    # --------------------------------------------------------

    if args.train_mode == "class_lora":

        grouped = split_by_class(train_records)

        all_labels = list(grouped.keys())

        for target_label in all_labels:
            print("\n================================")
            print("TRAINIERE LORA:", target_label)
            print("================================")

            # ----------------------------------------------------
            # Neues Modell
            # ----------------------------------------------------

            model, processor, target_modules = \
                load_model_and_processor(
                    args.model,
                    args.model_name
                )

            model = apply_lora(
                model,
                target_modules,
                args
            )

            # ----------------------------------------------------
            # WICHTIG:
            # gesamtes Dataset
            # NICHT nur target_label
            # ----------------------------------------------------

            train_dataset = ImageTextDataset(
                train_records,
                processor,
                args.model
            )

            val_dataset = ImageTextDataset(
                val_records,
                processor,
                args.model
            )

            test_dataset = ImageTextDataset(
                test_records,
                processor,
                args.model
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=build_collate_fn(
                    processor,
                    args.model
                )
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=build_collate_fn(
                    processor,
                    args.model
                )
            )

            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=build_collate_fn(
                    processor,
                    args.model
                )
            )

            output_dir = osp.join(
                args.output_dir,
                f"class_lora_{target_label}"
            )

            train_model(
                model,
                train_loader,
                args,
                args.model,
                output_dir,
                target_label=target_label
            )

            print("\nVALIDATION")

            evaluate_retrieval(
                model,
                val_loader,
                args.model
            )

            print("\nTEST")

            evaluate_retrieval(
                model,
                test_loader,
                args.model
            )

# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()