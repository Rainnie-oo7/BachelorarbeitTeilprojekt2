# -*- coding: utf-8 -*-

import os
from pathlib import Path
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
import pandas as pd
import numpy as np
import h5py

# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path("/home/b/PycharmProjects/ba2roco/cnn/dataset3")

CLASSES = [
    "histologie",
    "haut",
    "chart",
    "endoskopie",
    "mikroskopie",
    "chirurgie"
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ============================================================
# HELPER
# ============================================================

def infer_class_from_folder(path: Path):
    name = path.parts[-2]
    if "_" not in name:
        return None
    suffix = name.split("_")[-1].lower()
    return suffix if suffix in CLASS_TO_IDX else None


# ============================================================
# MAIN DATASET
# ============================================================

class MultiDataset(Dataset):
    def __init__(self):
        self.samples = []

        self.collect_folder_datasets()
        self.collect_docfigure()
        self.collect_chartqa()
        self.collect_pcam()

        print("Total samples:", len(self.samples))

        # class distribution
        counts = Counter([s["label"] for s in self.samples])
        print("Class distribution:", counts)

        self.compute_class_weights(counts)

    # --------------------------------------------------------
    # CLASS WEIGHTS
    # --------------------------------------------------------
    def compute_class_weights(self, counts):
        total = sum(counts.values())
        num_classes = len(CLASSES)

        weights = []
        for c in CLASSES:
            n_c = counts.get(c, 1)
            w = total / (num_classes * n_c)
            weights.append(w)

        weights = torch.tensor(weights, dtype=torch.float32)
        weights = torch.clamp(weights, max=10.0)

        self.class_weights = weights

    # --------------------------------------------------------
    # GENERIC FOLDER DATASETS
    # --------------------------------------------------------
    def collect_folder_datasets(self):
        print("Scanning folder datasets...")

        for root, _, files in os.walk(BASE_DIR):
            root_path = Path(root)

            for f in files:
                if not f.lower().endswith((".jpg", ".png", ".tif")):
                    continue

                path = root_path / f
                label = infer_class_from_folder(path)

                if label is None:
                    continue

                self.samples.append({
                    "type": "image",
                    "path": path,
                    "label": label
                })

    # --------------------------------------------------------
    # DOCFIGURE
    # --------------------------------------------------------
    def collect_docfigure(self):
        print("Loading DocFigure...")

        ann_dir = BASE_DIR / "DocFigure_chart/DocFigure_annotation/annotation"
        img_dir = BASE_DIR / "DocFigure_chart/DocFigure_image"

        allowed = {"bar chart", "line chart", "pie chart", "scatter plot"}

        for split in ["train.txt", "test.txt"]:
            path = ann_dir / split
            if not path.exists():
                continue

            with open(path) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 2:
                        continue

                    fname = parts[0].strip()
                    label = parts[1].strip().lower()

                    if label not in allowed:
                        continue

                    img_path = img_dir / fname
                    if img_path.exists():
                        self.samples.append({
                            "type": "image",
                            "path": img_path,
                            "label": "chart"
                        })

    # --------------------------------------------------------
    # CHARTQA PARQUET
    # --------------------------------------------------------
    def collect_chartqa(self):
        print("Loading ChartQA...")

        chartqa_dir = BASE_DIR / "ChartQA_chart"

        for pq in chartqa_dir.glob("*.parquet"):
            df = pd.read_parquet(pq)

            for _, row in df.iterrows():
                img_bytes = row["image"]

                self.samples.append({
                    "type": "bytes",
                    "image": img_bytes,
                    "label": "chart"
                })

    # --------------------------------------------------------
    # PCAM H5
    # --------------------------------------------------------
    def collect_pcam(self):
        print("Loading PCam...")

        pcam_dir = BASE_DIR / "PatchCamelyon_PCam_histologie"

        for h5_file in pcam_dir.glob("*_x.h5"):
            f = h5py.File(h5_file, "r")
            data = f["x"]

            for i in range(len(data)):
                self.samples.append({
                    "type": "h5",
                    "h5": h5_file,
                    "index": i,
                    "label": "histologie"
                })

    # --------------------------------------------------------
    # GET ITEM
    # --------------------------------------------------------
    def __getitem__(self, idx):
        s = self.samples[idx]
        label_idx = CLASS_TO_IDX[s["label"]]

        if s["type"] == "image":
            img = Image.open(s["path"]).convert("RGB")

        elif s["type"] == "bytes":
            img = Image.open(io.BytesIO(s["image"])).convert("RGB")

        elif s["type"] == "h5":
            with h5py.File(s["h5"], "r") as f:
                arr = f["x"][s["index"]]
            img = Image.fromarray(arr)

        else:
            raise ValueError("Unknown type")

        img = transform(img)
        return img, label_idx

    def __len__(self):
        return len(self.samples)


# ============================================================
# MODEL
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# TRAIN
# ============================================================

def train():
    dataset = MultiDataset()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model = SimpleCNN(len(CLASSES)).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=dataset.class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "cnn3.pth")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train()