# -*- coding: utf-8 -*-

import os
import io
import random
from pathlib import Path
from collections import Counter, defaultdict


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
import torchvision.transforms as transforms

from PIL import Image
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm
# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path("/home/b/PycharmProjects/ba2roco/cnn/dataset3")

CLASSES = ["histologie", "haut", "chart", "endoskopie", "mikroskopie", "chirurgie"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

IMG_SIZE = 224
DEVICE = "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PCam downsampling (wichtig!) Sind eh alle fast gleich Histologiebilder, 330k zu viel, restliche haben 8-32k
PCAM_STRIDE = 12  # daher jedes 12. Bild, ^ Class Weights allein reichen nicht

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

batch_size = 32
num_epochs = 5
seed = 42

# ============================================================
# DATASET
# ============================================================

class MultiDataset(Dataset):
    def __init__(self):
        self.samples = []

        self.load_chartqa()
        self.load_colorectal()
        self.load_dermnet()
        self.load_docfigure()
        self.load_kvasir()
        self.load_livecell()
        self.load_pcam()
        self.load_peir()
        self.load_surgical()
        self.load_pathmnist()

        print("\nTotal samples:", len(self.samples))
        self.print_class_distribution()

    # --------------------------------------------------------
    def add_image(self, path, label, source):
        self.samples.append({
            "type": "image",
            "path": path,
            "label": label,
            "source": source
        })

    def add_bytes(self, img_bytes, label, source):
        self.samples.append({
            "type": "bytes",
            "image": img_bytes,
            "label": label,
            "source": source
        })

    def add_h5(self, h5_path, idx, label, source):
        self.samples.append({
            "type": "h5",
            "h5": h5_path,
            "index": idx,
            "label": label,
            "source": source
        })

    # ============================================================
    # DATASETS
    # ============================================================

    def load_chartqa(self):
        print("Loading ChartQA...")
        d = BASE_DIR / "ChartQA_chart"

        files = [
            "train-00000-of-00003.parquet",
            "train-00001-of-00003.parquet",
            "train-00002-of-00003.parquet",
            "train-00003-of-00003.parquet",
            "test-00000-of-00001.parquet"
        ]

        for f in files:
            p = d / f
            if not p.exists():
                continue

            df = pd.read_parquet(p)

            for _, row in df.iterrows():
                self.add_bytes(row["image"], "chart", "chartqa")

    def load_colorectal(self):
        print("Loading Colorectal...")
        base = BASE_DIR / "Colorectal_histologie"

        for sub in base.glob("*/*"):
            for img in sub.glob("*.tif"):
                self.add_image(img, "histologie", "colorectal")

    def load_dermnet(self):
        print("Loading DermNet...")
        base = BASE_DIR / "Dermnet_koerperaussenechtbild_haut"

        for img in base.rglob("*.jpg"):
            self.add_image(img, "haut", "dermnet")

    def load_docfigure(self):
        print("Loading DocFigure...")

        ann = BASE_DIR / "DocFigure_chart/DocFigure_annotation/annotation"
        img_dir = BASE_DIR / "DocFigure_chart/DocFigure_image"

        EXCLUDE = {"3d objects", "medical images", "mask", "natural images"}

        for split in ["train.txt", "test.txt"]:
            f = ann / split
            if not f.exists():
                continue

            with open(f) as file:
                for line in file:
                    parts = line.strip().split(",")
                    if len(parts) < 2:
                        continue

                    fname = parts[0].strip()
                    label = parts[1].strip().lower()

                    if label in EXCLUDE:
                        continue

                    path = img_dir / fname
                    if path.exists():
                        self.add_image(path, "chart", "docfigure")

    def load_kvasir(self):
        print("Loading Kvasir...")
        base = BASE_DIR / "kvasir_endoskopie"

        for img in base.rglob("*.jpg"):
            self.add_image(img, "endoskopie", "kvasir")

    def load_livecell(self):
        print("Loading LIVECell...")
        base = BASE_DIR / "LIVEcell_mikroskopie"

        for img in base.rglob("*.tif"):
            self.add_image(img, "mikroskopie", "livecell")

    def load_pcam(self):
        print("Loading PCam...")
        base = BASE_DIR / "PatchCamelyon_PCam_histologie"

        for h5_file in base.glob("*_x.h5"):
            with h5py.File(h5_file, "r") as f:
                data = f["x"]
                for i in range(0, len(data), PCAM_STRIDE):
                    self.add_h5(h5_file, i, "histologie", "pcam")

    def load_peir(self):
        print("Loading PEIR...")
        base = BASE_DIR / "peir_chirurgie"

        for img in base.glob("*.jpg"):
            self.add_image(img, "chirurgie", "peir")

    def load_surgical(self):
        print("Loading Surgical...")
        base = BASE_DIR / "Surgical_gastrectomy_miccai2022_chirurgie"

        for img in base.rglob("*.jpg"):
            self.add_image(img, "chirurgie", "surgical")

    def load_pathmnist(self):
        print("Loading PathMNIST (fixed)...")

        base = BASE_DIR / "PathMNIST_histologie" / "pathmnist_224"
        splits = ["train", "val", "test"]

        for split in splits:
            img_path = base / f"{split}_images.npy"

            if not img_path.exists():
                print(f"⚠️ fehlt: {img_path}")
                continue

            images = np.load(img_path, mmap_mode="r")

            print(f"{split}: {images.shape}")

            for i in range(len(images)):
                self.samples.append({
                    "type": "npy",
                    "array": images,
                    "index": i,
                    "label": "histologie",
                    "source": f"pathmnist_{split}"
                })
    # ============================================================
    # STATS
    # ============================================================

    def print_class_distribution(self):
        counts = Counter([s["label"] for s in self.samples])
        print("Class distribution:", counts)

        total = sum(counts.values())
        num_classes = len(CLASSES)

        weights = []
        for c in CLASSES:
            n = counts.get(c, 1)
            w = total / (num_classes * n)
            weights.append(w)

        weights = torch.tensor(weights)
        weights = torch.clamp(weights, max=10)

        self.class_weights = weights
        print("Class weights:", weights)

    # ============================================================
    # GET ITEM
    # ============================================================

    def __getitem__(self, idx):
        s = self.samples[idx]

        try:
            label = CLASS_TO_IDX[s["label"]]

            t = s["type"]

            if t == "image":
                img = Image.open(s["path"]).convert("RGB")

            elif t == "bytes":
                img = Image.open(io.BytesIO(s["image"])).convert("RGB")

            elif t == "h5":
                with h5py.File(s["h5"], "r") as f:
                    arr = f["x"][s["index"]]
                img = Image.fromarray(arr)

            elif t == "npy":
                arr = s["array"][s["index"]]
                img = Image.fromarray(arr)

            else:
                raise ValueError(f"Unknown type: {t}")

            img = transform(img)
            return img, label

        except Exception as e:
            print("\n⚠Fehler im Sample:")
            print(s)
            print("Error:", e)

            new_idx = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_idx)

    def __len__(self):
        return len(self.samples)


# ============================================================
# MODEL
# ============================================================

class ThirdCNN(nn.Module):
    def __init__(self):
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
            nn.Linear(256, len(CLASSES))
        )

    def forward(self, x):
        return self.net(x)
# ============================================================
# globales Splitting soll vermieden werden, nur Splitting fuer DS jeweils
# ============================================================
def create_dataset_splits(dataset, train_ratio=0.85, val_ratio=0.06):

    # gruppieren nach source
    source_to_indices = defaultdict(list)

    for idx, s in enumerate(dataset.samples):
        source_to_indices[s["source"]].append(idx)

    train_indices = []
    val_indices = []
    test_indices = []

    for source, indices in source_to_indices.items():
        indices = list(indices)
        random.shuffle(indices)

        n = len(indices)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_indices += indices[:n_train]
        val_indices += indices[n_train:n_train + n_val]
        test_indices += indices[n_train + n_val:]

        print(f"{source}: train={n_train}, val={n_val}, test={n - n_train - n_val}")

    return train_indices, val_indices, test_indices


def create_loaders(dataset, batch_size=32):

    train_idx, val_idx, test_idx = create_dataset_splits(dataset)

    train_set = Subset(dataset, train_idx)
    val_set   = Subset(dataset, val_idx)
    test_set  = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def evaluate_model(model, loader, criterion, device, class_names):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    num_classes = len(class_names)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        eval_bar = tqdm(loader, desc="Evaluation", leave=False)

        for inputs, labels in eval_bar:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            for t, p in zip(labels.cpu(), preds.cpu()):
                confusion[t.long(), p.long()] += 1

            acc = correct / total if total > 0 else 0.0
            eval_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc * 100:.2f}%")

    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    # ---- Metrics ----
    per_class_metrics = []
    recalls = []

    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_f1_sum = 0.0

    weighted_precision_sum = 0.0
    weighted_recall_sum = 0.0
    weighted_f1_sum = 0.0

    for c in range(num_classes):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        support = confusion[c, :].sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class_metrics.append({
            "class_name": class_names[c],
            "support": support,
            "accuracy": recall,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

        recalls.append(recall)

        macro_precision_sum += precision
        macro_recall_sum += recall
        macro_f1_sum += f1

        weighted_precision_sum += precision * support
        weighted_recall_sum += recall * support
        weighted_f1_sum += f1 * support

    macro_precision = macro_precision_sum / num_classes
    macro_recall = macro_recall_sum / num_classes
    macro_f1 = macro_f1_sum / num_classes
    balanced_accuracy = sum(recalls) / num_classes

    weighted_precision = weighted_precision_sum / total
    weighted_recall = weighted_recall_sum / total
    weighted_f1 = weighted_f1_sum / total

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": confusion,
    }

def print_metrics(title, metrics):
    print(f"\n================ {title} ================")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
    print(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")

    print("\nPro Klasse:")
    for m in metrics["per_class_metrics"]:
        print(
            f"{m['class_name']}: "
            f"n={m['support']}, "
            f"acc={m['accuracy'] * 100:.2f}%, "
            f"precision={m['precision']:.4f}, "
            f"recall={m['recall']:.4f}, "
            f"f1={m['f1']:.4f}"
        )

    print("\nConfusion Matrix:")
    print(metrics["confusion_matrix"])
# ============================================================
# TRAIN
# ============================================================
def train():

    dataset = MultiDataset()

    train_loader, val_loader, test_loader = create_loaders(dataset)

    model = ThirdCNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=dataset.class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        DEVICE,
        epochs=5
    )
    # Validation
    val_metrics = evaluate_model(model, val_loader, criterion, DEVICE, CLASSES)
    print_metrics("Validation", val_metrics)

    test_metrics = evaluate_model(model, test_loader, criterion, DEVICE, CLASSES)
    print_metrics("Test", test_metrics)

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs):

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for inputs, labels in train_bar:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            acc = correct / total if total > 0 else 0.0

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{acc*100:.2f}%"
            )

        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"Train Acc: {acc*100:.2f}%")

    torch.save(model.state_dict(), "cnn_multiclass.pth")

if __name__ == "__main__":
    train()