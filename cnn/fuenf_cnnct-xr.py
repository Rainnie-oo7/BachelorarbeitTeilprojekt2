# -*- coding: utf-8 -*-

# Gemeinsame Medical Multiclass CNN Pipeline
import os
import os.path as osp
import re
import pickle
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
from tqdm import tqdm

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import nibabel as nib
except ImportError:
    nib = None

# ============================================================
# ROOTS
# ============================================================

DATASET_ROOTS = [
    # 1 CT
    Path("/media/b/Volume/TCIA_Lung_Phantom"),
    Path("/media/b/Volume/TCIA_TCGA-ESCA"),
    Path("/media/b/Volume/TCIA_TCGA-STAD"),
    Path("/media/b/Volume/QIN-Lung"),
    Path("/media/b/Volume/StageII-Colorectal-CT"),

    # 2 XRAY  /  - Angiographie
    Path("/home/b/PycharmProjects/ba2roco/cnn/dataset2"),
]
print("\n============ ROOT CHECK ==========")
for p in DATASET_ROOTS:
    print(p, "EXISTS:", p.exists())
#strikt Reihenfolge!!!
CLASSES = [
    "ct",
    "xray",
]

CLASS_TO_IDX = {
    c: i
    for i, c in enumerate(CLASSES)
}

IDX_TO_CLASS = {
    i: c
    for c, i in CLASS_TO_IDX.items()
}

# ============================================================
# Config
# ============================================================

MAX_PER_CLASS = 60800
SEED = 42
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0008  # noch mehr lernen moegl. als 10^-3

VALID_EXTENSIONS = {
    ".pt",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".dcm",
    ".nii",
    ".pck",
    ".tif",
    ".tiff",
}

# IMAGE HELPERS

def normalize_to_uint8(arr):

    arr = np.asarray(arr)

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    mn = float(arr.min())
    mx = float(arr.max())

    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = (arr - mn) / (mx - mn)

    arr = (arr * 255).clip(0, 255).astype(np.uint8)

    return arr


def ensure_image_tensor(tensor):

    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)

    tensor = tensor.detach().clone().float()

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    elif tensor.ndim == 3:

        if tensor.shape[-1] in (1, 3, 4):
            if tensor.shape[0] not in (1, 3, 4):
                tensor = tensor.permute(2, 0, 1)

    else:
        raise ValueError(f"Bad tensor shape: {tuple(tensor.shape)}")

    c, h, w = tensor.shape

    if c == 1:
        tensor = tensor.repeat(3, 1, 1)

    elif c == 4:
        tensor = tensor[:3]

    elif c != 3:
        raise ValueError(f"Unexpected channels: {c}")

    if tensor.max() > 1.5:
        tensor = tensor / 255.0

    return tensor


def pil_to_tensor_rgb(img):

    img = img.convert("RGB")

    arr = np.array(img)

    tensor = (
        torch.from_numpy(arr)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    return tensor


def load_normal_image(path):

    img = Image.open(path).convert("RGB")

    return pil_to_tensor_rgb(img)


def load_dicom(path):

    ds = pydicom.dcmread(str(path))

    arr = ds.pixel_array

    arr = np.asarray(arr)

    # ========================================================
    # 3D DICOM
    # ========================================================

    if arr.ndim == 3:

        # häufig:
        # (slices, H, W)
        # oder
        # (H, W, slices)

        # kleines Axis = Slice-Axis heuristisch
        slice_axis = np.argmin(arr.shape)

        center_idx = arr.shape[slice_axis] // 2

        arr = np.take(
            arr,
            center_idx,
            axis=slice_axis
        )

    # ========================================================
    # 4D DICOM
    # ========================================================

    elif arr.ndim == 4:

        # oft:
        # (time, slices, H, W)

        arr = arr[0]

        slice_axis = np.argmin(arr.shape)

        center_idx = arr.shape[slice_axis] // 2

        arr = np.take(
            arr,
            center_idx,
            axis=slice_axis
        )

    # ========================================================
    # Normalize
    # ========================================================

    arr = normalize_to_uint8(arr)

    img = Image.fromarray(arr).convert("RGB")

    return pil_to_tensor_rgb(img)


def load_pt(path):

    obj = torch.load(path, map_location="cpu")

    return ensure_image_tensor(obj)


def load_pck(path):

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, torch.Tensor):
        return ensure_image_tensor(obj)

    if isinstance(obj, np.ndarray):

        arr = obj

        if arr.ndim == 3:
            arr = arr[:, :, arr.shape[2] // 2]

        return ensure_image_tensor(torch.as_tensor(arr))

    if isinstance(obj, dict):

        for _, v in obj.items():

            if isinstance(v, np.ndarray):

                arr = v

                if arr.ndim == 3:
                    arr = arr[:, :, arr.shape[2] // 2]

                return ensure_image_tensor(torch.as_tensor(arr))

            if isinstance(v, torch.Tensor):

                tensor = v

                if tensor.ndim == 3:
                    if tensor.shape[0] not in (1, 3, 4):
                        tensor = tensor[:, :, tensor.shape[2] // 2]

                return ensure_image_tensor(tensor)

    raise ValueError(f"Cannot read pck: {path}")


def slice_indices_25_percent(depth):

    idxs = [
        int(depth * 0.25),
        int(depth * 0.50),
        int(depth * 0.75),
        int(depth * 1.00) - 1,
    ]

    idxs = [
        max(0, min(depth - 1, i))
        for i in idxs
    ]

    return sorted(set(idxs))


def load_nii_slice(path, slice_idx):

    img = nib.load(str(path))

    arr = img.get_fdata()

    if arr.ndim == 3:
        sl = arr[:, :, slice_idx]

    elif arr.ndim == 4:
        sl = arr[:, :, slice_idx, 0]

    else:
        raise ValueError(f"Unexpected NII shape: {arr.shape}")

    sl = normalize_to_uint8(sl)

    return pil_to_tensor_rgb(Image.fromarray(sl))

# ============================================================
# Dataset
# ============================================================

class UnifiedMedicalDataset(Dataset):

    def __init__(self, root_dirs, transform=None):

        self.root_dirs = root_dirs
        self.transform = transform

        self.classes = CLASSES
        self.class_to_idx = CLASS_TO_IDX

        self.samples = []

        self.collect_files()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        rec = self.samples[idx]

        path = rec["path"]
        label = rec["label_idx"]

        suffix = path.suffix.lower()

        if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            image = load_normal_image(path)

        elif suffix == ".dcm":
            image = load_dicom(path)

        elif suffix == ".pt":
            image = load_pt(path)

        elif suffix == ".pck":
            image = load_pck(path)

        elif suffix == ".nii":
            image = load_nii_slice(path, rec["slice_idx"])

        else:
            raise ValueError(f"Unsupported format: {path}")

        image = ensure_image_tensor(image)

        if self.transform:
            image = self.transform(image)

        return image, label

    # ========================================================
    # Hilffunktien
    # ========================================================

    def is_valid_file(self, path):

        path = Path(path)

        if path.name.startswith("."):
            return False

        return path.suffix.lower() in VALID_EXTENSIONS

    def is_mask(self, path):

        name = path.name.lower()

        return (
            "_mask" in name
            or "segmentation" in name
            or "_gt" in name
            or "_seg" in name
        )

    # ========================================================
    # Class inference
    # ========================================================

    def infer_class(self, path):

        joined = str(path).lower()

        # ====================================================
        # CT
        # ====================================================

        if any(x in joined for x in [
            "tcia_lung_phantom",
            "tcia_tcga-esca",
            "tcia_tcga-stad",
            "qin-lung",
            "stageii-colorectal-ct",
        ]):
            return "ct"

        # ====================================================
        # XRAY
        # ====================================================

        # alles aus dataset2 als XRAY
        if "dataset2" in joined:
            return "xray"

        return None

    # ========================================================
    # File Filters
    # ========================================================

    def should_take_file(self, path, label_name):

        if not self.is_valid_file(path):
            return False

        if self.is_mask(path):
            return False

        return True

    # ========================================================
    # Add sample
    # ========================================================

    def add_sample(self, path, label_name, slice_idx=None):

        self.samples.append({
            "path": path,
            "label_name": label_name,
            "label_idx": self.class_to_idx[label_name],
            "slice_idx": slice_idx,
        })

    # ========================================================
    # Collect Files
    # ========================================================

    def collect_files(self):

        rng = random.Random(SEED)

        grouped = defaultdict(list)

        for dataset_root in self.root_dirs:

            print(f"\nScanning: {dataset_root}")

            for root, _, files in os.walk(dataset_root):

                root_path = Path(root)

                for fname in files:

                    path = root_path / fname

                    label_name = self.infer_class(path)

                    if label_name is None:
                        continue

                    if not self.should_take_file(path, label_name):
                        continue

                    grouped[label_name].append(path)

        self.samples = []

        for class_name in CLASSES:

            files = grouped[class_name]

            rng.shuffle(files)

            selected = []

            for path in files:

                suffix = path.suffix.lower()

                if suffix == ".nii":

                    try:
                        nii = nib.load(str(path))
                        depth = nii.shape[2]
                    except Exception:
                        continue

                    for z in slice_indices_25_percent(depth):

                        selected.append((path, z))

                        if len(selected) >= MAX_PER_CLASS:
                            break

                else:
                    selected.append((path, None))

                if len(selected) >= MAX_PER_CLASS:
                    break

            for path, z in selected:
                self.add_sample(path, class_name, z)

            print(f"{class_name}: {len(selected)}")

# ============================================================
# Model
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

# ============================================================
# Class weights
# ============================================================


def compute_class_weights(dataset):

    counts = defaultdict(int)

    for rec in dataset.samples:
        counts[rec["label_name"]] += 1

    arr = torch.tensor([
        counts[c]
        for c in CLASSES
    ], dtype=torch.float32)

    total = arr.sum()

    weights = total / (len(CLASSES) * arr)

    weights = torch.sqrt(weights)

    weights = torch.clamp(weights, max=50.0)

    return weights

# ============================================================
# Evaluation
# ============================================================


def evaluate(model, loader, criterion, device):

    model.eval()

    total = 0
    correct = 0

    total_loss = 0.0

    with torch.no_grad():

        for inputs, labels in loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            preds = outputs.argmax(dim=1)

            total += labels.size(0)

            correct += (preds == labels).sum().item()

    acc = correct / total if total > 0 else 0.0

    return {
        "loss": total_loss / len(loader),
        "accuracy": acc,
    }

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )
    ])

    dataset = UnifiedMedicalDataset(
        root_dirs=DATASET_ROOTS,
        transform=transform
    )

    print(f"\nTotal samples: {len(dataset)}")

    if len(dataset) == 0:
        raise RuntimeError("No samples found")

    class_weights = compute_class_weights(dataset)

    print("\nClass weights:")

    for cls, w in zip(CLASSES, class_weights.tolist()):
        print(cls, w)

    total_size = len(dataset)

    train_size = int(0.79 * total_size)
    val_size = int(0.105 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    model = SimpleCNN(
        num_classes=len(CLASSES)
    ).to(device)

    # ============================================================
    # LOAD EXISTING MODEL vvvvvvvvvvvvAuskommentieren fuer Training
    # ============================================================

    # save_path = "cnnct-xr.pth"
    #
    # print(f"\nLoading model: {save_path}")
    #
    # model.load_state_dict(
    #     torch.load(
    #         save_path,
    #         map_location=device
    #     )
    # )
    #
    # model.eval()
    #
    # print("Model loaded.")

    # ============================================================
    # LOAD EXISTING MODEL ^^^^^^^^^^^^^^Auskommentieren fuer Training
    # ============================================================
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(NUM_EPOCHS):

        model.train()

        running_loss = 0.0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}"
        )

        for inputs, labels in train_bar:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            train_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        print(f"\nEpoch {epoch + 1}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy'] * 100:.2f}%")

    save_path = "cnnges.pth"

    torch.save(model.state_dict(), save_path)

    print(f"\nSaved model: {save_path}")

    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print("\n===== TEST =====")
    print(test_metrics)

# ============================================================
# PER-CLASS TEST EVALUATION
# ============================================================
    print("\n")
    print("=" * 70)
    print("PER-CLASS TEST EVALUATION")
    print("=" * 70)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for inputs, labels in tqdm(
                test_loader,
                desc="Per-Class Evaluation"):

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            preds = outputs.argmax(dim=1)

            all_preds.extend(
                preds.cpu().numpy().tolist()
            )

            all_labels.extend(
                labels.cpu().numpy().tolist()
            )

    # ============================================================
    # GLOBAL
    # ============================================================

    overall_acc = accuracy_score(
        all_labels,
        all_preds
    )

    print(f"\nOverall Accuracy: {overall_acc:.4f}")

    # ============================================================
    # PER CLASS
    # ============================================================

    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels,
        all_preds,
        labels=list(range(len(CLASSES))),
        zero_division=0
    )

    print("\n")
    print("=" * 70)
    print("PER CLASS METRICS")
    print("=" * 70)

    for i, cls in enumerate(CLASSES):

        print("\n" + "-" * 60)

        print(f"Class: {cls}")

        print(f"Precision : {precision[i]:.4f}")
        print(f"Recall    : {recall[i]:.4f}")
        print(f"F1-Score  : {f1[i]:.4f}")
        print(f"Support   : {support[i]}")

    # ============================================================
    # CLASSIFICATION REPORT
    # ============================================================

    print("\n")
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASSES,
        digits=4,
        zero_division=0
    )

    print(report)

    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    print("\n")
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    cm = confusion_matrix(
        all_labels,
        all_preds,
        labels=list(range(len(CLASSES)))
    )

    print(cm)

    # ============================================================
    # NORMALIZED CONFUSION MATRIX
    # ============================================================

    print("\n")
    print("=" * 70)
    print("NORMALIZED CONFUSION MATRIX")
    print("=" * 70)

    cm_norm = cm.astype(np.float32)

    row_sums = cm_norm.sum(axis=1, keepdims=True)

    cm_norm = np.divide(
        cm_norm,
        row_sums,
        where=row_sums != 0
    )

    np.set_printoptions(
        precision=3,
        suppress=True
    )

    print(cm_norm)