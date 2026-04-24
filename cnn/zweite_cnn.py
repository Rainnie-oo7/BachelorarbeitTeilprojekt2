import os
import os.path as osp
import re
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm

# import warnings   #pydicom ISOIR100 Warning wegmachen
# warnings.filterwarnings(
#     "ignore",
#     message="Unknown encoding.*",
#     module="pydicom"
# )

try:
    import pydicom
except ImportError:
    pydicom = None

try:
    import nibabel as nib
except ImportError:
    nib = None


# ============================================================
# Einstellungen
# ============================================================
ROOT_DIR = osp.normpath(osp.join(osp.dirname(__file__), "dataset"))

VALID_EXTENSIONS = {
    ".pt",
    ".jpg", ".jpeg", ".png", ".bmp",
    ".dcm",
    ".nii",
    ".pck",
}

CLASSES = [
    "xray",
    "xray_fluoroskopie_angiographie",
    "mrt_hirn",
    "mrt_body",
]

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ============================================================
# Bild-Hilfsfunktionen
# ============================================================

def normalize_to_uint8(arr):
    arr = np.asarray(arr)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if arr.size == 0:
        raise ValueError("Leeres Array.")

    mn = float(arr.min())
    mx = float(arr.max())

    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = (arr - mn) / (mx - mn)
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    return arr


def pil_to_tensor_rgb(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)

    img = img.convert("RGB")
    arr = np.array(img)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
    return tensor


def ensure_image_tensor(tensor):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)

    tensor = tensor.detach().clone().float()

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    elif tensor.ndim == 3:
        if tensor.shape[-1] in (1, 3, 4) and tensor.shape[0] not in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1).contiguous()

    else:
        raise ValueError(f"Nicht unterstuetzte Tensor-Form: {tuple(tensor.shape)}")

    c, h, w = tensor.shape

    if c == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif c == 4:
        tensor = tensor[:3]
    elif c != 3:
        raise ValueError(f"Unerwartete Kanalzahl: {c}, shape={tuple(tensor.shape)}")

    return tensor


def load_pt_as_tensor(path):
    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        return ensure_image_tensor(obj)

    return ensure_image_tensor(torch.as_tensor(obj))


def load_image_as_tensor(path):
    img = Image.open(path).convert("RGB")
    return pil_to_tensor_rgb(img)


def load_dicom_as_tensor(path):
    if pydicom is None:
        raise ImportError("pydicom fehlt. Installiere mit: pip install pydicom")

    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    arr = normalize_to_uint8(arr)
    img = Image.fromarray(arr).convert("RGB")
    return pil_to_tensor_rgb(img)


def slice_indices_25_percent(depth):
    """
    4 Schichten bei ca. 25%, 50%, 75%, 100%.
    Letzter Index wird auf depth-1 begrenzt.
    """
    if depth <= 0:
        return []

    idxs = [
        int(depth * 0.25),
        int(depth * 0.50),
        int(depth * 0.75),
        int(depth * 1.00) - 1,
    ]

    idxs = [max(0, min(depth - 1, i)) for i in idxs]
    return sorted(set(idxs))


def load_nii_slice_as_tensor(path, slice_idx):
    if nib is None:
        raise ImportError("nibabel fehlt. Installiere mit: pip install nibabel")

    img = nib.load(str(path))
    arr = img.get_fdata()

    if arr.ndim == 3:
        z = max(0, min(arr.shape[2] - 1, slice_idx))
        sl = arr[:, :, z]

    elif arr.ndim == 4:
        # 4D wird hier nur notfalls behandelt:
        # erstes Zeitfenster, mittlere/angegebene Z-Slice.
        z = max(0, min(arr.shape[2] - 1, slice_idx))
        sl = arr[:, :, z, 0]

    else:
        raise ValueError(f"Unerwartete NIfTI-Form {arr.shape} fuer {path}")

    sl = normalize_to_uint8(sl)
    return pil_to_tensor_rgb(Image.fromarray(sl))


def inspect_pck(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    print(f"\n[PCK INSPECT] {path}")
    print("type:", type(obj))

    if isinstance(obj, np.ndarray):
        print("shape:", obj.shape, "dtype:", obj.dtype)

    elif isinstance(obj, torch.Tensor):
        print("shape:", tuple(obj.shape), "dtype:", obj.dtype)

    elif isinstance(obj, dict):
        print("keys:", list(obj.keys())[:30])
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: ndarray {v.shape} {v.dtype}")
            elif isinstance(v, torch.Tensor):
                print(f"  {k}: tensor {tuple(v.shape)} {v.dtype}")
            else:
                print(f"  {k}: {type(v)}")

    return obj


def load_pck_as_tensor(path):
    """
    Robuster Minimal-Loader:
    - ndarray direkt
    - Tensor direkt
    - dict: nimmt erstes Array/Tensor mit mindestens 2 Dimensionen
    """
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, torch.Tensor):
        return ensure_image_tensor(obj)

    if isinstance(obj, np.ndarray):
        arr = obj
        if arr.ndim == 3:
            # falls Volumen: mittlere Slice
            arr = arr[:, :, arr.shape[2] // 2]
        return ensure_image_tensor(torch.as_tensor(arr))

    if isinstance(obj, dict):
        for _, v in obj.items():
            if isinstance(v, torch.Tensor) and v.ndim >= 2:
                if v.ndim == 3 and v.shape[0] not in (1, 3, 4):
                    v = v[:, :, v.shape[2] // 2]
                return ensure_image_tensor(v)

            if isinstance(v, np.ndarray) and v.ndim >= 2:
                arr = v
                if arr.ndim == 3:
                    arr = arr[:, :, arr.shape[2] // 2]
                return ensure_image_tensor(torch.as_tensor(arr))

    raise ValueError(f"Konnte .pck nicht als Bild laden: {path}")

#fuer MURA v1.1 Dataset, es gab unlesbare ._image.png
def is_hidden_or_mac_resource_file(path):
    return Path(path).name.startswith(".")
# ============================================================
# Dataset
# ============================================================

class FolderLabelMedicalDataset(Dataset):
    def __init__(self, root_dir, transform=None, verbose=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = CLASSES
        self.class_to_idx = CLASS_TO_IDX
        self.samples = []

        self.skipped = {
            "unsupported": 0,
            "mask_or_gt": 0,
            "unknown_label": 0,
            "ignored_dicom_spine": 0,
            "ignored_acdc_4d": 0,
        }

        self.collect_files()

        if verbose:
            self.print_summary()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        path = rec["path"]
        label = rec["label_idx"]

        suffix = path.suffix.lower()

        if suffix == ".pt":
            image = load_pt_as_tensor(path)

        elif suffix in {".jpg", ".jpeg", ".png", ".bmp"}:
            image = load_image_as_tensor(path)

        elif suffix == ".dcm":
            image = load_dicom_as_tensor(path)

        elif suffix == ".nii":
            image = load_nii_slice_as_tensor(path, rec["slice_idx"])

        elif suffix == ".pck":
            image = load_pck_as_tensor(path)

        else:
            raise ValueError(f"Nicht unterstuetztes Format: {path}")

        if self.transform:
            image = self.transform(image)

        return image, label

    def is_valid_file(self, path):
        path = Path(path)

        if is_hidden_or_mac_resource_file(path):
            return False

        return path.suffix.lower() in VALID_EXTENSIONS

    def is_mask_or_gt(self, path):
        name = path.name.lower()
        return (
            "_mask" in name
            or name.endswith("_gt.nii")
            or "_gt." in name
            or "segmentation" in str(path).lower()
        )

    def infer_label_from_path(self, path):
        parts = [p.lower() for p in path.parts]
        joined = str(path).lower()

        if "xray_fluoroskopie_angiographie" in parts:
            return "xray_fluoroskopie_angiographie"

        if "xray" in parts:
            return "xray"

        if "us" in parts:
            if "_mask" in path.name.lower():
                return None
            return "us"

        if "mrt_hirn" in parts:
            return "mrt_hirn"

        if "mrt_body" in parts:
            return "mrt_body"

        if "ct_kombimodalitaet_spect+ct_pet+ct" in parts:
            return "ct_kombimodalitaet_spect+ct_pet+ct"

        if "ct" in parts:
            if "spect" in joined and "ct" in joined:
                return "ct_kombimodalitaet_spect+ct_pet+ct"
            if "pet" in joined and "ct" in joined:
                return "ct_kombimodalitaet_spect+ct_pet+ct"
            return "ct"

        return None

    def should_take_file(self, path, label_name):
        suffix = path.suffix.lower()
        lower_path = str(path).lower()

        if not self.is_valid_file(path):
            self.skipped["unsupported"] += 1
            return False

        if self.is_mask_or_gt(path):
            self.skipped["mask_or_gt"] += 1
            return False

        # Spine: nur JPG, nicht die gleichnamigen DICOMs
        if label_name == "mrt_body" and "spine mri dataset" in lower_path:
            if suffix == ".dcm":
                self.skipped["ignored_dicom_spine"] += 1
                return False
            return suffix in {".jpg", ".jpeg", ".png"}

        # ACDC: nur frameXX.nii, keine gt, keine 4d
        if label_name == "mrt_body" and "automatedcardiacdiagnosischallenge_miccai17" in lower_path:
            name = path.name.lower()

            if name.endswith("_4d.nii"):
                self.skipped["ignored_acdc_4d"] += 1
                return False

            if suffix == ".nii" and "_frame" in name and not name.endswith("_gt.nii"):
                return True

            return False

        # Breast: nur DICOMs aus interessanten Serien
        if label_name == "mrt_body" and "breast-mri-nact-pilot" in lower_path:
            series_name = path.parent.name.lower()

            interesting = (
                "2dfse" in series_name
                or "ir3dfgre" in series_name
                or re.search(r"(^|[^a-z0-9])t1([^a-z0-9]|$)", series_name)
                or "locator" in series_name
            )

            return suffix == ".dcm" and interesting

        # Prostata: nur T1/T2-Serien, alles als mrt_body
        if label_name == "mrt_body" and "prostate_mri" in lower_path:
            series_names = [p.lower() for p in path.parts]
            has_t1_t2 = any(
                re.search(r"(^|[^a-z0-9])t1([^a-z0-9]|$)", p)
                or re.search(r"(^|[^a-z0-9])t2([^a-z0-9]|$)", p)
                for p in series_names
            )
            return suffix in {".dcm", ".jpg", ".jpeg", ".png", ".pt"} and has_t1_t2

        # Knee: .pck erstmal zulassen
        if label_name == "mrt_body" and "kneemridataset" in lower_path:
            return suffix == ".pck"

        # MRT Hirn: .nii wie gewünscht zulassen, außerdem pt/Bilder/DICOM
        if label_name == "mrt_hirn":
            return suffix in {".nii", ".pt", ".jpg", ".jpeg", ".png", ".bmp", ".dcm"}

        # Xray/US/CT: normale Bild-/Tensordateien
        return suffix in {".pt", ".jpg", ".jpeg", ".png", ".bmp", ".dcm"}

    def add_sample(self, path, label_name, slice_idx=None):
        self.samples.append({
            "path": path,
            "label_name": label_name,
            "label_idx": self.class_to_idx[label_name],
            "slice_idx": slice_idx,
        })

    def collect_files(self):
        self.samples = []

        for root, _, files in os.walk(self.root_dir):
            root_path = Path(root)

            for fname in files:
                path = root_path / fname

                # Dateien wie ._image2.png, .DS_Store, .hidden.png überspringen
                if path.name.startswith("."):
                    self.skipped["unsupported"] += 1
                    continue

                label_name = self.infer_label_from_path(path)

                if label_name is None:
                    self.skipped["unknown_label"] += 1
                    continue

                if not self.should_take_file(path, label_name):
                    continue

                suffix = path.suffix.lower()

                if suffix == ".nii":
                    if nib is None:
                        raise ImportError("nibabel fehlt. Installiere mit: pip install nibabel")

                    nii = nib.load(str(path))
                    arr_shape = nii.shape

                    if len(arr_shape) < 3:
                        continue

                    depth = arr_shape[2]
                    for z in slice_indices_25_percent(depth):
                        self.add_sample(path, label_name, slice_idx=z)

                else:
                    self.add_sample(path, label_name)

    def print_summary(self):
        print("\n================ DATASET CHECK ================")
        print(f"Root: {self.root_dir}")
        print(f"Anzahl Samples: {len(self.samples)}")
        print(f"Klassen: {self.classes}")
        print(f"class_to_idx: {self.class_to_idx}")

        counts = {c: 0 for c in self.classes}
        source_counts = {}

        for rec in self.samples:
            counts[rec["label_name"]] += 1

            p = str(rec["path"]).lower()
            source = "unknown"

            for key in [
                "automatedcardiacdiagnosischallenge_miccai17",
                "breast-mri-nact-pilot",
                "kneemridataset",
                "spine mri dataset",
                "prostate_mri",
            ]:
                if key in p:
                    source = key
                    break

            source_counts[source] = source_counts.get(source, 0) + 1

        print("\nAnzahl pro Klasse:")
        for k, v in counts.items():
            print(f"{k}: {v}")

        print("\nAnzahl pro Quelle:")
        for k, v in sorted(source_counts.items()):
            print(f"{k}: {v}")

        print("\nUebersprungene Dateien:")
        for k, v in self.skipped.items():
            print(f"{k}: {v}")

        print("\nErste 10 Samples:")
        for rec in self.samples[:10]:
            print(rec)


# ============================================================
# SimpleCNN bleibt gleich
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
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ============================================================
# Evaluation
# ============================================================

def count_subset_classes(subset, class_names):
    counts = {name: 0 for name in class_names}
    for idx in subset.indices:
        rec = subset.dataset.samples[idx]
        counts[rec["label_name"]] += 1
    return counts


def print_split_counts(name, counts):
    print(f"\n{name}-Verteilung:")
    for cls_name, count in counts.items():
        print(f"{cls_name}: {count}")


def compute_class_weights_from_counts(class_counts_dict, class_names, use_sqrt=True, max_weight=50.0):
    counts = torch.tensor([class_counts_dict[name] for name in class_names], dtype=torch.float32)
    counts[counts == 0] = 1.0

    total_samples = counts.sum()
    num_classes = len(class_names)

    weights = total_samples / (num_classes * counts)

    if use_sqrt:
        weights = torch.sqrt(weights)

    if max_weight is not None:
        weights = torch.clamp(weights, max=max_weight)

    return weights


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

            for t, p in zip(labels.cpu().tolist(), preds.cpu().tolist()):
                confusion[t, p] += 1

            acc = correct / total if total > 0 else 0.0
            eval_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc * 100:.2f}%")

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

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

    weighted_precision = weighted_precision_sum / total if total > 0 else 0.0
    weighted_recall = weighted_recall_sum / total if total > 0 else 0.0
    weighted_f1 = weighted_f1_sum / total if total > 0 else 0.0

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
# Hauptteil
# ============================================================

if __name__ == "__main__":
    batch_size = 32
    num_epochs = 5
    seed = 42

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = FolderLabelMedicalDataset(
        root_dir=ROOT_DIR,
        transform=transform,
        verbose=True,
    )

    if len(dataset) == 0:
        raise RuntimeError("Keine gueltigen Dateien gefunden.")

    sample_img, sample_label = dataset[0]
    print("\nSanity Check erstes Sample:")
    print(
        f"shape={sample_img.shape}, "
        f"dtype={sample_img.dtype}, "
        f"min={sample_img.min().item():.4f}, "
        f"max={sample_img.max().item():.4f}, "
        f"label={sample_label}, "
        f"class={dataset.classes[sample_label]}"
    )

    class_counts = {cls: 0 for cls in dataset.classes}
    for rec in dataset.samples:
        class_counts[rec["label_name"]] += 1

    class_weights = compute_class_weights_from_counts(
        class_counts_dict=class_counts,
        class_names=dataset.classes,
        use_sqrt=True,
        max_weight=50.0,
    )

    print("\nClass Weights:")
    for cls_name, weight in zip(dataset.classes, class_weights.tolist()):
        print(f"{cls_name}: {weight:.4f}")

    total_size = len(dataset)
    train_size = int(0.79 * total_size)
    val_size = int(0.105 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    print_split_counts("Train", count_subset_classes(train_dataset, dataset.classes))
    print_split_counts("Val", count_subset_classes(val_dataset, dataset.classes))
    print_split_counts("Test", count_subset_classes(test_dataset, dataset.classes))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    model = SimpleCNN(num_classes=len(dataset.classes)).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

        for inputs, labels in train_bar:
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        val_metrics = evaluate_model(model, val_loader, criterion, device, dataset.classes)

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val Accuracy: {val_metrics['accuracy'] * 100:.2f}%")
        print(f"Val Balanced Accuracy: {val_metrics['balanced_accuracy'] * 100:.2f}%")
        print(f"Val Macro F1: {val_metrics['macro_f1']:.4f}")
        print(f"Val Weighted F1: {val_metrics['weighted_f1']:.4f}")

    save_path = "convu_folderlabel_mrt_body_mrt_hirn.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModell gespeichert als {save_path}")

    test_metrics = evaluate_model(model, test_loader, criterion, device, dataset.classes)
    print_metrics("TESTERGEBNISSE", test_metrics)