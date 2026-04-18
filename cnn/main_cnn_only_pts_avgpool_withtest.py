import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm

VALID_EXTENSIONS = {'.pt'}


# ============================================================
# Hilfsfunktionen nur fuer Torch-Tensoren
# ============================================================

def load_tensor_file(path):
    """
    Laedt eine .pt-Datei und gibt einen Torch-Tensor zurueck.
    Erwartet Bilddaten als Tensor oder etwas, das in Tensor wandelbar ist.
    """
    obj = torch.load(path, map_location='cpu')

    if isinstance(obj, torch.Tensor):
        tensor = obj
    else:
        tensor = torch.as_tensor(obj)

    return tensor


def ensure_image_tensor(tensor):
    """
    Vereinheitlicht Tensoren auf Form C x H x W und 3 Kanaele.

    Erlaubte Eingaben:
    - H x W
    - C x H x W
    - H x W x C

    Ausgabe:
    - 3 x H x W
    - dtype float32
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)

    tensor = tensor.detach().clone()
    tensor = tensor.float()

    if tensor.ndim == 0:
        raise ValueError(f"Tensor ist skalar und kein Bild: shape={tuple(tensor.shape)}")

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    elif tensor.ndim == 3:
        if tensor.shape[-1] in (1, 3, 4) and tensor.shape[0] not in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1).contiguous()

    else:
        raise ValueError(f"Nicht unterstuetzte Tensor-Form fuer Bilddaten: {tuple(tensor.shape)}")

    if tensor.ndim != 3:
        raise ValueError(f"Tensor konnte nicht zu CxHxW vereinheitlicht werden: {tuple(tensor.shape)}")

    c, h, w = tensor.shape

    if c == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif c == 4:
        tensor = tensor[:3, :, :]
    elif c != 3:
        raise ValueError(f"Unerwartete Kanalzahl {c} in Tensor mit shape={tuple(tensor.shape)}")

    return tensor


# ============================================================
# Dataset-Klasse
# ============================================================

class NestedMedicalFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = [
            'xray',
            'xray_fluoroskopie_angiographie',
            'us',
            'mrt_hirn_flair',
            'mrt_hirn_t1',
            'mrt_hirn_t2',
            'mrt_hirn_t1_c',
            'mrt_prostata_t1',
            'mrt_prostata_t2',
            'ct',
            'ct_kombimodalitaet_spect+ct_pet+ct'
        ]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = []

        self.collect_files()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]

        image = load_tensor_file(img_path)
        image = ensure_image_tensor(image)

        if self.transform:
            image = self.transform(image)

        return image, label

    def is_valid_file(self, filename):
        ext = os.path.splitext(filename)[1].lower()
        return ext in VALID_EXTENSIONS

    def is_busi_mask(self, path):
        filename = os.path.basename(path).lower()
        return '_mask' in filename

    def collect_mrt_prostata_files(self):
        """
        Erwartete Struktur:
        dataset/mrt_prostata/PROSTATE_MRI/PROSTATE-MRI/...
        Nur Serienordner mit T1 oder T2 werden berücksichtigt.
        """
        base_dir = os.path.join(
            self.root_dir,
            'mrt_prostata',
            'PROSTATE_MRI',
            'PROSTATE-MRI'
        )

        if not os.path.isdir(base_dir):
            print(f"[WARNUNG] Prostata-Basisordner nicht gefunden: {base_dir}")
            return

        for level1 in os.listdir(base_dir):
            level1_path = os.path.join(base_dir, level1)
            if not os.path.isdir(level1_path):
                continue

            for level2 in os.listdir(level1_path):
                level2_path = os.path.join(level1_path, level2)
                if not os.path.isdir(level2_path):
                    continue

                for series_folder in os.listdir(level2_path):
                    series_path = os.path.join(level2_path, series_folder)
                    if not os.path.isdir(series_path):
                        continue

                    series_name = series_folder.lower()
                    class_name = None

                    if re.search(r'(^|[^a-z0-9])t2([^a-z0-9]|$)', series_name):
                        class_name = 'mrt_prostata_t2'
                    elif re.search(r'(^|[^a-z0-9])t1([^a-z0-9]|$)', series_name):
                        class_name = 'mrt_prostata_t1'
                    else:
                        continue

                    label = self.class_to_idx[class_name]

                    for file in os.listdir(series_path):
                        file_path = os.path.join(series_path, file)
                        if os.path.isfile(file_path) and self.is_valid_file(file):
                            self.images.append((file_path, label))

    def infer_class_from_path(self, full_path):
        norm_path = os.path.normpath(full_path)
        parts = norm_path.split(os.sep)
        lower_parts = [p.lower() for p in parts]

        # 1) US / Dataset_BUSI
        if 'us' in lower_parts and 'dataset_busi' in lower_parts:
            if self.is_busi_mask(full_path):
                return None
            return 'us'

        if 'us' in lower_parts:
            return 'us'

        # 2) Xray-Fluoroskopie-Angiographie
        if 'xray_fluoroskopie_angiographie' in lower_parts:
            return 'xray_fluoroskopie_angiographie'

        # 3) Generisches Xray
        if 'xray' in lower_parts:
            return 'xray'

        # 4) CT / CT-Kombimodalitaeten
        if 'ct' in lower_parts or any('ct' in p for p in lower_parts):
            for part in reversed(parts):
                pl = part.lower().strip()

                if 'spect' in pl and 'ct' in pl:
                    return 'ct_kombimodalitaet_spect+ct_pet+ct'

                if 'pet' in pl and 'ct' in pl:
                    return 'ct_kombimodalitaet_spect+ct_pet+ct'

                if re.search(r'spect[\s_\-]*ct|ct[\s_\-]*spect', pl):
                    return 'ct_kombimodalitaet_spect+ct_pet+ct'

                if re.search(r'pet[\s_\-]*ct|ct[\s_\-]*pet|petct', pl):
                    return 'ct_kombimodalitaet_spect+ct_pet+ct'

                if pl == 'ct':
                    return 'ct'

                if re.search(r'(^|[^a-z0-9])ct([^a-z0-9]|$)', pl):
                    return 'ct'

            return None

        # 5) MRT Hirn - Brain Tumor MRI Images 44 Classes
        if 'mrt_hirn' in lower_parts and 'brain tumor mri images 44 classes' in lower_parts:
            for part in reversed(parts):
                pl = part.lower().strip()
                if pl.endswith(' t1c+'):
                    return 'mrt_hirn_t1_c'
                if pl.endswith(' t1'):
                    return 'mrt_hirn_t1'
                if pl.endswith(' t2'):
                    return 'mrt_hirn_t2'
            return None

        # 6) MRT Hirn - Neurohacking_data-0.0/BRAINIX/DICOM
        if (
            'mrt_hirn' in lower_parts and
            'neurohacking_data-0.0' in lower_parts and
            'brainix' in lower_parts and
            'dicom' in lower_parts
        ):
            if 'flair' in lower_parts:
                return 'mrt_hirn_flair'
            if 't1' in lower_parts:
                return 'mrt_hirn_t1'
            if 't2' in lower_parts:
                return 'mrt_hirn_t2'
            return None

        # 7) MRT Prostata
        if (
            'mrt_prostata' in lower_parts and
            'prostate_mri' in lower_parts and
            'prostate-mri' in lower_parts
        ):
            for part in reversed(parts):
                pl = part.lower().strip()

                if re.search(r'(^|[^a-z0-9])t2([^a-z0-9]|$)', pl):
                    return 'mrt_prostata_t2'

                if re.search(r'(^|[^a-z0-9])t1([^a-z0-9]|$)', pl):
                    return 'mrt_prostata_t1'

            return None

        if 'mrt_hirn' in lower_parts:
            return None

        if 'mrt_prostata' in lower_parts:
            return None

        return None

    def collect_files(self):
        self.images = []

        for root, _, files in os.walk(self.root_dir):
            norm_root = os.path.normpath(root)
            lower_parts = [p.lower() for p in norm_root.split(os.sep)]

            if 'mrt_prostata' in lower_parts:
                continue

            for file in files:
                if not self.is_valid_file(file):
                    continue

                full_path = os.path.join(root, file)
                class_name = self.infer_class_from_path(full_path)

                if class_name is None:
                    continue

                label = self.class_to_idx[class_name]
                self.images.append((full_path, label))

        self.collect_mrt_prostata_files()


# ============================================================
# Einfaches CNN
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
# Evaluationsfunktionen
# ============================================================

def count_subset_classes(subset, class_names):
    counts = {name: 0 for name in class_names}
    for idx in subset.indices:
        _, label = subset.dataset.images[idx]
        counts[class_names[label]] += 1
    return counts


def print_split_counts(name, counts):
    print(f"\n{name}-Verteilung:")
    for cls_name, count in counts.items():
        print(f"{cls_name}: {count}")


def evaluate_model(model, loader, criterion, device, class_names):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    all_labels = []
    all_preds = []

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

            labels_cpu = labels.cpu()
            preds_cpu = preds.cpu()

            all_labels.extend(labels_cpu.tolist())
            all_preds.extend(preds_cpu.tolist())

            for t, p in zip(labels_cpu.tolist(), preds_cpu.tolist()):
                confusion[t, p] += 1

            acc = correct / total if total > 0 else 0.0
            eval_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc * 100:.2f}%")

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0

    per_class_metrics = []
    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_f1_sum = 0.0

    weighted_precision_sum = 0.0
    weighted_recall_sum = 0.0
    weighted_f1_sum = 0.0

    recalls = []

    for c in range(num_classes):
        tp = confusion[c, c].item()
        fp = confusion[:, c].sum().item() - tp
        fn = confusion[c, :].sum().item() - tp
        support = confusion[c, :].sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        class_acc = recall  # bei Single-Label-Klassifikation identisch mit Trefferquote der Klasse

        per_class_metrics.append({
            "class_name": class_names[c],
            "support": support,
            "accuracy": class_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

        macro_precision_sum += precision
        macro_recall_sum += recall
        macro_f1_sum += f1

        weighted_precision_sum += precision * support
        weighted_recall_sum += recall * support
        weighted_f1_sum += f1 * support

        recalls.append(recall)

    macro_precision = macro_precision_sum / num_classes if num_classes > 0 else 0.0
    macro_recall = macro_recall_sum / num_classes if num_classes > 0 else 0.0
    macro_f1 = macro_f1_sum / num_classes if num_classes > 0 else 0.0

    weighted_precision = weighted_precision_sum / total if total > 0 else 0.0
    weighted_recall = weighted_recall_sum / total if total > 0 else 0.0
    weighted_f1 = weighted_f1_sum / total if total > 0 else 0.0

    balanced_accuracy = sum(recalls) / num_classes if num_classes > 0 else 0.0

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
        "confusion_matrix": confusion
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
    root_dir = "dataset"
    batch_size = 32
    num_epochs = 2
    seed = 42

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = NestedMedicalFolder(root_dir=root_dir, transform=transform)

    print(f"Anzahl der Dateien: {len(dataset)}")
    print(f"Klassen: {dataset.classes}")
    print(f"class_to_idx: {dataset.class_to_idx}")

    class_counts = {cls: 0 for cls in dataset.classes}
    for _, label in dataset.images:
        cls_name = dataset.classes[label]
        class_counts[cls_name] += 1

    print("\nAnzahl pro Klasse im Gesamtdatensatz:")
    for cls_name, count in class_counts.items():
        print(f"{cls_name}: {count}")

    if len(dataset) == 0:
        raise RuntimeError("Keine gueltigen .pt-Dateien gefunden.")

    # Kurzer Sanity Check
    sample_img, sample_label = dataset[0]
    print("\nSanity Check erstes Sample:")
    print(
        f"shape={sample_img.shape}, "
        f"dtype={sample_img.dtype}, "
        f"min={sample_img.min().item():.4f}, "
        f"max={sample_img.max().item():.4f}, "
        f"label={sample_label}"
    )

    # Split: 70 / 15 / 15
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # Split-Verteilungen ausgeben
    train_counts = count_subset_classes(train_dataset, dataset.classes)
    val_counts = count_subset_classes(val_dataset, dataset.classes)
    test_counts = count_subset_classes(test_dataset, dataset.classes)

    print_split_counts("Train", train_counts)
    print_split_counts("Val", val_counts)
    print_split_counts("Test", test_counts)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    # Modell
    num_classes = len(dataset.classes)
    model = SimpleCNN(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device("cpu")
    model.to(device)

    # Training
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

    # Modell speichern
    torch.save(model.state_dict(), "convu_try_3eh.pth")
    print("\nModell gespeichert als convu_try_3eh.pth")

    # Finale Testevaluation
    test_metrics = evaluate_model(model, test_loader, criterion, device, dataset.classes)
    print_metrics("TESTERGEBNISSE", test_metrics)