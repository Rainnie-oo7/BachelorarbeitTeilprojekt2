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

    # auf float32
    tensor = tensor.float()

    # Leere / skalarartige Faelle vermeiden
    if tensor.ndim == 0:
        raise ValueError(f"Tensor ist skalar und kein Bild: shape={tuple(tensor.shape)}")

    # H x W -> 1 x H x W
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)

    # 3D-Fall
    elif tensor.ndim == 3:
        # Falls H x W x C vorliegt und letztes Maß Kanalzahl ist
        if tensor.shape[-1] in (1, 3, 4) and tensor.shape[0] not in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1).contiguous()

        # sonst nehmen wir an, es ist bereits C x H x W

    else:
        raise ValueError(f"Nicht unterstuetzte Tensor-Form fuer Bilddaten: {tuple(tensor.shape)}")

    # Jetzt sollte tensor C x H x W sein
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
# Hauptteil
# ============================================================

if __name__ == "__main__":
    root_dir = "dataset"

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

    print("\nAnzahl pro Klasse:")
    for cls_name, count in class_counts.items():
        print(f"{cls_name}: {count}")

    if len(dataset) == 0:
        raise RuntimeError("Keine gueltigen .pt-Dateien gefunden.")

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
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

    num_epochs = 2

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

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")

        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs = inputs.to(device, dtype=torch.float32)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                acc = 100 * correct / total if total > 0 else 0.0
                val_bar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2f}%")

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {running_loss / len(train_loader):.4f}, "
            f"Val Loss: {val_loss / len(val_loader):.4f}, "
            f"Val Accuracy: {100 * correct / total:.2f}%"
        )

    torch.save(model.state_dict(), "convu_try_3eh.pth")
    print("\nModell gespeichert als convu_try_3eh.pth")

"""
class_to_idx: {'xray': 0, 'xray_fluoroskopie_angiographie': 1, 'us': 2, 'mrt_hirn_flair': 3, 'mrt_hirn_t1': 4, 'mrt_hirn_t2': 5, 'mrt_hirn_t1_c': 6, 'mrt_prostata_t1': 7, 'mrt_prostata_t2': 8, 'ct': 9, 'ct_kombimodalitaet_spect+ct_pet+ct': 10}

Anzahl pro Klasse:
xray: 54415
xray_fluoroskopie_angiographie: 3000
us: 23804
mrt_hirn_flair: 22
mrt_hirn_t1: 1429
mrt_hirn_t2: 1385
mrt_hirn_t1_c: 1692
mrt_prostata_t1: 10
mrt_prostata_t2: 1656
ct: 98158
ct_kombimodalitaet_spect+ct_pet+ct: 418723

# ============
# Hinweis 1: Falls deine .pt-Dateien schon normalisiert gespeichert wurden
# ============
Dann solltest du nicht noch einmal normalisieren.
Im Moment nehme ich an, dass deine gespeicherten Tensoren noch rohe oder zumindest unnormalisierte Bildwerte enthalten.
Falls du beim Konvertieren schon auf [0, 1] oder [-1, 1] gebracht hast, muss man das anpassen.

# ============
# Sehr sinnvoller Test vor dem Training
# ============
Führe einmal vor dem Split das hier aus:

sample_img, sample_label = dataset[0]
print(sample_img.shape, sample_img.dtype, sample_img.min().item(), sample_img.max().item(), sample_label)

Ideal für dein Modell wäre danach etwa:

torch.Size([3, 224, 224]) torch.float32 ...

# ============
# Falls du rohe Intensitäten aus CT/MRT als .pt gespeichert hast
# ============
Dann ist oft besser, vor dem Normalize erst selbst zu skalieren, zum Beispiel pro Bild auf [0,1]. 
Sonst kann Normalize((0.5,...), (0.5,...)) auf sehr große Intensitätswerte ungünstig wirken.
Dann würde man in ensure_image_tensor() oder direkt danach noch so etwas einbauen:

tensor_min = tensor.min()
tensor_max = tensor.max()
if tensor_max > tensor_min:
    tensor = (tensor - tensor_min) / (tensor_max - tensor_min)
else:
    tensor = torch.zeros_like(tensor)

Das hängt davon ab, wie deine .pt-Dateien gespeichert wurden.
"""