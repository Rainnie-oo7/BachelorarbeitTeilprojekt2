import os
import re
import numpy as np
from PIL import Image
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm

# ============================================================
# Hilfsfunktionen zum Laden und Vereinheitlichen
# ============================================================

VALID_EXTENSIONS = {'.dcm', '.nii', '.jpeg', '.png', '.jpg', '.npy', '.tif', '.tiff'}


def normalize_to_uint8(arr):
    """
    Beliebiges numpy-Array robust auf uint8 [0,255] skalieren.
    """
    arr = np.asarray(arr)

    if arr.size == 0:
        raise ValueError("Leeres Array kann nicht verarbeitet werden.")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    min_val = arr.min()
    max_val = arr.max()

    if max_val == min_val:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = (arr - min_val) / (max_val - min_val)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def array_to_pil_rgb(arr):
    """
    Wandelt 2D/3D-Arrays in PIL-RGB um.
    - 2D: Graubild -> RGB
    - 3D:
        * wenn letzte Achse 3 oder 4 ist: als Bild behandeln
        * sonst mittleren Slice nehmen
    - 4D: ersten Zeitpunkt nehmen, dann weiter
    """
    arr = np.asarray(arr)

    # 4D -> ersten Zeitpunkt/Kanal nehmen
    if arr.ndim == 4:
        arr = arr[..., 0]

    # 3D-Fall
    if arr.ndim == 3:
        # Bereits H x W x C Bild?
        if arr.shape[-1] in (3, 4):
            arr = normalize_to_uint8(arr)
            if arr.shape[-1] == 4:
                return Image.fromarray(arr, mode='RGBA').convert("RGB")
            return Image.fromarray(arr).convert("RGB")

        # Sonst als Volumen interpretieren -> mittleren Slice nehmen
        mid = arr.shape[2] // 2
        arr = arr[:, :, mid]

    # 2D-Fall
    if arr.ndim == 2:
        arr = normalize_to_uint8(arr)
        return Image.fromarray(arr).convert("RGB")

    raise ValueError(f"Unerwartete Array-Form: {arr.shape}")


def load_file(path):
    """
    Laedt verschiedene Dateitypen und gibt IMMER ein PIL-RGB-Bild zurueck,
    damit torchvision-Transforms konsistent funktionieren.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
        return Image.open(path).convert("RGB")

    elif ext == '.dcm':
        dcm = pydicom.dcmread(path)
        arr = dcm.pixel_array
        return array_to_pil_rgb(arr)


    elif ext == '.npy':

        arr = np.load(path, allow_pickle=True)

        # Falls 0-dim Object-Array, Inhalt herausziehen

        if isinstance(arr, np.ndarray) and arr.dtype == object:

            if arr.shape == ():
                arr = arr.item()

        # Falls dict gespeichert wurde

        if isinstance(arr, dict):

            # haeufige Schluessel ausprobieren

            for key in ['image', 'img', 'array', 'data']:

                if key in arr:
                    arr = arr[key]

                    break

            else:

                raise ValueError(f"NPY-Datei enthaelt dict ohne bekannten Bild-Schluessel: {path}")

        # Falls Liste -> numpy

        if isinstance(arr, list):
            arr = np.array(arr)

        # Falls danach immer noch kein ndarray

        if not isinstance(arr, np.ndarray):
            raise ValueError(f"NPY-Datei konnte nicht als Bild interpretiert werden: {path} | Typ: {type(arr)}")

        return array_to_pil_rgb(arr)

    else:
        raise ValueError(f"Unbekannter Dateityp: {ext}")


# ============================================================
# Dataset-Klasse mit Sonderregeln
# ============================================================

class NestedMedicalFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Zielklassen explizit festlegen
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
        image = load_file(img_path)

        if self.transform:
            image = self.transform(image)

        return image, label

    def is_valid_file(self, filename):
        ext = os.path.splitext(filename)[1].lower()
        return ext in VALID_EXTENSIONS

    def is_busi_mask(self, path):
        """
        In us/Dataset_BUSI sollen Masken ignoriert werden:
        z.B.
        benign (1)_mask.png
        benign (4)_mask_1.png
        """
        filename = os.path.basename(path).lower()
        return '_mask' in filename

    def collect_mrt_prostata_files(self):
        """
        Erwartete Struktur:
        dataset/mrt_prostata/PROSTATE_MRI/PROSTATE_MRI/
            patient_1/
                study_1/
                    serie_1/
                    serie_2/
                    ...
            patient_2/
                study_2/
                    serie_1/
                    ...
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

        # 1. Schicht
        for level1 in os.listdir(base_dir):
            level1_path = os.path.join(base_dir, level1)
            if not os.path.isdir(level1_path):
                continue

            # 2. Schicht
            for level2 in os.listdir(level1_path):
                level2_path = os.path.join(level1_path, level2)
                if not os.path.isdir(level2_path):
                    continue

                # 3. Schicht = Serienordner
                for series_folder in os.listdir(level2_path):
                    series_path = os.path.join(level2_path, series_folder)
                    if not os.path.isdir(series_path):
                        continue

                    series_name = series_folder.lower()

                    class_name = None

                    # zuerst T2, dann T1
                    if re.search(r'(^|[^a-z0-9])t2([^a-z0-9]|$)', series_name):
                        class_name = 'mrt_prostata_t2'
                    elif re.search(r'(^|[^a-z0-9])t1([^a-z0-9]|$)', series_name):
                        class_name = 'mrt_prostata_t1'
                    else:
                        continue  # DWI, DCE usw. ignorieren

                    label = self.class_to_idx[class_name]

                    # Dateien direkt im Serienordner einsammeln
                    for file in os.listdir(series_path):
                        file_path = os.path.join(series_path, file)
                        if os.path.isfile(file_path) and self.is_valid_file(file):
                            self.images.append((file_path, label))

    def infer_class_from_path(self, full_path):
        """
        Bestimmt die Zielklasse anhand des Pfades.
        Gibt Klassennamen zurueck oder None, falls Datei ignoriert werden soll.
        """
        norm_path = os.path.normpath(full_path)
        parts = norm_path.split(os.sep)
        lower_parts = [p.lower() for p in parts]
        lower_path = norm_path.lower()

        # --------------------------------------------------------
        # 1) US / Dataset_BUSI
        # Ordner: us/Dataset_BUSI/.../benign|malignant|normal
        # Alle Nicht-Masken gehoeren einfach zur Klasse 'us'
        # --------------------------------------------------------
        if 'us' in lower_parts and 'dataset_busi' in lower_parts:
            if self.is_busi_mask(full_path):
                return None
            return 'us'

        # Generischer US-Fall
        if 'us' in lower_parts:
            return 'us'

        # --------------------------------------------------------
        # 2) Xray-Fluoroskopie-Angiographie
        # Muss vor xray geprueft werden
        # --------------------------------------------------------
        if 'xray_fluoroskopie_angiographie' in lower_parts:
            return 'xray_fluoroskopie_angiographie'

        # --------------------------------------------------------
        # 3) Generisches Xray
        # --------------------------------------------------------
        if 'xray' in lower_parts:
            return 'xray'

        # --------------------------------------------------------
        # 4) CT / CT-Kombimodalitaeten
        # Ordnername soll verwendet werden
        # Kombiklassen immer vor generischem CT pruefen
        # --------------------------------------------------------
        if 'ct' in lower_parts or any('ct' in p for p in lower_parts):
            for part in reversed(parts):
                pl = part.lower().strip()

                # Kombimodalitaeten
                if 'spect' in pl and 'ct' in pl:
                    return 'ct_kombimodalitaet_spect+ct_pet+ct'

                if 'pet' in pl and 'ct' in pl:
                    return 'ct_kombimodalitaet_spect+ct_pet+ct'

                # Optional auch fuer Schreibweisen wie "spect-ct", "pet_ct", "petct"
                if re.search(r'spect[\s_\-]*ct|ct[\s_\-]*spect', pl):
                    return 'ct_kombimodalitaet_spect+ct_pet+ct'

                if re.search(r'pet[\s_\-]*ct|ct[\s_\-]*pet|petct', pl):
                    return 'ct_kombimodalitaet_spect+ct_pet+ct'

                # Generisches CT
                if pl == 'ct':
                    return 'ct'

                if re.search(r'(^|[^a-z0-9])ct([^a-z0-9]|$)', pl):
                    return 'ct'

            return None

        # --------------------------------------------------------
        # 5) MRT Hirn - Brain Tumor MRI Images 44 Classes
        # Beispiele:
        # .../Brain Tumor MRI Images 44 Classes/Ependimoma T2/...
        # .../Brain Tumor MRI Images 44 Classes/Ganglioglioma T1/...
        # --------------------------------------------------------
        if 'mrt_hirn' in lower_parts and 'brain tumor mri images 44 classes' in lower_parts:
            for part in reversed(parts):
                pl = part.lower().strip()
                if pl.endswith(' t1'):
                    return 'mrt_hirn_t1'
                if pl.endswith(' t2'):
                    return 'mrt_hirn_t2'
                if pl.endswith(' t1c+'):
                    return 'mrt_hirn_t1_c'
            return None

        # --------------------------------------------------------
        # 6) MRT Hirn - Neurohacking_data-0.0/BRAINIX/DICOM
        # Klassen durch Ordner FLAIR / T1 / T2
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # 7) MRT Prostata - PROSTATE_MRI/PROSTATE-MRI
        # Nur Serienordner mit T1 oder T2 verwenden.
        # DWI, DCE, SShDWI, dSSh, SENSE, FAST, etc. ignorieren.
        # --------------------------------------------------------
        if (
                'mrt_prostata' in lower_parts and
                'prostate_mri' in lower_parts and
                'prostate-mri' in lower_parts
        ):
            for part in reversed(parts):
                pl = part.lower().strip()

                # zuerst T2, dann T1
                if re.search(r'(^|[^a-z0-9])t2([^a-z0-9]|$)', pl):
                    return 'mrt_prostata_t2'

                if re.search(r'(^|[^a-z0-9])t1([^a-z0-9]|$)', pl):
                    return 'mrt_prostata_t1'

            return None

        # Falls unter mrt_hirn oder mrt_prostata, aber keine passende Regel
        if 'mrt_hirn' in lower_parts:
            return None

        if 'mrt_prostata' in lower_parts:
            return None

        return None

    def collect_files(self):
        self.images = []

        # --------------------------------------------------------
        # 1) Standard-Rekursion über alles außer mrt_prostata
        # --------------------------------------------------------
        for root, _, files in os.walk(self.root_dir):
            norm_root = os.path.normpath(root)
            lower_parts = [p.lower() for p in norm_root.split(os.sep)]

            # mrt_prostata hier überspringen, wird separat behandelt
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

        # --------------------------------------------------------
        # 2) Spezialfall mrt_prostata gezielt einsammeln
        # --------------------------------------------------------
        self.collect_mrt_prostata_files()


# ============================================================
# Einfaches CNN
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============================================================
# Hauptteil
# ============================================================

if __name__ == "__main__":
    root_dir = "dataset_mini"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = NestedMedicalFolder(root_dir=root_dir, transform=transform)

    # Debug
    # print(f"Anzahl der Dateien: {len(dataset)}")
    # print(f"Klassen: {dataset.classes}")
    # print(f"class_to_idx: {dataset.class_to_idx}")
    # # Anzahl pro Klasse ausgeben
    # class_counts = {cls: 0 for cls in dataset.classes}
    # for _, label in dataset.images:
    #     cls_name = dataset.classes[label]
    #     class_counts[cls_name] += 1
    #
    # print("\nAnzahl pro Klasse:")
    # for cls_name, count in class_counts.items():
    #     print(f"{cls_name}: {count}")
    #
    # if len(dataset) > 15:
    #     exp = 15
    # else:
    #     exp = 0
    #
    # if len(dataset) > 0:
    #     print(f"\nBeispiel-Dateipfad: {dataset.images[exp][0]}")
    #     print(f"Beispiel-Label: {dataset.images[exp][1]} -> {dataset.classes[dataset.images[exp][1]]}")
    # else:
    #     raise RuntimeError("Keine gueltigen Dateien gefunden. Bitte Pfade/Regeln pruefen.")
    # print("len(dataset.classes)", len(dataset.classes))

    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Modell
    num_classes = len(dataset.classes)
    model = SimpleCNN(num_classes=num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    device = torch.device('cpu')
    # print(f"\nVerwendetes Geraet: {device}")
    model.to(device)

    # Training
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            train_bar.set_postfix(loss=loss.item())
            #Durchschnittlichen Loss anzeigen anstatt Batch-Loss
            # train_bar.set_postfix(loss=running_loss / (train_bar.n + 1))

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        val_bar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]")

        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                acc = 100 * correct / total if total > 0 else 0
                val_bar.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Train Loss: {running_loss / len(train_loader):.4f}, "
            f"Val Loss: {val_loss / len(val_loader):.4f}, "
            f"Val Accuracy: {100 * correct / total:.2f}%"
        )

    # Modell speichern
    torch.save(model.state_dict(), 'convu_try_1.pth')
    print("\nModell gespeichert als convu_try_1.pth")

"""
# ----------------------
# Debug-Output:
# ----------------------
class_to_idx: {
'xray': 0, 
'xray_fluoroskopie_angiographie': 1, 
'us': 2, 'mrt_hirn_flair': 3, 'mrt_hirn_t1': 4, 'mrt_hirn_t2': 5, 'mrt_hirn_t1_c': 6, 
'mrt_prostata_t1': 7, 'mrt_prostata_t2': 8, 
'ct': 9, 'ct_kombimodalitaet_spect+ct_pet+ct': 10}

Anzahl pro Klasse:
xray: 54683
xray_fluoroskopie_angiographie: 3000
us: 23804
mrt_hirn_flair: 22
mrt_hirn_t1: 1444
mrt_hirn_t2: 1385
mrt_hirn_t1_c: 1692
mrt_prostata_t1: 10
mrt_prostata_t2: 1656
ct: 98158
ct_kombimodalitaet_spect+ct_pet+ct: 418723
insgesamt 604577
"""