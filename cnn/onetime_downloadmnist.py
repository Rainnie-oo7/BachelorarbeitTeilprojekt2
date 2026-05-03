import os
from pathlib import Path
import os.path as osp
from medmnist import PathMNIST

BASE_DIR = osp.normpath(osp.join(osp.dirname(__file__), "dataset3"))
BASE_DIR = Path(BASE_DIR)

target_dir = BASE_DIR / "PathMNIST_histologie"


# Verzeichnis erstellen, falls es nicht existiert
os.makedirs(target_dir, exist_ok=True)

# Datensatz herunterladen und speichern
dataset = PathMNIST(
    split="val",
    download=True,
    size=224,
    root=target_dir  # Speichert die Daten im angegebenen Verzeichnis
)

print(f"Datensatz wurde erfolgreich unter {target_dir} gespeichert.")