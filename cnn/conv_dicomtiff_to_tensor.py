import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import pydicom
import nibabel as nib
import tifffile
# -------------------Ein einfacher Konverter fuer alle der 600k Bilder---------------------------------
# Grund:Das Training dauerte aktuell 10 h pro Epoch. Aktuell passiert das im Training: dcm.pixel_array
# Das passiert bei JEDEM Zugriff neu. Das ist extrem teuer. Daher einmal offline:
# Am einfachsten:                                       DICOM einmal zu PNG oder NumPy konvertieren
# Am kompatibelsten und am schnellsten im Training:     Bild zu Torch-Tensor kodieren/umwandeln
# -----------------------------------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = {
    ".dcm",
    ".nii",
    ".nii.gz",
    ".jpeg",
    ".png",
    ".jpg",
    ".npy",
    ".tif",
    ".tiff",
}


def has_supported_suffix(path: Path) -> bool:
    name_lower = path.name.lower()
    return any(name_lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS)


def strip_all_known_suffixes(path: Path) -> str:
    """
    Entfernt auch doppelte Suffixe wie .nii.gz sauber.
    """
    name = path.name
    lower = name.lower()

    for ext in sorted(SUPPORTED_EXTENSIONS, key=len, reverse=True):
        if lower.endswith(ext):
            return name[: -len(ext)]
    return path.stem


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """
    Normalisiert beliebige numerische Arrays nach uint8 [0,255].
    Für PNG-Ausgabe geeignet.
    """
    arr = np.asarray(arr)

    if arr.size == 0:
        raise ValueError("Leeres Array kann nicht normalisiert werden.")

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    min_val = arr.min()
    max_val = arr.max()

    if max_val <= min_val:
        return np.zeros(arr.shape, dtype=np.uint8)

    arr = (arr - min_val) / (max_val - min_val)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def ensure_2d_or_3d_image(arr: np.ndarray) -> np.ndarray:
    """
    Vereinheitlicht Bilddaten.
    Erlaubt:
    - 2D: H x W
    - 3D mit Kanalannahme: H x W x C oder C x H x W
    - NIfTI 3D Volumen: nimmt mittlere Schicht
    """
    arr = np.asarray(arr)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        # Fall 1: C x H x W mit kleinem Kanalmaß
        if arr.shape[0] in (1, 3, 4) and arr.shape[1] > 4 and arr.shape[2] > 4:
            # Für PNG lieber HWC
            return np.transpose(arr, (1, 2, 0))

        # Fall 2: H x W x C
        if arr.shape[-1] in (1, 3, 4):
            return arr

        # Sonst medizinisches Volumen -> mittlere Schicht entlang letzter Achse
        mid = arr.shape[-1] // 2
        return arr[:, :, mid]

    if arr.ndim == 4:
        # z. B. NIfTI mit H x W x D x C
        # Nimm mittlere Schicht und ersten Kanal
        mid = arr.shape[2] // 2
        arr = arr[:, :, mid, 0]
        return arr

    raise ValueError(f"Nicht unterstützte Array-Dimension: {arr.ndim}")


def load_dcm(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array
    return np.asarray(arr)


def load_nii(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    arr = img.get_fdata()
    return np.asarray(arr)


def load_pil_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        # Nicht zwangsweise RGB, um Grauwerte zu erhalten
        arr = np.array(img)
    return arr


def load_npy(path: Path) -> np.ndarray:
    try:
        arr = np.load(path, allow_pickle=False)
    except ValueError as e:
        if "Object arrays cannot be loaded when allow_pickle=False" in str(e):
            arr = np.load(path, allow_pickle=True)
        else:
            raise

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        # Einzelobjekt
        if arr.shape == ():
            obj = arr.item()

            if isinstance(obj, np.ndarray):
                return np.asarray(obj)

            if isinstance(obj, dict):
                # bevorzugt typischen Bild-Key
                for key in ["image", "img", "array", "data"]:
                    if key in obj:
                        return np.asarray(obj[key])
                raise ValueError(f"Dict in .npy enthält keinen bekannten Bild-Key: {list(obj.keys())}")

            if isinstance(obj, (list, tuple)):
                # erstes numerisches Element nehmen
                for x in obj:
                    try:
                        candidate = np.asarray(x)
                        if candidate.dtype != object:
                            return candidate
                    except Exception:
                        pass
                raise ValueError("Liste/Tuple in .npy enthält kein brauchbares numerisches Array.")

            raise ValueError(f"Nicht unterstützter Objekttyp in .npy: {type(obj)}")

        # Mehrere Objekte
        try:
            candidate = np.asarray(arr.tolist())
            if candidate.dtype != object:
                return candidate
        except Exception:
            pass

        raise ValueError(f"Objektarray in .npy konnte nicht sinnvoll in numerisches Bild umgewandelt werden: {path}")

    return np.asarray(arr)


def load_tiff(path: Path) -> np.ndarray:
    try:
        arr = tifffile.imread(str(path))
        return np.asarray(arr)
    except Exception as e:
        if "imagecodecs" in str(e).lower():
            raise RuntimeError(
                f"TIFF-Datei benötigt 'imagecodecs' zum Dekodieren: {path}"
            ) from e
        raise


def load_file_as_array(path: Path) -> np.ndarray:
    lower = path.name.lower()

    if lower.endswith(".nii.gz") or lower.endswith(".nii"):
        return load_nii(path)
    if lower.endswith(".dcm"):
        return load_dcm(path)
    if lower.endswith(".npy"):
        return load_npy(path)
    if lower.endswith(".tif") or lower.endswith(".tiff"):
        return load_tiff(path)
    if lower.endswith(".png") or lower.endswith(".jpg") or lower.endswith(".jpeg"):
        return load_pil_image(path)

    raise ValueError(f"Nicht unterstütztes Format: {path}")


def array_to_png(path_out: Path, arr: np.ndarray) -> None:
    arr = ensure_2d_or_3d_image(arr)

    # Falls 1-Kanal als H x W x 1 vorliegt -> H x W
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    arr_uint8 = normalize_to_uint8(arr)

    if arr_uint8.ndim == 2:
        img = Image.fromarray(arr_uint8, mode="L")
    elif arr_uint8.ndim == 3 and arr_uint8.shape[-1] == 3:
        img = Image.fromarray(arr_uint8, mode="RGB")
    elif arr_uint8.ndim == 3 and arr_uint8.shape[-1] == 4:
        img = Image.fromarray(arr_uint8, mode="RGBA")
    else:
        raise ValueError(f"PNG-Ausgabe nicht möglich für Form: {arr_uint8.shape}")

    img.save(path_out)


def array_to_tensor(arr: np.ndarray) -> torch.Tensor:
    arr = ensure_2d_or_3d_image(arr)
    arr = np.asarray(arr)

    if arr.ndim == 2:
        # 1 x H x W
        tensor = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
        return tensor

    if arr.ndim == 3:
        # H x W x C -> C x H x W
        tensor = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).contiguous()
        return tensor

    raise ValueError(f"Tensor-Konvertierung nicht möglich für Form: {arr.shape}")


def save_as_pt(path_out: Path, arr: np.ndarray) -> None:
    tensor = array_to_tensor(arr)
    torch.save(tensor, path_out)


def save_as_npy(path_out: Path, arr: np.ndarray) -> None:
    arr = ensure_2d_or_3d_image(arr)
    np.save(path_out, arr.astype(np.float32))


def convert_file(path: Path, output_mode: str) -> None:
    arr = load_file_as_array(path)
    base_name = strip_all_known_suffixes(path)

    if output_mode == "png":
        out_path = path.with_name(base_name + ".png")
        array_to_png(out_path, arr)

    elif output_mode == "pt":
        out_path = path.with_name(base_name + ".pt")
        save_as_pt(out_path, arr)

    elif output_mode == "npy":
        out_path = path.with_name(base_name + ".npy")
        save_as_npy(out_path, arr)

    else:
        raise ValueError(f"Unbekannter output_mode: {output_mode}")


def ask_user_choice() -> str:
    print("Bitte Zielformat wählen:")
    print("1 = .png")
    print("2 = .pt (Torch-Tensor)")
    print("3 = .npy")

    choice = input("Eingabe (1/2/3): ").strip()

    mapping = {
        "1": "png",
        "2": "pt",
        "3": "npy",
    }

    if choice not in mapping:
        raise ValueError("Ungültige Auswahl. Bitte 1, 2 oder 3 eingeben.")

    return mapping[choice]


def main():
    # root_input = input("Wurzelordner eingeben: ").strip()
    # if not root_input:
    #     raise ValueError("Kein Wurzelordner angegeben.")
    root_input = 'dataset'
    root_dir = Path(root_input).expanduser().resolve()
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"Ordner nicht gefunden: {root_dir}")

    output_mode = ask_user_choice()

    all_files = [p for p in root_dir.rglob("*") if p.is_file() and has_supported_suffix(p)]

    print(f"\nGefundene passende Dateien: {len(all_files)}")
    print(f"Zielformat: .{output_mode}\n")

    success = 0
    failed = 0

    for i, path in enumerate(all_files, start=1):
        try:
            convert_file(path, output_mode)
            success += 1
            print(f"[{i}/{len(all_files)}] OK:   {path}")
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(all_files)}] FEHLER: {path}")
            print(f"    Grund: {e}")

    print("\nFertig.")
    print(f"Erfolgreich: {success}")
    print(f"Fehlerhaft:  {failed}")


if __name__ == "__main__":
    main()

"""
Fertig.
Erfolgreich: 625742
Fehlerhaft:  4
welche nicht geprueft
"""