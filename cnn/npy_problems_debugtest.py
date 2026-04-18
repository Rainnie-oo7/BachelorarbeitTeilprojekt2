import numpy as np
from pathlib import Path

path = Path("/home/b/PycharmProjects/ba2roco/cnn/dataset_mini/us/Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images/ARRAY_FORMAT/P010_IMG1.npy")

arr = np.load(path, allow_pickle=True)

print("Typ:", type(arr))
print("dtype:", getattr(arr, "dtype", None))
print("shape:", getattr(arr, "shape", None))

if isinstance(arr, np.ndarray) and arr.dtype == object:
    print("Objektarray erkannt")
    if arr.shape == ():
        item = arr.item()
        print("Inhaltstyp:", type(item))
        if hasattr(item, "keys"):
            print("Dict-Keys:", list(item.keys()))