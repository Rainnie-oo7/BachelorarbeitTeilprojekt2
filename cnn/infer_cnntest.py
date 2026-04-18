import os
from PIL import Image
import numpy as np
import pydicom

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, Dataset

from main_cnn import SimpleCNN, array_to_pil_rgb

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Modell laden und verwenden
############
model = SimpleCNN(num_classes=8)
model.load_state_dict(torch.load('convu_try_1.pth'))
model.eval()

# Beispiel: Einen Bild klassifizieren
device = 'cpu'
# fuer dicom
path = 'Unbekanntes_bild.dcm'
dcm = pydicom.dcmread(path)
arr = dcm.pixel_array
img = array_to_pil_rgb(arr)
# img = Image.open('Unbekanntes_bild.png').convert('RGB')  # Hier .convert('RGB') hinzufügen
img_tensor = transform(img).unsqueeze(0).to(device)
output = model(img_tensor)
_, predicted = torch.max(output, 1)
print(f'Vorhersage: Klasse {predicted.item()}')