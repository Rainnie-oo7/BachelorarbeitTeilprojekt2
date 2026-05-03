import os
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers.utils import logging
from pathlib import Path
import os.path as osp
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers.utils import logging

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# Progress aktivieren
logging.set_verbosity_info()
logging.enable_progress_bar()

PATH = osp.normpath(osp.dirname(__file__))
BASE_DIR = Path(PATH)
# trainLoRA > project > user *this* >  / Dokumente / blipbase or
# LOCAL_BLIP_PATH = Path("/home/user/Dokumente/biomedbert")
LOCAL_BLIP_PATH = BASE_DIR.parent.parent.parent / "Dokumente" / "blipbase"
print("LOCAL_BLIP_PATH", LOCAL_BLIP_PATH)
print("EXISTS:", LOCAL_BLIP_PATH.exists())

device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", device)

print("Lade Modell...")
model = BlipForConditionalGeneration.from_pretrained(
    str(LOCAL_BLIP_PATH),
    local_files_only=True
).to(device)

processor = BlipProcessor.from_pretrained(str(LOCAL_BLIP_PATH), local_files_only=True)

model.eval()
print("Modell geladen.")

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
# print("Lade Bild...")
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
raw_image = Image.open("demo.jpg").convert('RGB')
print("Bild geladen.")
###################
print("Starte conditional captioning...")
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt")
# auf device schieben
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=30)
print("Output 1:")
print(processor.decode(out[0], skip_special_tokens=True))
# a photography of a man riding a bike with a cat on the back
###################
print("Starte unconditional captioning...")
inputs = processor(raw_image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=30)

print("Output 2:")
print(processor.decode(out[0], skip_special_tokens=True))
# a man riding a bike with a cat on the back
print("FERTIG")