import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers.utils import logging
from pathlib import Path
import os.path as osp

PATH = osp.normpath(osp.dirname(__file__))
BASE_DIR = Path(PATH)
# trainLoRA > project > user *this* / Dokumente / blipbase or
# LOCAL_BLIP_PATH = Path("/home/user/Dokumente/biomedbert")
LOCAL_BLIP_PATH = BASE_DIR.parent.parent.parent / "Dokumente" / "blipbase"
print("LOCAL_BLIP_PATH", LOCAL_BLIP_PATH)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
# Progress aktivieren
logging.set_verbosity_info()
logging.enable_progress_bar()
# Wenn du immer noch nichts siehst: Hugging Face speichert hier: ~/.cache/huggingface/
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model.save_pretrained(LOCAL_BLIP_PATH)
processor.save_pretrained(LOCAL_BLIP_PATH)
