# -*- coding: utf-8 -*-

"""
(Unter Ubuntu)
/home/user/Dokumente/local_models/
│
├── clip-vit-base-patch32/
├── blip-retrieval-base/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── preprocessor_config.json
│   └── ...
├── blip-image-captioning-base/
├── biomedblip-base/
└── biomedclip/
    ├── open_clip_pytorch_model.bin
    └── tokenizer.txt
"""

import os.path as osp
from pathlib import Path

import torch
import open_clip

from transformers import (
    AutoModel,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
    BlipForImageTextRetrieval,
)

# ============================================================
# PATHS
# ============================================================

PATH = osp.abspath(
    osp.join(osp.dirname(__file__), "../../../Dokumente")
)

BASE_DIR = Path(PATH)

LOCAL_MODEL_ROOT = BASE_DIR / "local_models"

LOCAL_CLIP_PATH = \
    LOCAL_MODEL_ROOT / "clip-vit-base-patch32"

LOCAL_BLIP_RETRIEVAL_PATH = \
    LOCAL_MODEL_ROOT / "blip-itm-base-coco"

LOCAL_BLIP_CAPTION_PATH = \
    LOCAL_MODEL_ROOT / "blip-image-captioning-base"

LOCAL_BIOMEDCLIP_PATH = \
    LOCAL_MODEL_ROOT / "biomedclip"

LOCAL_MODEL_ROOT.mkdir(parents=True, exist_ok=True)

# # ============================================================
# # CLIP
# # ============================================================
#
# print("\n==============================")
# print("DOWNLOAD CLIP")
# print("==============================")
#
# clip_model = CLIPModel.from_pretrained(
#     "openai/clip-vit-base-patch32"
# )
#
# clip_processor = CLIPProcessor.from_pretrained(
#     "openai/clip-vit-base-patch32"
# )
#
# clip_model.save_pretrained(LOCAL_CLIP_PATH)
# clip_processor.save_pretrained(LOCAL_CLIP_PATH)
#
# print("Gespeichert:", LOCAL_CLIP_PATH)
#
# # ============================================================
# # BLIP RETRIEVAL
# # ============================================================
#
# print("\n==============================")
# print("DOWNLOAD BLIP RETRIEVAL")
# print("==============================")
#
# blip_retrieval_model = BlipForImageTextRetrieval.from_pretrained(
#     "Salesforce/blip-itm-base-coco"
# )
#
# blip_retrieval_processor = BlipProcessor.from_pretrained(
#     "Salesforce/blip-itm-base-coco"
# )
#
# blip_retrieval_model.save_pretrained(
#     LOCAL_BLIP_RETRIEVAL_PATH
# )
#
# blip_retrieval_processor.save_pretrained(
#     LOCAL_BLIP_RETRIEVAL_PATH
# )
#
# print("Gespeichert:", LOCAL_BLIP_RETRIEVAL_PATH)
#
# # ============================================================
# # BLIP CAPTIONING
# # ============================================================
#
# print("\n==============================")
# print("DOWNLOAD BLIP CAPTIONING")
# print("==============================")
#
# blip_caption_model = BlipForConditionalGeneration.from_pretrained(
#     "Salesforce/blip-image-captioning-base"
# )
#
# blip_caption_processor = BlipProcessor.from_pretrained(
#     "Salesforce/blip-image-captioning-base"
# )
#
# blip_caption_model.save_pretrained(
#     LOCAL_BLIP_CAPTION_PATH
# )
#
# blip_caption_processor.save_pretrained(
#     LOCAL_BLIP_CAPTION_PATH
# )
#
# print("Gespeichert:", LOCAL_BLIP_CAPTION_PATH)

# # ============================================================
# # BIOMEDCLIP
# # ============================================================
#
# print("\n==============================")
# print("DOWNLOAD BIOMEDCLIP")
# print("==============================")
#
# LOCAL_BIOMEDCLIP_PATH.mkdir(
#     parents=True,
#     exist_ok=True
# )
#
# model, _, preprocess = open_clip.create_model_and_transforms(
#     "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
# )
#
# tokenizer = open_clip.get_tokenizer(
#     "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
# )

# # ============================================================
# # OPENCLIP STATE DICT SPEICHERN
# # ============================================================
#
# model_path = \
#     LOCAL_BIOMEDCLIP_PATH / "open_clip_pytorch_model.bin"
#
# torch.save(
#     model.state_dict(),
#     model_path
# )
#
# # ============================================================
# # TOKENIZER / CONFIG OPTIONAL SPEICHERN
# # ============================================================
#
# tokenizer_path = \
#     LOCAL_BIOMEDCLIP_PATH / "tokenizer.txt"
#
# with open(tokenizer_path, "w") as f:
#     f.write(
#         "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
#     )
#
# print("Gespeichert:", LOCAL_BIOMEDCLIP_PATH)

# ============================================================
# FERTIG
# ============================================================

print("\n==============================")
print("ALLE MODELLE HERUNTERGELADEN")
print("==============================")