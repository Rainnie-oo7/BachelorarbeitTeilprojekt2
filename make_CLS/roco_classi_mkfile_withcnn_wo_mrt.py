from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import re
import os
import os.path as osp
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from transformers import BertTokenizer, BertModel
import torchvision.transforms as transforms


# =========================================================
# Konfiguration
# =========================================================

OPERATING_PATH = Path(__file__).resolve().parent
DATA_PATH = osp.normpath(osp.join(osp.dirname(__file__), "../data"))
BASE_DIR = Path(DATA_PATH)

LOCAL_BIOMEDBERT_PATH = BASE_DIR.parent.parent.parent.parent / "Dokumente" / "biomedbert"
# make_CLS > project > user > *this* / Dokumente / biomedbert or
#LOCAL_BIOMEDBERT_PATH = Path("/home/user/Dokumente/biomedbert")

LOCAL_CNN_MODEL_PATH = Path(osp.normpath(osp.join(OPERATING_PATH, "../cnn", "convu_try_2_big.pth")))

OUTPUT_TXT = Path("cap_clsfy_rules_cnn_bert_wo_mrt.txt")
OUTPUT_CSV = Path("cap_clsfy_rules_cnn_bert.csv")

BATCH_SIZE = 32
MAX_LENGTH = 128
DEBUG_LIMIT = 50   # None fuer alles
DEVICE = "cpu"

USE_UNCLEAR_LABEL = True
UNCLEAR_LABEL = "Unklar"

MIN_MARGIN = 0.015
MIN_TOP1_SIM = 0.22

CNN_MIN_CONF = 0.55
FUSION_MIN_SCORE = 1.20


# =========================================================
# Klassen
# OHNE Oberklasse "mrt"
# =========================================================

ALL_CLASSES: List[str] = [
    "xray",
    "xray_fluoroskopie_angiographie",
    "us",
    "mrt_hirn_flair",
    "mrt_hirn_t1",
    "mrt_hirn_t2",
    "mrt_hirn_t1_c",
    "mrt_prostata_t1",
    "mrt_prostata_t2",
    "ct",
    "ct_kombimodalitaet_spect+ct_pet+ct",
]

# Falls dein altes CNN mit 10 oder 11 Klassen trainiert wurde,
# MUSS diese Liste exakt dem Training entsprechen.
# Beispiel hier ohne generische MRT-Klasse:
CNN_CLASSES: List[str] = [
    "xray",
    "xray_fluoroskopie_angiographie",
    "us",
    "mrt_hirn_flair",
    "mrt_hirn_t1",
    "mrt_hirn_t2",
    "mrt_hirn_t1_c",
    "mrt_prostata_t1",
    "mrt_prostata_t2",
    "ct",
    "ct_kombimodalitaet_spect+ct_pet+ct",
]

# nur noch fuer CT-Kombiklasse eine Hierarchie
PARENT_MAP: Dict[str, Optional[str]] = {
    "xray": None,
    "xray_fluoroskopie_angiographie": None,
    "us": None,
    "mrt_hirn_flair": None,
    "mrt_hirn_t1": None,
    "mrt_hirn_t2": None,
    "mrt_hirn_t1_c": None,
    "mrt_prostata_t1": None,
    "mrt_prostata_t2": None,
    "ct": None,
    "ct_kombimodalitaet_spect+ct_pet+ct": "ct",
}


def get_parent_or_self(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    parent = PARENT_MAP.get(label)
    return parent if parent is not None else label


# =========================================================
# Klassenbeschreibungen fuer BERT
# OHNE generische Klasse "mrt"
# =========================================================

CLASS_TEXTS: Dict[str, str] = {
    "xray": (
        "X-ray radiography, radiograph, plain radiograph, chest x-ray, abdominal x-ray, "
        "skeletal radiograph, projection radiography, conventional x-ray, AP view, PA view, "
        "lateral view, portable x-ray, panoramic radiograph, dorsoplantar projection, "
        "plain film, frontal radiograph, lateral radiograph."
    ),
    "xray_fluoroskopie_angiographie": (
        "Fluoroscopy, fluoroscopic x-ray imaging, real-time x-ray imaging, c-arm fluoroscopy, "
        "x-ray guided interventional procedure, fluoroscopic guidance, contrast fluoroscopy, "
        "angiography, angiographic image, digital subtraction angiography, DSA, catheter angiography, "
        "vascular intervention, angioplasty, balloon angioplasty, percutaneous transluminal angioplasty, "
        "PTA, PTCA, coronary angioplasty, stent placement, percutaneous coronary intervention, PCI."
    ),
    "us": (
        "Ultrasound imaging, sonography, ultrasonography, echography, echocardiography, "
        "B-mode ultrasound, doppler ultrasound, duplex sonography, endoscopic ultrasound, EUS, "
        "color doppler, power doppler, sonographic examination, ultrasonographic image."
    ),
    "mrt_hirn_flair": (
        "Brain MRI FLAIR sequence, cerebral MRI FLAIR, fluid-attenuated inversion recovery MRI, "
        "axial FLAIR brain image, cranial MRI FLAIR, neuro MRI FLAIR, head MRI FLAIR."
    ),
    "mrt_hirn_t1": (
        "Brain MRI T1-weighted image, cerebral MRI T1, cranial MRI T1-weighted, "
        "head MRI T1 sequence, neuro MRI T1, native T1-weighted brain MRI, pre-contrast T1 brain MRI."
    ),
    "mrt_hirn_t2": (
        "Brain MRI T2-weighted image, cerebral MRI T2, cranial MRI T2-weighted, "
        "head MRI T2 sequence, neuro MRI T2, axial T2-weighted brain MRI."
    ),
    "mrt_hirn_t1_c": (
        "Contrast-enhanced T1-weighted brain MRI, post-contrast T1 brain MRI, gadolinium-enhanced brain MRI, "
        "T1 plus contrast brain MRI, T1+C brain MRI, enhancing lesion on T1 MRI."
    ),
    "mrt_prostata_t1": (
        "Prostate MRI T1-weighted image, prostate MR T1 sequence, pelvic MRI of the prostate T1, "
        "axial T1 prostate MRI, prostate gland MRI T1-weighted."
    ),
    "mrt_prostata_t2": (
        "Prostate MRI T2-weighted image, prostate MR T2 sequence, pelvic MRI of the prostate T2, "
        "axial T2 prostate MRI, zonal anatomy of the prostate on T2-weighted MRI."
    ),
    "ct": (
        "Computed tomography, CT scan, computed tomographic image, axial CT, coronal CT, sagittal CT, "
        "contrast-enhanced CT, non-contrast CT, helical CT, multidetector CT, MDCT, HRCT, "
        "brain CT, chest CT, abdominal CT, pelvic CT, Hounsfield units."
    ),
    "ct_kombimodalitaet_spect+ct_pet+ct": (
        "Hybrid CT imaging with PET or SPECT, PET/CT, PET-CT, fused PET-CT image, hybrid PET/CT imaging, "
        "co-registered PET and CT, combined positron emission tomography and computed tomography, "
        "SPECT/CT, SPECT-CT, fused SPECT-CT image, hybrid single-photon emission computed tomography and CT, "
        "nuclear medicine hybrid imaging, tracer uptake co-registered with CT anatomy."
    ),
}


# =========================================================
# Regeln
# OHNE generische Ausgabeklasse "mrt"
# =========================================================

def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def contains_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


RULES: List[Tuple[str, List[str]]] = [
    (
        "mrt_hirn_t1_c",
        [
            r"\bbrain\b.*\bt1\+c\b",
            r"\bcerebr\w*\b.*\bt1\+c\b",
            r"\bpost-contrast\b.*\bbrain\b.*\bmri\b",
            r"\bgadolinium-enhanced\b.*\bbrain\b.*\bmri\b",
            r"\bcontrast-enhanced\b.*\bbrain\b.*\bmri\b",
            r"\bt1\+c\b.*\bbrain\b",
            r"\bt1 with contrast\b.*\bbrain\b",
        ],
    ),
    (
        "mrt_hirn_flair",
        [
            r"\bbrain\b.*\bflair\b",
            r"\bcranial\b.*\bflair\b",
            r"\bcerebr\w*\b.*\bflair\b",
            r"\bhead\b.*\bflair\b",
            r"\bflair\b.*\bbrain\b",
            r"\bfluid-attenuated inversion recovery\b",
        ],
    ),
    (
        "mrt_hirn_t1",
        [
            r"\bbrain\b.*\bt1\b",
            r"\bhead\b.*\bt1\b",
            r"\bcerebr\w*\b.*\bt1\b",
            r"\bcranial\b.*\bt1\b",
            r"\bt1-weighted\b.*\bbrain\b",
            r"\bbrain mri\b.*\bt1\b",
        ],
    ),
    (
        "mrt_hirn_t2",
        [
            r"\bbrain\b.*\bt2\b",
            r"\bhead\b.*\bt2\b",
            r"\bcerebr\w*\b.*\bt2\b",
            r"\bcranial\b.*\bt2\b",
            r"\bt2-weighted\b.*\bbrain\b",
            r"\bbrain mri\b.*\bt2\b",
        ],
    ),
    (
        "mrt_prostata_t2",
        [
            r"\bprostat\w*\b.*\bt2\b",
            r"\bt2-weighted\b.*\bprostat\w*\b",
            r"\bprostate mri\b.*\bt2\b",
        ],
    ),
    (
        "mrt_prostata_t1",
        [
            r"\bprostat\w*\b.*\bt1\b",
            r"\bt1-weighted\b.*\bprostat\w*\b",
            r"\bprostate mri\b.*\bt1\b",
        ],
    ),
    (
        "ct_kombimodalitaet_spect+ct_pet+ct",
        [
            r"\bpet\s*/\s*ct\b",
            r"\bpet-ct\b",
            r"\bspect\s*/\s*ct\b",
            r"\bspect-ct\b",
            r"\bhybrid pet\s*/\s*ct\b",
            r"\bhybrid spect\s*/\s*ct\b",
            r"\bfused pet-ct\b",
            r"\bfused spect-ct\b",
            r"\bco-registered pet and ct\b",
            r"\bcombined positron emission tomography and computed tomography\b",
            r"\bcombined single[- ]photon emission computed tomography and ct\b",
        ],
    ),
    (
        "xray_fluoroskopie_angiographie",
        [
            r"\bangioplast\w*\b",
            r"\bballoon angioplasty\b",
            r"\bpercutaneous transluminal angioplasty\b",
            r"\bpta\b",
            r"\bptca\b",
            r"\bcoronary angioplasty\b",
            r"\bstent placement\b",
            r"\bpercutaneous coronary intervention\b",
            r"\bpci\b",
            r"\bfluoroscopy\b",
            r"\bfluoroscopic\b",
            r"\bc-arm\b",
            r"\bx-ray guided\b",
            r"\bfluoroscopic guidance\b",
            r"\breal-time x-ray\b",
            r"\bangiograph\w*\b",
            r"\bdsa\b",
            r"\bcatheter angiography\b",
        ],
    ),
    (
        "ct",
        [
            r"\bct\b",
            r"\bct scan\b",
            r"\bcomputed tomography\b",
            r"\bcomputed tomograph\w*\b",
            r"\baxial ct\b",
            r"\bcoronal ct\b",
            r"\bsagittal ct\b",
            r"\bcontrast-enhanced ct\b",
            r"\bnon-contrast ct\b",
            r"\bhelical ct\b",
            r"\bmdct\b",
            r"\bhrct\b",
        ],
    ),
    (
        "us",
        [
            r"\bultrasound\b",
            r"\bsonograph\w*\b",
            r"\bultrasonograph\w*\b",
            r"\bechograph\w*\b",
            r"\bechocardiograph\w*\b",
            r"\bdoppler\b",
            r"\bduplex sonograph\w*\b",
            r"\bendoscopic ultrasound\b",
            r"\beus\b",
            r"\bb-mode ultrasound\b",
            r"\bcolor doppler\b",
            r"\bpower doppler\b",
        ],
    ),
    (
        "xray",
        [
            r"\bx-ray\b",
            r"\bxray\b",
            r"\bradiograph\b",
            r"\bradiographic\b",
            r"\bprojection radiography\b",
            r"\bap view\b",
            r"\bpa view\b",
            r"\blateral view\b",
            r"\bportable x-ray\b",
            r"\bpanoramic radiograph\b",
            r"\bdorsoplantar projection\b",
            r"\bplain film\b",
        ],
    ),
]

# generische MRT-Evidenz nur intern, NICHT als Klasse
GENERIC_MRT_PATTERNS = [
    r"\bmri\b",
    r"\bmr image\b",
    r"\bmr images\b",
    r"\bmr scan\b",
    r"\bmagnetic resonance\b",
    r"\bt1[- ]weighted\b",
    r"\bt2[- ]weighted\b",
    r"\bflair\b",
    r"\bdwi\b",
    r"\bdiffusion-weighted\b",
    r"\bfmri\b",
    r"\bt1\+c\b",
    r"\bgadolinium-enhanced\b",
]


def rule_based_classify(caption: str) -> Tuple[Optional[str], bool]:
    text = normalize_text(caption)

    for label, patterns in RULES:
        if contains_any(text, patterns):
            return label, False

    # nur generisches MRT erkannt, aber keine konkrete Unterklasse
    if contains_any(text, GENERIC_MRT_PATTERNS):
        return None, True

    return None, False


# =========================================================
# Captions laden
# =========================================================

def parse_captions_file(file_path: Path, split_name: str, domain_name: str, images_dir: Path) -> pd.DataFrame:
    rows = []

    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split("\t", 1)
            if len(parts) != 2:
                print(f"[WARN] Unerwartetes Format in {file_path} Zeile {line_number}: {line!r}")
                continue

            roco_id = parts[0].strip()
            caption = parts[1].strip()

            rows.append({
                "split": split_name,
                "domain": domain_name,
                "id": roco_id,
                "caption": caption,
                "image_path": str(images_dir / f"{roco_id}.jpg"),
            })

    return pd.DataFrame(rows)


def load_all_captions(base_dir: Path) -> pd.DataFrame:
    splits = ["train", "test", "validation"]
    domains = ["radiology", "non-radiology"]
    dfs = []

    for split in splits:
        for domain in domains:
            subset_dir = base_dir / split / domain
            captions_file = subset_dir / "captions.txt"
            images_dir = subset_dir / "images"

            if not captions_file.exists():
                print(f"[WARN] Datei fehlt: {captions_file}")
                continue
            if not images_dir.exists():
                print(f"[WARN] Bilderordner fehlt: {images_dir}")
                continue

            print(f"[INFO] Lese Captions aus: {captions_file}")
            dfs.append(parse_captions_file(captions_file, split, domain, images_dir))

    if not dfs:
        return pd.DataFrame(columns=["split", "domain", "id", "caption", "image_path"])

    return pd.concat(dfs, ignore_index=True)


# =========================================================
# BERT
# =========================================================

def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def encode_texts(
    texts: List[str],
    tokenizer: BertTokenizer,
    model: BertModel,
    batch_size: int = 32,
    max_length: int = 128,
    device: str = "cpu",
) -> torch.Tensor:
    all_embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded)

        pooled = mean_pooling(outputs.last_hidden_state, encoded["attention_mask"])
        pooled = F.normalize(pooled, p=2, dim=1)
        all_embeddings.append(pooled.cpu())

        end = min(start + batch_size, len(texts))
        print(f"[INFO] Encoded {end}/{len(texts)} Texte")

    return torch.cat(all_embeddings, dim=0)


@torch.no_grad()
def classify_bert_with_scores(
    texts: List[str],
    tokenizer: BertTokenizer,
    model: BertModel,
    label_to_embedding: Dict[str, torch.Tensor],
    candidate_labels: List[str],
    batch_size: int,
    max_length: int,
    device: str,
    min_margin: float,
    min_top1_sim: float,
) -> List[Dict[str, Any]]:
    if not texts:
        return []

    text_embs = encode_texts(
        texts=texts,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
    )

    class_embs = torch.stack([label_to_embedding[label] for label in candidate_labels], dim=0)
    sim_matrix = text_embs @ class_embs.T

    outputs = []
    for row in sim_matrix:
        values, indices = torch.topk(row, k=min(2, len(candidate_labels)))
        top1_idx = indices[0].item()
        top1_val = values[0].item()

        if len(values) > 1:
            top2_idx = indices[1].item()
            top2_val = values[1].item()
            margin = top1_val - top2_val
            top2_label = candidate_labels[top2_idx]
        else:
            top2_val = None
            margin = 999.0
            top2_label = None

        top1_label = candidate_labels[top1_idx]
        accepted = (top1_val >= min_top1_sim) and (margin >= min_margin)

        outputs.append({
            "label": top1_label if accepted else None,
            "top1_label": top1_label,
            "top1_score": top1_val,
            "top2_label": top2_label,
            "top2_score": top2_val,
            "margin": margin,
            "accepted": accepted,
        })

    return outputs


# =========================================================
# CNN
# =========================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int):
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


class SimpleCNNWrapper:
    def __init__(self, class_names: List[str], device: str = "cpu"):
        self.class_names = class_names
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.available = False
        self.model = None

    def load(self, model_path: Path):
        if not model_path.exists():
            print(f"[WARN] CNN-Modell nicht gefunden: {model_path}")
            self.available = False
            return

        self.model = SimpleCNN(num_classes=len(self.class_names))
        state_dict = torch.load(model_path, map_location=self.device)

        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as e:
            print("[ERROR] CNN-Gewichte konnten nicht geladen werden.")
            print(f"[ERROR] Erwartete Klassenanzahl: {len(self.class_names)}")
            raise e

        self.model.to(self.device)
        self.model.eval()
        self.available = True
        print(f"[INFO] CNN-Modell geladen: {model_path}")

    @torch.no_grad()
    def predict_one(self, image_path: str) -> Dict[str, Any]:
        if not self.available:
            return {
                "label": None,
                "top1_label": None,
                "top1_prob": None,
                "top2_label": None,
                "top2_prob": None,
                "margin": None,
                "accepted": False,
                "all_probs": {},
            }

        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = F.softmax(logits, dim=1)[0].detach().cpu()

        top_vals, top_idxs = torch.topk(probs, k=min(2, len(self.class_names)))
        top1_idx = top_idxs[0].item()
        top1_prob = top_vals[0].item()
        top1_label = self.class_names[top1_idx]

        if len(top_vals) > 1:
            top2_idx = top_idxs[1].item()
            top2_prob = top_vals[1].item()
            top2_label = self.class_names[top2_idx]
            margin = top1_prob - top2_prob
        else:
            top2_prob = None
            top2_label = None
            margin = 999.0

        accepted = top1_prob >= CNN_MIN_CONF

        return {
            "label": top1_label if accepted else None,
            "top1_label": top1_label,
            "top1_prob": top1_prob,
            "top2_label": top2_label,
            "top2_prob": top2_prob,
            "margin": margin,
            "accepted": accepted,
            "all_probs": {cls: probs[i].item() for i, cls in enumerate(self.class_names)},
        }


# =========================================================
# Fusion
# OHNE mrt-Oberklasse
# =========================================================

def build_vote_scores(
    rule_label: Optional[str],
    rule_generic_mrt: bool,
    bert_label: Optional[str],
    bert_score: Optional[float],
    cnn_label: Optional[str],
    cnn_score: Optional[float],
) -> Dict[str, float]:
    scores = defaultdict(float)

    if rule_label is not None:
        scores[rule_label] += 2.5
        parent = get_parent_or_self(rule_label)
        if parent != rule_label:
            scores[parent] += 0.8

    if bert_label is not None and bert_score is not None:
        scores[bert_label] += 1.2 + max(0.0, bert_score)
        parent = get_parent_or_self(bert_label)
        if parent != bert_label:
            scores[parent] += 0.5 + 0.3 * bert_score

    if cnn_label is not None and cnn_score is not None:
        scores[cnn_label] += 1.2 + max(0.0, cnn_score)
        parent = get_parent_or_self(cnn_label)
        if parent != cnn_label:
            scores[parent] += 0.5 + 0.3 * cnn_score

    # generische MRT-Evidenz gibt nur schwache Unterstützung
    # an alle MRT-Unterklassen, NICHT an eine mrt-Klasse
    if rule_generic_mrt:
        mrt_leafs = [
            "mrt_hirn_flair",
            "mrt_hirn_t1",
            "mrt_hirn_t2",
            "mrt_hirn_t1_c",
            "mrt_prostata_t1",
            "mrt_prostata_t2",
        ]
        for cls in mrt_leafs:
            scores[cls] += 0.15

    return dict(scores)


def choose_best_label(scores: Dict[str, float]) -> Tuple[Optional[str], float]:
    if not scores:
        return None, 0.0
    best_label, best_score = sorted(scores.items(), key=lambda x: x[1], reverse=True)[0]
    return best_label, best_score


# =========================================================
# Ausgabe
# =========================================================

def write_txt_output(df: pd.DataFrame, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(f"{row['id']}\t{row['final_class']}\n")


# =========================================================
# Hauptprogramm
# =========================================================

def main():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Lade Daten aus: {BASE_DIR}")

    df = load_all_captions(BASE_DIR)

    if df.empty:
        print("[ERROR] Keine Captions gefunden.")
        return

    if DEBUG_LIMIT is not None:
        df = df.head(DEBUG_LIMIT).copy()
        print(f"[INFO] DEBUG_LIMIT aktiv: {len(df)} Zeilen")

    print(f"[INFO] Gesamtzahl Captions: {len(df)}")

    if not LOCAL_BIOMEDBERT_PATH.exists():
        raise FileNotFoundError(f"Modellpfad nicht gefunden: {LOCAL_BIOMEDBERT_PATH}")

    print(f"[INFO] Lade lokales BiomedBERT aus: {LOCAL_BIOMEDBERT_PATH}")
    tokenizer = BertTokenizer.from_pretrained(str(LOCAL_BIOMEDBERT_PATH), local_files_only=True)
    model = BertModel.from_pretrained(str(LOCAL_BIOMEDBERT_PATH), local_files_only=True)
    model.to(DEVICE)
    model.eval()

    print("[INFO] Encode Klassenbeschreibungen ...")
    class_names_for_embeddings = list(CLASS_TEXTS.keys())
    class_texts_for_embeddings = [CLASS_TEXTS[c] for c in class_names_for_embeddings]

    class_emb_tensor = encode_texts(
        texts=class_texts_for_embeddings,
        tokenizer=tokenizer,
        model=model,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        device=DEVICE,
    )

    label_to_embedding = {
        label: class_emb_tensor[i]
        for i, label in enumerate(class_names_for_embeddings)
    }

    # CNN
    cnn = SimpleCNNWrapper(class_names=CNN_CLASSES, device=DEVICE)
    cnn.load(LOCAL_CNN_MODEL_PATH)

    # Regeln
    rule_results = df["caption"].apply(rule_based_classify)
    df["rule_label"] = [x[0] for x in rule_results]
    df["rule_generic_mrt"] = [x[1] for x in rule_results]

    # BERT flach ueber alle Ausgabeklassen
    print("[INFO] BERT-Klassifikation ...")
    bert_results = classify_bert_with_scores(
        texts=df["caption"].fillna("").astype(str).tolist(),
        tokenizer=tokenizer,
        model=model,
        label_to_embedding=label_to_embedding,
        candidate_labels=ALL_CLASSES,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        device=DEVICE,
        min_margin=MIN_MARGIN,
        min_top1_sim=MIN_TOP1_SIM,
    )

    df["bert_label"] = [x["label"] for x in bert_results]
    df["bert_top1"] = [x["top1_label"] for x in bert_results]
    df["bert_score"] = [x["top1_score"] for x in bert_results]
    df["bert_margin"] = [x["margin"] for x in bert_results]
    df["bert_accepted"] = [x["accepted"] for x in bert_results]

    # CNN
    print("[INFO] CNN-Klassifikation ...")
    cnn_results = []
    for i, image_path in enumerate(df["image_path"].tolist(), start=1):
        result = cnn.predict_one(image_path)
        cnn_results.append(result)
        if i % 50 == 0 or i == len(df):
            print(f"[INFO] CNN verarbeitet: {i}/{len(df)}")

    df["cnn_label"] = [x["label"] for x in cnn_results]
    df["cnn_top1"] = [x["top1_label"] for x in cnn_results]
    df["cnn_score"] = [x["top1_prob"] for x in cnn_results]
    df["cnn_margin"] = [x["margin"] for x in cnn_results]
    df["cnn_accepted"] = [x["accepted"] for x in cnn_results]

    # Fusion
    final_classes = []
    final_scores = []
    final_reasons = []

    for _, row in df.iterrows():
        scores = build_vote_scores(
            rule_label=row["rule_label"],
            rule_generic_mrt=bool(row["rule_generic_mrt"]),
            bert_label=row["bert_label"],
            bert_score=row["bert_score"],
            cnn_label=row["cnn_label"],
            cnn_score=row["cnn_score"],
        )

        final_label, final_score = choose_best_label(scores)

        if final_label is None:
            if USE_UNCLEAR_LABEL:
                final_label = UNCLEAR_LABEL
                final_reason = "no_evidence"
            else:
                final_reason = "no_evidence"
        else:
            if final_score < FUSION_MIN_SCORE:
                if USE_UNCLEAR_LABEL:
                    final_label = UNCLEAR_LABEL
                    final_reason = "low_fusion_score"
                else:
                    final_reason = "low_fusion_score"
            else:
                final_reason = "fused"

        # wichtiger Spezialfall:
        # Caption sagt nur allgemein MRI, aber keine konkrete Unterklasse,
        # und weder BERT noch CNN stützen eine bestimmte MRI-Unterklasse stark genug.
        if row["rule_generic_mrt"] and row["rule_label"] is None:
            if final_label in {
                "mrt_hirn_flair",
                "mrt_hirn_t1",
                "mrt_hirn_t2",
                "mrt_hirn_t1_c",
                "mrt_prostata_t1",
                "mrt_prostata_t2",
            } and final_score < 1.8:
                if USE_UNCLEAR_LABEL:
                    final_label = UNCLEAR_LABEL
                    final_reason = "generic_mri_only"

        final_classes.append(final_label)
        final_scores.append(final_score)
        final_reasons.append(final_reason)

    df["final_class"] = final_classes
    df["final_score"] = final_scores
    df["final_reason"] = final_reasons

    if USE_UNCLEAR_LABEL:
        df["final_class"] = df["final_class"].fillna(UNCLEAR_LABEL)

    print("\n[INFO] Erste 30 Ergebnisse:")
    preview_cols = [
        "id",
        "caption",
        "rule_label",
        "rule_generic_mrt",
        "bert_label",
        "bert_score",
        "cnn_label",
        "cnn_score",
        "final_class",
        "final_score",
        "final_reason",
    ]
    print(df[preview_cols].head(30).to_string(index=False))

    print("\n[INFO] Klassenverteilung final:")
    print(df["final_class"].value_counts(dropna=False).to_string())

    print(f"\n[INFO] Schreibe TXT: {OUTPUT_TXT}")
    write_txt_output(df, OUTPUT_TXT)

    print(f"[INFO] Schreibe CSV: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)

    print("[INFO] Fertig.")
    print(f"[INFO] TXT-Datei: {OUTPUT_TXT.resolve()}")
    print(f"[INFO] CSV-Datei: {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()