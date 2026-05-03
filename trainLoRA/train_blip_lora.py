import argparse
from pathlib import Path
import os.path as osp
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForConditionalGeneration
from peft import LoraConfig, get_peft_model

PATH = osp.normpath(osp.dirname(__file__))
BASE_DIR = Path(PATH)
# trainLoRA > project > user *this* >  / Dokumente / blipbase or
# LOCAL_BLIP_PATH = Path("/home/user/Dokumente/biomedbert")
LOCAL_BLIP_PATH = BASE_DIR.parent.parent.parent / "Dokumente" / "blipbase"
print("LOCAL_BLIP_PATH", LOCAL_BLIP_PATH)
print("EXISTS:", LOCAL_BLIP_PATH.exists())

class ImageCaptionDataset(Dataset):
    def __init__(self, csv_path, processor, modality=None, max_length=64):
        self.df = pd.read_csv(csv_path)

        if modality is not None:
            self.df = self.df[self.df["final_class"].astype(str).str.lower() == modality.lower()]

        self.df = self.df.reset_index(drop=True)
        self.processor = processor
        self.max_length = max_length

        if len(self.df) == 0:
            raise ValueError(f"Keine Daten für Modalität gefunden: {modality}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        caption = str(row["caption"])

        inputs = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in inputs.items()}

        labels = item["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        item["labels"] = labels

        return item


def freeze_vision_encoder(model):
    """
    Vision Encoder einfrieren.
    Dadurch wird nur der Text-/Decoder-Teil per LoRA angepasst.
    """
    if hasattr(model, "vision_model"):
        for param in model.vision_model.parameters():
            param.requires_grad = False
    else:
        print("WARNUNG: vision_model nicht gefunden.")


def find_lora_targets(model, mode="qv"):
    """
    Sucht automatisch passende Attention-Layer im BLIP-Textdecoder.

    Wichtig:
    BLIP benutzt in HuggingFace häufig:
        query, key, value

    Andere Modelle benutzen oft:
        q_proj, k_proj, v_proj

    Deshalb prüfen wir beides.
    """

    if mode == "qv":
        allowed = ("query", "value", "q_proj", "v_proj")
    elif mode == "qkv":
        allowed = ("query", "key", "value", "q_proj", "k_proj", "v_proj")
    else:
        raise ValueError("mode muss 'qv' oder 'qkv' sein")

    targets = []

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        # Vision Encoder nicht mit LoRA patchen
        if name.startswith("vision_model"):
            continue

        # Nur Attention-Projektionen
        last = name.split(".")[-1]
        if last in allowed:
            targets.append(name)

    if len(targets) == 0:
        print("\nKeine LoRA-Ziele gefunden. Verfügbare Linear-Layer:")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(name)
        raise RuntimeError("Keine passenden Q/K/V Layer gefunden.")

    print("\nGefundene LoRA-Zielmodule:")
    for t in targets:
        print("  ", t)

    return targets


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--modality", type=str, required=True)

    parser.add_argument(
        "--model_name",
        type=str,
        default="Salesforce/blip-image-captioning-base",
    )

    parser.add_argument(
        "--lora_mode",
        type=str,
        choices=["qv", "qkv"],
        default="qv",
    )

    parser.add_argument("--output_dir", type=str, default="../LoRAs/")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_length", type=int, default=64)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    args = parser.parse_args()

    device = "cpu"

    # processor = BlipProcessor.from_pretrained(args.model_name)
    # model = BlipForConditionalGeneration.from_pretrained(args.model_name)
    model = BlipForConditionalGeneration.from_pretrained(LOCAL_BLIP_PATH)
    processor = BlipProcessor.from_pretrained(LOCAL_BLIP_PATH)
    freeze_vision_encoder(model)

    target_modules = find_lora_targets(model, mode=args.lora_mode)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.to(device)
    model.train()

    dataset = ImageCaptionDataset(
        csv_path=args.csv,
        processor=processor,
        modality=args.modality,
        max_length=args.max_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cpu"))

    for epoch in range(args.epochs):
        total_loss = 0.0

        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for batch in progress:
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device == "cpu")):
                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}: avg_loss = {avg_loss:.4f}")

    save_dir = (Path(args.output_dir) / f"{args.lora_mode}_{args.modality}").resolve()
    print("ABSOLUTER SPEICHERPFAD:", save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    print(f"\nLoRA gespeichert unter: {save_dir}")


if __name__ == "__main__":
    main()