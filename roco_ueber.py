from pathlib import Path
import pandas as pd
import os
import os.path as osp

# =========================
# Konfiguration
# =========================
OPERATING_PATH  = osp.normpath(osp.join(osp.dirname(__file__), "data"))
BASE_DIR = Path(OPERATING_PATH)
HEAD_N = 100                       # zwischen 20 und 200 setzen


# =========================
# Hilfsfunktionen
# =========================
def parse_single_text_field(file_path: Path, value_name: str) -> pd.DataFrame:
    """
    Liest Dateien wie captions.txt ein:
    ROCO_81826\tModel bone showing the extent of graft insertion

    Gibt DataFrame mit Spalten:
    - id
    - <value_name>
    """
    rows = []

    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split("\t", 1)
            if len(parts) < 2:
                print(f"[WARN] Unerwartetes Format in {file_path} Zeile {line_number}: {line!r}")
                continue

            roco_id = parts[0].strip()
            value = parts[1].strip()

            rows.append({
                "id": roco_id,
                value_name: value
            })

    return pd.DataFrame(rows)


def parse_multi_value_field(file_path: Path, value_name: str) -> pd.DataFrame:
    """
    Liest Dateien wie cuis.txt, keywords.txt, semtypes.txt ein:
    ROCO_81826\t\tC0035139\tC0441587\t...
    oder
    ROCO_81826\tmodel\tinsertion\textent\tbone\tgraft
    oder
    ROCO_81826\tT061\tTherapeutic or Preventive Procedure\t...

    Gibt DataFrame mit Spalten:
    - id
    - <value_name>         -> zusammenhängender String
    - <value_name>_list    -> Liste der Tokens
    """
    rows = []

    with file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue

            parts = line.split("\t")
            if len(parts) < 1:
                print(f"[WARN] Unerwartetes Format in {file_path} Zeile {line_number}: {line!r}")
                continue

            roco_id = parts[0].strip()
            values = [p.strip() for p in parts[1:] if p.strip()]

            rows.append({
                "id": roco_id,
                value_name: " | ".join(values),
                f"{value_name}_list": values
            })

    return pd.DataFrame(rows)


def build_subset_dataframe(subset_dir: Path, split_name: str, domain_name: str) -> pd.DataFrame:
    """
    Baut DataFrame für genau einen Ordner wie:
    /Project/data/train/radiology
    """
    images_dir = subset_dir / "images"
    captions_file = subset_dir / "captions.txt"
    cuis_file = subset_dir / "cuis.txt"
    keywords_file = subset_dir / "keywords.txt"
    semtypes_file = subset_dir / "semtypes.txt"

    required_files = [captions_file, cuis_file, keywords_file, semtypes_file, images_dir]
    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(f"Fehlt: {path}")

    df_captions = parse_single_text_field(captions_file, "caption")
    df_cuis = parse_multi_value_field(cuis_file, "cuis")
    df_keywords = parse_multi_value_field(keywords_file, "keywords")
    df_semtypes = parse_multi_value_field(semtypes_file, "semtypes")

    df = df_captions.merge(df_cuis, on="id", how="outer")
    df = df.merge(df_keywords, on="id", how="outer")
    df = df.merge(df_semtypes, on="id", how="outer")

    df["split"] = split_name
    df["domain"] = domain_name
    df["image_path"] = df["id"].apply(lambda x: str(images_dir / f"{x}.jpg"))
    df["image_exists"] = df["image_path"].apply(lambda x: Path(x).exists())

    preferred_order = [
        "split",
        "domain",
        "id",
        "image_path",
        "image_exists",
        "caption",
        "cuis",
        "cuis_list",
        "keywords",
        "keywords_list",
        "semtypes",
        "semtypes_list",
    ]

    existing_cols = [c for c in preferred_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + remaining_cols]

    return df


def build_full_dataframe(base_dir: Path) -> pd.DataFrame:
    """
    Läuft über:
    - train/test/validation
    - radiology/non-radiology
    und baut einen Gesamt-DataFrame.
    """
    splits = ["train", "test", "validation"]
    domains = ["radiology", "non-radiology"]

    dfs = []

    for split in splits:
        for domain in domains:
            subset_dir = base_dir / split / domain
            print(f"[INFO] Verarbeite: {subset_dir}")
            df_subset = build_subset_dataframe(subset_dir, split, domain)
            dfs.append(df_subset)

    if not dfs:
        return pd.DataFrame()

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


# =========================
# Hauptteil
# =========================
def main():
    df = build_full_dataframe(BASE_DIR)

    print("\n[INFO] DataFrame erstellt.")
    print(f"[INFO] Anzahl Zeilen: {len(df)}")
    print(f"[INFO] Anzahl Spalten: {len(df.columns)}")

    print("\n[INFO] Spalten:")
    print(df.columns.tolist())

    print(f"\n[INFO] Erste {HEAD_N} Zeilen:")
    print(df.head(HEAD_N).to_string())

    # Optional:
    # df.to_csv("roco_v1_merged.csv", index=False)
    # df.to_parquet("roco_v1_merged.parquet", index=False)


if __name__ == "__main__":
    main()