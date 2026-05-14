# ============================================================
# MERGED_ROCO + EINEN WEITEREN CHECKPOINT MERGEN
# ============================================================

from pathlib import Path
import pandas as pd
from collections import defaultdict
from mk_nonoverlapping_class_sets import iterative_merge, print_stats, copy_images, build_item_id, CLASSES

# Bereits gemergter Datensatz
merged_csv = Path(
    "/home/b/Dokumente/ROCO-checkpoints_merged9/merged_roco.csv"
)

# Neuer Checkpoint
new_checkpoint_csv = Path(
    "/home/b/Dokumente/ROCO-checkpoints_log10/round_checkpoints/output_roco.csv"
)

print("Lade merged Datensatz...")
df_old = pd.read_csv(merged_csv)

print(df_old["final_label"].value_counts())

print("\nLade neuen Checkpoint...")
df_new = pd.read_csv(new_checkpoint_csv)

print(df_new["final_label"].value_counts())

# ============================================================
# BESTEHENDE IDs
# ============================================================

existing_ids = {}
per_class_counter = defaultdict(int)

for _, row in df_old.iterrows():

    item_id = build_item_id(row, mode="roco")

    label = row["final_label"]

    existing_ids[item_id] = label

    per_class_counter[label] += 1

# ============================================================
# ITERATIV ERGAENZEN
# ============================================================

added_rows = []

conflicts = 0
duplicates = 0
added = 0

MAX_PER_CLASS = 5000

print("\nFuelle Klassen iterativ auf...")

for _, row in df_new.iterrows():

    label = row["final_label"]

    if label not in CLASSES:
        continue

    # Klasse bereits voll?
    if per_class_counter[label] >= MAX_PER_CLASS:
        continue

    item_id = build_item_id(row, mode="roco")

    # ========================================================
    # ID bereits vorhanden
    # ========================================================

    if item_id in existing_ids:

        old_label = existing_ids[item_id]

        # gleicher Eintrag
        if old_label == label:

            duplicates += 1

            continue

        # Konflikt
        else:

            conflicts += 1

            continue

    # ========================================================
    # ECHTES NEUITEM
    # ========================================================

    added_rows.append(row)

    existing_ids[item_id] = label

    per_class_counter[label] += 1

    added += 1

# ============================================================
# FINAL
# ============================================================

if added_rows:

    df_added = pd.DataFrame(added_rows)

    final_df = pd.concat(
        [df_old, df_added],
        ignore_index=True
    )

else:

    final_df = df_old

# ============================================================
# SAVE
# ============================================================

output_dir = Path(
    "/home/b/Dokumente/ROCO-checkpoints_merged10"
)

output_dir.mkdir(parents=True, exist_ok=True)

out_csv = output_dir / "merged_roco.csv"

final_df.to_csv(out_csv, index=False)

print(f"\nGespeichert:")
print(out_csv)

# ============================================================
# BILDER KOPIEREN
# ============================================================

copy_images(
    final_df,
    output_dir
)

# ============================================================
# STATS
# ============================================================

print("\n==============================")
print("MERGE STATISTIK")
print("==============================")

print(f"Neu hinzugefuegt: {added}")
print(f"Doppelte ignoriert: {duplicates}")
print(f"Konflikte ignoriert: {conflicts}")

print("\nFinale Klassenverteilung:")
print(final_df["final_label"].value_counts())