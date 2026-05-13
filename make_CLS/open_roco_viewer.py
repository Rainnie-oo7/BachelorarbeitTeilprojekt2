# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from PIL import Image, ImageTk
from pathlib import Path

import pandas as pd
import json
import threading

# ============================================================
# IMAGE INDEX
# ============================================================

def build_image_index(image_dir):

    image_index = {}

    image_dir = Path(image_dir)

    for fp in image_dir.rglob("*"):

        if not fp.is_file():
            continue

        if fp.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        image_index[fp.name] = fp

    print(f"Indexed images: {len(image_index)}")

    return image_index


# ============================================================
# CONFIG
# ============================================================

MODE = "disagreements"

"""
MODES:

"final"
→ finale Round-Checkpoints

"disagreements"
→ nur Rule/CNN Konflikte

"early_rules"
→ reine Rule-basierte Early-Samples

"all"
→ alles kombiniert
"""

BASE_DIR = Path(
    "/home/b/PycharmProjects/ba2roco"
)

# ============================================================
# FINAL DATASET
# ============================================================

if MODE == "final":

    CSV_PATH = (
        BASE_DIR
        / "round_checkpoints"
        / "round_000.csv"
    )

    IMAGE_DIR = (
        BASE_DIR
        / "round_checkpoints"
        / "images_after_rounds"
    )

# ============================================================
# DISAGREEMENTS
# ============================================================

elif MODE == "disagreements":

    CSV_PATH = (
        BASE_DIR
        / "disagreements"
        / "rule_cnn_disagreements.csv"
    )

    IMAGE_DIR = (
        BASE_DIR
        / "disagreements"
        / "images"
    )

# ============================================================
# EARLY RULES
# ============================================================

elif MODE == "early_rules":

    CSV_PATH = (
        BASE_DIR
        / "early_rules_dataset"
        / "early_rules.csv"
    )

    IMAGE_DIR = (
        BASE_DIR
        / "early_rules_dataset"
        / "images"
    )

# ============================================================
# ALL
# ============================================================

elif MODE == "all":

    FINAL_CSV = (
        BASE_DIR
        / "round_checkpoints"
        / "round_005.csv"
    )

    DIS_CSV = (
        BASE_DIR
        / "disagreements"
        / "rule_cnn_disagreements.csv"
    )

    FINAL_IMG_DIR = (
        BASE_DIR
        / "round_checkpoints"
        / "images_after_rounds"
    )

    DIS_IMG_DIR = (
        BASE_DIR
        / "disagreements"
        / "images"
    )

    CSV_PATH = None
    IMAGE_DIR = None

else:

    raise ValueError(f"Unknown MODE: {MODE}")

print("\nCSV_PATH:")
print(CSV_PATH)

print("\nIMAGE_DIR:")
print(IMAGE_DIR)

# ============================================================
# IMAGE INDEX
# ============================================================

if MODE == "all":

    IMAGE_INDEX = {}

    IMAGE_INDEX.update(
        build_image_index(FINAL_IMG_DIR)
    )

    IMAGE_INDEX.update(
        build_image_index(DIS_IMG_DIR)
    )

else:

    IMAGE_INDEX = build_image_index(IMAGE_DIR)

# ============================================================
# UI CONFIG
# ============================================================

THUMB_SIZE = (900, 900)

BG = "#1e1e1e"
FG = "#dddddd"

FONT = ("Consolas", 10)

# ============================================================
# ASYNC IMAGE LOADER
# ============================================================

class AsyncImageLoader:

    def __init__(self):

        self.cache = {}
        self.lock = threading.Lock()

    def preload(self, paths):

        for p in paths:

            if p in self.cache:
                continue

            threading.Thread(
                target=self._worker,
                args=(p,),
                daemon=True
            ).start()

    def _worker(self, path):

        try:

            img = Image.open(path).convert("RGB")

            img.thumbnail(THUMB_SIZE)

            with self.lock:
                self.cache[path] = img

        except Exception:

            with self.lock:
                self.cache[path] = None

    def get(self, path):

        if path not in self.cache:
            self._worker(path)

        return self.cache.get(path)


# ============================================================
# BUILD IMAGE PATH
# ============================================================

def build_image_path(image_dir, row):

    label = row.get("final_label")

    if pd.isna(label) or not label:
        label = row.get("rule_pred", "unknown")

    label = str(label)

    # ============================================================
    # MODE=ALL
    # ============================================================

    if MODE == "all":

        source = row.get("viewer_source", "final")

        if source == "disagreement":
            base_dir = DIS_IMG_DIR
        else:
            base_dir = FINAL_IMG_DIR

    else:

        base_dir = image_dir

    # ============================================================
    # ROCO / PMC ID
    # ============================================================

    pmc_id = row.get("pmc_id")

    if pmc_id is None or pd.isna(pmc_id):
        return None

    pmc_id = str(pmc_id).strip()

    # ============================================================
    # SEARCH DIRECTORY
    # ============================================================

    class_dir = Path(base_dir) / label

    if not class_dir.exists():
        return None

    # ============================================================
    # Suche:
    # ROCO_22771_*.jpg
    # ============================================================

    exts = [".jpg", ".jpeg", ".png"]

    for fp in class_dir.iterdir():

        if not fp.is_file():
            continue

        if fp.suffix.lower() not in exts:
            continue

        if fp.stem.startswith(pmc_id):
            return fp

    # ============================================================
    # FALLBACK GLOBAL INDEX
    # ============================================================

    for name, fp in IMAGE_INDEX.items():

        stem = Path(name).stem

        if stem.startswith(pmc_id):
            return fp

    return None

# ============================================================
# MAIN VIEWER
# ============================================================

class ROCOViewer:

    def __init__(self, root):

        self.root = root

        self.root.title(f"ROCO DEBUG VIEWER [{MODE}]")

        self.root.geometry("1800x1000")

        self.root.configure(bg=BG)

        print("Lade CSV ...")

        # ============================================================
        # CSV LOAD
        # ============================================================

        if MODE == "all":

            df_final = pd.read_csv(FINAL_CSV, low_memory=False)

            df_dis = pd.read_csv(DIS_CSV, low_memory=False)

            df_final["viewer_source"] = "final"

            df_dis["viewer_source"] = "disagreement"

            self.df = pd.concat(
                [df_final, df_dis],
                ignore_index=True
            )

        else:

            self.df = pd.read_csv(CSV_PATH, low_memory=False)

        # ============================================================
        # FILTER
        # ============================================================

        if MODE == "disagreements":

            if "filter_reason" in self.df.columns:

                self.df = self.df[
                    self.df["filter_reason"]
                    .astype(str)
                    .str.contains("disagreement", case=False, na=False)
                ]

                print(
                    "Nur Disagreements geladen:",
                    len(self.df)
                )

        self.filtered_df = self.df.copy()

        if MODE == "all":
            self.image_dir = None
        else:
            self.image_dir = Path(IMAGE_DIR)

        self.loader = AsyncImageLoader()

        self.current_idx = 0

        self.build_ui()

        self.populate_list()

        self.root.bind("<Down>", lambda e: self.next_item())
        self.root.bind("<Up>", lambda e: self.prev_item())

        self.root.bind("<Right>", lambda e: self.next_item())
        self.root.bind("<Left>", lambda e: self.prev_item())

        self.root.bind("f", lambda e: self.focus_search())

    # ============================================================
    # UI
    # ============================================================

    def build_ui(self):

        main = ttk.Panedwindow(
            self.root,
            orient=tk.HORIZONTAL
        )

        main.pack(fill="both", expand=True)

        left = tk.Frame(main, bg=BG)
        center = tk.Frame(main, bg=BG)
        right = tk.Frame(main, bg=BG)

        main.add(left, weight=1)
        main.add(center, weight=2)
        main.add(right, weight=3)

        # ============================================================
        # LEFT
        # ============================================================

        search_frame = tk.Frame(left, bg=BG)
        search_frame.pack(fill="x")

        tk.Label(
            search_frame,
            text="Filter",
            bg=BG,
            fg=FG
        ).pack()

        self.search_entry = tk.Entry(search_frame)

        self.search_entry.pack(fill="x")

        tk.Button(
            search_frame,
            text="Apply",
            command=self.apply_filter
        ).pack(fill="x")

        tk.Button(
            search_frame,
            text="Only uncertain",
            command=self.filter_uncertain
        ).pack(fill="x")

        self.listbox = tk.Listbox(
            left,
            bg="#252526",
            fg=FG,
            font=FONT
        )

        self.listbox.pack(fill="both", expand=True)

        self.listbox.bind(
            "<<ListboxSelect>>",
            self.on_select
        )

        # ============================================================
        # CENTER
        # ============================================================

        self.image_label = tk.Label(
            center,
            bg=BG
        )

        self.image_label.pack(
            fill="both",
            expand=True
        )

        # ============================================================
        # RIGHT
        # ============================================================

        self.info = ScrolledText(
            right,
            bg="#111111",
            fg=FG,
            insertbackground=FG,
            font=FONT
        )

        self.info.pack(
            fill="both",
            expand=True
        )

        # ============================================================
        # STATUS
        # ============================================================

        self.status = tk.Label(
            self.root,
            text="READY",
            anchor="w",
            bg="#333333",
            fg="white"
        )

        self.status.pack(fill="x")

    # ============================================================
    # FILTER
    # ============================================================

    def apply_filter(self):

        val = self.search_entry.get().lower().strip()

        if not val:

            self.filtered_df = self.df.copy()

        else:

            mask = (
                self.df.astype(str)
                .apply(
                    lambda col: col.str.lower().str.contains(val, na=False)
                )
                .any(axis=1)
            )

            self.filtered_df = self.df[mask]

        self.populate_list()

    def filter_uncertain(self):

        if "uncertain" not in self.df.columns:
            return

        self.filtered_df = self.df[
            self.df["uncertain"] == True
        ]

        self.populate_list()

    # ============================================================
    # LIST
    # ============================================================

    def populate_list(self):

        self.listbox.delete(0, tk.END)

        for idx, row in self.filtered_df.iterrows():

            rule_pred = row.get("rule_pred", "?")
            cnn_pred = row.get("cnn_pred", "?")
            final_label = row.get("final_label", "?")
            cnn3_pred = row.get("cnn3_pred", "?")

            image_name = row.get("image_name", "?")

            txt = (
                f"{idx} | "
                f"RULE={rule_pred} | "
                f"CNN={cnn_pred} | "
                f"CNN3={cnn3_pred} | "
                f"FINAL={final_label} | "
                f"{image_name} | "
                f"{str(row.get('caption', ''))[:60]}"
            )

            self.listbox.insert(tk.END, txt)

            rule_pred = str(
                row.get("rule_pred", "")
            ).strip().lower()

            cnn_pred = str(
                row.get("cnn_pred", "")
            ).strip().lower()

            idx_listbox = self.listbox.size() - 1

            filter_reason = str(
                row.get("filter_reason", "")
            ).lower()

            # ============================================================
            # CNN3 FILTER
            # ============================================================

            if "cnn3" in filter_reason:

                self.listbox.itemconfig(
                    idx_listbox,
                    bg="#5a3b00",
                    fg="#ffd27f"
                )

                continue

            # ============================================================
            # AGREEMENT
            # ============================================================

            if rule_pred == cnn_pred:

                self.listbox.itemconfig(
                    idx_listbox,
                    bg="#143214",
                    fg="#aaffaa"
                )

            # ============================================================
            # DISAGREEMENT
            # ============================================================

            else:

                self.listbox.itemconfig(
                    idx_listbox,
                    bg="#401414",
                    fg="#ffaaaa"
                )

        self.status.config(
            text=f"{len(self.filtered_df)} Samples"
        )

    # ============================================================
    # NAVIGATION
    # ============================================================

    def next_item(self):

        if self.current_idx < len(self.filtered_df) - 1:

            self.current_idx += 1

            self.listbox.selection_clear(0, tk.END)

            self.listbox.selection_set(self.current_idx)

            self.show_current()

    def prev_item(self):

        if self.current_idx > 0:

            self.current_idx -= 1

            self.listbox.selection_clear(0, tk.END)

            self.listbox.selection_set(self.current_idx)

            self.show_current()

    # ============================================================
    # SELECT
    # ============================================================

    def on_select(self, event):

        if not self.listbox.curselection():
            return

        self.current_idx = self.listbox.curselection()[0]

        self.show_current()

    # ============================================================
    # SHOW
    # ============================================================

    def show_current(self):

        row = self.filtered_df.iloc[self.current_idx]

        self.show_image(row)

        self.show_text(row)

        self.preload_neighbors()

    # ============================================================
    # IMAGE
    # ============================================================

    def show_image(self, row):

        path = build_image_path(
            self.image_dir,
            row
        )

        if path is None:

            self.image_label.config(
                image="",
                text="NO IMAGE",
                fg="red"
            )

            return

        img = self.loader.get(path)

        if img is None:

            self.image_label.config(
                image="",
                text=f"NO IMAGE\n{path}",
                fg="red"
            )

            return

        self.tk_img = ImageTk.PhotoImage(img)

        self.image_label.config(
            image=self.tk_img
        )

    # ============================================================
    # TEXT
    # ============================================================

    def show_text(self, row):

        self.info.delete("1.0", tk.END)

        lines = []

        lines.append("=" * 80)

        rule_pred = row.get("rule_pred", "?")
        cnn_pred = row.get("cnn_pred", "?")

        is_agreement = (
            str(rule_pred).strip().lower()
            ==
            str(cnn_pred).strip().lower()
        )

        lines.append("")

        cnn3_pred = row.get("cnn3_pred", "?")
        cnn3_conf = row.get("cnn3_conf", "?")

        if is_agreement:

            lines.append("## AGREEMENT ##")

        else:

            lines.append("!! DISAGREEMENT !!")

        lines.append(f"RULE : {rule_pred}")
        lines.append(f"CNN  : {cnn_pred}")
        lines.append(f"CNN3 : {cnn3_pred}")
        lines.append(f"C3CF : {cnn3_conf}")

        lines.append("")

        for col in row.index:

            val = row[col]

            try:

                if isinstance(val, str):

                    vv = val.strip()

                    if vv.startswith("{") or vv.startswith("["):

                        try:

                            parsed = json.loads(vv)

                            val = json.dumps(
                                parsed,
                                indent=2
                            )

                        except:
                            pass

                lines.append(f"\n[{col}]")

                lines.append(str(val))

            except Exception as e:

                lines.append(f"{col}: ERROR {e}")

        self.info.insert(
            tk.END,
            "\n".join(lines)
        )

    # ============================================================
    # PRELOAD
    # ============================================================

    def preload_neighbors(self):

        paths = []

        for delta in [-2, -1, 1, 2]:

            idx = self.current_idx + delta

            if idx < 0 or idx >= len(self.filtered_df):
                continue

            row = self.filtered_df.iloc[idx]

            path = build_image_path(
                self.image_dir,
                row
            )

            if path is not None:
                paths.append(path)

        self.loader.preload(paths)

    # ============================================================
    # SEARCH FOCUS
    # ============================================================

    def focus_search(self):

        self.search_entry.focus_set()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    root = tk.Tk()

    app = ROCOViewer(root)

    root.mainloop()