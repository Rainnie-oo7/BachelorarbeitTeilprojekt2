# -*- coding: utf-8 -*-

"""
compare_all_fusions.py

Vergleicht ALLE fusionierten LoRA-Gewichte numerisch.

Ziel:
Prüfen, ob unterschiedliche Fusionsmethoden
tatsächlich verschiedene Gewichte erzeugen.

Verglichen werden:
- additive
- cosine
- hadamard
- layerwise
- norm
- svd
- weighted

für:
- ROCO
- PMC

Ausgabe:
- mittlere absolute Differenz
- maximale Differenz
- globale Gesamtdifferenz
- CSV Export

"""

from __future__ import annotations

import os
import os.path as osp
from pathlib import Path
from itertools import combinations

import torch
import pandas as pd
from safetensors.torch import load_file


# ============================================================
# ROOTS
# ============================================================

ROOTS = {

    "roco": "/home/b/PycharmProjects/ba2roco/LoRAs",

    "pmc": "/home/b/PycharmProjects/ba1pmc/LoRAs",
}


# ============================================================
# FIND ALL FUSIONS
# ============================================================

def find_all_fusions(root):

    root = Path(root)

    found = []

    for p in root.rglob("*"):

        if p.is_file():

            if p.name in [
                "adapter_model.bin",
                "adapter_model.safetensors"
            ]:

                found.append(str(p))

    return sorted(found)


# ============================================================
# LOAD STATE
# ============================================================

def load_state(path):

    if path.endswith(".safetensors"):

        return load_file(path)

    return torch.load(
        path,
        map_location="cpu"
    )


# ============================================================
# EXTRACT GROUP
# ============================================================

def extract_group_name(path):

    parent = Path(path).parent.name

    # Beispiel:
    # fusion_additive_class_lora_xray_class_lora_us

    methods = [
        "additive",
        "cosine",
        "hadamard",
        "layerwise",
        "norm",
        "svd",
        "weighted"
    ]

    for m in methods:

        parent = parent.replace(
            f"fusion_{m}_",
            ""
        )

    return parent


# ============================================================
# COMPARE
# ============================================================

def compare_states(sd1, sd2):

    total_mean = 0.0
    total_max = 0.0
    tensor_count = 0

    per_tensor = []

    common_keys = set(sd1.keys()) & set(sd2.keys())

    for k in sorted(common_keys):

        t1 = sd1[k].float()
        t2 = sd2[k].float()

        diff = (t1 - t2).abs()

        mean_diff = diff.mean().item()
        max_diff = diff.max().item()

        total_mean += mean_diff
        total_max = max(total_max, max_diff)

        tensor_count += 1

        per_tensor.append({
            "tensor": k,
            "mean_diff": mean_diff,
            "max_diff": max_diff
        })

    global_mean = total_mean / max(tensor_count, 1)

    return {

        "global_mean_diff": global_mean,
        "global_max_diff": total_max,
        "tensor_count": tensor_count,
        "per_tensor": per_tensor
    }


# ============================================================
# MAIN
# ============================================================

def main():

    all_results = []

    for dataset_name, root in ROOTS.items():

        print("\n================================================")
        print(dataset_name.upper())
        print("================================================")

        fusion_files = find_all_fusions(root)

        print("FOUND:", len(fusion_files))

        grouped = {}

        # ----------------------------------------------------
        # Gruppieren nach Fusionsexperiment
        # ----------------------------------------------------

        for path in fusion_files:

            group = extract_group_name(path)

            if group not in grouped:
                grouped[group] = []

            grouped[group].append(path)

        # ----------------------------------------------------
        # Jede Methode mit jeder vergleichen
        # ----------------------------------------------------

        for group_name, paths in grouped.items():

            print("\n--------------------------------------------")
            print(group_name)
            print("--------------------------------------------")

            for path1, path2 in combinations(paths, 2):

                name1 = Path(path1).parent.name
                name2 = Path(path2).parent.name

                print("\nCOMPARE:")
                print(name1)
                print("VS")
                print(name2)

                try:

                    sd1 = load_state(path1)
                    sd2 = load_state(path2)

                    result = compare_states(
                        sd1,
                        sd2
                    )

                    print(
                        "MEAN:",
                        result["global_mean_diff"]
                    )

                    print(
                        "MAX:",
                        result["global_max_diff"]
                    )

                    all_results.append({

                        "dataset": dataset_name,

                        "group": group_name,

                        "fusion_1": name1,

                        "fusion_2": name2,

                        "global_mean_diff":
                            result["global_mean_diff"],

                        "global_max_diff":
                            result["global_max_diff"],

                        "tensor_count":
                            result["tensor_count"]
                    })

                except Exception as e:

                    print("FAILED:", e)

    # ========================================================
    # SAVE
    # ========================================================

    df = pd.DataFrame(all_results)

    df = df.sort_values(
        by="global_mean_diff",
        ascending=False
    )

    output_csv = \
        "/home/b/Dokumente/fusion_weight_differences.csv"

    df.to_csv(
        output_csv,
        index=False
    )

    print("\n================================================")
    print("GESPEICHERT")
    print("================================================")

    print(output_csv)

    print("\nTOP DIFFERENCES:")
    print(df.head(20))


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()