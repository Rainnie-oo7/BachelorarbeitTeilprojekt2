import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# CSV laden
# =========================
csv_path = "erste_cnn_results.csv"

df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8")

# Erste Spalte als Index (optional)
df = df.set_index(df.columns[0])
metric_cols = ["Akkuranz", "Präzision", "Recall", "F1"]
# =========================
# Cleaning
# =========================
def clean_column(col):
    return (
        col.astype(str)
        .str.replace("%", "", regex=False)   # % entfernen
        .str.replace(",", ".", regex=False)  # Komma -> Punkt
        .astype(float)
    )

def clean_percent_column(col):
    return (
        col.astype(str)
        .str.replace("%", "", regex=False)   # % entfernen
        .str.replace(",", ".", regex=False)  # Komma -> Punkt
        .astype(float) / 100.0               # in [0,1] skalieren
    )

# Alle Spalten bereinigen
for col in ["Präzision", "Recall", "F1"]:
    df[col] = clean_column(df[col])

# NUR Akkuranz skalieren
df["Akkuranz"] = clean_percent_column(df["Akkuranz"])

heatmap_data = df[metric_cols]

# =========================
# Heatmap
# =========================
plt.figure(figsize=(8, 6))

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".3f",
    cmap="RdYlGn",
    linewidths=0.5,
    cbar=True,
    vmin=0.0,
    vmax=1.0
)

plt.title("Heatmap der Metriken")
plt.xlabel("Metriken")
plt.ylabel("Klasse / Modell")

plt.tight_layout()
plt.show()