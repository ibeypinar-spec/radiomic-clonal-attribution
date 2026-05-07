"""Kaydedilmis modelden SHAP analizi."""
import json, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings("ignore")
BASE_DIR = Path("C:/RadyomikAtribuisyon")

import joblib
bundle = joblib.load(BASE_DIR / "models/primer_imza.pkl")
pipe = bundle["pipeline"]
feature_names = bundle["feature_names"]
label_map = bundle["label_map"]
print(f"Model yuklendi. Label map: {label_map}")

# Veriyi yeniden yukle
FILENAME_TO_TYPE = {
    "tcga_luad": "akciger", "tcga_lusc": "akciger",
    "tcga_brca": "meme", "tcga_coad": "kolon", "tcga_prad": "prostat",
}
dfs = []
for csv in sorted((BASE_DIR / "features").glob("tcga_*_features.csv")):
    df = pd.read_csv(csv)
    stem = csv.stem.replace("_features", "")
    if "cancer_type" not in df.columns:
        df["cancer_type"] = FILENAME_TO_TYPE.get(stem)
    dfs.append(df)
combined = pd.concat(dfs, ignore_index=True)

feature_cols = [c for c in combined.columns if c not in ("patient_id", "cancer_type")]
X = combined[feature_cols].copy()
X = X.loc[:, X.std() > 0]
X = X.replace([float('inf'), float('-inf')], float('nan')).fillna(X.median())

import shap
scaler   = pipe.named_steps["scaler"]
selector = pipe.named_steps["selector"]
clf      = pipe.named_steps["clf"]

X_scaled = scaler.transform(X)
X_sel    = selector.transform(X_scaled)
sel_names = [feature_names[i] for i in selector.get_support(indices=True)]
print(f"Secilen ozellik sayisi: {len(sel_names)}")

print("SHAP hesaplaniyor...")
explainer   = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_sel)

if isinstance(shap_values, list):
    mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
elif np.array(shap_values).ndim == 3:
    mean_shap = np.abs(shap_values).mean(axis=(0, 2))
else:
    mean_shap = np.abs(shap_values).mean(axis=0)

mean_shap = mean_shap.flatten()
n_top = min(20, len(sel_names))
top_idx = np.argsort(mean_shap)[-n_top:][::-1]
top20 = [(sel_names[int(i)], float(mean_shap[int(i)])) for i in top_idx]

print("\nEn ayirt edici ozellikler:")
for name, val in top20:
    print(f"  {val:.4f}  {name}")

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(range(n_top), [v for _, v in top20][::-1])
ax.set_yticks(range(n_top))
ax.set_yticklabels([n for n, _ in top20][::-1], fontsize=8)
ax.set_xlabel("Ortalama |SHAP| degeri")
ax.set_title("En Ayirt Edici Radyomik Ozellikler (3 sinif)")
plt.tight_layout()
plt.savefig(BASE_DIR / "results/04_shap_ozet.png", dpi=150, bbox_inches="tight")
print(f"\nSHAP grafigi kaydedildi: {BASE_DIR}/results/04_shap_ozet.png")

with open(BASE_DIR / "results/04_top20_features.json", "w") as f:
    json.dump(top20, f, indent=2, ensure_ascii=False)
print("Tamamlandi.")
