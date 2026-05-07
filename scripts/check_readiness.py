# Sadece kontrol: mevcut CSV dosyalarini yukle
import pandas as pd
from pathlib import Path

BASE_DIR = Path("C:/RadyomikAtribuisyon")
FILENAME_TO_TYPE = {
    "tcga_luad": "akciger",
    "tcga_brca": "meme", 
    "tcga_coad": "kolon",
}

dfs = []
for csv in sorted((BASE_DIR / "features").glob("tcga_*_features.csv")):
    df = pd.read_csv(csv)
    stem = csv.stem.replace("_features", "")
    ctype = FILENAME_TO_TYPE.get(stem)
    if ctype:
        df["cancer_type"] = ctype
        dfs.append(df)
        print(f"  {csv.name}: {len(df)} hasta, type={ctype}, {df.shape[1]} sutun")

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nToplam: {len(combined)} hasta, {combined['cancer_type'].value_counts().to_dict()}")
    print(f"Ozellik sayisi: {combined.shape[1] - 2} (patient_id ve cancer_type haric)")
