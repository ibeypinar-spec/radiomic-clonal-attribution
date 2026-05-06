"""Pipeline ilerleme kontrolu — herhangi bir anda calistir"""
from pathlib import Path

BASE = Path("C:/RadyomikAtribuisyon")

datasets = ["luad", "lusc", "brca", "prad", "coad"]

print("=" * 55)
print("PIPELINE DURUM RAPORU")
print("=" * 55)

for ds in datasets:
    nifti   = len(list((BASE / f"nifti/{ds}").glob("*.nii.gz")))
    masks   = len(list((BASE / f"segmentations/{ds}").glob("*_mask.nii.gz")))
    feat_f  = BASE / f"features/tcga_{ds}_features.csv"
    feats   = 0
    if feat_f.exists():
        import pandas as pd
        feats = len(pd.read_csv(feat_f))

    bar_n = "█" * nifti + "░" * max(0, 5 - nifti // 20)
    print(f"\n{ds.upper()}")
    print(f"  FAZ1 NIfTI  : {nifti:3d} dosya")
    print(f"  FAZ2 Maske  : {masks:3d} / {nifti}")
    print(f"  FAZ3 Ozellik: {feats:3d} hasta")

model = BASE / "models/primer_imza.pkl"
roc   = BASE / "results/05_validasyon_roc.png"
print(f"\nFAZ4a Model  : {'✓ Hazir' if model.exists() else '— Bekliyor'}")
print(f"FAZ4b ROC    : {'✓ Hazir' if roc.exists() else '— Bekliyor'}")
print("=" * 55)
