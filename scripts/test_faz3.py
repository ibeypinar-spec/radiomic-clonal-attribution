"""FAZ 3 hizli test — ilk 3 hasta"""
import sys
sys.argv = ['03_feature_extraction.py', '--dataset', 'luad']
from pathlib import Path
BASE_DIR = Path("C:/RadyomikAtribuisyon")

cfg = {
    "nifti": BASE_DIR / "nifti/luad",
    "seg":   BASE_DIR / "segmentations/luad",
    "label": "akciger",
}
masks = list(cfg["seg"].glob("*_mask.nii.gz"))
niftis = list(cfg["nifti"].glob("*.nii.gz"))
print(f"NIfTI: {len(niftis)}, Maske: {len(masks)}")
print("Ilk 3 maske:", [m.name for m in masks[:3]])
