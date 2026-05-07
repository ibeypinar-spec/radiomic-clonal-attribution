import SimpleITK as sitk
import numpy as np
from pathlib import Path

BASE_DIR = Path("C:/RadyomikAtribuisyon")

# Ilk BRCA hastasini test et
pid = "TCGA-AO-A03M"
img_path = BASE_DIR / f"nifti/brca/{pid}.nii.gz"
mask_path = BASE_DIR / f"segmentations/brca/{pid}_mask.nii.gz"

print(f"Image: {img_path.exists()}")
print(f"Mask:  {mask_path.exists()}")

img = sitk.ReadImage(str(img_path))
mask = sitk.ReadImage(str(mask_path))

img_arr = sitk.GetArrayFromImage(img)
mask_arr = sitk.GetArrayFromImage(mask)

print(f"Image shape: {img_arr.shape}, dtype: {img_arr.dtype}")
print(f"Mask shape:  {mask_arr.shape}, dtype: {mask_arr.dtype}")
print(f"Mask unique: {np.unique(mask_arr)}")
print(f"Mask nonzero: {mask_arr.sum()} voxels")
print(f"Image spacing: {img.GetSpacing()}")
print(f"Mask spacing:  {mask.GetSpacing()}")
print(f"Image size: {img.GetSize()}")
print(f"Mask size:  {mask.GetSize()}")

# Geometry match?
print(f"Same direction: {img.GetDirection() == mask.GetDirection()}")
print(f"Same origin: {img.GetOrigin() == mask.GetOrigin()}")
