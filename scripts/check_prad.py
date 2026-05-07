import nibabel as nib
import numpy as np
import glob

files = sorted(glob.glob('C:/RadyomikAtribuisyon/nifti/prad/*.nii.gz'))[:3]
for f in files:
    img = nib.load(f)
    arr = img.get_fdata()
    print(f'File: {f.split("/")[-1]}')
    print(f'  Shape: {arr.shape}, dtype: {arr.dtype}')
    print(f'  Range: [{arr.min():.1f}, {arr.max():.1f}], mean: {arr.mean():.1f}')
    print(f'  Voxel size: {img.header.get_zooms()}')
