import nibabel as nib
import glob

files = sorted(glob.glob("C:/RadyomikAtribuisyon/nifti/brca/*.nii.gz"))[:10]
for f in files:
    img = nib.load(f)
    shape = img.shape
    zoom = img.header.get_zooms()[:3]
    pid = f.split("/")[-1].replace(".nii.gz","")
    print(f"{pid}: shape={shape}, voxel={[round(z,1) for z in zoom]}")
