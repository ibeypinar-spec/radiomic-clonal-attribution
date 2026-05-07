import nibabel as nib
import glob

files = sorted(glob.glob("C:/RadyomikAtribuisyon/nifti/prad/*.nii.gz"))
counts = {"3d": [], "4d": []}
for f in files:
    img = nib.load(f)
    shape = img.shape
    key = "4d" if len(shape) == 4 else "3d"
    counts[key].append(f.split("/")[-1] + f" {shape}")

print(f"3D: {len(counts['3d'])} dosya")
for x in counts["3d"][:5]:
    print(" ", x)
print(f"4D: {len(counts['4d'])} dosya")
for x in counts["4d"][:5]:
    print(" ", x)
