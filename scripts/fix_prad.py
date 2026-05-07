"""
PRAD icin T2 MRI hastalarini hazirlar:
- 4D dosyalari 3D'ye donusturur (ilk volume alir)
- Geçersiz modaliteleri (PET, CT) atar
- total_mr task ile roi_subset OLMADAN dener
"""
import nibabel as nib
import numpy as np
import subprocess
import os
from pathlib import Path
import tempfile
import SimpleITK as sitk

BASE_DIR = Path("C:/RadyomikAtribuisyon")
VALID_PRAD = ["TCGA-EJ-5495", "TCGA-EJ-5518", "TCGA-EJ-5542"]
nifti_dir = BASE_DIR / "nifti/prad"
seg_dir = BASE_DIR / "segmentations/prad"
seg_dir.mkdir(exist_ok=True)

for pid in VALID_PRAD:
    nifti_path = nifti_dir / f"{pid}.nii.gz"
    mask_path = seg_dir / f"{pid}_mask.nii.gz"
    
    if mask_path.exists():
        print(f"SKIP {pid} (maske var)")
        continue
    
    if not nifti_path.exists():
        print(f"ERROR {pid}: NIfTI dosya yok")
        continue
    
    # Load and check dimensions
    img = nib.load(str(nifti_path))
    shape = img.shape
    print(f"\n{pid}: shape={shape}")
    
    # Fix 4D -> 3D
    tmp_path = nifti_path
    if len(shape) == 4:
        arr = img.get_fdata()
        arr3d = arr[:,:,:,0]
        header = img.header.copy()
        header.set_data_shape(arr3d.shape)
        # Remove 4th dim info
        fixed = nib.Nifti1Image(arr3d, img.affine, header)
        tmp_3d = BASE_DIR / f"nifti/prad/{pid}_3d_tmp.nii.gz"
        nib.save(fixed, str(tmp_3d))
        tmp_path = tmp_3d
        print(f"  4D->3D: {shape} -> {arr3d.shape}")
    
    # Try TotalSegmentator WITHOUT roi_subset
    TOTALSEG = r"C:\Users\Lenovo\anaconda3\envs\radiomics\Scripts\TotalSegmentator.exe"
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            TOTALSEG,
            "-i", str(tmp_path),
            "-o", tmp,
            "--task", "total_mr"
        ]
        print(f"  Komut: {' '.join(cmd)}")
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            print(f"  returncode={proc.returncode}")
            if proc.stderr:
                # Show last 5 lines of stderr
                for line in proc.stderr.strip().split("\n")[-5:]:
                    print(f"  STDERR: {line}")
            
            prostate_f = Path(tmp) / "prostate.nii.gz"
            if prostate_f.exists():
                arr = sitk.GetArrayFromImage(sitk.ReadImage(str(prostate_f)))
                if arr.sum() > 0:
                    ref_img = sitk.ReadImage(str(tmp_path))
                    out_img = sitk.GetImageFromArray((arr > 0).astype(np.uint8))
                    out_img.CopyInformation(ref_img)
                    sitk.WriteImage(out_img, str(mask_path))
                    print(f"  OK: prostate segmented, nonzero={arr.sum()}")
                else:
                    print(f"  WARN: prostate.nii.gz bos")
            else:
                avail = [f.name for f in Path(tmp).glob("*.nii.gz")][:10]
                print(f"  ERROR: prostate.nii.gz yok. Mevcut: {avail}")
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT")
        except Exception as e:
            print(f"  EXCEPTION: {e}")

print("\nDone.")
