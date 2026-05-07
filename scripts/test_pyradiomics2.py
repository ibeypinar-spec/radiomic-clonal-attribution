"""
PyRadiomics crash diagnostic - testa ogni feature class separatamente
e prova diverse resoluzioni.
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)

import radiomics
from radiomics import featureextractor
from pathlib import Path
import time, gc
import numpy as np
import SimpleITK as sitk

BASE_DIR = Path("C:/RadyomikAtribuisyon")
pid = "TCGA-AO-A03M"
img_path = str(BASE_DIR / f"nifti/brca/{pid}.nii.gz")
mask_path = str(BASE_DIR / f"segmentations/brca/{pid}_mask.nii.gz")

print(f"=== PyRadiomics Crash Diagnostic ===")
print(f"Python: {sys.version}")
print(f"PyRadiomics: {radiomics.__version__}")

# Goruntu bilgisi
img = sitk.ReadImage(img_path)
msk = sitk.ReadImage(mask_path)
print(f"\nImage: size={img.GetSize()}, spacing={[round(s,3) for s in img.GetSpacing()]}")
arr_m = sitk.GetArrayFromImage(msk)
print(f"Mask: nonzero={np.sum(arr_m>0)}, unique={np.unique(arr_m).tolist()}")

# Tahmini 1mm resampling boyutu
size = img.GetSize()  # (x, y, z)
sp = img.GetSpacing()
resampled_size = [int(size[i]*sp[i]) for i in range(3)]
total_vox = resampled_size[0]*resampled_size[1]*resampled_size[2]
print(f"\n1mm resampling -> tahmini boyut: {resampled_size}, toplam voxel: {total_vox:,}")
print(f"Bellek tahmini: {total_vox*4/1024/1024:.0f} MB (float32)")

print("\n--- TEST 1: Sadece firstorder (no resampling) ---")
try:
    ext = featureextractor.RadiomicsFeatureExtractor()
    ext.disableAllFeatures()
    ext.enableFeaturesByName(firstorder=[])
    ext.settings['resampledPixelSpacing'] = None
    ext.settings['correctMask'] = True
    ext.settings['label'] = 1
    t0 = time.time()
    r = ext.execute(img_path, mask_path)
    feats = {k:v for k,v in r.items() if not k.startswith('diagnostics_')}
    print(f"  OK: {len(feats)} ozellik, {time.time()-t0:.1f}s")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  HATA: {e}")
gc.collect()

print("\n--- TEST 2: Sadece shape (no resampling) ---")
try:
    ext = featureextractor.RadiomicsFeatureExtractor()
    ext.disableAllFeatures()
    ext.enableFeaturesByName(shape=[])
    ext.settings['resampledPixelSpacing'] = None
    ext.settings['correctMask'] = True
    ext.settings['label'] = 1
    t0 = time.time()
    r = ext.execute(img_path, mask_path)
    feats = {k:v for k,v in r.items() if not k.startswith('diagnostics_')}
    print(f"  OK: {len(feats)} ozellik, {time.time()-t0:.1f}s")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  HATA: {e}")
gc.collect()

print("\n--- TEST 3: Shape + 2mm resampling ---")
try:
    ext = featureextractor.RadiomicsFeatureExtractor()
    ext.disableAllFeatures()
    ext.enableFeaturesByName(shape=[])
    ext.settings['resampledPixelSpacing'] = [2, 2, 2]
    ext.settings['correctMask'] = True
    ext.settings['label'] = 1
    t0 = time.time()
    r = ext.execute(img_path, mask_path)
    feats = {k:v for k,v in r.items() if not k.startswith('diagnostics_')}
    print(f"  OK: {len(feats)} ozellik, {time.time()-t0:.1f}s")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  HATA: {e}")
gc.collect()

print("\n--- TEST 4: Shape + 1mm resampling ---")
try:
    ext = featureextractor.RadiomicsFeatureExtractor()
    ext.disableAllFeatures()
    ext.enableFeaturesByName(shape=[])
    ext.settings['resampledPixelSpacing'] = [1, 1, 1]
    ext.settings['correctMask'] = True
    ext.settings['label'] = 1
    t0 = time.time()
    r = ext.execute(img_path, mask_path)
    feats = {k:v for k,v in r.items() if not k.startswith('diagnostics_')}
    print(f"  OK: {len(feats)} ozellik, {time.time()-t0:.1f}s")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  HATA: {e}")
gc.collect()

print("\n--- TEST 5: Tum featureler + 2mm resampling ---")
try:
    ext = featureextractor.RadiomicsFeatureExtractor()
    ext.disableAllFeatures()
    ext.enableFeaturesByName(shape=[], firstorder=[], glcm=[], glrlm=[], glszm=[])
    ext.settings['resampledPixelSpacing'] = [2, 2, 2]
    ext.settings['correctMask'] = True
    ext.settings['label'] = 1
    ext.settings['binWidth'] = 25
    t0 = time.time()
    r = ext.execute(img_path, mask_path)
    feats = {k:v for k,v in r.items() if not k.startswith('diagnostics_')}
    print(f"  OK: {len(feats)} ozellik, {time.time()-t0:.1f}s")
    for k, v in list(feats.items())[:3]:
        print(f"    {k}: {v}")
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  HATA: {e}")

print("\n=== TAMAMLANDI ===")
