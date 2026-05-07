"""
PyRadiomics test — shape kaldirilmis config (2mm resampling).
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
print("=== PyRadiomics Config Test (no shape, 2mm) ===", flush=True)

import radiomics
from radiomics import featureextractor
from pathlib import Path
import time

BASE_DIR = Path("C:/RadyomikAtribuisyon")
pid = "TCGA-AO-A03M"
img_path = str(BASE_DIR / f"nifti/brca/{pid}.nii.gz")
mask_path = str(BASE_DIR / f"segmentations/brca/{pid}_mask.nii.gz")

print(f"PyRadiomics: {radiomics.__version__}", flush=True)
print(f"Config: {BASE_DIR}/scripts/pyradiomics_config.yaml", flush=True)

print("\n--- TEST: firstorder+glcm+glrlm+glszm + 2mm (config dosyasi) ---", flush=True)
try:
    ext = featureextractor.RadiomicsFeatureExtractor(
        str(BASE_DIR / "scripts/pyradiomics_config.yaml"))
    t0 = time.time()
    r = ext.execute(img_path, mask_path)
    feats = {k:v for k,v in r.items() if not k.startswith('diagnostics_')}
    elapsed = time.time() - t0
    print(f"  BASARILI: {len(feats)} ozellik, {elapsed:.1f}s", flush=True)
    # Feature siniflari goster
    classes = {}
    for k in feats:
        parts = k.split('_')
        cls = parts[1] if len(parts) > 1 else 'other'
        classes[cls] = classes.get(cls, 0) + 1
    for cls, cnt in sorted(classes.items()):
        print(f"    {cls}: {cnt} ozellik", flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"  HATA: {e}", flush=True)

print("\n=== TAMAMLANDI ===", flush=True)
