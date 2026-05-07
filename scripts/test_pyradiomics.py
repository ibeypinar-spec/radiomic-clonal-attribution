import sys
sys.stdout.reconfigure(line_buffering=True)
print("PyRadiomics test basliyor...")

import radiomics
from radiomics import featureextractor
from pathlib import Path
import time

BASE_DIR = Path("C:/RadyomikAtribuisyon")
pid = "TCGA-AO-A03M"
img_path = str(BASE_DIR / f"nifti/brca/{pid}.nii.gz")
mask_path = str(BASE_DIR / f"segmentations/brca/{pid}_mask.nii.gz")

extractor = featureextractor.RadiomicsFeatureExtractor(str(BASE_DIR / "scripts/pyradiomics_config.yaml"))

print(f"Extracting features for {pid}...")
t0 = time.time()

try:
    result = extractor.execute(img_path, mask_path)
    elapsed = time.time() - t0
    
    features = {k: v for k, v in result.items() if not k.startswith("diagnostics_")}
    print(f"BASARILI: {len(features)} ozellik, {elapsed:.1f} saniye")
    for k, v in list(features.items())[:5]:
        print(f"  {k}: {v}")
except Exception as e:
    import traceback
    print(f"HATA: {e}")
    traceback.print_exc()
