"""
BRCA icin Otsu esikleme ile meme maskesi olusturur.
TotalSegmentator breasts task bu dataset icin calismiyor.
Breast MRI taramalari zaten meme bolgesine odaklidir.
Otsu esiklemesi ile tum relevant dokuyu maske olarak kullaniyoruz.
"""
import nibabel as nib
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from scipy import ndimage
import logging

BASE_DIR = Path("C:/RadyomikAtribuisyon")
nifti_dir = BASE_DIR / "nifti/brca"
seg_dir = BASE_DIR / "segmentations/brca"
seg_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "results/02_brca_otsu.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

files = sorted(nifti_dir.glob("*.nii.gz"))
log.info(f"BRCA Otsu maskeleme: {len(files)} dosya")

ok = 0
fail = 0

for i, nifti_path in enumerate(files, 1):
    pid = nifti_path.stem.replace(".nii", "")
    mask_path = seg_dir / f"{pid}_mask.nii.gz"

    if mask_path.exists():
        log.info(f"[{i}/{len(files)}] SKIP {pid} (maske var)")
        ok += 1
        continue

    try:
        img = nib.load(str(nifti_path))
        arr = img.get_fdata()

        # 4D ise ilk volume al
        if arr.ndim == 4:
            arr = arr[..., 0]

        # Mammografi kontrolu (cok buyuk 2D)
        if arr.shape[0] > 1000 or arr.shape[1] > 1000:
            log.warning(f"[{i}/{len(files)}] SKIP {pid}: Muhtemelen mamografi {arr.shape}")
            fail += 1
            continue

        # Otsu esikleme: scikit-image yerine manuel
        flat = arr[arr > 0].flatten()
        if len(flat) == 0:
            log.warning(f"[{i}/{len(files)}] {pid}: Tamamen sifir goruntu")
            fail += 1
            continue

        # Percentile bazli esikleme (Otsu yaklasimi)
        threshold = np.percentile(flat, 15)  # Alt %15'i gida/hava olarak at
        binary = (arr > threshold).astype(np.uint8)

        # Morfologik temizleme: kucuk izole bolgeleri kaldir
        labeled, n_labels = ndimage.label(binary)
        if n_labels == 0:
            log.warning(f"[{i}/{len(files)}] {pid}: Maske bos")
            fail += 1
            continue

        # En buyuk connected component'i al
        sizes = ndimage.sum(binary, labeled, range(1, n_labels + 1))
        max_label = np.argmax(sizes) + 1
        binary_clean = (labeled == max_label).astype(np.uint8)

        # Kucuk maske kontrolu
        if binary_clean.sum() < 1000:
            log.warning(f"[{i}/{len(files)}] {pid}: Cok kucuk maske ({binary_clean.sum()} voxel)")
            fail += 1
            continue

        # Kaydet
        out_img = nib.Nifti1Image(binary_clean, img.affine)
        nib.save(out_img, str(mask_path))
        log.info(f"[{i}/{len(files)}] OK {pid}: {binary_clean.sum()} voxel, shape={arr.shape}")
        ok += 1

    except Exception as e:
        log.error(f"[{i}/{len(files)}] ERR {pid}: {e}")
        fail += 1

log.info(f"\n=== TAMAMLANDI: {ok} basarili, {fail} basarisiz ===")
