"""
FAZ 2: Otomatik Segmentasyon — TotalSegmentator
GPU ile ~30 sn/hasta. Akciğer için tüm loblar birleştirilir.
Kullanim: python scripts/02_segmentation.py --dataset luad
"""

import os
import subprocess
import argparse
import json
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np

BASE_DIR = Path("C:/RadyomikAtribuisyon")

# Her dataset için hangi organ dosyalarını birleştireceğiz
DATASET_CFG = {
    "luad": {
        "nifti": BASE_DIR / "nifti/luad",
        "seg":   BASE_DIR / "segmentations/luad",
        "organs": [
            "lung_upper_lobe_left", "lung_lower_lobe_left",
            "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
        ],
        "roi_subset": True,
    },
    "lusc": {
        "nifti": BASE_DIR / "nifti/lusc",
        "seg":   BASE_DIR / "segmentations/lusc",
        "organs": [
            "lung_upper_lobe_left", "lung_lower_lobe_left",
            "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
        ],
        "roi_subset": True,
    },
    "brca": {
        "nifti": BASE_DIR / "nifti/brca",
        "seg":   BASE_DIR / "segmentations/brca",
        "organs": ["breast_left", "breast_right"],
        "roi_subset": True,
    },
    "prad": {
        "nifti": BASE_DIR / "nifti/prad",
        "seg":   BASE_DIR / "segmentations/prad",
        "organs": ["prostate"],
        "roi_subset": True,
    },
    "coad": {
        "nifti": BASE_DIR / "nifti/coad",
        "seg":   BASE_DIR / "segmentations/coad",
        "organs": ["colon"],
        "roi_subset": True,
    },
    "brain_mets": {
        "nifti": BASE_DIR / "nifti/brain_mets",
        "seg":   BASE_DIR / "segmentations/brain_mets",
        "organs": ["brain"],
        "roi_subset": True,
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "results/02_segmentation.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def merge_masks(organ_files: list[Path], reference_file: Path) -> np.ndarray | None:
    """Birden fazla organ maskesini tek binary maskeye birleştir."""
    import SimpleITK as sitk
    combined = None
    for f in organ_files:
        if not f.exists():
            continue
        arr = sitk.GetArrayFromImage(sitk.ReadImage(str(f)))
        if combined is None:
            combined = (arr > 0).astype(np.uint8)
        else:
            combined = np.logical_or(combined, arr > 0).astype(np.uint8)
    return combined


def segment_file(patient_id: str, nifti_path: Path, seg_out: Path, organs: list[str], roi_subset: bool = True) -> dict:
    import SimpleITK as sitk

    mask_path = seg_out / f"{patient_id}_mask.nii.gz"
    result = {"patient_id": patient_id, "status": None, "mask": None, "found_organs": []}

    if mask_path.exists():
        log.info(f"SKIP {patient_id} (maske zaten var)")
        return {"patient_id": patient_id, "status": "skipped", "mask": str(mask_path)}

    seg_out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cmd = ["TotalSegmentator", "-i", str(nifti_path), "-o", tmp]
        if roi_subset:
            cmd += ["--roi_subset"] + organs
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            # Hangi organ dosyaları bulundu?
            found = []
            for organ in organs:
                f = tmp_path / f"{organ}.nii.gz"
                if f.exists():
                    found.append(f)

            if not found:
                # tmp dizininde ne var göster (debug için)
                available = [f.stem.replace(".nii", "") for f in tmp_path.glob("*.nii.gz")][:10]
                log.warning(f"WARN {patient_id}: Hiçbir hedef organ bulunamadı. Mevcut: {available}")
                result["status"] = "no_mask"
                return result

            result["found_organs"] = [f.stem.replace(".nii", "") for f in found]

            # Referans görüntüyü oku (geometri için)
            ref_img = sitk.ReadImage(str(nifti_path))

            # Maskeleri birleştir
            combined = merge_masks(found, nifti_path)
            if combined is None or combined.sum() == 0:
                log.warning(f"WARN {patient_id}: Maske bos")
                result["status"] = "empty_mask"
                return result

            # Kaydet
            out_img = sitk.GetImageFromArray(combined)
            out_img.CopyInformation(ref_img)
            sitk.WriteImage(out_img, str(mask_path))

            result["status"] = "ok"
            result["mask"] = str(mask_path)
            log.info(f"OK   {patient_id} | organlar: {result['found_organs']}")

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            log.error(f"ERR  {patient_id}: Zaman asimi (10 dk)")
        except Exception as e:
            result["status"] = f"error: {e}"
            log.error(f"ERR  {patient_id}: {e}")

    return result


def main(dataset: str):
    cfg = DATASET_CFG[dataset]
    nifti_dir  = cfg["nifti"]
    seg_out    = cfg["seg"]
    organs     = cfg["organs"]

    if not nifti_dir.exists() or not list(nifti_dir.glob("*.nii.gz")):
        log.error(f"NIfTI dizini bos: {nifti_dir}. Once FAZ 1 calistirin.")
        return

    files = sorted(nifti_dir.glob("*.nii.gz"))
    log.info(f"Dataset: {dataset.upper()} | {len(files)} dosya | Hedef organlar: {organs}")

    results = []
    for i, f in enumerate(files, 1):
        patient_id = f.stem.replace(".nii", "")
        log.info(f"[{i}/{len(files)}] {f.name}")
        r = segment_file(patient_id, f, seg_out, organs, cfg.get("roi_subset", True))
        results.append(r)
        if i % 10 == 0:
            ok = sum(1 for x in results if x["status"] in ("ok", "skipped"))
            log.info(f"  --- Ilerleme: {i}/{len(files)}, {ok} basarili ---")

    ok   = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    fail = len(results) - ok
    log.info(f"\n=== TAMAMLANDI: {ok} basarili, {fail} basarisiz ===")

    report_path = BASE_DIR / f"results/02_{dataset}_segmentation_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "dataset": dataset, "total": len(results),
            "ok": ok, "fail": fail,
            "timestamp": datetime.now().isoformat(),
            "details": results,
        }, f, indent=2)
    log.info(f"Rapor: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CFG.keys()))
    args = parser.parse_args()
    main(args.dataset)
