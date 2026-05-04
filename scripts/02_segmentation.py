"""
FAZ 2: Otomatik Segmentasyon — TotalSegmentator
GPU ile ~30 sn/hasta, CPU ile ~8 dk/hasta.
Kullanim: python scripts/02_segmentation.py --dataset luad
"""

import os
import subprocess
import argparse
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("C:/RadyomikAtribuisyon")

DATASET_CFG = {
    "luad": {"nifti": BASE_DIR / "nifti/luad",   "seg": BASE_DIR / "segmentations/luad",   "task": "total", "target": "lung_tumor",    "fallback": "lung_upper_lobe_left"},
    "lusc": {"nifti": BASE_DIR / "nifti/lusc",   "seg": BASE_DIR / "segmentations/lusc",   "task": "total", "target": "lung_tumor",    "fallback": "lung_upper_lobe_left"},
    "brca": {"nifti": BASE_DIR / "nifti/brca",   "seg": BASE_DIR / "segmentations/brca",   "task": "total", "target": "breast",        "fallback": "breast"},
    "prad": {"nifti": BASE_DIR / "nifti/prad",   "seg": BASE_DIR / "segmentations/prad",   "task": "total", "target": "prostate",      "fallback": "prostate"},
    "coad": {"nifti": BASE_DIR / "nifti/coad",   "seg": BASE_DIR / "segmentations/coad",   "task": "total", "target": "colon",         "fallback": "small_bowel"},
    "brain_mets": {"nifti": BASE_DIR / "nifti/brain_mets", "seg": BASE_DIR / "segmentations/brain_mets", "task": "total", "target": "brain", "fallback": "brain"},
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


def segment_file(nifti_path: Path, seg_out: Path, task: str, target: str, fallback: str) -> dict:
    patient_id = nifti_path.stem.replace(".nii", "")
    mask_path = seg_out / f"{patient_id}_mask.nii.gz"

    if mask_path.exists():
        log.info(f"SKIP {patient_id} (maske zaten var)")
        return {"patient_id": patient_id, "status": "skipped", "mask": str(mask_path)}

    result = {"patient_id": patient_id, "status": None, "mask": None}
    seg_out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            "TotalSegmentator",
            "-i", str(nifti_path),
            "-o", tmp,
            "--task", task,
            "--ml",             # multi-label (tek dosya tum organlar)
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            # Hedef organ maskesini bul
            target_file = Path(tmp) / f"{target}.nii.gz"
            if not target_file.exists():
                target_file = Path(tmp) / f"{fallback}.nii.gz"
                if target_file.exists():
                    log.warning(f"WARN {patient_id}: '{target}' yok, '{fallback}' kullanildi")
                else:
                    result["status"] = "no_mask"
                    log.error(f"ERR  {patient_id}: Maske bulunamadi")
                    return result

            import shutil
            shutil.copy2(target_file, mask_path)
            result["status"] = "ok"
            result["mask"] = str(mask_path)
            log.info(f"OK   {patient_id} -> {mask_path.name}")

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            log.error(f"ERR  {patient_id}: Zaman asimi (10 dk)")
        except Exception as e:
            result["status"] = f"error: {e}"
            log.error(f"ERR  {patient_id}: {e}")

    return result


def main(dataset: str):
    cfg = DATASET_CFG[dataset]
    nifti_dir = cfg["nifti"]

    if not nifti_dir.exists() or not list(nifti_dir.glob("*.nii.gz")):
        log.error(f"NIfTI dizini bos: {nifti_dir}")
        log.error("Once 01_dicom_to_nifti.py calistirin.")
        return

    files = sorted(nifti_dir.glob("*.nii.gz"))
    log.info(f"Dataset: {dataset.upper()} | {len(files)} dosya bulundu")

    results = []
    for i, f in enumerate(files, 1):
        log.info(f"[{i}/{len(files)}] {f.name}")
        r = segment_file(f, cfg["seg"], cfg["task"], cfg["target"], cfg["fallback"])
        results.append(r)
        if i % 10 == 0:
            ok = sum(1 for x in results if x["status"] == "ok")
            log.info(f"  --- Ilerleme: {i}/{len(files)}, {ok} basarili ---")

    ok = sum(1 for r in results if r["status"] in ("ok", "skipped"))
    fail = len(results) - ok
    log.info(f"\n=== TAMAMLANDI: {ok} basarili, {fail} basarisiz ===")

    report_path = BASE_DIR / f"results/02_{dataset}_segmentation_report.json"
    with open(report_path, "w") as f:
        json.dump({"dataset": dataset, "total": len(results), "ok": ok, "fail": fail,
                   "timestamp": datetime.now().isoformat(), "details": results}, f, indent=2)
    log.info(f"Rapor: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CFG.keys()))
    args = parser.parse_args()
    main(args.dataset)
