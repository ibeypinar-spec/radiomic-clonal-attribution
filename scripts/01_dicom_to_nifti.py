"""
FAZ 1: DICOM -> NIfTI Donusumu
Her TCGA dataset icin DICOM klasorlerini tara ve NIfTI'ye donustur.
Kullanim: python scripts/01_dicom_to_nifti.py --dataset luad
"""

import os
import subprocess
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

BASE_DIR = Path("C:/RadyomikAtribuisyon")

DATASETS = {
    "luad": {"dicom": BASE_DIR / "data/tcga_luad", "nifti": BASE_DIR / "nifti/luad"},
    "lusc": {"dicom": BASE_DIR / "data/tcga_lusc", "nifti": BASE_DIR / "nifti/lusc"},
    "brca": {"dicom": BASE_DIR / "data/tcga_brca", "nifti": BASE_DIR / "nifti/brca"},
    "prad": {"dicom": BASE_DIR / "data/tcga_prad", "nifti": BASE_DIR / "nifti/prad"},
    "coad": {"dicom": BASE_DIR / "data/tcga_coad", "nifti": BASE_DIR / "nifti/coad"},
    "brain_mets": {"dicom": BASE_DIR / "data/brain_mets", "nifti": BASE_DIR / "nifti/brain_mets"},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "results/01_dicom_to_nifti.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def find_dicom_series(dicom_root: Path) -> list[Path]:
    """Her alt klasoru bir hasta serisi olarak kabul et."""
    series = []
    for item in sorted(dicom_root.iterdir()):
        if item.is_dir():
            dcm_files = list(item.rglob("*.dcm"))
            if not dcm_files:
                # Alt-alt klasorlere bak (TCIA yapisi: hasta/seri/dosyalar)
                sub_dirs = [d for d in item.iterdir() if d.is_dir()]
                for sub in sub_dirs:
                    if list(sub.rglob("*.dcm")):
                        series.append(sub)
            else:
                series.append(item)
    return series


def convert_series(series_dir: Path, output_dir: Path) -> dict:
    """Tek bir DICOM serisini NIfTI'ye donustur."""
    patient_id = series_dir.parent.name  # TCGA-XX-XXXX formatı
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {"patient_id": patient_id, "series": series_dir.name, "status": None, "output": None}

    cmd = [
        "dcm2niix",
        "-o", str(output_dir),
        "-f", f"{patient_id}",   # dosya adı = hasta ID
        "-z", "y",               # gzip sıkıştırma
        "-m", "y",               # birden fazla seriyi birleştir
        str(series_dir)
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        nifti_files = list(output_dir.glob(f"{patient_id}*.nii.gz"))

        if nifti_files:
            result["status"] = "ok"
            result["output"] = str(nifti_files[0])
            log.info(f"OK  {patient_id} -> {nifti_files[0].name}")
        else:
            result["status"] = "no_output"
            result["stderr"] = proc.stderr[-500:]
            log.warning(f"WARN {patient_id}: NIfTI olusturulamadi")
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        log.error(f"ERR  {patient_id}: Zaman asimi")
    except FileNotFoundError:
        log.error("dcm2niix bulunamadi! 'pip install dcm2niix' calistirin.")
        raise

    return result


def main(dataset: str):
    if dataset not in DATASETS:
        raise ValueError(f"Gecersiz dataset: {dataset}. Secenekler: {list(DATASETS.keys())}")

    cfg = DATASETS[dataset]
    dicom_root = cfg["dicom"]
    nifti_out = cfg["nifti"]

    if not dicom_root.exists():
        log.error(f"DICOM dizini bulunamadi: {dicom_root}")
        log.error(f"Once TCIA'dan {dataset.upper()} verisini indirin ve {dicom_root} altina koyun.")
        return

    series_list = find_dicom_series(dicom_root)
    log.info(f"Dataset: {dataset.upper()} | {len(series_list)} seri bulundu")

    results = []
    for i, series in enumerate(series_list, 1):
        log.info(f"[{i}/{len(series_list)}] {series.name}")
        r = convert_series(series, nifti_out)
        results.append(r)

    # Ozet
    ok = sum(1 for r in results if r["status"] == "ok")
    fail = len(results) - ok
    log.info(f"\n=== TAMAMLANDI: {ok} basarili, {fail} basarisiz ===")

    # JSON rapor kaydet
    report_path = BASE_DIR / f"results/01_{dataset}_conversion_report.json"
    with open(report_path, "w") as f:
        json.dump({"dataset": dataset, "total": len(results), "ok": ok, "fail": fail,
                   "timestamp": datetime.now().isoformat(), "details": results}, f, indent=2)
    log.info(f"Rapor: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DICOM -> NIfTI donusumu")
    parser.add_argument("--dataset", required=True, choices=list(DATASETS.keys()),
                        help="Donusturulecek dataset")
    args = parser.parse_args()
    main(args.dataset)
