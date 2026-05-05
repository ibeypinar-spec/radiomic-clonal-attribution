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

def find_tcia_root(dicom_root: Path) -> Path:
    """TCIA, indirilen veriyi ayni isimde alt klasore koyar: data/tcga_luad/tcga_luad/
    Bu fonksiyon gercek hasta klasorlerinin bulundugu dizini bulur."""
    # TCGA-XX-XXXX formatinda klasor var mi direkt bak
    patient_dirs = [d for d in dicom_root.iterdir() if d.is_dir() and d.name.startswith("TCGA-")]
    if patient_dirs:
        return dicom_root
    # Bir alt klasorde ara (TCIA yapisi)
    for sub in dicom_root.iterdir():
        if sub.is_dir() and sub.name != "metadata":
            patient_dirs = [d for d in sub.iterdir() if d.is_dir() and d.name.startswith("TCGA-")]
            if patient_dirs:
                return sub
    return dicom_root

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "results/01_dicom_to_nifti.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


def find_dicom_series(dicom_root: Path) -> list[tuple[str, Path]]:
    """TCIA yapisi: hasta_klasoru/StudyUID/SeriesUID/*.dcm
    Her hasta icin en uygun (en fazla DCM iceren) seri klasorunu dondur.
    Returns: [(patient_id, series_dir), ...]"""
    patient_root = find_tcia_root(dicom_root)
    series = []
    for patient_dir in sorted(patient_root.iterdir()):
        if not patient_dir.is_dir() or not patient_dir.name.startswith("TCGA-"):
            continue
        patient_id = patient_dir.name
        # En fazla DCM iceren seriyi sec
        best_series = None
        best_count = 0
        for series_dir in patient_dir.rglob("*"):
            if series_dir.is_dir():
                dcm_count = len(list(series_dir.glob("*.dcm")))
                if dcm_count > best_count:
                    best_count = dcm_count
                    best_series = series_dir
        if best_series and best_count > 0:
            series.append((patient_id, best_series))
    return series


def convert_series(patient_id: str, series_dir: Path, output_dir: Path) -> dict:
    """Tek bir DICOM serisini NIfTI'ye donustur."""
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
    log.info(f"Dataset: {dataset.upper()} | {len(series_list)} hasta bulundu")

    results = []
    for i, (patient_id, series) in enumerate(series_list, 1):
        # Zaten donusturulmusse atla
        if list(nifti_out.glob(f"{patient_id}*.nii.gz")):
            log.info(f"[{i}/{len(series_list)}] SKIP {patient_id}")
            results.append({"patient_id": patient_id, "status": "skipped"})
            continue
        log.info(f"[{i}/{len(series_list)}] {patient_id}")
        r = convert_series(patient_id, series, nifti_out)
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
