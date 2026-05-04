"""
FAZ 3: Radyomik Ozellik Cikarimi — PyRadiomics
~107 modalite-bagimsiz ozellik (shape + firstorder + glcm + glrlm + glszm).
Kullanim: python scripts/03_feature_extraction.py --dataset luad
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path("C:/RadyomikAtribuisyon")
CONFIG_PATH = BASE_DIR / "scripts/pyradiomics_config.yaml"

DATASET_CFG = {
    "luad":      {"nifti": BASE_DIR / "nifti/luad",      "seg": BASE_DIR / "segmentations/luad",      "label": "akciger"},
    "lusc":      {"nifti": BASE_DIR / "nifti/lusc",      "seg": BASE_DIR / "segmentations/lusc",      "label": "akciger"},
    "brca":      {"nifti": BASE_DIR / "nifti/brca",      "seg": BASE_DIR / "segmentations/brca",      "label": "meme"},
    "prad":      {"nifti": BASE_DIR / "nifti/prad",      "seg": BASE_DIR / "segmentations/prad",      "label": "prostat"},
    "coad":      {"nifti": BASE_DIR / "nifti/coad",      "seg": BASE_DIR / "segmentations/coad",      "label": "kolon"},
    "brain_mets":{"nifti": BASE_DIR / "nifti/brain_mets","seg": BASE_DIR / "segmentations/brain_mets","label": None},
}

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(BASE_DIR / "results/03_feature_extraction.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

import logging as pyrad_log
pyrad_log.getLogger("radiomics").setLevel(pyrad_log.ERROR)


def extract_patient(patient_id: str, img_path: Path, mask_path: Path, extractor) -> dict | None:
    import SimpleITK as sitk
    import numpy as np

    try:
        img = sitk.ReadImage(str(img_path))
        mask = sitk.ReadImage(str(mask_path))

        # Maske icindeki label degerlerini kontrol et
        arr = sitk.GetArrayFromImage(mask)
        labels = [int(v) for v in np.unique(arr) if v > 0]
        if not labels:
            log.warning(f"SKIP {patient_id}: Maske bos")
            return None

        # Config'deki label=1'i dene, yoksa ilk mevcut labeli kullan
        use_label = 1 if 1 in labels else labels[0]
        extractor.settings["label"] = use_label

        features = extractor.execute(img, mask)
        row = {"patient_id": patient_id}
        for k, v in features.items():
            if not k.startswith("diagnostics_"):
                try:
                    row[k] = float(v)
                except (TypeError, ValueError):
                    pass
        return row

    except Exception as e:
        log.error(f"ERR {patient_id}: {e}")
        return None


def main(dataset: str):
    import radiomics
    from radiomics import featureextractor

    cfg = DATASET_CFG[dataset]
    nifti_dir = cfg["nifti"]
    seg_dir = cfg["seg"]
    out_csv = BASE_DIR / f"features/tcga_{dataset}_features.csv"

    if not nifti_dir.exists():
        print(f"HATA: {nifti_dir} bulunamadi. Once FAZ 1-2 calistirin.")
        return

    extractor = featureextractor.RadiomicsFeatureExtractor(str(CONFIG_PATH))
    print(f"PyRadiomics v{radiomics.__version__} | Dataset: {dataset.upper()}")

    # Daha once islenmisleri atla (restart destegi)
    done_ids = set()
    if out_csv.exists():
        existing = pd.read_csv(out_csv)
        done_ids = set(existing["patient_id"].tolist())
        print(f"  {len(done_ids)} hasta zaten islenmis, devam ediliyor...")

    img_files = sorted(nifti_dir.glob("*.nii.gz"))
    print(f"  Toplam: {len(img_files)} dosya")

    rows = []
    failed = []

    for i, img_path in enumerate(img_files, 1):
        patient_id = img_path.stem.replace(".nii", "")

        if patient_id in done_ids:
            continue

        # Eslesen maske dosyasini bul
        mask_path = seg_dir / f"{patient_id}_mask.nii.gz"
        if not mask_path.exists():
            mask_candidates = list(seg_dir.glob(f"{patient_id}*.nii.gz"))
            if not mask_candidates:
                print(f"  [{i}/{len(img_files)}] SKIP {patient_id}: Maske yok")
                failed.append(patient_id)
                continue
            mask_path = mask_candidates[0]

        row = extract_patient(patient_id, img_path, mask_path, extractor)

        if row:
            if cfg["label"]:
                row["cancer_type"] = cfg["label"]
            rows.append(row)
            if i % 10 == 0:
                print(f"  [{i}/{len(img_files)}] {patient_id} OK — ara kayit yapiliyor")
                # Her 10 hastada ara kayit (crash korumasi)
                df_partial = pd.DataFrame(rows)
                if out_csv.exists():
                    df_partial = pd.concat([pd.read_csv(out_csv), df_partial], ignore_index=True)
                df_partial.to_csv(out_csv, index=False)
                rows = []
        else:
            failed.append(patient_id)

    # Son kayit
    if rows:
        df_new = pd.DataFrame(rows)
        if out_csv.exists():
            df_new = pd.concat([pd.read_csv(out_csv), df_new], ignore_index=True)
        df_new.to_csv(out_csv, index=False)

    total_ok = len(pd.read_csv(out_csv)) if out_csv.exists() else 0
    print(f"\n=== TAMAMLANDI: {total_ok} hasta, {len(failed)} basarisiz ===")
    print(f"Cikti: {out_csv}")

    if failed:
        fail_path = BASE_DIR / f"results/03_{dataset}_failed.json"
        with open(fail_path, "w") as f:
            json.dump(failed, f)
        print(f"Basarisiz hastalar: {fail_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CFG.keys()))
    args = parser.parse_args()
    main(args.dataset)
