# Radiomic Clonal Attribution

**Assigning brain metastases to primary tumors using radiomic signatures**

*AAKÜ Medical Faculty — Medical Oncology | İsmail Beypınar | v2.0*

## Overview

This project trains a radiomic signature model on large TCGA primary tumor datasets and validates it on a brain metastasis cohort (Pretreat-MetsToBrain-Masks, n=200) to attribute each metastasis to its most likely primary tumor type.

### Strategy

- **Train:** TCGA primary tumors (lung, breast, prostate, colon) → cancer-type-specific radiomic signatures
- **Test:** Brain metastasis MRI (Yale New Haven) → signature matching accuracy

This provides true external validation — the model never sees metastasis data during training.

## Datasets

| Cancer Type | TCGA Collection | Modality | Est. N |
|-------------|----------------|----------|--------|
| Lung (NSCLC) | TCGA-LUAD + TCGA-LUSC | CT | ~500+ |
| Breast | TCGA-BRCA | MRI | ~139 |
| Prostate | TCGA-PRAD | MRI | ~100 |
| Colon | TCGA-COAD | CT | ~188 |
| **Test set** | Pretreat-MetsToBrain-Masks | MRI | 200 |

## Pipeline

```
FAZ 1: DICOM → NIfTI      (dcm2niix)
FAZ 2: Segmentation        (TotalSegmentator)
FAZ 3: Feature extraction  (PyRadiomics, ~107 features, no wavelet)
FAZ 4: Model train + test  (LASSO + XGBoost + SHAP)
```

## Directory Structure

```
C:\RadyomikAtribuisyon\
├── data\           # Raw DICOM (not committed)
├── nifti\          # Converted NIfTI (not committed)
├── segmentations\  # TotalSegmentator masks (not committed)
├── features\       # PyRadiomics CSV outputs (not committed)
├── models\         # Trained models (not committed)
├── results\        # Figures and tables
├── scripts\        # Python scripts
└── notebooks\      # Exploratory notebooks
```

## Feature Strategy

Modality-independent features only:
- `original_shape` (14), `original_firstorder` (18), `original_glcm` (24), `original_glrlm` (16), `original_glszm` (16)
- **Wavelet features excluded** — too modality-sensitive

## Minimum Success Criteria

| Metric | Target |
|--------|--------|
| TCGA CV balanced accuracy | > 0.70 |
| Brain met. validation AUC | > 0.65 |
| NSCLC detection AUC | > 0.70 |
| Breast detection AUC | > 0.65 |

## Setup

```bash
conda activate radiomics
cd C:\RadyomikAtribuisyon
pip install pyradiomics dcm2niix TotalSegmentator xgboost shap
```

## Target Journals

- Cancer Imaging (IF ~5.0)
- Cancers — MDPI (IF ~5.2)
- European Radiology (IF ~7.0)
