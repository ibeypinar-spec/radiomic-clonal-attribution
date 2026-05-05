# Radiomic Clonal Attribution — Claude Code Kılavuzu

## Proje Amacı
Beyin metastazlarını primer tümörlerine (akciğer/meme/prostat/kolon) radyomik imza eşleştirmesiyle atamak.
- **Eğitim:** TCGA primer tümörleri (~900 hasta)
- **Test:** Pretreat-MetsToBrain-Masks (200 beyin metastazı, Yale)

## Ortam
```bash
conda activate radiomics
cd C:\RadyomikAtribuisyon
```

## Pipeline — Çalıştırma Sırası

| Adım | Komut | Süre (tahmini) |
|------|-------|----------------|
| FAZ 1 — DICOM→NIfTI | `python scripts/01_dicom_to_nifti.py --dataset luad` | ~5 dk/dataset |
| FAZ 2 — Segmentasyon | `python scripts/02_segmentation.py --dataset luad` | ~35 dk/dataset (GPU) |
| FAZ 3 — Özellik çıkarımı | `python scripts/03_feature_extraction.py --dataset luad` | ~20 dk/dataset |
| FAZ 4a — Model eğitimi | `python scripts/04_train_model.py` | ~10 dk |
| FAZ 4b — Validasyon | `python scripts/05_validate.py` | ~2 dk |

**Dataset seçenekleri:** `luad`, `lusc`, `brca`, `prad`, `coad`, `brain_mets`

## Dizin Yapısı
```
data/tcga_luad/tcga_luad/TCGA-XX-XXXX/[StudyUID]/[SeriesUID]/*.dcm  ← TCIA yapısı
nifti/luad/TCGA-XX-XXXX.nii.gz
segmentations/luad/TCGA-XX-XXXX_mask.nii.gz
features/tcga_luad_features.csv    ← cancer_type sütunu: "akciger"
models/primer_imza.pkl
results/                           ← ROC, SHAP, confusion matrix
```

## Özellik Stratejisi
- **Kullanılan:** shape (14), firstorder (18), glcm (24), glrlm (16), glszm (16) → ~107 özellik
- **Kullanılmayan:** Wavelet — modaliteye çok duyarlı
- **Config:** `scripts/pyradiomics_config.yaml`

## Etiket Haritası
```python
{"akciger": 0, "meme": 1, "prostat": 2, "kolon": 3}
# luad + lusc → "akciger"
```

## GPU Bilgisi
- NVIDIA GeForce RTX 5060, 8 GB VRAM, CUDA 13.0
- TotalSegmentator GPU'yu otomatik kullanır
- XGBoost: `device="cuda"` ile eğitilir

## Başarı Kriterleri
| Metrik | Hedef |
|--------|-------|
| TCGA CV balanced accuracy | > 0.70 |
| Beyin met. validasyon AUC | > 0.65 |
| NSCLC AUC | > 0.70 |
| Meme AUC | > 0.65 |

## Hedef Dergiler
1. Cancer Imaging (IF ~5.0)
2. Cancers / MDPI (IF ~5.2)
3. European Radiology (IF ~7.0)

## Sık Karşılaşılan Sorunlar
- `lung_tumor` maskesi bulunamazsa `lung_lower_lobe_left` fallback kullanılır
- Özellik çıkarımı crash olursa script kaldığı yerden devam eder (ara kayıt)
- XGBoost CUDA hatası → `device="cpu"` ile deneyin
