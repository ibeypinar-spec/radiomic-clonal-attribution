import pydicom
import glob
import os

# Check one DICOM from each patient
patients = ["TCGA-EJ-5495", "TCGA-VP-A878", "TCGA-J4-8198"]
for pid in patients:
    pattern = f"C:/RadyomikAtribuisyon/data/tcga_prad/tcga_prad/{pid}/**/*.dcm"
    dcms = glob.glob(pattern, recursive=True)
    if dcms:
        ds = pydicom.dcmread(dcms[0], stop_before_pixels=True)
        modality = getattr(ds, "Modality", "?")
        series_desc = getattr(ds, "SeriesDescription", "?")
        image_type = getattr(ds, "ImageType", "?")
        print(f"{pid}: Modality={modality}, Series={series_desc}, ImageType={image_type}")
    else:
        print(f"{pid}: DICOM bulunamadi")
