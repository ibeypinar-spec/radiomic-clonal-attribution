import pydicom
import glob
import os

brca_dir = "C:/RadyomikAtribuisyon/data/tcga_brca/tcga_brca"
patients = sorted(os.listdir(brca_dir))[:5]
for pid in patients:
    dcms = glob.glob(f"{brca_dir}/{pid}/**/*.dcm", recursive=True)
    if dcms:
        ds = pydicom.dcmread(dcms[0], stop_before_pixels=True)
        modality = getattr(ds, "Modality", "?")
        series = getattr(ds, "SeriesDescription", "?")
        print(f"{pid}: {modality} | {series}")
