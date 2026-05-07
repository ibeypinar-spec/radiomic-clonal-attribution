import pydicom
import glob
import os

prad_dir = "C:/RadyomikAtribuisyon/data/tcga_prad/tcga_prad"
for pid_dir in sorted(os.listdir(prad_dir)):
    patient_path = os.path.join(prad_dir, pid_dir)
    if not os.path.isdir(patient_path):
        continue
    dcms = glob.glob(os.path.join(patient_path, "**/*.dcm"), recursive=True)
    if dcms:
        ds = pydicom.dcmread(dcms[0], stop_before_pixels=True)
        modality = getattr(ds, "Modality", "?")
        series_desc = getattr(ds, "SeriesDescription", "?")
        print(f"{pid_dir}: {modality} | {series_desc}")
    else:
        print(f"{pid_dir}: DICOM yok")
