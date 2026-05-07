@echo off
echo Breasts modeli indiriliyor...
call C:\Users\Lenovo\anaconda3\Scripts\activate.bat radiomics
TotalSegmentator -i "C:\RadyomikAtribuisyon\nifti\brca\TCGA-AO-A03M.nii.gz" -o "C:\RadyomikAtribuisyon\segmentations\brca\test_tmp" --task breasts > C:\RadyomikAtribuisyon\results\brca_model_download.log 2>&1
echo Bitti: %errorlevel% >> C:\RadyomikAtribuisyon\results\brca_model_download.log
