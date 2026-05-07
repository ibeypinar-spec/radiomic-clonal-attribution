@echo off
echo MRI model indiriliyor (total_mr, ~232MB)...
call C:\Users\Lenovo\anaconda3\Scripts\activate.bat radiomics
TotalSegmentator -i "C:\RadyomikAtribuisyon\nifti\prad\TCGA-EJ-5495.nii.gz" -o "C:\RadyomikAtribuisyon\segmentations\prad\test_tmp" --task total_mr --roi_subset prostate > C:\RadyomikAtribuisyon\results\prad_model_download.log 2>&1
echo Bitti: %errorlevel% >> C:\RadyomikAtribuisyon\results\prad_model_download.log
