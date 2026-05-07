@echo off
set DATASET=%1
call C:\Users\Lenovo\anaconda3\Scripts\activate.bat radiomics
echo === FAZ 2: Segmentasyon [%DATASET%] ===
python C:\RadyomikAtribuisyon\scripts\02_segmentation.py --dataset %DATASET% >> C:\RadyomikAtribuisyon\results\pipeline_%DATASET%.log 2>&1
echo === FAZ 3: Ozellik Cikarimi [%DATASET%] ===
python C:\RadyomikAtribuisyon\scripts\03_feature_extraction.py --dataset %DATASET% >> C:\RadyomikAtribuisyon\results\pipeline_%DATASET%.log 2>&1
python C:\RadyomikAtribuisyon\scripts\fix_csv.py %DATASET% >> C:\RadyomikAtribuisyon\results\pipeline_%DATASET%.log 2>&1
echo === TAMAMLANDI [%DATASET%] === >> C:\RadyomikAtribuisyon\results\pipeline_%DATASET%.log
