@echo off
call C:\Users\Lenovo\anaconda3\Scripts\activate.bat radiomics
python C:\RadyomikAtribuisyon\scripts\03_feature_extraction.py --dataset luad > C:\RadyomikAtribuisyon\results\faz3_output.txt 2>&1
echo Exit code: %errorlevel% >> C:\RadyomikAtribuisyon\results\faz3_output.txt
