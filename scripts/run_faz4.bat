@echo off
call C:\Users\Lenovo\anaconda3\Scripts\activate.bat radiomics
python C:\RadyomikAtribuisyon\scripts\04_train_model.py > C:\RadyomikAtribuisyon\results\faz4_output.txt 2>&1
echo Exit code: %errorlevel% >> C:\RadyomikAtribuisyon\results\faz4_output.txt
