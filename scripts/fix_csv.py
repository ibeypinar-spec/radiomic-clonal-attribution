"""CSV duplikat kontrolu ve temizleme"""
import pandas as pd

path = "C:/RadyomikAtribuisyon/features/tcga_luad_features.csv"
df = pd.read_csv(path)
print(f"Ham: {len(df)} satir, {len(df.columns)} sutun, {df['patient_id'].nunique()} benzersiz hasta")

# Duplikatlari kaldir (her hastayi bir kez tut)
df_clean = df.drop_duplicates(subset="patient_id", keep="last")
print(f"Temiz: {len(df_clean)} satir")

# Kaydet
df_clean.to_csv(path, index=False)
print(f"Kaydedildi: {path}")
print(f"\ncancer_type dagılımı:\n{df_clean['cancer_type'].value_counts()}")
print(f"\nNaN sayisi: {df_clean.isna().sum().sum()}")
print(f"Inf sayisi: {(df_clean == float('inf')).sum().sum()}")
