import pandas as pd

for dataset in ["luad", "coad"]:
    try:
        df = pd.read_csv(f"C:/RadyomikAtribuisyon/features/tcga_{dataset}_features.csv")
        print(f"{dataset.upper()}: {df.shape[0]} hasta, {df.shape[1]} sutun")
        print(f"  Columns sample: {list(df.columns[:5])}")
        if "label" in df.columns:
            print(f"  Label: {df.label.unique()}")
    except Exception as e:
        print(f"{dataset}: {e}")
