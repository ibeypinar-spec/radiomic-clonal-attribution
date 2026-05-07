"""
FAZ 4a: Model Egitimi — TCGA verilerinden primer imza modeli
LASSO ozellik secimi + XGBoost 5-fold CV + SHAP analizi
Kullanim: python scripts/04_train_model.py
"""

import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

BASE_DIR = Path("C:/RadyomikAtribuisyon")

LABEL_MAP = {
    "akciger": 0,  # LUAD, LUSC
    "meme":    1,  # BRCA
    "kolon":   2,  # COAD
    "prostat": 3,  # PRAD (eger varsa)
}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}

# CSV dosya adi -> cancer_type eslesmesi
FILENAME_TO_TYPE = {
    "tcga_luad": "akciger",
    "tcga_lusc": "akciger",
    "tcga_brca": "meme",
    "tcga_coad": "kolon",
    "tcga_prad": "prostat",
}


def load_and_merge() -> pd.DataFrame:
    dfs = []
    for csv in sorted((BASE_DIR / "features").glob("tcga_*_features.csv")):
        df = pd.read_csv(csv)
        stem = csv.stem.replace("_features", "")  # e.g. tcga_luad

        # cancer_type sutunu yoksa dosya adindan cikar
        if "cancer_type" not in df.columns:
            ctype = FILENAME_TO_TYPE.get(stem)
            if ctype is None:
                print(f"WARN: {csv.name} eslestirilemedi, atlandi")
                continue
            df["cancer_type"] = ctype

        dfs.append(df)
        print(f"  {csv.name}: {len(df)} hasta, type={df['cancer_type'].iloc[0]}")

    if not dfs:
        raise FileNotFoundError("Hicbir TCGA feature CSV bulunamadi. Once FAZ 3 calistirin.")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nToplam: {len(combined)} hasta")
    print(combined["cancer_type"].value_counts())

    # Renumber labels to be contiguous 0..N-1
    present = sorted(combined["cancer_type"].unique())
    global LABEL_MAP, LABEL_NAMES
    LABEL_MAP = {t: i for i, t in enumerate(present)}
    LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}
    print(f"Label map: {LABEL_MAP}")
    return combined


def clean_features(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ("patient_id", "cancer_type")]
    X = df[feature_cols].copy()

    # Sabit kolonlari kaldir
    X = X.loc[:, X.std() > 0]
    # Inf/NaN doldur
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    y = df["cancer_type"].map(LABEL_MAP).values
    return X, y, X.columns.tolist()


def train(X, y, feature_names):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from xgboost import XGBClassifier
    import joblib

    # LASSO ile on-secim
    lasso = LogisticRegression(penalty="l1", solver="saga", C=0.1, max_iter=2000,
                               random_state=42)
    selector = SelectFromModel(lasso, threshold="mean")

    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42,
        device="cuda",          # RTX 5060 kullan
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", selector),
        ("clf", xgb),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("\n5-Fold CV basliyor...")
    scores = cross_validate(pipe, X, y, cv=cv, scoring="balanced_accuracy",
                            return_train_score=True, n_jobs=1)

    print(f"  Train BA: {scores['train_score'].mean():.3f} ± {scores['train_score'].std():.3f}")
    print(f"  Test  BA: {scores['test_score'].mean():.3f} ± {scores['test_score'].std():.3f}")

    # Son model tum veri ile
    pipe.fit(X, y)

    model_path = BASE_DIR / "models/primer_imza.pkl"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "feature_names": feature_names, "label_map": LABEL_MAP}, model_path)
    print(f"\nModel kaydedildi: {model_path}")

    results = {
        "train_ba_mean": float(scores["train_score"].mean()),
        "train_ba_std":  float(scores["train_score"].std()),
        "test_ba_mean":  float(scores["test_score"].mean()),
        "test_ba_std":   float(scores["test_score"].std()),
    }
    with open(BASE_DIR / "results/04_cv_sonuclari.json", "w") as f:
        json.dump(results, f, indent=2)

    return pipe, scores


def shap_analysis(pipe, X, feature_names):
    import shap

    print("\nSHAP analizi yapiliyor...")
    scaler = pipe.named_steps["scaler"]
    selector = pipe.named_steps["selector"]
    clf = pipe.named_steps["clf"]

    X_scaled = scaler.transform(X)
    X_sel = selector.transform(X_scaled)
    sel_names = [feature_names[i] for i in selector.get_support(indices=True)]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_sel)

    # Cok sinifli: ortalama mutlak SHAP
    # shap_values: list (eski SHAP) veya 3D array (yeni SHAP: n_samples x n_features x n_classes)
    if isinstance(shap_values, list):
        mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)
    elif shap_values.ndim == 3:
        mean_shap = np.abs(shap_values).mean(axis=(0, 2))  # samples x features x classes -> features
    else:
        mean_shap = np.abs(shap_values).mean(axis=0)

    mean_shap = mean_shap.flatten()  # kesinlikle 1D
    n_top = min(20, len(sel_names))
    top20_idx = np.argsort(mean_shap)[-n_top:][::-1]
    top20 = [(sel_names[int(i)], float(mean_shap[int(i)])) for i in top20_idx]

    print("\nEn ayirt edici 20 ozellik:")
    for name, val in top20:
        print(f"  {val:.4f}  {name}")

    # SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 8))
    top_names = [t[0] for t in top20]
    top_vals = [t[1] for t in top20]
    ax.barh(range(n_top), top_vals[::-1])
    ax.set_yticks(range(n_top))
    ax.set_yticklabels(top_names[::-1], fontsize=8)
    ax.set_xlabel("Ortalama |SHAP| degeri")
    ax.set_title("En Ayirt Edici 20 Radyomik Ozellik")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "results/04_shap_ozet.png", dpi=150, bbox_inches="tight")
    print(f"SHAP grafigi: {BASE_DIR}/results/04_shap_ozet.png")

    with open(BASE_DIR / "results/04_top20_features.json", "w") as f:
        json.dump(top20, f, indent=2, ensure_ascii=False)


def main():
    print("=== FAZ 4a: Model Egitimi ===")
    df = load_and_merge()

    n_classes = df["cancer_type"].nunique()
    if n_classes < 2:
        print(f"\nUYARI: Sadece {n_classes} sinif mevcut ({df['cancer_type'].unique()}).")
        print("Model egitimi icin en az 2 farkli dataset gerekli.")
        print("Once diger TCGA datasetlerini indirin (brca, prad, coad).")
        print("Mevcut veri kaydedildi, diger datasetler hazir olunca devam edin.")
        return

    X, y, feature_names = clean_features(df)
    print(f"\nOzellik matrisi: {X.shape[0]} hasta x {X.shape[1]} ozellik")
    pipe, scores = train(X, y, feature_names)
    shap_analysis(pipe, X, feature_names)
    print("\nEgitim tamamlandi.")


if __name__ == "__main__":
    main()
