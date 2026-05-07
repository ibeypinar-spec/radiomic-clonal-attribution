"""
FAZ 4b: Validasyon — Beyin Metastazlarinda Test
Egitilen primer imza modeli beyin mets kohortuna uygulanir.
Kullanim: python scripts/05_validate.py
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

TR_NAMES = {
    "akciger": "Akciğer", "meme": "Meme",
    "kolon": "Kolon", "prostat": "Prostat",
}
ALL_COLORS = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]


def main():
    import joblib
    from sklearn.metrics import (balanced_accuracy_score, roc_auc_score,
                                  confusion_matrix, classification_report)
    from sklearn.preprocessing import label_binarize

    print("=== FAZ 4b: Beyin Metastazi Validasyonu ===")

    # Model yukle
    model_path = BASE_DIR / "models/primer_imza.pkl"
    if not model_path.exists():
        print("HATA: Model bulunamadi. Once 04_train_model.py calistirin.")
        return
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    feature_names = bundle["feature_names"]
    label_map = bundle["label_map"]

    # Dinamik label setup
    LABEL_NAMES = {v: k for k, v in label_map.items()}
    LABEL_TR = {v: TR_NAMES.get(k, k.capitalize()) for k, v in label_map.items()}
    COLORS = ALL_COLORS[:len(label_map)]

    # Brain mets feature yukle
    bm_csv = BASE_DIR / "features/tcga_brain_mets_features.csv"
    if not bm_csv.exists():
        # Eski Jupyter'dan gelen all_features.csv'yi de dene
        alt_paths = list(BASE_DIR.rglob("all_features*.csv"))
        if alt_paths:
            bm_csv = alt_paths[0]
            print(f"  Alternatif feature dosyasi kullaniliyor: {bm_csv}")
        else:
            print("HATA: Brain mets feature CSV bulunamadi.")
            print("  Beklenen: features/tcga_brain_mets_features.csv")
            print("  Veya eski Jupyter: all_features.csv")
            return

    df = pd.read_csv(bm_csv)
    print(f"  Brain mets: {len(df)} hasta")

    if "cancer_type" not in df.columns:
        print("HATA: 'cancer_type' kolonu yok — gercek etiketler olmadan ROC hesaplanamaz.")
        # Etiket olmadan sadece tahmin yap
        X_bm = df[[c for c in df.columns if c in feature_names]].fillna(0)
        proba = pipe.predict_proba(X_bm)
        pred_labels = pipe.predict(X_bm)
        df["predicted"] = [LABEL_NAMES[p] for p in pred_labels]
        for i, name in LABEL_NAMES.items():
            df[f"prob_{name}"] = proba[:, i]
        out = BASE_DIR / "results/05_tahminler.csv"
        df.to_csv(out, index=False)
        print(f"Tahminler kaydedildi: {out}")
        return

    # Etiket varsa tam validasyon
    y_true = df["cancer_type"].map(label_map).values
    X_bm = df[[c for c in feature_names if c in df.columns]].copy()
    X_bm = X_bm.reindex(columns=feature_names, fill_value=0)
    X_bm = X_bm.replace([np.inf, -np.inf], np.nan).fillna(X_bm.median())

    y_pred = pipe.predict(X_bm)
    y_proba = pipe.predict_proba(X_bm)

    ba = balanced_accuracy_score(y_true, y_pred)
    print(f"\nBalanced Accuracy: {ba:.3f}")
    print("\nSiniflandirma Raporu:")
    print(classification_report(y_true, y_pred, target_names=list(LABEL_NAMES.values())))

    # Per-class AUC
    classes = sorted(label_map.values())
    y_bin = label_binarize(y_true, classes=classes)
    aucs = {}
    for i, cls in enumerate(classes):
        try:
            auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
            aucs[LABEL_NAMES[cls]] = round(auc, 3)
            print(f"  AUC {LABEL_NAMES[cls]}: {auc:.3f}")
        except Exception:
            pass

    # ROC egrileri
    fig, ax = plt.subplots(figsize=(8, 6))
    from sklearn.metrics import roc_curve
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_val = aucs.get(LABEL_NAMES[cls], 0)
        ax.plot(fpr, tpr, color=COLORS[i], lw=2,
                label=f"{LABEL_TR[cls]} (AUC={auc_val:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("1 - Özgüllük (FPR)")
    ax.set_ylabel("Duyarlılık (TPR)")
    ax.set_title("Beyin Metastazı Validasyonu — ROC Eğrileri")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "results/05_validasyon_roc.png", dpi=150, bbox_inches="tight")
    print(f"\nROC grafigi: {BASE_DIR}/results/05_validasyon_roc.png")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks(range(len(classes)))
    ax2.set_yticks(range(len(classes)))
    ax2.set_xticklabels([LABEL_TR[c] for c in classes])
    ax2.set_yticklabels([LABEL_TR[c] for c in classes])
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax2.set_xlabel("Tahmin")
    ax2.set_ylabel("Gerçek")
    ax2.set_title("Confusion Matrix")
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "results/05_confusion_matrix.png", dpi=150, bbox_inches="tight")
    print(f"Confusion matrix: {BASE_DIR}/results/05_confusion_matrix.png")

    # JSON ozet
    summary = {
        "balanced_accuracy": round(ba, 3),
        "per_class_auc": aucs,
        "n_patients": int(len(df)),
        "n_classes": len(classes),
    }
    with open(BASE_DIR / "results/05_validasyon_ozet.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nOzet: {BASE_DIR}/results/05_validasyon_ozet.json")


if __name__ == "__main__":
    main()
