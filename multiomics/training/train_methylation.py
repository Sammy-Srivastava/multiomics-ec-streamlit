from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False
    print("[XGBOOST IMPORT ERROR]", repr(e))

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix
from sklearn.impute import SimpleImputer

PROJECT = Path("/Users/samyaksrivastava/Desktop/new science fair thing")

X_PATH = Path("/Users/samyaksrivastava/Desktop/new science fair thing/UI_stuff/artifacts/methylation_gene_agg/M_gene_median_min3.parquet")
Y_PATH = Path("/Users/samyaksrivastava/Desktop/new science fair thing/labels_methylation_binary.csv")
MAP_PATH = Path("/Users/samyaksrivastava/Desktop/new science fair thing/methylation_sample_mapping_SAMPLE_to_GSM.csv")

OUT_OOF = PROJECT / "oof_methylation.csv"
OUT_FEATS = PROJECT / "methylation_feature_importance.csv"

OUT_FI_ALL_FOLDS = PROJECT / "fi_M_all_folds.csv"   #has feature_id, fold, importance

SEED = 42
N_SPLITS = 5

VAR_THRESHOLD = 1e-5
K_BEST = 5000
N_ESTIMATORS = 400

POS_LABEL = "EC"
NEG_LABEL = "ART"
ALLOWED_LABELS = {POS_LABEL, NEG_LABEL}

# ---helpers---
def load_sample_to_gsm_map(path: Path) -> dict[str, str]:
    m = pd.read_csv(path)
    if not {"old_sample", "new_sample"}.issubset(m.columns):
        raise ValueError(f"Expected columns old_sample,new_sample in {path}")
    m["old_sample"] = m["old_sample"].astype(str).str.strip()
    m["new_sample"] = m["new_sample"].astype(str).str.strip()
    return dict(zip(m["old_sample"], m["new_sample"]))

def load_X(path: Path, rename_map: dict[str, str] | None) -> pd.DataFrame:
    X_fxS = pd.read_parquet(path)
    if rename_map is not None:
        X_fxS = X_fxS.rename(columns=lambda c: rename_map.get(str(c).strip(), str(c).strip()))
    X = X_fxS.T
    X.index = X.index.astype(str).str.strip()
    return X

def load_y(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "sample_id" not in df.columns:
        raise ValueError(f"Expected a sample_id column in {path}. Found: {df.columns.tolist()}")

    df["sample_id"] = df["sample_id"].astype(str).str.strip()

    # Case A: string labels present
    label_col = None
    for c in ["label", "group", "phenotype", "status"]:
        if c in df.columns:
            label_col = c
            break

    if label_col is not None:
        df[label_col] = df[label_col].astype(str).str.strip()
        present = set(df[label_col].dropna().unique().tolist())
        print(f"[methylation] label column '{label_col}' uniques: {sorted(present)}")

        df = df[df[label_col].isin(ALLOWED_LABELS)].copy()
        if df.empty:
            raise ValueError(
                f"No rows left after filtering to {sorted(ALLOWED_LABELS)} in column '{label_col}'. "
                f"Found labels: {sorted(present)}"
            )

        y = df.set_index("sample_id")[label_col].map({NEG_LABEL: 0, POS_LABEL: 1}).astype(int)
        y.index = y.index.astype(str).str.strip()
        print(f"[methylation] Using task: {POS_LABEL}(1) vs {NEG_LABEL}(0) | counts={y.value_counts().to_dict()}")
        return y

    # Case B: binary y only
    if "y" not in df.columns:
        raise ValueError(
            f"Expected either a label/group column OR a y column in {path}. Found: {df.columns.tolist()}"
        )

    y = df.set_index("sample_id")["y"].astype(int)
    y.index = y.index.astype(str).str.strip()

    bad = sorted(set(y.unique()) - {0, 1})
    if bad:
        raise ValueError(f"y must be binary 0/1. Found extra values: {bad}")

    print(
        "WARNING: label column not found; using existing binary y as-is. "
        "Make sure this file is already EC vs ART (EC=1, ART=0) or it will not match your project task."
    )
    print(f"[methylation] y counts={y.value_counts().to_dict()}")
    return y

def align_Xy(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    common = X.index.intersection(y.index)
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs between X and y.")
    return X.loc[common].copy(), y.loc[common].copy()

# ---Fold-safe feature filters---
def fit_transform_feature_filters(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    var_thresh: float,
    k_best: int,
    feat_names_full: np.ndarray,
):
    vt = VarianceThreshold(threshold=var_thresh)
    X_tr2 = vt.fit_transform(X_tr)
    X_te2 = vt.transform(X_te)
    feats1 = feat_names_full[vt.get_support()]

    X_tr2 = np.where(np.isfinite(X_tr2), X_tr2, np.nan)
    X_te2 = np.where(np.isfinite(X_te2), X_te2, np.nan)

    imp = SimpleImputer(strategy="median")
    X_tr2 = imp.fit_transform(X_tr2)
    X_te2 = imp.transform(X_te2)

    k = int(min(k_best, X_tr2.shape[1]))
    if k < X_tr2.shape[1]:
        skb = SelectKBest(score_func=f_classif, k=k)
        X_tr2 = skb.fit_transform(X_tr2, y_tr)
        X_te2 = skb.transform(X_te2)
        feats2 = feats1[skb.get_support()]
        return X_tr2, X_te2, feats2, (vt, skb)

    return X_tr2, X_te2, feats1, (vt, None)

# ---xgb oof---
def xgb_cv_oof(X: pd.DataFrame, y: pd.Series):
    if not HAS_XGB:
        raise RuntimeError("xgboost is not installed in this interpreter.")

    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=int)
    feat_names_full = X.columns.to_numpy(dtype=str)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_proba = pd.Series(index=X.index, dtype=float)
    oof_pred = pd.Series(index=X.index, dtype=int)

    aucs, baccs, cms = [], [], []

    fi_rows: list[dict] = []

    for fold, (tr, te) in enumerate(cv.split(X_np, y_np), start=1):
        X_tr, y_tr = X_np[tr], y_np[tr]
        X_te, y_te = X_np[te], y_np[te]

        X_tr2, X_te2, fold_feats, _filters = fit_transform_feature_filters(
            X_tr, y_tr, X_te,
            var_thresh=VAR_THRESHOLD,
            k_best=K_BEST,
            feat_names_full=feat_names_full,
        )

        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        scale_pos_weight = (n_neg / max(1, n_pos))

        model = XGBClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.5,
            reg_lambda=1.0,
            reg_alpha=0.0,
            min_child_weight=1.0,
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            random_state=SEED + fold,
            scale_pos_weight=scale_pos_weight,
        )

        model.fit(X_tr2, y_tr)

        booster = model.get_booster()
        gain = booster.get_score(importance_type="gain")
        for j, feat in enumerate(fold_feats):
            kf = f"f{j}"
            fi_rows.append(
                {"feature_id": str(feat), "fold": int(fold), "importance": float(gain.get(kf, 0.0))}
            )

        proba = model.predict_proba(X_te2)[:, 1]
        pred = (proba >= 0.5).astype(int)

        oof_proba.iloc[te] = proba
        oof_pred.iloc[te] = pred

        aucs.append(roc_auc_score(y_te, proba))
        baccs.append(balanced_accuracy_score(y_te, pred))
        cms.append(confusion_matrix(y_te, pred, labels=[0, 1]))

        print(f"Fold {fold}: AUC={aucs[-1]:.3f} | BalAcc={baccs[-1]:.3f} | p={X_tr2.shape[1]} feats")

    mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs))
    mean_bacc, std_bacc = float(np.mean(baccs)), float(np.std(baccs))
    cm_sum = np.sum(cms, axis=0)

    print("\nCV summary")
    print(f"AUC mean±std: {mean_auc:.3f} + or - {std_auc:.3f}")
    print(f"BalAcc mean±std: {mean_bacc:.3f} + or - {std_bacc:.3f}")
    print("Confusion matrix (rows=true [0,1], cols=pred [0,1]):")
    print(cm_sum)
    
    fi_df = pd.DataFrame(fi_rows)
    fi_df.to_csv(OUT_FI_ALL_FOLDS, index=False)
    print(f"Saved (per-fold selected feature importances): {OUT_FI_ALL_FOLDS}")

    return oof_proba, oof_pred


def fit_final_xgb_and_export_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=int)

    vt = VarianceThreshold(threshold=VAR_THRESHOLD)
    X1 = vt.fit_transform(X_np)

    X1 = np.where(np.isfinite(X1), X1, np.nan)
    imp = SimpleImputer(strategy="median")
    X2 = imp.fit_transform(X1)

    k = int(min(K_BEST, X2.shape[1]))
    skb = None
    if k < X2.shape[1]:
        skb = SelectKBest(score_func=f_classif, k=k)
        X3 = skb.fit_transform(X2, y_np)
    else:
        X3 = X2

    n_pos = int((y_np == 1).sum())
    n_neg = int((y_np == 0).sum())
    scale_pos_weight = (n_neg / max(1, n_pos))

    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=SEED,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X3, y_np)

    feats_vt = X.columns[vt.get_support()]
    feats = feats_vt[skb.get_support()] if skb is not None else feats_vt

    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")

    rows = []
    for i, feat in enumerate(feats):
        kf = f"f{i}"
        rows.append((feat, float(score.get(kf, 0.0))))

    feat_df = pd.DataFrame(rows, columns=["feature", "gain_importance"])
    feat_df["gain_importance"] = feat_df["gain_importance"].fillna(0.0)
    feat_df = feat_df.sort_values("gain_importance", ascending=False)
    return feat_df


def main():
    if not X_PATH.exists():
        raise FileNotFoundError(f"X not found: {X_PATH}")
    if not Y_PATH.exists():
        raise FileNotFoundError(f"Y not found: {Y_PATH}")

    rename_map = load_sample_to_gsm_map(MAP_PATH) if MAP_PATH.exists() else None

    X = load_X(X_PATH, rename_map=rename_map)
    y = load_y(Y_PATH)
    X, y = align_Xy(X, y)

    oof_proba, oof_pred = xgb_cv_oof(X, y)

    out_df = pd.DataFrame(
        {
            "sample_id": oof_proba.index.astype(str),
            "proba": oof_proba.values,
            "pred": oof_pred.values,
            "y": y.loc[oof_proba.index].values,
        }
    )
    out_df.to_csv(OUT_OOF, index=False)
    print(f"\nSaved OOF predictions: {OUT_OOF}")

    feat_df = fit_final_xgb_and_export_features(X, y)
    feat_df.to_csv(OUT_FEATS, index=False)
    print(f"Saved feature importance: {OUT_FEATS}")


if __name__ == "__main__":
    main()
