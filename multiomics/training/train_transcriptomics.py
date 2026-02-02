from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.impute import SimpleImputer


# ----------------------------
# CONFIG
# ----------------------------
PROJECT = Path("/Users/samyaksrivastava/Desktop/new science fair thing")

RUN_DIR = Path("/Users/samyaksrivastava/Desktop/new science fair thing/UI_stuff/artifacts/harmonized/run_20260125_133200")
T_PATH = RUN_DIR / "T_harmonized.parquet"

Y_PATH = PROJECT / "labels_transcriptomics.csv"
MAP_PATH = PROJECT / "transcriptomics_sample_mapping_SAMPLE_to_GSM.csv"

OUT_OOF = PROJECT / "oof_transcriptomics.csv"
OUT_FEATS = PROJECT / "transcriptomics_feature_importance.csv"

# --- NEW: per-fold selected feature importances (stability-ready) ---
OUT_FI_ALL_FOLDS = PROJECT / "fi_T_all_folds.csv"   # feature_id, fold, importance

SEED = 42
N_SPLITS = 5
VAR_THRESHOLD = 1e-8

# model hyperparams
MODEL_C = 0.03
K_BEST = 100


# ----------------------------
# ID utilities
# ----------------------------
def dequote(s: str) -> str:
    s = str(s).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    return s.strip()

def normalize_id(s: str) -> str:
    return dequote(s).replace("\ufeff", "").strip()

def split_suffix(col: str):
    # e.g. EC5_03.signal -> (EC5_03, .signal)
    col = normalize_id(col)
    m = re.match(r"^(.*?)(\.(signal|pvalue))$", col, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)  # base, suffix
    return col, ""

def subject_from_sample_id(s: str) -> str:
    # ART01_plus -> ART01
    return str(s).split("_")[0]


# ----------------------------
# IO
# ----------------------------
def load_sample_map(path: Path) -> dict[str, str]:
    """
    Loads token->GSM mapping. Supports common column pairs:
      old_sample/new_sample OR sample/gsm OR sample_id/gsm OR s/gsm OR token/gsm OR old/new
    """
    m = pd.read_csv(path)
    m = m.rename(columns={c: c.strip().lower() for c in m.columns})

    candidates = [
        ("old_sample", "new_sample"),
        ("sample", "gsm"),
        ("sample_id", "gsm"),
        ("s", "gsm"),
        ("token", "gsm"),
        ("old", "new"),
    ]

    found = None
    for a, b in candidates:
        if a in m.columns and b in m.columns:
            found = (a, b)
            break
    if found is None:
        raise ValueError(f"Mapping file {path} missing expected columns. Found: {m.columns.tolist()}")

    a, b = found
    m = m.dropna(subset=[a, b]).copy()
    m[a] = m[a].astype(str).map(normalize_id)
    m[b] = m[b].astype(str).map(normalize_id)

    # prefer GSM targets when duplicates exist
    m["is_gsm"] = m[b].str.startswith("GSM", na=False)
    m = m.sort_values("is_gsm", ascending=False).drop_duplicates(subset=[a], keep="first")

    return dict(zip(m[a], m[b]))

def load_X(path: Path, rename_map: dict[str, str]) -> pd.DataFrame:
    """
    Reads features x samples parquet and returns samples x features.
    Handles paired channels (.signal/.pvalue) by expanding into feature suffixes:
      geneA__signal, geneA__pvalue, ...
    """
    X_fxS = pd.read_parquet(path)
    X_fxS.columns = X_fxS.columns.astype(str).map(normalize_id)

    # suffix-aware rename: token + .signal/.pvalue -> GSM + .signal/.pvalue
    def renamer(c: str) -> str:
        base, suf = split_suffix(c)
        mapped = rename_map.get(base, base)
        return f"{mapped}{suf}"

    X_fxS = X_fxS.rename(columns=renamer)

    cols = X_fxS.columns.astype(str)
    has_signal = cols.str.endswith(".signal").any()
    has_pvalue = cols.str.endswith(".pvalue").any()

    if has_signal or has_pvalue:
        def split_sc(c: str):
            m = re.match(r"^(.*)\.(signal|pvalue)$", str(c), flags=re.IGNORECASE)
            if m:
                return m.group(1), m.group(2).lower()
            return str(c), "value"

        sc = [split_sc(c) for c in cols]
        mi = pd.MultiIndex.from_tuples(sc, names=["sample", "channel"])
        X_fxS2 = X_fxS.copy()
        X_fxS2.columns = mi

        tmp = X_fxS2.T  # (sample,channel) x features

        parts = []
        for ch in tmp.index.get_level_values("channel").unique():
            part = tmp.xs(ch, level="channel", drop_level=True)  # sample x features
            part.columns = [f"{f}__{ch}" for f in part.columns]
            parts.append(part)

        X = pd.concat(parts, axis=1)
        X.index = X.index.astype(str).map(normalize_id)
        return X

    # non-channel: simple transpose
    X = X_fxS.T
    X.index = X.index.astype(str).map(normalize_id)
    return X

def load_y(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if not {"sample_id", "label"}.issubset(df.columns):
        raise ValueError(f"Labels file must have sample_id,label columns: {path}")

    df["sample_id"] = df["sample_id"].astype(str).map(normalize_id)
    lab = df["label"].astype(str).str.strip()

    # keep EC + ART (drop HC)
    keep = lab.isin(["EC", "ART"])
    df = df.loc[keep].copy()
    if df.empty:
        raise ValueError("No rows with labels in {EC, ART}.")

    # define binary task EC vs ART
    y = df.set_index("sample_id")["label"].map({"ART": 0, "EC": 1}).astype(int)
    y.index = y.index.astype(str).map(normalize_id)
    return y

def align_Xy(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    common = X.index.intersection(y.index)
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs between X and y.")
    X2 = X.loc[common].copy()
    y2 = y.loc[common].copy()
    return X2, y2


# ----------------------------
# Modeling
# ----------------------------
def pick_threshold_max_f1_no_collapse(y_true: np.ndarray, proba: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best = 0.5, -1.0
    for t in thresholds:
        pred = (proba >= t).astype(int)
        if pred.min() == pred.max():
            continue
        f1 = f1_score(y_true, pred, pos_label=1, zero_division=0)
        if f1 > best:
            best, best_t = f1, float(t)
    return best_t


def _fit_transform_with_meta(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    feat_names_full: np.ndarray,
    k_best: int,
):
    """
    Fold-safe preprocessing + returns selected feature names aligned to final columns.
      1) variance filter (train only)
      2) impute
      3) scale
      4) SelectKBest
    """
    vt = VarianceThreshold(threshold=VAR_THRESHOLD)
    X_tr1 = vt.fit_transform(X_tr)
    X_te1 = vt.transform(X_te)
    feats1 = feat_names_full[vt.get_support()]

    X_tr1 = np.where(np.isfinite(X_tr1), X_tr1, np.nan)
    X_te1 = np.where(np.isfinite(X_te1), X_te1, np.nan)

    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr1)
    X_te_imp = imp.transform(X_te1)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr2 = scaler.fit_transform(X_tr_imp)
    X_te2 = scaler.transform(X_te_imp)

    k = int(min(k_best, X_tr2.shape[1]))
    if k < 1:
        raise ValueError("No features survived filtering in this fold.")

    skb = SelectKBest(score_func=f_classif, k=k)
    X_tr3 = skb.fit_transform(X_tr2, y_tr)
    X_te3 = skb.transform(X_te2)
    feats2 = feats1[skb.get_support()]

    return X_tr3, X_te3, feats2


def logistic_cv_oof_tinyn_v2(X: pd.DataFrame, y: pd.Series, k_best: int, C: float):
    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=int)
    feat_names_full = X.columns.to_numpy(dtype=str)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_proba = pd.Series(index=X.index, dtype=float)
    oof_pred = pd.Series(index=X.index, dtype=int)

    aucs, baccs, cms = [], [], []

    # --- NEW: collect per-fold selected-feature importances for stability ---
    fi_rows: list[dict] = []

    for fold, (tr, te) in enumerate(cv.split(X_np, y_np), start=1):
        X_tr, y_tr = X_np[tr], y_np[tr]
        X_te, y_te = X_np[te], y_np[te]

        X_tr3, X_te3, fold_feats = _fit_transform_with_meta(
            X_tr, y_tr, X_te, feat_names_full=feat_names_full, k_best=k_best
        )

        clf = LogisticRegression(
            solver="liblinear",
            C=float(C),
            class_weight="balanced",
            max_iter=5000,
        )
        clf.fit(X_tr3, y_tr)

        # threshold chosen on TRAIN only
        p_tr = clf.predict_proba(X_tr3)[:, 1]
        thr = pick_threshold_max_f1_no_collapse(y_tr, p_tr)

        # --- NEW: per-fold feature importances = abs(coef) mapped to selected features ---
        coefs = clf.coef_.ravel()
        importances = np.abs(coefs)
        for fid, imp in zip(fold_feats, importances):
            fi_rows.append({"feature_id": str(fid), "fold": int(fold), "importance": float(imp)})

        p_te = clf.predict_proba(X_te3)[:, 1]
        pred_te = (p_te >= thr).astype(int)

        oof_proba.iloc[te] = p_te
        oof_pred.iloc[te] = pred_te

        aucs.append(roc_auc_score(y_te, p_te))
        baccs.append(balanced_accuracy_score(y_te, pred_te))
        cms.append(confusion_matrix(y_te, pred_te, labels=[0, 1]))

        print(f"Fold {fold}: thr={thr:.3f} | AUC={aucs[-1]:.3f} | BalAcc={baccs[-1]:.3f}")

    mean_auc, std_auc = float(np.mean(aucs)), float(np.std(aucs))
    mean_bacc, std_bacc = float(np.mean(baccs)), float(np.std(baccs))
    cm_sum = np.sum(cms, axis=0)

    print("\nCV summary")
    print(f"AUC mean±std: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"BalAcc mean±std: {mean_bacc:.3f} ± {std_bacc:.3f}")
    print("Confusion matrix (rows=true [0,1], cols=pred [0,1]):")
    print(cm_sum)

    # --- NEW: save fi_all_folds here so you don't lose it ---
    fi_df = pd.DataFrame(fi_rows)
    fi_df.to_csv(OUT_FI_ALL_FOLDS, index=False)
    print(f"Saved (per-fold selected feature importances): {OUT_FI_ALL_FOLDS}")

    summary = {
        "auc_mean": mean_auc,
        "auc_std": std_auc,
        "bacc_mean": mean_bacc,
        "bacc_std": std_bacc,
        "cm_sum": cm_sum,
        "n": int(X.shape[0]),
        "p": int(X.shape[1]),
    }
    return oof_proba, oof_pred, summary


def fit_final_logistic_and_export_features(X: pd.DataFrame, y: pd.Series, k_best: int, C: float) -> pd.DataFrame:
    vt = VarianceThreshold(threshold=VAR_THRESHOLD)
    X1 = vt.fit_transform(X.to_numpy(dtype=float))

    X1 = np.where(np.isfinite(X1), X1, np.nan)
    imp = SimpleImputer(strategy="median")
    X2 = imp.fit_transform(X1)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X3 = scaler.fit_transform(X2)

    k = int(min(k_best, X3.shape[1]))
    skb = SelectKBest(score_func=f_classif, k=k)
    X4 = skb.fit_transform(X3, y.to_numpy(dtype=int))

    clf = LogisticRegression(
        solver="liblinear",
        C=float(C),
        class_weight="balanced",
        max_iter=5000,
    )
    clf.fit(X4, y.to_numpy(dtype=int))

    # recover feature names
    feats_vt = X.columns[vt.get_support()]
    feats = feats_vt[skb.get_support()]
    coefs = clf.coef_.ravel()

    # NOTE: direction labels fixed for EC vs ART
    feat_df = pd.DataFrame(
        {
            "feature": feats,
            "coef": coefs,
            "abs_coef": np.abs(coefs),
            "direction": np.where(coefs > 0, "EC", "ART"),
        }
    ).sort_values("abs_coef", ascending=False)

    return feat_df


# ----------------------------
# MAIN
# ----------------------------
def main():
    # minimal checks
    if not T_PATH.exists():
        raise FileNotFoundError(f"Transcriptomics matrix not found: {T_PATH}")
    if not Y_PATH.exists():
        raise FileNotFoundError(f"Labels not found: {Y_PATH}")
    if not MAP_PATH.exists():
        raise FileNotFoundError(f"Mapping not found: {MAP_PATH}")

    rename_map = load_sample_map(MAP_PATH)

    X = load_X(T_PATH, rename_map=rename_map)
    y = load_y(Y_PATH)
    X, y = align_Xy(X, y)

    oof_proba, oof_pred, _summary = logistic_cv_oof_tinyn_v2(X, y, k_best=K_BEST, C=MODEL_C)

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

    feat_df = fit_final_logistic_and_export_features(X, y, k_best=K_BEST, C=MODEL_C)
    feat_df.to_csv(OUT_FEATS, index=False)
    print(f"Saved feature importance: {OUT_FEATS}")


if __name__ == "__main__":
    main()
