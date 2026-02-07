from __future__ import annotations

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.impute import SimpleImputer

DEFAULT_VAR_THRESHOLD = 1e-8
DEFAULT_MODEL_C = 0.03
DEFAULT_K_BEST = 100

# ---ID utilities---
def dequote(s: str) -> str:
    s = str(s).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    return s.strip()


def normalize_id(s: str) -> str:
    return dequote(s).replace("\ufeff", "").strip()


def split_suffix(col: str):
    col = normalize_id(col)
    m = re.match(r"^(.*?)(\.(signal|pvalue))$", col, flags=re.IGNORECASE)
    if m:
        return m.group(1), m.group(2)
    return col, ""

def load_sample_map(path: Path) -> dict[str, str]:
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


def load_X(path: Path, rename_map: dict[str, str] | None) -> pd.DataFrame:
    X_fxS = pd.read_parquet(path)
    X_fxS.columns = X_fxS.columns.astype(str).map(normalize_id)

    if rename_map:
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

        tmp = X_fxS2.T

        parts = []
        for ch in tmp.index.get_level_values("channel").unique():
            part = tmp.xs(ch, level="channel", drop_level=True)
            part.columns = [f"{f}__{ch}" for f in part.columns]
            parts.append(part)

        X = pd.concat(parts, axis=1)
        X.index = X.index.astype(str).map(normalize_id)
        return X

    # transpose
    X = X_fxS.T
    X.index = X.index.astype(str).map(normalize_id)
    return X


def load_y(path: Path, pos_label: str = "EC", neg_label: str = "ART") -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}

    sid_col = cols.get("sample_id") or cols.get("sample") or cols.get("id")
    if sid_col is None:
        raise ValueError(f"Labels file must include sample_id. Found: {df.columns.tolist()}")

    # prefer y if present or else label/target
    y_col = cols.get("y") or cols.get("label") or cols.get("target")
    if y_col is None:
        raise ValueError(f"Labels file must include y or label. Found: {df.columns.tolist()}")

    df[sid_col] = df[sid_col].astype(str).map(normalize_id)

    y_raw = df[y_col]

    y_num = pd.to_numeric(y_raw, errors="coerce")
    if y_num.notna().mean() > 0.95:
        y = y_num.dropna().astype(int)
        bad = sorted(set(y.unique()) - {0, 1})
        if bad:
            raise ValueError(f"Numeric y must be binary 0/1. Found extra values: {bad}")
        out = pd.Series(y.values, index=df.loc[y.index, sid_col].values, dtype=int)
        out.index = out.index.astype(str).map(normalize_id)
        return out

    # string labels
    lab = y_raw.astype(str).str.strip()
    keep = lab.isin([pos_label, neg_label])
    df = df.loc[keep].copy()
    if df.empty:
        raise ValueError(f"No rows with labels in {{{pos_label}, {neg_label}}}.")

    mapping = {neg_label: 0, pos_label: 1}
    out = df.set_index(sid_col)[y_col].astype(str).str.strip().map(mapping).astype(int)
    out.index = out.index.astype(str).map(normalize_id)
    return out


def align_Xy(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    common = X.index.intersection(y.index)
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs between X and y.")
    return X.loc[common].copy(), y.loc[common].copy()

# ---Modeling helpers---
def _safe_n_splits(y_np: np.ndarray, requested: int) -> int:
    n_pos = int((y_np == 1).sum())
    n_neg = int((y_np == 0).sum())
    max_possible = max(2, min(n_pos, n_neg))
    return int(min(int(requested), max_possible))


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
    var_threshold: float,
):
    vt = VarianceThreshold(threshold=float(var_threshold))
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

    k = int(min(int(k_best), X_tr2.shape[1]))
    if k < 1:
        raise ValueError("No features survived filtering in this fold.")

    skb = SelectKBest(score_func=f_classif, k=k)
    X_tr3 = skb.fit_transform(X_tr2, y_tr)
    X_te3 = skb.transform(X_te2)
    feats2 = feats1[skb.get_support()]

    return X_tr3, X_te3, feats2


def logistic_cv_oof(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int,
    seed: int,
    k_best: int,
    model_c: float,
    var_threshold: float,
    out_fi_all_folds: Path,
):
    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=int)
    feat_names_full = X.columns.to_numpy(dtype=str)

    n_splits_safe = _safe_n_splits(y_np, n_splits)
    if n_splits_safe != int(n_splits):
        print(f"Adjusted n_splits {n_splits} -> {n_splits_safe} due to class counts.")
    cv = StratifiedKFold(n_splits=n_splits_safe, shuffle=True, random_state=int(seed))

    oof_proba = pd.Series(index=X.index, dtype=float)
    oof_pred = pd.Series(index=X.index, dtype=int)

    aucs, baccs, cms = [], [], []
    fi_rows: list[dict] = []

    for fold, (tr, te) in enumerate(cv.split(X_np, y_np), start=1):
        X_tr, y_tr = X_np[tr], y_np[tr]
        X_te, y_te = X_np[te], y_np[te]

        X_tr3, X_te3, fold_feats = _fit_transform_with_meta(
            X_tr,
            y_tr,
            X_te,
            feat_names_full=feat_names_full,
            k_best=int(k_best),
            var_threshold=float(var_threshold),
        )

        clf = LogisticRegression(
            solver="liblinear",
            C=float(model_c),
            class_weight="balanced",
            max_iter=5000,
        )
        clf.fit(X_tr3, y_tr)

        # threshold chosen on train only
        p_tr = clf.predict_proba(X_tr3)[:, 1]
        thr = pick_threshold_max_f1_no_collapse(y_tr, p_tr)

        # per-fold feature importances = abs(coef)
        coefs = clf.coef_.ravel()
        importances = np.abs(coefs)
        for fid, imp in zip(fold_feats, importances):
            fi_rows.append({"feature_id": str(fid), "fold": int(fold), "importance": float(imp)})

        p_te = clf.predict_proba(X_te3)[:, 1]
        pred_te = (p_te >= thr).astype(int)

        oof_proba.iloc[te] = p_te
        oof_pred.iloc[te] = pred_te

        aucs.append(float(roc_auc_score(y_te, p_te)) if len(np.unique(y_te)) >= 2 else float("nan"))
        baccs.append(float(balanced_accuracy_score(y_te, pred_te)))
        cms.append(confusion_matrix(y_te, pred_te, labels=[0, 1]))

        print(f"[T] Fold {fold}: thr={thr:.3f} | AUC={aucs[-1]:.3f} | BalAcc={baccs[-1]:.3f}")

    mean_auc, std_auc = float(np.nanmean(aucs)), float(np.nanstd(aucs))
    mean_bacc, std_bacc = float(np.mean(baccs)), float(np.std(baccs))
    cm_sum = np.sum(cms, axis=0)

    print("\nCV summary")
    print(f"AUC mean plus or minus std: {mean_auc:.3f} plus or minus {std_auc:.3f}")
    print(f"BalAcc mean±std: {mean_bacc:.3f} plus or minus {std_bacc:.3f}")
    print("Confusion matrix (rows=true [0,1], cols=pred [0,1]):")
    print(cm_sum)

    out_fi_all_folds.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fi_rows).to_csv(out_fi_all_folds, index=False)
    print(f"Saved (per-fold selected feature importances): {out_fi_all_folds}")

    return oof_proba, oof_pred


def fit_final_and_export_features(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    k_best: int,
    model_c: float,
    var_threshold: float,
    pos_label: str,
    neg_label: str,
) -> pd.DataFrame:
    vt = VarianceThreshold(threshold=float(var_threshold))
    X1 = vt.fit_transform(X.to_numpy(dtype=float))

    X1 = np.where(np.isfinite(X1), X1, np.nan)
    imp = SimpleImputer(strategy="median")
    X2 = imp.fit_transform(X1)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X3 = scaler.fit_transform(X2)

    k = int(min(int(k_best), X3.shape[1]))
    skb = SelectKBest(score_func=f_classif, k=k)
    X4 = skb.fit_transform(X3, y.to_numpy(dtype=int))

    clf = LogisticRegression(
        solver="liblinear",
        C=float(model_c),
        class_weight="balanced",
        max_iter=5000,
    )
    clf.fit(X4, y.to_numpy(dtype=int))

    feats_vt = X.columns[vt.get_support()]
    feats = feats_vt[skb.get_support()]
    coefs = clf.coef_.ravel()

    feat_df = pd.DataFrame(
        {
            "feature": feats,
            "coef": coefs,
            "abs_coef": np.abs(coefs),
            "direction": np.where(coefs > 0, pos_label, neg_label),
        }
    ).sort_values("abs_coef", ascending=False)

    return feat_df

def main():
    ap = argparse.ArgumentParser(
        description="Train transcriptomics classifier with OOF CV; export feature importance + stability table."
    )

    ap.add_argument("--matrix", required=True, help="Path to T_harmonized.parquet (features x samples)")
    ap.add_argument("--out_dir", required=True, help="Output directory to write OOF + feature files")
    ap.add_argument("--labels", required=True, help="Labels CSV with sample_id,label OR sample_id,y")
    ap.add_argument(
        "--sample_map",
        required=False,
        default=None,
        help="Optional mapping CSV (token->GSM). If omitted, uses matrix columns as sample IDs.",
    )
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k_best", type=int, default=DEFAULT_K_BEST)
    ap.add_argument("--model_c", type=float, default=DEFAULT_MODEL_C)
    ap.add_argument("--var_threshold", type=float, default=DEFAULT_VAR_THRESHOLD)
    ap.add_argument("--pos_label", type=str, default="EC")
    ap.add_argument("--neg_label", type=str, default="ART")
    args = ap.parse_args()

    T_PATH = Path(args.matrix)
    OUT_DIR = Path(args.out_dir)
    Y_PATH = Path(args.labels)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_OOF = OUT_DIR / "oof_transcriptomics.csv"
    OUT_FEATS = OUT_DIR / "transcriptomics_feature_importance.csv"
    OUT_FI_ALL_FOLDS = OUT_DIR / "fi_T_all_folds.csv"

    if not T_PATH.exists():
        raise FileNotFoundError(f"Transcriptomics matrix not found: {T_PATH}")
    if not Y_PATH.exists():
        raise FileNotFoundError(f"Labels not found: {Y_PATH}")

    rename_map = None
    if args.sample_map:
        map_path = Path(args.sample_map)
        if not map_path.exists():
            raise FileNotFoundError(f"Sample map not found: {map_path}")
        rename_map = load_sample_map(map_path)
        print(f"[T] Using sample_map: {map_path}")
    else:
        print("[T] No sample_map provided (using matrix columns as sample IDs).")

    print(f"[T] matrix: {T_PATH}")
    print(f"[T] labels: {Y_PATH}")
    print(f"[T] out_dir: {OUT_DIR}")
    print(f"[T] pos_label={args.pos_label}, neg_label={args.neg_label}")

    X = load_X(T_PATH, rename_map=rename_map)
    y = load_y(Y_PATH, pos_label=args.pos_label, neg_label=args.neg_label)

    X, y = align_Xy(X, y)
    print(f" Aligned: X={X.shape} (samples x features) | y={y.value_counts().to_dict()}")

    oof_proba, oof_pred = logistic_cv_oof(
        X,
        y,
        n_splits=args.n_splits,
        seed=args.seed,
        k_best=args.k_best,
        model_c=args.model_c,
        var_threshold=args.var_threshold,
        out_fi_all_folds=OUT_FI_ALL_FOLDS,
    )

    out_df = pd.DataFrame(
        {
            "sample_id": oof_proba.index.astype(str),
            "proba": oof_proba.values,
            "pred": oof_pred.values,
            "y": y.loc[oof_proba.index].values,
        }
    )
    out_df.to_csv(OUT_OOF, index=False)
    print(f"Saved OOF predictions: {OUT_OOF}")

    feat_df = fit_final_and_export_features(
        X,
        y,
        k_best=args.k_best,
        model_c=args.model_c,
        var_threshold=args.var_threshold,
        pos_label=args.pos_label,
        neg_label=args.neg_label,
    )
    feat_df.to_csv(OUT_FEATS, index=False)
    print(f"Saved feature importance: {OUT_FEATS}")


if __name__ == "__main__":
    main()
