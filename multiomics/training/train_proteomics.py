#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, confusion_matrix


# ----------------------------
# DEFAULTS (match your current script)
# ----------------------------
SEED = 42
N_SPLITS_MAX = 5

VAR_THRESHOLD = 1e-8

K_GRID = [25, 50, 100, 200, 300]
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]

# representation control
# "mean"  : subject vector = mean(plus, minus) (recommended for tiny n)
# "blocks": subject vector = [PLUS__, MINUS__] concatenated (optional DELTA__)
REPRESENTATION = "mean"
INCLUDE_DELTA = False


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train proteomics unimodal model (subject-level nested CV, sample-level OOF for ensemble)."
    )
    ap.add_argument("--matrix", required=True, help="Path to harmonized proteomics matrix (features x samples).")
    ap.add_argument("--labels", required=True, help="CSV with columns sample_id,y (0/1). sample_id like ART01_plus.")
    ap.add_argument("--out_dir", required=True, help="Directory to write outputs.")

    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n_splits_max", type=int, default=N_SPLITS_MAX)

    ap.add_argument("--var_threshold", type=float, default=VAR_THRESHOLD)

    ap.add_argument("--k_grid", default="25,50,100,200,300", help="Comma-separated K grid for SelectKBest.")
    ap.add_argument("--c_grid", default="0.01,0.03,0.1,0.3,1.0", help="Comma-separated C grid for LogisticRegression.")

    ap.add_argument(
        "--representation",
        choices=["mean", "blocks"],
        default=REPRESENTATION,
        help="Subject-level representation. mean is recommended on tiny n.",
    )
    ap.add_argument(
        "--include_delta",
        action="store_true",
        help="If representation=blocks, optionally include DELTA__ = PLUS - MINUS (often hurts on tiny n).",
    )
    return ap.parse_args()


# ----------------------------
# ID parsing
# ----------------------------
def _clean_id(s: str) -> str:
    s = str(s).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    return s.strip()


def _tok(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", str(s)).upper()


def _parse_proteomics_sample_id(col: str) -> str | None:
    """
    Converts:
      Abundance:_F7:_126,_Sample,_ART01,_Pos,_B1  -> ART01_plus
      ...,_EC04,_Neg,_B2                         -> EC04_minus
    Drops:
      ...,_Norm,_Norm,_B1                        -> None
    """
    c = str(col).strip().strip('"').strip("'")
    parts = [p.strip() for p in c.split(",") if p.strip()]

    # already normalized?
    if re.fullmatch(r"(ART|EC|HC)\d{2}_(plus|minus)", c, flags=re.IGNORECASE):
        m = re.match(r"^((?:ART|EC|HC)\d{2})_(plus|minus)$", c, flags=re.IGNORECASE)
        if m:
            return f"{m.group(1).upper()}_{m.group(2).lower()}"
        return c

    subj = None
    sign = None

    for p in parts:
        pu = _tok(p)
        if pu == "NORM":
            return None

        if re.fullmatch(r"(ART|EC|HC)\d{2}", pu):
            subj = pu

        if pu in ("POS", "POSITIVE", "PLUS"):
            sign = "plus"
        elif pu in ("NEG", "NEGATIVE", "MINUS"):
            sign = "minus"

    if subj and sign:
        return f"{subj}_{sign}"

    # keep as-is (won’t align unless y has same ids)
    return c


def subject_from_sample_id(sample_id: str) -> str:
    return str(sample_id).split("_")[0]


# ----------------------------
# Load X/Y (sample-level)
# ----------------------------
def load_X_samples(p_path: Path) -> pd.DataFrame:
    if str(p_path).lower().endswith(".parquet"):
        X_fxS = pd.read_parquet(p_path)
    else:
        X_fxS = pd.read_csv(p_path, index_col=0)

    rename = {}
    drop_cols = []
    for c in X_fxS.columns.astype(str):
        new = _parse_proteomics_sample_id(c)
        if new is None:
            drop_cols.append(c)
        else:
            rename[c] = new

    if drop_cols:
        X_fxS = X_fxS.drop(columns=drop_cols)

    X_fxS = X_fxS.rename(columns=rename)
    X_fxS.columns = X_fxS.columns.astype(str).str.strip()

    # duplicates can happen after parsing (e.g., ART01 appears twice)
    if pd.Index(X_fxS.columns).duplicated().any():
        X_fxS = X_fxS.groupby(level=0, axis=1).mean()

    # samples x proteins
    X = X_fxS.T
    X.index = X.index.astype(str).str.strip()

    print("Sample-level X:", X.shape, "(samples x proteins)")
    print("Sample ids head:", X.index[:10].tolist())
    return X


def load_y_samples(y_path: Path) -> pd.Series:
    df = pd.read_csv(y_path)

    # Accept either `y` or `label`
    if "y" not in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "y"})

    if "sample_id" not in df.columns or "y" not in df.columns:
        raise ValueError(
            f"Expected columns sample_id and (y or label) in {y_path}. Found: {df.columns.tolist()}"
        )

    df["sample_id"] = df["sample_id"].astype(str).map(_clean_id)
    df["sample_id"] = df["sample_id"].str.replace(r"\s+", "", regex=True)

    y_raw = df.set_index("sample_id")["y"]
    y_raw.index = y_raw.index.astype(str).str.strip()

    # If numeric 0/1 already, keep it
    y_num = pd.to_numeric(y_raw, errors="coerce")
    if y_num.notna().all():
        y_out = y_num.astype(int)
        bad = sorted(set(y_out.unique()) - {0, 1})
        if bad:
            raise ValueError(f"Numeric y must be binary 0/1. Found extra values: {bad}")
        print("Sample-level y counts:", y_out.value_counts().to_dict())
        return y_out

    # Otherwise map strings like ART/EC/HC -> 0/1
    y_str = y_raw.astype(str).str.strip().str.upper()
    print("Raw label uniques:", sorted(y_str.unique().tolist()))

    # Positive class = EC (change if you want different binary task)
    mapping = {"EC": 1, "ART": 0, "HC": 0}

    unknown = sorted(set(y_str.unique()) - set(mapping.keys()))
    if unknown:
        raise ValueError(
            f"Unrecognized label values in labels file: {unknown}. "
            f"Expected subset of {sorted(mapping.keys())}"
        )

    y_out = y_str.map(mapping).astype(int)
    print("Sample-level y counts:", y_out.value_counts().to_dict())
    return y_out




def align_sample_level(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    common = X.index.intersection(y.index)
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs between X and y at sample-level.")
    return X.loc[common].copy(), y.loc[common].copy()


# ----------------------------
# Subject-level construction
# ----------------------------
def build_subject_mean(X_samp: pd.DataFrame, y_samp: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Subject vector = mean(plus, minus) over available rows.
    Usually most stable when n_subjects is small.
    """
    y_df = y_samp.to_frame("y")
    y_df["__subject__"] = y_df.index.map(subject_from_sample_id)

    inconsistent = y_df.groupby("__subject__")["y"].nunique()
    bad = inconsistent[inconsistent > 1]
    if len(bad) > 0:
        raise ValueError(f"Inconsistent y within subjects: {bad.index.tolist()}")

    y_subj = y_df.groupby("__subject__")["y"].agg(lambda v: int(v.mode().iloc[0]))

    plus_ids = [i for i in X_samp.index if str(i).endswith("_plus")]
    minus_ids = [i for i in X_samp.index if str(i).endswith("_minus")]
    feats = X_samp.columns.tolist()

    X_plus = X_samp.loc[plus_ids].copy()
    X_plus["__subject__"] = [subject_from_sample_id(i) for i in X_plus.index]
    X_plus = X_plus.groupby("__subject__")[feats].mean()

    X_minus = X_samp.loc[minus_ids].copy()
    X_minus["__subject__"] = [subject_from_sample_id(i) for i in X_minus.index]
    X_minus = X_minus.groupby("__subject__")[feats].mean()

    subj = y_subj.index.intersection(X_plus.index.union(X_minus.index))
    if len(subj) == 0:
        raise ValueError("No overlapping subjects between X and y at subject-level.")

    rows = []
    for s in subj:
        parts = []
        if s in X_plus.index:
            parts.append(X_plus.loc[s, feats].to_numpy(dtype=float))
        if s in X_minus.index:
            parts.append(X_minus.loc[s, feats].to_numpy(dtype=float))
        rows.append(np.mean(np.vstack(parts), axis=0))

    X_subj = pd.DataFrame(rows, index=subj, columns=feats)
    y_out = y_subj.loc[subj].astype(int)

    print(f"Subject-mean: X {X_subj.shape} | y {y_out.value_counts().to_dict()} | n_subjects={X_subj.shape[0]}")
    return X_subj, y_out


def build_subject_blocks(X_samp: pd.DataFrame, y_samp: pd.Series, include_delta: bool) -> tuple[pd.DataFrame, pd.Series]:
    """
    Subject vector = concatenation of PLUS__ and MINUS__ (and optionally DELTA__).
    """
    y_df = y_samp.to_frame("y")
    y_df["__subject__"] = y_df.index.map(subject_from_sample_id)

    inconsistent = y_df.groupby("__subject__")["y"].nunique()
    bad = inconsistent[inconsistent > 1]
    if len(bad) > 0:
        raise ValueError(f"Inconsistent y within subjects: {bad.index.tolist()}")

    y_subj = y_df.groupby("__subject__")["y"].agg(lambda v: int(v.mode().iloc[0]))

    plus_ids = [i for i in X_samp.index if str(i).endswith("_plus")]
    minus_ids = [i for i in X_samp.index if str(i).endswith("_minus")]
    feats = X_samp.columns.tolist()

    X_plus = X_samp.loc[plus_ids].copy()
    X_plus["__subject__"] = [subject_from_sample_id(i) for i in X_plus.index]
    X_plus = X_plus.groupby("__subject__")[feats].mean()

    X_minus = X_samp.loc[minus_ids].copy()
    X_minus["__subject__"] = [subject_from_sample_id(i) for i in X_minus.index]
    X_minus = X_minus.groupby("__subject__")[feats].mean()

    subj = y_subj.index.intersection(X_plus.index.union(X_minus.index))
    if len(subj) == 0:
        raise ValueError("No overlapping subjects between X and y at subject-level.")

    Xp = X_plus.reindex(subj)
    Xm = X_minus.reindex(subj)

    Xp.columns = [f"PLUS__{c}" for c in Xp.columns]
    Xm.columns = [f"MINUS__{c}" for c in Xm.columns]

    parts = [Xp, Xm]

    if include_delta:
        both = X_plus.index.intersection(X_minus.index).intersection(subj)
        if len(both) > 0:
            d = (X_plus.loc[both] - X_minus.loc[both])
            d.columns = [f"DELTA__{c}" for c in d.columns]
            parts.append(d.reindex(subj))

    X_subj = pd.concat(parts, axis=1)
    y_out = y_subj.loc[subj].astype(int)

    print(f"Subject-blocks: X {X_subj.shape} | y {y_out.value_counts().to_dict()} | n_subjects={X_subj.shape[0]}")
    return X_subj, y_out


# ----------------------------
# Pipeline pieces
# ----------------------------
def _fit_transform_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    k_best: int,
    var_threshold: float,
    min_present_frac: float = 0.7,
):
    """
    Fold-safe preprocessing:
      1) drop columns with too many NaNs in TRAIN
      2) variance filter
      3) impute
      4) scale
      5) SelectKBest
    """
    present_frac = np.mean(np.isfinite(X_tr), axis=0)
    keep = present_frac >= float(min_present_frac)

    if keep.sum() < 10:
        keep = present_frac >= 0.5
    if keep.sum() < 10:
        keep = present_frac > 0.0

    X_tr0 = X_tr[:, keep]
    X_te0 = X_te[:, keep]

    vt = VarianceThreshold(threshold=float(var_threshold))
    X_tr1 = vt.fit_transform(X_tr0)
    X_te1 = vt.transform(X_te0)

    X_tr1 = np.where(np.isfinite(X_tr1), X_tr1, np.nan)
    X_te1 = np.where(np.isfinite(X_te1), X_te1, np.nan)

    imp = SimpleImputer(strategy="median")
    X_tr2 = imp.fit_transform(X_tr1)
    X_te2 = imp.transform(X_te1)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr3 = scaler.fit_transform(X_tr2)
    X_te3 = scaler.transform(X_te2)

    k = int(min(int(k_best), X_tr3.shape[1]))
    if k < 1:
        raise ValueError("No features survived filtering in this fold.")
    skb = SelectKBest(score_func=f_classif, k=k)
    X_tr4 = skb.fit_transform(X_tr3, y_tr)
    X_te4 = skb.transform(X_te3)

    return X_tr4, X_te4


def _fit_transform_fold_with_meta(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    feat_names_full: np.ndarray,
    k_best: int,
    var_threshold: float,
    min_present_frac: float = 0.7,
):
    present_frac = np.mean(np.isfinite(X_tr), axis=0)
    keep = present_frac >= float(min_present_frac)

    if keep.sum() < 10:
        keep = present_frac >= 0.5
    if keep.sum() < 10:
        keep = present_frac > 0.0

    X_tr0 = X_tr[:, keep]
    X_te0 = X_te[:, keep]
    feats0 = feat_names_full[keep]

    vt = VarianceThreshold(threshold=float(var_threshold))
    X_tr1 = vt.fit_transform(X_tr0)
    X_te1 = vt.transform(X_te0)
    feats1 = feats0[vt.get_support()]

    X_tr1 = np.where(np.isfinite(X_tr1), X_tr1, np.nan)
    X_te1 = np.where(np.isfinite(X_te1), X_te1, np.nan)

    imp = SimpleImputer(strategy="median")
    X_tr2 = imp.fit_transform(X_tr1)
    X_te2 = imp.transform(X_te1)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr3 = scaler.fit_transform(X_tr2)
    X_te3 = scaler.transform(X_te2)

    k = int(min(int(k_best), X_tr3.shape[1]))
    if k < 1:
        raise ValueError("No features survived filtering in this fold.")
    skb = SelectKBest(score_func=f_classif, k=k)
    X_tr4 = skb.fit_transform(X_tr3, y_tr)
    X_te4 = skb.transform(X_te3)
    feats2 = feats1[skb.get_support()]

    return X_tr4, X_te4, feats2


def _safe_n_splits(y: np.ndarray, n_splits_max: int) -> int:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    max_possible = max(2, min(n_pos, n_neg))
    return int(min(int(n_splits_max), max_possible))


def nested_cv_oof_tuned(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_splits_max: int,
    var_threshold: float,
    k_grid: list[int],
    c_grid: list[float],
    out_fi_all_folds: Path,
):
    """
    Outer CV: estimate performance, produce OOF probabilities at SUBJECT level.
    Inner CV: select (K, C) maximizing mean AUC on training folds.

    Exports per-fold selected-feature importances for stability:
      feature_id, fold, importance (abs LR coef)
    """
    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=int)
    feat_names_full = X.columns.to_numpy(dtype=str)

    outer_splits = _safe_n_splits(y_np, n_splits_max)
    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=int(seed))

    oof_proba = pd.Series(index=X.index, dtype=float)
    oof_pred = pd.Series(index=X.index, dtype=int)

    aucs, baccs, cms = [], [], []
    chosen = []
    fi_rows: list[dict] = []

    for fold, (tr, te) in enumerate(outer.split(X_np, y_np), start=1):
        X_tr, y_tr = X_np[tr], y_np[tr]
        X_te, y_te = X_np[te], y_np[te]

        inner_splits = _safe_n_splits(y_tr, max(2, outer_splits - 1))
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=int(seed) + fold)

        best_score = -1e9
        best_k = None
        best_c = None

        for k in k_grid:
            for c in c_grid:
                inner_aucs = []
                for itr, ite in inner.split(X_tr, y_tr):
                    Xi_tr, yi_tr = X_tr[itr], y_tr[itr]
                    Xi_te, yi_te = X_tr[ite], y_tr[ite]

                    Xi_tr2, Xi_te2 = _fit_transform_fold(
                        Xi_tr, yi_tr, Xi_te, k_best=int(k), var_threshold=var_threshold
                    )

                    clf = LogisticRegression(
                        solver="liblinear",
                        C=float(c),
                        class_weight="balanced",
                        max_iter=5000,
                    )
                    clf.fit(Xi_tr2, yi_tr)
                    pi = clf.predict_proba(Xi_te2)[:, 1]

                    if len(np.unique(yi_te)) < 2:
                        continue
                    inner_aucs.append(roc_auc_score(yi_te, pi))

                if len(inner_aucs) == 0:
                    continue

                score = float(np.mean(inner_aucs))
                if score > best_score:
                    best_score = score
                    best_k = int(min(int(k), X_tr.shape[1]))
                    best_c = float(c)

        if best_k is None:
            best_k = int(min(100, X_tr.shape[1]))
            best_c = 0.1

        chosen.append({"fold": fold, "k": best_k, "C": best_c, "inner_auc": best_score})

        X_tr2, X_te2, fold_feats = _fit_transform_fold_with_meta(
            X_tr,
            y_tr,
            X_te,
            feat_names_full=feat_names_full,
            k_best=int(best_k),
            var_threshold=var_threshold,
        )

        clf = LogisticRegression(
            solver="liblinear",
            C=float(best_c),
            class_weight="balanced",
            max_iter=5000,
        )
        clf.fit(X_tr2, y_tr)

        coefs = clf.coef_.ravel()
        importances = np.abs(coefs)
        for fid, imp in zip(fold_feats, importances):
            fi_rows.append({"feature_id": str(fid), "fold": int(fold), "importance": float(imp)})

        p_te = clf.predict_proba(X_te2)[:, 1]
        pred_te = (p_te >= 0.5).astype(int)

        oof_proba.iloc[te] = p_te
        oof_pred.iloc[te] = pred_te

        fold_auc = float("nan") if len(np.unique(y_te)) < 2 else float(roc_auc_score(y_te, p_te))
        fold_bacc = float(balanced_accuracy_score(y_te, pred_te))
        cm = confusion_matrix(y_te, pred_te, labels=[0, 1])

        aucs.append(fold_auc)
        baccs.append(fold_bacc)
        cms.append(cm)

        print(
            f"[outer] Fold {fold}/{outer_splits}: best(k={best_k}, C={best_c}) | "
            f"AUC={fold_auc:.3f} | BalAcc={fold_bacc:.3f}"
        )

    mean_auc = float(np.nanmean(aucs))
    std_auc = float(np.nanstd(aucs))
    mean_bacc = float(np.mean(baccs))
    std_bacc = float(np.std(baccs))
    cm_sum = np.sum(cms, axis=0)

    print("\n[proteomics] NESTED-CV summary")
    print(f"AUC mean±std: {mean_auc:.3f} ± {std_auc:.3f}")
    print(f"BalAcc mean±std: {mean_bacc:.3f} ± {std_bacc:.3f}")
    print("Confusion matrix summed:")
    print(cm_sum)

    print("\nChosen hyperparams per fold (inner CV):")
    for row in chosen:
        print(f"  fold {row['fold']}: k={row['k']}, C={row['C']}, inner_auc={row['inner_auc']:.3f}")

    fi_df = pd.DataFrame(fi_rows)
    fi_df.to_csv(out_fi_all_folds, index=False)
    print("Saved (per-fold selected feature importances):", out_fi_all_folds)

    return oof_proba, oof_pred


def fit_final_and_export_features(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    n_splits_max: int,
    var_threshold: float,
    k_grid: list[int],
    c_grid: list[float],
) -> pd.DataFrame:
    """
    Fit a final model on all subject rows for interpretability export.
    Picks best (k, C) via CV on full dataset, then fits on all data.
    """
    X_np = X.to_numpy(dtype=float)
    y_np = y.to_numpy(dtype=int)

    splits = _safe_n_splits(y_np, n_splits_max)
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=int(seed))

    best_score = -1e9
    best_k = None
    best_c = None

    for k in k_grid:
        for c in c_grid:
            aucs = []
            for tr, te in cv.split(X_np, y_np):
                X_tr, y_tr = X_np[tr], y_np[tr]
                X_te, y_te = X_np[te], y_np[te]
                X_tr2, X_te2 = _fit_transform_fold(
                    X_tr, y_tr, X_te, k_best=int(k), var_threshold=var_threshold
                )
                clf = LogisticRegression(
                    solver="liblinear",
                    C=float(c),
                    class_weight="balanced",
                    max_iter=5000,
                )
                clf.fit(X_tr2, y_tr)
                p = clf.predict_proba(X_te2)[:, 1]
                if len(np.unique(y_te)) < 2:
                    continue
                aucs.append(roc_auc_score(y_te, p))
            if len(aucs) == 0:
                continue
            score = float(np.mean(aucs))
            if score > best_score:
                best_score = score
                best_k = int(min(int(k), X_np.shape[1]))
                best_c = float(c)

    if best_k is None:
        best_k = int(min(100, X_np.shape[1]))
        best_c = 0.1

    print(f"\n[final-fit] Using best(k={best_k}, C={best_c}) from global CV (mean AUC={best_score:.3f})")

    vt = VarianceThreshold(threshold=float(var_threshold))
    X1 = vt.fit_transform(X_np)

    X1 = np.where(np.isfinite(X1), X1, np.nan)
    imp = SimpleImputer(strategy="median")
    X2 = imp.fit_transform(X1)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X3 = scaler.fit_transform(X2)

    k = int(min(int(best_k), X3.shape[1]))
    skb = SelectKBest(score_func=f_classif, k=k)
    X4 = skb.fit_transform(X3, y_np)

    clf = LogisticRegression(
        solver="liblinear",
        C=float(best_c),
        class_weight="balanced",
        max_iter=5000,
    )
    clf.fit(X4, y_np)

    feats_vt = X.columns[vt.get_support()]
    feats = feats_vt[skb.get_support()]
    coefs = clf.coef_.ravel()

    feat_df = pd.DataFrame({"feature": feats, "coef": coefs, "abs_coef": np.abs(coefs)}).sort_values(
        "abs_coef", ascending=False
    )
    return feat_df


# ----------------------------
# MAIN
# ----------------------------
def main():
    args = parse_args()

    P_PATH = Path(args.matrix)
    Y_PATH = Path(args.labels)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Output stays ensemble-compatible (sample_id-level OOF)
    OUT_OOF = OUT_DIR / "proteomics_subject_mean_oof.csv"
    OUT_FEATS = OUT_DIR / "proteomics_subject_mean_feature_importance.csv"
    OUT_FI_ALL_FOLDS = OUT_DIR / "fi_P_all_folds.csv"

    seed = int(args.seed)
    n_splits_max = int(args.n_splits_max)
    var_threshold = float(args.var_threshold)

    k_grid = [int(x) for x in str(args.k_grid).split(",") if str(x).strip() != ""]
    c_grid = [float(x) for x in str(args.c_grid).split(",") if str(x).strip() != ""]

    rep = str(args.representation).lower()
    include_delta = bool(args.include_delta)

    print("\n=== RUN: PROTEOMICS (tuned, stable CV) ===")
    print("matrix:", P_PATH)
    print("labels:", Y_PATH)
    print("out_dir:", OUT_DIR)
    print(f"representation={rep}, include_delta={include_delta}")

    if not P_PATH.exists():
        raise FileNotFoundError(f"Proteomics matrix not found: {P_PATH}")
    if not Y_PATH.exists():
        raise FileNotFoundError(f"Labels not found: {Y_PATH}")

    X_samp = load_X_samples(P_PATH)
    y_samp = load_y_samples(Y_PATH)
    X_samp, y_samp = align_sample_level(X_samp, y_samp)

    # subject-level design matrix
    if rep == "mean":
        X_subj, y_subj = build_subject_mean(X_samp, y_samp)
    elif rep == "blocks":
        X_subj, y_subj = build_subject_blocks(X_samp, y_samp, include_delta=include_delta)
    else:
        raise ValueError("representation must be 'mean' or 'blocks'")

    # tuned subject-level OOF predictions (nested CV) + fi_all_folds export
    oof_proba_subj, _oof_pred_subj = nested_cv_oof_tuned(
        X_subj,
        y_subj,
        seed=seed,
        n_splits_max=n_splits_max,
        var_threshold=var_threshold,
        k_grid=k_grid,
        c_grid=c_grid,
        out_fi_all_folds=OUT_FI_ALL_FOLDS,
    )

    # Export OOF at sample_id-level for ensemble (each sample gets its subject's OOF probability)
    rows = []
    for sample_id, yval in y_samp.items():
        subj = subject_from_sample_id(sample_id)
        if subj in oof_proba_subj.index:
            rows.append((sample_id, float(oof_proba_subj.loc[subj]), int(yval)))

    out_df = pd.DataFrame(rows, columns=["sample_id", "proba", "y"])
    out_df["pred"] = (out_df["proba"] >= 0.5).astype(int)
    out_df.to_csv(OUT_OOF, index=False)
    print("Saved (sample_id-level OOF for ensemble):", OUT_OOF)

    feat_df = fit_final_and_export_features(
        X_subj,
        y_subj,
        seed=seed,
        n_splits_max=n_splits_max,
        var_threshold=var_threshold,
        k_grid=k_grid,
        c_grid=c_grid,
    )
    feat_df.to_csv(OUT_FEATS, index=False)
    print("Saved:", OUT_FEATS)


if __name__ == "__main__":
    main()
