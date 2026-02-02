#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss

# config
PROJECT = Path("/Users/samyaksrivastava/Desktop/new science fair thing")

OOF_METHYL = PROJECT / "oof_methylation.csv"
OOF_TRANS  = PROJECT / "oof_transcriptomics.csv"
OOF_PROT   = PROJECT / "proteomics_subject_mean_oof.csv"

WEIGHTS_JSON = PROJECT / "meta_optionA_weights.json"

OUT_DIR = PROJECT / "UI_stuff" / "artifacts" / "ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CSV        = OUT_DIR / "ensemble_oof.csv"
OUT_JSON       = OUT_DIR / "ensemble_metrics.json"
OUT_OVERLAP    = OUT_DIR / "ensemble_overlap_report.json"
OUT_INCONS_CSV = OUT_DIR / "ensemble_inconsistencies.csv"

THRESHOLD = 0.5


# Helpers
def _clean_id(s: str) -> str:
    s = str(s).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    return s.strip()

def _read_oof_any(path: Path, tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # id col
    if "sample_id" in df.columns:
        id_col = "sample_id"
    elif "subject" in df.columns:
        id_col = "subject"
    elif "id" in df.columns:
        id_col = "id"
    else:
        raise ValueError(f"[{tag}] Could not find id column in {path}. Columns={df.columns.tolist()}")

    if "proba" not in df.columns:
        raise ValueError(f"[{tag}] Missing 'proba' column in {path}. Columns={df.columns.tolist()}")

    if "y" not in df.columns:
        raise ValueError(f"[{tag}] Missing 'y' column in {path}. Columns={df.columns.tolist()}")

    out = pd.DataFrame({
        "sample_id": df[id_col].astype(str).map(_clean_id),
        f"p_{tag}": pd.to_numeric(df["proba"], errors="coerce"),
        f"y_{tag}": pd.to_numeric(df["y"], errors="coerce"),
    })

    out = out.dropna(subset=["sample_id"])
    out["sample_id"] = out["sample_id"].astype(str).str.strip()

    def _mode_or_nan(s: pd.Series):
        s2 = pd.to_numeric(s, errors="coerce").dropna()
        if s2.empty:
            return np.nan
        return float(s2.mode().iloc[0])

    out = out.groupby("sample_id", as_index=False).agg(
        {f"p_{tag}": "mean", f"y_{tag}": _mode_or_nan}
    )

    return out


def _load_weights(path: Path) -> dict[str, float]:
    obj = json.loads(path.read_text())
    w = obj.get("weights", obj)
    s = float(sum(w.values()))
    if s <= 0:
        raise ValueError("Weights sum to <= 0. Bad weights file.")
    return {k: float(v) / s for k, v in w.items()}


def _ensemble_row(row: pd.Series, weights: dict[str, float], p_cols: list[str]) -> float:
    num = 0.0
    den = 0.0
    for col in p_cols:
        tag = col.replace("p_", "")
        p = row[col]
        if pd.notna(p):
            w = float(weights.get(tag, 0.0))
            if w > 0:
                num += w * float(p)
                den += w
    return (num / den) if den > 0 else np.nan


def _metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y = y.astype(int)
    p = p.astype(float)

    n = int(len(y))
    pos = int(y.sum())
    neg = int(n - pos)

    auc = float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))
    pred = (p >= thr).astype(int)
    bacc = float(balanced_accuracy_score(y, pred))
    acc = float((pred == y).mean())
    brier = float(brier_score_loss(y, p))

    return {
        "n": n, "pos": pos, "neg": neg,
        "auc": auc, "bacc": bacc, "acc": acc, "brier": brier,
        "threshold": float(thr),
        "proba_mean": float(np.mean(p)),
        "proba_min": float(np.min(p)),
        "proba_max": float(np.max(p)),
    }


def _first_nonnull_rowwise(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """row-wise coalesce and take first non-null value across cols."""
    if not cols:
        return pd.Series([np.nan] * len(df), index=df.index)
    tmp = df[cols].copy()
    return tmp.bfill(axis=1).iloc[:, 0]


def main():
    if not WEIGHTS_JSON.exists():
        raise FileNotFoundError(f"Missing weights file: {WEIGHTS_JSON}. Run optionA_weight_report.py first.")

    weights = _load_weights(WEIGHTS_JSON)

    dfs = []
    present = {}

    if OOF_METHYL.exists():
        present["methylation"] = str(OOF_METHYL)
        dfs.append(_read_oof_any(OOF_METHYL, "methylation"))
    else:
        print(f"[WARN] Missing: {OOF_METHYL}")

    if OOF_TRANS.exists():
        present["transcriptomics"] = str(OOF_TRANS)
        dfs.append(_read_oof_any(OOF_TRANS, "transcriptomics"))
    else:
        print(f"[WARN] Missing: {OOF_TRANS}")

    if OOF_PROT.exists():
        present["proteomics"] = str(OOF_PROT)
        dfs.append(_read_oof_any(OOF_PROT, "proteomics"))
    else:
        print(f"[WARN] Missing: {OOF_PROT}")

    if not dfs:
        raise RuntimeError("No OOF files found. Nothing to ensemble.")

    # SAFE MERGE:
    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on="sample_id", how="outer")

    # Collect p_* and y_* columns
    p_cols = [c for c in df.columns if c.startswith("p_")]
    y_cols = [c for c in df.columns if c.startswith("y_")]


    def _row_inconsistent(vals: pd.Series) -> bool:
        v = [int(x) for x in vals.values.tolist() if pd.notna(x)]
        return (len(v) >= 2) and (len(set(v)) > 1)

    inconsistent_mask = df[y_cols].apply(_row_inconsistent, axis=1) if y_cols else pd.Series(False, index=df.index)
    inconsist_df = df.loc[inconsistent_mask, ["sample_id"] + y_cols + p_cols].copy()

    if len(inconsist_df) > 0:
        inconsist_df.to_csv(OUT_INCONS_CSV, index=False)
        print(f"[WARN] Found {len(inconsist_df)} sample_id rows with conflicting y across modalities.")
        print(f"[WARN] Wrote conflicts for inspection: {OUT_INCONS_CSV}")
    else:
        #delete stale file if it exists
        if OUT_INCONS_CSV.exists():
            try:
                OUT_INCONS_CSV.unlink()
            except Exception:
                pass

    #take first non-null across modalities
    df["y"] = _first_nonnull_rowwise(df, y_cols)

    # Diagnostics: overlap counts
    df["n_modalities_present"] = df[p_cols].notna().sum(axis=1).astype(int)

    # Label source_modality for single-modality rows
    df["source_modality"] = np.select(
        [
            df["n_modalities_present"].eq(1) & df.get("p_methylation", pd.Series([np.nan]*len(df))).notna(),
            df["n_modalities_present"].eq(1) & df.get("p_transcriptomics", pd.Series([np.nan]*len(df))).notna(),
            df["n_modalities_present"].eq(1) & df.get("p_proteomics", pd.Series([np.nan]*len(df))).notna(),
        ],
        ["methylation", "transcriptomics", "proteomics"],
        default="multi_or_unknown",
    )

    # Ensemble probability
    df["proba_ens"] = df.apply(lambda r: _ensemble_row(r, weights, p_cols), axis=1)
    df["pred_ens"] = (df["proba_ens"] >= THRESHOLD).astype(int)

    # Save ensemble OOF
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote: {OUT_CSV}")

    # Metrics 
    dfm = df.dropna(subset=["y", "proba_ens"]).copy()
    dfm["y"] = pd.to_numeric(dfm["y"], errors="coerce")
    dfm = dfm.dropna(subset=["y"])
    dfm["y"] = dfm["y"].astype(int)

    overall = _metrics(dfm["y"].to_numpy(), dfm["proba_ens"].to_numpy(), THRESHOLD)

    by_mod = {}
    for mod in ["methylation", "transcriptomics", "proteomics"]:
        sub = dfm[dfm["source_modality"] == mod]
        if len(sub) >= 2 and sub["y"].nunique() >= 2:
            by_mod[mod] = _metrics(sub["y"].to_numpy(), sub["proba_ens"].to_numpy(), THRESHOLD)
        else:
            by_mod[mod] = {"n": int(len(sub)), "note": "insufficient class variation for AUC"}

    payload = {
        "overall": overall,
        "by_source_modality": by_mod,
        "weights_used": weights,
        "inputs_present": present,
        "threshold": THRESHOLD,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[OK] wrote: {OUT_JSON}")

    # Overlap report
    overlap_counts = df["n_modalities_present"].value_counts().sort_index()
    overlap_payload = {
        "n_rows_total": int(len(df)),
        "counts_by_n_modalities_present": {int(k): int(v) for k, v in overlap_counts.items()},
        "n_rows_with_2plus_modalities": int((df["n_modalities_present"] >= 2).sum()),
        "n_rows_with_3_modalities": int((df["n_modalities_present"] >= 3).sum()),
        "example_rows_with_2plus": [],
        "note": (
            "If Option A truly has no shared IDs across modalities, n_rows_with_2plus_modalities will be 0 "
            "and proba_ens will equal the single modality probability on nearly all rows. That is expected."
        ),
    }

    ex = df[df["n_modalities_present"] >= 2].head(15)
    if len(ex) > 0:
        cols = ["sample_id", "y"] + p_cols + ["proba_ens", "pred_ens", "n_modalities_present"]
        overlap_payload["example_rows_with_2plus"] = ex[cols].to_dict(orient="records")

    OUT_OVERLAP.write_text(json.dumps(overlap_payload, indent=2))
    print(f"[OK] wrote: {OUT_OVERLAP}")

    # Console summary
    print("\n=== OVERLAP SUMMARY ===")
    for k, v in overlap_payload["counts_by_n_modalities_present"].items():
        print(f"  rows with {k} modality probs present: {v}")
    print(f"  rows with >=2 modalities present: {overlap_payload['n_rows_with_2plus_modalities']}")
    if len(inconsist_df) > 0:
        print(f"\n You have label collisions across modalities for the same sample_id.")
        print("Fix by prefixing IDs per modality OR by using a true cross-omic subject identifier.")


if __name__ == "__main__":
    main()
