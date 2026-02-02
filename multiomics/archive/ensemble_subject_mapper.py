#!/usr/bin/env python3
from __future__ import annotations

"""
Option A: Manual cross-modality subject-ID mapping + subject-level ensemble.

Goal:
  If modalities do not share sample IDs, we can still do true "late integration"
  by mapping each modality-specific sample_id to a common subject_id.

You provide a mapping CSV:
  UI_stuff/artifacts/ensemble/id_map.csv

Format (one row per sample_id you want to map):
  subject_id,modality,sample_id

Examples:
  S001,transcriptomics,GSM2335560
  S001,methylation,GSM1301904
  S001,proteomics,EC01_plus

Notes:
  - If proteomics has plus/minus replicates, map BOTH to the same subject_id.
  - You can start by copying the template and filling it out.

Reads:
  - oof_methylation.csv
  - oof_transcriptomics.csv
  - proteomics_subject_mean_oof.csv   (or sample-level if you want; see config)
  - meta_optionA_weights.json
  - id_map.csv (if present)

Writes:
  - id_map_template.csv (always; safe overwrite)
  - ensemble_subject_oof.csv
  - ensemble_subject_metrics.json
  - ensemble_subject_overlap_report.json
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss


# =========================
# CONFIG (EDIT ONCE)
# =========================
PROJECT = Path("/Users/samyaksrivastava/Desktop/new science fair thing")

OOF_METHYL = PROJECT / "oof_methylation.csv"
OOF_TRANS  = PROJECT / "oof_transcriptomics.csv"

# Keep your current file (subject-mean) by default.
# If you ever want sample-level proteomics OOF, point this at that file instead.
OOF_PROT   = PROJECT / "proteomics_subject_mean_oof.csv"

WEIGHTS_JSON = PROJECT / "meta_optionA_weights.json"

OUT_DIR = PROJECT / "UI_stuff" / "artifacts" / "ensemble"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAP_PATH = OUT_DIR / "id_map.csv"             # you will create/fill this
MAP_TEMPLATE = OUT_DIR / "id_map_template.csv"

OUT_CSV  = OUT_DIR / "ensemble_subject_oof.csv"
OUT_JSON = OUT_DIR / "ensemble_subject_metrics.json"
OUT_OVERLAP = OUT_DIR / "ensemble_subject_overlap_report.json"

THRESHOLD = 0.5


# =========================
# Helpers
# =========================
def _read_oof_any(path: Path, tag: str) -> pd.DataFrame:
    """
    Returns columns:
      sample_id, y, p_<tag>
    """
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
        "sample_id": df[id_col].astype(str).str.strip(),
        "y": pd.to_numeric(df["y"], errors="coerce"),
        f"p_{tag}": pd.to_numeric(df["proba"], errors="coerce"),
    }).dropna(subset=["sample_id"])

    # If duplicates exist, average proba; y = mode (or first non-null)
    def _mode_or_nan(s):
        s2 = pd.Series(s).dropna()
        if s2.empty:
            return np.nan
        return s2.mode().iloc[0]

    out = out.groupby("sample_id", as_index=False).agg(
        {"y": _mode_or_nan, f"p_{tag}": "mean"}
    )

    out["sample_id"] = out["sample_id"].astype(str).str.strip()
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    return out


def _load_weights(path: Path) -> dict[str, float]:
    obj = json.loads(path.read_text())
    w = obj.get("weights", obj)
    s = float(sum(w.values()))
    if s <= 0:
        raise ValueError("Weights sum to <= 0.")
    return {k: float(v) / s for k, v in w.items()}


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


def _write_template_map(m_df: pd.DataFrame, t_df: pd.DataFrame, p_df: pd.DataFrame) -> None:
    """
    Create a starter template with sample IDs so you can quickly fill subject_id.
    We will write 3 blocks (one per modality).
    """
    rows = []
    for tag, df in [("methylation", m_df), ("transcriptomics", t_df), ("proteomics", p_df)]:
        if df is None or df.empty:
            continue
        head_ids = df["sample_id"].astype(str).head(30).tolist()
        for sid in head_ids:
            rows.append({"subject_id": "", "modality": tag, "sample_id": sid})

    tmp = pd.DataFrame(rows, columns=["subject_id", "modality", "sample_id"])
    tmp.to_csv(MAP_TEMPLATE, index=False)


def _load_mapping(path: Path) -> pd.DataFrame:
    """
    Expected columns: subject_id, modality, sample_id
    """
    if not path.exists():
        return pd.DataFrame(columns=["subject_id", "modality", "sample_id"])

    mp = pd.read_csv(path)
    need = {"subject_id", "modality", "sample_id"}
    if not need.issubset(set(mp.columns)):
        raise ValueError(
            f"id_map.csv must have columns {sorted(list(need))}. "
            f"Found: {mp.columns.tolist()}"
        )

    mp = mp.copy()
    mp["subject_id"] = mp["subject_id"].astype(str).str.strip()
    mp["modality"] = mp["modality"].astype(str).str.strip().str.lower()
    mp["sample_id"] = mp["sample_id"].astype(str).str.strip()

    # Drop empty subject_id rows (placeholders)
    mp = mp[mp["subject_id"].notna() & (mp["subject_id"] != "")]
    mp = mp[mp["sample_id"].notna() & (mp["sample_id"] != "")]

    # sanity: allowed modalities
    allowed = {"methylation", "transcriptomics", "proteomics"}
    bad = sorted(set(mp["modality"]) - allowed)
    if bad:
        raise ValueError(f"Unknown modality values in id_map.csv: {bad}. Allowed={sorted(list(allowed))}")

    # If duplicates exist (same modality+sample_id mapped to multiple subjects), keep first and warn
    mp = mp.drop_duplicates(subset=["modality", "sample_id"], keep="first")
    return mp


def _apply_mapping(df: pd.DataFrame, mp: pd.DataFrame, tag: str) -> pd.DataFrame:
    """
    Adds subject_id via mapping; returns rows that got mapped.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["subject_id", "y", f"p_{tag}"])

    submap = mp[mp["modality"] == tag][["sample_id", "subject_id"]]
    out = df.merge(submap, on="sample_id", how="left")

    mapped = out.dropna(subset=["subject_id"]).copy()
    mapped["subject_id"] = mapped["subject_id"].astype(str).str.strip()

    # Keep only needed cols
    mapped = mapped[["subject_id", "y", f"p_{tag}"]]

    # If multiple sample_ids map to same subject_id within a modality:
    # average probabilities; y = mode
    def _mode_or_nan(s):
        s2 = pd.Series(s).dropna()
        if s2.empty:
            return np.nan
        return s2.mode().iloc[0]

    mapped = mapped.groupby("subject_id", as_index=False).agg(
        {"y": _mode_or_nan, f"p_{tag}": "mean"}
    )

    mapped["y"] = pd.to_numeric(mapped["y"], errors="coerce")
    return mapped


def _ensemble_weighted_mean(row: pd.Series, weights: dict[str, float], p_cols: list[str]) -> float:
    num = 0.0
    den = 0.0
    for col in p_cols:
        tag = col.replace("p_", "")
        p = row[col]
        if pd.notna(p):
            w = float(weights.get(tag, 0.0))
            num += w * float(p)
            den += w
    return (num / den) if den > 0 else np.nan


# =========================
# Main
# =========================
def main():
    # Load OOFs
    m = _read_oof_any(OOF_METHYL, "methylation") if OOF_METHYL.exists() else None
    t = _read_oof_any(OOF_TRANS,  "transcriptomics") if OOF_TRANS.exists() else None
    p = _read_oof_any(OOF_PROT,   "proteomics") if OOF_PROT.exists() else None

    if (m is None or m.empty) and (t is None or t.empty) and (p is None or p.empty):
        raise RuntimeError("No OOF files found/loaded.")

    # Template mapping to help you start
    _write_template_map(m, t, p)
    print(f"[OK] wrote mapping template: {MAP_TEMPLATE}")

    # Load weights
    if not WEIGHTS_JSON.exists():
        raise FileNotFoundError(f"Missing weights file: {WEIGHTS_JSON}. Create it with optionA_weight_report.py first.")
    weights = _load_weights(WEIGHTS_JSON)

    # Load mapping
    mp = _load_mapping(MAP_PATH)
    if mp.empty:
        print(f"[WARN] No mapping file found (or it is empty): {MAP_PATH}")
        print("       Fill in id_map.csv using id_map_template.csv as a starting point, then re-run this script.")
        # Still write empty outputs so UI doesn’t break
        empty = pd.DataFrame(columns=[
            "subject_id", "p_methylation", "p_transcriptomics", "p_proteomics",
            "y", "n_modalities_present", "proba_ens", "pred_ens"
        ])
        empty.to_csv(OUT_CSV, index=False)
        OUT_JSON.write_text(json.dumps({"note": "No id_map.csv provided; cannot build subject-level ensemble."}, indent=2))
        OUT_OVERLAP.write_text(json.dumps({"note": "No id_map.csv provided; overlap undefined."}, indent=2))
        print(f"[OK] wrote empty: {OUT_CSV}")
        print(f"[OK] wrote: {OUT_JSON}")
        print(f"[OK] wrote: {OUT_OVERLAP}")
        return

    # Apply mapping per modality
    m2 = _apply_mapping(m, mp, "methylation") if m is not None else pd.DataFrame()
    t2 = _apply_mapping(t, mp, "transcriptomics") if t is not None else pd.DataFrame()
    p2 = _apply_mapping(p, mp, "proteomics") if p is not None else pd.DataFrame()

    # Merge on subject_id (outer) and reconcile y
    dfs = [df for df in [m2, t2, p2] if df is not None and not df.empty]
    if not dfs:
        raise RuntimeError("Mapping file exists but did not match any sample IDs in the OOF files.")

    # Start with first
    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on="subject_id", how="outer", suffixes=("", "_dup"))

    # We may have multiple y columns (y_x, y_y) from merges; handle explicitly:
    # Collect any column that equals 'y' or startswith 'y_'.
    y_cols = [c for c in df.columns if c == "y" or c.startswith("y_")]
    if not y_cols:
        raise RuntimeError("No label columns after merge; unexpected.")

    # Build unified y: if multiple non-null and disagree, mark as NaN (and later drop for metrics)
    def _unify_y(row):
        vals = []
        for c in y_cols:
            v = row.get(c, np.nan)
            if pd.notna(v):
                vals.append(int(v))
        if not vals:
            return np.nan
        if len(set(vals)) == 1:
            return vals[0]
        return np.nan  # conflict

    df["y"] = df.apply(_unify_y, axis=1)
    # Drop old y columns
    for c in y_cols:
        if c != "y":
            df.drop(columns=[c], inplace=True, errors="ignore")

    # Count modalities present
    p_cols = [c for c in df.columns if c.startswith("p_")]
    df["n_modalities_present"] = df[p_cols].notna().sum(axis=1).astype(int)

    # Weighted ensemble probability (weights only apply when >=2 modalities present)
    df["proba_ens"] = df.apply(lambda r: _ensemble_weighted_mean(r, weights, p_cols), axis=1)
    df["pred_ens"] = (df["proba_ens"] >= THRESHOLD).astype(int)

    # Sort for readability
    df = df.sort_values(["n_modalities_present", "subject_id"], ascending=[False, True]).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] wrote: {OUT_CSV}")

    # Overlap report
    overlap_counts = df["n_modalities_present"].value_counts().sort_index()
    report = {
        "n_subjects_total": int(len(df)),
        "counts_by_n_modalities_present": {str(int(k)): int(v) for k, v in overlap_counts.items()},
        "n_subjects_with_2plus_modalities": int((df["n_modalities_present"] >= 2).sum()),
        "n_subjects_with_3_modalities": int((df["n_modalities_present"] >= 3).sum()),
        "example_subjects_with_2plus": df.loc[df["n_modalities_present"] >= 2, "subject_id"].head(25).tolist(),
        "note": "If you map IDs across modalities correctly, you should see some subjects with >=2 modalities present."
    }
    OUT_OVERLAP.write_text(json.dumps(report, indent=2))
    print(f"[OK] wrote: {OUT_OVERLAP}")

    # Metrics (only where y and proba_ens are present and y is not conflicting)
    dfm = df.dropna(subset=["y", "proba_ens"]).copy()
    dfm["y"] = dfm["y"].astype(int)

    payload = {
        "overall": _metrics(dfm["y"].to_numpy(), dfm["proba_ens"].to_numpy(), THRESHOLD) if len(dfm) else {"note": "no rows with labels"},
        "metrics_for_subjects_with_2plus_modalities": None,
        "weights_used": weights,
        "threshold": THRESHOLD,
        "n_rows_used_for_overall_metrics": int(len(dfm)),
    }

    df2 = dfm[dfm["n_modalities_present"] >= 2]
    if len(df2) >= 2 and df2["y"].nunique() >= 2:
        payload["metrics_for_subjects_with_2plus_modalities"] = _metrics(df2["y"].to_numpy(), df2["proba_ens"].to_numpy(), THRESHOLD)
        payload["n_subjects_with_2plus_used_for_metrics"] = int(len(df2))
    else:
        payload["metrics_for_subjects_with_2plus_modalities"] = {
            "n": int(len(df2)),
            "note": "insufficient rows or class variation for AUC; add more mapped overlaps"
        }
        payload["n_subjects_with_2plus_used_for_metrics"] = int(len(df2))

    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[OK] wrote: {OUT_JSON}")

    print("\n=== SUBJECT OVERLAP SUMMARY ===")
    print(f"  subjects with 1 modality present: {(df['n_modalities_present'] == 1).sum()}")
    print(f"  subjects with >=2 modalities present: {(df['n_modalities_present'] >= 2).sum()}")
    print(f"  subjects with 3 modalities present: {(df['n_modalities_present'] >= 3).sum()}")

    if (df['n_modalities_present'] >= 2).sum() == 0:
        print("\n[IMPORTANT] You still have zero overlap after mapping.")
        print("This means id_map.csv either is empty, has typos, or the listed sample_ids do not exist in the OOF files.")
        print("Open id_map_template.csv, copy sample IDs exactly, assign shared subject_id values, then re-run.")


if __name__ == "__main__":
    main()
