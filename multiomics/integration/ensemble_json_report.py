#!/usr/bin/env python3
from __future__ import annotations


from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss

PROJECT = Path("/Users/samyaksrivastava/Desktop/new science fair thing")

OOF_METHYL = PROJECT / "oof_methylation.csv"                 # sample_id,proba,(pred),y
OOF_TRANS  = PROJECT / "oof_transcriptomics.csv"             # sample_id,proba,(pred),y
OOF_PROT   = PROJECT / "proteomics_subject_mean_oof.csv"     # sample_id OR subject, proba, y

OUT_REPORT  = PROJECT / "meta_optionA_report.json"
OUT_WEIGHTS = PROJECT / "meta_optionA_weights.json"


WEIGHT_MODE = "auc"
EPS = 1e-6

CALIBRATION = "none" 


# Helpers
def _read_oof(path: Path, modality: str) -> pd.DataFrame:
    """
    Standardize OOF into columns: id, proba, y
    Accepts:
      - methyl/trans: sample_id, proba, y  (pred optional)
      - proteomics: subject OR sample_id, proba, y
    """
    if not path.exists():
        raise FileNotFoundError(f"[{modality}] Missing file: {path}")

    df = pd.read_csv(path)

    # normalize ID column to "id"
    if "id" not in df.columns:
        if modality == "proteomics" and "subject" in df.columns:
            df = df.rename(columns={"subject": "id"})
        elif "sample_id" in df.columns:
            df = df.rename(columns={"sample_id": "id"})
        else:
            raise ValueError(
                f"[{modality}] Could not find an ID column. "
                f"Expected one of: id, sample_id{', subject (proteomics)' if modality=='proteomics' else ''}. "
                f"Found columns: {df.columns.tolist()}"
            )

    # required columns
    required = {"id", "proba", "y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"[{modality}] Missing required columns: {sorted(missing)}. "
            f"Found columns: {df.columns.tolist()}"
        )

    out = df.loc[:, ["id", "proba", "y"]].copy()

    out["id"] = out["id"].astype(str).str.strip()
    out["proba"] = pd.to_numeric(out["proba"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")

    out = out.dropna(subset=["id", "proba", "y"])
    out["y"] = out["y"].astype(int)
    out["proba"] = out["proba"].astype(float).clip(0.0, 1.0)

    # drop duplicate ids
    out = out.drop_duplicates(subset=["id"], keep="first")

    # sanity check for y being binary
    bad = sorted(set(out["y"].unique()) - {0, 1})
    if bad:
        raise ValueError(f"[{modality}] y must be 0/1. Found extra values: {bad}")

    return out


def _metrics_from_oof(df: pd.DataFrame) -> dict:
    y = df["y"].to_numpy(dtype=int)
    p = df["proba"].to_numpy(dtype=float)

    n = int(df.shape[0])
    pos = int(y.sum())
    neg = int(n - pos)

    # both classes must be present
    auc = float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, p))

    pred05 = (p >= 0.5).astype(int)
    bacc05 = float(balanced_accuracy_score(y, pred05))
    brier = float(brier_score_loss(y, p))

    return {
        "n": n,
        "pos": pos,
        "neg": neg,
        "auc": auc,
        "bacc@0.5": bacc05,
        "brier": brier,
        "proba_min": float(np.min(p)) if n else float("nan"),
        "proba_mean": float(np.mean(p)) if n else float("nan"),
        "proba_max": float(np.max(p)) if n else float("nan"),
    }


def _derive_weights(metrics: dict, mode: str) -> dict[str, float]:
    mods = list(metrics.keys())
    if len(mods) == 0:
        raise ValueError("No modalities provided to weight derivation.")

    if mode == "equal":
        return {m: 1.0 / len(mods) for m in mods}

    if mode == "auc":
        raw = {}
        for m in mods:
            auc = metrics[m]["auc"]
            if np.isnan(auc):
                raw[m] = EPS
            else:
                raw[m] = max(float(auc) - 0.5, EPS)

        s = float(sum(raw.values()))
        return {m: float(raw[m] / s) for m in mods}

    raise ValueError(f"Unknown WEIGHT_MODE: {mode}")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    methyl = _read_oof(OOF_METHYL, "methylation")
    trans  = _read_oof(OOF_TRANS,  "transcriptomics")
    prot   = _read_oof(OOF_PROT,   "proteomics")

    ids_m = set(methyl["id"])
    ids_t = set(trans["id"])
    ids_p = set(prot["id"])

    overlap_all = ids_m & ids_t & ids_p

    print(f"[methyl] n={len(methyl)} | id head={methyl['id'].head().tolist()}")
    print(f"[trans ] n={len(trans)}  | id head={trans['id'].head().tolist()}")
    print(f"[prot  ] n={len(prot)}   | id head={prot['id'].head().tolist()}")

    print("\nOverlap diagnostics")
    print(f"  methyl: unique ids={len(ids_m)}")
    print(f"  trans : unique ids={len(ids_t)}")
    print(f"  prot  : unique ids={len(ids_p)}")
    print(f"  common ids across all provided modalities: {len(overlap_all)} (expected 0 for Option A)")

    metrics = {
        "methylation": _metrics_from_oof(methyl),
        "transcriptomics": _metrics_from_oof(trans),
        "proteomics": _metrics_from_oof(prot),
    }

    weights = _derive_weights(metrics, WEIGHT_MODE)

    report = {
        "option": "A",
        "reason": "No shared sample IDs across modalities; stacking/meta-model training is impossible.",
        "calibration": CALIBRATION,
        "weight_mode": WEIGHT_MODE,
        "weights": weights,
        "modality_metrics_from_oof": metrics,
        "inputs": {
            "methylation_oof": str(OOF_METHYL),
            "transcriptomics_oof": str(OOF_TRANS),
            "proteomics_oof": str(OOF_PROT),
        },
        "overlap": {
            "methylation_unique_ids": int(len(ids_m)),
            "transcriptomics_unique_ids": int(len(ids_t)),
            "proteomics_unique_ids": int(len(ids_p)),
            "common_ids_all_three": int(len(overlap_all)),
        },
        "notes": [
            "These weights are derived from unimodal OOF performance within each modality's cohort, not from cross-modal training.",
            "Use these weights only when a future *single subject* has multiple modalities available at inference time.",
            "If later you add probability calibration per modality, do calibration inside CV for each modality and then recompute OOF before re-weighting.",
        ],
    }

    weights_payload = {
        "weights": weights,
        "weight_mode": WEIGHT_MODE,
        "calibration": CALIBRATION,
    }

    _write_json(OUT_REPORT, report)
    _write_json(OUT_WEIGHTS, weights_payload)

    print(f"\nSaved report: {OUT_REPORT}")
    print(f"Saved weights: {OUT_WEIGHTS}")

    print("\nWeights to use at inference-time (Option A):")
    for k, v in weights.items():
        print(f"  {k}: {v:.3f}")

    print("\nInference rule (ONLY when multiple modalities exist for one subject):")
    print("  pred_ensemble = 1 if p_ensemble >= 0.5 else 0")


if __name__ == "__main__":
    main()
