#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================
# EDIT THESE 3 PATHS
# ==========================
TRANSCRIPT_FI = "transcriptomics_feature_importance.csv"
PROTEOMICS_FI = "proteomics_subject_mean_feature_importance.csv"  # subject-mean run
METHYL_FI     = "methylation_feature_importance.csv"

OUT_DIR = "board_figures"
TOPK = 10

# ==========================
# LOOK & FEEL
# ==========================
MODALITY_COLORS = {
    "transcriptomics": "#1f77b4",
    "proteomics": "#2c8ea0",
    "methylation": "#0effeb",
    "unknown": "#7f7f7f",
}

WRAP_WIDTH = 28          # wrap long feature names
MAX_LABEL_CHARS = 60     # hard cap for extreme names
USE_LOG_X_IF_SKEWED = True
SKEW_RATIO_THRESHOLD = 8.0   # if max/min > this, switch to log x-scale


# --------------------------
# Heuristics to auto-detect columns
# --------------------------
FEATURE_CANDIDATES = [
    "feature", "gene", "geneid", "symbol", "protein", "probe", "cpg", "id", "name"
]
SCORE_CANDIDATES = [
    "abs_coef", "absweight", "abs_weight", "importance", "gain",
    "coef", "weight", "score", "shap", "mean_abs_shap", "mean_shap"
]
FOLD_CANDIDATES = ["fold", "cv_fold", "outer_fold"]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower().strip())


def detect_column(df: pd.DataFrame, candidates) -> str | None:
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    return None


def choose_score_column(df: pd.DataFrame) -> str:
    for cand in SCORE_CANDIDATES:
        c = detect_column(df, [cand])
        if c is not None:
            return c

    # fallback: pick most numeric-looking column (excluding fold-like)
    numeric_scores = []
    fold_keys = {_norm(x) for x in FOLD_CANDIDATES}
    for c in df.columns:
        if _norm(c) in fold_keys:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        frac_numeric = s.notna().mean()
        if frac_numeric > 0.6:
            numeric_scores.append((frac_numeric, c))

    if not numeric_scores:
        raise ValueError("Could not find a numeric score/importance column.")

    numeric_scores.sort(reverse=True)
    return numeric_scores[0][1]


def choose_feature_column(df: pd.DataFrame) -> str:
    c = detect_column(df, FEATURE_CANDIDATES)
    if c is not None:
        return c

    # fallback: first non-numeric column with many unique values
    best = None
    for c in df.columns:
        frac_numeric = pd.to_numeric(df[c], errors="coerce").notna().mean()
        uniq = df[c].astype(str).nunique(dropna=True)
        if frac_numeric < 0.2 and uniq > 5:
            best = c
            break
    if best is None:
        raise ValueError("Could not find a feature/name column.")
    return best


def maybe_aggregate_across_folds(path: str) -> pd.DataFrame:
    """
    If file has a fold column, aggregate feature scores across folds using mean(abs(score)).
    Otherwise, return as-is.
    """
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path}: empty file")

    fold_col = detect_column(df, FOLD_CANDIDATES)
    feat_col = choose_feature_column(df)
    score_col = choose_score_column(df)

    keep_cols = [feat_col, score_col] + ([fold_col] if fold_col else [])
    df = df[keep_cols].copy()
    df = df.rename(columns={feat_col: "feature", score_col: "score"})
    df["feature"] = df["feature"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["feature", "score"])

    if fold_col:
        df["abs_score"] = df["score"].abs()
        agg = (
            df.groupby("feature", as_index=False)["abs_score"]
              .mean()
              .rename(columns={"abs_score": "score"})
        )
        return agg
    else:
        return df


def _truncate_and_wrap(s: str, wrap_width: int = WRAP_WIDTH) -> str:
    s = str(s)
    if len(s) > MAX_LABEL_CHARS:
        s = s[: MAX_LABEL_CHARS - 1] + "…"
    # wrap for nicer y-axis labels
    return "\n".join(textwrap.wrap(s, width=wrap_width)) if len(s) > wrap_width else s


def apply_clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    ax.title.set_fontsize(15)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)


def plot_top_features_lollipop(
    df: pd.DataFrame,
    title: str,
    out_prefix: Path,
    modality: str,
    topk: int = 10,
):
    """
    Lollipop plot:
      - thin line from 0 to score
      - dot at score
      - sorted by abs(score)
    """
    d = df.copy()
    d["rank_score"] = d["score"].abs()
    d = d.sort_values("rank_score", ascending=False).head(topk).copy()

    # Reverse for horizontal plot (largest at top)
    d = d.sort_values("rank_score", ascending=True).reset_index(drop=True)

    # Labels
    d["feature_label"] = d["feature"].apply(_truncate_and_wrap)

    xvals = d["rank_score"].to_numpy()
    y = np.arange(len(d))

    # Decide log-scale if extremely skewed
    use_log = False
    finite = xvals[np.isfinite(xvals)]
    if USE_LOG_X_IF_SKEWED and len(finite) >= 2:
        mn = np.min(finite[finite > 0]) if np.any(finite > 0) else np.nan
        mx = np.max(finite)
        if mn == mn and mx == mx and mn > 0 and (mx / mn) > SKEW_RATIO_THRESHOLD:
            use_log = True

    fig = plt.figure(figsize=(8.8, 5.6))
    ax = plt.gca()

    color = MODALITY_COLORS.get(modality.lower(), MODALITY_COLORS["unknown"])

    # Lollipop stems
    for yi, xv in zip(y, xvals):
        ax.hlines(yi, 0, xv, linewidth=2.0, alpha=0.85)

    # Lollipop heads
    ax.scatter(xvals, y, s=70, edgecolor="black", linewidth=0.7)

    # Aesthetics
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature_label"].tolist())
    ax.set_xlabel("Absolute feature contribution (ranked)")
    ax.set_title(title)

    # Grid
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)

    # Apply color to all stems + points (without specifying global styles)
    # Matplotlib draws hlines as LineCollections; easiest is set after creation by recoloring:
    # We'll recolor all lines/collections explicitly.
    for coll in ax.collections:
        # the scatter collection is included here; set facecolor to modality color
        try:
            coll.set_facecolor(color)
        except Exception:
            pass
    for line in ax.lines:
        line.set_color(color)

    # Log scale if needed
    if use_log:
        ax.set_xscale("log")
        ax.set_xlabel("Absolute feature contribution (log scale)")

    apply_clean_axes(ax)
    plt.tight_layout()

    fig.savefig(out_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Transcriptomics
    t_df = maybe_aggregate_across_folds(TRANSCRIPT_FI)
    plot_top_features_lollipop(
        t_df,
        f"Top {TOPK} Transcriptomic Features",
        out_dir / "top10_transcriptomics_features",
        modality="transcriptomics",
        topk=TOPK,
    )

    # Proteomics
    p_df = maybe_aggregate_across_folds(PROTEOMICS_FI)
    plot_top_features_lollipop(
        p_df,
        f"Top {TOPK} Proteomic Features",
        out_dir / "top10_proteomics_features",
        modality="proteomics",
        topk=TOPK,
    )

    # Methylation
    m_df = maybe_aggregate_across_folds(METHYL_FI)
    plot_top_features_lollipop(
        m_df,
        f"Top {TOPK} Methylation Features",
        out_dir / "top10_methylation_features",
        modality="methylation",
        topk=TOPK,
    )

    print("Wrote:")
    print(" -", out_dir / "top10_transcriptomics_features.png")
    print(" -", out_dir / "top10_proteomics_features.png")
    print(" -", out_dir / "top10_methylation_features.png")
    print(" - (PDF versions too)")


if __name__ == "__main__":
    main()
