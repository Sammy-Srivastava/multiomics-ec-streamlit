#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import ast
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TRANSCRIPTOMIC_ENRICH = "multiomics/results/transcriptomic_enrichment_results.csv"
PROTEOMIC_ENRICH     = "multiomics/results/proteomic_enrichment_results.csv"

OUT_DIR = "multiomics/results/board_figures/enrichment_viz"
TOPK_PER_MODALITY = 12


# ---visual settings---
CMAP = "Blues"
FIG_DPI = 300

# ---helpers---
def safe_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def wrap_label(s: str, width: int = 44) -> str:
    return "\n".join(textwrap.wrap(str(s), width=width))


def pick_first_existing(df: pd.DataFrame, cols: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in cols:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def parse_intersection(x) -> set[str]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return set()
    if isinstance(x, (list, tuple, set)):
        return set(map(str, x))
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            return set(ast.literal_eval(s))
        except Exception:
            pass
    return set(p.strip() for p in s.replace(";", ",").split(",") if p.strip())


def apply_clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    ax.title.set_fontsize(15)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)

# ---Enrichment load---
def load_enrichment(csv_path: str, modality: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    name_col = pick_first_existing(df, ["name", "term_name", "description_name"])
    p_col    = pick_first_existing(df, ["p_value", "pvalue", "pval"])
    sig_col  = pick_first_existing(df, ["significant", "is_significant"])
    inter_col = pick_first_existing(df, ["intersection", "genes", "overlap"])

    if name_col is None or p_col is None:
        raise ValueError(f"{csv_path} must contain term names and p-values.")

    out = df.copy()
    out["modality"] = modality
    out["term"] = out[name_col].astype(str)
    out["p_value"] = pd.to_numeric(out[p_col], errors="coerce")

    if sig_col is not None:
        out = out[safe_bool_series(out[sig_col])]

    out = out.dropna(subset=["term", "p_value"])
    out = out[out["p_value"] > 0]

    out["neglog10_p"] = -np.log10(out["p_value"])

    if inter_col is not None:
        out["intersection_size"] = out[inter_col].apply(lambda x: len(parse_intersection(x)))
    else:
        out["intersection_size"] = 1.0

    return out[["modality", "term", "neglog10_p", "intersection_size"]]


def select_top_terms(df: pd.DataFrame, topk: int) -> pd.DataFrame:
    return (
        df.sort_values(["modality", "neglog10_p"], ascending=[True, False])
          .groupby("modality", as_index=False)
          .head(topk)
          .drop_duplicates(subset=["modality", "term"])
          .copy()
    )

# ---dot plot---
def plot_dot_matrix(df: pd.DataFrame, out_path: Path):
    term_order = (
        df.groupby("term")["neglog10_p"]
          .max()
          .sort_values(ascending=True)
          .index.tolist()
    )

    modalities = ["transcriptomics", "proteomics"]
    mods_present = [m for m in modalities if (df["modality"] == m).any()]

    term_to_y = {t: i for i, t in enumerate(term_order)}
    mod_to_x = {m: i for i, m in enumerate(mods_present)}

    xs, ys, cs, ss = [], [], [], []

    sizes = df["intersection_size"].astype(float)
    smin, smax = sizes.min(), sizes.max()

    def scale_size(v):
        if smax <= smin + 1e-9:
            return 140
        t = (v - smin) / (smax - smin)
        return 80 + (t ** 0.7) * 420

    for _, r in df.iterrows():
        xs.append(mod_to_x[r["modality"]])
        ys.append(term_to_y[r["term"]])
        cs.append(r["neglog10_p"])
        ss.append(scale_size(r["intersection_size"]))

    fig = plt.figure(figsize=(11.6, 7.2))
    ax = plt.gca()

    sc = ax.scatter(
        xs, ys, c=cs, s=ss,
        cmap=CMAP, edgecolor="black",
        linewidth=0.6, alpha=0.95
    )

    ax.set_xticks(range(len(mods_present)))
    ax.set_xticklabels([m.title() for m in mods_present])
    ax.set_yticks(range(len(term_order)))
    ax.set_yticklabels([wrap_label(t) for t in term_order])

    ax.set_xlabel("Modality")
    ax.set_ylabel("Enriched terms (top)")
    ax.set_title("Enrichment Dot Plot (size = overlap, color = significance)")

    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)
    apply_clean_axes(ax)

    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$-\log_{10}(p\mathrm{-value})$")

    plt.tight_layout()
    # centering
    plt.subplots_adjust(left=0.46)

    fig.savefig(out_path.with_suffix(".png"), dpi=FIG_DPI)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = load_enrichment(TRANSCRIPTOMIC_ENRICH, "transcriptomics")
    p = load_enrichment(PROTEOMIC_ENRICH, "proteomics")

    keep = select_top_terms(pd.concat([t, p]), TOPK_PER_MODALITY)

    plot_dot_matrix(keep, out_dir / "enrichment_dotplot")

    keep.to_csv(out_dir / "enrichment_terms_used.csv", index=False)

    print("Generated centered enrichment dot plot.")


if __name__ == "__main__":
    main()
