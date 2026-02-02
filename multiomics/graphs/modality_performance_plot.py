from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_curve
)

oof_files = ['oof_methylation.csv', 'oof_transcriptomics.csv', 'proteomics_subject_mean_oof.csv']
OUT_DIR = 'board_figures'
out_name = 'modality_performance_auc'

# =========================
# VISUAL STYLE
# =========================
MODALITY_ORDER = ["transcriptomics", "proteomics", "methylation"]
MODALITY_COLORS = {
    "transcriptomics": "#1E5AA8",  # deep blue
    "proteomics": "#2E7BB4",       # muted blue
    "methylation": "#6BAED6",      # light blue
    "unknown": "#7f7f7f"
}

def apply_clean_axes(ax):
    """Minimalist axes styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    ax.title.set_fontsize(15)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)

def modality_color_list(modalities):
    return [MODALITY_COLORS.get(str(m).lower(), MODALITY_COLORS["unknown"]) for m in modalities]

# =========================
# OOF PARSER (UNCHANGED)
# =========================
def read_oof_loose_csv(path: Path) -> pd.DataFrame:
    allowed = {"transcriptomics", "proteomics", "methylation"}

    name = path.name.lower()
    inferred = None
    for m in allowed:
        if m in name:
            inferred = m
            break

    try:
        dfh = pd.read_csv(path, engine="python")
        cols = {c.lower(): c for c in dfh.columns}
        if "proba" in cols and "y" in cols:
            out = dfh[[cols["proba"], cols["y"]]].copy()
            out = out.rename(columns={cols["proba"]: "proba", cols["y"]: "y"})
            out["proba"] = pd.to_numeric(out["proba"], errors="coerce")
            out["y"] = pd.to_numeric(out["y"], errors="coerce")
            out = out.dropna(subset=["proba", "y"])
            out = out[out["y"].isin([0, 1])].copy()
            out["y"] = out["y"].astype(int)
            out["modality"] = inferred if inferred is not None else "unknown"
            return out[["modality", "proba", "y"]]
    except Exception:
        pass

    raw = pd.read_csv(path, header=None, engine="python", dtype=str, keep_default_na=False)
    raw = raw.replace("", np.nan).dropna(how="all").reset_index(drop=True)

    has_tokens = False
    sample_check = raw.head(200)
    for _, row in sample_check.iterrows():
        vals = [str(v).strip().lower() for v in row.tolist() if pd.notna(v) and str(v).strip() != ""]
        if any(v in allowed for v in vals):
            has_tokens = True
            break

    records = []

    if has_tokens:
        for _, row in raw.iterrows():
            vals = [v for v in row.tolist() if pd.notna(v)]
            vals_str = [str(v).strip() for v in vals if str(v).strip() != ""]

            modality = None
            for v in vals_str:
                vv = v.lower()
                if vv in allowed:
                    modality = vv
                    break
            if modality is None:
                continue

            y = None
            y_pos = None
            for i in range(len(vals_str) - 1, -1, -1):
                if vals_str[i] in ("0", "1"):
                    y = int(vals_str[i])
                    y_pos = i
                    break
            if y is None:
                continue

            proba = None
            for v in reversed(vals_str[:y_pos]):
                try:
                    f = float(v)
                except Exception:
                    continue
                if 0.0 <= f <= 1.0:
                    proba = f
                    break
            if proba is None:
                continue

            records.append({"modality": modality, "proba": proba, "y": y})
    else:
        if inferred is None:
            inferred = "unknown"

        for _, row in raw.iterrows():
            vals = [v for v in row.tolist() if pd.notna(v)]
            vals_str = [str(v).strip() for v in vals if str(v).strip() != ""]

            y = None
            y_pos = None
            for i in range(len(vals_str) - 1, -1, -1):
                if vals_str[i] in ("0", "1"):
                    y = int(vals_str[i])
                    y_pos = i
                    break
            if y is None:
                continue

            proba = None
            for v in reversed(vals_str[:y_pos]):
                try:
                    f = float(v)
                except Exception:
                    continue
                if 0.0 <= f <= 1.0:
                    proba = f
                    break
            if proba is None:
                continue

            records.append({"modality": inferred, "proba": proba, "y": y})

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(
            f"{path}: Could not parse any rows. "
            "If this persists, your file may not include both probability and y labels."
        )
    return df

# =========================
# METRICS
# =========================
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for modality, g in df.groupby('modality'):
        y = g['y'].to_numpy()
        p = g['proba'].to_numpy()

        auc = np.nan if len(np.unique(y)) < 2 else roc_auc_score(y, p)

        pred = (p >= 0.5).astype(int)
        bacc = balanced_accuracy_score(y, pred)
        brier = brier_score_loss(y, p)

        rows.append({
            "modality": modality,
            "n": int(len(g)),
            "pos": int((y == 1).sum()),
            "neg": int((y == 0).sum()),
            "auc": float(auc) if auc == auc else np.nan,
            "balanced_accuracy@0.5": float(bacc),
            "brier": float(brier),
        })

    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(MODALITY_ORDER)}
    out["order"] = out["modality"].map(order).fillna(999).astype(int)
    out = out.sort_values("order").drop(columns=["order"]).reset_index(drop=True)
    return out

# =========================
# BAR PLOTS (FIXED LABEL PLACEMENT)
# =========================
def plot_auc_bar(metrics: pd.DataFrame, out_prefix: Path):
    fig = plt.figure(figsize=(7.8, 4.8))
    ax = plt.gca()

    modalities = metrics['modality'].astype(str).str.lower().tolist()
    colors = modality_color_list(modalities)

    x = np.arange(len(metrics))
    auc_vals = metrics['auc'].to_numpy()

    bars = ax.bar(x, auc_vals, color=colors, edgecolor="black", linewidth=0.8)

    best_i = int(np.nanargmax(auc_vals))
    bars[best_i].set_linewidth(2.2)

    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in modalities])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('AUC')
    ax.set_title('Modality Performance (Out-of-Fold AUC)', pad=14)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    # --- FIX: keep value labels from colliding with title/top ---
    y_min, y_max = ax.get_ylim()
    top = y_max

    label_pad = 0.02      # normal offset above bar
    min_headroom = 0.07   # keep at least this much space below the top
    clamp_y = top - min_headroom

    for i, v in enumerate(auc_vals):
        if np.isnan(v):
            continue

        y_text = v + label_pad
        va = "bottom"

        # If it would get too close to the top/title region, place it inside the bar instead
        if y_text >= clamp_y:
            y_text = max(v - 0.035, 0.02)  # inside bar, readable
            va = "top"

        ax.text(i, y_text, f"{v:.3f}", ha="center", va=va, fontsize=11)

    footer = "  |  ".join(f"{r.modality}: n={r.n}" for r in metrics.itertuples(index=False))
    ax.text(0.5, -0.20, footer, transform=ax.transAxes, ha="center", va="top", fontsize=10)

    apply_clean_axes(ax)
    plt.tight_layout()

    fig.savefig(out_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)

def plot_bacc_bar(metrics: pd.DataFrame, out_prefix: Path):
    fig = plt.figure(figsize=(7.8, 4.8))
    ax = plt.gca()

    modalities = metrics['modality'].astype(str).str.lower().tolist()
    colors = modality_color_list(modalities)

    x = np.arange(len(metrics))
    bacc_vals = metrics['balanced_accuracy@0.5'].to_numpy()

    bars = ax.bar(x, bacc_vals, color=colors, edgecolor="black", linewidth=0.8)

    best_i = int(np.nanargmax(bacc_vals))
    bars[best_i].set_linewidth(2.2)

    ax.set_xticks(x)
    ax.set_xticklabels([m.title() for m in modalities])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Balanced Accuracy (threshold = 0.5)')
    ax.set_title('Modality Performance (Balanced Accuracy)', pad=14)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)

    # same collision-proof label logic
    y_min, y_max = ax.get_ylim()
    top = y_max
    label_pad = 0.02
    min_headroom = 0.07
    clamp_y = top - min_headroom

    for i, v in enumerate(bacc_vals):
        if np.isnan(v):
            continue

        y_text = v + label_pad
        va = "bottom"
        if y_text >= clamp_y:
            y_text = max(v - 0.035, 0.02)
            va = "top"
        ax.text(i, y_text, f"{v:.3f}", ha="center", va=va, fontsize=11)

    footer = "  |  ".join(f"{r.modality}: n={r.n}" for r in metrics.itertuples(index=False))
    ax.text(0.5, -0.20, footer, transform=ax.transAxes, ha="center", va="top", fontsize=10)

    apply_clean_axes(ax)
    plt.tight_layout()

    fig.savefig(out_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)

# =========================
# ROC PLOT (UPGRADED)
# =========================
def plot_roc_by_modality(all_df: pd.DataFrame, out_prefix: Path):
    fig = plt.figure(figsize=(7.8, 5.8))
    ax = plt.gca()

    present = set(all_df["modality"].astype(str).str.lower().unique().tolist())
    ordered_modalities = [m for m in MODALITY_ORDER if m in present] + sorted(list(present - set(MODALITY_ORDER)))

    # Chance line (subtle)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="#333333", alpha=0.7)

    for modality in ordered_modalities:
        g = all_df[all_df["modality"].astype(str).str.lower() == modality]
        y = g["y"].to_numpy()
        p = g["proba"].to_numpy()
        if len(np.unique(y)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)

        color = MODALITY_COLORS.get(modality, MODALITY_COLORS["unknown"])
        is_primary = (modality == "transcriptomics")

        lw = 3.4 if is_primary else 2.4
        alpha_line = 0.98 if is_primary else 0.90
        alpha_fill = 0.10 if is_primary else 0.06

        # Line
        ax.plot(
            fpr, tpr,
            linewidth=lw,
            color=color,
            alpha=alpha_line,
            label=f"{modality.title()} (AUC = {auc:.3f})"
        )
        # Fill under curve (subtle)
        ax.fill_between(fpr, 0, tpr, color=color, alpha=alpha_fill)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Out-of-Fold ROC Curves by Omic Modality", pad=12)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.25)
    ax.set_axisbelow(True)

    leg = ax.legend(loc="lower right", frameon=True, fontsize=10)
    leg.get_frame().set_alpha(0.92)

    apply_clean_axes(ax)
    plt.tight_layout()

    fig.savefig(out_prefix.with_suffix(".png"), dpi=300)
    fig.savefig(out_prefix.with_suffix(".pdf"))
    plt.close(fig)

# =========================
# MAIN
# =========================
def main():
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for f in oof_files:
        df = read_oof_loose_csv(Path(f))
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df["modality"] = all_df["modality"].astype(str).str.lower()

    metrics = compute_metrics(all_df)

    metrics_path = out_dir / "modality_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    plot_auc_bar(metrics, out_dir / out_name)
    plot_bacc_bar(metrics, out_dir / "modality_performance_bacc")
    plot_roc_by_modality(all_df, out_dir / "roc_by_modality")

    print("[OK] Generated:")
    print(" -", out_dir / f"{out_name}.png")
    print(" -", out_dir / f"{out_name}.pdf")
    print(" -", out_dir / "modality_performance_bacc.png")
    print(" -", out_dir / "modality_performance_bacc.pdf")
    print(" -", out_dir / "roc_by_modality.png")
    print(" -", out_dir / "roc_by_modality.pdf")
    print(" -", metrics_path)
    print("\nMetrics:")
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()
