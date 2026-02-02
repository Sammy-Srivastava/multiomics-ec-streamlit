from pathlib import Path
import json
import pandas as pd


def methylation_probe_to_gene(
    methyl_path: Path,
    probe_to_gene_path: Path,
    out_dir: Path,
    agg: str = "median",
    min_probes_per_gene: int = 2,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    X = pd.read_parquet(methyl_path)  # probes × samples
    map_df = pd.read_csv(probe_to_gene_path)

    required_cols = {"probe_id", "gene_id"}
    if not required_cols.issubset(map_df.columns):
        raise ValueError(f"probe_to_gene must contain {required_cols}")
    
    # Expect columns: probe_id, gene_id
    map_df["probe_id"] = map_df["probe_id"].astype(str).str.strip()
    map_df["gene_id"] = map_df["gene_id"].astype(str).str.strip()

    # Join probe → gene
    df = map_df.set_index("probe_id").join(X, how="inner")

    if df.empty:
        raise ValueError("No probes matched between methylation matrix and probe_to_gene map")

    sample_cols = X.columns.tolist()

    # Aggregate probes → gene
    if agg == "mean":
        Xg = df.groupby("gene_id")[sample_cols].mean()
    else:
        Xg = df.groupby("gene_id")[sample_cols].median()

    # Filter genes with too few probes
    counts = df.groupby("gene_id").size()
    keep = counts[counts >= min_probes_per_gene].index
    Xg = Xg.loc[keep]

    report = {
        "input_probes": int(X.shape[0]),
        "mapped_probes": int(df.index.nunique()),
        "input_genes_raw": int(counts.shape[0]),
        "output_genes": int(Xg.shape[0]),
        "aggregation": agg,
        "min_probes_per_gene": min_probes_per_gene,
        "probes_per_gene": {
            "min": int(counts.min()),
            "median": float(counts.median()),
            "max": int(counts.max()),
        },
    }

    Xg.to_parquet(out_dir / "M_gene.parquet")
    counts.to_frame("n_probes").to_parquet(out_dir / "M_gene_probe_counts.parquet")

    with open(out_dir / "M_gene_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report
