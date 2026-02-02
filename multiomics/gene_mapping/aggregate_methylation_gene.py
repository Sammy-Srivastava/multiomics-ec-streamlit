from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# CONFIG
PROJECT = Path("/Users/samyaksrivastava/Desktop/new science fair thing")

M_PATH = Path("/Users/samyaksrivastava/Desktop/new science fair thing/UI_stuff/artifacts/harmonized/20260124_122548/M_harmonized.parquet")

MAP_PATH = Path("/Users/samyaksrivastava/Desktop/new science fair thing/annotations/probe_to_gene.csv")

OUT_DIR = PROJECT / "UI_stuff" / "artifacts" / "methylation_gene_agg"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AGG = "median" 
MIN_PROBES_PER_GENE = 3

def agg_fn(name: str):
    if name == "mean":
        return np.nanmean
    if name == "median":
        return np.nanmedian
    raise ValueError("AGG must be 'mean' or 'median'")

def main():
    if not M_PATH.exists():
        raise FileNotFoundError(f"Methylation file not found: {M_PATH}")
    if not MAP_PATH.exists():
        raise FileNotFoundError(f"Mapping file not found: {MAP_PATH}")

    print("[LOAD] Methylation:", M_PATH)
    M_fxS = pd.read_parquet(M_PATH) 
    M_fxS.index = M_fxS.index.astype(str)
    M_fxS.columns = M_fxS.columns.astype(str)

    print("[LOAD] Mapping:", MAP_PATH)
    mp = pd.read_csv(MAP_PATH)
    if not {"probe_id", "gene_id"}.issubset(mp.columns):
        raise ValueError("Mapping file must contain columns: probe_id,gene_id")

    mp["probe_id"] = mp["probe_id"].astype(str)
    mp["gene_id"] = mp["gene_id"].astype(str)

    #keeping only probes present in methylation matrix
    probes_in_M = set(M_fxS.index)
    mp = mp[mp["probe_id"].isin(probes_in_M)].copy()

    if mp.empty:
        raise ValueError("No probe_id overlap between mapping file and M_harmonized index.")

    #counting probes per gene and filter
    gene_counts = mp.groupby("gene_id")["probe_id"].nunique().sort_values(ascending=False)
    keep_genes = gene_counts[gene_counts >= MIN_PROBES_PER_GENE].index
    mp = mp[mp["gene_id"].isin(keep_genes)].copy()

    if mp.empty:
        raise ValueError("After MIN_PROBES_PER_GENE filtering, no genes remain. Lower MIN_PROBES_PER_GENE (e.g., 2).")

    # gens to probe list build
    gene_to_probes = mp.groupby("gene_id")["probe_id"].apply(list)

    fn = agg_fn(AGG)
    out = []
    genes = []

    print(f"[AGG] Aggregating probes -> genes using {AGG}, min_probes={MIN_PROBES_PER_GENE}")
    for gene, probes in gene_to_probes.items():
        block = M_fxS.loc[probes].to_numpy(dtype=float)
        vec = fn(block, axis=0) 
        out.append(vec)
        genes.append(gene)

    G_gxS = pd.DataFrame(np.vstack(out), index=genes, columns=M_fxS.columns)

    out_path = OUT_DIR / f"M_gene_{AGG}_min{MIN_PROBES_PER_GENE}.parquet"
    G_gxS.to_parquet(out_path)

    print("\n[REPORT]")
    print("M probes x samples:", M_fxS.shape)
    print("Mapping rows after intersect:", mp.shape[0])
    print("Genes kept:", G_gxS.shape[0])
    print("Samples:", G_gxS.shape[1])
    print("Top genes by probe count:")
    print(gene_counts.head(10).to_string())
    print("\n[SAVED]", out_path)

if __name__ == "__main__":
    main()
