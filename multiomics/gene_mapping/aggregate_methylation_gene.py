from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _agg_fn(name: str):
    name = name.lower().strip()
    if name == "mean":
        return np.nanmean
    if name == "median":
        return np.nanmedian
    raise ValueError("AGG must be 'mean' or 'median'")


def aggregate_methylation_probes_to_genes(
    M_fxS: pd.DataFrame,
    mapping: pd.DataFrame,
    agg: str = "median",
    min_probes_per_gene: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """
    Inputs
    - M_fxS: probes x samples methylation matrix (index = probe IDs, columns = sample IDs)
    - mapping: DataFrame with columns ['probe_id','gene_id']

    Returns
    - G_gxS: genes x samples aggregated matrix
    - report: dict of counts / diagnostics
    """
    if not {"probe_id", "gene_id"}.issubset(mapping.columns):
        raise ValueError("Mapping file must contain columns: probe_id,gene_id")

    M = M_fxS.copy()
    M.index = M.index.astype(str)
    M.columns = M.columns.astype(str)

    mp = mapping.copy()
    mp["probe_id"] = mp["probe_id"].astype(str)
    mp["gene_id"] = mp["gene_id"].astype(str)

    # keep probes present in M
    probes_in_M = set(M.index)
    mp = mp[mp["probe_id"].isin(probes_in_M)].copy()
    if mp.empty:
        raise ValueError("No probe_id overlap between mapping file and methylation matrix index.")

    # count probes per gene and filter
    gene_counts = mp.groupby("gene_id")["probe_id"].nunique().sort_values(ascending=False)
    keep_genes = gene_counts[gene_counts >= int(min_probes_per_gene)].index
    mp = mp[mp["gene_id"].isin(keep_genes)].copy()
    if mp.empty:
        raise ValueError(
            f"After min_probes_per_gene={min_probes_per_gene}, no genes remain. "
            f"Try lowering to 2."
        )

    gene_to_probes = mp.groupby("gene_id")["probe_id"].apply(list)

    fn = _agg_fn(agg)
    out = []
    genes = []
    for gene, probes in gene_to_probes.items():
        block = M.loc[probes].to_numpy(dtype=float)  # probes x samples
        vec = fn(block, axis=0)                      # samples
        out.append(vec)
        genes.append(gene)

    G_gxS = pd.DataFrame(np.vstack(out), index=genes, columns=M.columns)

    report = {
        "methylation_shape_probes_x_samples": [int(M.shape[0]), int(M.shape[1])],
        "mapping_rows_after_intersect": int(mp.shape[0]),
        "genes_kept": int(G_gxS.shape[0]),
        "samples": int(G_gxS.shape[1]),
        "agg": agg,
        "min_probes_per_gene": int(min_probes_per_gene),
        "top_genes_by_probe_count": gene_counts.head(10).to_dict(),
    }
    return G_gxS, report
