import os
import re
import json
import argparse
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cg_re = re.compile(r"^cg\d{8}$", re.IGNORECASE)

def read_matrix(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.parquet':
        X = pd.read_parquet(path)
        X.index = X.index.astype(str)
        return X
    if ext == '.csv':
        df = pd.read_csv(path)
        if 'sample_id' in df.columns:
            df['sample_id'] = df['sample_id'].astype(str)
            df = df.set_index('sample_id')
        else:
            df = df.set_index(df.columns[0])
            df.index = df.index.astype(str)
        return df
    raise ValueError(f'Unsupported input format: {path}')

def coerce_numeric(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.apply(pd.to_numeric, errors='coerce')
    return Xn.astype(np.float32, copy=False)

def ensure_unique_columuns(G: pd.DataFrame) -> pd.DataFrame:
    if G.columns.duplicated().any():
        G = G.groupby(G.columns, axis=1).mean()
    return G

def methylation_to_genes(
    X_meth: pd.DataFrame,
    probe_map: pd.DataFrame,
    probe_col: str = 'probe_id',
    gene_col: str = 'gene_id'
) -> pd.DataFrame:
    probe_map = probe_map[[probe_col, gene_col]].dropna()
    probe_map[probe_col] = probe_map[probe_col].astype(str)
    probe_map[gene_col] = probe_map[gene_col].astype(str)
    
    probes_in_X = set(map(str, X_meth.columns))
    pm = probe_map[probe_map[probe_col].isin(probes_in_X)].copy()
    if pm.empty:
        raise ValueError('No overlap between methylation probes and probe to gene mapping')
    Xp = X_meth.loc[:, pm[probe_col].values].copy()
    Xp.columns = pm[gene_col].values
    G = Xp.T.groupby(level=0, axis=1).mean().T
    G = ensure_unique_columuns(G)
    return G

def pca_gene_signature(G: pd.DataFrame, k: int=10, random_state: int = 42) -> Tuple[pd.DataFrame, Dict]:
    G = coerce_numeric(G)
    n, p = G.shape
    if n < 2 or p < 2:
        raise ValueError(f"Need at least 2 samples and 2 genes. You have {G.shape}")
    
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(G.values)
    
    k_eff = min(k, n - 1, p)
    pca = PCA(n_components=k_eff, random_state=random_state)
    pca.fit(Xs)
    
    loadings = pca.components_[0].astype(np.float64)
    gene_ids = G.columns.astype(str).values
    
    out = pd.DataFrame({'gene_id': gene_ids, 'score': loadings})
    stats = {
        'samples': int(n),
        'genes': int(p),
        'k_eff': int(k_eff),
        'explained_variance_pc1': float(pca.explained_variance_ratio_[0]),
        'explained_variance_sum': float(pca.explained_variance_ratio_.sum())
    }
    return out, stats

def zscore_series(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return x * 0.0
    return (x - mu)/sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x_methyl', required=True, help='Methylation matrix (parquet/csv): samples x probes')
    ap.add_argument('--x_rna', required=True, help='Transcriptomics matrix (parquet/csv): samples x probes')
    ap.add_argument('--x_prot', required=True, help='Proteomics matrix (parquet/csv): samples x probes')
    ap.add_argument('--probe_to_gene', required=True, help='CSV mapping: probe_id, gene_id')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--pca_k', type=int, default=10)
    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    Xm = read_matrix(args.x_methyl)
    Xr = read_matrix(args.x_rna)
    Xp = read_matrix(args.x_prot)
    
    Xm = coerce_numeric(Xm)
    Xr = coerce_numeric(Xr)
    Xp = coerce_numeric(Xp)
    
    map_df = pd.read_csv(args.probe_to_gene)
    Gm = methylation_to_genes(Xm, map_df, probe_col='probe_id', gene_col='gene_id')
    
    Gr = ensure_unique_columuns(Xr)
    Gp = ensure_unique_columuns(Xp)
    
    sm, st_m = pca_gene_signature(Gm, k=args.pca_k)
    sr, st_r = pca_gene_signature(Gm, k=args.pca_k)
    sp, st_p = pca_gene_signature(Gm, k=args.pca_k)
    
    sm_path = os.path.join(args.out_dir, 'gene_signature_methylation.csv')
    sr_path = os.path.join(args.out_dir, 'gene_signature_transcriptomics.csv')
    sp_path = os.path.join(args.out_dir, 'gene_signature_proteomics.csv')
    sm.to_csv(sm_path, index=False)
    sr.to_csv(sr_path, index=False)
    sp.to_csv(sp_path, index=False)
    
    merged = (
        sm.rename(columns={'score': 'score_methyl'})
        .merge(sr.rename(columns={'score': 'score_rna'}), on='gene_id', how='inner')
        .merge(sp.rename(columns={'score': 'score_prot'}), on='gene_id', how='inner')
    )
    
    if merged.shape[0] < 50:
        print(f'Warning: Only {merged.shape[0]} overlapping genes across modalities')
        
    merged['z_methyl'] = zscore_series(merged['score_methyl'])
    merged['z_rna'] = zscore_series(merged['score_rna'])
    merged['z_prot'] = zscore_series(merged['score_prot'])
    merged['consensus_z_mean'] = zscore_series(merged[['z_methyl', 'z_rna', 'z_prot']].mean(axis=1))
    
    out_path = os.path.join(args.out_dir, 'late_integration_gene_scores.csv')
    merged.sort_values('consensus_z_mean', ascending=False).to_csv(out_path, index=False)
    
    report = {
        'inputs': {
            'x_methyl': args.x_methyl,
            'x_rna': args.x_rna,
            'x_prot': args.x_prot,
            'probe_to_gene': args.probe_to_gene
        },
        'per_omic_stats': {
            'methylation': st_m,
            'transcriptomics': st_r,
            'proteomics': st_p
        },
        'outputs': {
            'gene_signature_methylation': sm_path,
            'gene_signature_transcriptomics': sr_path,
            'gene_signature_proteomics': sp_path,
            'late_integrated_gene_score': out_path
        },
        'overlapping_genes': int(merged.shape[0]),
    }
    with open(os.path.join(args.out_dir, 'gene_late_integration_report.json'), 'w') as f:
        json.dump(report, f, indent=2)
        
    print('Saved:', out_path)
    print('Overlapping genes:', merged.shape[0])
    
if __name__ == '__main__':
    main()
    