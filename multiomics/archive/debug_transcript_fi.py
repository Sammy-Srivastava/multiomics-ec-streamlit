from pathlib import Path
import pandas as pd

PROJECT = Path("/Users/samyaksrivastava/Desktop/new science fair thing")
TRANS_FI = PROJECT / "transcriptomics_feature_importance.csv"
P2G = PROJECT / "probe_to_gene_T.csv"

fi = pd.read_csv(TRANS_FI)
p2g = pd.read_csv(P2G)

base = fi["feature"].astype(str).str.strip().str.split("__").str[0].str.strip()
mapped = base.isin(set(p2g["probe"].astype(str).str.strip()))

print("Unmapped probes:", base[~mapped].unique().tolist())
