#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import re

PROJECT = Path("/Users/samyaksrivastava/Desktop/new science fair thing")

PROT_FI   = PROJECT / "proteomics_feature_importance.csv"
PROT_RAW  = PROJECT / "CCR6_TMT_11plex_B1B2B3_fractions_HIV_Proteins.csv"
OUT       = PROJECT / "uniprot_to_gene.csv"

GN_RE = re.compile(r"\bGN=([A-Z0-9\-]+)\b")

def main():
    fi = pd.read_csv(PROT_FI, dtype=str)
    feat_col = None
    for c in fi.columns:
        if c.lower() in ("feature","protein","id","accession"):
            feat_col = c
            break
    if feat_col is None:
        raise ValueError(f"Could not find feature column in {PROT_FI}. Columns={fi.columns.tolist()}")

    feats = fi[feat_col].astype(str).str.strip()

    def norm_u(x: str) -> str:
        x = x.strip()
        if "|" in x and (x.startswith("sp|") or x.startswith("tr|")):
            parts = x.split("|")
            if len(parts) >= 2:
                x = parts[1].strip()
        if "-" in x and x.split("-")[-1].isdigit():
            x = x.split("-")[0]
        return x

    want = set(feats.map(norm_u).tolist())
    want.discard("")
    print("[INFO] unique proteomics FI features:", len(want))
    print("[INFO] example FI features:", list(sorted(want))[:20])

    df = pd.read_csv(PROT_RAW, sep=",", dtype=str, low_memory=False)

    if "Accession" not in df.columns or "Description" not in df.columns:
        raise ValueError(
            f"Expected Accession and Description columns in {PROT_RAW}. Got columns head: {df.columns.tolist()[:40]}")

    mapping = {}
    for _, r in df[["Accession","Description"]].dropna().iterrows():
        acc = norm_u(str(r["Accession"]))
        if acc not in want:
            continue
        m = GN_RE.search(str(r["Description"]))
        if not m:
            continue
        mapping[acc] = m.group(1).strip().upper()

    out = pd.DataFrame(sorted(mapping.items()), columns=["uniprot","gene"])
    out.to_csv(OUT, index=False)

    print(f"[OK] wrote {OUT} with {len(out)} mappings")
    print(f"[INFO] coverage: {len(out)}/{len(want)} = {len(out)/max(1,len(want)):.3f}")
    print(out.head(15).to_string(index=False))

if __name__ == "__main__":
    main()
