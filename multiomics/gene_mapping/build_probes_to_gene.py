#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import gzip
import io
import re
import pandas as pd
import numpy as np

PROJECT = Path("/Users/samyaksrivastava/Desktop/new science fair thing")
TRANS_FI = PROJECT / "transcriptomics_feature_importance.csv"
GPL = PROJECT / "GPL10558_HumanHT-12_V4_0_R1_15002873_B.txt"
OUT = PROJECT / "probe_to_gene_T.csv"

def open_text_maybe_gz(path: Path) -> io.TextIOBase:
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def detect_delim(line: str) -> str | None:
    if "\t" in line:
        return "\t"
    if "," in line:
        return ","
    return None


def find_header_row(path: Path) -> tuple[int, str]:
    """
Find line with ID token and gene-ish column
    """
    with open_text_maybe_gz(path) as f:
        for i, raw in enumerate(f):
            line = raw.strip("\n")
            if not line.strip():
                continue

            delim = detect_delim(line)
            if not delim:
                continue

            parts = [p.strip().strip('"').strip("'") for p in line.split(delim)]
            parts_low = [p.lower() for p in parts[:80]]

            #must have a probe/id column
            has_id = any(x in parts_low for x in ["id", "ilmnid", "ilmn_id", "probeid", "probe_id", "search_key"])
            #must have a gene-ish column
            has_gene = any(("symbol" in x) or ("gene" in x) for x in parts_low)

            if has_id and has_gene:
                return i, delim

    raise RuntimeError(f"Could not find a header row in {path.name}. This file likely isn't the right GPL annotation dump.")


def as_gene_list(cell: str) -> list[str]:
    if cell is None:
        return []
    s = str(cell).strip()
    if not s or s.upper() in {"NA", "N/A", "NONE"}:
        return []
    s = s.replace("///", ";").replace(",", ";")
    parts = [p.strip() for p in s.split(";") if p.strip()]
    # duplicate again and discard oultiers
    out = []
    for p in parts:
        out.append(p)
    return sorted(set(out))


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    if required:
        raise ValueError(f"Missing expected columns. Tried {candidates}. Columns head: {df.columns.tolist()[:60]}")
    return None


def main():
    if not GPL.exists():
        raise FileNotFoundError(f"Missing GPL file: {GPL}\nSet GPL to your actual GPL10558 txt(.gz) file path.")

    header_idx, delim = find_header_row(GPL)
    print(f"[INFO] Found header at line {header_idx} with delimiter {repr(delim)}")

    df = pd.read_csv(
        GPL,
        sep=delim,
        skiprows=header_idx,
        engine="python",
        dtype=str,
        on_bad_lines="skip"
    )

    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]

    probe_col = pick_col(df, ["ID", "IlmnID", "ILMN_ID", "ProbeID", "Probe_Id", "probe_id", "search_key"], required=True)

    gene_col = pick_col(
        df,
        [
            "Symbol",
            "Gene Symbol",
            "GeneSymbol",
            "GENE_SYMBOL",
            "Gene_Symbol",
            "gene_symbol",
            "Gene",
            "GENE",
            "gene",
            "gene_assignment",
            "Gene_Assignment",
            "ILMN_Gene",
            "IlmnGene",
        ],
        required=False
    )

    if gene_col is None:
        for c in df.columns:
            cl = str(c).lower()
            if ("symbol" in cl) or (("gene" in cl) and ("group" not in cl)):
                gene_col = c
                break

    if gene_col is None:
        raise RuntimeError(
            "Could not locate a gene symbol column in GPL file after reading it. "
            f"Columns head: {df.columns.tolist()[:80]}"
        )

    sub = df[[probe_col, gene_col]].dropna(subset=[probe_col])
    out_rows = []
    for _, r in sub.iterrows():
        probe = str(r[probe_col]).strip()
        genes = as_gene_list(r[gene_col])
        if probe and genes:
            out_rows.append((probe, genes[0]))

    out_df = pd.DataFrame(out_rows, columns=["probe", "gene"])
    out_df["probe"] = out_df["probe"].astype(str).str.strip()
    out_df["gene"] = out_df["gene"].astype(str).str.strip()
    out_df = out_df[out_df["gene"].str.lower().ne("nan")]
    out_df = out_df[out_df["gene"].ne("")]

    # drop empties
    out_df = out_df[(out_df["probe"] != "") & (out_df["gene"] != "")]
    out_df = out_df.drop_duplicates(subset=["probe"], keep="first")

    if out_df.empty:
        raise RuntimeError("Produced an empty probe->gene table. GPL parsing succeeded but gene column content was unusable.")

    out_df.to_csv(OUT, index=False)
    print(f"[OK] wrote {OUT} with {len(out_df)} mappings")
    print("[INFO] probe head:", out_df["probe"].head(5).tolist())
    print("[INFO] gene head:", out_df["gene"].head(5).tolist())

    #coverage check against transcriptomics FI
    if TRANS_FI.exists():
        fi = pd.read_csv(TRANS_FI)
        feat = fi["feature"].astype(str).str.strip()
        base = feat.str.split("__").str[0].str.strip()
        mapped = base.isin(set(out_df["probe"]))
        print(f"[CHECK] Transcriptomics FI rows: {len(fi)}")
        print(f"[CHECK] Unique FI probes: {base.nunique()}")
        print(f"[CHECK] Mapped fraction: {mapped.mean():.3f}")
        print(f"[CHECK] Mapped count: {mapped.sum()}/{len(fi)}")
        print("[CHECK] Unmapped examples:", base[~mapped].head(20).tolist())
    else:
        print("[WARN] transcriptomics_feature_importance.csv not found; skipping coverage check.")

if __name__ == "__main__":
    main()
