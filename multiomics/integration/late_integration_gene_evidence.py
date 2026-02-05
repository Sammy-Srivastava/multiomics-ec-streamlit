#!/usr/bin/env python3
from __future__ import annotations

"""
Late integration WITHOUT shared samples (final robust version)

Key guarantees:
- NEVER references t_gene / p_gene / m_gene before they are created (fixes your UnboundLocalError).
- Transcriptomics is OPTIONAL (will be skipped safely if mapping is unavailable).
- Methylation supports:
    A) FI already uses gene symbols  -> use directly
    B) FI uses cg######## probes     -> map via EPIC manifest
- Proteomics supports UniProt->gene mapping (recommended).

Outputs:
  - UI_stuff/artifacts/late_integration/gene_evidence_table.csv
  - UI_stuff/artifacts/late_integration/convergent_genes.csv
  - UI_stuff/artifacts/late_integration/run_report.json
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

# =========================
# CONFIG (EDIT ONCE)
# =========================
PROJECT = Path("/Users/samyaksrivastava/Desktop/new science fair thing")

TRANS_FI = PROJECT / "transcriptomics_feature_importance.csv"
PROT_FI  = PROJECT / "proteomics_feature_importance.csv"
METH_FI  = PROJECT / "methylation_feature_importance.csv"

PROBE2GENE_T  = PROJECT / "resources" / "mappings" / "probe_to_gene_T.csv"  # optional, for ILMN_ probes
GPL10558      = PROJECT / "GPL10558_HumanHT-12_V4_0_R1_15002873_B.txt"  # optional best-effort
EPIC_MANIFEST = PROJECT / "MethylationEPIC_v-1-0_B4.csv"               # required if methylation FI is cg probes
UNIPROT2GENE  = PROJECT / "uniprot_to_gene.csv"                        # optional but recommended

OUT_DIR = PROJECT / "UI_stuff" / "artifacts" / "late_integration"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_EVIDENCE = OUT_DIR / "gene_evidence_table.csv"
OUT_CONVERG  = OUT_DIR / "convergent_genes.csv"
OUT_REPORT   = OUT_DIR / "run_report.json"

TOP_N = 300
MIN_MODALITIES = 2


# =========================
# Utilities
# =========================
def _read_any_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    common = dict(engine="python", dtype=str, on_bad_lines="skip")
    for comment in [None, "#", "!"]:
        kw = dict(common)
        if comment is not None:
            kw["comment"] = comment

        # sniff
        try:
            df = pd.read_csv(path, sep=None, **kw)
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass

        for sep in ["\t", ",", "|", ";"]:
            try:
                df = pd.read_csv(path, sep=sep, **kw)
                if df.shape[1] >= 2:
                    return df
            except Exception:
                pass

    raise RuntimeError(f"Could not parse {path} as a 2+ column table with common delimiters.")


def _pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    if required:
        raise ValueError(f"Missing expected columns {candidates}. Columns head: {df.columns.tolist()[:50]}")
    return None


def _as_gene_list(cell) -> list[str]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    s = str(cell).strip()
    if not s or s.upper() in {"NA", "N/A", "NONE"}:
        return []
    s = s.replace("///", ";").replace(",", ";")
    parts = [p.strip() for p in s.split(";") if p.strip()]
    out = []
    for p in parts:
        if " " in p and len(p) > 10:
            out.extend([x.strip() for x in p.split() if x.strip()])
        else:
            out.append(p)
    return sorted(set(out))


def _norm_gene(g) -> str:
    if g is None:
        return ""
    s = str(g).strip().upper()
    if not s or s in {"NA", "N/A", "NONE"}:
        return ""
    return s


def _percentile_rank(x: pd.Series) -> pd.Series:
    r = x.rank(method="average", ascending=True)
    return (r - 1) / (len(r) - 1) if len(r) > 1 else pd.Series([1.0] * len(r), index=x.index)


def _looks_like_gene_symbol(s: str) -> bool:
    if not s:
        return False
    s = s.strip()
    if s.startswith("ILMN_"):
        return False
    if len(s) > 24:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9\-\._]*", s))


# =========================
# Transcriptomics mapping
# =========================
def load_probe2gene_optional(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    df = _read_any_table(path)
    pcol = _pick_col(df, ["probe", "Probe", "ID", "ilmn", "IlmnID", "ILMN_ID"], required=True)
    gcol = _pick_col(df, ["gene", "Gene", "Symbol", "GeneSymbol", "Gene Symbol"], required=True)

    m: dict[str, list[str]] = {}
    for _, r in df[[pcol, gcol]].dropna().iterrows():
        probe = str(r[pcol]).strip()
        genes = _as_gene_list(r[gcol])
        if probe and genes:
            m[probe] = genes
    return m


def load_gpl10558_map_best_effort(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    header_idx = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            if s.startswith(("ID\t", "ID,", "ILMN_ID\t", "IlmnID\t")):
                header_idx = i
                break
            if ("ID" in s.split("\t")[:6] or "ID" in s.split(",")[:6]) and ("Symbol" in s or "Gene" in s):
                header_idx = i
                break
    if header_idx is None:
        return {}
    try:
        df = pd.read_csv(path, skiprows=header_idx, sep=None, engine="python", dtype=str, on_bad_lines="skip")
        df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]
        probe_col = _pick_col(df, ["ID", "IlmnID", "ilmn_id", "Probe_Id", "ProbeID", "ILMN_ID"], required=True)
        sym_col = _pick_col(df, ["Symbol", "Gene Symbol", "GeneSymbol", "GENE_SYMBOL", "Gene_Symbol"], required=False)
        if sym_col is None:
            sym_col = _pick_col(df, ["Gene", "GENE", "gene", "gene_assignment", "Gene_Assignment"], required=False)
        if sym_col is None:
            return {}
        m: dict[str, list[str]] = {}
        sub = df[[probe_col, sym_col]].dropna(subset=[probe_col])
        for _, row in sub.iterrows():
            probe = str(row[probe_col]).strip()
            genes = _as_gene_list(row[sym_col])
            if probe and genes:
                m[probe] = genes
        return m
    except Exception:
        return {}


def transcriptomics_features_to_genes(fi_path: Path, probe2gene: dict[str, list[str]] | None) -> pd.DataFrame:
    df = _read_any_table(fi_path)
    feat_col = _pick_col(df, ["feature", "Feature", "probe", "Probe", "id", "ID"])
    score_col = _pick_col(df, ["abs_coef", "gain_importance", "importance", "coef", "weight", "score"], required=False)
    if score_col is None:
        raise ValueError("Transcriptomics FI has no recognizable score column (abs_coef/importance/coef/etc).")

    tmp = df[[feat_col, score_col]].copy()
    tmp.columns = ["feature", "score_raw"]
    tmp["feature"] = tmp["feature"].astype(str).str.strip()
    tmp["score_raw"] = pd.to_numeric(tmp["score_raw"], errors="coerce")
    tmp = tmp.dropna(subset=["feature", "score_raw"])
    tmp["base"] = tmp["feature"].str.split("__").str[0].str.strip()

    sample = tmp["base"].head(200).tolist()
    gene_like_frac = float(np.mean([_looks_like_gene_symbol(x) for x in sample])) if sample else 0.0

    # gene symbol path
    if gene_like_frac >= 0.7:
        out = tmp.rename(columns={"base": "gene"})[["gene", "score_raw"]].copy()
        out["gene"] = out["gene"].map(_norm_gene)
        out = out[out["gene"].astype(bool)]
        out = out.groupby("gene", as_index=False)["score_raw"].max()
        out = out.rename(columns={"score_raw": "score"})
        out["modality"] = "transcriptomics"
        return out

    # probe path
    if not probe2gene:
        raise RuntimeError("Transcriptomics FI looks like probes, but no probe_to_gene_T.csv / usable GPL mapping found.")

    rows = []
    for _, r in tmp.iterrows():
        probe = r["base"]
        score = float(r["score_raw"])
        genes = probe2gene.get(probe, [])
        for g in genes:
            rows.append((g, score))

    out = pd.DataFrame(rows, columns=["gene", "score"])
    if out.empty:
        raise RuntimeError("Transcriptomics mapping produced 0 genes (probe IDs likely do not match mapping).")

    out["gene"] = out["gene"].map(_norm_gene)
    out = out[out["gene"].astype(bool)]
    out = out.groupby("gene", as_index=False)["score"].max()
    out["modality"] = "transcriptomics"
    return out


# =========================
# Methylation mapping
# =========================
_CG_RE = re.compile(r"(cg\d{8})", flags=re.IGNORECASE)

def _normalize_methyl_probe(x: str) -> str:
    if x is None:
        return ""
    s = str(x).strip().strip('"').strip("'")
    if not s:
        return ""
    base = s.split("__")[0].strip()
    m = _CG_RE.search(base)
    if m:
        return m.group(1).lower()
    return base.lower()


def load_epic_manifest_map(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing EPIC manifest: {path}")

    header_idx = None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            if "IlmnID" in s.split(","):
                header_idx = i
                break
    if header_idx is None:
        raise ValueError("Could not locate EPIC manifest header row containing 'IlmnID'.")

    df = pd.read_csv(path, skiprows=header_idx, sep=",", engine="python", dtype=str, on_bad_lines="skip")
    df.columns = [str(c).strip().strip('"').strip("'") for c in df.columns]

    probe_col = _pick_col(df, ["IlmnID", "ID", "Name", "ProbeID", "probe_id", "cgid"], required=True)
    gene_col = _pick_col(df, ["UCSC_RefGene_Name", "RefGene_Name", "Gene", "GENE", "Gene_Name"], required=False)
    if gene_col is None:
        raise ValueError("EPIC manifest loaded but no gene column found (UCSC_RefGene_Name/RefGene_Name/etc).")

    m: dict[str, list[str]] = {}
    for _, row in df[[probe_col, gene_col]].dropna(subset=[probe_col]).iterrows():
        probe = str(row[probe_col]).strip().strip('"').strip("'").lower()
        genes = _as_gene_list(row[gene_col])
        if probe and genes:
            m[probe] = genes
    return m


def methylation_features_to_genes(fi_path: Path, epic_map: dict[str, list[str]] | None) -> pd.DataFrame:
    df = _read_any_table(fi_path)
    feat_col = _pick_col(df, ["feature", "Feature", "probe", "Probe", "id", "ID"])
    score_col = _pick_col(df, ["gain_importance", "importance", "abs_coef", "coef", "score", "weight"], required=False)
    if score_col is None:
        raise ValueError("Methylation FI has no recognizable score column (gain_importance/importance/coef/etc).")

    tmp = df[[feat_col, score_col]].copy()
    tmp.columns = ["feature", "score_raw"]
    tmp["feature"] = tmp["feature"].astype(str).str.strip()
    tmp["score_raw"] = pd.to_numeric(tmp["score_raw"], errors="coerce")
    tmp = tmp.dropna(subset=["feature", "score_raw"])
    tmp["base"] = tmp["feature"].str.split("__").str[0].str.strip()

    sample = tmp["base"].head(200).tolist()
    gene_like_frac = float(np.mean([_looks_like_gene_symbol(x) for x in sample])) if sample else 0.0
    cg_like_frac = float(np.mean([bool(re.fullmatch(r"(?i)cg\d{8}", str(x).strip())) for x in sample])) if sample else 0.0

    # gene symbol path
    if gene_like_frac >= 0.7 and cg_like_frac < 0.2:
        out = tmp.rename(columns={"base": "gene"})[["gene", "score_raw"]].copy()
        out["gene"] = out["gene"].map(_norm_gene)
        out = out[out["gene"].astype(bool)]
        out = out.groupby("gene", as_index=False)["score_raw"].max()
        out = out.rename(columns={"score_raw": "score"})
        out["modality"] = "methylation"
        return out

    # probe path (cg#######)
    if not epic_map:
        raise RuntimeError("Methylation FI looks like probes (cg########) but EPIC manifest mapping was not loaded.")

    rows = []
    for _, r in tmp.iterrows():
        probe = _normalize_methyl_probe(r["base"])
        score = float(r["score_raw"])
        genes = epic_map.get(probe, [])
        for g in genes:
            rows.append((g, score))

    out = pd.DataFrame(rows, columns=["gene", "score"])
    if out.empty:
        raise RuntimeError(
            "No methylation probes mapped to genes. "
            "This usually means FI probe IDs do not match EPIC manifest probe IDs."
        )

    out["gene"] = out["gene"].map(_norm_gene)
    out = out[out["gene"].astype(bool)]
    out = out.groupby("gene", as_index=False)["score"].max()
    out["modality"] = "methylation"
    return out


# =========================
# Proteomics mapping
# =========================
def load_uniprot_map_optional(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    df = _read_any_table(path)
    ucol = _pick_col(df, ["uniprot", "UniProt", "accession", "Accession", "Entry"], required=True)
    gcol = _pick_col(df, ["gene", "Gene", "GeneSymbol", "Symbol"], required=True)
    m: dict[str, str] = {}
    for _, r in df[[ucol, gcol]].dropna().iterrows():
        u = str(r[ucol]).strip()
        g = _norm_gene(r[gcol])
        if u and g:
            m[u] = g
    return m


def proteomics_features_to_genes(fi_path: Path, u2g: dict[str, str]) -> pd.DataFrame:
    df = _read_any_table(fi_path)
    feat_col = _pick_col(df, ["feature", "Feature", "protein", "Protein", "id", "ID"])
    score_col = _pick_col(df, ["abs_coef", "importance", "gain_importance", "coef", "score", "weight"], required=False)
    if score_col is None:
        raise ValueError("Proteomics FI has no recognizable score column (abs_coef/importance/coef/etc).")

    tmp = df[[feat_col, score_col]].copy()
    tmp.columns = ["uniprot", "score_raw"]
    tmp["uniprot"] = tmp["uniprot"].astype(str).str.strip()
    tmp["score_raw"] = pd.to_numeric(tmp["score_raw"], errors="coerce")
    tmp = tmp.dropna(subset=["uniprot", "score_raw"])

    tmp["gene"] = tmp["uniprot"].map(u2g).fillna(tmp["uniprot"])
    tmp["gene"] = tmp["gene"].map(_norm_gene)
    tmp = tmp[tmp["gene"].astype(bool)]

    out = tmp.groupby("gene", as_index=False)["score_raw"].max()
    out = out.rename(columns={"score_raw": "score"})
    out["modality"] = "proteomics"
    return out


# =========================
# Main
# =========================
def main():
    report = {
        "inputs": {
            "transcript_fi": str(TRANS_FI),
            "proteomics_fi": str(PROT_FI),
            "methylation_fi": str(METH_FI),
            "probe_to_gene_T_optional": str(PROBE2GENE_T),
            "gpl10558_optional": str(GPL10558),
            "epic_manifest": str(EPIC_MANIFEST),
            "uniprot_to_gene_optional": str(UNIPROT2GENE),
        },
        "warnings": [],
    }

    # ---- mappings ----
    probe2gene_t = load_probe2gene_optional(PROBE2GENE_T)
    if probe2gene_t:
        report["warnings"].append(f"Using transcriptomics probe->gene cache: {PROBE2GENE_T} ({len(probe2gene_t)} probes)")
    else:
        gpl_map = load_gpl10558_map_best_effort(GPL10558)
        if gpl_map:
            probe2gene_t = gpl_map
            report["warnings"].append(f"Using transcriptomics mapping from GPL best-effort: {GPL10558} ({len(probe2gene_t)} probes)")
        else:
            report["warnings"].append(
                "No usable transcriptomics probe->gene mapping found. "
                "If transcriptomics FI uses ILMN_ probes, provide probe_to_gene_T.csv (probe,gene)."
            )
            probe2gene_t = {}

    epic_map = None
    try:
        epic_map = load_epic_manifest_map(EPIC_MANIFEST)
    except Exception as e:
        report["warnings"].append(f"EPIC manifest not usable/needed: {e}")
        epic_map = None

    u2g = load_uniprot_map_optional(UNIPROT2GENE)
    if not u2g:
        report["warnings"].append(
            "No UniProt->Gene mapping file found. Proteomics will be integrated using UniProt IDs as gene keys. "
            "To improve, create uniprot_to_gene.csv with columns (uniprot,gene)."
        )

    # ---- gene-level FI per modality (create variables BEFORE any diagnostics) ----
    p_gene = proteomics_features_to_genes(PROT_FI, u2g)
    m_gene = methylation_features_to_genes(METH_FI, epic_map)

    try:
        t_gene = transcriptomics_features_to_genes(TRANS_FI, probe2gene_t if probe2gene_t else None)
    except Exception as e:
        report["warnings"].append(f"Transcriptomics excluded: {e}")
        t_gene = pd.DataFrame(columns=["gene", "score", "modality"])

    # ---- overlap diagnostics (SAFE) ----
    T = set(t_gene["gene"]) if len(t_gene) else set()
    P = set(p_gene["gene"]) if len(p_gene) else set()
    M = set(m_gene["gene"]) if len(m_gene) else set()

    print("\n=== OVERLAP DIAGNOSTICS ===")
    print("T genes:", len(T), "P genes:", len(P), "M genes:", len(M))
    print("T∩P:", len(T & P))
    print("T∩M:", len(T & M))
    print("M∩P:", len(M & P))
    print("T∩M∩P:", len(T & M & P))
    print("Example M∩P (up to 25):", list(sorted(M & P))[:25])
    print("=== END DIAGNOSTICS ===\n")

    report["mapped_counts"] = {
        "transcriptomics_genes": int(len(t_gene)),
        "proteomics_genes": int(len(p_gene)),
        "methylation_genes": int(len(m_gene)),
    }

    # ---- union table ----
    genes_union = set(p_gene["gene"]) | set(m_gene["gene"])
    if len(t_gene):
        genes_union |= set(t_gene["gene"])
    ev = pd.DataFrame({"gene": sorted(genes_union)})

    ev = ev.merge(p_gene[["gene", "score"]].rename(columns={"score": "score_proteomics"}), on="gene", how="left")
    ev = ev.merge(m_gene[["gene", "score"]].rename(columns={"score": "score_methylation"}), on="gene", how="left")
    if len(t_gene):
        ev = ev.merge(t_gene[["gene", "score"]].rename(columns={"score": "score_transcriptomics"}), on="gene", how="left")
    else:
        ev["score_transcriptomics"] = np.nan

    # ---- percentile ranks ----
    for mod in ["transcriptomics", "proteomics", "methylation"]:
        sc = f"score_{mod}"
        rk = f"rank_{mod}"
        mask = ev[sc].notna()
        sub = ev.loc[mask, sc]
        if len(sub) >= 2:
            ev.loc[mask, rk] = _percentile_rank(sub)
        elif len(sub) == 1:
            ev.loc[mask, rk] = 1.0
        else:
            ev[rk] = np.nan

    rank_cols = ["rank_transcriptomics", "rank_proteomics", "rank_methylation"]
    ev["n_modalities"] = ev[rank_cols].notna().sum(axis=1)
    ev["integrated_score"] = ev[rank_cols].mean(axis=1, skipna=True)

    def _mods_present(row) -> str:
        mods = []
        if pd.notna(row.get("rank_transcriptomics")): mods.append("T")
        if pd.notna(row.get("rank_proteomics")): mods.append("P")
        if pd.notna(row.get("rank_methylation")): mods.append("M")
        return "".join(mods)

    ev["modalities_present"] = ev.apply(_mods_present, axis=1)

    ev_sorted = ev.sort_values(["n_modalities", "integrated_score"], ascending=[False, False])
    convergent = ev_sorted[ev_sorted["n_modalities"] >= MIN_MODALITIES].head(TOP_N).copy()

    ev_sorted.to_csv(OUT_EVIDENCE, index=False)
    convergent.to_csv(OUT_CONVERG, index=False)

    report["outputs"] = {
        "gene_evidence_table": str(OUT_EVIDENCE),
        "convergent_genes": str(OUT_CONVERG),
        "run_report": str(OUT_REPORT),
    }
    report["summary"] = {
        "n_genes_union": int(len(ev_sorted)),
        "n_convergent_genes_topN": int(len(convergent)),
        "n_genes_supported_by_2plus": int((ev_sorted["n_modalities"] >= 2).sum()),
        "n_genes_supported_by_3": int((ev_sorted["n_modalities"] >= 3).sum()),
        "top5_convergent": convergent["gene"].head(5).tolist() if len(convergent) else [],
    }

    OUT_REPORT.write_text(json.dumps(report, indent=2))

    print("[OK] wrote:", OUT_EVIDENCE)
    print("[OK] wrote:", OUT_CONVERG)
    print("[OK] wrote:", OUT_REPORT)
    print("\n=== SUMMARY ===")
    print(json.dumps(report["summary"], indent=2))
    if report["warnings"]:
        print("\n=== WARNINGS ===")
        for w in report["warnings"]:
            print("-", w)


if __name__ == "__main__":
    main()
