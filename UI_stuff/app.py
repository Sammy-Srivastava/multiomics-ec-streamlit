# UI_stuff/app.py
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import sys

import os
import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# Path setup
# =============================================================================
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

upload_dir = project_root / "UI_stuff" / "data" / "uploads"
artifact_dir = project_root / "UI_stuff" / "artifacts"
log_dir = project_root / "UI_stuff" / "logs"

for d in (upload_dir, artifact_dir, log_dir):
    d.mkdir(parents=True, exist_ok=True)

HARMONIZED_ROOT = artifact_dir / "harmonized"
HARMONIZED_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_PROBE2GENE_PATH = project_root / "annotations" / "probe_to_gene.csv"

ALLOWED_UPLOAD_EXT = {".csv", ".tsv", ".txt", ".parquet", ".gz"}
BAD_EXT = (".raw", ".tar", ".mzml", ".wiff", ".d", ".cdf")

OMIC_KEYS = {"M": "Methylation", "T": "Transcriptomics", "P": "Proteomics", "Mb": "Metabolomics", "G": "Genomics"}
_CANON_WS = re.compile(r"\s+")

LABEL_COL_ALIASES = {
    "sample_id": {"sample_id", "sample", "id", "gsm", "geo_accession", "subject", "patient"},
    "label": {"label", "group", "class", "y", "phenotype", "condition"},
}

IS_CLOUD = bool(st.secrets.get('STREAMLIT_CLOUD', False)) or bool(os.environ.get('STREAMLIT_SERVER_RUNNING', ''))

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(page_title="Omics Harmonizer", layout="wide")
st.title("Omics Harmonizer Dashboard")
st.caption("Upload → Harmonize → Train unimodal models → Combine → Board exports")


# =============================================================================
# Utilities
# =============================================================================
def safe_filename(name: str) -> str:
    name = name.replace("\\", "_").replace("/", "_")
    return "".join(c for c in name if c.isalnum() or c in "._-")


def now_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def human_bytes(n: int) -> str:
    if n is None:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:,.1f} {u}"
        x /= 1024.0
    return f"{x:,.1f} PB"


def validate_input_file(p: Path) -> Tuple[bool, str]:
    p = Path(p)
    if not p.exists():
        return False, "File does not exist."
    if p.stat().st_size == 0:
        return False, "File is empty."
    name_lower = p.name.lower()
    if name_lower.endswith(BAD_EXT):
        return False, f"Unsupported binary/raw file type: {p.suffix}"
    if (p.suffix.lower() not in ALLOWED_UPLOAD_EXT) and (not name_lower.endswith((".tsv.gz", ".txt.gz", ".csv.gz"))):
        return False, f"Unsupported extension: {p.suffix}"
    return True, "OK"


def save_uploaded_file(uploaded) -> Path:
    fname = safe_filename(uploaded.name)
    out_path = upload_dir / fname
    uploaded.seek(0)
    with open(out_path, "wb") as f:
        shutil.copyfileobj(uploaded, f)
    return out_path


@st.cache_data(show_spinner=False)
def list_uploaded_relpaths_cached(upload_dir_str: str) -> List[str]:
    ud = Path(upload_dir_str)
    subfiles = [p for p in ud.rglob("*") if p.is_file()]
    return sorted({str(p.relative_to(ud)) for p in subfiles})


@st.cache_data(show_spinner=False)
def list_runs_cached(root_str: str) -> List[str]:
    rr = Path(root_str)
    runs = [p for p in rr.glob("*") if p.is_dir()]
    runs = sorted(runs, key=lambda p: p.name, reverse=True)
    return [p.name for p in runs]


def read_matrix_shape(path: Path) -> Tuple[int, int]:
    p = Path(path)
    if not p.exists():
        return (0, 0)
    try:
        if str(p).lower().endswith(".parquet"):
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p, index_col=0)
        return (int(df.shape[0]), int(df.shape[1]))
    except Exception:
        return (0, 0)


@st.cache_data(show_spinner=False)
def read_matrix_head_cached(path_str: str, nrows: int = 25) -> pd.DataFrame:
    p = Path(path_str)
    name_lower = p.name.lower()

    if name_lower.endswith(".parquet"):
        df = pd.read_parquet(p)
        df.columns = df.columns.astype(str).str.strip()
        df.index = df.index.astype(str).str.strip()
        return df.head(nrows)

    compression = "gzip" if name_lower.endswith(".gz") else None
    sep = "\t" if name_lower.endswith((".tsv", ".txt", ".tsv.gz", ".txt.gz")) else ","

    try:
        df = pd.read_csv(p, sep=sep, index_col=0, compression=compression, nrows=nrows, engine="python")
    except Exception as e:
        msg = str(e)
        if "Duplicate column names" not in msg and "duplicate" not in msg.lower():
            raise

        # ---- Fallback: manual header -> unique -> read with header=None ----
        # Read just the first line (header) ourselves

        if compression == "gzip":
            import gzip
            with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
                header_line = f.readline()
        else:
            header_line = p.read_text(encoding="utf-8", errors="ignore").splitlines()[0]

        raw_cols = [c.strip() for c in header_line.rstrip("\n").split(sep)]
        raw_cols = make_unique_columns(raw_cols)

        df = pd.read_csv(
            p,
            sep=sep,
            header=None,
            names=raw_cols,
            skiprows=1,
            index_col=0,
            compression=compression,
            nrows=nrows,
            engine="python",
        )

    df.columns = df.columns.astype(str).str.strip()
    df.index = df.index.astype(str).str.strip()
    return df



def canonicalize_sample_ids_basic(cols: pd.Index) -> pd.Index:
    return cols.astype(str).str.strip().str.replace(_CANON_WS, "_", regex=True)


def canonicalize_sample_ids(omic_key: str, cols: pd.Index) -> pd.Index:
    cols = cols.astype(str)
    if omic_key == "P":
        out = []
        for c in cols:
            m2 = re.search(r"\b(EC\d+|ART\d+|HC\d+)\b", c, flags=re.I)
            out.append(m2.group(1).upper() if m2 else c.strip())
        return pd.Index(out)
    return canonicalize_sample_ids_basic(pd.Index(cols))

def make_unique_columns(cols: List[str]) -> List[str]:
    """
    Ensure column names are unique by suffixing duplicates: col, col__2, col__3, ...
    """
    seen: Dict[str, int] = {}
    out: List[str] = []
    for c in cols:
        c = str(c)
        if c not in seen:
            seen[c] = 1
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out


def collapse_duplicate_columns_numeric(df: pd.DataFrame, how: str = "mean") -> pd.DataFrame:
    """
    If df has duplicate column names, combine duplicates across columns using mean/median (numeric only).
    Non-numeric values -> NaN.
    """
    if not df.columns.duplicated().any():
        return df

    X = df.apply(pd.to_numeric, errors="coerce")

    if how == "median":
        agg = np.nanmedian
    else:
        agg = np.nanmean

    grouped: Dict[str, np.ndarray] = {}
    for col in pd.Index(df.columns).unique():
        block = X.loc[:, df.columns == col].to_numpy(dtype=float)
        grouped[col] = agg(block, axis=1)

    return pd.DataFrame(grouped, index=df.index)

    
# =============================================================================
# Labels: upload + validate
# =============================================================================
def _pick_col(df: pd.DataFrame, alias_set: set) -> Optional[str]:
    cols = {str(c).strip().lower(): c for c in df.columns}
    for a in alias_set:
        if a in cols:
            return cols[a]
    return None


@st.cache_data(show_spinner=False)
def read_labels_cached(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    name_lower = p.name.lower()
    sep = "\t" if name_lower.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(p, sep=sep, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def validate_labels_df(df: pd.DataFrame) -> Tuple[bool, str, Optional[pd.DataFrame]]:
    if df is None or df.empty:
        return False, "Labels file is empty.", None

    sid_col = _pick_col(df, LABEL_COL_ALIASES["sample_id"])
    lab_col = _pick_col(df, LABEL_COL_ALIASES["label"])
    if sid_col is None or lab_col is None:
        return False, f"Labels must include sample_id + label (found {df.columns.tolist()})", None

    out = df[[sid_col, lab_col]].copy()
    out.columns = ["sample_id", "label"]
    out["sample_id"] = out["sample_id"].astype(str).str.strip()
    out["label"] = out["label"].astype(str).str.strip()
    out = out.dropna()
    out = out[(out["sample_id"] != "") & (out["label"] != "")]
    out = out.drop_duplicates("sample_id", keep="first")
    if out.empty:
        return False, "After cleaning, labels are empty.", None
    return True, "OK", out


def labels_template_df() -> pd.DataFrame:
    return pd.DataFrame({"sample_id": ["GSM0000001", "GSM0000002", "GSM0000003"], "label": ["EC", "ART", "EC"]})


# =============================================================================
# Probe→gene mapping (Option B: default or upload)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_default_probe2gene() -> pd.DataFrame:
    if not DEFAULT_PROBE2GENE_PATH.exists():
        raise FileNotFoundError(f"Default mapping not found: {DEFAULT_PROBE2GENE_PATH}")
    mp = pd.read_csv(DEFAULT_PROBE2GENE_PATH)
    if not {"probe_id", "gene_id"}.issubset(mp.columns):
        raise ValueError("Default annotations/probe_to_gene.csv must contain columns: probe_id,gene_id")
    mp["probe_id"] = mp["probe_id"].astype(str)
    mp["gene_id"] = mp["gene_id"].astype(str)
    return mp


def load_uploaded_probe2gene(uploaded_file) -> pd.DataFrame:
    mp = pd.read_csv(uploaded_file)
    if not {"probe_id", "gene_id"}.issubset(mp.columns):
        raise ValueError("Uploaded mapping must contain columns: probe_id,gene_id")
    mp["probe_id"] = mp["probe_id"].astype(str)
    mp["gene_id"] = mp["gene_id"].astype(str)
    return mp


def aggregate_methylation_probes_to_genes(
    M_fxS: pd.DataFrame,
    mapping: pd.DataFrame,
    agg: str = "median",
    min_probes_per_gene: int = 3,
) -> Tuple[pd.DataFrame, dict]:
    if agg not in {"mean", "median"}:
        raise ValueError("agg must be 'mean' or 'median'")

    M_fxS = M_fxS.copy()
    M_fxS.index = M_fxS.index.astype(str)
    M_fxS.columns = M_fxS.columns.astype(str)

    mp = mapping.copy()
    mp["probe_id"] = mp["probe_id"].astype(str)
    mp["gene_id"] = mp["gene_id"].astype(str)

    probes_in_M = set(M_fxS.index)
    mp = mp[mp["probe_id"].isin(probes_in_M)].copy()
    if mp.empty:
        raise ValueError("No probe_id overlap between mapping file and methylation matrix index.")

    gene_counts = mp.groupby("gene_id")["probe_id"].nunique().sort_values(ascending=False)
    keep_genes = gene_counts[gene_counts >= int(min_probes_per_gene)].index
    mp = mp[mp["gene_id"].isin(keep_genes)].copy()
    if mp.empty:
        raise ValueError("No genes remain after min_probes_per_gene filtering.")

    gene_to_probes = mp.groupby("gene_id")["probe_id"].apply(list)

    fn = np.nanmedian if agg == "median" else np.nanmean
    out = []
    genes = []

    for gene, probes in gene_to_probes.items():
        block = M_fxS.loc[probes].to_numpy(dtype=float)
        vec = fn(block, axis=0)
        out.append(vec)
        genes.append(gene)

    G_gxS = pd.DataFrame(np.vstack(out), index=genes, columns=M_fxS.columns)

    report = {
        "M_shape": [int(M_fxS.shape[0]), int(M_fxS.shape[1])],
        "mapping_rows_after_intersect": int(mp.shape[0]),
        "genes_kept": int(G_gxS.shape[0]),
        "samples": int(G_gxS.shape[1]),
        "agg": agg,
        "min_probes_per_gene": int(min_probes_per_gene),
        "top_genes_by_probe_count": gene_counts.head(10).to_dict(),
    }
    return G_gxS, report


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.header("Control Panel")
    if st.button("Clear cache + rerun"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Uploads: {upload_dir}")
    existing = [p for p in sorted(upload_dir.glob("*")) if p.is_file()]
    if existing:
        to_delete = st.multiselect("Delete uploads", options=[p.name for p in existing], default=[])
        if st.button("Delete selected") and to_delete:
            for name in to_delete:
                try:
                    (upload_dir / name).unlink(missing_ok=True)
                except Exception:
                    pass
            st.cache_data.clear()
            st.rerun()

    st.divider()
    st.subheader("Open a previous run")
    run_names = list_runs_cached(str(HARMONIZED_ROOT))
    if run_names:
        run_choice = st.selectbox("Run", options=["(none)"] + run_names, index=0)
        if run_choice != "(none)":
            run_dir = HARMONIZED_ROOT / run_choice
            rp = run_dir / "run_report.json"
            if rp.exists():
                st.json(json.loads(rp.read_text(encoding="utf-8")))
            lp = run_dir / "log.txt"
            if lp.exists():
                st.code("\n".join(lp.read_text(encoding="utf-8").splitlines()[-120:]))


# =============================================================================
# Tabs
# =============================================================================
tab_upload, tab_select, tab_preview, tab_harmonize, tab_train, tab_combine, tab_board = st.tabs(
    ["1) Upload", "2) Select", "3) Preview", "4) Harmonize", "5) Train", "6) Combine", "7) Board exports"]
)


# =============================================================================
# 1) Upload
# =============================================================================
with tab_upload:
    st.header("1) Upload files")
    uploaded_files = st.file_uploader(
        "Upload omics matrix file(s) (CSV/TSV/TXT/Parquet, optional .gz)",
        type=["csv", "tsv", "txt", "parquet", "gz"],
        accept_multiple_files=True,
    )

    st.subheader("Labels (recommended)")
    labels_upload = st.file_uploader("Upload labels (CSV/TSV/TXT)", type=["csv", "tsv", "txt"], accept_multiple_files=False)

    c1, c2 = st.columns([1, 3])
    with c1:
        tmpl = labels_template_df().to_csv(index=False).encode("utf-8")
        st.download_button("Download labels template", data=tmpl, file_name="labels_template.csv", mime="text/csv")
    with c2:
        st.caption("Labels must contain columns like sample_id,label (sample IDs should match your matrices).")

    if uploaded_files:
        saved = []
        for uf in uploaded_files:
            p = save_uploaded_file(uf)
            saved.append(p.name)
        st.success(f"Saved {len(saved)} file(s).")
        st.cache_data.clear()

    if "labels_path" not in st.session_state:
        st.session_state["labels_path"] = None

    if labels_upload is not None:
        lp = save_uploaded_file(labels_upload)
        st.session_state["labels_path"] = str(lp)
        st.success(f"Saved labels: {lp.name}")
        st.cache_data.clear()

    st.divider()
    relpaths = list_uploaded_relpaths_cached(str(upload_dir))
    st.subheader("Uploads folder contents")
    st.write(f"{len(relpaths)} file(s)")
    with st.expander("Show list"):
        st.write(relpaths)


# =============================================================================
# 2) Select
# =============================================================================
with tab_select:
    st.header("2) Select inputs")
    relpaths = list_uploaded_relpaths_cached(str(upload_dir))
    if not relpaths:
        st.info("No uploads found. Go to Upload first.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    with c1:
        methyl_file = st.selectbox("Methylation (M)", options=["(none)"] + relpaths, index=0, key="sb_m")
    with c2:
        rna_file = st.selectbox("Transcriptomics (T)", options=["(none)"] + relpaths, index=0, key="sb_t")
    with c3:
        prot_file = st.selectbox("Proteomics (P)", options=["(none)"] + relpaths, index=0, key="sb_p")

    c4, c5 = st.columns(2)
    with c4:
        metab_file = st.selectbox("Metabolomics (Mb)", options=["(none)"] + relpaths, index=0, key="sb_mb")
    with c5:
        gen_file = st.selectbox("Genomics (G)", options=["(none)"] + relpaths, index=0, key="sb_g")

    st.subheader("Harmonization settings")
    sample_strategy = st.selectbox("Sample handling", ["union", "intersection"], index=0, key="sb_strategy")
    out_format = st.selectbox("Output format", ["parquet", "csv"], index=0, key="sb_outfmt")
    preview_rows = st.number_input("Preview rows (head only)", min_value=5, max_value=200, value=25, step=5, key="ni_preview_rows")

    st.divider()
    st.subheader("Labels status")
    labels_path_str = st.session_state.get("labels_path")
    if labels_path_str:
        try:
            raw_lab = read_labels_cached(labels_path_str)
            ok, msg, cleaned = validate_labels_df(raw_lab)
            if ok and cleaned is not None:
                st.success(f"Labels OK: {Path(labels_path_str).name}")
                st.write(cleaned["label"].value_counts())
                with st.expander("Labels preview"):
                    st.dataframe(cleaned.head(100))
            else:
                st.error(f"Labels invalid: {msg}")
        except Exception as e:
            st.error(f"Failed to read labels: {e}")
    else:
        st.warning("No labels uploaded yet (training will require labels).")


# =============================================================================
# 3) Preview
# =============================================================================
with tab_preview:
    st.header("3) Preview (head only)")
    st.caption("This does not read full files. It’s just to sanity-check formatting.")

    in_paths: Dict[str, Path] = {}
    if st.session_state.get("sb_m") and st.session_state["sb_m"] != "(none)":
        in_paths["M"] = upload_dir / st.session_state["sb_m"]
    if st.session_state.get("sb_t") and st.session_state["sb_t"] != "(none)":
        in_paths["T"] = upload_dir / st.session_state["sb_t"]
    if st.session_state.get("sb_p") and st.session_state["sb_p"] != "(none)":
        in_paths["P"] = upload_dir / st.session_state["sb_p"]
    if st.session_state.get("sb_mb") and st.session_state["sb_mb"] != "(none)":
        in_paths["Mb"] = upload_dir / st.session_state["sb_mb"]
    if st.session_state.get("sb_g") and st.session_state["sb_g"] != "(none)":
        in_paths["G"] = upload_dir / st.session_state["sb_g"]

    preview_rows = int(st.session_state.get("ni_preview_rows", 25))

    if st.button("Run preview"):
        if not in_paths:
            st.warning("Select at least one input in the Select tab.")
        else:
            for k, p in in_paths.items():
                ok, msg = validate_input_file(p)
                st.markdown(f"### {k} — {OMIC_KEYS.get(k, k)}")
                st.write(f"File: `{p.name}` ({human_bytes(p.stat().st_size)})")
                if not ok:
                    st.error(msg)
                    continue
                try:
                    head = read_matrix_head_cached(str(p), nrows=preview_rows)
                    head.columns = canonicalize_sample_ids(k, head.columns)
                    if k == "P":
                        head = collapse_duplicate_columns_numeric(head, how="mean")
                    st.write(f"Head shape: {head.shape}")
                    st.write("Columns head:", head.columns.astype(str).tolist()[:12])
                    st.write("Index head:", head.index.astype(str).tolist()[:12])
                    st.dataframe(head)
                except Exception as e:
                    st.error(f"Preview failed: {e}")


# =============================================================================
# 4) Harmonize (+ Option B methylation gene aggregation)
# =============================================================================
with tab_harmonize:
    st.header("4) Harmonize")
    st.caption("Writes a harmonized matrix per modality into a new run folder.")

    in_paths = {}
    if st.session_state.get("sb_m") and st.session_state["sb_m"] != "(none)":
        in_paths["M"] = upload_dir / st.session_state["sb_m"]
    if st.session_state.get("sb_t") and st.session_state["sb_t"] != "(none)":
        in_paths["T"] = upload_dir / st.session_state["sb_t"]
    if st.session_state.get("sb_p") and st.session_state["sb_p"] != "(none)":
        in_paths["P"] = upload_dir / st.session_state["sb_p"]
    if st.session_state.get("sb_mb") and st.session_state["sb_mb"] != "(none)":
        in_paths["Mb"] = upload_dir / st.session_state["sb_mb"]
    if st.session_state.get("sb_g") and st.session_state["sb_g"] != "(none)":
        in_paths["G"] = upload_dir / st.session_state["sb_g"]

    sample_strategy = st.session_state.get("sb_strategy", "union")
    out_format = st.session_state.get("sb_outfmt", "parquet")

    if not in_paths:
        st.warning("Select at least one input in the Select tab.")
        st.stop()

    st.subheader("Optional (M only): aggregate probes → genes (Option B)")
    st.write("Uses the default mapping unless you upload a custom probe-to-gene file.")

    do_m_gene = st.checkbox("Create gene-level methylation matrix after harmonization", value=False)
    agg_choice = st.selectbox("Aggregation", ["median", "mean"], index=0, disabled=not do_m_gene)
    min_probes = st.number_input("Min probes per gene", min_value=1, max_value=50, value=3, step=1, disabled=not do_m_gene)

    use_custom_map = st.checkbox(
        "Upload custom probe-to-gene mapping (optional)",
        value=False,
        disabled=not do_m_gene,
        help="If unchecked, the default annotations/probe_to_gene.csv is used.",
    )
    uploaded_map = None
    if do_m_gene and use_custom_map:
        uploaded_map = st.file_uploader("Upload probe-to-gene CSV", type=["csv"])

    if st.button("Run harmonization", type="primary"):
        for k, p in in_paths.items():
            ok, msg = validate_input_file(p)
            if not ok:
                st.error(f"{k}: {msg}")
                st.stop()

        run_id = now_run_id()
        run_dir = HARMONIZED_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        log_path = run_dir / "log.txt"
        log_path.write_text(f"[{datetime.now().isoformat()}] Harmonization start: {run_id}\n", encoding="utf-8")

        inputs_snapshot = {k: str(p) for k, p in in_paths.items()}
        (run_dir / "inputs.json").write_text(json.dumps(inputs_snapshot, indent=2), encoding="utf-8")

        try:
            with st.spinner("Running multiharmonize..."):
                import harmonizing_stuff.data_harmonization.multiharmonize as mh  # local import

                res = mh.multiharmonize(
                    in_paths=in_paths,
                    out_dir=run_dir,
                    out_format=out_format,
                    sample_strategy=sample_strategy,
                )

            outputs = {k: str(v) for k, v in res.paths.items()}

            @dataclass
            class RunReport:
                run_id: str
                timestamp: str
                sample_strategy: str
                out_format: str
                inputs: Dict[str, str]
                outputs: Dict[str, str]
                notes: str = ""

            report = RunReport(
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                sample_strategy=sample_strategy,
                out_format=out_format,
                inputs=inputs_snapshot,
                outputs=outputs,
                notes="Harmonized matrices are written by multiharmonize.",
            )
            (run_dir / "run_report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("Outputs:\n" + json.dumps(outputs, indent=2) + "\n")

            st.success("Harmonization complete.")
            st.write("Run:", run_id)
            st.write({k: Path(v).name for k, v in outputs.items()})

            # ---- Option B: M probes -> genes ----
            if do_m_gene and "M" in outputs:
                m_path = Path(outputs["M"])
                if not m_path.exists():
                    st.warning("M harmonized output missing; skipping gene aggregation.")
                else:
                    st.subheader("Methylation gene-level aggregation (Option B)")
                    with st.spinner("Loading M harmonized matrix..."):
                        M_fxS = pd.read_parquet(m_path) if m_path.suffix.lower() == ".parquet" else pd.read_csv(m_path, index_col=0)

                    # mapping selection
                    if uploaded_map is not None:
                        mapping = load_uploaded_probe2gene(uploaded_map)
                        # Save a copy into the run folder for reproducibility (not the repo)
                        saved_map_path = run_dir / "custom_probe_to_gene.csv"
                        mapping.to_csv(saved_map_path, index=False)
                        st.info(f"Using CUSTOM mapping (saved to {saved_map_path.name}).")
                    else:
                        mapping = load_default_probe2gene()
                        st.info("Using DEFAULT mapping (annotations/probe_to_gene.csv).")

                    with st.spinner("Aggregating probes → genes..."):
                        G_gxS, rep = aggregate_methylation_probes_to_genes(
                            M_fxS=M_fxS,
                            mapping=mapping,
                            agg=str(agg_choice),
                            min_probes_per_gene=int(min_probes),
                        )

                    out_gene_path = run_dir / f"M_gene_{agg_choice}_min{int(min_probes)}.{out_format}"
                    if out_format == "parquet":
                        G_gxS.to_parquet(out_gene_path)
                    else:
                        G_gxS.to_csv(out_gene_path)

                    (run_dir / "methylation_gene_agg_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

                    st.success("Saved gene-level methylation matrix.")
                    st.write("Gene matrix:", out_gene_path.name)
                    st.json(rep)

            st.cache_data.clear()

        except Exception as e:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{datetime.now().isoformat()}] ERROR: {repr(e)}\n")
            st.error("Harmonization failed.")
            st.exception(e)


# =============================================================================
# Helpers for later tabs
# =============================================================================
def get_run_outputs(run_dir: Path) -> Dict[str, Path]:
    rp = run_dir / "run_report.json"
    if not rp.exists():
        return {}
    j = json.loads(rp.read_text(encoding="utf-8"))
    outs = j.get("outputs", {}) or {}
    return {k: Path(v) for k, v in outs.items()}


def find_training_script(name: str) -> Optional[Path]:
    candidates = [project_root / "multiomics" / "scripts" / name, project_root / "multiomics" / "training" / name]
    for p in candidates:
        if p.exists():
            return p
    return None


def script_supports_flag(script_path: Path, flag: str) -> bool:
    try:
        return flag in script_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    


# =============================================================================
# 5) Train
# =============================================================================
with tab_train:
    st.header("5) Train unimodal models (OOF)")

    run_names = list_runs_cached(str(HARMONIZED_ROOT))
    if not run_names:
        st.info("No runs available. Harmonize first.")
        st.stop()

    pick = st.selectbox("Choose a harmonization run", options=run_names, index=0, key="sb_train_run")
    run_dir = HARMONIZED_ROOT / pick
    outs = get_run_outputs(run_dir)

    # Prefer gene-level methylation if it exists (Option B output)
    gene_candidates = sorted(run_dir.glob("M_gene_*.parquet")) + sorted(run_dir.glob("M_gene_*.csv"))
    if gene_candidates:
        st.info(f"Detected gene-level methylation matrix: {gene_candidates[0].name} (will use this for M)")
        outs["M"] = gene_candidates[0]

    labels_path_str = st.session_state.get("labels_path")
    if not labels_path_str:
        st.error("Upload labels first (Upload tab).")
        st.stop()

    raw_lab = read_labels_cached(labels_path_str)
    ok, msg, cleaned = validate_labels_df(raw_lab)
    if not ok:
        st.error(f"Labels invalid: {msg}")
        st.stop()
    st.success(f"Labels loaded: {Path(labels_path_str).name}")

    n_splits = st.number_input("CV folds", min_value=3, max_value=10, value=5, step=1)
    seed = st.number_input("Seed", min_value=0, max_value=10000, value=42, step=1)
    pos_label = st.text_input("Positive label", value="EC")
    neg_label = st.text_input("Negative label", value="ART")

    oof_dir = run_dir / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)

    if st.button("Run training", type="primary"):
        scripts = {
            "T": find_training_script("train_transcriptomics.py"),
            "M": find_training_script("train_methylation.py"),
            "P": find_training_script("train_proteomics.py"),
        }

        def _supports_any(sp: Path, flags: list[str]) -> bool:
            try:
                txt = sp.read_text(encoding="utf-8", errors="ignore")
                return any(f in txt for f in flags)
            except Exception:
                return False

        cmds = []
        for k in ["T", "M", "P"]:
            mat = outs.get(k)
            sp = scripts.get(k)
            if mat is None or sp is None:
                continue

            r, c = read_matrix_shape(mat)
            if r == 0 or c == 0:
                st.warning(f"Skipping {k}: empty matrix shape={(r, c)}")
                continue

            cmd = [
                sys.executable,
                str(sp),
            ]

            t_map_path = None
            candidate = project_root / 'transcriptomics_sample_mapping_SAMPLE_to_GSM.csv'
            if candidate.exists():
                t_map_path = candidate
            
            # Matrix flag (most of your scripts now support this)
            if _supports_any(sp, ["--matrix"]):
                cmd += ["--matrix", str(mat)]

            # Out dir flag
            if _supports_any(sp, ["--out_dir"]):
                cmd += ["--out_dir", str(oof_dir)]

            # Labels flag (your T script requires it; others may ignore)
            if _supports_any(sp, ["--labels"]):
                cmd += ["--labels", str(Path(labels_path_str))]

            # Transcriptomics-only: sample_map required
            if k == "T" and _supports_any(sp, ["--sample_map"]):
                cmd += ["--sample_map", str(t_map_path)]

            # Optional CV + seed flags
            # Optional CV + seed flags
            if _supports_any(sp, ["--n_splits_max"]):
                cmd += ["--n_splits_max", str(int(n_splits))]
            elif _supports_any(sp, ["--n_splits"]):
                cmd += ["--n_splits", str(int(n_splits))]
            if _supports_any(sp, ["--seed"]):
                cmd += ["--seed", str(int(seed))]

            # Optional EC/ART flags (only if the script uses them)
            if _supports_any(sp, ["--pos_label"]):
                cmd += ["--pos_label", pos_label]
            if _supports_any(sp, ["--neg_label"]):
                cmd += ["--neg_label", neg_label]

            cmds.append((k, cmd))

        if not cmds:
            st.error("No training commands built (missing scripts or matrices).")
            st.stop()

        with st.spinner("Training..."):
            for k, cmd in cmds:
                p = subprocess.run(cmd, capture_output=True, text=True)
                if p.returncode != 0:
                    st.error(f"Training failed for {k}:")
                    st.code(" ".join(cmd))
                    if p.stdout:
                        st.code(p.stdout[-8000:])
                    if p.stderr:
                        st.code(p.stderr[-8000:])
                    st.stop()

        st.success("Training complete.")
        st.write("OOF files:", [p.name for p in sorted(oof_dir.glob("*.csv"))])
    
    if IS_CLOUD:
        st.warning('Training is disabled on Streamlit Cloud')
        st.stop()

# =============================================================================
# 6) Combine
# =============================================================================
with tab_combine:
    st.header("6) Combine results (weights)")

    run_names = list_runs_cached(str(HARMONIZED_ROOT))
    if not run_names:
        st.info("No runs available.")
        st.stop()

    pick = st.selectbox("Choose a run", options=run_names, index=0, key="sb_combine_run")
    run_dir = HARMONIZED_ROOT / pick
    oof_dir = run_dir / "oof"
    combine_dir = run_dir / "combine"
    fig_dir = run_dir / "figures"
    combine_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- helpers must be defined BEFORE use ----
    def pick_first_existing(paths: List[Path]) -> Optional[Path]:
        for p in paths:
            if p.exists():
                return p
        return None

    def find_oof_file(oof_dir: Path, preferred: List[str], patterns: List[str]) -> Optional[Path]:
        exact = pick_first_existing([oof_dir / name for name in preferred])
        if exact is not None:
            return exact
        for pat in patterns:
            hits = sorted(oof_dir.glob(pat))
            hits = sorted(hits, key=lambda x: (len(x.name), x.name))
            if hits:
                return hits[0]
        return None

    oof_T = find_oof_file(
        oof_dir,
        preferred=["oof_transcriptomics.csv"],
        patterns=["*transcript*oof*.csv", "oof*T*.csv", "*T*oof*.csv"],
    )
    oof_M = find_oof_file(
        oof_dir,
        preferred=["oof_methylation.csv"],
        patterns=["*methyl*oof*.csv", "oof*M*.csv", "*M*oof*.csv"],
    )
    oof_P = find_oof_file(
        oof_dir,
        preferred=["proteomics_subject_mean_oof.csv"],
        patterns=["*prote*oof*.csv", "*P*oof*.csv"],
    )

    st.write(
        {
            "T": str(oof_T) if oof_T else None,
            "M": str(oof_M) if oof_M else None,
            "P": str(oof_P) if oof_P else None,
        }
    )

    available = {k: f for k, f in [("T", oof_T), ("M", oof_M), ("P", oof_P)] if f is not None and f.exists()}
    missing = [k for k in ["T", "M", "P"] if k not in available]

    if missing:
        st.info(f"Proceeding without: {missing}")

    if len(available) < 1:
        st.error("No OOF files found. Add at least one modality OOF in the run/oof folder.")
        st.write("Found in oof/:", [p.name for p in sorted(oof_dir.glob('*.csv'))])
        st.stop()

    from sklearn.metrics import roc_auc_score  # noqa: E402

    def read_oof(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        proba_col = cols.get("proba") or cols.get("prob") or cols.get("p") or cols.get("yhat")
        y_col = cols.get("y") or cols.get("label") or cols.get("target")
        if not (proba_col and y_col):
            raise ValueError(f"OOF must contain proba + y columns (found {df.columns.tolist()})")
        out = pd.DataFrame(
            {
                "proba": pd.to_numeric(df[proba_col], errors="coerce"),
                "y": pd.to_numeric(df[y_col], errors="coerce"),
            }
        ).dropna()
        out["y"] = out["y"].astype(int)
        return out

    def auc_of(df: pd.DataFrame) -> float:
        y = df["y"].to_numpy()
        p = df["proba"].to_numpy()
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))

    def normalize_weights(aucs: Dict[str, float]) -> Dict[str, float]:
        vals = {k: (v if np.isfinite(v) and v > 0 else 0.0) for k, v in aucs.items()}
        s = float(sum(vals.values()))
        if s <= 0:
            n = len(vals)
            return {k: 1.0 / n for k in vals}
        return {k: float(v / s) for k, v in vals.items()}

    if st.button("Compute weights", type="primary"):
        name_map = {"T": "Transcriptomics", "M": "Methylation", "P": "Proteomics"}

        aucs = {}
        for k, path in available.items():
            aucs[name_map[k]] = auc_of(read_oof(path))

        weights = normalize_weights(aucs)

        out_json = combine_dir / "combined_weights.json"
        out_json.write_text(json.dumps({"aucs": aucs, "weights": weights}, indent=2), encoding="utf-8")

        st.success("Saved combined_weights.json")
        st.json({"aucs": aucs, "weights": weights})

        import matplotlib.pyplot as plt  # noqa: E402

        fig = plt.figure()
        plt.bar(list(aucs.keys()), list(aucs.values()))
        plt.ylim(0, 1)
        plt.ylabel("OOF AUC")
        plt.title("Model performance by modality")
        fig_path = fig_dir / "modality_performance_auc.png"
        fig.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        st.image(str(fig_path), caption="OOF AUC by modality")
        
# =============================================================================
# 7) Board exports (simple zip)
# =============================================================================
with tab_board:
    st.header("7) Board exports (ZIP)")

    run_names = list_runs_cached(str(HARMONIZED_ROOT))
    if not run_names:
        st.info("No runs available.")
        st.stop()

    pick = st.selectbox("Choose a run", options=run_names, index=0, key="sb_board_run")
    run_dir = HARMONIZED_ROOT / pick

    pack_dir = run_dir / "board_pack"
    pack_dir.mkdir(parents=True, exist_ok=True)

    include = []
    for rel in [
        "run_report.json",
        "methylation_gene_agg_report.json",
        "inputs.json",
        "combine/combined_weights.json",
        "figures/modality_performance_auc.png",
    ]:
        p = run_dir / rel
        if p.exists():
            include.append(p)

    # add all pngs/csvs that exist in common folders
    for folder in ["figures", "oof", "combine"]:
        fdir = run_dir / folder
        if fdir.exists():
            include += [p for p in fdir.glob("*.png")]
            include += [p for p in fdir.glob("*.csv")]
            include += [p for p in fdir.glob("*.json")]

    out_zip = pack_dir / "board_pack.zip"

    if st.button("Build ZIP", type="primary"):
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(set(include)):
                if p.exists() and p.is_file():
                    z.write(p, arcname=str(p.relative_to(run_dir)))
        st.success("Board ZIP created.")
        st.download_button("Download board_pack.zip", data=out_zip.read_bytes(), file_name="board_pack.zip", mime="application/zip")
        with st.expander("Included files"):
            st.write([str(p.relative_to(run_dir)) for p in sorted(set(include))])
