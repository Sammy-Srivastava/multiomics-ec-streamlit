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

import numpy as np
import pandas as pd
import streamlit as st



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

ALLOWED_UPLOAD_EXT = {".csv", ".tsv", ".txt", ".parquet", ".gz", ".zip"}
BAD_EXT = (".raw", ".tar", ".mzml", ".wiff", ".d", ".cdf")

OMIC_KEYS = {
    "M": "Methylation",
    "T": "Transcriptomic",
    "P": "Proteomic",
    "Mb": "Metabolomic",
    "G": "Genomic",
}

_CANON_WS = re.compile(r"\s+")


# 1. Page config

st.set_page_config(page_title="Omics Harmonizer", layout="wide")
st.title("Omics Harmonizer Dashboard")
st.caption("Harmonize + train + Option A weights. Optimized for speed and reliability.")


# 2. Small utilities

def safe_filename(name: str) -> str:
    """Sanitize user-provided filenames for local storage."""
    name = name.replace("\\", "_").replace("/", "_")
    return "".join(c for c in name if c.isalnum() or c in "._-")


def now_run_id() -> str:
    """Unique run directory label."""
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def human_bytes(n: int) -> str:
    """print byte sizes"""
    if n is None:
        return "?"
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:,.1f} {u}"
        x /= 1024.0
    return f"{x:,.1f} PB"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)


def is_within_directory(directory: Path, target: Path) -> bool:
    """Prevent zip slip."""
    try:
        directory = directory.resolve()
        target = target.resolve()
        return str(target).startswith(str(directory))
    except Exception:
        return False

# 3. Upload helpers

def save_uploaded_file(uploaded) -> Path:
    """Uploadfile using chunk downloading"""
    fname = safe_filename(uploaded.name)
    out_path = upload_dir / fname
    CHUNK = 16 * 1024 * 1024  # 16MB
    uploaded.seek(0)
    with open(out_path, "wb") as f:
        while True:
            chunk = uploaded.read(CHUNK)
            if not chunk:
                break
            f.write(chunk)
    return out_path


def safe_extract_zip(zip_path: Path, out_dir: Path, max_members: int = 2000) -> List[Path]:
    """Extract ZIP safely"""
    extracted: List[Path] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        members = z.infolist()
        if len(members) > max_members:
            raise ValueError(f"ZIP has {len(members)} members; exceeds limit {max_members}.")

        for info in members:
            if info.is_dir():
                continue

            raw_name = info.filename
            if raw_name.endswith("/"):
                continue

            dest = out_dir / safe_filename(Path(raw_name).name)
            if not is_within_directory(out_dir, dest):
                raise ValueError(f"Blocked ZIP path traversal attempt: {raw_name}")

            with z.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

            extracted.append(dest)

    return extracted


@st.cache_data(show_spinner=False)
def list_uploaded_relpaths_cached(upload_dir_str: str) -> List[str]:
    """Cached list of all uploaded files"""
    ud = Path(upload_dir_str)
    subfiles = [p for p in ud.rglob("*") if p.is_file()]
    return sorted({str(p.relative_to(ud)) for p in subfiles})


@st.cache_data(show_spinner=False)
def list_runs_cached(root_str: str) -> List[str]:
    """Cached list of harmonization run directories"""
    rr = Path(root_str)
    runs = [p for p in rr.glob("*") if p.is_dir()]
    runs = sorted(runs, key=lambda p: p.name, reverse=True)
    return [p.name for p in runs]


def validate_input_file(p: Path) -> Tuple[bool, str]:
    """sanity checks"""
    p = Path(p)
    if not p.exists():
        return False, "File does not exist."
    if p.stat().st_size == 0:
        return False, "File is empty."
    name_lower = p.name.lower()
    if name_lower.endswith(BAD_EXT):
        return False, f"Unsupported binary/raw file type: {p.suffix}"
    if (p.suffix.lower() not in ALLOWED_UPLOAD_EXT) and (
        not name_lower.endswith((".tsv.gz", ".txt.gz", ".csv.gz"))
    ):
        return False, f"Unsupported extension: {p.suffix}"
    return True, "OK"


# 4. Sample ID canonicalization and sample map

def canonicalize_sample_ids_basic(cols: pd.Index) -> pd.Index:
    return cols.astype(str).str.strip().str.replace(_CANON_WS, "_", regex=True)


def canonicalize_sample_ids(omic_key: str, cols: pd.Index) -> pd.Index:
    """Canonicalize sample IDs for preview."""
    cols = cols.astype(str)

    if omic_key == "P":
        out = []
        for c in cols:
            m = re.search(r",_Sample,_(.+?),_", c)
            if m:
                out.append(m.group(1).strip())
                continue
            m2 = re.search(r"\b(EC\d+|ART\d+|HC\d+)\b", c, flags=re.I)
            out.append(m2.group(1).upper() if m2 else c.strip())
        return pd.Index(out)

    return canonicalize_sample_ids_basic(pd.Index(cols))


def load_sample_map_csv(path: Optional[Path]) -> Optional[dict]:
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    cols = {str(c).strip().lower(): c for c in df.columns}
    if "old" not in cols or "new" not in cols:
        return None

    mp = {}
    for o, n in zip(df[cols["old"]].astype(str), df[cols["new"]].astype(str)):
        mp[o.strip()] = n.strip()
    return mp


def apply_sample_map(cols: pd.Index, sample_map: Optional[dict]) -> pd.Index:
    if not sample_map:
        return cols
    return pd.Index([sample_map.get(str(c), str(c)) for c in cols.astype(str)])

# 5) Head-only matrix reading + diagnostics

@st.cache_data(show_spinner=False)
def read_matrix_head_cached(path_str: str, nrows: int = 25) -> pd.DataFrame:
    """HEAD-only preview reader."""
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
        df = pd.read_csv(p, sep=sep, index_col=0, compression=compression, nrows=nrows, engine="c")
    except Exception:
        df = pd.read_csv(
            p,
            sep=sep,
            index_col=0,
            compression="infer",
            nrows=nrows,
            engine="python",
            encoding="latin1",
            on_bad_lines="skip",
        )

    df.columns = df.columns.astype(str).str.strip()
    df.index = df.index.astype(str).str.strip()
    return df


def matrix_quick_diagnostics_from_head(df_head: pd.DataFrame) -> dict:
    """Heuristic stats from a small preview"""
    if df_head is None:
        return {"ok": False, "reason": "df is None"}

    n_feat, n_samp = df_head.shape

    sample = df_head.iloc[: min(50, n_feat), : min(20, n_samp)].copy()
    coerced = sample.apply(pd.to_numeric, errors="coerce")
    total = coerced.size
    arr = coerced.to_numpy(dtype=float, na_value=np.nan)

    non_na = int(np.isfinite(arr).sum())
    na_rate = float(pd.isna(sample).mean().mean()) if total > 0 else 0.0

    col_count = df_head.shape[1]
    idx_is_sample_like = df_head.index.to_series().astype(str).str.match(r"^[A-Za-z]*\d+.*", na=False).mean()
    maybe_transposed = (col_count < 4 and n_feat > 100) or (
        idx_is_sample_like > 0.8 and n_feat < 5000 and n_samp > 5000
    )

    return {
        "ok": True,
        "shape_head_only": [int(n_feat), int(n_samp)],
        "na_rate_preview": float(na_rate),
        "numeric_rate_preview": float(non_na / total) if total > 0 else 0.0,
        "maybe_transposed": bool(maybe_transposed),
        "index_head": df_head.index.astype(str).tolist()[:10],
        "columns_head": df_head.columns.astype(str).tolist()[:10],
    }

# 6) Sanity helpers for methylation empty-output debugging

DET_P_PATTERNS = [
    r"detection\s*p",
    r"detection_p",
    r"det\.?\s*p",
    r"\bdetp\b",
    r"\bpval\b",
    r"p-value",
    r"\bpvalue\b",
]
METH_SIG_PATTERNS = [
    r"unmethylated",
    r"methylated",
    r"\bunmeth\b",
    r"\bmeth\b",
    r"\bsignal\b",
    r"_u\b",
    r"_m\b",
]


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


def methylation_forensics(raw_m_path: Optional[Path], out_m_path: Optional[Path], preview_rows: int = 25) -> dict:
    report = {
        "raw_path": str(raw_m_path) if raw_m_path else None,
        "out_path": str(out_m_path) if out_m_path else None,
        "raw_exists": bool(raw_m_path and raw_m_path.exists()),
        "out_exists": bool(out_m_path and out_m_path.exists()),
        "out_shape": [0, 0],
        "raw_head_shape": None,
        "raw_columns_head": None,
        "detp_column_hits_n": None,
        "detp_column_hits_examples": None,
        "signal_column_hits_n": None,
        "signal_column_hits_examples": None,
        "numeric_rate_preview": None,
        "notes": [],
    }

    if out_m_path and out_m_path.exists():
        r, c = read_matrix_shape(out_m_path)
        report["out_shape"] = [r, c]

    if raw_m_path and raw_m_path.exists():
        try:
            head = read_matrix_head_cached(str(raw_m_path), nrows=int(preview_rows))
            cols = [str(x) for x in head.columns.tolist()]

            detp_hits: List[str] = []
            for pat in DET_P_PATTERNS:
                rx = re.compile(pat, flags=re.I)
                detp_hits.extend([c for c in cols if rx.search(c)])

            sig_hits: List[str] = []
            for pat in METH_SIG_PATTERNS:
                rx = re.compile(pat, flags=re.I)
                sig_hits.extend([c for c in cols if rx.search(c)])

            block = head.iloc[: min(50, head.shape[0]), : min(20, head.shape[1])].copy()
            coerced = block.apply(pd.to_numeric, errors="coerce")
            total = coerced.size
            non_na = int(np.isfinite(coerced.to_numpy(dtype=float, na_value=np.nan)).sum())
            numeric_rate = float(non_na / total) if total else 0.0

            report["raw_head_shape"] = [int(head.shape[0]), int(head.shape[1])]
            report["raw_columns_head"] = cols[:40]
            report["detp_column_hits_n"] = int(len(set(detp_hits)))
            report["detp_column_hits_examples"] = sorted(set(detp_hits))[:25]
            report["signal_column_hits_n"] = int(len(set(sig_hits)))
            report["signal_column_hits_examples"] = sorted(set(sig_hits))[:25]
            report["numeric_rate_preview"] = float(numeric_rate)

            if report["detp_column_hits_n"] == 0:
                report["notes"].append(
                    "No obvious detection p-value columns found in RAW head. "
                    "If the methylation normalizer expects detP, filtering/parsing may fail."
                )
            if report["signal_column_hits_n"] == 0:
                report["notes"].append(
                    "No obvious methylated/unmethylated signal columns found in RAW head. "
                    "If the normalizer expects intensity pairs, parsing may fail."
                )
            if numeric_rate < 0.5:
                report["notes"].append(
                    "Low numeric rate in RAW preview block. Many values may be strings -> NaNs -> dropped by filters."
                )
        except Exception as e:
            report["notes"].append(f"RAW head forensics failed: {repr(e)}")

    return report


def require_nonempty_outputs(outputs: Dict[str, str]) -> None:
    """stop if any output is missing or empty."""
    bad = []
    shapes = {}

    for k, outp in outputs.items():
        p = Path(outp)
        if not p.exists():
            bad.append((k, "missing output file", outp))
            continue
        r, c = read_matrix_shape(p)
        shapes[k] = (r, c)
        if r == 0 or c == 0:
            bad.append((k, f"empty matrix shape={(r, c)}", outp))

    if bad:
        st.error("Harmonization produced invalid output(s).")
        st.write("Output shapes:", shapes)
        st.write("Problems:", bad)
        st.stop()


def get_run_outputs(run_dir: Path) -> Dict[str, Path]:
    """Read outputs mapping from run_report.json."""
    rp = run_dir / "run_report.json"
    if not rp.exists():
        return {}
    j = json.loads(rp.read_text(encoding="utf-8"))
    outs = j.get("outputs", {}) or {}

    out_paths: Dict[str, Path] = {}
    for k, v in outs.items():
        try:
            out_paths[k] = Path(v)
        except Exception:
            pass
    return out_paths


# 7) Sidebar: uploads & prior runs

with st.sidebar:
    st.header("Uploads / Runs")

    if st.button("Refresh file lists", key="btn_refresh_lists"):
        st.cache_data.clear()
        st.rerun()

    st.caption(f"Uploads dir: {upload_dir}")

    existing_top = [p for p in sorted(upload_dir.glob("*")) if p.is_file()]
    if existing_top:
        to_delete = st.multiselect(
            "Select uploads to delete",
            options=[p.name for p in existing_top],
            default=[],
            key="ms_delete_uploads",
        )
        if st.button("Delete selected uploads", type="secondary", key="btn_delete_uploads") and to_delete:
            for name in to_delete:
                try:
                    (upload_dir / name).unlink(missing_ok=True)
                except Exception:
                    pass
            st.cache_data.clear()
            st.rerun()

    st.divider()
    st.subheader("Prior runs")

    run_names = list_runs_cached(str(HARMONIZED_ROOT))
    if not run_names:
        st.info("No runs yet.")
    else:
        run_choice = st.selectbox("Open run", options=["(none)"] + run_names, index=0, key="sb_open_run")
        if run_choice != "(none)":
            run_dir = HARMONIZED_ROOT / run_choice
            report_path = run_dir / "run_report.json"
            log_path = run_dir / "log.txt"
            forensics_path = run_dir / "methylation_forensics.json"

            if report_path.exists():
                st.write("run_report.json:")
                st.json(json.loads(report_path.read_text(encoding="utf-8")))
            else:
                st.warning("No run_report.json found.")

            if forensics_path.exists():
                st.write("methylation_forensics.json:")
                st.json(json.loads(forensics_path.read_text(encoding="utf-8")))

            if log_path.exists():
                st.write("log.txt (tail):")
                txt = log_path.read_text(encoding="utf-8")
                st.code("\n".join(txt.splitlines()[-120:]))
            else:
                st.info("No log.txt found.")


# =============================================================================
# 8) Stage 1: Upload
# =============================================================================
st.header("1) Upload files")
st.write("Upload matrices (CSV/TSV/TXT/Parquet, optionally .gz). ZIP uploads are supported (optional extraction).")

uploaded_files = st.file_uploader(
    "Upload dataset file(s)",
    type=["csv", "tsv", "txt", "parquet", "gz", "zip"],
    accept_multiple_files=True,
    key="uploader_main",
)

zip_extract = st.checkbox("If ZIP uploaded: extract contents into uploads/", value=False, key="cb_zip_extract")
zip_extract_limit = st.number_input(
    "ZIP max files to extract", min_value=50, max_value=5000, value=2000, step=50, key="ni_zip_limit"
)

if uploaded_files:
    saved_names = []
    extracted_names = []

    for uf in uploaded_files:
        p = save_uploaded_file(uf)
        saved_names.append(p.name)

        if p.suffix.lower() == ".zip" and zip_extract:
            try:
                z_out = upload_dir / (p.stem + "_extracted")
                extracted = safe_extract_zip(p, z_out, max_members=int(zip_extract_limit))
                extracted_names.extend([str(ep.relative_to(upload_dir)) for ep in extracted])
            except Exception as e:
                st.error(f"ZIP extraction failed for {p.name}: {e}")

    st.success(f"Saved {len(saved_names)} file(s).")
    if extracted_names:
        st.info(f"Extracted {len(extracted_names)} file(s) from ZIP(s).")
        with st.expander("Extracted file list"):
            st.write(extracted_names)

    st.cache_data.clear()

st.divider()


# =============================================================================
# 9) Stage 2: Select inputs
# =============================================================================
st.header("2) Select inputs for harmonization")

relpaths = list_uploaded_relpaths_cached(str(upload_dir))
if not relpaths:
    st.info("No uploads found. Upload files above.")
    st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    methyl_file = st.selectbox("Methylation (M)", options=["(none)"] + relpaths, index=0, key="sb_m")
with c2:
    rna_file = st.selectbox("Transcriptomic (T)", options=["(none)"] + relpaths, index=0, key="sb_t")
with c3:
    prot_file = st.selectbox("Proteomic (P)", options=["(none)"] + relpaths, index=0, key="sb_p")

c4, c5, _ = st.columns(3)
with c4:
    metab_file = st.selectbox("Metabolomic (Mb)", options=["(none)"] + relpaths, index=0, key="sb_mb")
with c5:
    gen_file = st.selectbox("Genomic (G)", options=["(none)"] + relpaths, index=0, key="sb_g")

st.subheader("Harmonization settings (speed-first defaults)")
sample_strategy = st.selectbox("Sample strategy", ["union", "intersection"], index=0, key="sb_strategy")
out_format = st.selectbox("Output format", ["parquet", "csv"], index=0, key="sb_outfmt")

st.caption(
    "Performance guidance: parquet + union is fastest. "
    "Intersection is only appropriate if you expect overlapping samples."
)

st.subheader("Optional: Sample mapping for ID alignment")
st.write("Upload a CSV with columns: old,new (case-insensitive). Example: SAMPLE1 -> GSM1301871.")
sample_map_file = st.selectbox(
    "Sample map CSV",
    options=["(none)"] + [p for p in relpaths if p.lower().endswith(".csv")],
    index=0,
    key="sb_samplemap",
)

preview_rows = st.number_input(
    "Preview rows (head only)", min_value=5, max_value=200, value=25, step=5, key="ni_preview_rows"
)

st.divider()


def resolve_selected_inputs() -> Dict[str, Path]:
    """Convert dropdown selections into absolute file paths."""
    in_paths: Dict[str, Path] = {}
    if methyl_file != "(none)":
        in_paths["M"] = upload_dir / methyl_file
    if rna_file != "(none)":
        in_paths["T"] = upload_dir / rna_file
    if prot_file != "(none)":
        in_paths["P"] = upload_dir / prot_file
    if metab_file != "(none)":
        in_paths["Mb"] = upload_dir / metab_file
    if gen_file != "(none)":
        in_paths["G"] = upload_dir / gen_file
    return in_paths


in_paths = resolve_selected_inputs()


# =============================================================================
# 10) Stage 3: Preview / diagnostics (HEAD ONLY)
# =============================================================================
st.header("3) Validate & preview (optional)")
if st.button("Run preview/diagnostics (head only)", type="secondary", key="btn_preview"):
    if not in_paths:
        st.warning("Select at least one input omic above.")
    else:
        sample_map_path = None if sample_map_file == "(none)" else (upload_dir / sample_map_file)
        sample_map = load_sample_map_csv(sample_map_path)

        cols = st.columns(min(3, len(in_paths)))
        for i, (k, p) in enumerate(in_paths.items()):
            block = cols[i % len(cols)]
            with block:
                st.markdown(f"**{k} — {OMIC_KEYS.get(k, k)}**")
                ok, msg = validate_input_file(p)
                st.write(f"File: `{p.name}`")
                st.write(f"Size: {human_bytes(p.stat().st_size)}")
                if not ok:
                    st.error(msg)
                    continue

                try:
                    head = read_matrix_head_cached(str(p), nrows=int(preview_rows))
                    head.columns = canonicalize_sample_ids(k, head.columns)
                    head.columns = apply_sample_map(head.columns, sample_map)

                    d = matrix_quick_diagnostics_from_head(head)
                    if d.get("maybe_transposed"):
                        st.warning("May be transposed (heuristic). Verify features×samples orientation.")
                    st.write(f"Head shape: {tuple(d['shape_head_only'])}")
                    st.write(f"Preview NA rate: {d['na_rate_preview']:.3f}")
                    st.write(f"Preview numeric rate: {d['numeric_rate_preview']:.3f}")
                    st.write("Sample IDs head:", d["columns_head"])
                    with st.expander("Preview table (head)"):
                        st.dataframe(head)
                except Exception as e:
                    st.error(f"Preview failed: {e}")
else:
    st.info("Preview is skipped by default to keep the app fast.")

st.divider()


# =============================================================================
# 11) Stage 4: Run harmonization
# =============================================================================
st.header("4) Run harmonization")
run_btn = st.button("Run Harmonization", type="primary", key="btn_run_harmonize")

if run_btn:
    if not in_paths:
        st.error("Select at least one input omic.")
        st.stop()

    for k, p in in_paths.items():
        ok, msg = validate_input_file(p)
        if not ok:
            st.error(f"{k}: {msg}")
            st.stop()

    sample_map_path = None if sample_map_file == "(none)" else (upload_dir / sample_map_file)
    sample_map = load_sample_map_csv(sample_map_path)

    run_id = now_run_id()
    out_dir = HARMONIZED_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "log.txt"
    write_text(log_path, f"[{datetime.now().isoformat()}] Starting harmonization run {run_id}\n")

    inputs_snapshot = {k: str(p) for k, p in in_paths.items()}
    write_text(out_dir / "inputs.json", json.dumps(inputs_snapshot, indent=2))

    try:
        with st.spinner("Running harmonization..."):
            append_text(log_path, f"sample_strategy={sample_strategy}, out_format={out_format}\n")
            append_text(log_path, "Inputs:\n" + json.dumps(inputs_snapshot, indent=2) + "\n")

            # Lazy import so Streamlit loads faster
            import harmonizing_stuff.data_harmonization.multiharmonize as mh

            # NOTE: fast_mode removed; rely on multiharmonize defaults/implementation.
            res = mh.multiharmonize(
                in_paths=in_paths,
                out_dir=out_dir,
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
            notes=(
                "UI does not rewrite outputs after harmonization (speed). "
                "If sample ID canonicalization / sample_map must be applied to outputs, do it inside multiharmonize."
            ),
        )
        (out_dir / "run_report.json").write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")

        append_text(log_path, f"[{datetime.now().isoformat()}] Harmonization complete.\n")
        append_text(log_path, "Outputs:\n" + json.dumps(outputs, indent=2) + "\n")

        st.success("Harmonization complete.")
        st.write("Run directory:", str(out_dir))

        st.subheader("Output files")
        for k, outp in outputs.items():
            p = Path(outp)
            rel = p.relative_to(project_root) if project_root in p.parents else p
            st.write(f"- **{k}**: `{rel}` ({human_bytes(p.stat().st_size) if p.exists() else 'missing'})")

        st.subheader("Sanity checks (fail-fast)")
        shapes = {k: read_matrix_shape(Path(outp)) for k, outp in outputs.items()}
        st.write("Output shapes:", shapes)

        # Auto methylation forensics if empty
        if "M" in outputs:
            mr, mc = shapes.get("M", (0, 0))
            if mr == 0 or mc == 0:
                st.error("Methylation harmonized output is empty. Running methylation forensics...")
                raw_m = inputs_snapshot.get("M")
                raw_m_path = Path(raw_m) if raw_m else None
                m_out = Path(outputs["M"])
                rep = methylation_forensics(raw_m_path, m_out, preview_rows=int(preview_rows))
                (out_dir / "methylation_forensics.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
                st.json(rep)
                st.stop()

        require_nonempty_outputs(outputs)

        # Quick overlap preview from HEAD ONLY
        overlap_preview = {}
        for k, outp in outputs.items():
            try:
                head = read_matrix_head_cached(outp, nrows=1)
                cols = canonicalize_sample_ids(k, head.columns)
                cols = apply_sample_map(cols, sample_map)
                overlap_preview[k] = list(cols[:20])
            except Exception:
                overlap_preview[k] = ["(preview failed)"]

        st.subheader("Sample ID preview (canonicalized + sample_map applied in preview only)")
        st.json(overlap_preview)

        with st.expander("Preview harmonized outputs (HEAD ONLY)"):
            for k, outp in outputs.items():
                p = Path(outp)
                if not p.exists():
                    st.warning(f"{k}: missing output at {p}")
                    continue
                try:
                    head = read_matrix_head_cached(str(p), nrows=int(preview_rows))
                    st.markdown(f"**{k}** — head shape={head.shape}")
                    st.dataframe(head)
                except Exception as e:
                    st.warning(f"{k}: head preview failed: {e}")

        st.cache_data.clear()

    except Exception as e:
        append_text(log_path, f"[{datetime.now().isoformat()}] ERROR: {repr(e)}\n")
        st.error("Harmonization failed.")
        st.exception(e)

st.divider()


# =============================================================================
# 12) Stage 5: Browse runs (main page)
# =============================================================================
st.header("5) Browse recent runs")

run_names = list_runs_cached(str(HARMONIZED_ROOT))
if not run_names:
    st.info("No runs found yet.")
else:
    pick = st.selectbox("Select a run to view", options=run_names, index=0, key="sb_run_pick_main")
    rd = HARMONIZED_ROOT / pick
    rp = rd / "run_report.json"
    lp = rd / "log.txt"
    fp = rd / "methylation_forensics.json"

    c1, c2 = st.columns(2)
    with c1:
        st.write("Run report")
        if rp.exists():
            st.json(json.loads(rp.read_text(encoding="utf-8")))
        else:
            st.warning("run_report.json not found.")
        if fp.exists():
            st.write("Methylation forensics")
            st.json(json.loads(fp.read_text(encoding="utf-8")))

    with c2:
        st.write("Log (tail)")
        if lp.exists():
            txt = lp.read_text(encoding="utf-8")
            st.code("\n".join(txt.splitlines()[-160:]))
        else:
            st.info("log.txt not found.")

st.divider()


# =============================================================================
# 13) Stage 6: Train unimodal models (OOF)
# =============================================================================
st.header("6) Train unimodal models (generate OOF predictions)")

run_names = list_runs_cached(str(HARMONIZED_ROOT))
if not run_names:
    st.info("No runs available. Run harmonization first.")
    st.stop()

train_pick = st.selectbox("Select run to train on", options=run_names, index=0, key="sb_train_run")
train_dir = HARMONIZED_ROOT / train_pick
outs = get_run_outputs(train_dir)

t_h = outs.get("T")
m_h = outs.get("M")
p_h = outs.get("P")

st.write("Detected harmonized outputs:")
st.write({"T": str(t_h) if t_h else None, "M": str(m_h) if m_h else None, "P": str(p_h) if p_h else None})

st.subheader("Training settings")
n_splits = st.number_input("CV folds (n_splits)", min_value=3, max_value=10, value=5, step=1, key="ni_cv")
seed = st.number_input("Random seed", min_value=0, max_value=10_000, value=42, step=1, key="ni_seed")
pos_label = st.text_input("Positive label (for AUC)", value="EC", key="ti_poslabel")
neg_label = st.text_input("Negative label (for AUC)", value="ART", key="ti_neglabel")

oof_dir = train_dir / "oof"
oof_dir.mkdir(parents=True, exist_ok=True)

train_btn = st.button("Run unimodal training (write OOF CSVs)", type="primary", key="btn_train_unimodal")


def _tail(s: str, n: int = 4000) -> str:
    return s[-n:] if s else ""


def _copy_if_exists(src: Path, dst: Path) -> bool:
    try:
        if src.exists() and src.is_file() and src.stat().st_size > 0:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            return True
    except Exception:
        return False
    return False


def _collect_oof_files_to_run_oof_dir(project_root: Path, oof_dir: Path) -> dict:
    """
    Training scripts sometimes dump outputs into project root.
    This collects/copies canonical outputs into run/oof/ with expected filenames.
    """
    canon = {
        "T": "oof_transcriptomics.csv",
        "M": "oof_methylation.csv",
        "P": "proteomics_subject_mean_oof.csv",
    }

    candidates = {
        "T": [oof_dir / "oof_transcriptomics.csv", project_root / "oof_transcriptomics.csv"],
        "M": [oof_dir / "oof_methylation.csv", project_root / "oof_methylation.csv"],
        "P": [
            oof_dir / "proteomics_subject_mean_oof.csv",
            project_root / "proteomics_subject_mean_oof.csv",
            oof_dir / "proteomic_subject_mean_oof.csv",
            project_root / "proteomic_subject_mean_oof.csv",
        ],
    }

    copied = {}
    for k, cand_list in candidates.items():
        dst = oof_dir / canon[k]
        ok = False
        for src in cand_list:
            if _copy_if_exists(src, dst):
                ok = True
                copied[k] = {"from": str(src), "to": str(dst)}
                break
        if not ok:
            copied[k] = {"from": None, "to": str(dst)}

    return copied


if train_btn:
    # Skip empty harmonized matrices (prevents training crashes)
    valid: Dict[str, Path] = {}
    for k, mat in [("T", t_h), ("M", m_h), ("P", p_h)]:
        if not mat:
            continue
        r, c = read_matrix_shape(Path(mat))
        if r == 0 or c == 0:
            st.warning(f"Skipping {k}: harmonized matrix is empty shape={(r, c)}")
            continue
        valid[k] = Path(mat)

    if not valid:
        st.error("No non-empty harmonized matrices available for training.")
        st.stop()

    scripts = {
        "T": project_root / "multiomics" / "scripts" / "train_transcriptomics.py",
        "M": project_root / "multiomics" / "scripts" / "train_methylation.py",
        "P": project_root / "multiomics" / "scripts" / "train_proteomics.py",
    }

    cmds = []
    if "T" in valid and scripts["T"].exists():
        cmds.append(
            [
                "python3",
                str(scripts["T"]),
                "--matrix",
                str(valid["T"]),
                "--out_dir",
                str(oof_dir),
                "--n_splits",
                str(int(n_splits)),
                "--seed",
                str(int(seed)),
                "--pos_label",
                pos_label,
                "--neg_label",
                neg_label,
            ]
        )
    if "M" in valid and scripts["M"].exists():
        cmds.append(
            [
                "python3",
                str(scripts["M"]),
                "--matrix",
                str(valid["M"]),
                "--out_dir",
                str(oof_dir),
                "--n_splits",
                str(int(n_splits)),
                "--seed",
                str(int(seed)),
                "--pos_label",
                pos_label,
                "--neg_label",
                neg_label,
            ]
        )
    if "P" in valid and scripts["P"].exists():
        cmds.append(
            [
                "python3",
                str(scripts["P"]),
                "--matrix",
                str(valid["P"]),
                "--out_dir",
                str(oof_dir),
                "--n_splits",
                str(int(n_splits)),
                "--seed",
                str(int(seed)),
                "--pos_label",
                pos_label,
                "--neg_label",
                neg_label,
            ]
        )

    if not cmds:
        st.error("No training scripts found for the selected non-empty modalities. Check script paths.")
        st.write({k: str(v) for k, v in scripts.items()})
        st.stop()

    with st.spinner("Training unimodal models..."):
        logs = []
        for cmd in cmds:
            p = subprocess.run(cmd, capture_output=True, text=True)
            logs.append(
                {
                    "cmd": " ".join(cmd),
                    "returncode": p.returncode,
                    "stdout_tail": _tail(p.stdout),
                    "stderr_tail": _tail(p.stderr),
                }
            )
            if p.returncode != 0:
                st.error(f"Training failed for: {' '.join(cmd)}")
                if p.stdout:
                    st.code(_tail(p.stdout, 8000))
                if p.stderr:
                    st.code(_tail(p.stderr, 8000))
                st.stop()

    st.subheader("Collecting OOF outputs into this run (canonical filenames)")
    collected = _collect_oof_files_to_run_oof_dir(project_root=project_root, oof_dir=oof_dir)
    st.json(collected)

    st.success("Unimodal training complete. OOF files are ready for Stage 7.")
    with st.expander("Training logs"):
        st.json(logs)

    st.write("OOF directory:", str(oof_dir))
    st.write("Files:", [p.name for p in sorted(oof_dir.glob("*.csv"))])

st.divider()


# =============================================================================
# 14) Stage 7: Option A weights + board figures
# =============================================================================
st.header("7) Option A weights + board figures")

ensemble_dir = train_dir / "ensemble"
fig_dir = train_dir / "board_figures"
ensemble_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)

oof_T = oof_dir / "oof_transcriptomics.csv"
oof_M = oof_dir / "oof_methylation.csv"
oof_P = oof_dir / "proteomics_subject_mean_oof.csv"

st.write("Expected OOF files (canonical):")
st.write({"T": str(oof_T), "M": str(oof_M), "P": str(oof_P)})

missing_oof = [name for name, f in [("T", oof_T), ("M", oof_M), ("P", oof_P)] if not f.exists()]
if missing_oof:
    st.info(f"Waiting for OOF files. Missing: {missing_oof}")

from sklearn.metrics import roc_auc_score  # noqa: E402


def read_oof(path: Path, modality: str) -> pd.DataFrame:
    """Load an OOF file and standardize column names to sample_id/proba/y/modality."""
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}

    proba_col = cols.get("proba") or cols.get("prob") or cols.get("p") or cols.get("yhat")
    y_col = cols.get("y") or cols.get("label") or cols.get("target")
    sid_col = cols.get("sample_id") or cols.get("sample") or cols.get("id")

    if not (proba_col and y_col):
        raise ValueError(f"{modality}: OOF must contain proba and y columns (found {df.columns.tolist()})")

    out = pd.DataFrame(
        {
            "sample_id": df[sid_col].astype(str) if sid_col else np.arange(len(df)).astype(str),
            "proba": pd.to_numeric(df[proba_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce"),
            "modality": modality,
        }
    ).dropna(subset=["proba", "y"])
    out["y"] = out["y"].astype(int)
    return out


def modality_auc(oof: pd.DataFrame) -> float:
    """Compute AUC for a single modality."""
    y = oof["y"].to_numpy()
    p = oof["proba"].to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def normalized_weights(aucs: Dict[str, float]) -> Dict[str, float]:
    """Convert AUCs to nonnegative weights summing to 1 (fallback to uniform)."""
    vals = {k: (v if np.isfinite(v) and v > 0 else 0.0) for k, v in aucs.items()}
    s = float(sum(vals.values()))
    if s <= 0:
        n = len(vals)
        return {k: 1.0 / n for k in vals}
    return {k: float(v / s) for k, v in vals.items()}


ens_btn = st.button("Build weights + export figures", type="primary", key="btn_ensemble")

if ens_btn:
    available = {}
    for mod, f in [("transcriptomics", oof_T), ("methylation", oof_M), ("proteomics", oof_P)]:
        if f.exists():
            available[mod] = f

    if not available:
        st.error("No OOF files found. Run Stage 6 training first.")
        st.stop()

    aucs: Dict[str, float] = {}
    for mod, f in available.items():
        df = read_oof(f, mod)
        aucs[mod] = modality_auc(df)

    w = normalized_weights(aucs)

    weights_path = ensemble_dir / "weights.json"
    weights_path.write_text(json.dumps({"aucs": aucs, "weights": w}, indent=2), encoding="utf-8")

    st.success("Weights computed and saved.")
    st.json({"aucs": aucs, "weights": w})

    # Board figure: AUC bar plot
    import matplotlib.pyplot as plt  # noqa: E402

    mods = list(aucs.keys())
    vals = [aucs[m] for m in mods]

    fig = plt.figure()
    plt.bar(mods, vals)
    plt.ylim(0, 1)
    plt.ylabel("OOF AUC")
    plt.title("Modality performance (OOF AUC)")
    fig_path = fig_dir / "modality_performance_auc.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    st.image(str(fig_path), caption="Modality performance (OOF AUC)")
    st.write("Saved:")
    st.write({"weights.json": str(weights_path), "auc_figure": str(fig_path)})

# --- Stage 8 remains unchanged from your original ---
# -------------------------
# Stage 8: Top features + essential board exports
# -------------------------
st.header("8) Top features + essential board exports (board pack)")

board_tables_dir = train_dir / "board_tables"
board_tables_dir.mkdir(parents=True, exist_ok=True)

board_pack_dir = train_dir / "board_pack"
board_pack_dir.mkdir(parents=True, exist_ok=True)

top_n = st.number_input(
    "Top N features per modality",
    min_value=10,
    max_value=500,
    value=50,
    step=10,
    key="ni_topn",
)
min_freq = st.number_input(
    "Min fold frequency (0-1) for stability table",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.05,
    key="ni_minfreq",
)


def _find_first_existing(paths: List[Path]) -> Optional[Path]:
    """Return the first path that exists, else None."""
    for p in paths:
        if p and p.exists():
            return p
    return None


def _find_oof_file(oof_dir: Path, candidates: List[str]) -> Optional[Path]:
    """
    Find an OOF file robustly:
      1) Prefer explicit canonical candidate filenames in oof_dir
      2) Else fallback to glob + pick something modality-looking
    """
    for name in candidates:
        p = oof_dir / name
        if p.exists():
            return p

    globs: List[Path] = []
    for pat in ("*oof*.csv", "*.csv"):
        globs.extend(sorted(oof_dir.glob(pat)))

    # Prefer files that look like the right modality
    for p in globs:
        s = p.name.lower()
        if any(k in s for k in ("transcript", "methyl", "proteo")):
            return p

    return globs[0] if globs else None


def _load_importance_csv(path: Path) -> pd.DataFrame:
    """Load a feature-importance CSV with stripped column names."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _standardize_top_features(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    """
    Accepts common formats:
      - Logistic: feature, coef, abs_coef, direction
      - XGB: feature, gain_importance
      - Generic: feature/feature_id + importance/score
    Returns a standardized table: feature, score, modality (+ extras if present).
    """
    cols = {c.lower(): c for c in df.columns}

    # Feature id column
    fcol = cols.get("feature") or cols.get("feature_id") or cols.get("id") or cols.get("name") or df.columns[0]

    # Score column
    score_col = None
    for cand in ("abs_coef", "gain_importance", "importance", "gain", "score"):
        if cand in cols:
            score_col = cols[cand]
            break

    if score_col is None:
        # Fallback: abs(coef) if present
        if "coef" in cols:
            df["_abs_coef_"] = pd.to_numeric(df[cols["coef"]], errors="coerce").abs()
            score_col = "_abs_coef_"
        else:
            raise ValueError(f"{modality}: cannot find an importance/score column in {df.columns.tolist()}")

    out = pd.DataFrame(
        {
            "feature": df[fcol].astype(str),
            "score": pd.to_numeric(df[score_col], errors="coerce"),
            "modality": modality,
        }
    )

    # Carry useful extras if present
    for extra in ("coef", "direction"):
        if extra in cols:
            out[extra] = df[cols[extra]]

    out = out.dropna(subset=["score"]).sort_values("score", ascending=False)
    return out


def _load_stability(fi_path: Path, modality: str) -> pd.DataFrame:
    """
    Expects a per-fold importance CSV like: feature_id, fold, importance
    Produces stability table: feature, freq, mean_importance, folds_seen, stability_score
    """
    fi = pd.read_csv(fi_path)
    fi.columns = [str(c).strip() for c in fi.columns]
    cols = {c.lower(): c for c in fi.columns}

    fcol = cols.get("feature_id") or cols.get("feature") or fi.columns[0]
    icol = cols.get("importance") or cols.get("score") or cols.get("abs_coef")
    foldcol = cols.get("fold")

    if icol is None or foldcol is None:
        raise ValueError(
            f"{modality}: stability file must contain feature_id/feature, fold, importance/score. "
            f"Found {fi.columns.tolist()}"
        )

    fi[fcol] = fi[fcol].astype(str)
    fi[foldcol] = pd.to_numeric(fi[foldcol], errors="coerce")
    fi[icol] = pd.to_numeric(fi[icol], errors="coerce")
    fi = fi.dropna(subset=[fcol, foldcol, icol]).copy()

    n_folds = int(fi[foldcol].nunique()) if fi.shape[0] else 0
    g = fi.groupby(fcol, as_index=False).agg(
        mean_importance=(icol, "mean"),
        freq=(foldcol, lambda x: float(x.nunique()) / max(1, n_folds)),
        folds_seen=(foldcol, "nunique"),
    )
    g["stability_score"] = g["mean_importance"] * g["freq"]
    g["modality"] = modality
    g = g.rename(columns={fcol: "feature"}).sort_values(
        ["stability_score", "mean_importance"], ascending=False
    )
    return g


def _save_df(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


def _plot_roc_and_cm(oof_path: Path, modality: str) -> Dict[str, str]:
    """
    Save ROC + Confusion Matrix plots into fig_dir and CM CSV into board_tables_dir.
    Returns a dict of saved paths.
    """
    from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay, confusion_matrix
    import matplotlib.pyplot as plt

    o = read_oof(oof_path, modality)  # Stage 7 reader (sample_id/proba/y)
    y = o["y"].to_numpy()
    p = o["proba"].to_numpy()

    saved: Dict[str, str] = {}

    # ROC
    if len(np.unique(y)) >= 2:
        fpr, tpr, _ = roc_curve(y, p)
        roc_auc = auc(fpr, tpr)
        fig = plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{modality} ROC (AUC={roc_auc:.3f})")
        roc_path = fig_dir / f"{modality}_roc.png"
        fig.savefig(roc_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved["roc"] = str(roc_path)

    # Confusion matrix (threshold 0.5)
    pred = (p >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])

    fig = plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["0", "1"])
    disp.plot(values_format="d")
    plt.title(f"{modality} Confusion Matrix (thr=0.5)")
    cm_path = fig_dir / f"{modality}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    saved["cm"] = str(cm_path)

    cm_csv = board_tables_dir / f"{modality}_confusion_matrix.csv"
    pd.DataFrame(cm, index=["true0", "true1"], columns=["pred0", "pred1"]).to_csv(cm_csv)
    saved["cm_csv"] = str(cm_csv)

    return saved


def _zip_board_pack(out_zip: Path, include_paths: List[Path]) -> None:
    """Create a zip with paths stored relative to train_dir for neatness."""
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in include_paths:
            if not p or not p.exists() or not p.is_file():
                continue
            arc = p.relative_to(train_dir) if train_dir in p.parents else p.name
            z.write(p, arcname=str(arc))


stage8_btn = st.button(
    "Build Stage 8 board exports (top features, stability, ROC/CM, zip)",
    type="primary",
    key="btn_stage8",
)

if stage8_btn:
    # --- 1) locate OOF files robustly ---
    oof_T8 = _find_oof_file(oof_dir, ["oof_transcriptomics.csv"])
    oof_M8 = _find_oof_file(oof_dir, ["oof_methylation.csv"])
    oof_P8 = _find_oof_file(oof_dir, ["proteomics_subject_mean_oof.csv", "proteomic_subject_mean_oof.csv"])

    oof_map: Dict[str, Optional[Path]] = {
        "transcriptomics": oof_T8,
        "methylation": oof_M8,
        "proteomics": oof_P8,
    }

    st.write("Detected OOF files for Stage 8:")
    st.write({k: (str(v) if v else None) for k, v in oof_map.items()})

    # --- 2) locate feature importance + stability files ---
    feat_map: Dict[str, Optional[Path]] = {
        "transcriptomics": _find_first_existing(
            [
                train_dir / "oof" / "transcriptomics_feature_importance.csv",
                project_root / "transcriptomics_feature_importance.csv",
            ]
        ),
        "methylation": _find_first_existing(
            [
                train_dir / "oof" / "methylation_feature_importance.csv",
                project_root / "methylation_feature_importance.csv",
            ]
        ),
        "proteomics": _find_first_existing(
            [
                train_dir / "oof" / "proteomics_subject_mean_feature_importance.csv",
                project_root / "proteomics_subject_mean_feature_importance.csv",
            ]
        ),
    }

    stability_map: Dict[str, Optional[Path]] = {
        "transcriptomics": _find_first_existing([train_dir / "oof" / "fi_T_all_folds.csv", project_root / "fi_T_all_folds.csv"]),
        "methylation": _find_first_existing([train_dir / "oof" / "fi_M_all_folds.csv", project_root / "fi_M_all_folds.csv"]),
        "proteomics": _find_first_existing([train_dir / "oof" / "fi_P_all_folds.csv", project_root / "fi_P_all_folds.csv"]),
    }

    st.write("Detected feature importance files:")
    st.write({k: (str(v) if v else None) for k, v in feat_map.items()})

    st.write("Detected per-fold stability files:")
    st.write({k: (str(v) if v else None) for k, v in stability_map.items()})

    exported_paths: List[Path] = []

    # --- 3) export top N features per modality ---
    top_tables: List[pd.DataFrame] = []
    for mod, fpath in feat_map.items():
        if not fpath:
            continue
        try:
            raw = _load_importance_csv(fpath)
            top = _standardize_top_features(raw, mod).head(int(top_n))
            out_csv = board_tables_dir / f"top_{int(top_n)}_{mod}_features.csv"
            _save_df(top, out_csv)
            exported_paths.append(out_csv)
            top_tables.append(top)
        except Exception as e:
            st.warning(f"Top-features export failed for {mod}: {e}")

    if top_tables:
        st.subheader("Top features (preview)")
        st.dataframe(pd.concat(top_tables, ignore_index=True).head(200))

    # --- 4) export stability tables (frequency across folds) ---
    stab_tables: List[pd.DataFrame] = []
    for mod, fpath in stability_map.items():
        if not fpath:
            continue
        try:
            stab = _load_stability(fpath, mod)
            stab = stab[stab["freq"] >= float(min_freq)].head(int(top_n))
            out_csv = board_tables_dir / f"stable_top_{int(top_n)}_{mod}_features_minfreq_{min_freq:.2f}.csv"
            _save_df(stab, out_csv)
            exported_paths.append(out_csv)
            stab_tables.append(stab)
        except Exception as e:
            st.warning(f"Stability export failed for {mod}: {e}")

    if stab_tables:
        st.subheader("Feature stability (preview)")
        st.dataframe(pd.concat(stab_tables, ignore_index=True).head(200))

    # --- 5) export ROC + confusion matrix plots per modality ---
    plot_paths: List[Path] = []
    for mod, oof_path in oof_map.items():
        if not oof_path or not oof_path.exists():
            continue
        try:
            saved = _plot_roc_and_cm(oof_path, mod)
            for _, sp in saved.items():
                p = Path(sp)
                if p.exists():
                    exported_paths.append(p)
                    plot_paths.append(p)
        except Exception as e:
            st.warning(f"ROC/CM export failed for {mod}: {e}")

    if plot_paths:
        st.subheader("ROC / Confusion Matrix outputs")
        for p in plot_paths:
            if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                st.image(str(p), caption=p.name)

    # --- 6) include weights.json + auc figure if they exist ---
    weights_json = ensemble_dir / "weights.json"
    auc_fig = fig_dir / "modality_performance_auc.png"
    for p in (weights_json, auc_fig):
        if p.exists():
            exported_paths.append(p)

    # --- 7) zip it ---
    board_zip = board_pack_dir / "board_pack.zip"
    _zip_board_pack(board_zip, include_paths=exported_paths)

    st.success("Stage 8 exports complete.")
    st.write("Board pack zip:", str(board_zip))
    st.write(
        "Exported files:",
        [
            str(p.relative_to(train_dir)) if train_dir in p.parents else str(p)
            for p in exported_paths
        ],
    )
