#!/usr/bin/env python3
from __future__ import annotations

import sys
import re
import gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd


# ---------- hard-coded labeling ----------
SUPPRESSED_VALUE = 10.0
VIREMIC_THRESHOLD = 50.0

# output names (same folder as methylation file)
OUT_RENAMED = "methylation_RENAMED_GSM.txt.gz"
OUT_LABELS = "labels_methylation.csv"
OUT_MAPPING = "methylation_sample_mapping_SAMPLE_to_GSM.csv"

GSM_RE = re.compile(r"^GSM\d+$", re.IGNORECASE)
SAMPLE_RE = re.compile(r"^SAMPLE(\d+)$", re.IGNORECASE)


def die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def open_text(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt", errors="ignore")
    return path.open("r", errors="ignore")


def open_text_write_gz(path: Path):
    return gzip.open(path, "wt")


def _strip_cell(x: str) -> str:
    return x.strip().strip('"').strip()


def read_series_row(series_path: Path, prefix: str) -> Optional[List[str]]:
    """
    Return the tab-separated payload cells (excluding the first token) for the first row that startswith prefix.
    Example: prefix="!Sample_geo_accession"
    """
    with open_text(series_path) as f:
        for line in f:
            if line.startswith(prefix):
                parts = line.rstrip("\n").split("\t")[1:]
                return [_strip_cell(p) for p in parts]
    return None


def read_series_gsms(series_path: Path) -> List[str]:
    row = read_series_row(series_path, "!Sample_geo_accession")
    if not row:
        return []
    return [p for p in row if GSM_RE.match(p)]


def read_series_characteristics(series_path: Path, gsm_ids: List[str]) -> pd.DataFrame:
    char_rows: List[List[str]] = []
    with open_text(series_path) as f:
        for line in f:
            if line.startswith("!Sample_characteristics_ch1"):
                parts = [_strip_cell(p) for p in line.rstrip("\n").split("\t")[1:]]
                char_rows.append(parts)

    if not char_rows:
        die("No !Sample_characteristics_ch1 lines found in series matrix.")

    min_len = min(len(r) for r in char_rows)
    if min_len == 0:
        die("Characteristics lines present but empty.")

    gsm_ids = gsm_ids[:min_len]
    char_rows = [r[:min_len] for r in char_rows]

    per_sample = []
    for j, gsm in enumerate(gsm_ids):
        fields: Dict[str, str] = {}
        for r in char_rows:
            cell = r[j]
            if ":" in cell:
                k, v = cell.split(":", 1)
                fields[k.strip()] = v.strip()
        per_sample.append(fields)

    meta = pd.DataFrame(per_sample, index=gsm_ids)
    meta.index.name = "GSM"
    return meta


def choose_viral_load_col(meta: pd.DataFrame) -> str:
    if "HIV.Viral.Load" in meta.columns:
        return "HIV.Viral.Load"

    candidates = []
    for c in meta.columns:
        cl = c.lower()
        if "viral" in cl and "load" in cl:
            candidates.append(c)

    if not candidates:
        die("Could not find a viral load field in characteristics (looked for keys containing both 'viral' and 'load').")

    # If multiple, pick the shortest key as a reasonable heuristic (often the canonical one).
    candidates.sort(key=lambda x: (len(x), x))
    return candidates[0]


def parse_viral_load_value(x) -> float:
    """
    Tolerant parsing for viral load values:
    - numeric strings
    - strings like "<50", "50 copies/mL", "undetectable"
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return float("nan")

    s = str(x).strip().lower()
    if s in {"undetectable", "undet", "not detected", "negative"}:
        return 0.0

    # handle "<50" style
    if s.startswith("<"):
        m = re.search(r"(\d+(\.\d+)?)", s)
        return float(m.group(1)) if m else float("nan")

    # grab first number anywhere in the string
    m = re.search(r"(\d+(\.\d+)?)", s)
    if m:
        return float(m.group(1))

    return float("nan")


def make_groups(meta: pd.DataFrame, vl_col: str) -> pd.Series:
    vl = meta[vl_col].map(parse_viral_load_value)
    group = pd.Series(["unknown"] * len(vl), index=vl.index, dtype="object")
    group.loc[vl <= SUPPRESSED_VALUE] = "suppressed"
    group.loc[vl > VIREMIC_THRESHOLD] = "viremic"
    return group


def find_sample_placeholders_from_header(header_cols: List[str]) -> List[str]:
    s = set()
    for c in header_cols:
        m = re.match(r"^(SAMPLE\d+)\b", c, flags=re.IGNORECASE)
        if m:
            s.add(m.group(1).upper())

    if not s:
        die("No SAMPLE# placeholders found in methylation header.")

    def key(x: str):
        m2 = re.match(r"^SAMPLE(\d+)$", x, flags=re.IGNORECASE)
        return int(m2.group(1)) if m2 else 10**12

    return sorted(s, key=key)


def build_sample_to_gsm_mapping(series_path: Path, sample_placeholders: List[str]) -> Dict[str, str]:
    """
    Ensures correct alignment:
    - reads !Sample_geo_accession -> GSM list (positional)
    - ensures we have same count as SAMPLE# placeholders
    - maps SAMPLE1..SAMPLEn in numeric order to GSMs in positional order

    Note: This assumes the methylation file's SAMPLE# columns correspond to the series matrix sample ordering.
    That is the common GEO convention for "non-normalized" tables shipped with the series.
    """
    gsm_row = read_series_row(series_path, "!Sample_geo_accession")
    if not gsm_row:
        die("Series matrix missing !Sample_geo_accession line.")

    gsm_ids = [_strip_cell(x) for x in gsm_row]
    gsm_ids = [g for g in gsm_ids if GSM_RE.match(g)]

    if len(gsm_ids) != len(sample_placeholders):
        die(
            f"Count mismatch: found {len(sample_placeholders)} SAMPLE# placeholders in methylation header "
            f"but {len(gsm_ids)} GSMs in series matrix. Ensure you used the matching GSE series matrix."
        )

    # SAMPLE# are already sorted numerically; GSM list is positional.
    return dict(zip(sample_placeholders, gsm_ids))


def main() -> None:
    if len(sys.argv) != 3:
        die("Usage: python3 multiomics/scripts/methylation_fast_onefile.py <GSE*_series_matrix.txt(.gz)> <methylation_non_normalized.txt(.gz)>")

    series_path = Path(sys.argv[1]).expanduser().resolve()
    meth_path = Path(sys.argv[2]).expanduser().resolve()

    if not series_path.exists():
        die(f"Series matrix not found: {series_path}")
    if not meth_path.exists():
        die(f"Methylation file not found: {meth_path}")

    out_dir = meth_path.parent
    out_renamed = out_dir / OUT_RENAMED
    out_labels = out_dir / OUT_LABELS
    out_mapping = out_dir / OUT_MAPPING

    # 1) Read methylation header (stream)
    with open_text(meth_path) as fin:
        header = fin.readline()
        if not header:
            die("Methylation file is empty.")

        delim = "\t" if header.count("\t") > header.count(",") else ","
        header_cols = [_strip_cell(h) for h in header.rstrip("\n").split(delim)]

        # detect SAMPLE placeholders among data columns
        sample_placeholders = find_sample_placeholders_from_header(header_cols[1:])

    # 2) Build mapping SAMPLE# -> GSM#### from series matrix (aligned)
    mapping = build_sample_to_gsm_mapping(series_path, sample_placeholders)

    # 3) Build labels from characteristics (tiny)
    gsm_ids = list(mapping.values())
    meta = read_series_characteristics(series_path, gsm_ids)
    vl_col = choose_viral_load_col(meta)
    groups = make_groups(meta, vl_col)

    # 4) Stream-transform methylation file (FAST)
    def rename_col(c: str) -> str:
        m = re.match(r"^(SAMPLE\d+)(.*)$", c, flags=re.IGNORECASE)
        if not m:
            return c
        old = m.group(1).upper()
        rest = m.group(2)
        return mapping.get(old, old) + rest

    with open_text(meth_path) as fin, open_text_write_gz(out_renamed) as fout:
        header = fin.readline()
        delim = "\t" if header.count("\t") > header.count(",") else ","
        header_cols = [_strip_cell(h) for h in header.rstrip("\n").split(delim)]
        renamed_header_cols = [header_cols[0]] + [rename_col(c) for c in header_cols[1:]]
        fout.write(delim.join(renamed_header_cols) + "\n")
        for line in fin:
            fout.write(line)

    # 5) Write mapping + labels
    pd.DataFrame(
        {"old_sample": list(mapping.keys()), "new_sample": [mapping[k] for k in mapping.keys()]}
    ).to_csv(out_mapping, index=False)

    pd.DataFrame({"sample_id": groups.index, "group": groups.values}).to_csv(out_labels, index=False)

    vc = groups.value_counts(dropna=False).to_dict()
    print(f"Wrote renamed methylation: {out_renamed}")
    print(f"Wrote labels: {out_labels}")
    print(f"Wrote mapping: {out_mapping}")
    print(f"Viral load field used: {vl_col}")
    print(f"Label counts: {vc}")


if __name__ == "__main__":
    main()
