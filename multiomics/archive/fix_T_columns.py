#!/usr/bin/env python3
from pathlib import Path
import re
import pandas as pd
import numpy as np

# --------- EDIT THESE PATHS ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

IN_PATH = PROJECT_ROOT / 'UI_stuff/artifacts/harmonized/20260124_122548/T_harmonized.parquet'
# If yours is csv/tsv, change IN_PATH and read below.

OUT_MATRIX_PATH = PROJECT_ROOT / "UI_stuff" / "artifacts" / "transcriptomics_prepped" / "T_fixed.parquet"
OUT_META_PATH   = PROJECT_ROOT / "UI_stuff" / "artifacts" / "transcriptomics_prepped" / "T_sample_metadata.csv"

# If True: average Pos/Neg (and any other tech reps) into one column per subject_id (EC01/ART01/HC01/etc.)
COLLAPSE_TECH_REPS = True
# ------------------------------------


def read_matrix(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path, index_col=0)
    if suf in [".tsv", ".txt"]:
        return pd.read_csv(path, sep="\t", index_col=0)
    raise ValueError(f"Unsupported file type: {path}")


def parse_col(col: str):
    """
    Expected patterns like:
      Abundance:_F7:_128C,_Sample,_EC01,_Pos,_B1
      Abundance:_F8:_127N,_Sample,_ART03,_Neg,_B2
      ...,_Sample,_HC02,_Pos,_B1
      ...,_Sample,_Norm,_Norm,_B1   (special 'Norm' channel)
    Returns dict or None if doesn't match.
    """
    c = str(col)

    # primary: split by comma
    parts = [p.strip() for p in c.split(",")]
    if len(parts) < 5:
        return None

    # We expect "... , _Sample , <subject> , <posneg/norm> , <batch>"
    # Sometimes tokens look like "_Sample" exactly.
    if not any(p.endswith("_Sample") or p == "_Sample" or p == "Sample" for p in parts):
        return None

    # Find the sample token position
    sample_idx = None
    for i, p in enumerate(parts):
        if p.endswith("_Sample") or p == "_Sample" or p == "Sample":
            sample_idx = i
            break
    if sample_idx is None or sample_idx + 3 >= len(parts):
        return None

    subject_id = parts[sample_idx + 1].strip()   # EC01 / ART02 / HC02 / Norm
    rep        = parts[sample_idx + 2].strip()   # Pos / Neg / Norm
    batch      = parts[sample_idx + 3].strip()   # B1 / B2 / B3

    # Pull fraction/channel if present
    frac = None
    chan = None
    m = re.search(r"_F(\d+):_([^,]+)", c)  # captures F7 and 128C-ish
    if m:
        frac = f"F{m.group(1)}"
        chan = m.group(2)

    # Infer group label from subject_id prefix
    # ECxx -> EC, ARTxx -> ART, HCxx -> HC, Norm -> Norm/Ref
    group = None
    if re.match(r"^EC\d+$", subject_id):
        group = "EC"
    elif re.match(r"^ART\d+$", subject_id):
        group = "ART"
    elif re.match(r"^HC\d+$", subject_id):
        group = "HC"
    elif subject_id.lower() == "norm":
        group = "NORM"
    else:
        # unknown naming; keep as-is
        group = "UNKNOWN"

    return {
        "original_col": c,
        "subject_id": subject_id,
        "group": group,
        "rep": rep,
        "batch": batch,
        "fraction": frac,
        "channel": chan,
    }


def main():
    X = read_matrix(IN_PATH)
    cols = list(map(str, X.columns))
    print(f"Loaded matrix: {X.shape[0]} features x {X.shape[1]} columns")

    parsed = []
    for c in cols:
        info = parse_col(c)
        if info is not None:
            parsed.append(info)

    if len(parsed) == 0:
        raise ValueError(
            "None of the columns matched the expected TMT '...,_Sample,<ID>,<Rep>,<Batch>' pattern. "
            "This file may not be the expected proteomics-style matrix."
        )

    meta = pd.DataFrame(parsed)
    meta.to_csv(OUT_META_PATH, index=False)
    print(f"[SAVED] sample metadata: {OUT_META_PATH}")
    print(meta.head(10).to_string(index=False))

    # If not collapsing tech reps, rename columns to something readable (subject/rep/batch)
    if not COLLAPSE_TECH_REPS:
        rename_map = {}
        for row in parsed:
            new = f"{row['subject_id']}__{row['rep']}__{row['batch']}"
            rename_map[row["original_col"]] = new
        X2 = X.rename(columns=rename_map)
        X2.to_parquet(OUT_MATRIX_PATH)
        print(f"[SAVED] renamed matrix (no collapsing): {OUT_MATRIX_PATH}")
        return

    # Collapse tech reps: average all columns that share the same subject_id (optionally exclude NORM)
    grouped = meta.groupby("subject_id")["original_col"].apply(list).to_dict()

    out_cols = []
    out_data = []

    for sid, col_list in grouped.items():
        block = X[col_list]  # features x k replicates
        # average across columns
        avg = block.mean(axis=1)
        out_cols.append(sid)
        out_data.append(avg.to_numpy())

    Xc = pd.DataFrame(np.column_stack(out_data), index=X.index, columns=out_cols)

    # Optional: drop Norm reference if you don’t want it as a “sample”
    # Xc = Xc.drop(columns=["Norm"], errors="ignore")

    Xc.to_parquet(OUT_MATRIX_PATH)
    print(f"[SAVED] collapsed matrix (averaged tech reps): {OUT_MATRIX_PATH}")
    print(f"Collapsed shape: {Xc.shape[0]} features x {Xc.shape[1]} subjects")
    print("Subjects head:", list(Xc.columns)[:20])


if __name__ == "__main__":
    main()
