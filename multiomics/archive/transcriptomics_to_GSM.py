from __future__ import annotations
from pathlib import Path
import pandas as pd

PROJECT = Path("/Users/samyaksrivastava/Desktop/new science fair thing")
SERIES_MATRIX = PROJECT / "GSE87620_series_matrix.txt"
OUT_MAP = PROJECT / "transcriptomics_sample_mapping_SAMPLE_to_GSM.csv"

# You found S-style IDs here:
SAMPLE_FIELD = "!Sample_description"

def get_vals(lines, field):
    for ln in lines:
        if ln.startswith(field + "\t"):
            parts = ln.split("\t")[1:]
            return [x.strip().strip('"') for x in parts if x.strip()]
    return None

def main():
    if not SERIES_MATRIX.exists():
        raise FileNotFoundError(SERIES_MATRIX)

    lines = SERIES_MATRIX.read_text(errors="ignore").splitlines()
    gsm = get_vals(lines, "!Sample_geo_accession")
    old = get_vals(lines, SAMPLE_FIELD)

    if gsm is None:
        raise ValueError("Could not find !Sample_geo_accession in series matrix.")
    if old is None:
        raise ValueError(f"Could not find {SAMPLE_FIELD} in series matrix.")

    if len(gsm) != len(old):
        raise ValueError(f"Length mismatch: {len(gsm)} GSMs vs {len(old)} in {SAMPLE_FIELD}")

    df = pd.DataFrame({"old_sample": old, "new_sample": gsm})
    df["old_sample"] = df["old_sample"].astype(str).str.strip()
    df["new_sample"] = df["new_sample"].astype(str).str.strip()

    df = df[df["new_sample"].str.startswith("GSM")].copy()
    df = df.drop_duplicates(subset=["old_sample"], keep="first")

    df.to_csv(OUT_MAP, index=False)
    print("[SAVED]", OUT_MAP)
    print(df.head(15).to_string(index=False))
    print("\nMapped rows:", len(df))

if __name__ == "__main__":
    main()
