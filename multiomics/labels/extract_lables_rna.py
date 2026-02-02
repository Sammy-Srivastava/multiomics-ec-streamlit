import pandas as pd

series_matrix = "GSE87620_series_matrix.txt"
out = "labels_transcriptomics.csv"

# -----------------------------
# Helpers
# -----------------------------
def dequote(s: str) -> str:
    if s is None:
        return s
    s = s.strip()
    # strip matching single/double quotes
    if (len(s) >= 2) and ((s[0] == s[-1]) and s[0] in ['"', "'"]):
        s = s[1:-1]
    return s.strip()

def normalize(x: str) -> str:
    x = dequote(str(x)).lower()

    # Strip the common prefix if present
    x = x.replace("infection status:", "").strip()

    # Healthy controls
    if "hiv-" in x.replace(" ", "") or "hiv negative" in x or "seronegative" in x or "donor" in x and "hiv-" in x.replace(" ", ""):
        return "HC"

    # Elite controllers
    if "elite controller" in x or ("elite" in x and "controller" in x) or "controller" in x:
        return "EC"

    # Chronic treated == ART-treated chronic infection
    if "chronic treated" in x:
        return "ART"

    # Other treated patterns (backup)
    if "treated" in x or "art" in x or "cart" in x or "haart" in x or "suppressed" in x:
        return "ART"

    # Viremic / untreated patterns (backup)
    if "viremic" in x or "untreated" in x or "naive" in x:
        return "Viremic"

    return "Unknown"



def score_row(vals):
    labs = [normalize(v) for v in vals]
    unknown = sum(1 for l in labs if l == "Unknown")
    known = len(labs) - unknown
    return unknown, known, labs

def looks_like_label_row(vals):
    # heuristic: if many values contain these tokens, it's likely the group/status row
    joined = " | ".join(dequote(str(v)).lower() for v in vals[: min(len(vals), 10)])
    keywords = ["disease", "status", "group", "phenotype", "hiv", "infection"]
    return any(k in joined for k in keywords)

# -----------------------------
# Parse series matrix
# -----------------------------
sample_ids = []
characteristics_rows = []

with open(series_matrix, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if line.startswith("!Sample_geo_accession"):
            sample_ids = [dequote(x) for x in line.rstrip("\n").split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            vals = [dequote(x) for x in line.rstrip("\n").split("\t")[1:]]
            # Keep all rows; we'll pick the best one later
            characteristics_rows.append(vals)

if not sample_ids:
    raise RuntimeError("Could not find !Sample_geo_accession in series matrix.")

if not characteristics_rows:
    raise RuntimeError("Could not find any !Sample_characteristics_ch1 lines in series matrix.")

# Ensure consistent length
n = len(sample_ids)
characteristics_rows = [row for row in characteristics_rows if len(row) == n]
if not characteristics_rows:
    raise RuntimeError("No characteristics rows matched the number of samples (length mismatch).")

# -----------------------------
# Choose best label row
# -----------------------------

# Prefer rows that actually contain group-defining tokens for EC/ART/Viremic/HC
TARGET_TOKENS = ["elite", "controller", "viremic", "haart", "cart", "art", "treated", "aviremic"]

def contains_target_tokens(row):
    text = " ".join(dequote(str(v)).lower() for v in row)
    return any(tok in text for tok in TARGET_TOKENS)

candidate_rows = [r for r in characteristics_rows if contains_target_tokens(r)]
rows_to_search = candidate_rows if candidate_rows else characteristics_rows


best = None
for row in rows_to_search:
    unknown, known, labs = score_row(row)

    # Prefer rows with fewer Unknowns; break ties by more known;
    # if still tied, prefer rows that look like they contain 'status/group' keys.
    candidate = (unknown, -known, 0 if looks_like_label_row(row) else 1)
    if best is None or candidate < best["candidate"]:
        best = {"candidate": candidate, "row": row, "labs": labs, "unknown": unknown, "known": known}

labels = best["labs"]

df = pd.DataFrame({"sample_id": sample_ids, "label": labels})

# -----------------------------
# Safety check with diagnostics
# -----------------------------
if (df["label"] == "Unknown").any():
    bad = df.loc[df["label"] == "Unknown", "sample_id"].tolist()[:10]
    # Print a quick hint about what the chosen row looks like for those samples
    chosen_row = best["row"]
    bad_idx = df.index[df["label"] == "Unknown"].tolist()[:5]
    preview = [(sample_ids[i], chosen_row[i]) for i in bad_idx]
    raise ValueError(
        f"Unknown labels for samples (showing up to 10): {bad}\n"
        f"Chosen characteristics row produced {best['unknown']} Unknown out of {len(sample_ids)} samples.\n"
        f"Preview (sample_id, raw_characteristic): {preview}\n"
        f"If this dataset uses different wording, add rules in normalize()."
    )

df.to_csv(out, index=False)
print(f"Saved {out} (n={len(df)}) using best characteristics row: Unknown={best['unknown']}, Known={best['known']}")
