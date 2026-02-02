import pandas as pd


ANN_PATH = "annotations/methylation_annotation_raw.csv"
OUT_PATH = "annotations/probe_to_gene.csv"

import pandas as pd

def find_header_line(path: str) -> int:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            # Your file's real header starts with "IlmnID,Name,..."
            if line.startswith("IlmnID,Name,"):
                return i
    raise ValueError("Could not find the annotation table header (IlmnID,Name,).")

header_line = find_header_line(ANN_PATH)

ann = pd.read_csv(
    ANN_PATH,
    sep=",",
    engine="python",
    skiprows=header_line
)

print("Loaded annotation table:", ann.shape)
print("First columns:", list(ann.columns)[:20])

probe_col = 'IlmnID'

promoter_groups = {"TSS200", "TSS1500", "5'UTR", "1stExon"}

ann = ann[[probe_col, 'UCSC_RefGene_Name', 'UCSC_RefGene_Group']]

ann = ann.dropna(subset=['UCSC_RefGene_Name', 'UCSC_RefGene_Group'])

ann = ann[
    ann['UCSC_RefGene_Group']
    .str.contains('|'.join(promoter_groups), regex=True, na=False)
]

ann["UCSC_RefGene_Name"] = ann["UCSC_RefGene_Name"].str.split(";")
ann = ann.explode("UCSC_RefGene_Name")

ann["UCSC_RefGene_Name"] = ann["UCSC_RefGene_Name"].str.strip()
ann = ann[ann["UCSC_RefGene_Name"] != ""]

probe_to_gene = ann.rename(
    columns={
        probe_col: 'probe_id',
        'UCSC_RefGene_Name': 'gene_id',
    }
)[['probe_id', 'gene_id']]

probe_to_gene = probe_to_gene.drop_duplicates()

OUT_PATH = 'annotations/probe_to_gene.csv'
probe_to_gene.to_csv(OUT_PATH, index=False)

print('Saved:', OUT_PATH)
print('Rows:', len(probe_to_gene))
print(probe_to_gene.head())