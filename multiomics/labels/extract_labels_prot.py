import pandas as pd

series_matrix = 'GSE157198_series_matrix.txt'
out = 'labels_transcriptomics.csv'

sample_ids = []
group_vals = None

with open(series_matrix, 'r') as f:
    for line in f:
        if line.startswith('!Sample_geo_accession'):
            sample_ids = [x.replace('"', '') for x in line.strip().split('\t')[1:]]

        if line.startswith('!Sample_characteristics_ch1'):
            vals = [v.replace('"', '').lower() for v in line.strip().split('\t')[1:]]
            # THIS is the phenotype row
            if all(v.startswith('group:') for v in vals):
                group_vals = vals

if group_vals is None:
    raise RuntimeError('Could not find group labels in series matrix.')

def normalize(x):
    # remove "group:"
    x = x.split(':', 1)[1].strip()

    if x.startswith('ec'):
        return 'EC'
    if x.startswith('art-naive') or 'naive' in x:
        return 'Viremic'
    if x.startswith('art'):
        return 'ART'
    if x.startswith('healthy'):
        return 'HC'
    return 'Unknown'

labels = [normalize(x) for x in group_vals]

df = pd.DataFrame({
    'sample_id': sample_ids,
    'label': labels
})

# Safety check (now should PASS)
if (df['label'] == 'Unknown').any():
    bad = df.loc[df['label'] == 'Unknown', 'sample_id'].tolist()
    raise ValueError(f'Unknown labels for samples: {bad}')

df.to_csv(out, index=False)
print('Saved', out)
