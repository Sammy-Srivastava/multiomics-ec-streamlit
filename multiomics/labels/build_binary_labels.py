from __future__ import annotations

from pathlib import Path
import pandas as pd

from multiomics.labels.label_mapping import to_ec_binary

project = Path('/Users/samyaksrivastava/Desktop/new science fair thing')

lab_P = project / 'labels_proteomics.csv'
lab_T = project / 'labels_transcriptomics.csv'
lab_M = project / 'labels_methylation.csv'

out_P = project / 'binary_proteomics.csv'
out_T = project / 'binary_transcriptomics.csv'
out_M = project / 'binary_methylation.csv'

#altering group columns
def normalize_group_col(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'group' not in df.columns:
        if 'label' in df.columns:
            df = df.rename(columns={'label': 'group'})
        elif 'labels' in df.columns:
            df = df.rename(columns={'labels': 'group'})
    if 'sample_id' not in df.columns:
        raise ValueError(f'Missing sample_id in columns: {list(df.columns)}')
    if 'group' not in df.columns:
        raise ValueError(f'Missing group/label in columns: {list(df.columns)}')
    return df

#turning into binary
def make_binary(df: pd.DataFrame, name:str) -> pd.DataFrame:
    df =  normalize_group_col(df)
    df['y'] = df['group'].apply(to_ec_binary)
    out = df.loc[df['y'].notna(), ['sample_id', 'y']].copy()
    out['y'] = out['y'].astype(int)
    print(f"\n{name}:")
    print("  total rows:", len(df))
    print("  usable rows:", len(out))
    print("  y counts:", out["y"].value_counts().to_dict())
    return out

def main():
    dfp = pd.read_csv(lab_P)
    outp = make_binary(dfp, 'Proteomics')
    outp.to_csv(out_P, index=False)
    
    dft = pd.read_csv(lab_T)
    outt = make_binary(dft, 'Transcriptomics')
    outt.to_csv(out_T, index=False)
    
    dfm = pd.read_csv(lab_M)
    outm = make_binary(dfm, 'Methylation')
    outm.to_csv(out_M, index=False)
    
    print("\nWrote:")
    print(" ", out_P)
    print(" ", out_T)
    print(" ", out_M)
    
if __name__ == '__main__':
    main()
    