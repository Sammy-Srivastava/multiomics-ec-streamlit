import json
from pathlib import Path

import numpy as np
import pandas as pd

from harmonizing_stuff.classes_omic_normalizer.base import BaseNormalizer


class TranscriptomicNormalizer(BaseNormalizer):
    def __init__(
        self,
        min_expression=1.0,
        min_sample_fraction=0.2,
        min_library_percentile=5,
        scale=True,
        log1p=True,
    ):
        super().__init__(method="transcriptomic_qc")
        self.min_expression = float(min_expression)
        self.min_sample_fraction = float(min_sample_fraction)
        self.min_library_percentile = float(min_library_percentile)
        self.scale = bool(scale)
        self.log1p = bool(log1p)

        self._metadata = None
        self._raw_X = None
        self._X_qc = None
        self._gene_means = None
        self._gene_stds = None
        self._sample_ids = None

        self.fitted = False
        self.report = {}

    #HELPERS
    def read_counts_matrix(self, path: str) -> pd.DataFrame:
        p = Path(path)
        name = p.name.lower()

        # Parquet file read
        if name.endswith(".parquet"):
            df = pd.read_parquet(p)
            if df is None or df.shape[0] == 0 or df.shape[1] == 0:
                raise ValueError(f"Transcriptomics: parquet read produced empty matrix. File={path}")

            df.index = df.index.astype(str).str.strip()
            df.columns = df.columns.astype(str).str.strip()

            # Ensure numeric (parquet sometimes preserves object dtype if created oddly)
            df = df.apply(pd.to_numeric, errors="coerce")

            if df.shape[1] < 2:
                raise ValueError(
                    f"Transcriptomics: parsed parquet but <2 sample cols. Shape={df.shape}. "
                    f"Columns head={df.columns.tolist()[:10]}"
                )
            return df

        sep = "\t" if name.endswith((".tsv", ".txt", ".tsv.gz", ".txt.gz")) else ","
        compression = "gzip" if name.endswith(".gz") else None

        # C path (fast path)
        last_err = None
        try:
            df = pd.read_csv(
                p,
                sep=sep,
                index_col=0,
                compression=compression,
                engine="c",
                low_memory=True,
            )
        except Exception as e:
            last_err = e
            df = None

        # tolerant fallback
        if df is None:
            for enc in ("utf-8", "latin1", "cp1252"):
                try:
                    df = pd.read_csv(
                        p,
                        sep=sep,
                        index_col=0,
                        compression="infer",
                        engine="python",
                        encoding=enc,
                        on_bad_lines="skip",
                    )
                    break
                except Exception as e:
                    last_err = e
                    df = None

        # GEO series matrix fall back
        if df is None:
            for enc in ("utf-8", "latin1", "cp1252"):
                try:
                    df = pd.read_csv(
                        p,
                        sep="\t",
                        index_col=0,
                        compression="infer",
                        engine="python",
                        encoding=enc,
                        comment="!",
                        on_bad_lines="skip",
                    )
                    if df is not None and df.shape[0] > 0 and df.shape[1] > 1:
                        break
                except Exception as e:
                    last_err = e
                    df = None

        if df is None:
            raise ValueError(
                f"Failed to read {p.name}. Likely wrong file type or encoding. "
                f"Last error: {repr(last_err)}"
            )

        if df.shape[0] == 0 or df.shape[1] == 0:
            raise ValueError(f"File parsed to empty matrix shape={df.shape}. File={path}")

        df.index = df.index.astype(str).str.strip()
        df.columns = df.columns.astype(str).str.strip()

        # Numeric coercion once (avoid repeating this in load_files)
        df = df.apply(pd.to_numeric, errors="coerce")

        if df.shape[1] < 2:
            raise ValueError(
                f"Parsed text but <2 sample cols. Shape={df.shape}. "
                f"Columns head={df.columns.tolist()[:10]}"
            )

        # Reduce memory footprint (safe for counts/log)
        # df = df.astype("float32")

        return df

    def load_files(self, raw_counts_file: str, series_matrix_file: str = None):
        self._raw_X = self.read_counts_matrix(raw_counts_file)

        # sample ids
        self._sample_ids = list(self._raw_X.columns)

        # Metadata parsing 
        if series_matrix_file:
            metadata_dict = {}
            for enc in ("utf-8", "latin1", "cp1252"):
                try:
                    with open(series_matrix_file, "r", encoding=enc, errors="replace") as f:
                        for line in f:
                            if line.startswith("!Sample_characteristics_ch1"):
                                values = line.strip().split("\t")[1:]
                                clean_vals = [v.split(":", 1)[1].strip() if ":" in v else v for v in values]
                                key = f"characteristics_{len(metadata_dict)}"
                                metadata_dict[key] = clean_vals
                            elif line.startswith("!Sample_source_name_ch1"):
                                values = line.strip().split("\t")[1:]
                                metadata_dict["cell_type"] = [v.strip() for v in values]
                    break
                except Exception:
                    metadata_dict = {}
                    continue

            if metadata_dict:
                md = pd.DataFrame(metadata_dict)
                md.insert(0, "sample_id", self._sample_ids)
                md.columns = md.columns.str.lower()
                md = md.map(lambda x: x.replace('"', "") if isinstance(x, str) else x)
                md.replace("NA", np.nan, inplace=True)
                self._metadata = md
            else:
                self._metadata = None

        return self

    # QC / Normalization
    def fit(self, X: pd.DataFrame = None, metadata=None):
        if X is None:
            if self._raw_X is None:
                raise ValueError("No input X provided and no loaded raw matrix.")
            X = self._raw_X

        # Ensure numeric
        X = X.apply(pd.to_numeric, errors="coerce")

        self.report["initial_genes"] = int(X.shape[0])
        self.report["initial_samples"] = int(X.shape[1])

        # Gene filter
        expressed_mask = X > self.min_expression
        expression_fraction = expressed_mask.mean(axis=1)
        genes_to_keep = expression_fraction >= self.min_sample_fraction
        X = X.loc[genes_to_keep]
        self.report["genes_removed"] = int((~genes_to_keep).sum())

        # Sample filter (library size)
        library_sizes = X.sum(axis=0, skipna=True)
        cutoff = np.percentile(library_sizes.values, self.min_library_percentile)
        samples_to_keep = library_sizes >= cutoff
        X = X.loc[:, samples_to_keep]
        self.report["samples_removed"] = int((~samples_to_keep).sum())

        # log transform
        if self.log1p:
            X_work = np.log2(X + 1.0)
            self.report["log_transform"] = "log2(x+1)"
        else:
            X_work = X.copy()
            self.report["log_transform"] = "none"

        self._X_qc = X_work.copy()

        # Scaling params
        self._gene_means = X_work.mean(axis=1, skipna=True)
        self._gene_stds = X_work.std(axis=1, skipna=True).replace(0, 1.0)

        self.report["scaling"] = "zscore" if self.scale else "none"
        self.report["zero_variance_genes"] = int((X_work.std(axis=1, skipna=True) == 0).sum())
        self.report["final_genes"] = int(self._X_qc.shape[0])
        self.report["final_samples"] = int(self._X_qc.shape[1])

        self.fitted = True
        return self

    def transform(self):
        if not self.fitted or self._X_qc is None:
            raise RuntimeError("Must call fit() before transform()")

        X = self._X_qc.copy()

        if self.scale:
            X = X.sub(self._gene_means, axis=0).div(self._gene_stds, axis=0)
            self.report["scaling_applied"] = True
        else:
            self.report["scaling_applied"] = False

        return X

    def get_qc_matrix(self):
        if not self.fitted or self._X_qc is None:
            raise RuntimeError("Must call fit() before get_qc_matrix()")
        return self._X_qc.copy()

    def get_metadata(self):
        return self._metadata.copy() if self._metadata is not None else None

    def save_report(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2)
