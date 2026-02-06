from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd


class ProteomicNormalizer:
    def __init__(
        self,
        prefer_lfq: bool = True,
        min_nonzero_fraction: float = 0.2,
        feature_level: str = "gene", 
    ):
        self.prefer_lfq = bool(prefer_lfq)
        self.min_nonzero_fraction = float(min_nonzero_fraction)
        if feature_level not in {"gene", "protein"}:
            raise ValueError("feature_level must be 'gene' or 'protein'")
        self.feature_level = feature_level

        self.report = {}
        self.fitted = False
        self._X_raw: Optional[pd.DataFrame] = None
        self._X_norm: Optional[pd.DataFrame] = None

    #---helpers---
    
    @staticmethod
    def _is_parquet_file(p: Path) -> bool:
        try:
            with open(p, "rb") as f:
                head = f.read(4)
            return head == b"PAR1"
        except Exception:
            return False

    @staticmethod
    def _guess_sep(name_lower: str) -> str:
        return "\t" if name_lower.endswith((".tsv", ".txt", ".tsv.gz", ".txt.gz")) else ","

    @staticmethod
    def _guess_compression(name_lower: str):
        return "gzip" if name_lower.endswith(".gz") else None

    #Read only headers
    @staticmethod
    def _read_header_columns(p: Path, sep: str, compression) -> List[str]:
        last_err = None
        for enc in ("utf-8", "latin1", "cp1252"):
            try:
                header_df = pd.read_csv(
                    p,
                    sep=sep,
                    compression=compression,
                    engine="python",
                    encoding=enc,
                    nrows=0,
                    on_bad_lines="skip",
                )
                cols = [str(c).strip() for c in header_df.columns.tolist()]
                return cols
            except Exception as e:
                last_err = e
        raise ValueError(f"Proteomics: failed header read for {p.name}. Last error: {repr(last_err)}")

    #column selection logic
    @staticmethod
    def _pick_id_candidates(cols: List[str]) -> Tuple[str, Optional[str]]:
        #Choosing best ID columns based on MaxQuant conventions.
        protein_col = None
        for c in ("Majority protein IDs", "Protein IDs"):
            if c in cols:
                protein_col = c
                break
        if protein_col is None:
            protein_col = cols[0] if cols else "Protein IDs"

        gene_col = "Gene names" if "Gene names" in cols else None
        return protein_col, gene_col

    def _detect_quant_columns_from_header(self, cols: List[str]) -> Tuple[List[str], str, callable]:
        '''Detect quant columns without reading the full file.
        Keeping original strings to match exact columns in usecols'''
        lfq_cols = [c for c in cols if c.startswith("LFQ intensity ")]
        intensity_cols = [c for c in cols if c.startswith("Intensity ")]
        peptides_cols = [c for c in cols if c.startswith("Peptides ")]

        reporter_cols = [c for c in cols if ("reporter" in c.lower() and "intensity" in c.lower())]
        abundance_cols = [c for c in cols if any(k in c.lower() for k in ("abundance", "area", "quantity", "signal"))]

        if self.prefer_lfq and lfq_cols:
            return lfq_cols, "LFQ intensity", (lambda c: str(c).replace("LFQ intensity ", "").strip())
        if intensity_cols:
            return intensity_cols, "Intensity", (lambda c: str(c).replace("Intensity ", "").strip())
        if reporter_cols:
            return reporter_cols, "Reporter intensity", (lambda c: str(c).strip())
        if peptides_cols:
            covid_like = [c for c in peptides_cols if ("COVID_" in c or "covid_" in c)]
            quant_cols = covid_like if covid_like else peptides_cols
            return quant_cols, "Peptides", (lambda c: str(c).replace("Peptides ", "").strip())
        if abundance_cols:
            return abundance_cols, "Abundance-like", (lambda c: str(c).strip())

        return [], "none", (lambda c: str(c).strip())

    # Robust read of selected columns (usecols)
    @staticmethod
    def _read_usecols(p: Path, sep: str, compression, usecols: List[str]) -> pd.DataFrame:
        #fast path
        try:
            df = pd.read_csv(
                p,
                sep=sep,
                compression=compression,
                engine="c",
                low_memory=True,
                usecols=usecols,
            )
            df.columns = df.columns.astype(str).str.strip()
            return df
        except Exception:
            pass
        
        #fallbacks
        last_err = None
        for enc in ("utf-8", "latin1", "cp1252"):
            try:
                df = pd.read_csv(
                    p,
                    sep=sep,
                    compression="infer",
                    engine="python",
                    encoding=enc,
                    usecols=usecols,
                    on_bad_lines="skip",
                )
                df.columns = df.columns.astype(str).str.strip()
                return df
            except Exception as e:
                last_err = e

        raise ValueError(f"Failed to read selected columns for {p.name}. Last error: {repr(last_err)}")

    #API
    def fit(self, file_path: str):
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(file_path)

        name = p.name.lower()

        bad_ext = (".raw", ".mzml", ".wiff", ".d", ".cdf", ".thermo", ".mzxml")
        if name.endswith(bad_ext):
            raise ValueError("Selected a raw mass-spec file. Use a processed quant table (MaxQuant proteinGroups.txt or a CSV/TSV/Parquet quant table).")

        # read once, parquet is already columnar and fast
        if name.endswith(".parquet") or self._is_parquet_file(p):
            df = pd.read_parquet(p)
            if df is None or df.shape[0] == 0 or df.shape[1] == 0:
                raise ValueError("parquet read produced an empty table.")
            df.columns = df.columns.astype(str).str.strip()

            protein_col, gene_col = self._pick_id_candidates(df.columns.tolist())
            quant_cols, quant_type, clean = self._detect_quant_columns_from_header(df.columns.tolist())
            if not quant_cols:
                preview = df.columns.astype(str).tolist()[:120]
                raise ValueError(
                    "Proteomics: no quantitative columns found. Expected columns like "
                    "'LFQ intensity ', 'Intensity ', 'Peptides ', reporter intensities, or abundance-like fields.\n"
                    f"Columns preview: {preview}"
                )

            work = df[[c for c in [protein_col, gene_col] if c is not None] + quant_cols].copy()
            return self._fit_from_df(work, protein_col, gene_col, quant_cols, quant_type, clean)

        # header then select columns then read only those columns
        sep = self._guess_sep(name)
        compression = self._guess_compression(name)

        cols = self._read_header_columns(p, sep=sep, compression=compression)
        if not cols:
            raise ValueError("Proteomics: header read produced no columns (file malformed?).")

        protein_col, gene_col = self._pick_id_candidates(cols)
        quant_cols, quant_type, clean = self._detect_quant_columns_from_header(cols)

        if not quant_cols:
            raise ValueError(
                "Proteomics: no quantitative columns found from header. "
                "Expected 'LFQ intensity ', 'Intensity ', 'Peptides ', reporter intensities, or abundance-like fields."
            )

        usecols = [protein_col] + ([gene_col] if gene_col else []) + quant_cols
        df = self._read_usecols(p, sep=sep, compression=compression, usecols=usecols)

        return self._fit_from_df(df, protein_col, gene_col, quant_cols, quant_type, clean)

    def _fit_from_df(
        self,
        df: pd.DataFrame,
        protein_col: str,
        gene_col: Optional[str],
        quant_cols: List[str],
        quant_type: str,
        clean,
    ):
        # Pull quant matrix
        X = df[quant_cols].copy()
        X.columns = [clean(c) for c in X.columns]
        X.columns = X.columns.astype(str).str.strip()

        # Deduplicate sample columns if present
        if pd.Index(X.columns).duplicated().any():
            X = X.loc[:, ~pd.Index(X.columns).duplicated()]

        # doing numeric coercion once
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.replace(0, np.nan)

        used_feature = "protein"

        # Gene aggregation if requested
        if self.feature_level == "gene" and gene_col is not None and gene_col in df.columns:
            g = df[gene_col].astype(str).str.strip()
            g = g.replace({"": np.nan, "nan": np.nan, "None": np.nan})
            g = g.map(lambda s: str(s).split(";")[0].strip() if pd.notna(s) else np.nan)

            frac_gene_present = float(g.notna().mean()) if len(g) else 0.0
            if frac_gene_present >= 0.10:
                X["__gene__"] = g.values
                X = X.dropna(subset=["__gene__"])
                X = X.set_index("__gene__")
                X.index.name = "gene_symbol"
                used_feature = "gene"
                X = X.groupby(level=0).median()

        if used_feature == "protein":
            pid = df[protein_col].astype(str).str.strip() if protein_col in df.columns else pd.Series([f"row{i}" for i in range(len(df))])
            pid = pid.replace({"": np.nan, "nan": np.nan, "None": np.nan})
            X.index = pid.values[: X.shape[0]]
            X.index.name = "protein_id"

        # Coverage filter
        keep = X.notna().mean(axis=1) >= float(self.min_nonzero_fraction)
        self.report["features_removed_low_coverage"] = int((~keep).sum())
        X = X.loc[keep]

        self._X_raw = X
        self.fitted = True

        self.report.update(
            {
                "protein_id_col": protein_col,
                "gene_col_used": (gene_col if used_feature == "gene" else None),
                "feature_level_used": used_feature,
                "quant_type": quant_type,
                "initial_samples_detected": int(len(quant_cols)),
                "final_features": int(X.shape[0]),
                "final_samples": int(X.shape[1]),
                "missing_fraction": float(X.isna().mean().mean()) if X.size else 1.0,
                "samples_preview": X.columns.astype(str).tolist()[:20],
                "feature_head": X.index.astype(str).tolist()[:10],
            }
        )
        return self

    def transform(self, log_transform: bool = True, zscore: bool = True) -> pd.DataFrame:
        if not self.fitted or self._X_raw is None:
            raise RuntimeError("Call fit() first")

        X = self._X_raw.copy()

        if log_transform:
            X = np.log2(X + 1.0)
            self.report["log_transform_applied"] = True
        else:
            self.report["log_transform_applied"] = False

        if zscore:
            mu = X.mean(axis=1, skipna=True)
            sd = X.std(axis=1, skipna=True).replace(0, 1.0)
            X = X.sub(mu, axis=0).div(sd, axis=0)
            X = X.replace([np.inf, -np.inf], np.nan)
            self.report["zscore_applied"] = True
        else:
            self.report["zscore_applied"] = False

        self._X_norm = X
        return X

    def fit_transform(self, file_path: str, **kwargs) -> pd.DataFrame:
        self.fit(file_path)
        return self.transform(**kwargs)

    def get_matrix(self, normalized: bool = False) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Must call fit() first")
        if normalized:
            if self._X_norm is None:
                raise RuntimeError("Call transform() first")
            return self._X_norm.copy()
        return self._X_raw.copy()

    def get_report(self) -> dict:
        if not self.fitted:
            raise RuntimeError("Must call fit() first")
        return self.report.copy()

    def save_report(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2)
