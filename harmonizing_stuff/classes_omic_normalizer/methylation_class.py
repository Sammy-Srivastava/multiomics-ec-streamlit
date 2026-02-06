import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

from harmonizing_stuff.classes_omic_normalizer.base import BaseNormalizer


class MethylationNormalizer(BaseNormalizer):

    def __init__(
        self,
        detection_p_threshold: float = 0.01,
        max_probe_failure: float = 0.10,
        max_sample_failure: float = 0.05,
        epsilon: float = 1e-6,
        output: str = "M",
        zscore: bool = True,
        na_frac_threshold: float = 0.10,
    ):
        super().__init__(method="methylation_qc")
        self.detection_p_threshold = detection_p_threshold
        self.max_probe_failure = max_probe_failure
        self.max_sample_failure = max_sample_failure
        self.epsilon = epsilon
        self.output = output
        self.zscore = zscore
        self.na_frac_threshold = na_frac_threshold

        self.fitted = False
        self.report: dict = {}
        self._beta_qc_final: pd.DataFrame | None = None
        self._beta: pd.DataFrame | None = None
        self._detection_p: pd.DataFrame | None = None
        self.max_probes: int | None = None

    #---helpers---
    
    def _clean_columns(self, cols):
        #cleans columns
        cleaned = [str(c).strip() for c in cols]
        counts = {}
        unique_cols = []
        for c in cleaned:
            if c in counts:
                counts[c] += 1
                unique_cols.append(f"{c}_{counts[c]}")
            else:
                counts[c] = 0
                unique_cols.append(c)
        return unique_cols

    def _sort_columns(self, cols):
        #sorts columns using suffix
        cols = [str(c).strip() for c in cols if str(c).strip()]

        def numeric_suffix_key(s: str):
            m = re.search(r"(\d+)", s)
            return int(m.group(1)) if m else None

        c_like = [c for c in cols if re.match(r"^C\d+", c)]
        nc_like = [c for c in cols if re.match(r"^NC\d+", c)]
        rest = [c for c in cols if c not in set(c_like) and c not in set(nc_like)]

        if c_like or nc_like:
            c_like_sorted = sorted(c_like, key=lambda x: numeric_suffix_key(x) if numeric_suffix_key(x) is not None else 10**12)
            nc_like_sorted = sorted(nc_like, key=lambda x: numeric_suffix_key(x) if numeric_suffix_key(x) is not None else 10**12)
            rest_sorted = sorted(rest)
            return c_like_sorted + nc_like_sorted + rest_sorted

        def fallback_key(x: str):
            n = numeric_suffix_key(x)
            return (0, n, x) if n is not None else (1, 10**12, x)

        return sorted(cols, key=fallback_key)

    def _read_geo_table(self, path: str) -> pd.DataFrame:
        # reads produced matrix
        p = Path(path)
        name = p.name.lower()

        if name.endswith(".parquet"):
            df = pd.read_parquet(p)
            if df is None or df.shape[0] == 0 or df.shape[1] == 0:
                raise ValueError(f"Methylation parquet read produced empty matrix shape={None if df is None else df.shape}. File: {path}")
            df.index = df.index.astype(str).str.strip().str.replace('"', "", regex=False)
            df.columns = df.columns.astype(str).str.strip().str.replace('"', "", regex=False)
            return df

        df = None
        last_err = None
        attempts = [
            dict(sep="\t", compression="infer", engine="python"),
            dict(sep="\t", compression="infer"),
            dict(sep=",", compression="infer"),
        ]
        for kw in attempts:
            try:
                df = pd.read_csv(path, index_col=0, **kw)
                break
            except Exception as e:
                last_err = e

        if df is None:
            raise ValueError(f"Could not read methylation file: {path}. Last error: {last_err}")

        if df.shape[0] == 0 or df.shape[1] == 0:
            raise ValueError(
                f"Methylation file parsed to empty matrix shape={df.shape}. "
                f"Likely wrong delimiter/format. File: {path}"
            )

        df.index = df.index.astype(str).str.strip().str.replace('"', "", regex=False)
        df.columns = df.columns.astype(str).str.strip().str.replace('"', "", regex=False)
        return df

    def _intensity_maps(self, cols: pd.Index):
        u_cols = [c for c in cols if re.search(r"unmethylated\s*signal", c, flags=re.I)]
        m_cols = [c for c in cols if re.search(r"methylated\s*signal", c, flags=re.I)]
        p_cols = [c for c in cols if re.search(r"detection\s*[\._ ]\s*pval|detection\s*pval|detection_p", c, flags=re.I)]

        def base_name(col: str) -> str:
            s = str(col)
            s = re.sub(r"unmethylated\s*signal", "", s, flags=re.I)
            s = re.sub(r"methylated\s*signal", "", s, flags=re.I)
            s = re.sub(r"detection\s*[\._ ]\s*pval", "", s, flags=re.I)
            s = re.sub(r"detection\s*pval", "", s, flags=re.I)
            s = re.sub(r"detection_p", "", s, flags=re.I)
            return s.strip().strip('"').strip()

        u_map = {base_name(c): c for c in u_cols}
        m_map = {base_name(c): c for c in m_cols}
        p_map = {base_name(c): c for c in p_cols}

        shared = sorted(set(u_map.keys()) & set(m_map.keys()))
        return shared, u_map, m_map, p_map

    def _signal_pvalue_pairs(self, cols: pd.Index) -> tuple[list[str], dict[str, str], dict[str, str]]:
        #detecting unusual wording and turning it into recognized ones
        sig_map = {}
        pval_map = {}

        for c in cols.astype(str):
            cl = c.lower().strip()
            if cl.endswith(".signal"):
                base = c[: -len(".signal")].strip()
                sig_map[base] = c
            elif cl.endswith(".pvalue"):
                base = c[: -len(".pvalue")].strip()
                pval_map[base] = c
            elif cl.endswith(".pval"):
                base = c[: -len(".pval")].strip()
                pval_map[base] = c

        shared = sorted(set(sig_map.keys()) & set(pval_map.keys()))
        return shared, sig_map, pval_map

    def _classify_columns(self, df: pd.DataFrame):
        # treating pvalues as detectionpvals
        beta_cols, detp_cols = [], []
        for col in df.columns:
            col_lower = str(col).lower()
            if (
                "detection_p" in col_lower
                or "detection p" in col_lower
                or "detp" in col_lower
                or col_lower.endswith(".pvalue")
                or col_lower.endswith(".pval")
                or re.search(r"\bpvalue\b", col_lower)
                or re.search(r"\bpval\b", col_lower)
            ):
                detp_cols.append(col)
            else:
                beta_cols.append(col)
        return beta_cols, detp_cols

    #---API---
    def load_file(self, path: str, require_detection_pval: bool = True):
        df = self._read_geo_table(path)
        cols = pd.Index(df.columns.astype(str))

        # Debug
        print("RAW methylation df.columns head:", df.columns[:10].tolist())

        # 1. Intensity format (unmethylated/methylated)
        has_u = cols.to_series().str.contains(r"unmethylated\s*signal", case=False, regex=True).any()
        has_m = cols.to_series().str.contains(r"methylated\s*signal", case=False, regex=True).any()

        if has_u and has_m:
            shared, u_map, m_map, p_map = self._intensity_maps(cols)
            if len(shared) == 0:
                raise ValueError(
                    "Methylation intensities detected, but no sample IDs had both"
                    "Unmethylated and Methylated signal columns."
                )

            beta_dict = {}
            detp_dict = {}

            for sid in shared:
                U = pd.to_numeric(df[u_map[sid]], errors="coerce")
                M = pd.to_numeric(df[m_map[sid]], errors="coerce")
                beta_dict[sid] = M / (M + U + self.epsilon)

                if sid in p_map:
                    detp_dict[sid] = pd.to_numeric(df[p_map[sid]], errors="coerce")

            beta = pd.DataFrame(beta_dict, index=df.index)
            beta.columns = self._sort_columns(self._clean_columns(beta.columns.tolist()))
            beta = beta.loc[:, beta.columns]
            self._beta = beta

            if detp_dict:
                detp = pd.DataFrame(detp_dict, index=df.index)
                detp.columns = self._clean_columns(detp.columns.tolist())
                detp = detp.reindex(columns=beta.columns)
                self._detection_p = detp
            else:
                if require_detection_pval:
                    raise ValueError("No Detection Pvalues found. Re-run with require_detection_pval=False.")
                self._detection_p = None

            self.report["input_format"] = "intensity_unmeth_meth"
            return self

        # pair sample signal and sample pvalues
        shared, sig_map, pval_map = self._signal_pvalue_pairs(cols)
        if len(shared) > 0:
            beta_dict = {}
            detp_dict = {}
            for sid in shared:
                beta_dict[sid] = pd.to_numeric(df[sig_map[sid]], errors="coerce")
                detp_dict[sid] = pd.to_numeric(df[pval_map[sid]], errors="coerce")

            beta = pd.DataFrame(beta_dict, index=df.index)
            detp = pd.DataFrame(detp_dict, index=df.index)

            beta.columns = self._sort_columns(self._clean_columns(beta.columns.tolist()))
            detp.columns = self._clean_columns(detp.columns.tolist())
            detp = detp.reindex(columns=beta.columns)

            self._beta = beta
            self._detection_p = detp

            self.report["input_format"] = "paired_signal_pvalue"
            return self

        beta_cols, detp_cols = self._classify_columns(df)

        if len(detp_cols) == 0:
            if require_detection_pval:
                raise ValueError("No detection p-value columns found.")
            beta_cols_clean = self._clean_columns(beta_cols)
            beta_map = dict(zip(beta_cols, beta_cols_clean))
            beta = df[beta_cols].rename(columns=beta_map)
            beta = beta[self._sort_columns(beta.columns.tolist())] if len(beta.columns) > 0 else beta
            self._beta = beta
            self._detection_p = None
            self.report["input_format"] = "beta_only_no_detp"
            return self

        beta_cols_clean = self._clean_columns(beta_cols)

        detp_cols_clean = self._clean_columns(
            [re.sub(r"(_Detection_Pval|Detection Pval|\.Detection\.Pval|\.pvalue|\.pval)$", "", str(c), flags=re.I) for c in detp_cols]
        )

        beta_map = dict(zip(beta_cols, beta_cols_clean))
        detp_map = dict(zip(detp_cols, detp_cols_clean))

        shared_samples = sorted(set(beta_cols_clean) & set(detp_cols_clean))
        if len(shared_samples) == 0:
            raise ValueError("No shared samples found between beta and detection p-value columns.")

        shared_samples = self._sort_columns(shared_samples)

        beta = df[beta_cols].rename(columns=beta_map).reindex(columns=shared_samples)
        detection_p = df[detp_cols].rename(columns=detp_map).reindex(columns=shared_samples)

        self._beta = beta
        self._detection_p = detection_p
        self.report["input_format"] = "beta_plus_detp_split"
        return self

    def fit(self):
        if self._beta is None:
            raise ValueError("Call load_file() before fit()")

        beta = self._beta.copy()

        if self.max_probes is not None and beta.shape[0] > self.max_probes:
            beta = beta.sample(n=self.max_probes, random_state=0)
            self.report["probe_downsampled_to"] = int(self.max_probes)

        self.report["initial_probes"] = int(beta.shape[0])
        self.report["initial_samples"] = int(beta.shape[1])

        # masking Detection P-values
        if self._detection_p is not None:
            detection_p = self._detection_p.copy()
            mask_fail = detection_p > self.detection_p_threshold
            beta = beta.mask(mask_fail)
            self.report["masked_values"] = int(mask_fail.sum().sum())
        else:
            self.report["masked_values"] = 0

        # Probe-level QC
        probe_keep = beta.isna().mean(axis=1) <= self.max_probe_failure
        beta = beta.loc[probe_keep]
        self.report["probes_removed"] = int((~probe_keep).sum())

        # Sample-level QC
        sample_keep = beta.isna().mean(axis=0) <= self.max_sample_failure
        beta = beta.loc[:, sample_keep]
        self.report["samples_removed"] = int((~sample_keep).sum())

        # Remove low-variance probes
        probe_var = beta.var(axis=1, skipna=True)
        beta = beta.loc[probe_var > 1e-5]

        # Mask extreme values
        beta = beta.mask(beta.abs() > 1e6) 

        # nan fraction filtering
        probe_keep2 = beta.isna().mean(axis=1) <= self.na_frac_threshold
        sample_keep2 = beta.isna().mean(axis=0) <= self.na_frac_threshold
        beta = beta.loc[probe_keep2, sample_keep2]

        # convert to M-values if values look like beta in [0,1]
        if str(self.output).lower() == "m":
            vals = beta.to_numpy(dtype=float, copy=False)
            finite = vals[np.isfinite(vals)]
            if finite.size > 0:
                q01 = float(np.quantile(finite, 0.01))
                q99 = float(np.quantile(finite, 0.99))
            else:
                q01, q99 = 0.0, 0.0

            self.report["pre_M_q01"] = q01
            self.report["pre_M_q99"] = q99

            looks_like_beta = (q01 >= -0.05) and (q99 <= 1.05)

            if looks_like_beta:
                beta = beta.clip(lower=self.epsilon, upper=1.0 - self.epsilon)
                beta = np.log2((beta + self.epsilon) / (1 - beta + self.epsilon))
                self.report["beta_to_M_converted"] = True
            else:
                # prevent wipeout from invalid conversion
                self.report["beta_to_M_converted"] = False
                self.report["beta_to_M_skip_reason"] = f"Values not in [0,1] (q01={q01:.4g}, q99={q99:.4g}); leaving as-is."

        # Z-score per probe
        if self.zscore:
            row_mean = beta.mean(axis=1, skipna=True)
            row_std = beta.std(axis=1, skipna=True).replace(0, 1.0)
            beta = beta.sub(row_mean, axis=0).div(row_std, axis=0)

        self._beta_qc_final = beta.copy()
        self.report["final_probes"] = int(beta.shape[0])
        self.report["final_samples"] = int(beta.shape[1])

        self.fitted = True
        return self

    def transform(self):
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        return self._beta_qc_final.copy()

    def fit_transform(self):
        self.fit()
        return self.transform()

    def save_matrix(self, path: str):
        if not self.fitted:
            raise RuntimeError("Call fit() first")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._beta_qc_final.to_csv(path)

    def print_report(self):
        print("QC Report:")
        for k, v in self.report.items():
            print(f"{k}: {v}")

    def save_report(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.report, f, indent=2)
