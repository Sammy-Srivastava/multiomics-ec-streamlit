from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import time
import pandas as pd
from typing import Dict, Optional, Callable


from harmonizing_stuff.classes_omic_normalizer.methylation_class import MethylationNormalizer
from harmonizing_stuff.classes_omic_normalizer.proteomic_class import ProteomicNormalizer
from harmonizing_stuff.classes_omic_normalizer.transcriptomic_class import TranscriptomicNormalizer


_CANON_WS = re.compile(r"\s+")


def _canonicalize_sample_id_basic(s: str) -> str:
    return _CANON_WS.sub("_", str(s).strip())


def _apply_sample_map(cols: pd.Index, sample_map: Optional[dict]) -> pd.Index:
    if not sample_map:
        return cols
    return pd.Index([sample_map.get(str(c), str(c)) for c in cols])


def _dedup_columns_median(X: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate sample columns by median."""
    cols = pd.Index(X.columns)
    if not cols.duplicated().any():
        return X
    # roupby on columns, take median across duplicate columns
    return X.groupby(by=X.columns, axis=1).median()


def _write_matrix(X: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        X.to_parquet(path, index=True, compression="snappy")
    else:
        X.to_csv(path, index=True)


def _cap_rows_by_variance_fast(X: pd.DataFrame, max_rows: int, max_samples_for_var: int = 60) -> pd.DataFrame:
    """
    Keep only top rows by variance (quick display / demo cap).
    Uses up to max_samples_for_var columns for variance estimate for speed.
    """
    if not max_rows or max_rows <= 0 or X.shape[0] <= max_rows:
        return X

    if X.shape[1] > max_samples_for_var:
        v = X.iloc[:, :max_samples_for_var].var(axis=1, skipna=True)
    else:
        v = X.var(axis=1, skipna=True)

    top = v.nlargest(max_rows).index
    return X.loc[top]


def _fingerprint_run(payload: dict) -> str:
    s = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


@dataclass
class MultiHarmonizeResult:
    out_dir: Path
    paths: Dict[str, str]
    report: Dict


def multiharmonize(
    in_paths: Dict[str, Path],
    out_dir: Path,
    *,
    sample_strategy: str = "union",               
    union_missing: str = "keep_nan",              
    out_format: str = "parquet",                  

    transcriptomics_series_matrix: Optional[Path] = None,  

    # Caps
    cap_M_probes: int = 20000,
    cap_T_genes: int = 8000,
    cap_P_features: int = 4000,

    # Caching
    reuse_if_same_inputs: bool = True,

    # ID mapping/canonicalization
    sample_map: Optional[dict] = None,
    canonicalize_fn: Optional[Callable[[str, pd.Index], pd.Index]] = None,
) -> MultiHarmonizeResult:
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if sample_strategy not in {"intersection", "union"}:
        raise ValueError("sample_strategy must be 'intersection' or 'union'")
    if out_format not in {"parquet", "csv"}:
        raise ValueError("out_format must be 'parquet' or 'csv'")
    if union_missing not in {"keep_nan", "fill_zero"}:
        raise ValueError("union_missing must be 'keep_nan' or 'fill_zero'")
    if not in_paths:
        raise ValueError("No valid omics provided in in_paths.")

    # ---Cache fingerprint---
    inputs_meta = []
    for k in sorted(in_paths.keys()):
        p = Path(in_paths[k])
        st = p.stat()
        inputs_meta.append((k, str(p.resolve()), int(st.st_size), int(st.st_mtime)))

    demo_caps = {"M": int(cap_M_probes), "T": int(cap_T_genes), "P": int(cap_P_features)}

    payload = dict(
        inputs=inputs_meta,
        sample_strategy=sample_strategy,
        union_missing=union_missing if sample_strategy == "union" else None,
        out_format=out_format,
        series_matrix=str(Path(transcriptomics_series_matrix).resolve()) if transcriptomics_series_matrix else "",
        demo_caps=demo_caps,
        has_sample_map=bool(sample_map),
        has_canonicalize_fn=bool(canonicalize_fn),
    )
    run_fp = _fingerprint_run(payload)

    fp_path = out_dir / "_run_fingerprint.txt"
    report_path = out_dir / "harmonize_report.json"

    if reuse_if_same_inputs and fp_path.exists() and report_path.exists():
        try:
            old_fp = fp_path.read_text(encoding="utf-8").strip()
            if old_fp == run_fp:
                rep = json.loads(report_path.read_text(encoding="utf-8"))
                ext = ".parquet" if out_format == "parquet" else ".csv"
                paths: Dict[str, str] = {}
                for omic in rep.get("omics_present", []):
                    p = out_dir / f"{omic}_harmonized{ext}"
                    if p.exists():
                        paths[omic] = str(p)
                if paths:
                    return MultiHarmonizeResult(out_dir=out_dir, paths=paths, report=rep)
        except Exception:
            pass

    # ---Helpers---
    def _canon_and_map(omic: str, X: pd.DataFrame) -> pd.DataFrame:
        if canonicalize_fn is not None:
            X.columns = canonicalize_fn(omic, pd.Index(X.columns))
        else:
            X.columns = pd.Index([_canonicalize_sample_id_basic(c) for c in X.columns])

        X.columns = _apply_sample_map(pd.Index(X.columns), sample_map)
        X = _dedup_columns_median(X)
        return X

    timings: Dict[str, float] = {}
    matrices: Dict[str, pd.DataFrame] = {}
    reports: Dict[str, Dict] = {}

    # ---Methylation---
    if "M" in in_paths:
        t = time.time()
        norm = MethylationNormalizer(zscore=True, output="M")
        norm.load_file(str(in_paths["M"]), require_detection_pval=True)
        norm.fit()
        Xm = norm.transform()

        Xm = _cap_rows_by_variance_fast(Xm, int(cap_M_probes), max_samples_for_var=60)
        Xm = _canon_and_map("M", Xm)

        matrices["M"] = Xm
        reports["M"] = norm.report
        timings["M_s"] = round(time.time() - t, 3)

    # ---Transcriptomics---
    if "T" in in_paths:
        t = time.time()
        norm = TranscriptomicNormalizer(scale=True, log1p=True)
        norm.load_files(
            str(in_paths["T"]),
            str(transcriptomics_series_matrix) if transcriptomics_series_matrix else None,
        )
        norm.fit()
        Xt = norm.transform()

        Xt = _cap_rows_by_variance_fast(Xt, int(cap_T_genes), max_samples_for_var=60)
        Xt = _canon_and_map("T", Xt)

        matrices["T"] = Xt
        reports["T"] = norm.report
        timings["T_s"] = round(time.time() - t, 3)

        md = norm.get_metadata()
        if md is not None:
            md.to_csv(out_dir / "T_metadata.csv", index=False)

    # ---Proteomics---
    if "P" in in_paths:
        t = time.time()
        norm = ProteomicNormalizer(prefer_lfq=True, min_nonzero_fraction=0.2, feature_level="gene")
        norm.fit(str(in_paths["P"]))
        Xp = norm.transform(log_transform=True, zscore=True)

        Xp = _cap_rows_by_variance_fast(Xp, int(cap_P_features), max_samples_for_var=60)
        Xp = _canon_and_map("P", Xp)

        matrices["P"] = Xp
        reports["P"] = norm.get_report()
        timings["P_s"] = round(time.time() - t, 3)

    if not matrices:
        raise ValueError("No valid omics produced a matrix.")

    # ---Align samples across omics---
    t_align = time.time()
    sample_sets = {k: set(v.columns.tolist()) for k, v in matrices.items()}

    aligned_mats: Dict[str, pd.DataFrame] = {}
    aligned_samples: Optional[list[str]] = None

    if len(matrices) == 1:
        aligned_mats = matrices
        aligned_samples = list(next(iter(matrices.values())).columns)

    else:
        if sample_strategy == "intersection":
            it = iter(sample_sets.values())
            aligned_set = set(next(it))
            for s in it:
                aligned_set.intersection_update(s)

            aligned_samples = sorted(aligned_set)
            if not aligned_samples:
                preview = {k: sorted(list(v))[:20] for k, v in sample_sets.items()}
                raise ValueError(f"No overlapping samples across selected omics. Sample previews: {preview}")

            for omic, X in matrices.items():
                aligned_mats[omic] = X.reindex(columns=aligned_samples)

        else:
            aligned_set = set()
            for s in sample_sets.values():
                aligned_set.update(s)

            aligned_samples = sorted(aligned_set)

            for omic, X in matrices.items():
                Xr = X.reindex(columns=aligned_samples)
                if union_missing == "fill_zero":
                    Xr = Xr.fillna(0.0)
                aligned_mats[omic] = Xr

    timings["align_s"] = round(time.time() - t_align, 3)

    # ---Write outputs and reports---
    t_write = time.time()
    ext = ".parquet" if out_format == "parquet" else ".csv"
    paths: Dict[str, str] = {}

    for omic, X in aligned_mats.items():
        out_path = out_dir / f"{omic}_harmonized{ext}"
        _write_matrix(X, out_path)
        paths[omic] = str(out_path)

        rep_path = out_dir / f"{omic}_report.json"
        rep_path.write_text(json.dumps(reports.get(omic, {}), indent=2), encoding="utf-8")

    timings["write_s"] = round(time.time() - t_write, 3)

    report = {
        "run_fingerprint": run_fp,
        "demo_caps": demo_caps,
        "sample_strategy": sample_strategy,
        "union_missing": union_missing if sample_strategy == "union" else None,
        "omics_present": sorted(list(aligned_mats.keys())),
        "aligned_samples_n": (len(aligned_samples) if aligned_samples is not None else None),
        "aligned_samples_preview": (aligned_samples[:25] if aligned_samples is not None else None),
        "per_omic_shapes": {
            k: {"features": int(v.shape[0]), "samples": int(v.shape[1])} for k, v in aligned_mats.items()
        },
        "sample_sets_preview": {k: sorted(list(sample_sets[k]))[:25] for k in sample_sets},
        "timings_seconds": timings,
        "total_seconds": round(time.time() - t0, 3),
    }

    (out_dir / "harmonize_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fp_path.write_text(run_fp, encoding="utf-8")

    return MultiHarmonizeResult(out_dir=out_dir, paths=paths, report=report)