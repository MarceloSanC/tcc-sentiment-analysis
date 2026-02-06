from __future__ import annotations

from pathlib import Path
from typing import Any
from datetime import datetime, timezone
import logging
import json

import numpy as np
import pandas as pd


class DataQualityReporter:
    """
    Utility to produce data-quality reports for auditing.
    Output is a single JSON object per run.
    """

    @staticmethod
    def _as_date_series(df: pd.DataFrame, date_col: str) -> pd.Series:
        series = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        return series.dt.normalize()

    @staticmethod
    def generate_report(
        df: pd.DataFrame,
        *,
        date_col: str | None = None,
        key_cols: list[str] | None = None,
        expected_dtypes: dict[str, str] | None = None,
        value_ranges: dict[str, tuple[float, float]] | None = None,
        validation_rules: list[dict[str, Any]] | None = None,
        comparison_rules: list[dict[str, Any]] | None = None,
        business_days: bool = True,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "columns": list(df.columns),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
            "nulls": df.isna().sum().to_dict(),
        }

        report["rows_with_nulls"] = int(df.isna().any(axis=1).sum())

        if expected_dtypes:
            mismatches = {}
            for col, expected in expected_dtypes.items():
                if col in df.columns:
                    actual = str(df.dtypes[col])
                    if actual != expected:
                        mismatches[col] = {"expected": expected, "actual": actual}
            report["dtype_mismatches"] = mismatches

        invalid_rows_mask = None
        if key_cols:
            missing_keys = [c for c in key_cols if c not in df.columns]
            report["key_cols"] = key_cols
            report["missing_key_cols"] = missing_keys
            if not missing_keys:
                report["duplicate_keys"] = int(df.duplicated(subset=key_cols).sum())
                invalid_keys = df[key_cols].isna().any(axis=1)
                report["invalid_key_rows"] = int(invalid_keys.sum())
                invalid_rows_mask = invalid_keys if invalid_rows_mask is None else (invalid_rows_mask | invalid_keys)
        else:
            report["duplicate_rows"] = int(df.duplicated().sum())

        numeric = df.select_dtypes(include=["number"])
        if not numeric.empty:
            report["zeros"] = (numeric == 0).sum().to_dict()
            report["rows_with_zeros"] = int((numeric == 0).any(axis=1).sum())
            numeric_values = numeric.apply(pd.to_numeric, errors="coerce")
            invalid_inf = ~np.isfinite(numeric_values)
            report["invalid_numeric_inf"] = invalid_inf.sum().to_dict()
            stats = numeric.describe().T
            report["numeric_stats"] = stats.to_dict(orient="index")

        if value_ranges:
            out_of_range = {}
            for col, (low, high) in value_ranges.items():
                if col in df.columns:
                    series = pd.to_numeric(df[col], errors="coerce")
                    mask = (series < low) | (series > high)
                    out_of_range[col] = int(mask.sum())
                    invalid_rows_mask = mask if invalid_rows_mask is None else (invalid_rows_mask | mask)
            report["out_of_range"] = out_of_range

        if validation_rules:
            rule_results: dict[str, int] = {}
            missing_rule_columns: dict[str, str] = {}
            for rule in validation_rules:
                name = str(rule.get("name", "unnamed_rule"))
                col = rule.get("column")
                op = rule.get("op")
                if not col or col not in df.columns:
                    missing_rule_columns[name] = str(col)
                    continue
                series = df[col]
                mask = None
                if op == "not_null":
                    mask = series.isna()
                elif op == "nonnegative":
                    values = pd.to_numeric(series, errors="coerce")
                    mask = values < 0
                elif op == "between":
                    low = rule.get("min")
                    high = rule.get("max")
                    values = pd.to_numeric(series, errors="coerce")
                    mask = (values < low) | (values > high)
                elif op == "in":
                    values = rule.get("values", [])
                    mask = ~series.isin(values)
                if mask is not None:
                    rule_results[name] = int(mask.sum())
                    invalid_rows_mask = mask if invalid_rows_mask is None else (invalid_rows_mask | mask)
            report["validation_rules"] = rule_results
            if missing_rule_columns:
                report["validation_missing_columns"] = missing_rule_columns

        if comparison_rules:
            comp_results: dict[str, int] = {}
            missing_comp_cols: dict[str, list[str]] = {}
            for rule in comparison_rules:
                name = str(rule.get("name", "unnamed_comparison"))
                left = rule.get("left")
                right = rule.get("right")
                op = rule.get("op")
                if left not in df.columns or right not in df.columns:
                    missing_comp_cols[name] = [str(left), str(right)]
                    continue
                left_vals = pd.to_numeric(df[left], errors="coerce")
                right_vals = pd.to_numeric(df[right], errors="coerce")
                if op == ">=":
                    mask = left_vals < right_vals
                elif op == "<=":
                    mask = left_vals > right_vals
                elif op == ">":
                    mask = left_vals <= right_vals
                elif op == "<":
                    mask = left_vals >= right_vals
                else:
                    continue
                mask = mask | left_vals.isna() | right_vals.isna()
                comp_results[name] = int(mask.sum())
                invalid_rows_mask = mask if invalid_rows_mask is None else (invalid_rows_mask | mask)
            report["comparison_rules"] = comp_results
            if missing_comp_cols:
                report["comparison_missing_columns"] = missing_comp_cols

        if date_col and date_col in df.columns:
            dates = DataQualityReporter._as_date_series(df, date_col)
            invalid_dates = dates.isna()
            report["date_col"] = date_col
            report["date_min"] = dates.min().date().isoformat() if not dates.isna().all() else None
            report["date_max"] = dates.max().date().isoformat() if not dates.isna().all() else None
            report["invalid_date_rows"] = int(invalid_dates.sum())
            invalid_rows_mask = invalid_dates if invalid_rows_mask is None else (invalid_rows_mask | invalid_dates)

            day_counts = dates.value_counts(dropna=True).sort_index()
            report["per_day_counts"] = {d.date().isoformat(): int(v) for d, v in day_counts.items()}

            dow_counts = dates.dt.dayofweek.value_counts(dropna=True).sort_index()
            report["per_day_of_week_counts"] = {int(k): int(v) for k, v in dow_counts.items()}

            year_counts = dates.dt.year.value_counts(dropna=True).sort_index()
            report["per_year_counts"] = {int(k): int(v) for k, v in year_counts.items()}

            if not dates.isna().all():
                min_day = dates.min().date()
                max_day = dates.max().date()
                if business_days:
                    expected = pd.bdate_range(min_day, max_day, tz="UTC")
                else:
                    expected = pd.date_range(min_day, max_day, freq="D", tz="UTC")
                missing = expected.difference(dates.dropna().unique())
                report["missing_days_count"] = int(len(missing))
                report["missing_days_all"] = [d.date().isoformat() for d in missing]

                unique_sorted = pd.Series(sorted(dates.dropna().unique()))
                gaps = unique_sorted.diff().dt.days
                gap_mask = gaps > 1
                report["gaps_count"] = int(gap_mask.sum())
                if gap_mask.any():
                    gap_sizes = gaps[gap_mask].astype(int).tolist()
                    report["gaps_sizes_all"] = gap_sizes

        if invalid_rows_mask is not None:
            report["invalid_rows_total"] = int(invalid_rows_mask.sum())

        return report

    @staticmethod
    def write_report(
        report: dict[str, Any],
        output_dir: Path,
        *,
        prefix: str,
    ) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"{prefix}_report_{ts}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def report_exists(output_dir: Path, prefix: str) -> bool:
        return any(output_dir.glob(f"{prefix}_report_*.json"))

    @staticmethod
    def report_from_parquet(
        parquet_path: Path,
        *,
        prefix: str,
        date_col: str | None = None,
        key_cols: list[str] | None = None,
        expected_dtypes: dict[str, str] | None = None,
        value_ranges: dict[str, tuple[float, float]] | None = None,
        validation_rules: list[dict[str, Any]] | None = None,
        comparison_rules: list[dict[str, Any]] | None = None,
        business_days: bool = True,
    ) -> Path:
        logger = logging.getLogger(__name__)
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as exc:
            logger.warning(
                "Data quality report skipped (failed to read parquet)",
                extra={"path": str(parquet_path.resolve()), "error": str(exc)},
            )
            return parquet_path
        report = DataQualityReporter.generate_report(
            df,
            date_col=date_col,
            key_cols=key_cols,
            expected_dtypes=expected_dtypes,
            value_ranges=value_ranges,
            validation_rules=validation_rules,
            comparison_rules=comparison_rules,
            business_days=business_days,
        )
        report["source_path"] = str(parquet_path.resolve())
        report["file_size_bytes"] = parquet_path.stat().st_size
        report["file_mtime_utc"] = (
            datetime.fromtimestamp(parquet_path.stat().st_mtime, tz=timezone.utc).isoformat()
        )
        report["file_hash_sha256"] = DataQualityReporter.file_hash(parquet_path)
        report["generated_at"] = datetime.now(timezone.utc).isoformat()
        return DataQualityReporter.write_report(
            report,
            parquet_path.parent / "reports",
            prefix=prefix,
        )

    @staticmethod
    def file_hash(path: Path) -> str:
        import hashlib

        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
