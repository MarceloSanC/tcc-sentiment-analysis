from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class _SidecarSchema:
    required_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...] = ()


_MARGINAL_COVERAGE_SCHEMA = _SidecarSchema(
    required_columns=(
        "run_id",
        "split",
        "horizon",
        "coverage_q10",
        "coverage_q50",
        "coverage_q90",
        "picp_q10_q90",
        "coverage_error_picp",
        "mpiw",
        "n_obs",
    ),
    numeric_columns=(
        "horizon",
        "coverage_q10",
        "coverage_q50",
        "coverage_q90",
        "picp_q10_q90",
        "coverage_error_picp",
        "mpiw",
        "n_obs",
    ),
)

_DM_SCHEMA = _SidecarSchema(
    required_columns=(
        "horizon",
        "baseline_model_version",
        "dm_stat",
        "pvalue_two_sided",
        "pvalue_one_sided_less",
        "pvalue_adj_holm",
        "n_obs_effective",
        "hac_lag_used",
        "hln_applied",
        "direction",
        "dedup_rule",
        "seed_aggregation",
    ),
    numeric_columns=(
        "horizon",
        "dm_stat",
        "pvalue_two_sided",
        "pvalue_one_sided_less",
        "pvalue_adj_holm",
        "n_obs_effective",
        "hac_lag_used",
    ),
)

_DM_18_SCHEMA = _SidecarSchema(
    required_columns=("fold_name", "analysis_role", *_DM_SCHEMA.required_columns),
    numeric_columns=_DM_SCHEMA.numeric_columns,
)

_DELTA_PINBALL_SCHEMA = _SidecarSchema(
    required_columns=(
        "horizon",
        "baseline_model_version",
        "mean_pinball_tft_post_guardrail",
        "mean_pinball_baseline_post_guardrail",
        "delta_mean_pinball_rel",
    ),
    numeric_columns=(
        "horizon",
        "mean_pinball_tft_post_guardrail",
        "mean_pinball_baseline_post_guardrail",
        "delta_mean_pinball_rel",
    ),
)

_TIER_VERDICT_SCHEMA = _SidecarSchema(
    required_columns=(
        "hypothesis",
        "horizon",
        "tier",
        "criteria_passed_dict",
        "numerical_inputs",
        "justification",
    ),
    numeric_columns=("horizon",),
)


class ParquetPhaseBTierSidecarWriter:
    def __init__(self, output_dir: str | Path = "data/analytics/reports/phase_b") -> None:
        self.output_dir = Path(output_dir)

    @staticmethod
    def _safe_cohort_id(cohort_id: str) -> str:
        text = str(cohort_id).strip()
        if not text:
            raise ValueError("cohort_id must not be empty")
        if "/" in text or "\\" in text:
            raise ValueError("cohort_id must not contain path separators")
        return text

    @staticmethod
    def _validate(df: pd.DataFrame, schema: _SidecarSchema, *, name: str) -> pd.DataFrame:
        missing = sorted(set(schema.required_columns) - set(df.columns))
        if missing:
            raise ValueError(f"{name} sidecar missing required columns: {missing}")
        out = df.copy()
        for column in schema.numeric_columns:
            converted = pd.to_numeric(out[column], errors="coerce")
            invalid = out[column].notna() & converted.isna()
            if invalid.any():
                raise ValueError(f"{name} sidecar column '{column}' must be numeric")
            out[column] = converted
        return out

    def _path(self, cohort_id: str, filename: str) -> Path:
        safe = self._safe_cohort_id(cohort_id)
        return self.output_dir / f"cohort={safe}" / filename

    def _write(
        self,
        df: pd.DataFrame,
        cohort_id: str,
        filename: str,
        schema: _SidecarSchema,
    ) -> Path:
        path = self._path(cohort_id, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        validated = self._validate(df, schema, name=filename)
        validated.to_parquet(path, index=False)
        return path

    def write_marginal_coverage(self, df: pd.DataFrame, cohort_id: str) -> Path:
        return self._write(
            df,
            cohort_id,
            "phase_b_marginal_coverage.parquet",
            _MARGINAL_COVERAGE_SCHEMA,
        )

    def write_dm_family_6(self, df: pd.DataFrame, cohort_id: str) -> Path:
        return self._write(
            df,
            cohort_id,
            "phase_b_dm_family_6.parquet",
            _DM_SCHEMA,
        )

    def write_dm_family_18_sensitivity(self, df: pd.DataFrame, cohort_id: str) -> Path:
        return self._write(
            df,
            cohort_id,
            "phase_b_dm_family_18_sensitivity.parquet",
            _DM_18_SCHEMA,
        )

    def write_delta_pinball(self, df: pd.DataFrame, cohort_id: str) -> Path:
        return self._write(
            df,
            cohort_id,
            "phase_b_delta_pinball.parquet",
            _DELTA_PINBALL_SCHEMA,
        )

    def write_tier_verdict(self, df: pd.DataFrame, cohort_id: str) -> Path:
        return self._write(
            df,
            cohort_id,
            "phase_b_tier_verdict.parquet",
            _TIER_VERDICT_SCHEMA,
        )

