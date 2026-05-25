from __future__ import annotations

import math

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.domain.services.phase_b_tier_policy import default_phase_b_policy

_QUANTILE_COLUMN_BY_LEVEL = {
    0.1: "quantile_p10_post_guardrail",
    0.5: "quantile_p50_post_guardrail",
    0.9: "quantile_p90_post_guardrail",
    10: "quantile_p10_post_guardrail",
    50: "quantile_p50_post_guardrail",
    90: "quantile_p90_post_guardrail",
}
_COVERAGE_COLUMN_BY_LEVEL = {
    0.1: "coverage_q10",
    0.5: "coverage_q50",
    0.9: "coverage_q90",
    10: "coverage_q10",
    50: "coverage_q50",
    90: "coverage_q90",
}


def _normalize_level(level: float) -> float:
    value = float(level)
    if value > 1:
        value = value / 100.0
    if value not in {0.1, 0.5, 0.9}:
        raise ValueError(f"unsupported quantile level: {level}")
    return value


def _finite_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric[np.isfinite(numeric)]
    return float(numeric.mean()) if len(numeric) else math.nan


def compute_marginal_coverage(
    fact_oos_filtered: pd.DataFrame,
    run_ids: set[str],
    quantile_levels: Iterable[float],
) -> pd.DataFrame:
    levels = [_normalize_level(level) for level in quantile_levels]
    if not levels:
        raise ValueError("quantile_levels must not be empty")
    required = {"run_id", "split", "horizon", "y_true"}
    required.update(_QUANTILE_COLUMN_BY_LEVEL[level] for level in levels)
    missing = sorted(required - set(fact_oos_filtered.columns))
    if missing:
        raise ValueError(f"fact_oos_filtered missing required columns: {missing}")

    df = fact_oos_filtered.copy()
    df["run_id"] = df["run_id"].astype(str)
    df = df[df["run_id"].isin({str(r) for r in run_ids})].copy()
    if df.empty:
        return pd.DataFrame(
            columns=[
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
            ]
        )

    rows: list[dict[str, object]] = []
    group_cols = ["run_id", "split", "horizon"]
    for keys, group in df.groupby(group_cols, dropna=False):
        g = group.copy()
        for col in ["y_true", *[_QUANTILE_COLUMN_BY_LEVEL[level] for level in levels]]:
            g[col] = pd.to_numeric(g[col], errors="coerce")
        subset_cols = ["y_true", *[_QUANTILE_COLUMN_BY_LEVEL[level] for level in levels]]
        g = g.dropna(subset=subset_cols)

        row: dict[str, object] = {
            "run_id": str(keys[0]),
            "split": str(keys[1]),
            "horizon": int(keys[2]) if pd.notna(keys[2]) else None,
            "n_obs": int(len(g)),
        }
        for level in [0.1, 0.5, 0.9]:
            out_col = _COVERAGE_COLUMN_BY_LEVEL[level]
            if level not in levels or g.empty:
                row[out_col] = math.nan
                continue
            q_col = _QUANTILE_COLUMN_BY_LEVEL[level]
            row[out_col] = float((g["y_true"] <= g[q_col]).mean())

        q10_col = _QUANTILE_COLUMN_BY_LEVEL[0.1]
        q90_col = _QUANTILE_COLUMN_BY_LEVEL[0.9]
        if {0.1, 0.9}.issubset(set(levels)) and not g.empty:
            picp = float(((g[q10_col] <= g["y_true"]) & (g["y_true"] <= g[q90_col])).mean())
            mpiw = _finite_mean(g[q90_col] - g[q10_col])
        else:
            picp = math.nan
            mpiw = math.nan
        row["picp_q10_q90"] = picp
        row["coverage_error_picp"] = (
            float(picp - default_phase_b_policy().nominal_picp_q10_q90)
            if math.isfinite(picp)
            else math.nan
        )
        row["mpiw"] = mpiw
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

