"""Quantile-cluster gold builders.

4 builders covering the OOS quantile contract (raw + post_guardrail),
guardrail audit, degeneracy report, and config-level aggregation of
`gold_prediction_metrics_by_run_split_horizon`. Migrated verbatim from
`RefreshAnalyticsStoreUseCase._build_gold_*` (R-22.1); the
`PredictionMetricsByConfigGoldBuilder` is Tier 2 and consumes the
run/split/horizon DataFrame via `ctx.gold_outputs`.
"""
from __future__ import annotations

import logging

from typing import ClassVar

import numpy as np
import pandas as pd

from src.domain.services.gold_builders.base import (
    BuildContext,
    GoldBuilder,
    GoldBuilderSnapshot,
)
from src.domain.services.quantile_contract_analyzer import (
    QuantileContractAnalyzer,
    QuantileDegeneracyThresholds,
)

logger = logging.getLogger(__name__)


RAW_QUANTILE_COLUMNS: tuple[str, str, str] = ("quantile_p10", "quantile_p50", "quantile_p90")
POST_GUARDRAIL_QUANTILE_COLUMNS: tuple[str, str, str] = (
    "quantile_p10_post_guardrail",
    "quantile_p50_post_guardrail",
    "quantile_p90_post_guardrail",
)
PROBABILISTIC_METRIC_COLUMNS: tuple[str, ...] = (
    "pinball_q10",
    "pinball_q50",
    "pinball_q90",
    "mean_pinball",
    "picp",
    "mpiw",
    "pred_interval_width",
    "coverage_error",
    "confidence_calibrated",
    "prob_up",
    "prob_down",
)
POINT_METRIC_COLUMNS: tuple[str, ...] = (
    "rmse",
    "mae",
    "mape",
    "smape",
    "directional_accuracy",
    "bias",
)


def prediction_metric_columns() -> list[str]:
    return [
        *POINT_METRIC_COLUMNS,
        *(f"{m}_raw" for m in PROBABILISTIC_METRIC_COLUMNS),
        *(f"{m}_post_guardrail" for m in PROBABILISTIC_METRIC_COLUMNS),
    ]


def _safe_iqr(series: pd.Series) -> float:
    # Cat C filter produces all-NaN groups for point/degenerate runs.
    # `np.nanpercentile` warns on all-NaN — drop first to silence the
    # warning at the source while preserving the (NaN) result.
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return float("nan")
    return float(np.percentile(cleaned, 75) - np.percentile(cleaned, 25))


def _pinball_loss(y_true: pd.Series, y_pred_q: pd.Series, quantile: float) -> pd.Series:
    q = float(quantile)
    diff = y_true - y_pred_q
    return np.maximum(q * diff, (q - 1.0) * diff)


def _prob_up_from_quantiles(q10: pd.Series, q50: pd.Series, q90: pd.Series) -> pd.Series:
    # Piecewise-linear CDF using q10/q90 anchors: CDF(q10)=0.1, CDF(q90)=0.9.
    width = (q90 - q10).astype(float)
    safe_width = width.where(width.abs() > 1e-12, np.nan)

    cdf0 = 0.1 + 0.8 * ((0.0 - q10) / safe_width)
    cdf0 = cdf0.clip(lower=0.1, upper=0.9)

    cdf0 = np.where(q10 > 0.0, 0.0, cdf0)
    cdf0 = np.where(q90 < 0.0, 1.0, cdf0)

    fallback = np.where(q50 > 0.0, 1.0, np.where(q50 < 0.0, 0.0, 0.5))
    cdf0 = np.where(np.isnan(cdf0), fallback, cdf0)

    return pd.Series(1.0 - cdf0, index=q10.index, dtype="float64")


def _build_metrics_single_contract(
    dim_run: pd.DataFrame,
    fact_oos_predictions: pd.DataFrame,
    fact_config: pd.DataFrame | None,
    *,
    quantile_columns: tuple[str, str, str],
) -> pd.DataFrame:
    """Single-contract per-row → grouped probabilistic + point metrics.

    Called twice by `PredictionMetricsByRunSplitHorizonGoldBuilder`
    (raw + post_guardrail) and by `QuantileGuardrailAuditGoldBuilder`
    (before + after) to produce the deltas. Body lifted verbatim from
    the monolith's
    `_build_gold_prediction_metrics_by_run_split_horizon_single_contract`.
    """
    if fact_oos_predictions.empty:
        return pd.DataFrame()

    q10_col, q50_col, q90_col = quantile_columns
    df = fact_oos_predictions.copy()

    required = [
        "run_id",
        "split",
        "horizon",
        "y_true",
        "y_pred",
        q10_col,
        q50_col,
        q90_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return pd.DataFrame()

    for c in ["horizon", "y_true", "y_pred", q10_col, q50_col, q90_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    valid = df.dropna(subset=["horizon", "y_true", "y_pred", q10_col, q50_col, q90_col]).copy()
    if valid.empty:
        return pd.DataFrame()

    # Stage 9: Cat C filter — probabilistic eligibility per row. Mode-gate
    # (quantile vs point) via fact_config + anti-degeneracy on raw quantiles
    # (detect collapse emitted by the model before the guardrail masks it).
    # Point runs receive no filter.
    if (
        fact_config is not None
        and not fact_config.empty
        and "prediction_mode" in fact_config.columns
        and "run_id" in fact_config.columns
    ):
        valid = valid.merge(
            fact_config[["run_id", "prediction_mode"]].drop_duplicates("run_id"),
            on="run_id",
            how="left",
        )
    else:
        valid["prediction_mode"] = None

    q10_raw_col, _, q90_raw_col = RAW_QUANTILE_COLUMNS
    if q10_raw_col in valid.columns and q90_raw_col in valid.columns:
        p10_raw = pd.to_numeric(valid[q10_raw_col], errors="coerce")
        p90_raw = pd.to_numeric(valid[q90_raw_col], errors="coerce")
        is_non_degenerate = (p10_raw != p90_raw) & p10_raw.notna() & p90_raw.notna()
    else:
        is_non_degenerate = pd.Series(False, index=valid.index)
    is_quantile_mode = valid["prediction_mode"].astype(str).str.lower() == "quantile"
    prob_eligible_mask = (is_quantile_mode & is_non_degenerate).astype(bool)
    valid["_prob_eligible"] = prob_eligible_mask

    valid["horizon"] = valid["horizon"].astype(int)
    valid["error"] = valid["y_pred"] - valid["y_true"]
    valid["abs_error"] = valid["error"].abs()
    valid["sq_error"] = valid["error"] ** 2
    valid["pred_interval_width"] = valid[q90_col] - valid[q10_col]
    valid["covered_80"] = (
        (valid["y_true"] >= valid[q10_col]) & (valid["y_true"] <= valid[q90_col])
    ).astype(float)

    denom = valid["y_true"].replace(0.0, np.nan)
    valid["ape"] = valid["abs_error"] / denom.abs()
    smape_denom = valid["y_true"].abs() + valid["y_pred"].abs()
    valid["smape_row"] = 2.0 * valid["abs_error"] / smape_denom.replace(0.0, np.nan)

    valid["da_row"] = (np.sign(valid["y_true"]) == np.sign(valid["y_pred"])).astype(float)
    valid["pinball_q10_row"] = _pinball_loss(valid["y_true"], valid[q10_col], 0.1)
    valid["pinball_q50_row"] = _pinball_loss(valid["y_true"], valid[q50_col], 0.5)
    valid["pinball_q90_row"] = _pinball_loss(valid["y_true"], valid[q90_col], 0.9)
    valid["pinball_mean_row"] = (
        valid["pinball_q10_row"] + valid["pinball_q50_row"] + valid["pinball_q90_row"]
    ) / 3.0
    valid["prob_up_row"] = _prob_up_from_quantiles(
        valid[q10_col], valid[q50_col], valid[q90_col]
    )

    # Per-row mask applies ONLY to probabilistic columns. Point columns
    # (sq_error, abs_error, ape, smape_row, da_row, error) stay for all
    # runs (point + quantile + degenerate).
    prob_row_cols = (
        "pinball_q10_row",
        "pinball_q50_row",
        "pinball_q90_row",
        "pinball_mean_row",
        "prob_up_row",
        "covered_80",
        "pred_interval_width",
    )
    for col in prob_row_cols:
        if col in valid.columns:
            valid.loc[~prob_eligible_mask, col] = np.nan

    group_cols = [
        c
        for c in [
            "run_id",
            "asset",
            "feature_set_name",
            "config_signature",
            "split",
            "fold",
            "seed",
            "horizon",
        ]
        if c in valid.columns
    ]

    agg = (
        valid.groupby(group_cols, dropna=False)
        .agg(
            n_samples=("y_true", "count"),
            n_probabilistic_samples=("_prob_eligible", "sum"),
            rmse=("sq_error", lambda x: float(np.sqrt(np.mean(x)))),
            mae=("abs_error", "mean"),
            mape=("ape", "mean"),
            smape=("smape_row", "mean"),
            directional_accuracy=("da_row", "mean"),
            bias=("error", "mean"),
            pinball_q10=("pinball_q10_row", "mean"),
            pinball_q50=("pinball_q50_row", "mean"),
            pinball_q90=("pinball_q90_row", "mean"),
            mean_pinball=("pinball_mean_row", "mean"),
            picp=("covered_80", "mean"),
            mpiw=("pred_interval_width", "mean"),
            pred_interval_width=("pred_interval_width", "mean"),
            prob_up=("prob_up_row", "mean"),
        )
        .reset_index()
    )

    # Stage 9.2: is_quantile_genuine is per-group (run_id, split, horizon),
    # computed dynamically (NOT persisted in silver). Permissive definition:
    # True iff the group contributed >=1 eligible row.
    agg["n_probabilistic_samples"] = (
        pd.to_numeric(agg["n_probabilistic_samples"], errors="coerce").fillna(0).astype(int)
    )
    agg["is_quantile_genuine"] = agg["n_probabilistic_samples"] > 0

    agg["coverage_nominal"] = 0.80
    agg["coverage_error"] = agg["picp"] - agg["coverage_nominal"]
    agg["prob_down"] = 1.0 - agg["prob_up"]
    calibration_term = (
        1.0 - (agg["coverage_error"].abs() / agg["coverage_nominal"])
    ).clip(lower=0.0, upper=1.0)
    width_term = 1.0 / (1.0 + agg["pred_interval_width"].clip(lower=0.0))
    agg["confidence_calibrated"] = calibration_term * width_term

    if not dim_run.empty and "run_id" in dim_run.columns:
        keep = [
            c
            for c in [
                "run_id",
                "model_version",
                "feature_set_hash",
                "parent_sweep_id",
                "trial_number",
                "status",
            ]
            if c in dim_run.columns
        ]
        if keep:
            agg = agg.merge(dim_run[keep].drop_duplicates("run_id"), on="run_id", how="left")

    return agg


class PredictionMetricsByRunSplitHorizonGoldBuilder(GoldBuilder):
    output_table = "gold_prediction_metrics_by_run_split_horizon"
    requires = ("dim_run", "fact_oos_predictions", "fact_config")

    # Per-process flag so the warning emits at most once across builds.
    # Reset by the orchestrator at the start of each refresh.
    _POST_GUARDRAIL_MISSING_WARNING_EMITTED: ClassVar[bool] = False

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        dim_run = snapshot.get("dim_run")
        fact_oos_predictions = snapshot.get("fact_oos_predictions")
        fact_config = snapshot.get("fact_config")

        raw_metrics = _build_metrics_single_contract(
            dim_run,
            fact_oos_predictions,
            fact_config,
            quantile_columns=RAW_QUANTILE_COLUMNS,
        )
        if raw_metrics.empty:
            return pd.DataFrame()

        key_cols = [
            c
            for c in [
                "run_id",
                "asset",
                "feature_set_name",
                "config_signature",
                "split",
                "fold",
                "seed",
                "horizon",
                "model_version",
                "feature_set_hash",
                "parent_sweep_id",
                "trial_number",
                "status",
            ]
            if c in raw_metrics.columns
        ]
        probabilistic_cols = [
            c for c in PROBABILISTIC_METRIC_COLUMNS if c in raw_metrics.columns
        ]
        unique_cols = [c for c in raw_metrics.columns if c not in probabilistic_cols]

        out = raw_metrics[unique_cols].copy()
        raw_renamed = raw_metrics[key_cols + probabilistic_cols].copy().rename(
            columns={m: f"{m}_raw" for m in probabilistic_cols}
        )
        out = out.merge(raw_renamed, on=key_cols, how="left")

        def _emit_deltas(frame: pd.DataFrame) -> pd.DataFrame:
            for m in probabilistic_cols:
                raw_col = f"{m}_raw"
                post_col = f"{m}_post_guardrail"
                if raw_col in frame.columns and post_col in frame.columns:
                    frame[f"delta_{m}_post_minus_raw"] = frame[post_col] - frame[raw_col]
            return frame

        def _set_alias_post_primary(frame: pd.DataFrame, base: str) -> None:
            post_col = f"{base}_post_guardrail"
            raw_col = f"{base}_raw"
            if post_col in frame.columns:
                frame[base] = frame[post_col]
            elif raw_col in frame.columns:
                frame[base] = frame[raw_col]

        def _set_alias_raw_only(frame: pd.DataFrame, base: str) -> None:
            raw_col = f"{base}_raw"
            if raw_col in frame.columns:
                frame[base] = frame[raw_col]

        missing_post = [
            c for c in POST_GUARDRAIL_QUANTILE_COLUMNS if c not in fact_oos_predictions.columns
        ]
        if missing_post:
            if not PredictionMetricsByRunSplitHorizonGoldBuilder._POST_GUARDRAIL_MISSING_WARNING_EMITTED:
                logger.warning(
                    "Post-guardrail quantile columns missing; gold probabilistic post-guardrail metrics will be NaN",
                    extra={"missing_columns": missing_post},
                )
                PredictionMetricsByRunSplitHorizonGoldBuilder._POST_GUARDRAIL_MISSING_WARNING_EMITTED = True
            for m in probabilistic_cols:
                out[f"{m}_post_guardrail"] = np.nan
            for base in ("confidence_calibrated", "prob_up", "prob_down"):
                _set_alias_raw_only(out, base)
            return _emit_deltas(out)

        post_metrics = _build_metrics_single_contract(
            dim_run,
            fact_oos_predictions,
            fact_config,
            quantile_columns=POST_GUARDRAIL_QUANTILE_COLUMNS,
        )
        if post_metrics.empty:
            for m in probabilistic_cols:
                out[f"{m}_post_guardrail"] = np.nan
            for base in ("confidence_calibrated", "prob_up", "prob_down"):
                _set_alias_raw_only(out, base)
            return _emit_deltas(out)

        post_probabilistic_cols = [c for c in probabilistic_cols if c in post_metrics.columns]
        post_renamed = post_metrics[key_cols + post_probabilistic_cols].copy().rename(
            columns={m: f"{m}_post_guardrail" for m in post_probabilistic_cols}
        )
        out = out.merge(post_renamed, on=key_cols, how="left")
        for m in probabilistic_cols:
            col = f"{m}_post_guardrail"
            if col not in out.columns:
                out[col] = np.nan
        for base in ("confidence_calibrated", "prob_up", "prob_down"):
            _set_alias_post_primary(out, base)
        # Cat A: guardrail effect directly queryable on the primary
        # contract. Functionally replaces gold_quantile_guardrail_audit
        # (still materialized for compatibility until Phase B closes).
        return _emit_deltas(out)


class QuantileGuardrailAuditGoldBuilder(GoldBuilder):
    output_table = "gold_quantile_guardrail_audit"
    requires = ("dim_run", "fact_oos_predictions", "fact_config")

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        dim_run = snapshot.get("dim_run")
        fact_oos_predictions = snapshot.get("fact_oos_predictions")
        fact_config = snapshot.get("fact_config")

        required = {
            "run_id",
            "split",
            "horizon",
            "quantile_p10",
            "quantile_p50",
            "quantile_p90",
            "quantile_p10_post_guardrail",
            "quantile_p50_post_guardrail",
            "quantile_p90_post_guardrail",
        }
        if fact_oos_predictions.empty or not required.issubset(set(fact_oos_predictions.columns)):
            return pd.DataFrame()

        base_metrics = _build_metrics_single_contract(
            dim_run,
            fact_oos_predictions,
            fact_config,
            quantile_columns=RAW_QUANTILE_COLUMNS,
        )
        post_metrics = _build_metrics_single_contract(
            dim_run,
            fact_oos_predictions,
            fact_config,
            quantile_columns=POST_GUARDRAIL_QUANTILE_COLUMNS,
        )
        if base_metrics.empty or post_metrics.empty:
            return pd.DataFrame()

        key_cols = [
            c
            for c in [
                "run_id",
                "asset",
                "feature_set_name",
                "config_signature",
                "split",
                "fold",
                "seed",
                "horizon",
                "model_version",
                "feature_set_hash",
                "parent_sweep_id",
                "trial_number",
                "status",
            ]
            if c in base_metrics.columns and c in post_metrics.columns
        ]
        metric_cols = [
            c
            for c in [
                "mean_pinball",
                "picp",
                "mpiw",
                "pred_interval_width",
                "coverage_error",
                "confidence_calibrated",
            ]
            if c in base_metrics.columns and c in post_metrics.columns
        ]

        before = base_metrics[key_cols + metric_cols].copy().rename(
            columns={m: f"{m}_before" for m in metric_cols}
        )
        after = post_metrics[key_cols + metric_cols].copy().rename(
            columns={m: f"{m}_after" for m in metric_cols}
        )
        out = before.merge(after, on=key_cols, how="inner")
        if out.empty:
            return pd.DataFrame()

        for m in metric_cols:
            out[f"delta_{m}_after_minus_before"] = out[f"{m}_after"] - out[f"{m}_before"]

        qdf = fact_oos_predictions.copy()
        for c in [
            "horizon",
            "quantile_p10",
            "quantile_p50",
            "quantile_p90",
            "quantile_p10_post_guardrail",
            "quantile_p50_post_guardrail",
            "quantile_p90_post_guardrail",
            "quantile_guardrail_applied",
        ]:
            if c in qdf.columns:
                qdf[c] = pd.to_numeric(qdf[c], errors="coerce")

        group_cols = [c for c in ["run_id", "split", "horizon"] if c in qdf.columns]
        crossing = (
            qdf.assign(
                crossing_before=(
                    (qdf["quantile_p10"] > qdf["quantile_p50"])
                    | (qdf["quantile_p50"] > qdf["quantile_p90"])
                ).astype(float),
                crossing_after=(
                    (qdf["quantile_p10_post_guardrail"] > qdf["quantile_p50_post_guardrail"])
                    | (qdf["quantile_p50_post_guardrail"] > qdf["quantile_p90_post_guardrail"])
                ).astype(float),
                negative_width_before=(
                    (qdf["quantile_p90"] - qdf["quantile_p10"]) < 0.0
                ).astype(float),
                negative_width_after=(
                    (qdf["quantile_p90_post_guardrail"] - qdf["quantile_p10_post_guardrail"]) < 0.0
                ).astype(float),
                guardrail_applied=qdf.get(
                    "quantile_guardrail_applied", pd.Series(0.0, index=qdf.index)
                ).fillna(0.0),
            )
            .groupby(group_cols, dropna=False)
            .agg(
                n_rows=("run_id", "count"),
                crossing_before_count=("crossing_before", "sum"),
                crossing_after_count=("crossing_after", "sum"),
                negative_width_before_count=("negative_width_before", "sum"),
                negative_width_after_count=("negative_width_after", "sum"),
                guardrail_applied_count=("guardrail_applied", "sum"),
            )
            .reset_index()
        )

        for c in [
            "crossing_before_count",
            "crossing_after_count",
            "negative_width_before_count",
            "negative_width_after_count",
            "guardrail_applied_count",
        ]:
            crossing[c] = pd.to_numeric(crossing[c], errors="coerce").fillna(0.0).astype(int)

        n_rows = pd.to_numeric(crossing["n_rows"], errors="coerce").replace(0, np.nan)
        crossing["crossing_before_rate"] = crossing["crossing_before_count"] / n_rows
        crossing["crossing_after_rate"] = crossing["crossing_after_count"] / n_rows
        crossing["negative_width_before_rate"] = crossing["negative_width_before_count"] / n_rows
        crossing["negative_width_after_rate"] = crossing["negative_width_after_count"] / n_rows
        crossing["guardrail_applied_rate"] = crossing["guardrail_applied_count"] / n_rows

        out = out.merge(crossing, on=["run_id", "split", "horizon"], how="left")
        return out


class QuantileDegeneracyReportGoldBuilder(GoldBuilder):
    output_table = "gold_quantile_degeneracy_report"
    requires = ("fact_oos_predictions", "fact_config")

    def __init__(self, *, thresholds: QuantileDegeneracyThresholds | None = None) -> None:
        self._thresholds = thresholds or QuantileDegeneracyThresholds()

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        fact_oos_predictions = snapshot.get("fact_oos_predictions")
        fact_config = snapshot.get("fact_config")

        columns = [
            "parent_sweep_id",
            "split",
            "horizon",
            "prediction_mode",
            "n_rows",
            "p10_eq_p90_count",
            "p10_eq_p90_rate",
            "p10_eq_p50_eq_p90_count",
            "p10_eq_p50_eq_p90_rate",
            "gate_passed",
        ]
        metrics = QuantileContractAnalyzer.analyze_degeneracy(
            fact_oos_predictions,
            fact_config,
        )
        if not metrics:
            return pd.DataFrame(columns=columns)

        rows = []
        for item in metrics:
            mode = (item.prediction_mode or "").strip().lower()
            gate_failed = (
                mode == "quantile"
                and item.n_rows >= self._thresholds.min_rows_for_gate
                and item.p10_eq_p90_rate >= self._thresholds.max_p10_eq_p90_rate
            )
            rows.append(
                {
                    "parent_sweep_id": item.parent_sweep_id,
                    "split": item.split,
                    "horizon": item.horizon,
                    "prediction_mode": item.prediction_mode,
                    "n_rows": item.n_rows,
                    "p10_eq_p90_count": item.p10_eq_p90_count,
                    "p10_eq_p90_rate": item.p10_eq_p90_rate,
                    "p10_eq_p50_eq_p90_count": item.p10_eq_p50_eq_p90_count,
                    "p10_eq_p50_eq_p90_rate": item.p10_eq_p50_eq_p90_rate,
                    "gate_passed": not gate_failed,
                }
            )
        return pd.DataFrame(rows, columns=columns).sort_values(
            ["parent_sweep_id", "split", "horizon", "prediction_mode"],
            na_position="last",
        ).reset_index(drop=True)


class PredictionMetricsByConfigGoldBuilder(GoldBuilder):
    output_table = "gold_prediction_metrics_by_config"
    requires = ()
    requires_gold = ("gold_prediction_metrics_by_run_split_horizon",)

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        metrics_run_split_h = ctx.gold_outputs["gold_prediction_metrics_by_run_split_horizon"]
        if metrics_run_split_h.empty:
            return pd.DataFrame()

        cols = [
            "asset",
            "feature_set_name",
            "parent_sweep_id",
            "config_signature",
            "split",
            "horizon",
        ]
        if not set(cols).issubset(set(metrics_run_split_h.columns)):
            return pd.DataFrame()

        metric_cols = prediction_metric_columns()
        available_metric_cols = [c for c in metric_cols if c in metrics_run_split_h.columns]

        frame = metrics_run_split_h.copy()
        if "n_samples" in frame.columns:
            frame["n_samples"] = pd.to_numeric(frame["n_samples"], errors="coerce").fillna(0)

        grouped = frame.groupby(cols, dropna=False)
        agg_spec: dict[str, tuple[str, str]] = {"n_runs": ("run_id", "count")}
        if "n_samples" in frame.columns:
            agg_spec["n_oos"] = ("n_samples", "sum")
        out = grouped.agg(**agg_spec).reset_index()
        if "n_oos" in out.columns:
            out["n_oos"] = pd.to_numeric(out["n_oos"], errors="coerce").fillna(0).astype(int)

        for m in available_metric_cols:
            g = (
                grouped[m]
                .agg(["mean", "std"])
                .reset_index()
                .rename(columns={"mean": f"mean_{m}", "std": f"std_{m}"})
            )
            out = out.merge(g, on=cols, how="left")
            iqr = (
                grouped[m]
                .agg(_safe_iqr)
                .reset_index()
                .rename(columns={m: f"iqr_{m}"})
            )
            out = out.merge(iqr, on=cols, how="left")

        return out
