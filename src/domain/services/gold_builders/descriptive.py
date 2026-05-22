"""Descriptive-cluster gold builders.

6 builders mixing Tier 1 (silver/dim only — Ic95, FeatureSetImpact,
OosConsolidated, OosQualityReport) and Tier 2 (consume
`gold_prediction_metrics_by_run_split_horizon` —
PredictionMetricsByHorizon, PredictionCalibration). The R-E suffix fix
(monolith :1486 `df.merge(..., suffixes=("", "_dim"))`) is preserved
inside `OosQualityReportGoldBuilder.build()`; a regression test in
`tests/unit/use_cases/test_refresh_analytics_store_use_case.py` (kept
under that path) continues to gate it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.services.gold_builders.base import (
    BuildContext,
    GoldBuilder,
    GoldBuilderSnapshot,
)
from src.domain.services.gold_builders.pairwise import _pairwise_group_cols
from src.domain.services.gold_builders.quantile import (
    _safe_iqr,
    prediction_metric_columns,
)
from src.domain.services.gold_builders.ranking import _base_join_runs_split_metrics


def _build_ic95_from_base(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    df = base[base["split"] == "test"].copy()
    if df.empty:
        return pd.DataFrame()
    metrics = [
        ("rmse", "test_rmse"),
        ("mae", "test_mae"),
        ("directional_accuracy", "test_da"),
    ]
    rows: list[dict[str, object]] = []
    for metric_col, metric_name in metrics:
        grouped = (
            df.groupby(
                ["asset", "feature_set_name", "parent_sweep_id", "config_signature"],
                dropna=False,
            )[metric_col]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        grouped["std"] = grouped["std"].fillna(0.0)
        grouped["se"] = grouped["std"] / np.sqrt(grouped["count"].clip(lower=1))
        grouped["ci95_low"] = grouped["mean"] - (1.96 * grouped["se"])
        grouped["ci95_high"] = grouped["mean"] + (1.96 * grouped["se"])
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "asset": row["asset"],
                    "feature_set_name": row["feature_set_name"],
                    "parent_sweep_id": row["parent_sweep_id"],
                    "config_signature": row["config_signature"],
                    "metric": metric_name,
                    "n": int(row["count"]),
                    "mean": float(row["mean"]),
                    "std": float(row["std"]),
                    "se": float(row["se"]),
                    "ci95_low": float(row["ci95_low"]),
                    "ci95_high": float(row["ci95_high"]),
                }
            )
    return pd.DataFrame(rows)


def _build_feature_set_impact_from_base(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    df = base.copy()
    metric_cols = ["rmse", "mae", "directional_accuracy"]
    out_rows: list[dict[str, object]] = []
    group_cols = ["asset", "feature_set_name", "parent_sweep_id", "split"]
    for metric_col in metric_cols:
        grouped = (
            df.groupby(group_cols, dropna=False)[metric_col]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        for _, row in grouped.iterrows():
            out_rows.append(
                {
                    "asset": row["asset"],
                    "feature_set_name": row["feature_set_name"],
                    "parent_sweep_id": row["parent_sweep_id"],
                    "split": row["split"],
                    "metric": metric_col,
                    "n_runs": int(row["count"]),
                    "mean_value": float(row["mean"]),
                    "std_value": float(row["std"]) if pd.notna(row["std"]) else 0.0,
                }
            )
    return pd.DataFrame(out_rows)


class Ic95GoldBuilder(GoldBuilder):
    output_table = "gold_ic95_by_config_metric"
    requires = ("dim_run", "fact_split_metrics")

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        base = _base_join_runs_split_metrics(
            snapshot.get("dim_run"), snapshot.get("fact_split_metrics")
        )
        return _build_ic95_from_base(base)


class FeatureSetImpactGoldBuilder(GoldBuilder):
    output_table = "gold_feature_set_impact"
    requires = ("dim_run", "fact_split_metrics")

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        base = _base_join_runs_split_metrics(
            snapshot.get("dim_run"), snapshot.get("fact_split_metrics")
        )
        return _build_feature_set_impact_from_base(base)


class OosConsolidatedGoldBuilder(GoldBuilder):
    output_table = "gold_oos_consolidated"
    requires = ("dim_run", "fact_oos_predictions")

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        dim_run = snapshot.get("dim_run")
        fact_oos_predictions = snapshot.get("fact_oos_predictions")
        if fact_oos_predictions.empty:
            return pd.DataFrame()
        if dim_run.empty:
            return fact_oos_predictions.copy()
        cols = [
            "run_id",
            "model_version",
            "feature_set_hash",
            "parent_sweep_id",
            "trial_number",
            "status",
        ]
        keep = [c for c in cols if c in dim_run.columns]
        dim_trim = dim_run[keep].copy()
        return fact_oos_predictions.merge(dim_trim, on="run_id", how="left")


class OosQualityReportGoldBuilder(GoldBuilder):
    output_table = "gold_oos_quality_report"
    requires = ("dim_run", "fact_oos_predictions")

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        dim_run = snapshot.get("dim_run")
        fact_oos_predictions = snapshot.get("fact_oos_predictions")

        if fact_oos_predictions.empty:
            return pd.DataFrame()
        required = [
            "run_id",
            "split",
            "horizon",
            "timestamp_utc",
            "target_timestamp_utc",
            "y_true",
            "y_pred",
        ]
        missing = [c for c in required if c not in fact_oos_predictions.columns]
        if missing:
            return pd.DataFrame(
                [{"scope": "dataset", "status": "invalid", "detail": f"missing_columns={missing}"}]
            )

        df = fact_oos_predictions.copy()
        for c in ["horizon", "y_true", "y_pred", "quantile_p10", "quantile_p90"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df["target_timestamp_utc"] = pd.to_datetime(
            df["target_timestamp_utc"], utc=True, errors="coerce"
        )

        if not dim_run.empty and "run_id" in dim_run.columns:
            keep = [
                c
                for c in [
                    "run_id",
                    "asset",
                    "feature_set_name",
                    "config_signature",
                    "parent_sweep_id",
                    "split_signature",
                    "split_fingerprint",
                ]
                if c in dim_run.columns
            ]
            # Stage R-E fix: when fact_oos_predictions and dim_run both carry
            # `asset`/`feature_set_name`/`config_signature`, the merge would
            # otherwise produce asset=None in the gold report (downstream
            # gold_quality_statistics_report then read quality_passed_all=False
            # even when all checks passed). Keep fact_oos side as canonical
            # (`suffixes=("", "_dim")`) so groupby finds the actual asset
            # value. Regression guard in tests/unit/use_cases.
            df = df.merge(
                dim_run[keep].drop_duplicates("run_id"),
                on="run_id",
                how="left",
                suffixes=("", "_dim"),
            )

        key_cols = ["run_id", "split", "horizon", "timestamp_utc", "target_timestamp_utc"]
        dup_mask = df.duplicated(subset=key_cols, keep=False)
        df["_dup"] = dup_mask.astype(int)
        df["_target_before_ts"] = (df["target_timestamp_utc"] < df["timestamp_utc"]).fillna(False).astype(int)
        df["_y_true_null"] = df["y_true"].isna().astype(int)
        df["_y_pred_null"] = df["y_pred"].isna().astype(int)
        if {"quantile_p10", "quantile_p90"}.issubset(set(df.columns)):
            width = df["quantile_p90"] - df["quantile_p10"]
            df["_pred_interval_negative"] = ((~width.isna()) & (width < 0.0)).astype(int)
        else:
            df["_pred_interval_negative"] = 0

        rows: list[dict[str, object]] = []
        group_cols = [
            c
            for c in [
                "run_id",
                "asset",
                "feature_set_name",
                "config_signature",
                "parent_sweep_id",
                "split",
                "horizon",
            ]
            if c in df.columns
        ]
        for _, g in df.groupby(group_cols, dropna=False):
            first = g.iloc[0]
            tgt = g["target_timestamp_utc"].dropna().sort_values()
            monotonic = bool(tgt.is_monotonic_increasing)
            rows.append(
                {
                    "scope": "run_split_horizon",
                    "run_id": first.get("run_id"),
                    "asset": first.get("asset"),
                    "feature_set_name": first.get("feature_set_name"),
                    "config_signature": first.get("config_signature"),
                    "parent_sweep_id": first.get("parent_sweep_id"),
                    "split": first.get("split"),
                    "horizon": int(first.get("horizon")) if pd.notna(first.get("horizon")) else None,
                    "n_rows": int(len(g)),
                    "n_duplicate_key_rows": int(g["_dup"].sum()),
                    "n_target_before_timestamp": int(g["_target_before_ts"].sum()),
                    "n_y_true_null": int(g["_y_true_null"].sum()),
                    "n_y_pred_null": int(g["_y_pred_null"].sum()),
                    "n_negative_interval_width": int(g["_pred_interval_negative"].sum()),
                    "target_timestamp_monotonic": monotonic,
                    "passed": bool(
                        int(g["_dup"].sum()) == 0
                        and int(g["_target_before_ts"].sum()) == 0
                        and int(g["_y_true_null"].sum()) == 0
                        and int(g["_y_pred_null"].sum()) == 0
                        and int(g["_pred_interval_negative"].sum()) == 0
                        and monotonic
                    ),
                }
            )

        align_rows: list[dict[str, object]] = []
        align_group_cols = _pairwise_group_cols(df)
        if (
            {"config_signature", "target_timestamp_utc"}.issubset(set(df.columns))
            and align_group_cols
        ):
            for keys, g in df.groupby(align_group_cols, dropna=False):
                per_cfg = {}
                for cfg, gc in g.groupby("config_signature", dropna=False):
                    per_cfg[str(cfg)] = set(gc["target_timestamp_utc"].dropna().astype(str).tolist())
                if len(per_cfg) < 2:
                    continue
                sets = list(per_cfg.values())
                intersection = set.intersection(*sets) if sets else set()
                union = set.union(*sets) if sets else set()
                min_count = min((len(s) for s in sets), default=0)
                max_count = max((len(s) for s in sets), default=0)
                exact = all(s == sets[0] for s in sets[1:])
                has_sig = "split_signature" in align_group_cols
                row = {
                    "scope": "alignment_group",
                    "asset": keys[0] if len(keys) > 0 else None,
                    "parent_sweep_id": keys[1] if len(keys) > 1 else None,
                    "split_signature": keys[2] if has_sig and len(keys) > 2 else None,
                    "split": keys[3]
                    if has_sig and len(keys) > 3
                    else (keys[2] if len(keys) > 2 else None),
                    "horizon": int(keys[4])
                    if has_sig and len(keys) > 4 and pd.notna(keys[4])
                    else (int(keys[3]) if len(keys) > 3 and pd.notna(keys[3]) else None),
                    "n_configs": int(len(per_cfg)),
                    "target_intersection_count": int(len(intersection)),
                    "target_union_count": int(len(union)),
                    "target_min_count_per_config": int(min_count),
                    "target_max_count_per_config": int(max_count),
                    "target_exact_alignment": bool(exact),
                    "target_jaccard_alignment": float(len(intersection) / len(union))
                    if len(union) > 0
                    else 1.0,
                    "passed": bool(exact),
                }
                align_rows.append(row)

        return pd.DataFrame(rows + align_rows)


class PredictionMetricsByHorizonGoldBuilder(GoldBuilder):
    output_table = "gold_prediction_metrics_by_horizon"
    requires = ()
    requires_gold = ("gold_prediction_metrics_by_run_split_horizon",)

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        metrics_run_split_h = ctx.gold_outputs["gold_prediction_metrics_by_run_split_horizon"]
        if metrics_run_split_h.empty:
            return pd.DataFrame()

        cols = ["asset", "feature_set_name", "parent_sweep_id", "split", "horizon"]
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


class PredictionCalibrationGoldBuilder(GoldBuilder):
    output_table = "gold_prediction_calibration"
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

        keep = [
            c
            for c in [
                "run_id",
                *cols,
                "n_samples",
                "pinball_q10_raw",
                "pinball_q50_raw",
                "pinball_q90_raw",
                "mean_pinball_raw",
                "picp_raw",
                "mpiw_raw",
                "pred_interval_width_raw",
                "coverage_error_raw",
                "confidence_calibrated_raw",
                "pinball_q10_post_guardrail",
                "pinball_q50_post_guardrail",
                "pinball_q90_post_guardrail",
                "mean_pinball_post_guardrail",
                "picp_post_guardrail",
                "mpiw_post_guardrail",
                "pred_interval_width_post_guardrail",
                "coverage_error_post_guardrail",
                "confidence_calibrated_post_guardrail",
                "prob_up_raw",
                "prob_up_post_guardrail",
                "prob_down_raw",
                "prob_down_post_guardrail",
                "coverage_nominal",
                "prob_up",
                "prob_down",
                "confidence_calibrated",
            ]
            if c in metrics_run_split_h.columns
        ]
        return metrics_run_split_h[keep].copy()
