"""Confidence-cluster gold builders.

8 builders covering risk, generalization, robustness, feature impact,
local SHAP-style summaries, and the Tier 3 quality+decision rollups.
Mix of Tier 1 (risk, feature_contrib_local_summary), Tier 2
(generalization, robustness, feature_impact_by_horizon) and Tier 3
(model_decision_final, quality_statistics_report,
quality_run_sweep_summary).
"""
from __future__ import annotations

import json
import logging

from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd

from src.domain.services.gold_builders.base import (
    BuildContext,
    GoldBuilder,
    GoldBuilderSnapshot,
    normalize_parent_sweep_id_for_merge,
)
from src.domain.services.gold_builders.quantile import (
    PredictionMetricsByRunSplitHorizonGoldBuilder,
    prediction_metric_columns,
)

logger = logging.getLogger(__name__)


class PredictionRiskGoldBuilder(GoldBuilder):
    output_table = "gold_prediction_risk"
    requires = ("dim_run", "fact_oos_predictions")

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        dim_run = snapshot.get("dim_run")
        fact_oos_predictions = snapshot.get("fact_oos_predictions")

        if fact_oos_predictions.empty:
            return pd.DataFrame()

        required = ["run_id", "split", "horizon", "y_pred"]
        missing = [c for c in required if c not in fact_oos_predictions.columns]
        if missing:
            return pd.DataFrame()

        df = fact_oos_predictions.copy()
        for c in ["horizon", "y_pred"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["horizon", "y_pred"]).copy()
        if df.empty:
            return pd.DataFrame()

        df["horizon"] = df["horizon"].astype(int)
        df["expected_move_row"] = df["y_pred"].abs()
        df["downside_risk_row"] = np.maximum(-df["y_pred"], 0.0)

        # Cat B: var_10/es_10_approx require quantile monotonicity (Jorion
        # 2007; Acerbi & Tasche 2002). Under raw with crossing the number
        # loses risk interpretation; use exclusively post-guardrail columns.
        post_q10 = "quantile_p10_post_guardrail"
        post_q50 = "quantile_p50_post_guardrail"
        if post_q10 in df.columns and post_q50 in df.columns:
            df[post_q10] = pd.to_numeric(df[post_q10], errors="coerce")
            df[post_q50] = pd.to_numeric(df[post_q50], errors="coerce")
            df["var_10_row"] = df[post_q10]
            df["es_10_approx_row"] = 1.125 * df[post_q10] - 0.125 * df[post_q50]
            df["es_10_approx_row"] = np.minimum(df["es_10_approx_row"], df["var_10_row"])
        else:
            if not PredictionMetricsByRunSplitHorizonGoldBuilder._POST_GUARDRAIL_MISSING_WARNING_EMITTED:
                logger.warning(
                    "Post-guardrail quantile columns missing; gold probabilistic post-guardrail metrics will be NaN",
                    extra={
                        "missing_columns": [
                            c for c in (post_q10, post_q50) if c not in df.columns
                        ]
                    },
                )
                PredictionMetricsByRunSplitHorizonGoldBuilder._POST_GUARDRAIL_MISSING_WARNING_EMITTED = True
            df["var_10_row"] = np.nan
            df["es_10_approx_row"] = np.nan

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
            if c in df.columns
        ]

        out = (
            df.groupby(group_cols, dropna=False)
            .agg(
                n_samples=("y_pred", "count"),
                expected_move=("expected_move_row", "mean"),
                downside_risk=("downside_risk_row", "mean"),
                var_10=("var_10_row", "mean"),
                es_10_approx=("es_10_approx_row", "mean"),
            )
            .reset_index()
        )

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
                out = out.merge(dim_run[keep].drop_duplicates("run_id"), on="run_id", how="left")

        return out


class PredictionGeneralizationGapGoldBuilder(GoldBuilder):
    output_table = "gold_prediction_generalization_gap"
    requires = ()
    requires_gold = ("gold_prediction_metrics_by_run_split_horizon",)

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        metrics_run_split_h = ctx.gold_outputs["gold_prediction_metrics_by_run_split_horizon"]
        if metrics_run_split_h.empty:
            return pd.DataFrame()

        req = {
            "asset",
            "feature_set_name",
            "parent_sweep_id",
            "config_signature",
            "horizon",
            "split",
        }
        if not req.issubset(set(metrics_run_split_h.columns)):
            return pd.DataFrame()

        metric_cols = [
            c for c in prediction_metric_columns() if c in metrics_run_split_h.columns
        ]
        if not metric_cols:
            return pd.DataFrame()

        grouped = metrics_run_split_h.groupby(
            ["asset", "feature_set_name", "parent_sweep_id", "config_signature", "horizon", "split"],
            dropna=False,
        )

        means = grouped[metric_cols].mean().reset_index()
        test_df = means[means["split"].astype(str) == "test"].drop(columns=["split"])
        val_df = means[means["split"].astype(str) == "val"].drop(columns=["split"])
        if test_df.empty or val_df.empty:
            return pd.DataFrame()

        key_cols = ["asset", "feature_set_name", "parent_sweep_id", "config_signature", "horizon"]
        out = test_df.merge(val_df, on=key_cols, how="inner", suffixes=("_test", "_val"))
        if out.empty:
            return pd.DataFrame()

        for m in metric_cols:
            out[f"gap_{m}_test_minus_val"] = out[f"{m}_test"] - out[f"{m}_val"]

        keep = key_cols + [c for c in out.columns if c.startswith("gap_")]
        return out[keep].copy()


class PredictionRobustnessByHorizonGoldBuilder(GoldBuilder):
    output_table = "gold_prediction_robustness_by_horizon"
    requires = ()
    requires_gold = ("gold_prediction_metrics_by_run_split_horizon",)

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        metrics_run_split_h = ctx.gold_outputs["gold_prediction_metrics_by_run_split_horizon"]
        if metrics_run_split_h.empty:
            return pd.DataFrame()

        req = {
            "asset",
            "feature_set_name",
            "parent_sweep_id",
            "config_signature",
            "horizon",
            "split",
        }
        if not req.issubset(set(metrics_run_split_h.columns)):
            return pd.DataFrame()

        df = metrics_run_split_h[metrics_run_split_h["split"].astype(str) == "test"].copy()
        if df.empty:
            return pd.DataFrame()

        metric_cols = [c for c in prediction_metric_columns() if c in df.columns]
        if not metric_cols:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        gcols = ["asset", "feature_set_name", "parent_sweep_id", "config_signature", "horizon"]
        for keys, g in df.groupby(gcols, dropna=False):
            for m in metric_cols:
                series = pd.to_numeric(g[m], errors="coerce").dropna()
                if series.empty:
                    continue
                n = int(len(series))
                mean = float(series.mean())
                std = float(series.std(ddof=1)) if n > 1 else 0.0
                se = std / np.sqrt(max(1, n))
                rows.append(
                    {
                        "asset": keys[0],
                        "feature_set_name": keys[1],
                        "parent_sweep_id": keys[2],
                        "config_signature": keys[3],
                        "horizon": int(keys[4]) if pd.notna(keys[4]) else None,
                        "metric": m,
                        "n_runs": n,
                        "mean": mean,
                        "std": std,
                        "median": float(np.nanmedian(series)),
                        "iqr": float(np.nanpercentile(series, 75) - np.nanpercentile(series, 25)),
                        "ci95_low": float(mean - 1.96 * se),
                        "ci95_high": float(mean + 1.96 * se),
                    }
                )
        return pd.DataFrame(rows)


class FeatureImpactByHorizonGoldBuilder(GoldBuilder):
    output_table = "gold_feature_impact_by_horizon"
    requires = ("fact_model_artifacts",)
    requires_gold = ("gold_prediction_metrics_by_run_split_horizon",)

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        fact_model_artifacts = snapshot.get("fact_model_artifacts")
        metrics_run_split_h = ctx.gold_outputs["gold_prediction_metrics_by_run_split_horizon"]

        if fact_model_artifacts.empty or metrics_run_split_h.empty:
            return pd.DataFrame()
        if not {"run_id", "feature_importance_json"}.issubset(set(fact_model_artifacts.columns)):
            return pd.DataFrame()
        req = {"run_id", "asset", "feature_set_name", "parent_sweep_id", "split", "horizon"}
        if not req.issubset(set(metrics_run_split_h.columns)):
            return pd.DataFrame()

        split_h = metrics_run_split_h[
            ["run_id", "asset", "feature_set_name", "parent_sweep_id", "split", "horizon"]
        ].drop_duplicates()
        split_h = split_h[split_h["split"].astype(str).isin(["val", "test"])].copy()
        if split_h.empty:
            return pd.DataFrame()

        rows: list[dict[str, object]] = []
        for _, art in fact_model_artifacts.iterrows():
            run_id = art.get("run_id")
            raw = art.get("feature_importance_json")
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, list):
                continue
            horizons = split_h[split_h["run_id"] == run_id]
            if horizons.empty:
                continue
            for _, hz in horizons.iterrows():
                for item in parsed:
                    if not isinstance(item, dict) or "feature" not in item:
                        continue
                    rows.append(
                        {
                            "run_id": run_id,
                            "asset": hz.get("asset"),
                            "feature_set_name": hz.get("feature_set_name"),
                            "parent_sweep_id": hz.get("parent_sweep_id"),
                            "split": hz.get("split"),
                            "horizon": int(hz.get("horizon")) if pd.notna(hz.get("horizon")) else None,
                            "feature_name": str(item.get("feature")),
                            "delta_rmse": pd.to_numeric(item.get("delta_rmse"), errors="coerce"),
                            "delta_mae": pd.to_numeric(item.get("delta_mae"), errors="coerce"),
                            "baseline_rmse": pd.to_numeric(item.get("baseline_rmse"), errors="coerce"),
                            "baseline_mae": pd.to_numeric(item.get("baseline_mae"), errors="coerce"),
                            "method": "global_importance_reused_by_horizon",
                        }
                    )

        if not rows:
            return pd.DataFrame()
        detail = pd.DataFrame(rows)
        agg = (
            detail.groupby(
                [
                    "asset",
                    "feature_set_name",
                    "parent_sweep_id",
                    "split",
                    "horizon",
                    "feature_name",
                    "method",
                ],
                dropna=False,
            )
            .agg(
                n_runs=("run_id", "count"),
                mean_delta_rmse=("delta_rmse", "mean"),
                std_delta_rmse=("delta_rmse", "std"),
                mean_delta_mae=("delta_mae", "mean"),
                std_delta_mae=("delta_mae", "std"),
            )
            .reset_index()
        )
        return agg


class FeatureContribLocalSummaryGoldBuilder(GoldBuilder):
    output_table = "gold_feature_contrib_local_summary"
    requires = ("dim_run", "fact_feature_contrib_local")

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        fact_feature_contrib_local = snapshot.get("fact_feature_contrib_local")
        dim_run = snapshot.get("dim_run")

        if fact_feature_contrib_local.empty:
            return pd.DataFrame()

        req = {
            "asset",
            "feature_set_name",
            "horizon",
            "feature_name",
            "contribution",
            "abs_contribution",
        }
        if not req.issubset(set(fact_feature_contrib_local.columns)):
            return pd.DataFrame()

        df = fact_feature_contrib_local.copy()
        if "run_id" in df.columns and {"run_id", "parent_sweep_id"}.issubset(set(dim_run.columns)):
            dim_trim = normalize_parent_sweep_id_for_merge(
                dim_run[["run_id", "parent_sweep_id"]].drop_duplicates("run_id")
            )
            df = df.merge(dim_trim, on="run_id", how="left")
            df = normalize_parent_sweep_id_for_merge(df)
        else:
            df["parent_sweep_id"] = None
        df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
        df["contribution"] = pd.to_numeric(df["contribution"], errors="coerce")
        df["abs_contribution"] = pd.to_numeric(df["abs_contribution"], errors="coerce")
        if "feature_rank" in df.columns:
            df["feature_rank"] = pd.to_numeric(df["feature_rank"], errors="coerce")
        if "timestamp_utc" in df.columns:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")

        df = df.dropna(subset=["horizon", "contribution", "abs_contribution", "feature_name"]).copy()
        if df.empty:
            return pd.DataFrame()
        df["horizon"] = df["horizon"].astype(int)
        if "method" not in df.columns:
            df["method"] = "unknown"

        group_cols = [
            "asset",
            "feature_set_name",
            "parent_sweep_id",
            "horizon",
            "feature_name",
            "method",
        ]
        agg = (
            df.groupby(group_cols, dropna=False)
            .agg(
                n_rows=("contribution", "count"),
                n_inference_runs=("inference_run_id", "nunique")
                if "inference_run_id" in df.columns
                else ("contribution", "count"),
                mean_contribution=("contribution", "mean"),
                mean_abs_contribution=("abs_contribution", "mean"),
                median_abs_contribution=("abs_contribution", "median"),
                std_abs_contribution=("abs_contribution", "std"),
                top3_frequency=(
                    "feature_rank",
                    lambda s: float(
                        (pd.to_numeric(pd.Series(s), errors="coerce") <= 3).fillna(False).mean()
                    ),
                )
                if "feature_rank" in df.columns
                else ("contribution", lambda s: np.nan),
                positive_share=(
                    "contribution",
                    lambda s: float(
                        (pd.to_numeric(pd.Series(s), errors="coerce") > 0).fillna(False).mean()
                    ),
                ),
            )
            .reset_index()
        )

        stability_rows: list[dict[str, object]] = []
        if "inference_run_id" in df.columns:
            stab_group_cols = ["asset", "feature_set_name", "parent_sweep_id", "horizon", "method"]
            for keys, g in df.groupby(stab_group_cols, dropna=False):
                run_top: dict[str, set[str]] = {}
                for rid, rg in g.groupby("inference_run_id", dropna=False):
                    top = (
                        rg.groupby("feature_name", dropna=False)["abs_contribution"]
                        .mean()
                        .sort_values(ascending=False)
                        .head(3)
                    )
                    run_top[str(rid)] = set(top.index.astype(str).tolist())

                vals: list[float] = []
                run_ids = sorted(run_top.keys())
                if len(run_ids) >= 2:
                    for a, b in combinations(run_ids, 2):
                        sa = run_top.get(a, set())
                        sb = run_top.get(b, set())
                        if not sa and not sb:
                            vals.append(1.0)
                        else:
                            union = sa | sb
                            inter = sa & sb
                            vals.append(float(len(inter) / len(union)) if union else 1.0)
                stability_rows.append(
                    {
                        "asset": keys[0],
                        "feature_set_name": keys[1],
                        "parent_sweep_id": keys[2],
                        "horizon": int(keys[3]) if pd.notna(keys[3]) else None,
                        "method": keys[4],
                        "local_top3_jaccard_mean": float(np.mean(vals)) if vals else np.nan,
                        "local_top3_jaccard_pairs": int(len(vals)),
                    }
                )

        if stability_rows:
            stab = pd.DataFrame(stability_rows)
            agg = agg.merge(
                stab,
                on=["asset", "feature_set_name", "parent_sweep_id", "horizon", "method"],
                how="left",
            )

        return agg


def _build_model_decision_final(
    metrics_by_config: pd.DataFrame,
    robustness_by_horizon: pd.DataFrame,
    generalization_gap: pd.DataFrame,
    dm_results: pd.DataFrame,
    mcs_results: pd.DataFrame,
    win_rate_results: pd.DataFrame,
    paired_intersection: pd.DataFrame,
    *,
    primary_quantile_contract: Literal["raw", "post_guardrail"] = "post_guardrail",
) -> pd.DataFrame:
    """Tier 3 rollup composing 7 upstream gold tables into the final
    config-level decision view. Kept module-level so unit tests can
    target it directly (the monolith exposed a `@staticmethod` of the
    same name; the migrated tests call this function instead).
    """
    if primary_quantile_contract not in {"raw", "post_guardrail"}:
        raise ValueError("primary_quantile_contract must be one of: raw, post_guardrail")
    if metrics_by_config.empty:
        return pd.DataFrame()

    req = {"asset", "feature_set_name", "config_signature", "split", "horizon"}
    if not req.issubset(set(metrics_by_config.columns)):
        return pd.DataFrame()

    base = metrics_by_config[metrics_by_config["split"].astype(str) == "test"].copy()
    if base.empty:
        return pd.DataFrame()

    primary_metric_map = {
        f"mean_pinball_q10_{primary_quantile_contract}": "mean_pinball_q10",
        f"mean_pinball_q50_{primary_quantile_contract}": "mean_pinball_q50",
        f"mean_pinball_q90_{primary_quantile_contract}": "mean_pinball_q90",
        f"mean_mean_pinball_{primary_quantile_contract}": "mean_mean_pinball",
        f"mean_picp_{primary_quantile_contract}": "mean_picp",
        f"mean_mpiw_{primary_quantile_contract}": "mean_mpiw",
    }

    keep = [
        c
        for c in [
            "asset",
            "feature_set_name",
            "config_signature",
            "split",
            "horizon",
            "n_runs",
            "mean_rmse",
            "std_rmse",
            "mean_mae",
            "std_mae",
            "mean_directional_accuracy",
            "std_directional_accuracy",
            *primary_metric_map.keys(),
        ]
        if c in base.columns
    ]
    out = base[keep].copy()
    out = out.rename(columns={src: dst for src, dst in primary_metric_map.items() if src in out.columns})
    out["primary_quantile_contract"] = primary_quantile_contract
    out["config_label"] = out["feature_set_name"].astype(str) + "|" + out["config_signature"].astype(str)
    out = normalize_parent_sweep_id_for_merge(out)

    if not robustness_by_horizon.empty and {
        "asset",
        "feature_set_name",
        "config_signature",
        "horizon",
        "metric",
        "ci95_low",
        "ci95_high",
    }.issubset(set(robustness_by_horizon.columns)):
        ci = robustness_by_horizon.copy()
        ci = ci[ci["metric"].astype(str).isin(["rmse", "mae", "directional_accuracy"])].copy()
        ci_wide = (
            ci.pivot_table(
                index=["asset", "feature_set_name", "config_signature", "horizon"],
                columns="metric",
                values=["ci95_low", "ci95_high"],
                aggfunc="first",
            )
            .reset_index()
        )
        ci_wide.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).strip("_") if isinstance(col, tuple) else str(col)
            for col in ci_wide.columns
        ]
        out = out.merge(
            ci_wide,
            on=["asset", "feature_set_name", "config_signature", "horizon"],
            how="left",
        )

    if not generalization_gap.empty and {
        "asset",
        "feature_set_name",
        "config_signature",
        "horizon",
    }.issubset(set(generalization_gap.columns)):
        primary_gap_map = {
            f"gap_mean_pinball_{primary_quantile_contract}_test_minus_val": "gap_mean_pinball_test_minus_val",
            f"gap_picp_{primary_quantile_contract}_test_minus_val": "gap_picp_test_minus_val",
            f"gap_mpiw_{primary_quantile_contract}_test_minus_val": "gap_mpiw_test_minus_val",
        }
        gap_keep = [
            c
            for c in [
                "asset",
                "feature_set_name",
                "config_signature",
                "horizon",
                "gap_rmse_test_minus_val",
                "gap_mae_test_minus_val",
                "gap_directional_accuracy_test_minus_val",
                *primary_gap_map.keys(),
            ]
            if c in generalization_gap.columns
        ]
        gap = generalization_gap[gap_keep].rename(
            columns={src: dst for src, dst in primary_gap_map.items() if src in gap_keep}
        )
        out = out.merge(
            gap, on=["asset", "feature_set_name", "config_signature", "horizon"], how="left"
        )

    dm_rows: list[dict[str, object]] = []
    if not dm_results.empty and {
        "asset",
        "parent_sweep_id",
        "split",
        "horizon",
        "left_config",
        "right_config",
        "pvalue_two_sided",
        "mean_loss_diff_left_minus_right",
    }.issubset(set(dm_results.columns)):
        for keys, g in dm_results.groupby(
            ["asset", "parent_sweep_id", "split", "horizon"], dropna=False
        ):
            winners: dict[str, int] = {}
            losers: dict[str, int] = {}
            for _, r in g.iterrows():
                p = pd.to_numeric(r.get("pvalue_two_sided"), errors="coerce")
                d = pd.to_numeric(r.get("mean_loss_diff_left_minus_right"), errors="coerce")
                lcfg = str(r.get("left_config"))
                rcfg = str(r.get("right_config"))
                if pd.isna(p) or pd.isna(d) or p >= 0.05:
                    continue
                if d < 0:
                    winners[lcfg] = winners.get(lcfg, 0) + 1
                    losers[rcfg] = losers.get(rcfg, 0) + 1
                elif d > 0:
                    winners[rcfg] = winners.get(rcfg, 0) + 1
                    losers[lcfg] = losers.get(lcfg, 0) + 1
            configs = set(list(winners.keys()) + list(losers.keys()))
            for cfg in configs:
                dm_rows.append(
                    {
                        "asset": keys[0],
                        "parent_sweep_id": keys[1],
                        "split": keys[2],
                        "horizon": int(keys[3]) if pd.notna(keys[3]) else None,
                        "config_label": cfg,
                        "dm_significant_wins": int(winners.get(cfg, 0)),
                        "dm_significant_losses": int(losers.get(cfg, 0)),
                        "dm_net_wins": int(winners.get(cfg, 0) - losers.get(cfg, 0)),
                    }
                )
    dm_summary = pd.DataFrame(dm_rows)

    mcs_summary = pd.DataFrame()
    if not mcs_results.empty and {
        "asset",
        "parent_sweep_id",
        "split",
        "horizon",
        "config_label",
        "selected_in_mcs_alpha_0_05",
    }.issubset(set(mcs_results.columns)):
        mcs_summary = mcs_results[
            ["asset", "parent_sweep_id", "split", "horizon", "config_label", "selected_in_mcs_alpha_0_05"]
        ].copy()
        mcs_summary = mcs_summary.rename(
            columns={"selected_in_mcs_alpha_0_05": "mcs_selected_alpha_0_05"}
        )

    wr_rows: list[dict[str, object]] = []
    if not win_rate_results.empty and {
        "asset",
        "parent_sweep_id",
        "split",
        "horizon",
        "left_config",
        "right_config",
        "left_win_rate_ex_ties",
        "right_win_rate_ex_ties",
    }.issubset(set(win_rate_results.columns)):
        recs: dict[tuple[object, object, object, object, str], list[float]] = {}
        for _, r in win_rate_results.iterrows():
            key_left = (
                r.get("asset"),
                r.get("parent_sweep_id"),
                r.get("split"),
                r.get("horizon"),
                str(r.get("left_config")),
            )
            key_right = (
                r.get("asset"),
                r.get("parent_sweep_id"),
                r.get("split"),
                r.get("horizon"),
                str(r.get("right_config")),
            )
            lw = pd.to_numeric(r.get("left_win_rate_ex_ties"), errors="coerce")
            rw = pd.to_numeric(r.get("right_win_rate_ex_ties"), errors="coerce")
            if pd.notna(lw):
                recs.setdefault(key_left, []).append(float(lw))
            if pd.notna(rw):
                recs.setdefault(key_right, []).append(float(rw))
        for (asset, sweep, split, horizon, cfg), vals in recs.items():
            horizon_value = pd.to_numeric(horizon, errors="coerce")
            wr_rows.append(
                {
                    "asset": asset,
                    "parent_sweep_id": sweep,
                    "split": split,
                    "horizon": int(horizon_value) if pd.notna(horizon_value) else None,
                    "config_label": cfg,
                    "win_rate_ex_ties_mean": float(np.mean(vals)),
                }
            )
    wr_summary = pd.DataFrame(wr_rows)

    dm_summary = normalize_parent_sweep_id_for_merge(dm_summary)
    mcs_summary = normalize_parent_sweep_id_for_merge(mcs_summary)
    wr_summary = normalize_parent_sweep_id_for_merge(wr_summary)

    sweep_map = pd.DataFrame()
    for df in [mcs_summary, dm_summary, wr_summary]:
        if not df.empty and {
            "asset",
            "split",
            "horizon",
            "config_label",
            "parent_sweep_id",
        }.issubset(set(df.columns)):
            part = df[["asset", "split", "horizon", "config_label", "parent_sweep_id"]].drop_duplicates()
            sweep_map = pd.concat([sweep_map, part], ignore_index=True) if not sweep_map.empty else part
    if not sweep_map.empty:
        sweep_map = sweep_map.drop_duplicates(["asset", "split", "horizon", "config_label"])
        out = out.merge(sweep_map, on=["asset", "split", "horizon", "config_label"], how="left")
    else:
        out["parent_sweep_id"] = None

    for df in [dm_summary, mcs_summary, wr_summary]:
        if df.empty:
            continue
        merge_cols = [
            c
            for c in ["asset", "parent_sweep_id", "split", "horizon", "config_label"]
            if c in df.columns and c in out.columns
        ]
        if not merge_cols:
            continue
        out = out.merge(df.drop_duplicates(merge_cols), on=merge_cols, how="left")

    if not paired_intersection.empty and {
        "asset",
        "parent_sweep_id",
        "split",
        "horizon",
    }.issubset(set(paired_intersection.columns)):
        inter_keep = [
            c
            for c in [
                "asset",
                "parent_sweep_id",
                "split",
                "horizon",
                "n_common",
                "n_union",
                "coverage_ratio",
                "aligned_exact",
                "target_intersection_count",
                "target_exact_alignment",
                "target_jaccard_alignment",
                "pairwise_ready_dm",
                "pairwise_ready_mcs",
            ]
            if c in paired_intersection.columns
        ]
        pi = normalize_parent_sweep_id_for_merge(paired_intersection[inter_keep])
        key_cols = ["asset", "parent_sweep_id", "split", "horizon"]
        if not pi.empty and pi.duplicated(subset=key_cols).any():
            agg: dict[str, str] = {}
            for col in ["n_common", "target_intersection_count"]:
                if col in pi.columns:
                    agg[col] = "min"
            for col in ["n_union"]:
                if col in pi.columns:
                    agg[col] = "max"
            for col in ["coverage_ratio", "target_jaccard_alignment"]:
                if col in pi.columns:
                    agg[col] = "min"
            for col in [
                "aligned_exact",
                "target_exact_alignment",
                "pairwise_ready_dm",
                "pairwise_ready_mcs",
            ]:
                if col in pi.columns:
                    agg[col] = "all"
            if agg:
                pi = pi.groupby(key_cols, dropna=False).agg(agg).reset_index()
            else:
                pi = pi.drop_duplicates(key_cols)
        out = normalize_parent_sweep_id_for_merge(out)
        pi = normalize_parent_sweep_id_for_merge(pi)
        out = out.merge(pi, on=key_cols, how="left")

    out = normalize_parent_sweep_id_for_merge(out)
    decision_rank_cols = ["asset", "parent_sweep_id", "horizon"]
    out["rank_rmse"] = (
        out.groupby(decision_rank_cols, dropna=False)["mean_rmse"].rank(method="min", ascending=True)
        if "mean_rmse" in out.columns
        else np.nan
    )
    out["rank_mae"] = (
        out.groupby(decision_rank_cols, dropna=False)["mean_mae"].rank(method="min", ascending=True)
        if "mean_mae" in out.columns
        else np.nan
    )
    out["rank_da"] = (
        out.groupby(decision_rank_cols, dropna=False)["mean_directional_accuracy"].rank(
            method="min", ascending=False
        )
        if "mean_directional_accuracy" in out.columns
        else np.nan
    )

    for col in ["pairwise_ready_dm", "pairwise_ready_mcs", "target_exact_alignment"]:
        if col not in out.columns:
            out[col] = False
    _dm = out["pairwise_ready_dm"]
    _mcs = out["pairwise_ready_mcs"]
    _tgt = out["target_exact_alignment"]
    out["academic_decision_ready"] = (
        _dm.where(_dm.notna(), False).astype(bool)
        & _mcs.where(_mcs.notna(), False).astype(bool)
        & _tgt.where(_tgt.notna(), False).astype(bool)
    )

    return out.sort_values(
        ["asset", "parent_sweep_id", "horizon", "rank_rmse", "rank_mae"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)


class ModelDecisionFinalGoldBuilder(GoldBuilder):
    output_table = "gold_model_decision_final"
    requires = ()
    requires_gold = (
        "gold_prediction_metrics_by_config",
        "gold_prediction_robustness_by_horizon",
        "gold_prediction_generalization_gap",
        "gold_dm_pairwise_results",
        "gold_mcs_results",
        "gold_win_rate_pairwise_results",
        "gold_paired_oos_intersection_by_horizon",
    )

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        return _build_model_decision_final(
            metrics_by_config=ctx.gold_outputs["gold_prediction_metrics_by_config"],
            robustness_by_horizon=ctx.gold_outputs["gold_prediction_robustness_by_horizon"],
            generalization_gap=ctx.gold_outputs["gold_prediction_generalization_gap"],
            dm_results=ctx.gold_outputs["gold_dm_pairwise_results"],
            mcs_results=ctx.gold_outputs["gold_mcs_results"],
            win_rate_results=ctx.gold_outputs["gold_win_rate_pairwise_results"],
            paired_intersection=ctx.gold_outputs["gold_paired_oos_intersection_by_horizon"],
            primary_quantile_contract=ctx.primary_quantile_contract,
        )


class QualityStatisticsReportGoldBuilder(GoldBuilder):
    output_table = "gold_quality_statistics_report"
    requires = ()
    requires_gold = (
        "gold_oos_quality_report",
        "gold_dm_pairwise_results",
        "gold_mcs_results",
        "gold_win_rate_pairwise_results",
    )

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        quality_report = ctx.gold_outputs["gold_oos_quality_report"]
        dm_results = ctx.gold_outputs["gold_dm_pairwise_results"]
        mcs_results = ctx.gold_outputs["gold_mcs_results"]
        win_rate_results = ctx.gold_outputs["gold_win_rate_pairwise_results"]

        if (
            quality_report.empty
            and dm_results.empty
            and mcs_results.empty
            and win_rate_results.empty
        ):
            return pd.DataFrame()

        rows: list[dict[str, object]] = []

        q = quality_report.copy() if not quality_report.empty else pd.DataFrame()
        if not q.empty and "scope" in q.columns:
            q = q[q["scope"].astype(str).isin(["run_split_horizon", "alignment_group"])].copy()

        keys = ["asset", "parent_sweep_id", "split", "horizon"]

        q_by_group: dict[tuple, dict[str, object]] = {}
        if not q.empty and set(keys).issubset(set(q.columns)):
            for k, g in q.groupby(keys, dropna=False):
                q_by_group[k] = {
                    "quality_rows": int(len(g)),
                    "quality_passed_all": bool(
                        g.get("passed", pd.Series(dtype=bool)).fillna(False).all()
                    ),
                }

        dm_by_group: dict[tuple, dict[str, object]] = {}
        if not dm_results.empty and set(keys).issubset(set(dm_results.columns)):
            for k, g in dm_results.groupby(keys, dropna=False):
                dm_by_group[k] = {
                    "dm_pairs": int(len(g)),
                    "dm_min_pvalue": float(
                        pd.to_numeric(g.get("pvalue_two_sided"), errors="coerce").min()
                    ),
                    "aligned_timestamps_dm": int(
                        pd.to_numeric(g.get("aligned_timestamps"), errors="coerce").max()
                    ),
                    "n_configs_dm": int(pd.to_numeric(g.get("n_configs"), errors="coerce").max()),
                }

        mcs_by_group: dict[tuple, dict[str, object]] = {}
        if not mcs_results.empty and set(keys).issubset(set(mcs_results.columns)):
            for k, g in mcs_results.groupby(keys, dropna=False):
                selected = pd.to_numeric(g.get("selected_in_mcs_alpha_0_05"), errors="coerce")
                mcs_by_group[k] = {
                    "mcs_models": int(len(g)),
                    "mcs_selected_models": int(selected.fillna(0).astype(int).sum())
                    if not selected.empty
                    else 0,
                    "aligned_timestamps_mcs": int(
                        pd.to_numeric(g.get("aligned_timestamps"), errors="coerce").max()
                    ),
                    "n_configs_mcs": int(pd.to_numeric(g.get("n_configs"), errors="coerce").max()),
                }

        wr_by_group: dict[tuple, dict[str, object]] = {}
        if not win_rate_results.empty and set(keys).issubset(set(win_rate_results.columns)):
            for k, g in win_rate_results.groupby(keys, dropna=False):
                wr_by_group[k] = {
                    "win_rate_pairs": int(len(g)),
                    "aligned_timestamps_win_rate": int(
                        pd.to_numeric(g.get("aligned_timestamps"), errors="coerce").max()
                    ),
                }

        all_keys = (
            set(q_by_group.keys())
            | set(dm_by_group.keys())
            | set(mcs_by_group.keys())
            | set(wr_by_group.keys())
        )
        for k in sorted(all_keys, key=lambda x: tuple(str(v) for v in x)):
            base = {
                "asset": k[0],
                "parent_sweep_id": k[1],
                "split": k[2],
                "horizon": k[3],
                "quality_rows": 0,
                "quality_passed_all": False,
                "dm_pairs": 0,
                "mcs_models": 0,
                "mcs_selected_models": 0,
                "win_rate_pairs": 0,
                "aligned_timestamps_dm": 0,
                "aligned_timestamps_mcs": 0,
                "aligned_timestamps_win_rate": 0,
                "n_configs_dm": 0,
                "n_configs_mcs": 0,
            }
            base.update(q_by_group.get(k, {}))
            base.update(dm_by_group.get(k, {}))
            base.update(mcs_by_group.get(k, {}))
            base.update(wr_by_group.get(k, {}))
            base["dm_available"] = bool(base["dm_pairs"] > 0)
            base["mcs_available"] = bool(base["mcs_models"] > 0)
            base["win_rate_available"] = bool(base["win_rate_pairs"] > 0)
            base["statistics_ready"] = bool(
                base["quality_passed_all"] and base["dm_available"] and base["mcs_available"]
            )
            rows.append(base)

        return pd.DataFrame(rows)


class QualityRunSweepSummaryGoldBuilder(GoldBuilder):
    output_table = "gold_quality_run_sweep_summary"
    requires = ("dim_run",)
    requires_gold = ("gold_oos_quality_report", "gold_quality_statistics_report")

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        quality_report = ctx.gold_outputs["gold_oos_quality_report"]
        quality_statistics = ctx.gold_outputs["gold_quality_statistics_report"]
        dim_run = snapshot.get("dim_run")

        rows: list[dict[str, object]] = []

        if not quality_report.empty and {"scope", "run_id", "passed"}.issubset(
            set(quality_report.columns)
        ):
            qr = quality_report[quality_report["scope"].astype(str) == "run_split_horizon"].copy()
            if not qr.empty:
                run_ag = (
                    qr.groupby("run_id", dropna=False)
                    .agg(
                        quality_rows=("passed", "count"),
                        quality_passed_rows=(
                            "passed",
                            lambda s: int(pd.Series(s).fillna(False).astype(bool).sum()),
                        ),
                    )
                    .reset_index()
                )
                run_ag["quality_passed_all"] = run_ag["quality_rows"] == run_ag["quality_passed_rows"]
                if not dim_run.empty and "run_id" in dim_run.columns:
                    keep = [
                        c
                        for c in [
                            "run_id",
                            "asset",
                            "parent_sweep_id",
                            "feature_set_name",
                            "config_signature",
                            "model_version",
                        ]
                        if c in dim_run.columns
                    ]
                    run_ag = run_ag.merge(
                        dim_run[keep].drop_duplicates("run_id"), on="run_id", how="left"
                    )
                run_ag["scope"] = "run"
                rows.extend(run_ag.to_dict(orient="records"))

        if not quality_statistics.empty and {"asset", "parent_sweep_id"}.issubset(
            set(quality_statistics.columns)
        ):
            qs = quality_statistics.copy()
            sweep_ag = (
                qs.groupby(["asset", "parent_sweep_id"], dropna=False)
                .agg(
                    group_rows=("statistics_ready", "count"),
                    statistics_ready_rows=(
                        "statistics_ready",
                        lambda s: int(pd.Series(s).fillna(False).astype(bool).sum()),
                    ),
                    dm_available_rows=(
                        "dm_available",
                        lambda s: int(pd.Series(s).fillna(False).astype(bool).sum()),
                    ),
                    mcs_available_rows=(
                        "mcs_available",
                        lambda s: int(pd.Series(s).fillna(False).astype(bool).sum()),
                    ),
                    win_rate_available_rows=(
                        "win_rate_available",
                        lambda s: int(pd.Series(s).fillna(False).astype(bool).sum()),
                    ),
                )
                .reset_index()
            )
            sweep_ag["scope"] = "sweep"
            rows.extend(sweep_ag.to_dict(orient="records"))

        return pd.DataFrame(rows)
