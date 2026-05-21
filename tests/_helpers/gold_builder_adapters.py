"""Transitional bridge between legacy direct-args test signatures and
the new R-22 ``GoldBuilder`` API.

Maps the legacy ``RefreshAnalyticsStoreUseCase._build_gold_*``
direct-args call shape to ``GoldBuilder.build(snapshot, ctx)`` by
wiring DataFrames into a :class:`GoldBuilderSnapshot` plus a
:class:`BuildContext` and invoking the production builder. This keeps
the existing unit-test ergonomics without forcing the cluster files
to expose extra public functions.

Slated for removal once existing test files
(``test_refresh_analytics_store_use_case.py``,
``test_inference_to_gold_parent_sweep_id.py``) are refactored to
construct ``GoldBuilderSnapshot`` and ``BuildContext`` directly.
Tracked as a follow-up in
``docs/07_reports/option_b_execution_log_2026-05-19.md`` under the
R-22 fix-up section. The helpers live under ``tests/_helpers/`` and
are NOT imported by production code.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from src.domain.services.gold_builders import (
    BuildContext,
    DmPairwiseResultsGoldBuilder,
    FeatureContribLocalSummaryGoldBuilder,
    FeatureImpactByHorizonGoldBuilder,
    GoldBuilderSnapshot,
    McsResultsGoldBuilder,
    OosConsolidatedGoldBuilder,
    OosQualityReportGoldBuilder,
    PairedOosIntersectionByHorizonGoldBuilder,
    PredictionCalibrationGoldBuilder,
    PredictionGeneralizationGapGoldBuilder,
    PredictionMetricsByConfigGoldBuilder,
    PredictionMetricsByHorizonGoldBuilder,
    PredictionMetricsByRunSplitHorizonGoldBuilder,
    PredictionRiskGoldBuilder,
    PredictionRobustnessByHorizonGoldBuilder,
    QualityRunSweepSummaryGoldBuilder,
    QualityStatisticsReportGoldBuilder,
    QuantileDegeneracyReportGoldBuilder,
    QuantileGuardrailAuditGoldBuilder,
    WinRatePairwiseResultsGoldBuilder,
)
from src.domain.services.gold_builders.confidence import _build_model_decision_final
from src.domain.services.gold_builders.descriptive import (
    _build_feature_set_impact_from_base,
    _build_ic95_from_base,
)
from src.domain.services.gold_builders.ranking import (
    _base_join_runs_split_metrics,
    _build_consistency_topk_from_base,
    _build_ranking_by_config_from_base,
    _build_runs_long_from_base,
)
from src.domain.services.quantile_contract_analyzer import QuantileDegeneracyThresholds


def _snapshot(**tables: pd.DataFrame) -> GoldBuilderSnapshot:
    return GoldBuilderSnapshot(tables=dict(tables))


# ---------------- ranking + descriptive (base-keyed) cluster ----------------


def build_gold_runs_long(base: pd.DataFrame) -> pd.DataFrame:
    return _build_runs_long_from_base(base)


def build_gold_ranking_by_config(base: pd.DataFrame) -> pd.DataFrame:
    return _build_ranking_by_config_from_base(base)


def build_gold_consistency_topk(base: pd.DataFrame) -> pd.DataFrame:
    return _build_consistency_topk_from_base(base)


def build_gold_ic95(base: pd.DataFrame) -> pd.DataFrame:
    return _build_ic95_from_base(base)


def build_gold_feature_set_impact(base: pd.DataFrame) -> pd.DataFrame:
    return _build_feature_set_impact_from_base(base)


# ---------------- quantile / pairwise / descriptive (direct-DF input) ----


def build_gold_prediction_metrics_by_run_split_horizon(
    dim_run: pd.DataFrame,
    fact_oos_predictions: pd.DataFrame,
    fact_config: pd.DataFrame | None = None,
    *,
    quantile_columns: tuple[str, str, str] | None = None,
) -> pd.DataFrame:
    """Test adapter for the run/split/horizon metrics builder.

    The monolith static method accepted an optional `quantile_columns`
    overriding which quantile triple to consume — used by callers that
    wanted a single contract (raw OR post_guardrail) rather than the
    full raw+post output. The new builder always emits raw+post; for
    single-contract callers we shell out to the internal helper.
    """
    if quantile_columns is not None:
        from src.domain.services.gold_builders.quantile import (
            _build_metrics_single_contract,
        )

        return _build_metrics_single_contract(
            dim_run,
            fact_oos_predictions,
            fact_config,
            quantile_columns=quantile_columns,
        )
    snap = _snapshot(
        dim_run=dim_run,
        fact_oos_predictions=fact_oos_predictions,
        fact_config=fact_config if fact_config is not None else pd.DataFrame(),
    )
    return PredictionMetricsByRunSplitHorizonGoldBuilder().build(snap, BuildContext())


def build_gold_quantile_guardrail_audit(
    dim_run: pd.DataFrame,
    fact_oos_predictions: pd.DataFrame,
    fact_config: pd.DataFrame | None = None,
) -> pd.DataFrame:
    snap = _snapshot(
        dim_run=dim_run,
        fact_oos_predictions=fact_oos_predictions,
        fact_config=fact_config if fact_config is not None else pd.DataFrame(),
    )
    return QuantileGuardrailAuditGoldBuilder().build(snap, BuildContext())


def build_gold_quantile_degeneracy_report(
    fact_oos_predictions: pd.DataFrame,
    fact_config: pd.DataFrame,
    *,
    thresholds: QuantileDegeneracyThresholds | None = None,
) -> pd.DataFrame:
    snap = _snapshot(fact_oos_predictions=fact_oos_predictions, fact_config=fact_config)
    return QuantileDegeneracyReportGoldBuilder(thresholds=thresholds).build(snap, BuildContext())


def build_gold_prediction_metrics_by_config(metrics_run_split_h: pd.DataFrame) -> pd.DataFrame:
    ctx = BuildContext()
    ctx.gold_outputs["gold_prediction_metrics_by_run_split_horizon"] = metrics_run_split_h
    return PredictionMetricsByConfigGoldBuilder().build(_snapshot(), ctx)


def build_gold_prediction_metrics_by_horizon(metrics_run_split_h: pd.DataFrame) -> pd.DataFrame:
    ctx = BuildContext()
    ctx.gold_outputs["gold_prediction_metrics_by_run_split_horizon"] = metrics_run_split_h
    return PredictionMetricsByHorizonGoldBuilder().build(_snapshot(), ctx)


def build_gold_prediction_calibration(metrics_run_split_h: pd.DataFrame) -> pd.DataFrame:
    ctx = BuildContext()
    ctx.gold_outputs["gold_prediction_metrics_by_run_split_horizon"] = metrics_run_split_h
    return PredictionCalibrationGoldBuilder().build(_snapshot(), ctx)


def build_gold_prediction_risk(
    dim_run: pd.DataFrame, fact_oos_predictions: pd.DataFrame
) -> pd.DataFrame:
    snap = _snapshot(dim_run=dim_run, fact_oos_predictions=fact_oos_predictions)
    return PredictionRiskGoldBuilder().build(snap, BuildContext())


def build_gold_oos_consolidated(
    dim_run: pd.DataFrame, fact_oos_predictions: pd.DataFrame
) -> pd.DataFrame:
    snap = _snapshot(dim_run=dim_run, fact_oos_predictions=fact_oos_predictions)
    return OosConsolidatedGoldBuilder().build(snap, BuildContext())


def build_gold_oos_quality_report(
    dim_run: pd.DataFrame, fact_oos_predictions: pd.DataFrame
) -> pd.DataFrame:
    snap = _snapshot(dim_run=dim_run, fact_oos_predictions=fact_oos_predictions)
    return OosQualityReportGoldBuilder().build(snap, BuildContext())


def build_gold_dm_pairwise_results(
    dim_run: pd.DataFrame, fact_oos_predictions: pd.DataFrame
) -> pd.DataFrame:
    snap = _snapshot(dim_run=dim_run, fact_oos_predictions=fact_oos_predictions)
    return DmPairwiseResultsGoldBuilder().build(snap, BuildContext())


def build_gold_mcs_results(
    dim_run: pd.DataFrame, fact_oos_predictions: pd.DataFrame
) -> pd.DataFrame:
    snap = _snapshot(dim_run=dim_run, fact_oos_predictions=fact_oos_predictions)
    return McsResultsGoldBuilder().build(snap, BuildContext())


def build_gold_paired_oos_intersection_by_horizon(
    dim_run: pd.DataFrame,
    fact_oos_predictions: pd.DataFrame,
    *,
    target_horizons: tuple[int, ...] = (1, 7, 30),
) -> pd.DataFrame:
    snap = _snapshot(dim_run=dim_run, fact_oos_predictions=fact_oos_predictions)
    return PairedOosIntersectionByHorizonGoldBuilder(target_horizons=target_horizons).build(
        snap, BuildContext()
    )


def build_gold_win_rate_pairwise_results(
    dim_run: pd.DataFrame, fact_oos_predictions: pd.DataFrame
) -> pd.DataFrame:
    snap = _snapshot(dim_run=dim_run, fact_oos_predictions=fact_oos_predictions)
    return WinRatePairwiseResultsGoldBuilder().build(snap, BuildContext())


def build_gold_prediction_generalization_gap(metrics_run_split_h: pd.DataFrame) -> pd.DataFrame:
    ctx = BuildContext()
    ctx.gold_outputs["gold_prediction_metrics_by_run_split_horizon"] = metrics_run_split_h
    return PredictionGeneralizationGapGoldBuilder().build(_snapshot(), ctx)


def build_gold_prediction_robustness_by_horizon(metrics_run_split_h: pd.DataFrame) -> pd.DataFrame:
    ctx = BuildContext()
    ctx.gold_outputs["gold_prediction_metrics_by_run_split_horizon"] = metrics_run_split_h
    return PredictionRobustnessByHorizonGoldBuilder().build(_snapshot(), ctx)


def build_gold_feature_impact_by_horizon(
    fact_model_artifacts: pd.DataFrame, metrics_run_split_h: pd.DataFrame
) -> pd.DataFrame:
    ctx = BuildContext()
    ctx.gold_outputs["gold_prediction_metrics_by_run_split_horizon"] = metrics_run_split_h
    snap = _snapshot(fact_model_artifacts=fact_model_artifacts)
    return FeatureImpactByHorizonGoldBuilder().build(snap, ctx)


def build_gold_feature_contrib_local_summary(
    fact_feature_contrib_local: pd.DataFrame, dim_run: pd.DataFrame
) -> pd.DataFrame:
    snap = _snapshot(
        fact_feature_contrib_local=fact_feature_contrib_local, dim_run=dim_run
    )
    return FeatureContribLocalSummaryGoldBuilder().build(snap, BuildContext())


def build_gold_model_decision_final(
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
    return _build_model_decision_final(
        metrics_by_config=metrics_by_config,
        robustness_by_horizon=robustness_by_horizon,
        generalization_gap=generalization_gap,
        dm_results=dm_results,
        mcs_results=mcs_results,
        win_rate_results=win_rate_results,
        paired_intersection=paired_intersection,
        primary_quantile_contract=primary_quantile_contract,
    )


def build_gold_quality_statistics_report(
    quality_report: pd.DataFrame,
    dm_results: pd.DataFrame,
    mcs_results: pd.DataFrame,
    win_rate_results: pd.DataFrame,
) -> pd.DataFrame:
    ctx = BuildContext()
    ctx.gold_outputs["gold_oos_quality_report"] = quality_report
    ctx.gold_outputs["gold_dm_pairwise_results"] = dm_results
    ctx.gold_outputs["gold_mcs_results"] = mcs_results
    ctx.gold_outputs["gold_win_rate_pairwise_results"] = win_rate_results
    return QualityStatisticsReportGoldBuilder().build(_snapshot(), ctx)


def build_gold_quality_run_sweep_summary(
    quality_report: pd.DataFrame,
    quality_statistics: pd.DataFrame,
    dim_run: pd.DataFrame,
) -> pd.DataFrame:
    ctx = BuildContext()
    ctx.gold_outputs["gold_oos_quality_report"] = quality_report
    ctx.gold_outputs["gold_quality_statistics_report"] = quality_statistics
    snap = _snapshot(dim_run=dim_run)
    return QualityRunSweepSummaryGoldBuilder().build(snap, ctx)


# Re-export the helper used by tests that synthesize a "base" join shape.
base_join_runs_split_metrics = _base_join_runs_split_metrics

__all__ = [
    "base_join_runs_split_metrics",
    "build_gold_consistency_topk",
    "build_gold_dm_pairwise_results",
    "build_gold_feature_contrib_local_summary",
    "build_gold_feature_impact_by_horizon",
    "build_gold_feature_set_impact",
    "build_gold_ic95",
    "build_gold_mcs_results",
    "build_gold_model_decision_final",
    "build_gold_oos_consolidated",
    "build_gold_oos_quality_report",
    "build_gold_paired_oos_intersection_by_horizon",
    "build_gold_prediction_calibration",
    "build_gold_prediction_generalization_gap",
    "build_gold_prediction_metrics_by_config",
    "build_gold_prediction_metrics_by_horizon",
    "build_gold_prediction_metrics_by_run_split_horizon",
    "build_gold_prediction_risk",
    "build_gold_prediction_robustness_by_horizon",
    "build_gold_quality_run_sweep_summary",
    "build_gold_quality_statistics_report",
    "build_gold_quantile_degeneracy_report",
    "build_gold_quantile_guardrail_audit",
    "build_gold_ranking_by_config",
    "build_gold_runs_long",
    "build_gold_win_rate_pairwise_results",
]
