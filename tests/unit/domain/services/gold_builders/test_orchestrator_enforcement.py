"""Orchestrator-level enforcement tests for R-22.

Guards three failure modes the registry was redesigned to surface
fast:

1. A builder requires a silver/dim table that is not in the snapshot.
2. A Tier 2/3 builder requires a gold table that is not yet in
   `ctx.gold_outputs` (typically caused by a registration order bug
   or by an upstream builder that the user filtered out).
3. The `primary_quantile_contract` context flag is propagated to
   builders that consume it (specifically `ModelDecisionFinalGoldBuilder`).

The tests construct a `GoldBuildersRegistry` directly (DI via the
`registry=` constructor param) so they do not exercise the full
silver-loading machinery — only the orchestrator's enforcement
pre-check loop.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.domain.services.gold_builders import (
    BuildContext,
    GoldBuilder,
    GoldBuilderRequirementError,
    GoldBuilderSnapshot,
    GoldBuildersRegistry,
    ModelDecisionFinalGoldBuilder,
    OosQualityReportGoldBuilder,
    PredictionMetricsByConfigGoldBuilder,
    RunsLongGoldBuilder,
)
from src.use_cases.refresh_analytics_store_use_case import RefreshAnalyticsStoreUseCase


def _make_use_case(tmp_path: Path, registry: GoldBuildersRegistry) -> RefreshAnalyticsStoreUseCase:
    silver = tmp_path / "silver"
    silver.mkdir(parents=True, exist_ok=True)
    gold = tmp_path / "gold"
    return RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold,
        registry=registry,
    )


def test_orchestrator_raises_when_required_silver_table_absent(tmp_path: Path) -> None:
    """`RunsLongGoldBuilder` requires `dim_run` and `fact_split_metrics`.
    With an empty silver dir those tables are missing from the snapshot
    -> `GoldBuilderRequirementError` raised before build() is called.
    """
    reg = GoldBuildersRegistry()
    reg.register(RunsLongGoldBuilder())
    use_case = _make_use_case(tmp_path, reg)
    # Force the snapshot to have neither dim_run nor fact_split_metrics
    # by stubbing _load_silver_dim_snapshot to return an empty snapshot.
    use_case._load_silver_dim_snapshot = lambda _scope: GoldBuilderSnapshot(tables={})
    with pytest.raises(GoldBuilderRequirementError, match="snapshot missing tables"):
        use_case.execute()


def test_orchestrator_raises_when_required_gold_table_missing(tmp_path: Path) -> None:
    """`PredictionMetricsByConfigGoldBuilder` requires
    `gold_prediction_metrics_by_run_split_horizon` in `ctx.gold_outputs`.
    Without registering its upstream Tier 1 builder the dependency is
    absent -> `GoldBuilderRequirementError` raised with a
    'registration order bug' hint.
    """
    reg = GoldBuildersRegistry()
    reg.register(PredictionMetricsByConfigGoldBuilder())
    use_case = _make_use_case(tmp_path, reg)
    use_case._load_silver_dim_snapshot = lambda _scope: GoldBuilderSnapshot(tables={})
    with pytest.raises(
        GoldBuilderRequirementError, match="registration order bug"
    ):
        use_case.execute()


def test_orchestrator_passes_when_requirements_satisfied(tmp_path: Path) -> None:
    """Smoke: a builder with empty `requires` runs to completion even
    when the snapshot has no silver tables (the builder itself returns
    an empty DataFrame in that case)."""

    class _NoopBuilder(GoldBuilder):
        output_table = "gold_noop"
        requires = ()

        def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
            return pd.DataFrame({"x": [1]})

    reg = GoldBuildersRegistry()
    reg.register(_NoopBuilder())
    use_case = _make_use_case(tmp_path, reg)
    use_case._load_silver_dim_snapshot = lambda _scope: GoldBuilderSnapshot(tables={})
    result = use_case.execute()
    assert "gold_noop" in result.outputs


def test_oos_quality_report_builder_preserves_asset_after_dim_run_merge() -> None:
    """Regression guard for the R-E suffix fix migrated into
    `OosQualityReportGoldBuilder.build()`: when both
    fact_oos_predictions and dim_run carry `asset`, the merge must use
    `suffixes=("", "_dim")` so the fact_oos side stays canonical. If
    someone removes the suffix the report would emit asset=None,
    which downstream `gold_quality_statistics_report` would interpret
    as `quality_passed_all=False` even when every check passed.
    """
    builder = OosQualityReportGoldBuilder()
    fact = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "split": "test",
                "horizon": 1,
                "timestamp_utc": "2024-01-01T00:00:00Z",
                "target_timestamp_utc": "2024-01-02T00:00:00Z",
                "y_true": 0.0,
                "y_pred": 0.0,
            }
        ]
    )
    dim = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "parent_sweep_id": "sw1",
            }
        ]
    )
    snap = GoldBuilderSnapshot(tables={"dim_run": dim, "fact_oos_predictions": fact})
    out = builder.build(snap, BuildContext())
    run_rows = out[out["scope"] == "run_split_horizon"]
    assert (run_rows["asset"] == "AAPL").all()
    assert run_rows["asset"].notna().all()


@pytest.mark.parametrize("contract", ["raw", "post_guardrail"])
def test_model_decision_final_uses_primary_quantile_contract(contract: str) -> None:
    """Cat A: ModelDecisionFinalGoldBuilder reads `ctx.primary_quantile_contract`
    to choose which `mean_pinball_*_{raw|post_guardrail}` columns to
    promote to the contract-agnostic alias. Different contracts should
    pick different upstream columns; we verify that by injecting
    distinct values into raw vs post_guardrail and asserting which one
    surfaces.
    """
    metrics_by_config = pd.DataFrame(
        [
            {
                "asset": "AAPL",
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "split": "test",
                "horizon": 1,
                "n_runs": 1,
                "mean_rmse": 0.1,
                "std_rmse": 0.0,
                "mean_mae": 0.1,
                "std_mae": 0.0,
                "mean_directional_accuracy": 0.7,
                "std_directional_accuracy": 0.0,
                "mean_mean_pinball_raw": 9.9,
                "mean_mean_pinball_post_guardrail": 1.1,
                "mean_picp_raw": 0.99,
                "mean_picp_post_guardrail": 0.11,
                "mean_mpiw_raw": 9.0,
                "mean_mpiw_post_guardrail": 1.0,
                "mean_pinball_q10_raw": 0.0,
                "mean_pinball_q10_post_guardrail": 0.0,
                "mean_pinball_q50_raw": 0.0,
                "mean_pinball_q50_post_guardrail": 0.0,
                "mean_pinball_q90_raw": 0.0,
                "mean_pinball_q90_post_guardrail": 0.0,
            }
        ]
    )

    empty = pd.DataFrame()
    ctx = BuildContext(primary_quantile_contract=contract)
    ctx.gold_outputs.update(
        {
            "gold_prediction_metrics_by_config": metrics_by_config,
            "gold_prediction_robustness_by_horizon": empty,
            "gold_prediction_generalization_gap": empty,
            "gold_dm_pairwise_results": empty,
            "gold_mcs_results": empty,
            "gold_win_rate_pairwise_results": empty,
            "gold_paired_oos_intersection_by_horizon": empty,
        }
    )
    out = ModelDecisionFinalGoldBuilder().build(GoldBuilderSnapshot(), ctx)
    assert not out.empty
    row = out.iloc[0]
    if contract == "raw":
        assert float(row["mean_mean_pinball"]) == 9.9
        assert float(row["mean_picp"]) == 0.99
        assert float(row["mean_mpiw"]) == 9.0
    else:
        assert float(row["mean_mean_pinball"]) == 1.1
        assert float(row["mean_picp"]) == 0.11
        assert float(row["mean_mpiw"]) == 1.0
    assert row["primary_quantile_contract"] == contract
