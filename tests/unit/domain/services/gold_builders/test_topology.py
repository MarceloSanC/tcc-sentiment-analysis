"""Topology guard for `build_default_registry()` registration order.

R-22.2.bis: builders may declare `requires_gold = (...)` referencing
gold tables produced by Tier 1/2 builders earlier in the registry.
The orchestrator enforces this at execute() time by raising
`GoldBuilderRequirementError` when an upstream is missing, but
catching the bug at *registration* time (i.e., this static test) is
faster and obviates a real run.

The test rebuilds the registry, walks it in registration order, and
asserts that every `requires_gold` dependency was registered strictly
before the consumer. Violations indicate a reordering bug in
`build_default_registry()`.
"""
from __future__ import annotations

import pytest

from src.domain.services.gold_builders import (
    build_default_registry,
)


def test_registry_topology_respects_requires_gold() -> None:
    """For each builder B with `requires_gold=(X, ...)`, X must appear
    in the registry strictly before B. R-22.2.bis Opcao A+B hibrida.
    """
    reg = build_default_registry()
    seen: set[str] = set()
    for b in reg:
        for dep in b.requires_gold:
            assert dep in seen, (
                f"{b.output_table} declares requires_gold={b.requires_gold} "
                f"but {dep} is not yet registered. Registration order in "
                f"build_default_registry() violates topology."
            )
        seen.add(b.output_table)


def test_registry_emits_25_gold_builders() -> None:
    """Sanity: the migration produced exactly 25 builders (verified via
    grep of the monolith pre-R-22; the registry size locks the
    bit-identical contract per ADR-0005)."""
    assert len(build_default_registry()) == 25


def test_registry_output_tables_match_monolith_emit_sequence() -> None:
    """The 25 output_tables, in order, must match the parquet emission
    sequence of the legacy `RefreshAnalyticsStoreUseCase.execute()`.
    Locks the byte-identical contract (smoke regression in
    `tests/integration/test_gold_builders_byte_identical_archive.py`
    relies on this order to diff per-table parquets).
    """
    expected = [
        "gold_runs_long",
        "gold_ranking_by_config",
        "gold_consistency_topk",
        "gold_ic95_by_config_metric",
        "gold_feature_set_impact",
        "gold_oos_consolidated",
        "gold_prediction_metrics_by_run_split_horizon",
        "gold_quantile_guardrail_audit",
        "gold_quantile_degeneracy_report",
        "gold_prediction_metrics_by_config",
        "gold_prediction_metrics_by_horizon",
        "gold_prediction_calibration",
        "gold_prediction_risk",
        "gold_prediction_generalization_gap",
        "gold_prediction_robustness_by_horizon",
        "gold_feature_impact_by_horizon",
        "gold_feature_contrib_local_summary",
        "gold_oos_quality_report",
        "gold_dm_pairwise_results",
        "gold_mcs_results",
        "gold_win_rate_pairwise_results",
        "gold_paired_oos_intersection_by_horizon",
        "gold_model_decision_final",
        "gold_quality_statistics_report",
        "gold_quality_run_sweep_summary",
    ]
    actual = [b.output_table for b in build_default_registry()]
    assert actual == expected, (
        f"Registry order diverges from monolith emit sequence; "
        f"the byte-identical contract under ADR-0005 §Consequences "
        f"requires this exact sequence. Diff:\n"
        f"expected: {expected}\n  actual: {actual}"
    )


@pytest.mark.parametrize(
    "tier_2_output, expected_requires_gold",
    [
        ("gold_prediction_metrics_by_config", ("gold_prediction_metrics_by_run_split_horizon",)),
        ("gold_prediction_metrics_by_horizon", ("gold_prediction_metrics_by_run_split_horizon",)),
        ("gold_prediction_calibration", ("gold_prediction_metrics_by_run_split_horizon",)),
        ("gold_prediction_generalization_gap", ("gold_prediction_metrics_by_run_split_horizon",)),
        ("gold_prediction_robustness_by_horizon", ("gold_prediction_metrics_by_run_split_horizon",)),
        ("gold_feature_impact_by_horizon", ("gold_prediction_metrics_by_run_split_horizon",)),
    ],
)
def test_tier_2_builders_declare_expected_requires_gold(
    tier_2_output: str, expected_requires_gold: tuple[str, ...]
) -> None:
    """Each Tier 2 builder must declare its gold dependency so the
    orchestrator can fail fast when registration order is wrong."""
    reg = build_default_registry()
    builder = next(b for b in reg if b.output_table == tier_2_output)
    assert builder.requires_gold == expected_requires_gold


def test_model_decision_final_requires_all_seven_upstream_gold_tables() -> None:
    """Tier 3 rollup: composes 7 upstream gold tables (metrics_by_config,
    robustness, generalization_gap, dm, mcs, win_rate, paired_intersection)
    in addition to ctx.primary_quantile_contract.
    """
    reg = build_default_registry()
    builder = next(b for b in reg if b.output_table == "gold_model_decision_final")
    assert set(builder.requires_gold) == {
        "gold_prediction_metrics_by_config",
        "gold_prediction_robustness_by_horizon",
        "gold_prediction_generalization_gap",
        "gold_dm_pairwise_results",
        "gold_mcs_results",
        "gold_win_rate_pairwise_results",
        "gold_paired_oos_intersection_by_horizon",
    }


def test_quality_statistics_report_requires_quality_plus_pairwise() -> None:
    reg = build_default_registry()
    builder = next(b for b in reg if b.output_table == "gold_quality_statistics_report")
    assert set(builder.requires_gold) == {
        "gold_oos_quality_report",
        "gold_dm_pairwise_results",
        "gold_mcs_results",
        "gold_win_rate_pairwise_results",
    }


def test_quality_run_sweep_summary_requires_quality_report_and_statistics() -> None:
    reg = build_default_registry()
    builder = next(b for b in reg if b.output_table == "gold_quality_run_sweep_summary")
    assert set(builder.requires_gold) == {
        "gold_oos_quality_report",
        "gold_quality_statistics_report",
    }
