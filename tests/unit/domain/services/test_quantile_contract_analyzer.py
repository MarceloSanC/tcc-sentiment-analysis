from __future__ import annotations

import pandas as pd

from src.domain.services.quantile_contract_analyzer import (
    QuantileBlockAThresholds,
    QuantileContractAnalyzer,
    QuantileDegeneracyThresholds,
)


def test_analyze_and_evaluate_block_a_with_valid_quantiles() -> None:
    df = pd.DataFrame(
        [
            {"quantile_p10": -0.01, "quantile_p50": 0.00, "quantile_p90": 0.01},
            {"quantile_p10": -0.02, "quantile_p50": -0.01, "quantile_p90": 0.00},
        ]
    )
    metrics = QuantileContractAnalyzer.analyze(df)
    assert metrics.crossing_bruto_count == 0
    assert metrics.negative_interval_width_count == 0

    out = QuantileContractAnalyzer.evaluate_block_a(
        metrics,
        thresholds=QuantileBlockAThresholds(
            max_crossing_bruto_rate=0.001,
            max_negative_interval_width_count=0,
            max_crossing_post_guardrail_rate=0.0,
            require_post_guardrail=False,
        ),
    )
    assert out.passed is True


def test_evaluate_block_a_fails_on_crossing_rate_threshold() -> None:
    df = pd.DataFrame(
        [
            {"quantile_p10": 0.03, "quantile_p50": 0.02, "quantile_p90": 0.01},
            {"quantile_p10": -0.02, "quantile_p50": -0.01, "quantile_p90": 0.00},
        ]
    )
    metrics = QuantileContractAnalyzer.analyze(df)
    out = QuantileContractAnalyzer.evaluate_block_a(
        metrics,
        thresholds=QuantileBlockAThresholds(max_crossing_bruto_rate=0.10),
    )
    assert out.passed is False
    assert "crossing_bruto_rate=" in out.detail


def test_filter_scope_by_parent_sweep_split_and_horizon() -> None:
    oos = pd.DataFrame(
        [
            {"run_id": "r1", "split": "test", "horizon": 1, "quantile_p10": 0.0, "quantile_p50": 0.1, "quantile_p90": 0.2},
            {"run_id": "r1", "split": "val", "horizon": 1, "quantile_p10": 0.0, "quantile_p50": 0.1, "quantile_p90": 0.2},
            {"run_id": "r2", "split": "test", "horizon": 1, "quantile_p10": 0.0, "quantile_p50": 0.1, "quantile_p90": 0.2},
        ]
    )
    dim = pd.DataFrame(
        [
            {"run_id": "r1", "parent_sweep_id": "0_2_3_explicit"},
            {"run_id": "r2", "parent_sweep_id": "0_2_1_explicit"},
        ]
    )

    scoped = QuantileContractAnalyzer.filter_scope(
        fact_oos_predictions=oos,
        dim_run=dim,
        parent_sweep_prefixes=["0_2_3_"],
        splits=["test"],
        horizons=[1],
    )
    assert len(scoped) == 1
    assert set(scoped["run_id"].astype(str)) == {"r1"}
    assert set(scoped["split"].astype(str)) == {"test"}


def test_analyze_degeneracy_prefers_fact_config_parent_sweep_and_reports_rates() -> None:
    oos = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "parent_sweep_id": "stale",
                "split": "test",
                "horizon": 1,
                "quantile_p10": 0.5,
                "quantile_p50": 0.5,
                "quantile_p90": 0.5,
            },
            {
                "run_id": "r1",
                "parent_sweep_id": "stale",
                "split": "test",
                "horizon": 1,
                "quantile_p10": 0.4,
                "quantile_p50": 0.5,
                "quantile_p90": 0.6,
            },
        ]
    )
    fact_config = pd.DataFrame(
        [{"run_id": "r1", "prediction_mode": "quantile", "parent_sweep_id": "sw_cfg"}]
    )

    metrics = QuantileContractAnalyzer.analyze_degeneracy(oos, fact_config)

    assert len(metrics) == 1
    assert metrics[0].parent_sweep_id == "sw_cfg"
    assert metrics[0].prediction_mode == "quantile"
    assert metrics[0].n_rows == 2
    assert metrics[0].p10_eq_p90_count == 1
    assert metrics[0].p10_eq_p90_rate == 0.5
    assert metrics[0].p10_eq_p50_eq_p90_count == 1
    assert metrics[0].p10_eq_p50_eq_p90_rate == 0.5


def test_evaluate_degeneracy_fails_quantile_group_with_91_25_percent_degeneracy() -> None:
    rows = []
    for idx in range(1600):
        degenerate = idx < 1460
        rows.append(
            {
                "run_id": "r1",
                "split": "test",
                "horizon": 1,
                "quantile_p10": 0.5 if degenerate else 0.4,
                "quantile_p50": 0.5,
                "quantile_p90": 0.5 if degenerate else 0.6,
            }
        )
    fact_config = pd.DataFrame(
        [{"run_id": "r1", "prediction_mode": "quantile", "parent_sweep_id": "sw1"}]
    )

    metrics = QuantileContractAnalyzer.analyze_degeneracy(pd.DataFrame(rows), fact_config)
    evaluation = QuantileContractAnalyzer.evaluate_degeneracy(
        metrics,
        thresholds=QuantileDegeneracyThresholds(
            min_rows_for_gate=1000,
            max_p10_eq_p90_rate=0.05,
        ),
    )

    assert metrics[0].p10_eq_p90_rate == 0.9125
    assert evaluation.passed is False
    assert "p10_eq_p90_rate=0.91250000" in evaluation.detail


def test_analyze_degeneracy_uses_raw_quantiles_not_post_guardrail() -> None:
    oos = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "split": "test",
                "horizon": 1,
                "quantile_p10": 0.5,
                "quantile_p50": 0.5,
                "quantile_p90": 0.5,
                "quantile_p10_post_guardrail": 0.4,
                "quantile_p50_post_guardrail": 0.5,
                "quantile_p90_post_guardrail": 0.6,
            }
            for _ in range(10)
        ]
    )
    fact_config = pd.DataFrame(
        [{"run_id": "r1", "prediction_mode": "quantile", "parent_sweep_id": "sw1"}]
    )

    metrics = QuantileContractAnalyzer.analyze_degeneracy(oos, fact_config)
    evaluation = QuantileContractAnalyzer.evaluate_degeneracy(
        metrics,
        thresholds=QuantileDegeneracyThresholds(
            min_rows_for_gate=10,
            max_p10_eq_p90_rate=0.05,
        ),
    )

    assert metrics[0].p10_eq_p90_rate == 1.0
    assert metrics[0].p10_eq_p50_eq_p90_rate == 1.0
    assert evaluation.passed is False
    assert "p10_eq_p90_rate=1.00000000" in evaluation.detail


def test_evaluate_degeneracy_ignores_point_mode_even_when_collapsed() -> None:
    oos = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "split": "test",
                "horizon": 1,
                "quantile_p10": 0.5,
                "quantile_p50": 0.5,
                "quantile_p90": 0.5,
            }
            for _ in range(1000)
        ]
    )
    fact_config = pd.DataFrame(
        [{"run_id": "r1", "prediction_mode": "point", "parent_sweep_id": "sw1"}]
    )

    evaluation = QuantileContractAnalyzer.evaluate_degeneracy(
        QuantileContractAnalyzer.analyze_degeneracy(oos, fact_config),
        thresholds=QuantileDegeneracyThresholds(min_rows_for_gate=1000, max_p10_eq_p90_rate=0.05),
    )

    assert evaluation.passed is True
    assert "point_groups_ignored=1" in evaluation.detail


def test_evaluate_degeneracy_treats_small_quantile_group_as_diagnostic_only() -> None:
    oos = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "split": "test",
                "horizon": 1,
                "quantile_p10": 0.5,
                "quantile_p50": 0.5,
                "quantile_p90": 0.5,
            }
            for _ in range(999)
        ]
    )
    fact_config = pd.DataFrame(
        [{"run_id": "r1", "prediction_mode": "quantile", "parent_sweep_id": "sw1"}]
    )

    evaluation = QuantileContractAnalyzer.evaluate_degeneracy(
        QuantileContractAnalyzer.analyze_degeneracy(oos, fact_config),
        thresholds=QuantileDegeneracyThresholds(min_rows_for_gate=1000, max_p10_eq_p90_rate=0.05),
    )

    assert evaluation.passed is True
    assert "small_quantile_groups_diagnostic_only=1" in evaluation.detail
