from __future__ import annotations

import math

import pandas as pd
import pytest

from src.domain.services.dm_tft_vs_baseline import (
    build_loss_diff_series,
    compute_dm_family,
    compute_dm_hln_hac,
)
from src.domain.services.fold_dedup_resolver import FoldPeriod


def _row(run_id: str, ts: str, loss: float, *, horizon: int = 1) -> dict[str, object]:
    q = 2.0 * float(loss)
    return {
        "run_id": run_id,
        "split": "test",
        "horizon": horizon,
        "target_timestamp_utc": pd.Timestamp(ts, tz="UTC"),
        "y_true": 0.0,
        "quantile_p10_post_guardrail": q,
        "quantile_p50_post_guardrail": q,
        "quantile_p90_post_guardrail": q,
    }


def _fixtures() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, FoldPeriod]]:
    dim = pd.DataFrame(
        [
            {"run_id": "tft_wf1_s1", "fold": "wf_1", "model_version": "tft", "seed": 1},
            {"run_id": "tft_wf1_s2", "fold": "wf_1", "model_version": "tft", "seed": 2},
            {"run_id": "tft_wf2_s1", "fold": "wf_2", "model_version": "tft", "seed": 1},
            {"run_id": "tft_wf2_s2", "fold": "wf_2", "model_version": "tft", "seed": 2},
            {"run_id": "base_wf1_s1", "fold": "wf_1", "model_version": "baseline", "seed": 1},
            {"run_id": "base_wf1_s2", "fold": "wf_1", "model_version": "baseline", "seed": 2},
            {"run_id": "base_wf2_s1", "fold": "wf_2", "model_version": "baseline", "seed": 1},
            {"run_id": "base_wf2_s2", "fold": "wf_2", "model_version": "baseline", "seed": 2},
        ]
    )
    fact = pd.DataFrame(
        [
            _row("tft_wf1_s1", "2020-01-04", 0.50),
            _row("tft_wf1_s2", "2020-01-04", 0.54),
            _row("base_wf1_s1", "2020-01-04", 0.60),
            _row("base_wf1_s2", "2020-01-04", 0.64),
            _row("tft_wf1_s1", "2020-01-10", 0.90),
            _row("base_wf1_s1", "2020-01-10", 0.95),
            _row("tft_wf2_s1", "2020-01-10", 0.10),
            _row("tft_wf2_s2", "2020-01-10", 0.12),
            _row("base_wf2_s1", "2020-01-10", 0.20),
            _row("base_wf2_s2", "2020-01-10", 0.22),
        ]
    )
    fold_periods = {
        "wf_1": FoldPeriod("wf_1", pd.Timestamp("2020-01-01", tz="UTC")),
        "wf_2": FoldPeriod("wf_2", pd.Timestamp("2020-01-05", tz="UTC")),
    }
    return fact, dim, fold_periods


def test_build_loss_diff_series_dedups_fold_and_aggregates_seed_mean() -> None:
    fact, dim, fold_periods = _fixtures()
    series = build_loss_diff_series(
        fact,
        dim,
        {"tft_wf1_s1", "tft_wf1_s2", "tft_wf2_s1", "tft_wf2_s2"},
        {"base_wf1_s1", "base_wf1_s2", "base_wf2_s1", "base_wf2_s2"},
        fold_periods,
        horizon=1,
        split="test",
    )
    assert series[pd.Timestamp("2020-01-04", tz="UTC")] == pytest.approx(-0.10)
    assert series[pd.Timestamp("2020-01-10", tz="UTC")] == pytest.approx(-0.10)


def test_build_loss_diff_series_excludes_missing_common_support() -> None:
    fact, dim, fold_periods = _fixtures()
    fact = pd.concat([fact, pd.DataFrame([_row("tft_wf2_s1", "2020-02-01", 0.1)])])
    series = build_loss_diff_series(
        fact,
        dim,
        {"tft_wf2_s1"},
        {"base_wf2_s1"},
        fold_periods,
        horizon=1,
        split="test",
    )
    assert pd.Timestamp("2020-02-01", tz="UTC") not in set(series.index)


def test_build_loss_diff_series_validates_required_columns() -> None:
    fact, dim, fold_periods = _fixtures()
    with pytest.raises(ValueError, match="fact_oos missing"):
        build_loss_diff_series(
            fact.drop(columns=["y_true"]),
            dim,
            {"tft_wf1_s1"},
            {"base_wf1_s1"},
            fold_periods,
            horizon=1,
            split="test",
        )


def test_compute_dm_hln_hac_uses_horizon_lag_and_one_sided_less() -> None:
    result = compute_dm_hln_hac(pd.Series([-0.4, -0.2, -0.3, -0.5, -0.1]), horizon=7)
    assert result.hac_lag_used == 6
    assert result.hln_applied is True
    assert result.n_obs == 5
    assert result.dm_stat < 0
    assert 0 <= result.pvalue_one_sided_less <= result.pvalue_two_sided <= 1


def test_compute_dm_hln_hac_handles_zero_variance_equal_losses() -> None:
    result = compute_dm_hln_hac(pd.Series([0.0, 0.0, 0.0, 0.0, 0.0]), horizon=1)
    assert result.dm_stat == 0.0
    assert result.pvalue_two_sided == 1.0
    assert result.pvalue_one_sided_less == 0.5


def test_compute_dm_family_emits_one_row_per_horizon_and_baseline() -> None:
    fact, dim, fold_periods = _fixtures()
    out = compute_dm_family(
        fact,
        dim,
        {"tft_wf1_s1", "tft_wf1_s2", "tft_wf2_s1", "tft_wf2_s2"},
        {
            "baseline_a": {"base_wf1_s1", "base_wf1_s2", "base_wf2_s1", "base_wf2_s2"},
            "baseline_b": {"base_wf2_s1", "base_wf2_s2"},
        },
        fold_periods,
        horizons=[1, 7],
    )
    assert len(out) == 4
    assert set(out["baseline_model_version"]) == {"baseline_a", "baseline_b"}
    assert set(out["horizon"]) == {1, 7}
    assert out["loss_function"].eq("pinball_loss_post_guardrail").all()


def test_compute_dm_hln_hac_returns_nan_for_insufficient_observations() -> None:
    result = compute_dm_hln_hac(pd.Series([1.0]), horizon=1)
    assert result.n_obs == 1
    assert math.isnan(result.dm_stat)

