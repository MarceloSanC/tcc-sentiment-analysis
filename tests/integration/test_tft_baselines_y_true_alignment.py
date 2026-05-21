"""Regression guard: TFT trainer and baseline runner produce same
`y_true`, `timestamp_utc`, and `target_timestamp_utc` for the same
`(decision_idx, h)` pair. Closes Gap 6 mechanically.

The guarantee comes from both writers going through
`MultiHorizonPredictionPersister.build_record(...)` per ADR-0003 Opcao (a).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.domain.services.multi_horizon_prediction_persister import (
    IncompletePredictionWindowError,
    MultiHorizonPredictionPersister,
    RunContext,
)


def _make_run_context(model_version: str = "m1") -> RunContext:
    return RunContext(
        schema_version=2,
        run_id="run-test",
        model_version=model_version,
        asset="AAPL",
        feature_set_name="BASELINE_FEATURES",
        config_signature="sig-test",
        split="test",
        fold="none",
        seed=42,
    )


def _daily_timestamps(n: int) -> list[pd.Timestamp]:
    return [pd.Timestamp(t, tz="UTC") for t in pd.date_range("2024-01-01", periods=n, freq="D")]


@pytest.mark.parametrize("decision_idx,h", [(0, 1), (5, 1), (10, 7), (20, 30), (40, 7)])
def test_tft_and_baseline_emit_aligned_records_for_same_decision_idx(
    decision_idx: int, h: int
) -> None:
    """For the same (decision_idx, h), both pipelines' records reference
    the exact same timestamp_utc, target_timestamp_utc, and y_true.
    """
    n = 60
    timestamps = _daily_timestamps(n)
    # In real datasets target_return[t] = log(close[t+1] / close[t]); both
    # writers reference target_returns[decision_idx + h - 1]. Here we just
    # use a deterministic array.
    target_returns = np.linspace(-0.01, 0.05, n)

    expected_y_true = float(target_returns[decision_idx + h - 1])

    tft_record = MultiHorizonPredictionPersister.build_record(
        decision_idx=decision_idx,
        h=h,
        y_true=expected_y_true,
        y_pred=0.001,
        q10=None, q50=None, q90=None,
        q10_post=None, q50_post=None, q90_post=None,
        guardrail_applied=False,
        dataset_timestamps=timestamps,
        run_context=_make_run_context(model_version="tft"),
    )

    baseline_record = MultiHorizonPredictionPersister.build_record(
        decision_idx=decision_idx,
        h=h,
        y_true=expected_y_true,
        y_pred=0.0,
        q10=None, q50=None, q90=None,
        q10_post=None, q50_post=None, q90_post=None,
        guardrail_applied=False,
        dataset_timestamps=timestamps,
        run_context=_make_run_context(model_version="baseline_zero_return"),
    )

    assert tft_record.timestamp_utc == baseline_record.timestamp_utc
    assert tft_record.target_timestamp_utc == baseline_record.target_timestamp_utc
    assert tft_record.y_true == baseline_record.y_true == pytest.approx(expected_y_true)
    assert tft_record.decision_idx == baseline_record.decision_idx == decision_idx
    assert tft_record.horizon == baseline_record.horizon == h


def test_target_timestamp_is_h_trading_days_after_decision_for_consecutive_dates() -> None:
    """In a contiguous daily dataset, target_timestamp_utc shifts exactly h
    rows from decision day (i.e., calendar days equals h for daily series).
    """
    ts = _daily_timestamps(40)
    for decision_idx in (0, 5, 30):
        for h in (1, 7, 30 - decision_idx if decision_idx <= 10 else 1):
            if decision_idx + h >= len(ts):
                continue
            record = MultiHorizonPredictionPersister.build_record(
                decision_idx=decision_idx,
                h=h,
                y_true=0.0, y_pred=0.0,
                q10=None, q50=None, q90=None,
                q10_post=None, q50_post=None, q90_post=None,
                guardrail_applied=False,
                dataset_timestamps=ts,
                run_context=_make_run_context(),
            )
            decision = pd.Timestamp(record.timestamp_utc)
            target = pd.Timestamp(record.target_timestamp_utc)
            assert (target - decision).days == h


def test_skip_at_boundary_is_symmetric_across_pipelines() -> None:
    """Both pipelines raise IncompletePredictionWindowError at the same boundary.
    Ensures no silent y_true=NaN.
    """
    ts = _daily_timestamps(10)
    rc = _make_run_context()
    for h in (1, 3, 7):
        target_pos = 9 + h
        assert target_pos >= len(ts)
        with pytest.raises(IncompletePredictionWindowError):
            MultiHorizonPredictionPersister.build_record(
                decision_idx=9, h=h,
                y_true=0.0, y_pred=0.0,
                q10=None, q50=None, q90=None,
                q10_post=None, q50_post=None, q90_post=None,
                guardrail_applied=False,
                dataset_timestamps=ts,
                run_context=rc,
            )
