from __future__ import annotations

import pandas as pd
import pytest

from src.domain.services.multi_horizon_prediction_persister import (
    IncompletePredictionWindowError,
    MultiHorizonPredictionPersister,
    RunContext,
)
from src.infrastructure.schemas.analytics_store_schema import (
    FACT_OOS_PREDICTIONS_SCHEMA,
)


def _make_run_context(**overrides) -> RunContext:
    base = dict(
        schema_version=2,
        run_id="run-1",
        model_version="tft-v1",
        asset="AAPL",
        feature_set_name="BASELINE_FEATURES",
        config_signature="sig-1",
        split="test",
        fold="none",
        seed=42,
    )
    base.update(overrides)
    return RunContext(**base)


def _make_daily_timestamps(n: int, start: str = "2024-01-01") -> list[pd.Timestamp]:
    return [pd.Timestamp(t, tz="UTC") for t in pd.date_range(start, periods=n, freq="D")]


def test_build_record_h1_returns_next_day_target_ts() -> None:
    ts = _make_daily_timestamps(30)
    record = MultiHorizonPredictionPersister.build_record(
        decision_idx=10, h=1,
        y_true=0.01, y_pred=0.012,
        q10=None, q50=None, q90=None,
        q10_post=None, q50_post=None, q90_post=None,
        guardrail_applied=False,
        dataset_timestamps=ts,
        run_context=_make_run_context(),
    )
    assert record.timestamp_utc == ts[10].isoformat()
    assert record.target_timestamp_utc == ts[11].isoformat()
    assert record.horizon == 1
    assert record.decision_idx == 10


def test_build_record_h7_returns_7_ahead() -> None:
    ts = _make_daily_timestamps(30)
    record = MultiHorizonPredictionPersister.build_record(
        decision_idx=10, h=7,
        y_true=0.005, y_pred=0.007,
        q10=None, q50=None, q90=None,
        q10_post=None, q50_post=None, q90_post=None,
        guardrail_applied=False,
        dataset_timestamps=ts,
        run_context=_make_run_context(),
    )
    assert record.timestamp_utc == ts[10].isoformat()
    assert record.target_timestamp_utc == ts[17].isoformat()
    assert record.horizon == 7


def test_y_true_is_propagated_unchanged() -> None:
    ts = _make_daily_timestamps(30)
    record = MultiHorizonPredictionPersister.build_record(
        decision_idx=5, h=3,
        y_true=-0.0234, y_pred=0.0100,
        q10=None, q50=None, q90=None,
        q10_post=None, q50_post=None, q90_post=None,
        guardrail_applied=False,
        dataset_timestamps=ts,
        run_context=_make_run_context(),
    )
    assert record.y_true == pytest.approx(-0.0234)
    assert record.y_pred == pytest.approx(0.0100)
    assert record.error == pytest.approx(0.0100 - (-0.0234))
    assert record.abs_error == pytest.approx(abs(0.0100 - (-0.0234)))
    assert record.sq_error == pytest.approx((0.0100 - (-0.0234)) ** 2)


def test_boundary_raises_incomplete_window() -> None:
    ts = _make_daily_timestamps(10)
    with pytest.raises(IncompletePredictionWindowError):
        MultiHorizonPredictionPersister.build_record(
            decision_idx=9, h=1,
            y_true=0.0, y_pred=0.0,
            q10=None, q50=None, q90=None,
            q10_post=None, q50_post=None, q90_post=None,
            guardrail_applied=False,
            dataset_timestamps=ts,
            run_context=_make_run_context(),
        )


def test_boundary_raises_when_decision_idx_at_last_position() -> None:
    """decision_idx == len(ts) - 1 with h>=1 must raise (target_pos out of range)."""
    ts = _make_daily_timestamps(10)
    with pytest.raises(IncompletePredictionWindowError):
        MultiHorizonPredictionPersister.build_record(
            decision_idx=9, h=2,
            y_true=0.0, y_pred=0.0,
            q10=None, q50=None, q90=None,
            q10_post=None, q50_post=None, q90_post=None,
            guardrail_applied=False,
            dataset_timestamps=ts,
            run_context=_make_run_context(),
        )


def test_h_must_be_positive() -> None:
    ts = _make_daily_timestamps(10)
    with pytest.raises(ValueError, match="h must be >= 1"):
        MultiHorizonPredictionPersister.build_record(
            decision_idx=5, h=0,
            y_true=0.0, y_pred=0.0,
            q10=None, q50=None, q90=None,
            q10_post=None, q50_post=None, q90_post=None,
            guardrail_applied=False,
            dataset_timestamps=ts,
            run_context=_make_run_context(),
        )


def test_decision_idx_must_be_non_negative() -> None:
    ts = _make_daily_timestamps(10)
    with pytest.raises(ValueError, match="decision_idx must be >= 0"):
        MultiHorizonPredictionPersister.build_record(
            decision_idx=-1, h=1,
            y_true=0.0, y_pred=0.0,
            q10=None, q50=None, q90=None,
            q10_post=None, q50_post=None, q90_post=None,
            guardrail_applied=False,
            dataset_timestamps=ts,
            run_context=_make_run_context(),
        )


def test_trading_day_arithmetic_skips_weekend_gaps() -> None:
    """target_ts is looked up in dataset_timestamps; calendar gaps (weekends/holidays)
    are skipped naturally because the dataset is itself trading-day indexed.
    """
    raw = pd.to_datetime([
        "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
        "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
    ], utc=True).tolist()
    record = MultiHorizonPredictionPersister.build_record(
        decision_idx=3, h=1,
        y_true=0.0, y_pred=0.0,
        q10=None, q50=None, q90=None,
        q10_post=None, q50_post=None, q90_post=None,
        guardrail_applied=False,
        dataset_timestamps=raw,
        run_context=_make_run_context(),
    )
    assert pd.Timestamp(record.timestamp_utc) == raw[3]
    assert pd.Timestamp(record.target_timestamp_utc) == raw[4]
    diff_days = (pd.Timestamp(record.target_timestamp_utc) - pd.Timestamp(record.timestamp_utc)).days
    assert diff_days == 3


def test_run_context_propagated() -> None:
    ts = _make_daily_timestamps(30)
    rc = _make_run_context(
        run_id="custom-run", model_version="v9",
        asset="MSFT", feature_set_name="FUNDAMENTAL_FEATURES",
        config_signature="sig-xyz", split="val", fold="fold-3", seed=20260517,
    )
    record = MultiHorizonPredictionPersister.build_record(
        decision_idx=1, h=2,
        y_true=0.0, y_pred=0.0,
        q10=None, q50=None, q90=None,
        q10_post=None, q50_post=None, q90_post=None,
        guardrail_applied=False,
        dataset_timestamps=ts,
        run_context=rc,
    )
    assert record.run_id == "custom-run"
    assert record.model_version == "v9"
    assert record.asset == "MSFT"
    assert record.feature_set_name == "FUNDAMENTAL_FEATURES"
    assert record.config_signature == "sig-xyz"
    assert record.split == "val"
    assert record.fold == "fold-3"
    assert record.seed == 20260517


def test_quantile_fields_preserved_when_provided() -> None:
    ts = _make_daily_timestamps(30)
    record = MultiHorizonPredictionPersister.build_record(
        decision_idx=2, h=1,
        y_true=0.0, y_pred=0.0,
        q10=-0.01, q50=0.0, q90=0.01,
        q10_post=-0.01, q50_post=0.0, q90_post=0.01,
        guardrail_applied=False,
        dataset_timestamps=ts,
        run_context=_make_run_context(),
    )
    assert record.quantile_p10 == -0.01
    assert record.quantile_p50 == 0.0
    assert record.quantile_p90 == 0.01
    assert record.quantile_p10_post_guardrail == -0.01
    assert record.quantile_p50_post_guardrail == 0.0
    assert record.quantile_p90_post_guardrail == 0.01
    assert record.quantile_guardrail_applied == 0


def test_quantile_guardrail_applied_encoded_as_int() -> None:
    ts = _make_daily_timestamps(30)
    record = MultiHorizonPredictionPersister.build_record(
        decision_idx=2, h=1,
        y_true=0.0, y_pred=0.0,
        q10=0.0, q50=0.0, q90=0.0,
        q10_post=0.0, q50_post=0.0, q90_post=0.0,
        guardrail_applied=True,
        dataset_timestamps=ts,
        run_context=_make_run_context(),
    )
    assert record.quantile_guardrail_applied == 1
    assert isinstance(record.quantile_guardrail_applied, int)


def test_year_reflects_decision_day() -> None:
    """Per Opcao (a), year column reflects decision_day year (not target year)."""
    ts = [pd.Timestamp(t, tz="UTC") for t in pd.date_range("2023-12-29", periods=10, freq="D")]
    record = MultiHorizonPredictionPersister.build_record(
        decision_idx=0, h=5,
        y_true=0.0, y_pred=0.0,
        q10=None, q50=None, q90=None,
        q10_post=None, q50_post=None, q90_post=None,
        guardrail_applied=False,
        dataset_timestamps=ts,
        run_context=_make_run_context(),
    )
    assert record.year == 2023
    assert pd.Timestamp(record.target_timestamp_utc).year == 2024


def test_to_dict_keys_include_all_schema_required_columns() -> None:
    """PredictionRecord.to_dict() must include every column required by FACT_OOS_PREDICTIONS_SCHEMA."""
    ts = _make_daily_timestamps(30)
    record = MultiHorizonPredictionPersister.build_record(
        decision_idx=2, h=1,
        y_true=0.0, y_pred=0.0,
        q10=0.0, q50=0.0, q90=0.0,
        q10_post=0.0, q50_post=0.0, q90_post=0.0,
        guardrail_applied=False,
        dataset_timestamps=ts,
        run_context=_make_run_context(),
    )
    row = record.to_dict()
    for col in FACT_OOS_PREDICTIONS_SCHEMA.required_columns:
        assert col in row, f"missing required column in record.to_dict(): {col}"


def test_decision_idx_column_present_in_record() -> None:
    """Materializa o invariante Opcao (a) no schema (Gap 6 closure)."""
    ts = _make_daily_timestamps(30)
    record = MultiHorizonPredictionPersister.build_record(
        decision_idx=7, h=3,
        y_true=0.0, y_pred=0.0,
        q10=None, q50=None, q90=None,
        q10_post=None, q50_post=None, q90_post=None,
        guardrail_applied=False,
        dataset_timestamps=ts,
        run_context=_make_run_context(),
    )
    assert "decision_idx" in record.to_dict()
    assert record.decision_idx == 7


def test_record_is_immutable() -> None:
    ts = _make_daily_timestamps(30)
    record = MultiHorizonPredictionPersister.build_record(
        decision_idx=1, h=1,
        y_true=0.0, y_pred=0.0,
        q10=None, q50=None, q90=None,
        q10_post=None, q50_post=None, q90_post=None,
        guardrail_applied=False,
        dataset_timestamps=ts,
        run_context=_make_run_context(),
    )
    with pytest.raises((AttributeError, Exception)):
        record.y_true = 1.0  # type: ignore[misc]


def test_keyword_only_signature() -> None:
    """All build_record args must be keyword-only (matches QuantileGuardrailService pattern)."""
    ts = _make_daily_timestamps(30)
    rc = _make_run_context()
    with pytest.raises(TypeError):
        MultiHorizonPredictionPersister.build_record(  # type: ignore[misc]
            1, 1, 0.0, 0.0, None, None, None, None, None, None, False, ts, rc,
        )
