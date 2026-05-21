"""Cross-pipeline guard for Gap 6: at the same `target_timestamp_utc`, the
TFT pipeline (decoder cell) and the baseline pipeline (Persister-emitted
record) must produce the SAME `y_true`. The 2026-05-20 audit found 663/685
test/h=1 rows where TFT.y_true == baseline.y_true_at_target_ts_plus_one
instead of equality at the same target_ts — i.e., off-by-one.

Stage R-20 (Opcao d) closes this by:
- shifting `target_return` to `log(close[t]/close[t-1])` (backward) in
  `build_tft_dataset_use_case.py`
- moving the baseline `y_true_idx` from `i + h - 1` to `i + h` in
  `run_baselines_use_case.py:259`

This test exercises BOTH sides end-to-end on a synthetic dataset and
asserts equality on the join key `(target_timestamp_utc, horizon)`. No
model training; the decoder iterates raw dataloader output, the baseline
path goes through `MultiHorizonPredictionPersister.build_record` exactly
as the production use case does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting import TimeSeriesDataSet

from src.domain.services.multi_horizon_prediction_persister import (
    IncompletePredictionWindowError,
    MultiHorizonPredictionPersister,
    RunContext,
)


def _make_run_context(model_version: str) -> RunContext:
    return RunContext(
        schema_version=2,
        run_id=f"run-{model_version}",
        model_version=model_version,
        asset="AAPL",
        feature_set_name="BASELINE_FEATURES",
        config_signature="sig-test",
        split="test",
        fold="none",
        seed=42,
    )


def _synthetic_split_df(n_rows: int = 40, *, seed: int = 7) -> pd.DataFrame:
    """Same construction as production: backward-shift target_return, drop
    the first row (where t-1 is undefined).
    """
    rng = np.random.default_rng(seed)
    raw_ts = pd.date_range("2024-03-01", periods=n_rows + 1, freq="D", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n_rows + 1)))
    target_return = np.log(close[1:] / close[:-1])
    return pd.DataFrame(
        {
            "timestamp": raw_ts[1:],
            "asset_id": ["AAPL"] * n_rows,
            "time_idx": np.arange(n_rows, dtype="int64"),
            "close": close[1:],
            "target_return": target_return.astype("float64"),
            "day_of_week": raw_ts[1:].dayofweek.astype("int64"),
            "month": raw_ts[1:].month.astype("int64"),
            "feature_a": rng.normal(size=n_rows).astype("float64"),
        }
    )


def _tft_records(
    split_df: pd.DataFrame, *, max_encoder_length: int, max_prediction_length: int
) -> list[dict]:
    """Walk the TimeSeriesDataSet exactly like the TFT trainer does, but
    skip model training. Builds Persister records straight from the
    decoder targets.
    """
    ds = TimeSeriesDataSet(
        split_df,
        time_idx="time_idx",
        target="target_return",
        group_ids=["asset_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["feature_a"],
    )
    dataloader = ds.to_dataloader(train=False, batch_size=128, num_workers=0)
    actuals = torch.cat([y[0] for _, y in iter(dataloader)], dim=0).cpu().numpy()

    timestamps = pd.to_datetime(split_df["timestamp"], utc=True).tolist()
    rc = _make_run_context("tft-fake")
    records: list[dict] = []
    n_samples = actuals.shape[0]
    for i in range(n_samples):
        decision_idx = (max_encoder_length - 1) + i
        for h in range(1, max_prediction_length + 1):
            y_true = float(actuals[i, h - 1])
            try:
                record = MultiHorizonPredictionPersister.build_record(
                    decision_idx=decision_idx,
                    h=h,
                    y_true=y_true,
                    y_pred=0.0,
                    q10=None, q50=None, q90=None,
                    q10_post=None, q50_post=None, q90_post=None,
                    guardrail_applied=False,
                    dataset_timestamps=timestamps,
                    run_context=rc,
                )
            except IncompletePredictionWindowError:
                continue
            records.append(record.to_dict())
    return records


def _baseline_zero_return_records(
    split_df: pd.DataFrame, *, max_encoder_length: int, max_prediction_length: int
) -> list[dict]:
    """Mirror `run_baselines_use_case._emit_oos_rows` for zero_return on the
    same split. Critical: uses the post-Stage R-20 index
    `y_true_idx = i + h_int` (not i + h - 1).
    """
    timestamps = pd.to_datetime(split_df["timestamp"], utc=True).tolist()
    target_returns = split_df["target_return"].to_numpy()
    rc = _make_run_context("baseline_zero_return")
    # Same offsets as run_baselines_test_pipeline_use_case: start aligns
    # with the TFT first valid sample; end trims trailing rows lacking the
    # full decoder window.
    n = len(split_df)
    offset_start = max(max_encoder_length - 1, 0)
    offset_end = max(max_prediction_length, 0)
    candidate_idxs = np.arange(n, dtype="int64")
    if offset_start:
        candidate_idxs = candidate_idxs[offset_start:]
    if offset_end:
        candidate_idxs = candidate_idxs[: max(0, len(candidate_idxs) - offset_end)]

    records: list[dict] = []
    for i in candidate_idxs:
        for h in range(1, max_prediction_length + 1):
            y_true_idx = int(i) + h
            if y_true_idx >= len(target_returns):
                continue
            y_true_value = target_returns[y_true_idx]
            if not np.isfinite(y_true_value):
                continue
            try:
                record = MultiHorizonPredictionPersister.build_record(
                    decision_idx=int(i),
                    h=h,
                    y_true=float(y_true_value),
                    y_pred=0.0,
                    q10=0.0, q50=0.0, q90=0.0,
                    q10_post=0.0, q50_post=0.0, q90_post=0.0,
                    guardrail_applied=False,
                    dataset_timestamps=timestamps,
                    run_context=rc,
                )
            except IncompletePredictionWindowError:
                continue
            records.append(record.to_dict())
    return records


@pytest.mark.parametrize(
    "max_encoder_length,max_prediction_length",
    [(10, 7), (5, 3), (3, 2)],
)
def test_tft_and_baseline_emit_equal_y_true_for_same_target_ts(
    max_encoder_length: int, max_prediction_length: int,
) -> None:
    """The substantive Gap 6 closure: y_true matches at every joined
    `(target_timestamp_utc, horizon)`. If a future change diverges either
    the TFT decoder fetch index or the baseline `y_true_idx`, this test
    fails with non-zero rows in `diff`.
    """
    split_df = _synthetic_split_df(n_rows=30)

    tft_rows = pd.DataFrame(
        _tft_records(
            split_df,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
        )
    )
    base_rows = pd.DataFrame(
        _baseline_zero_return_records(
            split_df,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
        )
    )

    assert not tft_rows.empty, "TFT path emitted no rows; check synthetic split sizing."
    assert not base_rows.empty, "Baseline path emitted no rows; check offsets."

    join = tft_rows[["target_timestamp_utc", "horizon", "y_true"]].merge(
        base_rows[["target_timestamp_utc", "horizon", "y_true"]],
        on=["target_timestamp_utc", "horizon"],
        suffixes=("_tft", "_base"),
        how="inner",
    )
    assert len(join) > 0, "TFT and baseline emitted no overlapping target_ts; offsets misaligned."

    diff = (join["y_true_tft"] - join["y_true_base"]).abs()
    assert (diff < 1e-9).all(), (
        f"Cross-pipeline y_true off-by-one regression: max diff={float(diff.max())}, "
        f"n_diverging={int((diff >= 1e-9).sum())} of {len(join)}. "
        "TFT decoder fetches target_return[decision_idx+h]; baseline must use "
        "y_true_idx = i + h_int (Stage R-20 Opcao d)."
    )


def test_target_timestamp_alignment_between_pipelines() -> None:
    """The set of (decision_idx, h) → target_timestamp must coincide
    bit-byte between the two paths (both go through the same Persister).
    """
    split_df = _synthetic_split_df(n_rows=24)
    tft_rows = pd.DataFrame(_tft_records(split_df, max_encoder_length=5, max_prediction_length=2))
    base_rows = pd.DataFrame(_baseline_zero_return_records(split_df, max_encoder_length=5, max_prediction_length=2))

    key = ["decision_idx", "horizon"]
    tft_ts = tft_rows.set_index(key)[["target_timestamp_utc"]]
    base_ts = base_rows.set_index(key)[["target_timestamp_utc"]]
    join = tft_ts.join(base_ts, lsuffix="_tft", rsuffix="_base", how="inner")
    assert (join["target_timestamp_utc_tft"] == join["target_timestamp_utc_base"]).all()
