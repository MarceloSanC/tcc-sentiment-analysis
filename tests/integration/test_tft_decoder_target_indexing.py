"""Adapter integration test: verify that the pytorch_forecasting decoder
fetches `target_return` at the index ADR-0003 (amendado em Stage R-20,
Opcao d) prescribes for the y_true the persister will store.

Mechanic: for a TimeSeriesDataSet with max_encoder_length=L,
- sample i (no random sampling) has its encoder spanning rows [i, i+L-1].
  `decision_idx = i + L - 1` per pytorch_forecasting convention (used by
  `train_tft_model_use_case._persist_fact_oos_predictions`).
- The decoder slot at offset (h-1) corresponds to the row at position
  `decision_idx + h` in the split DataFrame.

This is the SAME index `MultiHorizonPredictionPersister.build_record(...)`
queries for `target_timestamp`. With the backward-shift formula for
`target_return` (Stage R-20 Opcao d), the row at `decision_idx + h`
also holds the y_true the baseline persists for the same
(decision_idx, h), closing Gap 6 at the source.

NO model training. Construct dataset + dataloader, iterate, assert.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pytorch_forecasting import TimeSeriesDataSet


def _synthetic_split_df(n_rows: int = 30, *, seed: int = 42) -> pd.DataFrame:
    """Synthetic split mirroring real dataset_tft shape (close, target_return,
    time_idx, asset_id, day_of_week, month).

    target_return is computed via the canonical backward-shift formula
    (`log(close[t]/close[t-1])`); first row dropped to align with how
    `build_tft_dataset_use_case` persists the column.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_rows + 1, freq="D", tz="UTC")
    # Geometric-ish close path so target_return is well-defined.
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, size=n_rows + 1)))
    target_return = np.log(close[1:] / close[:-1])
    df = pd.DataFrame(
        {
            "timestamp": timestamps[1:],
            "asset_id": ["AAPL"] * n_rows,
            "time_idx": np.arange(n_rows, dtype="int64"),
            "close": close[1:],
            "target_return": target_return.astype("float64"),
            "day_of_week": timestamps[1:].dayofweek.astype("int64"),
            "month": timestamps[1:].month.astype("int64"),
            "feature_a": rng.normal(size=n_rows).astype("float64"),
        }
    )
    return df


@pytest.mark.parametrize("max_encoder_length,max_prediction_length", [(10, 7), (5, 3), (1, 1)])
def test_decoder_target_value_matches_target_return_at_decision_plus_h(
    max_encoder_length: int, max_prediction_length: int,
) -> None:
    df = _synthetic_split_df(n_rows=30)
    ds = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target_return",
        group_ids=["asset_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["feature_a"],
    )
    dataloader = ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    actuals = torch.cat([y[0] for _, y in iter(dataloader)], dim=0).cpu().numpy()

    # For each sample, recover decision_idx by matching the encoder end.
    # The TimeSeriesDataSet emits samples sequentially per group_id; without
    # random sampling sample i has decision_idx = (max_encoder_length - 1) + i.
    target_returns = df["target_return"].to_numpy()
    n_samples = actuals.shape[0]
    assert n_samples > 0, "Synthetic split too short for given encoder/decoder lengths."

    for i in range(n_samples):
        decision_idx = (max_encoder_length - 1) + i
        for h in range(1, max_prediction_length + 1):
            expected = target_returns[decision_idx + h]
            got = float(actuals[i, h - 1])
            assert got == pytest.approx(expected, abs=1e-9), (
                f"sample={i} decision_idx={decision_idx} h={h}: "
                f"decoder y[0][{i},{h - 1}]={got} vs target_return[{decision_idx + h}]={expected}"
            )


def test_decoder_off_by_one_regression_guard() -> None:
    """If a future refactor reverts to forward `target_return` shift OR
    changes `y_true_idx = i + h - 1` in the baseline, the decoder vs
    persister-prescribed index would diverge. This test re-affirms the
    convention that closed Gap 6.
    """
    df = _synthetic_split_df(n_rows=20)
    max_encoder_length = 5
    max_prediction_length = 2
    ds = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target_return",
        group_ids=["asset_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=["feature_a"],
    )
    dataloader = ds.to_dataloader(train=False, batch_size=64, num_workers=0)
    actuals = torch.cat([y[0] for _, y in iter(dataloader)], dim=0).cpu().numpy()

    target_returns = df["target_return"].to_numpy()
    # First sample: decision_idx = max_encoder_length - 1 = 4.
    decision_idx = max_encoder_length - 1
    decoder_h1 = float(actuals[0, 0])
    # The Opcao (d) post-amendment claim: decoder y[0][0,0] equals
    # target_return at decision_idx + 1 (NOT decision_idx).
    assert decoder_h1 == pytest.approx(target_returns[decision_idx + 1], abs=1e-9)
    # Equally important: it MUST NOT equal target_return[decision_idx] (which
    # would be the pre-fix Opcao (a) baseline y_true at h=1, exposing the
    # off-by-one).
    if not np.isclose(target_returns[decision_idx], target_returns[decision_idx + 1]):
        assert decoder_h1 != pytest.approx(target_returns[decision_idx], abs=1e-12)
