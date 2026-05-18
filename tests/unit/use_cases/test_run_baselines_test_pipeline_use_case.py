"""Unit tests for RunBaselinesTestPipelineUseCase (Stage F.0.2).

Covers:
- No-op behaviour when `baselines` section is missing.
- parent_sweep_id derivation from `output_subdir`.
- Warmup offset derivation from `training_config.max_encoder_length`
  and end offset from `max_prediction_length - 1`.
- Forwarding of `replica_seeds` and `walk_forward.folds`.
- Window override per baseline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adapters.parquet_analytics_run_repository import ParquetAnalyticsRunRepository
from src.use_cases.run_baselines_test_pipeline_use_case import (
    RunBaselinesTestPipelineUseCase,
)
from src.use_cases.run_baselines_use_case import RunBaselinesUseCase


def _write_dataset(path: Path, n_days: int = 400) -> None:
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    returns = rng.normal(0.0, 0.01, size=n_days)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["AAPL"] * n_days,
            "time_idx": np.arange(n_days, dtype="int64"),
            "target_return": returns,
        }
    )
    df.to_parquet(path, index=False)


def _make_pipeline(silver: Path) -> RunBaselinesTestPipelineUseCase:
    repo = ParquetAnalyticsRunRepository(output_dir=silver)
    runner = RunBaselinesUseCase(analytics_run_repository=repo)
    return RunBaselinesTestPipelineUseCase(baselines_runner=runner)


def _load_oos(silver: Path) -> pd.DataFrame:
    base = silver / "fact_oos_predictions"
    if not base.exists():
        return pd.DataFrame()
    files = sorted(base.rglob("*.parquet"))
    return (
        pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        if files
        else pd.DataFrame()
    )


def _load_dim(silver: Path) -> pd.DataFrame:
    base = silver / "dim_run"
    files = sorted(base.rglob("*.parquet"))
    return (
        pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        if files
        else pd.DataFrame()
    )


def test_no_op_when_baselines_section_missing(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    ds = tmp_path / "dataset.parquet"
    _write_dataset(ds)
    pipeline = _make_pipeline(silver)

    config = {
        "output_subdir": "sw_no_baselines",
        "training_config": {"max_encoder_length": 30, "max_prediction_length": 1},
        "split_config": {
            "train_start": "2024-01-01",
            "train_end": "2024-06-30",
            "val_start": "2024-07-01",
            "val_end": "2024-08-31",
            "test_start": "2024-09-01",
            "test_end": "2024-10-26",
        },
    }
    result = pipeline.execute(asset="AAPL", config=config, dataset_path=ds)
    assert result.parent_sweep_id is None
    assert result.baselines_persisted_total == 0
    assert any("no 'baselines' section" in w for w in result.warnings)


def test_parent_sweep_id_derived_from_output_subdir(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    ds = tmp_path / "dataset.parquet"
    _write_dataset(ds)
    pipeline = _make_pipeline(silver)

    config = {
        "output_subdir": "sw_smoke_v1",
        "training_config": {"max_encoder_length": 10, "max_prediction_length": 1},
        "split_config": {
            "train_start": "2024-01-01",
            "train_end": "2024-06-30",
            "val_start": "2024-07-01",
            "val_end": "2024-08-31",
            "test_start": "2024-09-01",
            "test_end": "2024-10-26",
        },
        "baselines": [{"name": "zero_return", "config": {}}],
        "replica_seeds": [42],
    }
    result = pipeline.execute(asset="AAPL", config=config, dataset_path=ds)
    assert result.parent_sweep_id == "sw_smoke_v1"
    dim = _load_dim(silver)
    assert "sw_smoke_v1" in dim["parent_sweep_id"].astype(str).tolist()


def test_offset_start_aligns_with_max_encoder_length(tmp_path: Path) -> None:
    """With max_encoder_length=30, the first emitted train target_timestamp
    must be at calendar offset == 30 trading rows past the train_start row.
    The dataset uses daily frequency so trading rows == calendar rows."""
    silver = tmp_path / "silver"
    ds = tmp_path / "dataset.parquet"
    _write_dataset(ds)
    pipeline = _make_pipeline(silver)

    config = {
        "output_subdir": "sw_offset",
        "training_config": {"max_encoder_length": 30, "max_prediction_length": 1},
        "split_config": {
            "train_start": "2024-01-01",
            "train_end": "2024-06-30",
            "val_start": "2024-07-01",
            "val_end": "2024-08-31",
            "test_start": "2024-09-01",
            "test_end": "2024-10-26",
        },
        "baselines": [{"name": "zero_return", "config": {}}],
        "replica_seeds": [42],
    }
    pipeline.execute(asset="AAPL", config=config, dataset_path=ds)
    oos = _load_oos(silver)
    train_rows = oos[oos["split"] == "train"]
    assert not train_rows.empty
    first_ts = pd.Timestamp(train_rows["timestamp_utc"].min())
    # Row 0 is 2024-01-01; row 30 (offset start) is 2024-01-31.
    assert first_ts == pd.Timestamp("2024-01-31T00:00:00+00:00")


def test_offset_end_aligns_with_max_prediction_length(tmp_path: Path) -> None:
    """With max_prediction_length=7, the last 6 idxs of each split are dropped
    (matching the TFT trainer's last-decision rule for multi-horizon).
    """
    silver = tmp_path / "silver"
    ds = tmp_path / "dataset.parquet"
    _write_dataset(ds, n_days=120)
    pipeline = _make_pipeline(silver)

    config = {
        "output_subdir": "sw_end_offset",
        "training_config": {
            "max_encoder_length": 0,
            "max_prediction_length": 7,
            "evaluation_horizons": [1],
        },
        "split_config": {
            "train_start": "2024-01-01",
            "train_end": "2024-01-31",
            "val_start": "2024-02-01",
            "val_end": "2024-02-29",
            "test_start": "2024-03-01",
            "test_end": "2024-03-31",
        },
        "baselines": [{"name": "zero_return", "config": {}}],
        "replica_seeds": [42],
    }
    pipeline.execute(asset="AAPL", config=config, dataset_path=ds)
    oos = _load_oos(silver)
    val_rows = oos[oos["split"] == "val"]
    # Val span: 2024-02-01 to 2024-02-29 inclusive = 29 rows.
    # offset_end = max_pred - 1 = 6.  Expected emitted = 29 - 6 = 23.
    assert len(val_rows) == 23


def test_window_override_per_baseline(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    ds = tmp_path / "dataset.parquet"
    _write_dataset(ds, n_days=400)
    pipeline = _make_pipeline(silver)

    config = {
        "output_subdir": "sw_window",
        "training_config": {"max_encoder_length": 5, "max_prediction_length": 1},
        "split_config": {
            "train_start": "2024-01-01",
            "train_end": "2024-06-30",
            "val_start": "2024-07-01",
            "val_end": "2024-08-31",
            "test_start": "2024-09-01",
            "test_end": "2024-10-26",
        },
        "baselines": [
            {"name": "historical_mean_rolling", "config": {"window_days": 10}}
        ],
        "replica_seeds": [42],
    }
    pipeline.execute(asset="AAPL", config=config, dataset_path=ds)
    config_silver = _load_dim(silver)
    # No config emitted? fact_config_silver = ParquetAnalyticsRunRepository writes
    # to fact_config table. We can read it via load:
    config_files = sorted((silver / "fact_config").rglob("*.parquet"))
    cfg = pd.concat(
        [pd.read_parquet(p) for p in config_files], ignore_index=True
    )
    assert (cfg["max_encoder_length"] == 10).any()


def test_walk_forward_folds_emit_per_fold_sweep_id(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    ds = tmp_path / "dataset.parquet"
    _write_dataset(ds, n_days=400)
    pipeline = _make_pipeline(silver)

    config = {
        "output_subdir": "sw_wf",
        "training_config": {"max_encoder_length": 5, "max_prediction_length": 1},
        "walk_forward": {
            "enabled": True,
            "folds": [
                {
                    "name": "wf_1",
                    "train_start": "2024-01-01",
                    "train_end": "2024-04-30",
                    "val_start": "2024-05-01",
                    "val_end": "2024-06-30",
                    "test_start": "2024-07-01",
                    "test_end": "2024-08-31",
                },
                {
                    "name": "wf_2",
                    "train_start": "2024-01-01",
                    "train_end": "2024-06-30",
                    "val_start": "2024-07-01",
                    "val_end": "2024-08-31",
                    "test_start": "2024-09-01",
                    "test_end": "2024-10-31",
                },
            ],
        },
        "baselines": [{"name": "zero_return", "config": {}}],
        "replica_seeds": [42],
    }
    result = pipeline.execute(asset="AAPL", config=config, dataset_path=ds)
    assert result.folds_processed == 2
    dim = _load_dim(silver)
    sweep_ids = set(dim["parent_sweep_id"].astype(str).tolist())
    assert "sw_wf__wf_1" in sweep_ids
    assert "sw_wf__wf_2" in sweep_ids


def test_rejects_unsupported_baseline_name(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    ds = tmp_path / "dataset.parquet"
    _write_dataset(ds)
    pipeline = _make_pipeline(silver)
    config = {
        "output_subdir": "sw_bad",
        "training_config": {"max_encoder_length": 5, "max_prediction_length": 1},
        "split_config": {
            "train_start": "2024-01-01",
            "train_end": "2024-06-30",
            "val_start": "2024-07-01",
            "val_end": "2024-08-31",
            "test_start": "2024-09-01",
            "test_end": "2024-10-26",
        },
        "baselines": [{"name": "unsupported_arch", "config": {}}],
    }
    with pytest.raises(ValueError, match="unsupported baseline"):
        pipeline.execute(asset="AAPL", config=config, dataset_path=ds)


def test_rejects_missing_output_subdir(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    ds = tmp_path / "dataset.parquet"
    _write_dataset(ds)
    pipeline = _make_pipeline(silver)
    config = {
        "training_config": {"max_encoder_length": 5, "max_prediction_length": 1},
        "split_config": {
            "train_start": "2024-01-01",
            "train_end": "2024-06-30",
            "val_start": "2024-07-01",
            "val_end": "2024-08-31",
            "test_start": "2024-09-01",
            "test_end": "2024-10-26",
        },
        "baselines": [{"name": "zero_return", "config": {}}],
    }
    with pytest.raises(ValueError, match="output_subdir"):
        pipeline.execute(asset="AAPL", config=config, dataset_path=ds)
