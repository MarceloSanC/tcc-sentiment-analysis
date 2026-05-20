from __future__ import annotations

import pandas as pd
import pytest

from src.adapters.parquet_analytics_run_repository import ParquetAnalyticsRunRepository
from src.infrastructure.schemas.analytics_store_schema import ANALYTICS_SCHEMA_VERSION
from src.interfaces.analytics_run_repository import DuplicateKeyError


def _row(run_id: str) -> dict:
    return {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "run_id": run_id,
        "execution_id": None,
        "parent_sweep_id": None,
        "trial_number": None,
        "fold": None,
        "seed": 42,
        "asset": "AAPL",
        "feature_set_name": "B",
        "feature_set_hash": "hash",
        "feature_list_ordered_json": '["open","close"]',
        "config_signature": "cfg",
        "split_fingerprint": "spl",
        "model_version": "v1",
        "checkpoint_path_final": "/tmp/final.pt",
        "checkpoint_path_best": "/tmp/best.ckpt",
        "git_commit": "abc",
        "pipeline_version": "0.1",
        "library_versions_json": "{}",
        "hardware_info_json": "{}",
        "status": "ok",
        "duration_total_seconds": 0.0,
        "eta_recorded_seconds": 0.0,
        "retries": 0,
        "created_at_utc": "2026-01-01T00:00:00Z",
    }


def _snapshot_row(run_id: str, *, n_samples_train: int = 3) -> dict:
    return {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "run_id": run_id,
        "asset": "AAPL",
        "parent_sweep_id": None,
        "dataset_start_utc": "2026-01-01T00:00:00Z",
        "dataset_end_utc": "2026-01-05T00:00:00Z",
        "train_start_utc": "2026-01-01T00:00:00Z",
        "train_end_utc": "2026-01-03T00:00:00Z",
        "val_start_utc": "2026-01-04T00:00:00Z",
        "val_end_utc": "2026-01-04T00:00:00Z",
        "test_start_utc": "2026-01-05T00:00:00Z",
        "test_end_utc": "2026-01-05T00:00:00Z",
        "warmup_policy": "strict_fail",
        "required_warmup_count": 0,
        "warmup_applied": "false",
        "effective_train_start_utc": "20260101",
        "n_samples_train": n_samples_train,
        "n_samples_val": 1,
        "n_samples_test": 1,
        "dataset_fingerprint": "dfp",
        "split_fingerprint": "sfp",
    }


def _oos_row(
    run_id: str,
    *,
    timestamp_utc: str = "2026-01-01T00:00:00+00:00",
    target_timestamp_utc: str = "2026-01-01T00:00:00+00:00",
    year: int = 2026,
    y_pred: float = 0.2,
    decision_idx: int = 0,
) -> dict:
    return {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "run_id": run_id,
        "model_version": "v1",
        "asset": "AAPL",
        "feature_set_name": "B",
        "config_signature": "cfg",
        "split": "test",
        "fold": "wf_1",
        "seed": 7,
        "horizon": 1,
        "decision_idx": decision_idx,
        "timestamp_utc": timestamp_utc,
        "target_timestamp_utc": target_timestamp_utc,
        "y_true": 0.1,
        "y_pred": y_pred,
        "error": y_pred - 0.1,
        "abs_error": abs(y_pred - 0.1),
        "sq_error": (y_pred - 0.1) ** 2,
        "quantile_p10": 0.05,
        "quantile_p50": y_pred,
        "quantile_p90": 0.3,
        "year": year,
    }


def test_upsert_dim_run_by_run_id(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)

    repo.upsert_dim_run(_row("r1"))
    repo.upsert_dim_run({**_row("r1"), "model_version": "v2"})

    path = tmp_path / "dim_run" / "asset=AAPL" / "sweep_id=__none__" / "dim_run.parquet"
    df = pd.read_parquet(path)

    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "r1"
    assert df.iloc[0]["model_version"] == "v2"



def test_overwrite_policy_first_write_creates_parquet(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)

    repo.append_fact_run_snapshot(_snapshot_row("r1"))

    path = tmp_path / "fact_run_snapshot" / "asset=AAPL" / "sweep_id=__none__" / "fact_run_snapshot.parquet"
    df = pd.read_parquet(path)

    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "r1"


def test_overwrite_policy_appends_when_pk_is_distinct(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)

    repo.append_fact_run_snapshot(_snapshot_row("r1"))
    repo.append_fact_run_snapshot(_snapshot_row("r2"))

    path = tmp_path / "fact_run_snapshot" / "asset=AAPL" / "sweep_id=__none__" / "fact_run_snapshot.parquet"
    df = pd.read_parquet(path)

    assert len(df) == 2
    assert set(df["run_id"].tolist()) == {"r1", "r2"}


def test_overwrite_policy_raises_on_pk_collision_without_flag(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)
    row = _oos_row("r1")

    repo.append_fact_oos_predictions([row])

    with pytest.raises(DuplicateKeyError, match="fact_oos_predictions"):
        repo.append_fact_oos_predictions([{**row, "y_pred": 0.9}])


def test_overwrite_policy_replaces_colliding_rows_when_enabled(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)
    original = _oos_row("r1", y_pred=0.2)
    preserved = _oos_row(
        "r1",
        timestamp_utc="2026-01-02T00:00:00+00:00",
        target_timestamp_utc="2026-01-02T00:00:00+00:00",
        y_pred=0.3,
    )

    repo.append_fact_oos_predictions([original, preserved])
    repo.append_fact_oos_predictions(
        [{**original, "y_pred": 0.9, "quantile_p50": 0.9}],
        overwrite=True,
    )

    path = tmp_path / "fact_oos_predictions" / "asset=AAPL" / "feature_set_name=B" / "year=2026" / "fact_oos_predictions.parquet"
    df = pd.read_parquet(path).sort_values("target_timestamp_utc").reset_index(drop=True)

    assert len(df) == 2
    assert float(df.iloc[0]["y_pred"]) == 0.9
    assert float(df.iloc[1]["y_pred"]) == 0.3


def test_overwrite_policy_only_checks_collisions_within_partition_path(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)
    base = {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "run_id": "r1",
        "asset": "AAPL",
        "parent_sweep_id": "swp",
        "epoch": 0,
        "train_loss": 0.5,
        "val_loss": 0.6,
        "epoch_time_seconds": 1.2,
        "best_epoch": 0,
        "stopped_epoch": 2,
        "early_stop_reason": "max_epochs",
    }

    repo.append_fact_epoch_metrics([{**base, "fold": "wf_1"}])
    repo.append_fact_epoch_metrics([{**base, "fold": "wf_2"}])

    path_1 = tmp_path / "fact_epoch_metrics" / "asset=AAPL" / "sweep_id=swp" / "fold=wf_1" / "fact_epoch_metrics.parquet"
    path_2 = tmp_path / "fact_epoch_metrics" / "asset=AAPL" / "sweep_id=swp" / "fold=wf_2" / "fact_epoch_metrics.parquet"

    assert len(pd.read_parquet(path_1)) == 1
    assert len(pd.read_parquet(path_2)) == 1


def test_append_fact_run_snapshot_and_split_refs(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)

    snapshot = {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "run_id": "r1",
        "asset": "AAPL",
        "parent_sweep_id": None,
        "dataset_start_utc": "2026-01-01T00:00:00Z",
        "dataset_end_utc": "2026-01-05T00:00:00Z",
        "train_start_utc": "2026-01-01T00:00:00Z",
        "train_end_utc": "2026-01-03T00:00:00Z",
        "val_start_utc": "2026-01-04T00:00:00Z",
        "val_end_utc": "2026-01-04T00:00:00Z",
        "test_start_utc": "2026-01-05T00:00:00Z",
        "test_end_utc": "2026-01-05T00:00:00Z",
        "warmup_policy": "strict_fail",
        "required_warmup_count": 0,
        "warmup_applied": "false",
        "effective_train_start_utc": "20260101",
        "n_samples_train": 3,
        "n_samples_val": 1,
        "n_samples_test": 1,
        "dataset_fingerprint": "dfp",
        "split_fingerprint": "sfp",
    }
    repo.append_fact_run_snapshot(snapshot)

    split_rows = [
        {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": "r1",
            "split": "train",
            "artifact_path": None,
            "artifact_hash": "h1",
            "timestamps_compact_json": "[]",
        },
        {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": "r1",
            "split": "val",
            "artifact_path": None,
            "artifact_hash": "h2",
            "timestamps_compact_json": "[]",
        },
    ]
    repo.append_fact_split_timestamps_ref(split_rows)

    snapshot_path = tmp_path / "fact_run_snapshot" / "asset=AAPL" / "sweep_id=__none__" / "fact_run_snapshot.parquet"
    split_ref_path = tmp_path / "fact_split_timestamps_ref" / "run_id=r1" / "fact_split_timestamps_ref.parquet"

    sdf = pd.read_parquet(snapshot_path)
    rdf = pd.read_parquet(split_ref_path)

    assert len(sdf) == 1
    assert sdf.iloc[0]["run_id"] == "r1"
    assert len(rdf) == 2
    assert set(rdf["split"].tolist()) == {"train", "val"}


def test_append_fact_config(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)

    row = {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "run_id": "r1",
        "training_run_id": "r1",
        "asset": "AAPL",
        "parent_sweep_id": None,
        "prediction_mode": "quantile",
        "loss_name": "QuantileLoss",
        "quantile_levels_json": "[0.1,0.5,0.9]",
        "max_encoder_length": 60,
        "max_prediction_length": 1,
        "batch_size": 64,
        "max_epochs": 20,
        "learning_rate": 0.001,
        "hidden_size": 16,
        "attention_head_size": 2,
        "dropout": 0.1,
        "hidden_continuous_size": 8,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.0,
        "scaler_type": "StandardScaler",
        "training_config_json": "{}",
        "dataset_parameters_json": "{}",
        "search_space_json": "{}",
        "objective_name": "val_loss",
        "objective_direction": "minimize",
    }
    repo.append_fact_config(row)

    path = tmp_path / "fact_config" / "asset=AAPL" / "sweep_id=__none__" / "fact_config.parquet"
    df = pd.read_parquet(path)
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "r1"
    assert df.iloc[0]["prediction_mode"] == "quantile"


def test_append_fact_failures(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)

    row = {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "run_id": "r1",
        "execution_id": None,
        "asset": "AAPL",
        "failed_at_utc": "2026-01-01T00:00:00Z",
        "stage": "train_execute",
        "error_type": "RuntimeError",
        "error_message": "boom",
        "trace_hash": "h",
        "traceback_excerpt": "trace",
        "entrypoint": "src.main_train_tft",
        "cmdline": "python -m src.main_train_tft --asset AAPL",
        "stdout_truncated": "",
        "stderr_truncated": "boom",
    }
    repo.append_fact_failures(row)

    path = tmp_path / "fact_failures" / "asset=AAPL" / "fact_failures.parquet"
    df = pd.read_parquet(path)
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "r1"
    assert df.iloc[0]["stage"] == "train_execute"


def test_append_fact_epoch_metrics(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)

    rows = [
        {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": "r1",
            "asset": "AAPL",
            "parent_sweep_id": "swp",
            "fold": "wf_1",
            "epoch": 0,
            "train_loss": 0.5,
            "val_loss": 0.6,
            "epoch_time_seconds": 1.2,
            "best_epoch": 0,
            "stopped_epoch": 2,
            "early_stop_reason": "max_epochs",
        },
        {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": "r1",
            "asset": "AAPL",
            "parent_sweep_id": "swp",
            "fold": "wf_1",
            "epoch": 1,
            "train_loss": 0.4,
            "val_loss": 0.5,
            "epoch_time_seconds": 1.1,
            "best_epoch": 1,
            "stopped_epoch": 2,
            "early_stop_reason": "max_epochs",
        },
    ]
    repo.append_fact_epoch_metrics(rows)

    path = tmp_path / "fact_epoch_metrics" / "asset=AAPL" / "sweep_id=swp" / "fold=wf_1" / "fact_epoch_metrics.parquet"
    df = pd.read_parquet(path)
    assert len(df) == 2
    assert set(df["epoch"].tolist()) == {0, 1}


def test_append_fact_oos_predictions(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)

    rows = [
        {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": "r1",
            "model_version": "v1",
            "asset": "AAPL",
            "feature_set_name": "B",
            "config_signature": "cfg",
            "split": "test",
            "fold": "wf_1",
            "seed": 7,
            "horizon": 1,
            "decision_idx": 0,
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "target_timestamp_utc": "2026-01-01T00:00:00+00:00",
            "y_true": 0.1,
            "y_pred": 0.2,
            "error": 0.1,
            "abs_error": 0.1,
            "sq_error": 0.01,
            "quantile_p10": 0.05,
            "quantile_p50": 0.2,
            "quantile_p90": 0.3,
            "year": 2026,
        }
    ]
    repo.append_fact_oos_predictions(rows)

    path = tmp_path / "fact_oos_predictions" / "asset=AAPL" / "feature_set_name=B" / "year=2026" / "fact_oos_predictions.parquet"
    df = pd.read_parquet(path)
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "r1"
    assert df.iloc[0]["model_version"] == "v1"
    assert float(df.iloc[0]["quantile_p50"]) == 0.2


def test_append_fact_model_artifacts(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)

    row = {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "run_id": "r1",
        "asset": "AAPL",
        "model_version": "v1",
        "checkpoint_path_final": "/tmp/model_state.pt",
        "checkpoint_path_best": None,
        "config_path": "/tmp/config.json",
        "scaler_path": None,
        "encoder_path": None,
        "feature_importance_json": "[]",
        "attention_summary_json": "{}",
        "logs_ref_json": "{}",
    }
    repo.append_fact_model_artifacts(row)

    path = (
        tmp_path
        / "fact_model_artifacts"
        / "asset=AAPL"
        / "fact_model_artifacts.parquet"
    )
    df = pd.read_parquet(path)
    assert len(df) == 1
    assert df.iloc[0]["run_id"] == "r1"


def test_append_fact_split_metrics(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)

    rows = [
        {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": "r1",
            "asset": "AAPL",
            "parent_sweep_id": "swp",
            "split": "train",
            "rmse": 0.1,
            "mae": 0.1,
            "mape": 0.0,
            "smape": 0.0,
            "directional_accuracy": 0.6,
            "n_samples": 10,
        },
        {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": "r1",
            "asset": "AAPL",
            "parent_sweep_id": "swp",
            "split": "test",
            "rmse": 0.2,
            "mae": 0.2,
            "mape": 0.0,
            "smape": 0.0,
            "directional_accuracy": 0.55,
            "n_samples": 5,
        },
    ]
    repo.append_fact_split_metrics(rows)

    path = tmp_path / "fact_split_metrics" / "asset=AAPL" / "sweep_id=swp" / "fact_split_metrics.parquet"
    df = pd.read_parquet(path)
    assert len(df) == 2
    assert set(df["split"].tolist()) == {"train", "test"}


def test_append_bridge_run_features_and_fact_inference_runs(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)

    bridge_rows = [
        {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": "r1",
            "feature_order": 0,
            "feature_name": "close",
        },
        {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": "r1",
            "feature_order": 1,
            "feature_name": "volume",
        },
    ]
    repo.append_bridge_run_features(bridge_rows)

    inf_row = {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "run_id": "r1",
        "training_run_id": "r1",
        "inference_run_id": "inf_1",
        "model_version": "20260308_010101_B",
        "asset": "AAPL",
        "inference_start_utc": "2026-03-01T00:00:00+00:00",
        "inference_end_utc": "2026-03-05T00:00:00+00:00",
        "overwrite": "false",
        "batch_size": 64,
        "status": "ok",
        "inferred_count": 5,
        "skipped_count": 1,
        "upserts_count": 4,
        "duration_seconds": 1.5,
    }
    repo.append_fact_inference_runs(inf_row)

    bridge_path = tmp_path / "bridge_run_features" / "run_id=r1" / "bridge_run_features.parquet"
    inf_path = tmp_path / "fact_inference_runs" / "asset=AAPL" / "fact_inference_runs.parquet"

    bdf = pd.read_parquet(bridge_path)
    idf = pd.read_parquet(inf_path)
    assert len(bdf) == 2
    assert list(bdf.sort_values("feature_order")["feature_name"]) == ["close", "volume"]
    assert len(idf) == 1
    assert idf.iloc[0]["inference_run_id"] == "inf_1"


def test_append_fact_inference_predictions(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)

    rows = [
        {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "inference_run_id": "inf_1",
            "run_id": "r1",
            "training_run_id": "r1",
            "model_version": "20260308_010101_B",
            "asset": "AAPL",
            "feature_set_name": "BASELINE_FEATURES",
            "features_used_csv": "open,high,low,close,volume",
            "model_path": "data/models/AAPL/runs/20260308_010101_B",
            "split": "inference",
            "horizon": 1,
            "timestamp_utc": "2026-03-05T00:00:00+00:00",
            "target_timestamp_utc": "2026-03-05T00:00:00+00:00",
            "y_true": None,
            "y_pred": 0.0123,
            "error": None,
            "abs_error": None,
            "sq_error": None,
            "quantile_p10": -0.01,
            "quantile_p50": 0.0123,
            "quantile_p90": 0.025,
            "year": 2026,
            "created_at_utc": "2026-03-05T01:00:00+00:00",
        }
    ]
    repo.append_fact_inference_predictions(rows)

    path = (
        tmp_path
        / "fact_inference_predictions"
        / "asset=AAPL"
        / "model_version=20260308_010101_B"
        / "year=2026"
        / "fact_inference_predictions.parquet"
    )
    df = pd.read_parquet(path)
    assert len(df) == 1
    assert df.iloc[0]["inference_run_id"] == "inf_1"
    assert float(df.iloc[0]["y_pred"]) == 0.0123


def test_append_fact_feature_contrib_local(tmp_path) -> None:
    repo = ParquetAnalyticsRunRepository(output_dir=tmp_path)

    rows = [
        {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "inference_run_id": "inf_1",
            "run_id": "r1",
            "training_run_id": "r1",
            "model_version": "20260308_010101_B",
            "asset": "AAPL",
            "feature_set_name": "BASELINE_FEATURES",
            "split": "inference",
            "horizon": 1,
            "timestamp_utc": "2026-03-05T00:00:00+00:00",
            "target_timestamp_utc": "2026-03-05T00:00:00+00:00",
            "feature_name": "close",
            "feature_rank": 1,
            "contribution": 0.01,
            "abs_contribution": 0.01,
            "contribution_sign": "positive",
            "method": "local_magnitude_signed_v1",
            "year": 2026,
            "created_at_utc": "2026-03-05T01:00:00+00:00",
        }
    ]
    repo.append_fact_feature_contrib_local(rows)

    path = (
        tmp_path
        / "fact_feature_contrib_local"
        / "asset=AAPL"
        / "model_version=20260308_010101_B"
        / "year=2026"
        / "fact_feature_contrib_local.parquet"
    )
    df = pd.read_parquet(path)
    assert len(df) == 1
    assert df.iloc[0]["feature_name"] == "close"
