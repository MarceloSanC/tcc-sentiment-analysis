from __future__ import annotations

import pandas as pd
import pytest

from src.domain.services.scope_spec import ScopeSpec
from src.use_cases.validate_analytics_quality_use_case import (
    ValidateAnalyticsQualityUseCase,
)


def _write_table(base, table_name: str, rows: list[dict], parts: dict[str, str] | None = None) -> None:
    table_dir = base / table_name
    if parts:
        for k, v in parts.items():
            table_dir = table_dir / f"{k}={v}"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(table_dir / f"{table_name}.parquet", index=False)


def _seed_minimal_valid_silver(silver) -> None:
    _write_table(
        silver,
        "dim_run",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "execution_id": None,
                "asset": "AAPL",
                "feature_set_name": "B",
                "feature_set_hash": "fh",
                "feature_list_ordered_json": "[]",
                "config_signature": "cfg1",
                "split_fingerprint": "sp1",
                "model_version": "v1",
                "parent_sweep_id": "sw1",
                "checkpoint_path_final": "/tmp/final.pt",
                "checkpoint_path_best": "/tmp/best.ckpt",
                "git_commit": "abc",
                "pipeline_version": "0.1",
                "library_versions_json": "{}",
                "hardware_info_json": "{}",
                "status": "ok",
                "duration_total_seconds": 1.0,
                "eta_recorded_seconds": 0.0,
                "retries": 0,
                "created_at_utc": "2026-01-01T00:00:00+00:00",
            }
        ],
        {"asset": "AAPL", "sweep_id": "sw1"},
    )
    _write_table(
        silver,
        "fact_run_snapshot",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "asset": "AAPL",
                "parent_sweep_id": "sw1",
                "dataset_start_utc": "2026-01-01T00:00:00+00:00",
                "dataset_end_utc": "2026-01-10T00:00:00+00:00",
                "train_start_utc": "2026-01-01T00:00:00+00:00",
                "train_end_utc": "2026-01-06T00:00:00+00:00",
                "val_start_utc": "2026-01-07T00:00:00+00:00",
                "val_end_utc": "2026-01-08T00:00:00+00:00",
                "test_start_utc": "2026-01-09T00:00:00+00:00",
                "test_end_utc": "2026-01-10T00:00:00+00:00",
                "warmup_policy": "strict_fail",
                "required_warmup_count": 0,
                "warmup_applied": "false",
                "effective_train_start_utc": "2026-01-01T00:00:00+00:00",
                "n_samples_train": 6,
                "n_samples_val": 2,
                "n_samples_test": 2,
                "dataset_fingerprint": "dfp",
                "split_fingerprint": "sfp",
            }
        ],
        {"asset": "AAPL", "sweep_id": "sw1"},
    )
    _write_table(
        silver,
        "fact_split_metrics",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "asset": "AAPL",
                "parent_sweep_id": "sw1",
                "split": "test",
                "rmse": 0.1,
                "mae": 0.1,
                "mape": 0.0,
                "smape": 0.0,
                "directional_accuracy": 0.5,
                "n_samples": 2,
            }
        ],
        {"asset": "AAPL", "sweep_id": "sw1"},
    )
    _write_table(
        silver,
        "fact_config",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "asset": "AAPL",
                "parent_sweep_id": "sw1",
                "prediction_mode": "quantile",
                "loss_name": "QuantileLoss",
                "quantile_levels_json": "[0.1,0.5,0.9]",
                "max_encoder_length": 3,
                "max_prediction_length": 1,
                "batch_size": 8,
                "max_epochs": 2,
                "learning_rate": 0.001,
                "hidden_size": 8,
                "attention_head_size": 2,
                "dropout": 0.1,
                "hidden_continuous_size": 4,
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 0.0,
                "scaler_type": "none",
                "training_config_json": "{}",
                "dataset_parameters_json": "{}",
                "search_space_json": "{}",
                "objective_name": "val_loss",
                "objective_direction": "minimize",
            }
        ],
        {"asset": "AAPL", "sweep_id": "sw1"},
    )
    _write_table(
        silver,
        "fact_epoch_metrics",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "asset": "AAPL",
                "parent_sweep_id": "sw1",
                "fold": "none",
                "epoch": 0,
                "train_loss": 0.8,
                "val_loss": 0.9,
            }
        ],
        {"asset": "AAPL", "sweep_id": "__none__", "fold": "none"},
    )
    _write_table(
        silver,
        "fact_oos_predictions",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "model_version": "v1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg1",
                "split": "test",
                "fold": "none",
                "seed": 0,
                "horizon": 1,
                "timestamp_utc": "2026-01-09T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-09T00:00:00+00:00",
                "y_true": 0.1,
                "y_pred": 0.2,
                "error": 0.1,
                "abs_error": 0.1,
                "sq_error": 0.01,
                "quantile_p10": 0.1,
                "quantile_p50": 0.2,
                "quantile_p90": 0.3,
                "year": 2026,
            }
        ],
        {"asset": "AAPL", "feature_set_name": "B", "year": "2026"},
    )
    _write_table(
        silver,
        "fact_model_artifacts",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "asset": "AAPL",
                "model_version": "v1",
                "checkpoint_path_final": "/tmp/final.pt",
                "checkpoint_path_best": "/tmp/best.ckpt",
                "config_path": "/tmp/config.json",
                "scaler_path": None,
                "encoder_path": None,
                "feature_importance_json": "[]",
                "attention_summary_json": "{\"available\": false}",
                "logs_ref_json": "{}",
            }
        ],
        {"asset": "AAPL"},
    )
    _write_table(
        silver,
        "bridge_run_features",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "feature_order": 0,
                "feature_name": "close",
            }
        ],
        {"run_id": "r1"},
    )


def test_validate_analytics_quality_passes_on_valid_dataset(tmp_path) -> None:
    silver = tmp_path / "silver"
    _seed_minimal_valid_silver(silver)

    result = ValidateAnalyticsQualityUseCase(analytics_silver_dir=silver).execute()
    assert result.passed is True
    assert len(result.checks) > 0
    assert all(bool(item["passed"]) for item in result.checks)


def test_validate_analytics_quality_fails_when_quantile_contract_is_broken(tmp_path) -> None:
    silver = tmp_path / "silver"
    _seed_minimal_valid_silver(silver)

    # Break quantile contract for official run
    _write_table(
        silver,
        "fact_oos_predictions",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "model_version": "v1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg1",
                "split": "test",
                "fold": "none",
                "seed": 0,
                "horizon": 1,
                "timestamp_utc": "2026-01-10T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-10T00:00:00+00:00",
                "y_true": 0.1,
                "y_pred": 0.2,
                "error": 0.1,
                "abs_error": 0.1,
                "sq_error": 0.01,
                "quantile_p10": None,
                "quantile_p50": 0.2,
                "quantile_p90": 0.3,
                "year": 2026,
            }
        ],
        {"asset": "AAPL", "feature_set_name": "B", "year": "2026"},
    )

    result = ValidateAnalyticsQualityUseCase(analytics_silver_dir=silver).execute()
    assert result.passed is False
    assert any(
        (item["check"] == "official_contract_quantile_attention" and not bool(item["passed"]))
        for item in result.checks
    )


def test_validate_analytics_quality_fails_on_oos_duplicate_key(tmp_path) -> None:
    silver = tmp_path / "silver"
    _seed_minimal_valid_silver(silver)

    _write_table(
        silver,
        "fact_oos_predictions",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "model_version": "v1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg1",
                "split": "test",
                "fold": "none",
                "seed": 0,
                "horizon": 1,
                "timestamp_utc": "2026-01-09T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-09T00:00:00+00:00",
                "y_true": 0.1,
                "y_pred": 0.2,
                "error": 0.1,
                "abs_error": 0.1,
                "sq_error": 0.01,
                "quantile_p10": 0.1,
                "quantile_p50": 0.2,
                "quantile_p90": 0.3,
                "year": 2026,
            },
            {
                "schema_version": 1,
                "run_id": "r1",
                "model_version": "v1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg1",
                "split": "test",
                "fold": "none",
                "seed": 0,
                "horizon": 1,
                "timestamp_utc": "2026-01-09T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-09T00:00:00+00:00",
                "y_true": 0.1,
                "y_pred": 0.2,
                "error": 0.1,
                "abs_error": 0.1,
                "sq_error": 0.01,
                "quantile_p10": 0.1,
                "quantile_p50": 0.2,
                "quantile_p90": 0.3,
                "year": 2026,
            },
        ],
        {"asset": "AAPL", "feature_set_name": "B", "year": "2026"},
    )

    result = ValidateAnalyticsQualityUseCase(analytics_silver_dir=silver).execute()
    assert result.passed is False
    assert any(
        (item["check"] == "oos_unique_key" and not bool(item["passed"]))
        for item in result.checks
    )


def test_validate_analytics_quality_fails_on_oos_non_numeric_values(tmp_path) -> None:
    silver = tmp_path / "silver"
    _seed_minimal_valid_silver(silver)

    _write_table(
        silver,
        "fact_oos_predictions",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "model_version": "v1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "config_signature": "cfg1",
                "split": "test",
                "fold": "none",
                "seed": 0,
                "horizon": 1,
                "timestamp_utc": "2026-01-10T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-10T00:00:00+00:00",
                "y_true": "abc",
                "y_pred": 0.2,
                "error": 0.1,
                "abs_error": 0.1,
                "sq_error": 0.01,
                "quantile_p10": 0.1,
                "quantile_p50": 0.2,
                "quantile_p90": 0.3,
                "year": 2026,
            }
        ],
        {"asset": "AAPL", "feature_set_name": "B", "year": "2026"},
    )

    result = ValidateAnalyticsQualityUseCase(analytics_silver_dir=silver).execute()
    assert result.passed is False
    assert any(
        (item["check"] == "oos_numeric_types" and not bool(item["passed"]))
        for item in result.checks
    )


def test_validate_analytics_quality_fails_on_oos_missing_expected_horizon(tmp_path) -> None:
    silver = tmp_path / "silver"
    _seed_minimal_valid_silver(silver)

    _write_table(
        silver,
        "fact_config",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "asset": "AAPL",
                "parent_sweep_id": "sw1",
                "prediction_mode": "quantile",
                "loss_name": "QuantileLoss",
                "quantile_levels_json": "[0.1,0.5,0.9]",
                "evaluation_horizons_json": "[1,7]",
                "max_encoder_length": 3,
                "max_prediction_length": 7,
                "batch_size": 8,
                "max_epochs": 2,
                "learning_rate": 0.001,
                "hidden_size": 8,
                "attention_head_size": 2,
                "dropout": 0.1,
                "hidden_continuous_size": 4,
                "early_stopping_patience": 2,
                "early_stopping_min_delta": 0.0,
                "scaler_type": "none",
                "training_config_json": "{}",
                "dataset_parameters_json": "{}",
                "search_space_json": "{}",
                "objective_name": "val_loss",
                "objective_direction": "minimize",
            }
        ],
        {"asset": "AAPL", "sweep_id": "sw1"},
    )

    result = ValidateAnalyticsQualityUseCase(analytics_silver_dir=silver).execute()
    assert result.passed is False
    assert any(
        (item["check"] == "oos_horizon_coverage" and not bool(item["passed"]))
        for item in result.checks
    )


def test_validate_analytics_quality_fails_on_oos_supervised_nulls(tmp_path) -> None:
    silver = tmp_path / "silver"
    _seed_minimal_valid_silver(silver)

    _write_table(
        silver,
        "fact_oos_predictions",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "model_version": "v1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "config_signature": "cfg1",
                "split": "test",
                "fold": "none",
                "seed": 0,
                "horizon": 1,
                "timestamp_utc": "2026-01-10T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-10T00:00:00+00:00",
                "y_true": None,
                "y_pred": 0.2,
                "error": 0.1,
                "abs_error": 0.1,
                "sq_error": 0.01,
                "quantile_p10": 0.1,
                "quantile_p50": 0.2,
                "quantile_p90": 0.3,
                "year": 2026,
            }
        ],
        {"asset": "AAPL", "feature_set_name": "B", "year": "2026"},
    )

    result = ValidateAnalyticsQualityUseCase(analytics_silver_dir=silver).execute()
    assert result.passed is False
    assert any(
        (item["check"] == "oos_supervised_nulls" and not bool(item["passed"]))
        for item in result.checks
    )


def test_validate_analytics_quality_fails_on_negative_interval_width(tmp_path) -> None:
    silver = tmp_path / "silver"
    _seed_minimal_valid_silver(silver)

    _write_table(
        silver,
        "fact_oos_predictions",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "model_version": "v1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "config_signature": "cfg1",
                "split": "test",
                "fold": "none",
                "seed": 0,
                "horizon": 1,
                "timestamp_utc": "2026-01-10T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-10T00:00:00+00:00",
                "y_true": 0.1,
                "y_pred": 0.2,
                "error": 0.1,
                "abs_error": 0.1,
                "sq_error": 0.01,
                "quantile_p10": 0.4,
                "quantile_p50": 0.2,
                "quantile_p90": 0.3,
                "year": 2026,
            }
        ],
        {"asset": "AAPL", "feature_set_name": "B", "year": "2026"},
    )

    result = ValidateAnalyticsQualityUseCase(analytics_silver_dir=silver).execute()
    assert result.passed is False
    assert any(
        (item["check"] == "oos_interval_width_non_negative" and not bool(item["passed"]))
        for item in result.checks
    )


def test_validate_analytics_quality_fails_on_oos_pairwise_target_alignment(tmp_path) -> None:
    silver = tmp_path / "silver"
    _seed_minimal_valid_silver(silver)

    _write_table(
        silver,
        "dim_run",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "execution_id": None,
                "asset": "AAPL",
                "feature_set_name": "B",
                "feature_set_hash": "fh",
                "feature_list_ordered_json": "[]",
                "config_signature": "cfg1",
                "split_fingerprint": "sp1",
                "model_version": "v1",
                "checkpoint_path_final": "/tmp/final.pt",
                "checkpoint_path_best": "/tmp/best.ckpt",
                "git_commit": "abc",
                "pipeline_version": "0.1",
                "library_versions_json": "{}",
                "hardware_info_json": "{}",
                "status": "ok",
                "duration_total_seconds": 1.0,
                "eta_recorded_seconds": 0.0,
                "retries": 0,
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "parent_sweep_id": "sw1",
            },
            {
                "schema_version": 1,
                "run_id": "r2",
                "execution_id": None,
                "asset": "AAPL",
                "feature_set_name": "B",
                "feature_set_hash": "fh",
                "feature_list_ordered_json": "[]",
                "config_signature": "cfg2",
                "split_fingerprint": "sp2",
                "model_version": "v2",
                "checkpoint_path_final": "/tmp/final.pt",
                "checkpoint_path_best": "/tmp/best.ckpt",
                "git_commit": "abc",
                "pipeline_version": "0.1",
                "library_versions_json": "{}",
                "hardware_info_json": "{}",
                "status": "ok",
                "duration_total_seconds": 1.0,
                "eta_recorded_seconds": 0.0,
                "retries": 0,
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "parent_sweep_id": "sw1",
            },
        ],
        {"asset": "AAPL", "sweep_id": "sw1"},
    )

    _write_table(
        silver,
        "fact_oos_predictions",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "model_version": "v1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "config_signature": "cfg1",
                "split": "test",
                "fold": "none",
                "seed": 0,
                "horizon": 1,
                "timestamp_utc": "2026-01-09T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-09T00:00:00+00:00",
                "y_true": 0.1,
                "y_pred": 0.2,
                "error": 0.1,
                "abs_error": 0.1,
                "sq_error": 0.01,
                "quantile_p10": 0.1,
                "quantile_p50": 0.2,
                "quantile_p90": 0.3,
                "year": 2026,
            },
            {
                "schema_version": 1,
                "run_id": "r2",
                "model_version": "v2",
                "asset": "AAPL",
                "feature_set_name": "B",
                "config_signature": "cfg2",
                "split": "test",
                "fold": "none",
                "seed": 0,
                "horizon": 1,
                "timestamp_utc": "2026-01-10T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-10T00:00:00+00:00",
                "y_true": 0.1,
                "y_pred": 0.2,
                "error": 0.1,
                "abs_error": 0.1,
                "sq_error": 0.01,
                "quantile_p10": 0.1,
                "quantile_p50": 0.2,
                "quantile_p90": 0.3,
                "year": 2026,
            },
        ],
        {"asset": "AAPL", "feature_set_name": "B", "year": "2026"},
    )

    result = ValidateAnalyticsQualityUseCase(analytics_silver_dir=silver).execute()
    assert result.passed is False
    assert any(
        (item["check"] == "oos_pairwise_target_alignment" and not bool(item["passed"]))
        for item in result.checks
    )


def test_validate_analytics_quality_fails_when_dm_mcs_not_materialized_in_gold(tmp_path) -> None:
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _seed_minimal_valid_silver(silver)

    _write_table(
        silver,
        "dim_run",
        [
            {
                "schema_version": 1,
                "run_id": "ra",
                "execution_id": None,
                "asset": "AAPL",
                "feature_set_name": "B",
                "feature_set_hash": "fh",
                "feature_list_ordered_json": "[]",
                "config_signature": "cfg1",
                "split_fingerprint": "sp1",
                "model_version": "v1",
                "checkpoint_path_final": "/tmp/final.pt",
                "checkpoint_path_best": "/tmp/best.ckpt",
                "git_commit": "abc",
                "pipeline_version": "0.1",
                "library_versions_json": "{}",
                "hardware_info_json": "{}",
                "status": "ok",
                "duration_total_seconds": 1.0,
                "eta_recorded_seconds": 0.0,
                "retries": 0,
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "parent_sweep_id": "sw1",
            },
            {
                "schema_version": 1,
                "run_id": "rb",
                "execution_id": None,
                "asset": "AAPL",
                "feature_set_name": "B",
                "feature_set_hash": "fh",
                "feature_list_ordered_json": "[]",
                "config_signature": "cfg2",
                "split_fingerprint": "sp2",
                "model_version": "v2",
                "checkpoint_path_final": "/tmp/final.pt",
                "checkpoint_path_best": "/tmp/best.ckpt",
                "git_commit": "abc",
                "pipeline_version": "0.1",
                "library_versions_json": "{}",
                "hardware_info_json": "{}",
                "status": "ok",
                "duration_total_seconds": 1.0,
                "eta_recorded_seconds": 0.0,
                "retries": 0,
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "parent_sweep_id": "sw1",
            },
        ],
        {"asset": "AAPL", "sweep_id": "sw1"},
    )

    rows: list[dict] = []
    for i in range(1, 6):
        day = f"2026-01-0{i}T00:00:00+00:00"
        rows.append(
            {
                "schema_version": 1,
                "run_id": "ra",
                "model_version": "v1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "config_signature": "cfg1",
                "split": "test",
                "fold": "none",
                "seed": 0,
                "horizon": 1,
                "timestamp_utc": day,
                "target_timestamp_utc": day,
                "y_true": 0.01 * i,
                "y_pred": 0.012 * i,
                "error": 0.002 * i,
                "abs_error": abs(0.002 * i),
                "sq_error": (0.002 * i) ** 2,
                "quantile_p10": 0.0,
                "quantile_p50": 0.012 * i,
                "quantile_p90": 0.02 * i,
                "year": 2026,
            }
        )
        rows.append(
            {
                "schema_version": 1,
                "run_id": "rb",
                "model_version": "v2",
                "asset": "AAPL",
                "feature_set_name": "B",
                "config_signature": "cfg2",
                "split": "test",
                "fold": "none",
                "seed": 0,
                "horizon": 1,
                "timestamp_utc": day,
                "target_timestamp_utc": day,
                "y_true": 0.01 * i,
                "y_pred": 0.008 * i,
                "error": -0.002 * i,
                "abs_error": abs(-0.002 * i),
                "sq_error": (-0.002 * i) ** 2,
                "quantile_p10": -0.01,
                "quantile_p50": 0.008 * i,
                "quantile_p90": 0.018 * i,
                "year": 2026,
            }
        )

    _write_table(
        silver,
        "fact_oos_predictions",
        rows,
        {"asset": "AAPL", "feature_set_name": "B", "year": "2026"},
    )

    result = ValidateAnalyticsQualityUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold,
    ).execute()
    assert result.passed is False
    assert any(
        (item["check"] == "dm_mcs_persisted_executable" and not bool(item["passed"]))
        for item in result.checks
    )


def test_validate_analytics_quality_fails_when_inference_runs_have_no_predictions(tmp_path) -> None:
    silver = tmp_path / "silver"
    _seed_minimal_valid_silver(silver)

    _write_table(
        silver,
        "fact_inference_runs",
        [
            {
                "schema_version": 1,
                "run_id": None,
                "inference_run_id": "inf_1",
                "model_version": "v1",
                "asset": "AAPL",
                "inference_start_utc": "2026-01-09T00:00:00+00:00",
                "inference_end_utc": "2026-01-09T00:00:00+00:00",
                "overwrite": "false",
                "batch_size": 64,
                "status": "ok",
                "inferred_count": 1,
                "skipped_count": 0,
                "upserts_count": 1,
                "duration_seconds": 0.1,
            }
        ],
        {"asset": "AAPL"},
    )

    result = ValidateAnalyticsQualityUseCase(analytics_silver_dir=silver).execute()
    assert result.passed is False
    assert any(
        (item["check"] == "inference_predictions_continuity" and not bool(item["passed"]))
        for item in result.checks
    )


def test_validate_analytics_quality_fails_on_missing_calibrated_confidence_by_horizon(tmp_path) -> None:
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _seed_minimal_valid_silver(silver)

    _write_table(
        gold,
        "gold_prediction_metrics_by_run_split_horizon",
        [
            {
                "run_id": "r1",
                "split": "test",
                "horizon": 1,
                "confidence_calibrated": None,
            }
        ],
    )

    result = ValidateAnalyticsQualityUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold,
    ).execute()
    assert result.passed is False
    assert any(
        (item["check"] == "gold_confidence_calibrated_by_horizon" and not bool(item["passed"]))
        for item in result.checks
    )


def test_validate_analytics_quality_fails_on_missing_n_oos_in_gold_metrics_by_config(tmp_path) -> None:
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _seed_minimal_valid_silver(silver)

    _write_table(
        gold,
        "gold_prediction_metrics_by_run_split_horizon",
        [
            {
                "run_id": "r1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg1",
                "split": "test",
                "horizon": 1,
                "n_samples": 2,
                "confidence_calibrated": 0.7,
            }
        ],
    )
    _write_table(
        gold,
        "gold_prediction_metrics_by_config",
        [
            {
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg1",
                "split": "test",
                "horizon": 1,
                "n_runs": 1,
            }
        ],
    )

    result = ValidateAnalyticsQualityUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold,
    ).execute()
    assert result.passed is False
    assert any(
        (item["check"] == "gold_metrics_by_config_n_oos_contract" and not bool(item["passed"]))
        for item in result.checks
    )


def test_validate_analytics_quality_passes_n_oos_contract_when_consistent(tmp_path) -> None:
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _seed_minimal_valid_silver(silver)

    _write_table(
        gold,
        "gold_prediction_metrics_by_run_split_horizon",
        [
            {
                "run_id": "r1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg1",
                "split": "test",
                "horizon": 1,
                "n_samples": 2,
                "confidence_calibrated": 0.7,
            },
            {
                "run_id": "r2",
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg1",
                "split": "test",
                "horizon": 1,
                "n_samples": 3,
                "confidence_calibrated": 0.8,
            },
        ],
    )
    _write_table(
        gold,
        "gold_prediction_metrics_by_config",
        [
            {
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg1",
                "split": "test",
                "horizon": 1,
                "n_runs": 2,
                "n_oos": 5,
            }
        ],
    )

    result = ValidateAnalyticsQualityUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold,
    ).execute()
    check = next(item for item in result.checks if item["check"] == "gold_metrics_by_config_n_oos_contract")
    assert check["passed"] is True


def test_n_oos_contract_passes_per_sweep(tmp_path) -> None:
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _seed_minimal_valid_silver(silver)

    _write_table(
        gold,
        "gold_prediction_metrics_by_run_split_horizon",
        [
            {
                "run_id": "r1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg_sw1",
                "split": "test",
                "horizon": 1,
                "n_samples": 2,
                "confidence_calibrated": 0.7,
            },
            {
                "run_id": "r2",
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw2",
                "config_signature": "cfg_sw2",
                "split": "test",
                "horizon": 1,
                "n_samples": 3,
                "confidence_calibrated": 0.8,
            },
        ],
    )
    _write_table(
        gold,
        "gold_prediction_metrics_by_config",
        [
            {
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg_sw1",
                "split": "test",
                "horizon": 1,
                "n_runs": 1,
                "n_oos": 2,
            },
            {
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw2",
                "config_signature": "cfg_sw2",
                "split": "test",
                "horizon": 1,
                "n_runs": 1,
                "n_oos": 3,
            },
        ],
    )

    result = ValidateAnalyticsQualityUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold,
        scope_spec=ScopeSpec.create(
            scope_mode="cohort_decision",
            parent_sweep_prefixes=["sw1"],
        ),
    ).execute()

    check = next(item for item in result.checks if item["check"] == "gold_metrics_by_config_n_oos_contract")
    assert check["passed"] is True
    assert "mismatch_with_run_level=0" in str(check["detail"])


def test_n_oos_contract_passes_with_legacy_shared_config_signature(tmp_path) -> None:
    """Regression test for M5-Q4: same config_signature across sweeps must not collapse."""
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _seed_minimal_valid_silver(silver)

    _write_table(
        gold,
        "gold_prediction_metrics_by_run_split_horizon",
        [
            {
                "run_id": "r1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg_shared",
                "split": "test",
                "horizon": 1,
                "n_samples": 2,
                "confidence_calibrated": 0.7,
            },
            {
                "run_id": "r2",
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw2",
                "config_signature": "cfg_shared",
                "split": "test",
                "horizon": 1,
                "n_samples": 3,
                "confidence_calibrated": 0.8,
            },
        ],
    )
    _write_table(
        gold,
        "gold_prediction_metrics_by_config",
        [
            {
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg_shared",
                "split": "test",
                "horizon": 1,
                "n_runs": 1,
                "n_oos": 2,
            },
            {
                "asset": "AAPL",
                "feature_set_name": "B",
                "parent_sweep_id": "sw2",
                "config_signature": "cfg_shared",
                "split": "test",
                "horizon": 1,
                "n_runs": 1,
                "n_oos": 3,
            },
        ],
    )

    result = ValidateAnalyticsQualityUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold,
    ).execute()

    check = next(item for item in result.checks if item["check"] == "gold_metrics_by_config_n_oos_contract")
    assert check["passed"] is True
    assert "mismatch_with_run_level=0" in str(check["detail"])


def test_validate_analytics_quality_block_a_check_passes_on_minimal_valid_dataset(tmp_path) -> None:
    silver = tmp_path / "silver"
    _seed_minimal_valid_silver(silver)

    result = ValidateAnalyticsQualityUseCase(analytics_silver_dir=silver).execute()
    block_a = next(item for item in result.checks if item["check"] == "oos_quantile_block_a_acceptance")
    assert block_a["passed"] is True
    assert "total_rows=1" in str(block_a["detail"])
    assert "crossing_bruto_rate=" in str(block_a["detail"])


def test_validate_analytics_quality_block_a_scope_filters_parent_sweep_prefix(tmp_path) -> None:
    silver = tmp_path / "silver"
    _seed_minimal_valid_silver(silver)

    # Ensure in-scope run has parent_sweep_id and add out-of-scope run with broken quantiles.
    _write_table(
        silver,
        "dim_run",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "execution_id": None,
                "asset": "AAPL",
                "feature_set_name": "B",
                "feature_set_hash": "fh",
                "feature_list_ordered_json": "[]",
                "config_signature": "cfg1",
                "split_fingerprint": "sp1",
                "model_version": "v1",
                "checkpoint_path_final": "/tmp/final.pt",
                "checkpoint_path_best": "/tmp/best.ckpt",
                "git_commit": "abc",
                "pipeline_version": "0.1",
                "library_versions_json": "{}",
                "hardware_info_json": "{}",
                "status": "ok",
                "duration_total_seconds": 1.0,
                "eta_recorded_seconds": 0.0,
                "retries": 0,
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "parent_sweep_id": "0_2_3_ok",
            }
        ],
        {"asset": "AAPL", "sweep_id": "sw1"},
    )

    _write_table(
        silver,
        "dim_run",
        [
            {
                "schema_version": 1,
                "run_id": "r2",
                "execution_id": None,
                "asset": "AAPL",
                "feature_set_name": "B",
                "feature_set_hash": "fh",
                "feature_list_ordered_json": "[]",
                "config_signature": "cfg2",
                "split_fingerprint": "sp2",
                "model_version": "v2",
                "checkpoint_path_final": "/tmp/final.pt",
                "checkpoint_path_best": "/tmp/best.ckpt",
                "git_commit": "abc",
                "pipeline_version": "0.1",
                "library_versions_json": "{}",
                "hardware_info_json": "{}",
                "status": "ok",
                "duration_total_seconds": 1.0,
                "eta_recorded_seconds": 0.0,
                "retries": 0,
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "parent_sweep_id": "0_2_1_bad",
            }
        ],
        {"asset": "AAPL", "sweep_id": "sw2"},
    )

    _write_table(
        silver,
        "fact_oos_predictions",
        [
            {
                "schema_version": 1,
                "run_id": "r2",
                "model_version": "v2",
                "asset": "AAPL",
                "feature_set_name": "B",
                "config_signature": "cfg2",
                "split": "test",
                "fold": "none",
                "seed": 0,
                "horizon": 1,
                "timestamp_utc": "2026-01-10T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-10T00:00:00+00:00",
                "y_true": 0.1,
                "y_pred": 0.2,
                "error": 0.1,
                "abs_error": 0.1,
                "sq_error": 0.01,
                "quantile_p10": 0.4,
                "quantile_p50": 0.2,
                "quantile_p90": 0.3,
                "year": 2026,
            }
        ],
        {"asset": "AAPL", "feature_set_name": "B", "year": "2026"},
    )

    # Scope should only include r1 from parent_sweep_id 0_2_3_.
    result = ValidateAnalyticsQualityUseCase(
        analytics_silver_dir=silver,
        block_a_parent_sweep_prefixes=["0_2_3_"],
        block_a_splits=["test"],
        block_a_horizons=[1],
    ).execute()

    block_a = next(item for item in result.checks if item["check"] == "oos_quantile_block_a_acceptance")
    assert block_a["passed"] is True


def test_validate_analytics_quality_scope_metadata_is_appended_to_checks(tmp_path) -> None:
    silver = tmp_path / "silver"
    _seed_minimal_valid_silver(silver)

    result = ValidateAnalyticsQualityUseCase(
        analytics_silver_dir=silver,
        scope_spec=ScopeSpec.create(scope_mode="global_health"),
    ).execute()

    assert len(result.checks) > 0
    assert all("scope_mode=global_health" in str(item["detail"]) for item in result.checks)


def test_validate_analytics_quality_legacy_block_a_scope_warns_deprecation(tmp_path) -> None:
    silver = tmp_path / "silver"
    _seed_minimal_valid_silver(silver)

    with pytest.warns(DeprecationWarning, match="deprecated"):
        ValidateAnalyticsQualityUseCase(
            analytics_silver_dir=silver,
            block_a_parent_sweep_prefixes=["0_2_3_"],
        ).execute()


def test_validate_analytics_quality_scope_spec_overrides_legacy_scope_filters(tmp_path) -> None:
    silver = tmp_path / "silver"
    _seed_minimal_valid_silver(silver)

    result = ValidateAnalyticsQualityUseCase(
        analytics_silver_dir=silver,
        scope_spec=ScopeSpec.create(scope_mode="global_health"),
        block_a_parent_sweep_prefixes=["0_2_3_"],
        block_a_splits=["test"],
        block_a_horizons=[1],
    ).execute()
    block_a = next(item for item in result.checks if item["check"] == "oos_quantile_block_a_acceptance")
    assert block_a["passed"] is True
    assert "scope_mode=global_health" in str(block_a["detail"])
