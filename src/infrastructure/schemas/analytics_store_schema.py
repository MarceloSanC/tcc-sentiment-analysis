from __future__ import annotations

import json

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral, Real
from pathlib import Path
from typing import Any

ANALYTICS_SCHEMA_VERSION = 2

ANALYTICS_RUN_STATUSES = {"ok", "failed", "partial_failed"}
ANALYTICS_PREDICTION_MODES = {"point", "quantile"}


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_feature_set_hash(features_ordered: list[str]) -> str:
    return _sha256_text("|".join(features_ordered))


def compute_config_signature(training_config: dict[str, Any]) -> str:
    cleaned = dict(training_config)
    for volatile_key in ("created_at", "started_at", "ended_at", "timestamp"):
        cleaned.pop(volatile_key, None)
    return _sha256_text(_canonical_json(cleaned))


def compute_split_fingerprint(
    *,
    train_timestamps: list[str],
    val_timestamps: list[str],
    test_timestamps: list[str],
) -> str:
    payload = {
        "train": sorted(train_timestamps),
        "val": sorted(val_timestamps),
        "test": sorted(test_timestamps),
    }
    return _sha256_text(_canonical_json(payload))


def compute_dataset_fingerprint(
    *,
    asset: str,
    timestamp_min: str,
    timestamp_max: str,
    row_count: int,
    close_sum: float,
    volume_sum: float,
    parquet_file_hash: str,
) -> str:
    payload = {
        "asset": asset,
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
        "row_count": int(row_count),
        "close_sum": float(close_sum),
        "volume_sum": float(volume_sum),
        "parquet_file_hash": parquet_file_hash,
    }
    return _sha256_text(_canonical_json(payload))


def compute_dataset_fingerprint_from_file(
    *,
    asset: str,
    timestamp_min: str,
    timestamp_max: str,
    row_count: int,
    close_sum: float,
    volume_sum: float,
    parquet_path: Path,
) -> str:
    return compute_dataset_fingerprint(
        asset=asset,
        timestamp_min=timestamp_min,
        timestamp_max=timestamp_max,
        row_count=row_count,
        close_sum=close_sum,
        volume_sum=volume_sum,
        parquet_file_hash=_sha256_file(parquet_path),
    )


def compute_run_id(
    *,
    asset: str,
    feature_set_hash: str,
    trial_number: int | None,
    fold: str | None,
    seed: int | None,
    model_version: str,
    config_signature: str,
    split_signature: str,
    pipeline_version: str,
) -> str:
    payload = {
        "asset": asset,
        "feature_set_hash": feature_set_hash,
        "trial_number": trial_number,
        "fold": fold,
        "seed": seed,
        "model_version": model_version,
        "config_signature": config_signature,
        "split_signature": split_signature,
        "pipeline_version": pipeline_version,
    }
    return _sha256_text(_canonical_json(payload))


def derive_run_status(
    *,
    train_completed: bool,
    has_epoch_metrics: bool,
    has_oos_predictions: bool,
    failed: bool,
) -> str:
    if failed:
        return "failed"
    if train_completed and has_epoch_metrics and has_oos_predictions:
        return "ok"
    if train_completed:
        return "partial_failed"
    return "failed"


@dataclass(frozen=True)
class AnalyticsTableSchema:
    name: str
    schema_version: int
    columns: dict[str, str]
    required_columns: tuple[str, ...]
    logical_pk: tuple[str, ...]
    partition_by: tuple[str, ...]
    update_policy: str  # upsert | append-only


DIM_RUN_SCHEMA = AnalyticsTableSchema(
    name="dim_run",
    schema_version=ANALYTICS_SCHEMA_VERSION,
    columns={
        "schema_version": "int64",
        "run_id": "string",
        "execution_id": "string",
        "parent_sweep_id": "string",
        "trial_number": "int64",
        "fold": "string",
        "seed": "int64",
        "asset": "string",
        "feature_set_name": "string",
        "feature_set_hash": "string",
        "feature_list_ordered_json": "string",
        "config_signature": "string",
        "split_fingerprint": "string",
        "model_version": "string",
        "checkpoint_path_final": "string",
        "checkpoint_path_best": "string",
        "git_commit": "string",
        "pipeline_version": "string",
        "library_versions_json": "string",
        "hardware_info_json": "string",
        "status": "string",
        "duration_total_seconds": "float64",
        "eta_recorded_seconds": "float64",
        "retries": "int64",
        "created_at_utc": "string",
    },
    required_columns=(
        "schema_version",
        "run_id",
        "model_version",
        "asset",
        "feature_set_name",
        "feature_set_hash",
        "feature_list_ordered_json",
        "config_signature",
        "split_fingerprint",
        "model_version",
        "status",
        "pipeline_version",
        "created_at_utc",
    ),
    logical_pk=("run_id",),
    partition_by=("asset", "parent_sweep_id"),
    update_policy="upsert",
)

FACT_RUN_SNAPSHOT_SCHEMA = AnalyticsTableSchema(
    name="fact_run_snapshot",
    schema_version=ANALYTICS_SCHEMA_VERSION,
    columns={
        "schema_version": "int64",
        "run_id": "string",
        "asset": "string",
        "parent_sweep_id": "string",
        "dataset_start_utc": "string",
        "dataset_end_utc": "string",
        "train_start_utc": "string",
        "train_end_utc": "string",
        "val_start_utc": "string",
        "val_end_utc": "string",
        "test_start_utc": "string",
        "test_end_utc": "string",
        "warmup_policy": "string",
        "required_warmup_count": "int64",
        "warmup_applied": "string",
        "effective_train_start_utc": "string",
        "n_samples_train": "int64",
        "n_samples_val": "int64",
        "n_samples_test": "int64",
        "dataset_fingerprint": "string",
        "split_fingerprint": "string",
    },
    required_columns=(
        "schema_version",
        "run_id",
        "asset",
        "dataset_start_utc",
        "dataset_end_utc",
        "train_start_utc",
        "train_end_utc",
        "val_start_utc",
        "val_end_utc",
        "test_start_utc",
        "test_end_utc",
        "warmup_policy",
        "n_samples_train",
        "n_samples_val",
        "n_samples_test",
        "dataset_fingerprint",
        "split_fingerprint",
    ),
    logical_pk=("run_id",),
    partition_by=("asset", "parent_sweep_id"),
    update_policy="append-only",
)

FACT_CONFIG_SCHEMA = AnalyticsTableSchema(
    name="fact_config",
    schema_version=ANALYTICS_SCHEMA_VERSION,
    columns={
        "schema_version": "int64",
        "run_id": "string",
        "asset": "string",
        "parent_sweep_id": "string",
        "prediction_mode": "string",
        "loss_name": "string",
        "quantile_levels_json": "string",
        "evaluation_horizons_json": "string",
        "max_encoder_length": "int64",
        "max_prediction_length": "int64",
        "batch_size": "int64",
        "max_epochs": "int64",
        "learning_rate": "float64",
        "hidden_size": "int64",
        "attention_head_size": "int64",
        "dropout": "float64",
        "hidden_continuous_size": "int64",
        "early_stopping_patience": "int64",
        "early_stopping_min_delta": "float64",
        "scaler_type": "string",
        "training_config_json": "string",
        "dataset_parameters_json": "string",
        "search_space_json": "string",
        "objective_name": "string",
        "objective_direction": "string",
    },
    required_columns=(
        "schema_version",
        "run_id",
        "asset",
        "prediction_mode",
        "loss_name",
        "quantile_levels_json",
        "max_encoder_length",
        "max_prediction_length",
        "batch_size",
        "max_epochs",
        "learning_rate",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "training_config_json",
        "dataset_parameters_json",
        "objective_name",
        "objective_direction",
    ),
    logical_pk=("run_id",),
    partition_by=("asset", "parent_sweep_id"),
    update_policy="append-only",
)

FACT_EPOCH_METRICS_SCHEMA = AnalyticsTableSchema(
    name="fact_epoch_metrics",
    schema_version=ANALYTICS_SCHEMA_VERSION,
    columns={
        "schema_version": "int64",
        "run_id": "string",
        "asset": "string",
        "parent_sweep_id": "string",
        "fold": "string",
        "epoch": "int64",
        "train_loss": "float64",
        "val_loss": "float64",
        "epoch_time_seconds": "float64",
        "best_epoch": "int64",
        "stopped_epoch": "int64",
        "early_stop_reason": "string",
    },
    required_columns=(
        "schema_version",
        "run_id",
        "asset",
        "fold",
        "epoch",
        "train_loss",
        "val_loss",
    ),
    logical_pk=("run_id", "epoch"),
    partition_by=("asset", "parent_sweep_id", "fold"),
    update_policy="append-only",
)

FACT_SPLIT_METRICS_SCHEMA = AnalyticsTableSchema(
    name="fact_split_metrics",
    schema_version=ANALYTICS_SCHEMA_VERSION,
    columns={
        "schema_version": "int64",
        "run_id": "string",
        "asset": "string",
        "parent_sweep_id": "string",
        "split": "string",
        "rmse": "float64",
        "mae": "float64",
        "mape": "float64",
        "smape": "float64",
        "directional_accuracy": "float64",
        "n_samples": "int64",
    },
    required_columns=(
        "schema_version",
        "run_id",
        "split",
        "rmse",
        "mae",
        "directional_accuracy",
        "n_samples",
    ),
    logical_pk=("run_id", "split"),
    partition_by=("asset", "parent_sweep_id"),
    update_policy="append-only",
)

FACT_OOS_PREDICTIONS_SCHEMA = AnalyticsTableSchema(
    name="fact_oos_predictions",
    schema_version=ANALYTICS_SCHEMA_VERSION,
    columns={
        "schema_version": "int64",
        "run_id": "string",
        "model_version": "string",
        "asset": "string",
        "feature_set_name": "string",
        "split": "string",
        "fold": "string",
        "seed": "int64",
        "horizon": "int64",
        # decision_idx materializes the ADR-0003 anchor convention:
        # timestamp_utc = dataset_timestamps[decision_idx]. The column is
        # the schema-level invariant (mechanical > procedural).
        "decision_idx": "int64",
        "timestamp_utc": "string",
        "target_timestamp_utc": "string",
        "y_true": "float64",
        "y_pred": "float64",
        "error": "float64",
        "abs_error": "float64",
        "sq_error": "float64",
        "quantile_p10": "float64",
        "quantile_p50": "float64",
        "quantile_p90": "float64",
        "quantile_p10_post_guardrail": "float64",
        "quantile_p50_post_guardrail": "float64",
        "quantile_p90_post_guardrail": "float64",
        "quantile_guardrail_applied": "int64",
        "year": "int64",
    },
    required_columns=(
        "schema_version",
        "run_id",
        "model_version",
        "asset",
        "feature_set_name",
        "split",
        "horizon",
        "decision_idx",
        "timestamp_utc",
        "target_timestamp_utc",
        "y_true",
        "y_pred",
        "error",
        "abs_error",
        "sq_error",
        "quantile_p10",
        "quantile_p50",
        "quantile_p90",
        "year",
    ),
    logical_pk=("run_id", "split", "horizon", "timestamp_utc", "target_timestamp_utc"),
    partition_by=("asset", "feature_set_name", "year"),
    update_policy="append-only",
)

FACT_FAILURES_SCHEMA = AnalyticsTableSchema(
    name="fact_failures",
    schema_version=ANALYTICS_SCHEMA_VERSION,
    columns={
        "schema_version": "int64",
        "run_id": "string",
        "execution_id": "string",
        "asset": "string",
        "failed_at_utc": "string",
        "stage": "string",
        "error_type": "string",
        "error_message": "string",
        "trace_hash": "string",
        "traceback_excerpt": "string",
        "entrypoint": "string",
        "cmdline": "string",
        "stdout_truncated": "string",
        "stderr_truncated": "string",
    },
    required_columns=(
        "schema_version",
        "run_id",
        "failed_at_utc",
        "stage",
        "error_type",
        "error_message",
        "trace_hash",
    ),
    logical_pk=("run_id", "failed_at_utc", "stage"),
    partition_by=("asset",),
    update_policy="append-only",
)

FACT_MODEL_ARTIFACTS_SCHEMA = AnalyticsTableSchema(
    name="fact_model_artifacts",
    schema_version=ANALYTICS_SCHEMA_VERSION,
    columns={
        "schema_version": "int64",
        # run_id and training_run_id both point to the producing dim_run.run_id.
        "run_id": "string",
        "training_run_id": "string",
        "asset": "string",
        "model_version": "string",
        "checkpoint_path_final": "string",
        "checkpoint_path_best": "string",
        "config_path": "string",
        "scaler_path": "string",
        "encoder_path": "string",
        "feature_importance_json": "string",
        "attention_summary_json": "string",
        "logs_ref_json": "string",
    },
    required_columns=(
        "schema_version",
        "run_id",
        "model_version",
        "checkpoint_path_final",
        "config_path",
        "feature_importance_json",
        "attention_summary_json",
    ),
    logical_pk=("run_id",),
    partition_by=("asset",),
    update_policy="append-only",
)

FACT_INFERENCE_RUNS_SCHEMA = AnalyticsTableSchema(
    name="fact_inference_runs",
    schema_version=ANALYTICS_SCHEMA_VERSION,
    columns={
        "schema_version": "int64",
        # run_id/training_run_id: FK to dim_run.run_id for the loaded training artifact.
        # inference_run_id: PK of the inference batch created by RunTFTInferenceUseCase.
        "run_id": "string",
        "training_run_id": "string",
        "inference_run_id": "string",
        "model_version": "string",
        "asset": "string",
        "inference_start_utc": "string",
        "inference_end_utc": "string",
        "overwrite": "string",
        "batch_size": "int64",
        "status": "string",
        "inferred_count": "int64",
        "skipped_count": "int64",
        "upserts_count": "int64",
        "duration_seconds": "float64",
    },
    required_columns=(
        "schema_version",
        "inference_run_id",
        "model_version",
        "asset",
        "inference_start_utc",
        "inference_end_utc",
        "status",
    ),
    logical_pk=("inference_run_id",),
    partition_by=("asset",),
    update_policy="append-only",
)

FACT_INFERENCE_PREDICTIONS_SCHEMA = AnalyticsTableSchema(
    name="fact_inference_predictions",
    schema_version=ANALYTICS_SCHEMA_VERSION,
    columns={
        "schema_version": "int64",
        # inference_run_id: FK to the inference batch; run_id/training_run_id: FK to dim_run.run_id.
        "inference_run_id": "string",
        "run_id": "string",
        "training_run_id": "string",
        "model_version": "string",
        "asset": "string",
        "feature_set_name": "string",
        "features_used_csv": "string",
        "model_path": "string",
        "split": "string",
        "horizon": "int64",
        # decision_idx materializes the ADR-0003 anchor. Column mirrors
        # fact_oos_predictions; not required here yet because the inference
        # writer is migrated post-Stage 20 (see ADR-0003 §C1).
        "decision_idx": "int64",
        "timestamp_utc": "string",
        "target_timestamp_utc": "string",
        "y_true": "float64",
        "y_pred": "float64",
        "error": "float64",
        "abs_error": "float64",
        "sq_error": "float64",
        "quantile_p10": "float64",
        "quantile_p50": "float64",
        "quantile_p90": "float64",
        "quantile_p10_post_guardrail": "float64",
        "quantile_p50_post_guardrail": "float64",
        "quantile_p90_post_guardrail": "float64",
        "quantile_guardrail_applied": "int64",
        "year": "int64",
        "created_at_utc": "string",
    },
    required_columns=(
        "schema_version",
        "inference_run_id",
        "model_version",
        "asset",
        "feature_set_name",
        "features_used_csv",
        "model_path",
        "split",
        "horizon",
        "timestamp_utc",
        "target_timestamp_utc",
        "y_pred",
        "year",
        "created_at_utc",
    ),
    logical_pk=("inference_run_id", "horizon", "timestamp_utc", "target_timestamp_utc"),
    partition_by=("asset", "model_version", "year"),
    update_policy="append-only",
)

FACT_FEATURE_CONTRIB_LOCAL_SCHEMA = AnalyticsTableSchema(
    name="fact_feature_contrib_local",
    schema_version=ANALYTICS_SCHEMA_VERSION,
    columns={
        "schema_version": "int64",
        # inference_run_id: FK to the inference batch; run_id/training_run_id: FK to dim_run.run_id.
        "inference_run_id": "string",
        "run_id": "string",
        "training_run_id": "string",
        "model_version": "string",
        "asset": "string",
        "feature_set_name": "string",
        "split": "string",
        "horizon": "int64",
        "timestamp_utc": "string",
        "target_timestamp_utc": "string",
        "feature_name": "string",
        "feature_rank": "int64",
        "contribution": "float64",
        "abs_contribution": "float64",
        "contribution_sign": "string",
        "method": "string",
        "year": "int64",
        "created_at_utc": "string",
    },
    required_columns=(
        "schema_version",
        "inference_run_id",
        "model_version",
        "asset",
        "feature_set_name",
        "split",
        "horizon",
        "timestamp_utc",
        "target_timestamp_utc",
        "feature_name",
        "feature_rank",
        "contribution",
        "abs_contribution",
        "contribution_sign",
        "method",
        "year",
        "created_at_utc",
    ),
    logical_pk=("inference_run_id", "horizon", "timestamp_utc", "target_timestamp_utc", "feature_name"),
    partition_by=("asset", "model_version", "year"),
    update_policy="append-only",
)

BRIDGE_RUN_FEATURES_SCHEMA = AnalyticsTableSchema(
    name="bridge_run_features",
    schema_version=ANALYTICS_SCHEMA_VERSION,
    columns={
        "schema_version": "int64",
        "run_id": "string",
        "feature_order": "int64",
        "feature_name": "string",
    },
    required_columns=("schema_version", "run_id", "feature_order", "feature_name"),
    logical_pk=("run_id", "feature_order"),
    partition_by=(),
    update_policy="append-only",
)

FACT_SPLIT_TIMESTAMPS_REF_SCHEMA = AnalyticsTableSchema(
    name="fact_split_timestamps_ref",
    schema_version=ANALYTICS_SCHEMA_VERSION,
    columns={
        "schema_version": "int64",
        "run_id": "string",
        "split": "string",
        "artifact_path": "string",
        "artifact_hash": "string",
        "timestamps_compact_json": "string",
    },
    required_columns=(
        "schema_version",
        "run_id",
        "split",
    ),
    logical_pk=("run_id", "split"),
    partition_by=("run_id",),
    update_policy="append-only",
)


ANALYTICS_TABLE_SCHEMAS: dict[str, AnalyticsTableSchema] = {
    s.name: s
    for s in (
        DIM_RUN_SCHEMA,
        FACT_RUN_SNAPSHOT_SCHEMA,
        FACT_CONFIG_SCHEMA,
        FACT_SPLIT_METRICS_SCHEMA,
        FACT_EPOCH_METRICS_SCHEMA,
        FACT_OOS_PREDICTIONS_SCHEMA,
        FACT_MODEL_ARTIFACTS_SCHEMA,
        FACT_FAILURES_SCHEMA,
        FACT_INFERENCE_RUNS_SCHEMA,
        FACT_INFERENCE_PREDICTIONS_SCHEMA,
        FACT_FEATURE_CONTRIB_LOCAL_SCHEMA,
        BRIDGE_RUN_FEATURES_SCHEMA,
        FACT_SPLIT_TIMESTAMPS_REF_SCHEMA,
    )
}



def _is_int_like(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_float_like(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_value_of_dtype(value: Any, dtype: str) -> bool:
    if dtype == "string":
        return isinstance(value, str)
    if dtype == "int64":
        return _is_int_like(value)
    if dtype == "float64":
        return _is_float_like(value)
    return True


def validate_table_payload(table_name: str, rows: list[dict[str, Any]]) -> None:
    if table_name not in ANALYTICS_TABLE_SCHEMAS:
        raise ValueError(f"Unknown analytics table: {table_name}")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Payload must be a non-empty list of rows")

    schema = ANALYTICS_TABLE_SCHEMAS[table_name]
    required = set(schema.required_columns)
    pk_cols = schema.logical_pk

    seen_pk: set[tuple[Any, ...]] = set()

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {idx} must be a dictionary")

        missing_required = [c for c in required if c not in row]
        if missing_required:
            raise ValueError(
                f"{table_name} row {idx} missing required columns: {sorted(missing_required)}"
            )

        if "schema_version" not in row:
            raise ValueError(f"{table_name} row {idx} missing schema_version")
        if row["schema_version"] != schema.schema_version:
            raise ValueError(
                f"{table_name} row {idx} has invalid schema_version={row['schema_version']} "
                f"(expected={schema.schema_version})"
            )

        for col in required:
            if row.get(col) is None:
                raise ValueError(f"{table_name} row {idx} has null required field: {col}")

        if table_name == "dim_run":
            status = str(row.get("status")).strip().lower()
            if status not in ANALYTICS_RUN_STATUSES:
                raise ValueError(f"{table_name} row {idx} has invalid status: {status}")

        if table_name == "fact_config":
            mode = str(row.get("prediction_mode")).strip().lower()
            if mode not in ANALYTICS_PREDICTION_MODES:
                raise ValueError(f"{table_name} row {idx} has invalid prediction_mode: {mode}")

        for col, value in row.items():
            if col not in schema.columns or value is None:
                continue
            expected_dtype = schema.columns[col]
            if not _is_value_of_dtype(value, expected_dtype):
                raise ValueError(
                    f"{table_name} row {idx} has invalid dtype for '{col}': "
                    f"expected={expected_dtype}, got={type(value).__name__}"
                )

        pk = tuple(row.get(c) for c in pk_cols)
        if any(v is None for v in pk):
            raise ValueError(f"{table_name} row {idx} has null value in PK columns {pk_cols}")
        if pk in seen_pk:
            raise ValueError(f"{table_name} payload has duplicate logical PK: {pk}")
        seen_pk.add(pk)
