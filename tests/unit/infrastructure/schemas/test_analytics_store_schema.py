from __future__ import annotations

from src.infrastructure.schemas.analytics_store_schema import (
    ANALYTICS_SCHEMA_VERSION,
    ANALYTICS_TABLE_SCHEMAS,
    BRIDGE_RUN_FEATURES_SCHEMA,
    DIM_RUN_SCHEMA,
    FACT_CONFIG_SCHEMA,
    FACT_EPOCH_METRICS_SCHEMA,
    FACT_FAILURES_SCHEMA,
    FACT_FEATURE_CONTRIB_LOCAL_SCHEMA,
    FACT_INFERENCE_PREDICTIONS_SCHEMA,
    FACT_INFERENCE_RUNS_SCHEMA,
    FACT_MODEL_ARTIFACTS_SCHEMA,
    FACT_OOS_PREDICTIONS_SCHEMA,
    FACT_RUN_SNAPSHOT_SCHEMA,
    FACT_SPLIT_METRICS_SCHEMA,
    FACT_SPLIT_TIMESTAMPS_REF_SCHEMA,
    compute_config_signature,
    compute_dataset_fingerprint,
    compute_feature_set_hash,
    compute_run_id,
    compute_split_fingerprint,
    derive_run_status,
    validate_table_payload,
)


def test_feature_set_hash_depends_on_feature_order() -> None:
    a = compute_feature_set_hash(["open", "close"])
    b = compute_feature_set_hash(["close", "open"])
    assert a != b


def test_config_signature_ignores_volatile_keys() -> None:
    c1 = compute_config_signature({"lr": 0.001, "created_at": "x"})
    c2 = compute_config_signature({"created_at": "y", "lr": 0.001})
    assert c1 == c2


def _base_training_config() -> dict[str, object]:
    return {
        "asset": "AAPL",
        "feature_set_name": "all_features",
        "parent_sweep_id": "phase_b_confirmatory_20260514",
        "fold": "wf_1",
        "trial_number": 7,
        "seed": 20260514,
        "learning_rate": 0.001,
        "hidden_size": 64,
    }


def test_config_signature_changes_when_parent_sweep_id_changes() -> None:
    base = _base_training_config()
    base_signature = compute_config_signature(base)

    with_other_parent = {**base, "parent_sweep_id": "phase_b_confirmatory_other"}
    with_other_fold = {**base, "fold": "wf_2"}
    with_other_trial = {**base, "trial_number": 8}
    with_other_seed = {**base, "seed": 20260515}

    assert compute_config_signature(with_other_parent) != base_signature
    assert compute_config_signature(with_other_fold) != base_signature
    assert compute_config_signature(with_other_trial) != base_signature
    assert compute_config_signature(with_other_seed) != base_signature


def test_config_signature_stable_across_volatile_keys() -> None:
    base = _base_training_config()
    with_volatile_timestamps = {
        **base,
        "created_at": "2026-05-14T10:00:00+00:00",
        "started_at": "2026-05-14T10:01:00+00:00",
        "ended_at": "2026-05-14T10:30:00+00:00",
        "timestamp": "2026-05-14T10:31:00+00:00",
    }
    with_other_volatile_timestamps = {
        **base,
        "created_at": "2026-05-15T10:00:00+00:00",
        "started_at": "2026-05-15T10:01:00+00:00",
        "ended_at": "2026-05-15T10:30:00+00:00",
        "timestamp": "2026-05-15T10:31:00+00:00",
    }

    assert compute_config_signature(with_volatile_timestamps) == compute_config_signature(
        with_other_volatile_timestamps
    )


def test_split_fingerprint_is_order_invariant_inside_each_split() -> None:
    s1 = compute_split_fingerprint(
        train_timestamps=["2024-01-02", "2024-01-01"],
        val_timestamps=["2024-01-03"],
        test_timestamps=["2024-01-04"],
    )
    s2 = compute_split_fingerprint(
        train_timestamps=["2024-01-01", "2024-01-02"],
        val_timestamps=["2024-01-03"],
        test_timestamps=["2024-01-04"],
    )
    assert s1 == s2


def test_dataset_fingerprint_changes_when_inputs_change() -> None:
    f1 = compute_dataset_fingerprint(
        asset="AAPL",
        timestamp_min="2024-01-01T00:00:00Z",
        timestamp_max="2024-01-02T00:00:00Z",
        row_count=2,
        close_sum=201.0,
        volume_sum=3000.0,
        parquet_file_hash="h1",
    )
    f2 = compute_dataset_fingerprint(
        asset="AAPL",
        timestamp_min="2024-01-01T00:00:00Z",
        timestamp_max="2024-01-02T00:00:00Z",
        row_count=2,
        close_sum=202.0,
        volume_sum=3000.0,
        parquet_file_hash="h1",
    )
    assert f1 != f2


def test_run_id_deterministic() -> None:
    rid1 = compute_run_id(
        asset="AAPL",
        feature_set_hash="fsh",
        trial_number=1,
        fold="wf_1",
        seed=7,
        model_version="v1",
        config_signature="cfg",
        split_signature="spl",
        pipeline_version="0.1",
    )
    rid2 = compute_run_id(
        asset="AAPL",
        feature_set_hash="fsh",
        trial_number=1,
        fold="wf_1",
        seed=7,
        model_version="v1",
        config_signature="cfg",
        split_signature="spl",
        pipeline_version="0.1",
    )
    assert rid1 == rid2


def test_derive_run_status_rules() -> None:
    assert (
        derive_run_status(
            train_completed=True,
            has_epoch_metrics=True,
            has_oos_predictions=True,
            failed=False,
        )
        == "ok"
    )
    assert (
        derive_run_status(
            train_completed=True,
            has_epoch_metrics=False,
            has_oos_predictions=True,
            failed=False,
        )
        == "partial_failed"
    )
    assert (
        derive_run_status(
            train_completed=False,
            has_epoch_metrics=False,
            has_oos_predictions=False,
            failed=True,
        )
        == "failed"
    )


def test_all_tables_have_schema_version_required() -> None:
    for schema in ANALYTICS_TABLE_SCHEMAS.values():
        assert "schema_version" in schema.columns
        assert "schema_version" in schema.required_columns
        assert schema.schema_version == ANALYTICS_SCHEMA_VERSION


def test_schema_version_bumped_to_2() -> None:
    assert ANALYTICS_SCHEMA_VERSION == 2


def test_partitioning_and_pk_contracts() -> None:
    assert DIM_RUN_SCHEMA.partition_by == ("asset", "parent_sweep_id")
    assert FACT_RUN_SNAPSHOT_SCHEMA.partition_by == ("asset", "parent_sweep_id")
    assert FACT_CONFIG_SCHEMA.partition_by == ("asset", "parent_sweep_id")
    assert FACT_EPOCH_METRICS_SCHEMA.partition_by == ("asset", "parent_sweep_id", "fold")
    assert FACT_OOS_PREDICTIONS_SCHEMA.partition_by == ("asset", "feature_set_name", "year")
    assert FACT_OOS_PREDICTIONS_SCHEMA.logical_pk == (
        "run_id",
        "split",
        "horizon",
        "timestamp_utc",
        "target_timestamp_utc",
    )
    assert FACT_FAILURES_SCHEMA.update_policy == "append-only"
    assert BRIDGE_RUN_FEATURES_SCHEMA.logical_pk == ("run_id", "feature_order")
    assert FACT_SPLIT_METRICS_SCHEMA.logical_pk == ("run_id", "split")
    assert FACT_INFERENCE_RUNS_SCHEMA.logical_pk == ("inference_run_id",)
    assert FACT_INFERENCE_PREDICTIONS_SCHEMA.logical_pk == ("inference_run_id", "horizon", "timestamp_utc", "target_timestamp_utc")
    assert FACT_INFERENCE_PREDICTIONS_SCHEMA.partition_by == ("asset", "model_version", "year")
    assert FACT_FEATURE_CONTRIB_LOCAL_SCHEMA.partition_by == ("asset", "model_version", "year")
    assert FACT_FEATURE_CONTRIB_LOCAL_SCHEMA.logical_pk == ("inference_run_id", "horizon", "timestamp_utc", "target_timestamp_utc", "feature_name")
    assert FACT_SPLIT_TIMESTAMPS_REF_SCHEMA.logical_pk == ("run_id", "split")


def test_quantile_columns_are_mandatory_in_oos_schema() -> None:
    required = set(FACT_OOS_PREDICTIONS_SCHEMA.required_columns)
    assert {"quantile_p10", "quantile_p50", "quantile_p90"}.issubset(required)
    assert "model_version" in required


def test_analytics_table_coverage_matches_checklist_core_tables() -> None:
    names = set(ANALYTICS_TABLE_SCHEMAS.keys())
    expected = {
        "dim_run",
        "fact_run_snapshot",
        "fact_config",
        "fact_split_metrics",
        "fact_epoch_metrics",
        "fact_oos_predictions",
        "fact_model_artifacts",
        "fact_failures",
        "fact_inference_runs",
        "fact_inference_predictions",
        "fact_feature_contrib_local",
        "bridge_run_features",
        "fact_split_timestamps_ref",
    }
    assert expected.issubset(names)


def test_dim_run_contains_traceability_fields() -> None:
    cols = set(DIM_RUN_SCHEMA.columns.keys())
    required = {
        "parent_sweep_id",
        "trial_number",
        "fold",
        "seed",
        "asset",
        "feature_set_name",
        "model_version",
        "split_fingerprint",
        "checkpoint_path_final",
        "checkpoint_path_best",
        "git_commit",
        "pipeline_version",
        "library_versions_json",
        "hardware_info_json",
        "status",
        "duration_total_seconds",
        "eta_recorded_seconds",
        "retries",
    }
    assert required.issubset(cols)


def test_fact_model_artifacts_requires_attention_and_importance() -> None:
    req = set(FACT_MODEL_ARTIFACTS_SCHEMA.required_columns)
    assert "feature_importance_json" in req
    assert "attention_summary_json" in req


def test_inference_schemas_declare_nullable_training_run_id() -> None:
    inference_schemas = (
        FACT_INFERENCE_RUNS_SCHEMA,
        FACT_INFERENCE_PREDICTIONS_SCHEMA,
        FACT_FEATURE_CONTRIB_LOCAL_SCHEMA,
    )
    for schema in inference_schemas:
        assert "training_run_id" in schema.columns
        assert "training_run_id" not in schema.required_columns

    assert "training_run_id" in FACT_MODEL_ARTIFACTS_SCHEMA.columns


def test_validate_table_payload_rejects_missing_required_column() -> None:
    row = {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "run_id": "r1",
        "asset": "AAPL",
        # missing feature_set_name
        "feature_set_hash": "h",
        "feature_list_ordered_json": "[]",
        "config_signature": "c",
        "split_fingerprint": "s",
        "model_version": "v1",
        "status": "ok",
        "pipeline_version": "0.1",
        "created_at_utc": "2026-01-01T00:00:00Z",
    }
    try:
        validate_table_payload("dim_run", [row])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "missing required columns" in str(exc)


def test_validate_table_payload_rejects_schema_version_mismatch() -> None:
    row = {
        "schema_version": 999,
        "run_id": "r1",
        "asset": "AAPL",
        "feature_set_name": "BASELINE_FEATURES",
        "feature_set_hash": "h",
        "feature_list_ordered_json": "[]",
        "config_signature": "c",
        "split_fingerprint": "s",
        "model_version": "v1",
        "status": "ok",
        "pipeline_version": "0.1",
        "created_at_utc": "2026-01-01T00:00:00Z",
    }
    try:
        validate_table_payload("dim_run", [row])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "invalid schema_version" in str(exc)


def test_validate_table_payload_rejects_duplicate_pk() -> None:
    row = {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "run_id": "r1",
        "asset": "AAPL",
        "feature_set_name": "BASELINE_FEATURES",
        "feature_set_hash": "h",
        "feature_list_ordered_json": "[]",
        "config_signature": "c",
        "split_fingerprint": "s",
        "model_version": "v1",
        "status": "ok",
        "pipeline_version": "0.1",
        "created_at_utc": "2026-01-01T00:00:00Z",
    }
    try:
        validate_table_payload("dim_run", [row, row])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "duplicate logical PK" in str(exc)


def test_validate_dim_run_requires_split_fingerprint() -> None:
    row = {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "run_id": "r1",
        "asset": "AAPL",
        "feature_set_name": "BASELINE_FEATURES",
        "feature_set_hash": "h",
        "feature_list_ordered_json": "[]",
        "config_signature": "c",
        "model_version": "v1",
        "status": "ok",
        "pipeline_version": "0.1",
        "created_at_utc": "2026-01-01T00:00:00Z",
    }
    try:
        validate_table_payload("dim_run", [row])
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "missing required columns" in str(exc)


def test_validate_table_payload_accepts_fact_inference_predictions_minimal_row() -> None:
    row = {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "inference_run_id": "inf_1",
        "run_id": None,
        "model_version": "20260308_010101_B",
        "asset": "AAPL",
        "feature_set_name": "BASELINE_FEATURES",
        "features_used_csv": "open,close",
        "model_path": "data/models/AAPL/runs/20260308_010101_B",
        "split": "inference",
        "horizon": 1,
        "timestamp_utc": "2026-03-08T00:00:00+00:00",
        "target_timestamp_utc": "2026-03-08T00:00:00+00:00",
        "y_true": None,
        "y_pred": 0.01,
        "error": None,
        "abs_error": None,
        "sq_error": None,
        "quantile_p10": -0.01,
        "quantile_p50": 0.01,
        "quantile_p90": 0.03,
        "year": 2026,
        "created_at_utc": "2026-03-08T00:00:01+00:00",
    }
    validate_table_payload("fact_inference_predictions", [row])


def test_validate_table_payload_accepts_fact_feature_contrib_local_minimal_row() -> None:
    row = {
        "schema_version": ANALYTICS_SCHEMA_VERSION,
        "inference_run_id": "inf_1",
        "run_id": None,
        "model_version": "20260308_010101_B",
        "asset": "AAPL",
        "feature_set_name": "BASELINE_FEATURES",
        "split": "inference",
        "horizon": 1,
        "timestamp_utc": "2026-03-08T00:00:00+00:00",
        "target_timestamp_utc": "2026-03-08T00:00:00+00:00",
        "feature_name": "close",
        "feature_rank": 1,
        "contribution": 0.01,
        "abs_contribution": 0.01,
        "contribution_sign": "positive",
        "method": "local_magnitude_signed_v1",
        "year": 2026,
        "created_at_utc": "2026-03-08T00:00:01+00:00",
    }
    validate_table_payload("fact_feature_contrib_local", [row])


def test_guardrail_quantile_columns_declared_in_oos_and_inference_schemas() -> None:
    oos_cols = set(FACT_OOS_PREDICTIONS_SCHEMA.columns.keys())
    inf_cols = set(FACT_INFERENCE_PREDICTIONS_SCHEMA.columns.keys())
    expected = {
        "quantile_p10_post_guardrail",
        "quantile_p50_post_guardrail",
        "quantile_p90_post_guardrail",
        "quantile_guardrail_applied",
    }
    assert expected.issubset(oos_cols)
    assert expected.issubset(inf_cols)
