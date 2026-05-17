# src/infrastructure/schemas/model_artifact_schema.py

MODEL_METADATA_FIELDS = {
    "model_type",
    "asset_id",
    "created_at",
    "training_run_id",
    "training_window",
    "features_used",
    "scaler_type",
    "version",
}

TFT_TRAINING_DEFAULTS = {
    "max_encoder_length": 60,
    "max_prediction_length": 1,
    "evaluation_horizons": [1, 7, 30],
    "batch_size": 64,
    "max_epochs": 20,
    "learning_rate": 5e-4,
    "hidden_size": 32,
    "attention_head_size": 2,
    "dropout": 0.1,
    "hidden_continuous_size": 8,
    "seed": 42,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.0,
    "prediction_mode": "quantile",
    "quantile_levels": [0.1, 0.5, 0.9],
    "warmup_policy": "strict_fail",
    "min_samples_train": 1,
    "min_samples_val": 1,
    "min_samples_test": 1,
    "quality_gate_max_nan_ratio_per_feature": 1.0,
    "quality_gate_min_temporal_coverage_days": 1,
    "quality_gate_require_unique_timestamps": True,
    "quality_gate_require_monotonic_timestamps": True,
    "store_split_timestamps_ref": False,
    "evaluate_train_split": False,
    "compute_feature_importance": False,
}

TFT_TRAINING_LIMITS = {
    "max_encoder_length": {"min": 2},
    "max_prediction_length": {"min": 1},
    "batch_size": {"min": 1},
    "max_epochs": {"min": 1},
    "learning_rate": {"min_exclusive": 0.0},
    "hidden_size": {"min": 1},
    "attention_head_size": {"min": 1},
    "dropout": {"min": 0.0, "max": 1.0},
    "hidden_continuous_size": {"min": 1},
    "early_stopping_patience": {"min": 0},
    "early_stopping_min_delta": {"min": 0.0},
    "min_samples_train": {"min": 1},
    "min_samples_val": {"min": 1},
    "min_samples_test": {"min": 1},
    "quality_gate_max_nan_ratio_per_feature": {"min": 0.0, "max": 1.0},
    "quality_gate_min_temporal_coverage_days": {"min": 1},
}

TFT_WARMUP_POLICIES = {"strict_fail", "drop_leading"}

TFT_SPLIT_DEFAULTS = {
    "train_start": "20100101",
    "train_end": "20221231",
    "val_start": "20230101",
    "val_end": "20241231",
    "test_start": "20250101",
    "test_end": "20251231",
}


def validate_tft_training_config(config: dict) -> None:
    if "max_prediction_length" in config and "max_encoder_length" in config:
        if int(config["max_prediction_length"]) > int(config["max_encoder_length"]):
            raise ValueError(
                "Invalid config: max_prediction_length must be <= max_encoder_length"
            )

    for key, rule in TFT_TRAINING_LIMITS.items():
        if key not in config or config[key] is None:
            continue

        value = config[key]
        if "min" in rule and value < rule["min"]:
            raise ValueError(f"Invalid config: {key} must be >= {rule['min']}")
        if "max" in rule and value > rule["max"]:
            raise ValueError(f"Invalid config: {key} must be <= {rule['max']}")
        if "min_exclusive" in rule and value <= rule["min_exclusive"]:
            raise ValueError(
                f"Invalid config: {key} must be > {rule['min_exclusive']}"
            )

    if "quality_gate_require_unique_timestamps" in config and not isinstance(
        config["quality_gate_require_unique_timestamps"], bool
    ):
        raise ValueError(
            "Invalid config: quality_gate_require_unique_timestamps must be boolean"
        )
    if "quality_gate_require_monotonic_timestamps" in config and not isinstance(
        config["quality_gate_require_monotonic_timestamps"], bool
    ):
        raise ValueError(
            "Invalid config: quality_gate_require_monotonic_timestamps must be boolean"
        )
    if "store_split_timestamps_ref" in config and not isinstance(
        config["store_split_timestamps_ref"], bool
    ):
        raise ValueError(
            "Invalid config: store_split_timestamps_ref must be boolean"
        )
    if "evaluate_train_split" in config and not isinstance(
        config["evaluate_train_split"], bool
    ):
        raise ValueError("Invalid config: evaluate_train_split must be boolean")
    if "compute_feature_importance" in config and not isinstance(
        config["compute_feature_importance"], bool
    ):
        raise ValueError("Invalid config: compute_feature_importance must be boolean")

    warmup_policy = str(config.get("warmup_policy", "strict_fail")).strip().lower()
    if warmup_policy not in TFT_WARMUP_POLICIES:
        raise ValueError(
            f"Invalid config: warmup_policy must be one of {sorted(TFT_WARMUP_POLICIES)}"
        )

    prediction_mode = str(config.get("prediction_mode", "quantile")).strip().lower()
    if prediction_mode not in {"point", "quantile"}:
        raise ValueError("Invalid config: prediction_mode must be one of ['point', 'quantile']")

    quantile_levels = config.get("quantile_levels", [0.1, 0.5, 0.9])
    if quantile_levels is not None:
        if not isinstance(quantile_levels, list) or not quantile_levels:
            raise ValueError("Invalid config: quantile_levels must be a non-empty list")
        for q in quantile_levels:
            if not isinstance(q, (int, float)):
                raise ValueError("Invalid config: quantile_levels values must be numeric")
            if float(q) <= 0.0 or float(q) >= 1.0:
                raise ValueError("Invalid config: quantile_levels must be in (0,1)")

    evaluation_horizons = config.get("evaluation_horizons", [1, 7, 30])
    if evaluation_horizons is not None:
        if not isinstance(evaluation_horizons, list) or not evaluation_horizons:
            raise ValueError("Invalid config: evaluation_horizons must be a non-empty list")
        normalized: list[int] = []
        for h in evaluation_horizons:
            if not isinstance(h, (int, float)):
                raise ValueError("Invalid config: evaluation_horizons values must be numeric")
            hv = int(h)
            if hv < 1:
                raise ValueError("Invalid config: evaluation_horizons must be >= 1")
            normalized.append(hv)
        max_prediction_length = int(
            config.get("max_prediction_length", TFT_TRAINING_DEFAULTS["max_prediction_length"])
        )
        filtered = sorted({h for h in normalized if h <= max_prediction_length})
        if not filtered:
            raise ValueError(
                "Invalid config: evaluation_horizons has no valid value <= max_prediction_length "
                f"({max_prediction_length})"
            )
        # Keep normalized/compatible horizons in the validated payload.
        config["evaluation_horizons"] = filtered
