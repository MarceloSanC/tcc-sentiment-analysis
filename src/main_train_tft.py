from __future__ import annotations

import argparse
import json
import logging
import sys

from pathlib import Path

import yaml

from src.adapters.local_tft_model_repository import LocalTFTModelRepository
from src.adapters.parquet_analytics_run_repository import ParquetAnalyticsRunRepository
from src.adapters.parquet_tft_dataset_repository import ParquetTFTDatasetRepository
from src.adapters.pytorch_forecasting_tft_trainer import PytorchForecastingTFTTrainer
from src.infrastructure.schemas.model_artifact_schema import (
    TFT_SPLIT_DEFAULTS,
    TFT_TRAINING_DEFAULTS,
    validate_tft_training_config,
)
from src.use_cases.train_tft_model_use_case import TrainTFTModelUseCase
from src.utils.asset_periods import resolve_training_split
from src.utils.feature_token_parser import (
    normalize_feature_tokens,
    parse_feature_tokens,
)
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import load_data_paths

logger = logging.getLogger(__name__)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y"}:
        return True
    if v in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_quantile_levels(value: str) -> list[float]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "quantile-levels must be a JSON list (e.g. '[0.1, 0.5, 0.9]')"
        ) from exc
    if not isinstance(parsed, list) or not parsed:
        raise argparse.ArgumentTypeError("quantile-levels must be a non-empty list")
    out: list[float] = []
    for q in parsed:
        if not isinstance(q, (int, float)):
            raise argparse.ArgumentTypeError("quantile-levels values must be numeric")
        out.append(float(q))
    return out


def _parse_horizons(value: str) -> list[int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "evaluation-horizons must be a JSON list (e.g. '[1, 7, 30]')"
        ) from exc
    if not isinstance(parsed, list) or not parsed:
        raise argparse.ArgumentTypeError("evaluation-horizons must be a non-empty list")
    out: list[int] = []
    for h in parsed:
        if not isinstance(h, (int, float)):
            raise argparse.ArgumentTypeError("evaluation-horizons values must be numeric")
        hv = int(h)
        if hv < 1:
            raise argparse.ArgumentTypeError("evaluation-horizons values must be >= 1")
        out.append(hv)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TFT model from dataset_tft with configurable feature sets and split."
    )
    parser.add_argument("--asset", required=True, help="Asset symbol. Example: AAPL")
    parser.add_argument(
        "--features",
        type=str,
        required=False,
        help=(
            "Comma-separated feature tokens. Accepts group tokens "
            "(BASELINE_FEATURES, TECHNICAL_FEATURES, SENTIMENT_FEATURES, "
            "FUNDAMENTAL_FEATURES) and/or explicit columns "
            "(e.g. close,ema_50,sentiment_score)."
        ),
    )
    parser.add_argument(
        "--config-json",
        type=str,
        required=False,
        help=(
            "Path to a JSON file with training parameters. "
            "Merge order: defaults <- JSON <- CLI."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        required=False,
        help=(
            "Optional base directory for model artifacts. "
            "If omitted, uses configured default path from data_paths."
        ),
    )
    parser.add_argument(
        "--max-encoder-length",
        type=int,
        help=f"Encoder length (>=2). Default: {TFT_TRAINING_DEFAULTS['max_encoder_length']}",
    )
    parser.add_argument(
        "--max-prediction-length",
        type=int,
        help=(
            f"Prediction horizon (>=1, <= encoder). Default: "
            f"{TFT_TRAINING_DEFAULTS['max_prediction_length']}"
        ),
    )
    parser.add_argument(
        "--evaluation-horizons",
        type=_parse_horizons,
        help=(
            "Horizons to persist/evaluate as JSON list. "
            f"Default: {TFT_TRAINING_DEFAULTS['evaluation_horizons']}"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help=f"Batch size (>=1). Default: {TFT_TRAINING_DEFAULTS['batch_size']}",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        help=f"Max epochs (>=1). Default: {TFT_TRAINING_DEFAULTS['max_epochs']}",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help=f"Learning rate (>0). Default: {TFT_TRAINING_DEFAULTS['learning_rate']}",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        help=f"Hidden size (>=1). Default: {TFT_TRAINING_DEFAULTS['hidden_size']}",
    )
    parser.add_argument(
        "--attention-head-size",
        type=int,
        help=(
            "Attention head size (>=1). "
            f"Default: {TFT_TRAINING_DEFAULTS['attention_head_size']}"
        ),
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help=f"Dropout [0,1]. Default: {TFT_TRAINING_DEFAULTS['dropout']}",
    )
    parser.add_argument(
        "--hidden-continuous-size",
        type=int,
        help=(
            "Hidden size for continuous vars (>=1). "
            f"Default: {TFT_TRAINING_DEFAULTS['hidden_continuous_size']}"
        ),
    )
    parser.add_argument(
        "--seed", type=int, help=f"Random seed. Default: {TFT_TRAINING_DEFAULTS['seed']}"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help=(
            "Early stopping patience (>=0). "
            f"Default: {TFT_TRAINING_DEFAULTS['early_stopping_patience']}"
        ),
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        help=(
            "Early stopping min delta (>=0). "
            f"Default: {TFT_TRAINING_DEFAULTS['early_stopping_min_delta']}"
        ),
    )
    parser.add_argument(
        "--prediction-mode",
        type=str,
        choices=["point", "quantile"],
        help=f"Prediction mode. Default: {TFT_TRAINING_DEFAULTS['prediction_mode']}",
    )
    parser.add_argument(
        "--quantile-levels",
        type=_parse_quantile_levels,
        help=(
            "Quantile levels as JSON list. "
            f"Default: {TFT_TRAINING_DEFAULTS['quantile_levels']}"
        ),
    )
    parser.add_argument(
        "--warmup-policy",
        type=str,
        choices=["strict_fail", "drop_leading"],
        help=f"Warmup policy. Default: {TFT_TRAINING_DEFAULTS['warmup_policy']}",
    )
    parser.add_argument(
        "--min-samples-train",
        type=int,
        help=f"Minimum train samples. Default: {TFT_TRAINING_DEFAULTS['min_samples_train']}",
    )
    parser.add_argument(
        "--min-samples-val",
        type=int,
        help=f"Minimum validation samples. Default: {TFT_TRAINING_DEFAULTS['min_samples_val']}",
    )
    parser.add_argument(
        "--min-samples-test",
        type=int,
        help=f"Minimum test samples. Default: {TFT_TRAINING_DEFAULTS['min_samples_test']}",
    )
    parser.add_argument(
        "--quality-gate-max-nan-ratio-per-feature",
        type=float,
        help=(
            "Max NaN ratio per feature allowed in quality gate. "
            f"Default: {TFT_TRAINING_DEFAULTS['quality_gate_max_nan_ratio_per_feature']}"
        ),
    )
    parser.add_argument(
        "--quality-gate-min-temporal-coverage-days",
        type=int,
        help=(
            "Minimum temporal coverage (days) for quality gate. "
            f"Default: {TFT_TRAINING_DEFAULTS['quality_gate_min_temporal_coverage_days']}"
        ),
    )
    parser.add_argument(
        "--quality-gate-require-unique-timestamps",
        type=_parse_bool,
        help=(
            "Require unique timestamps in quality gate. "
            f"Default: {TFT_TRAINING_DEFAULTS['quality_gate_require_unique_timestamps']}"
        ),
    )
    parser.add_argument(
        "--quality-gate-require-monotonic-timestamps",
        type=_parse_bool,
        help=(
            "Require monotonic timestamps in quality gate. "
            f"Default: {TFT_TRAINING_DEFAULTS['quality_gate_require_monotonic_timestamps']}"
        ),
    )
    parser.add_argument(
        "--store-split-timestamps-ref",
        type=_parse_bool,
        help=(
            "Persist compact split timestamp references in analytics. "
            f"Default: {TFT_TRAINING_DEFAULTS['store_split_timestamps_ref']}"
        ),
    )
    parser.add_argument(
        "--evaluate-train-split",
        type=_parse_bool,
        help=(
            "Run post-train prediction/metrics for train split. "
            f"Default: {TFT_TRAINING_DEFAULTS['evaluate_train_split']}"
        ),
    )
    parser.add_argument(
        "--compute-feature-importance",
        type=_parse_bool,
        help=(
            "Compute permutation feature importance after training. "
            f"Default: {TFT_TRAINING_DEFAULTS['compute_feature_importance']}"
        ),
    )
    parser.add_argument(
        "--parent-sweep-id",
        type=str,
        help="Optional sweep lineage identifier persisted in analytics (e.g. optuna_0_1_5_*).",
    )
    parser.add_argument("--train-start", type=str, help="Train start date (yyyymmdd)")
    parser.add_argument("--train-end", type=str, help="Train end date (yyyymmdd)")
    parser.add_argument("--val-start", type=str, help="Validation start date (yyyymmdd)")
    parser.add_argument("--val-end", type=str, help="Validation end date (yyyymmdd)")
    parser.add_argument("--test-start", type=str, help="Test start date (yyyymmdd)")
    parser.add_argument("--test-end", type=str, help="Test end date (yyyymmdd)")
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run ablation experiments in addition to main training (disabled by default).",
    )
    parser.add_argument(
        "--overwrite-on-collision",
        action="store_true",
        help="Overwrite colliding analytics silver rows instead of failing on logical PK collision.",
    )
    return parser.parse_args()


def _load_json_config(path: str | None) -> dict:
    if not path:
        return {}
    config_path = Path(path)
    if not config_path.exists() or not config_path.is_file():
        raise ValueError(f"Config JSON not found: {path}")
    try:
        content = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file: {path}") from exc
    if not isinstance(content, dict):
        raise ValueError("Config JSON root must be an object")
    return content


def _load_train_quality_gate_defaults() -> dict:
    cfg_path = Path(__file__).parent.parent / "config" / "quality" / "dataset_quality.yaml"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    section = payload.get("train_guard_rail", {})
    if not isinstance(section, dict):
        return {}
    return {
        "quality_gate_max_nan_ratio_per_feature": section.get(
            "max_nan_ratio_per_feature"
        ),
        "quality_gate_min_temporal_coverage_days": section.get(
            "min_temporal_coverage_days"
        ),
        "quality_gate_require_unique_timestamps": section.get(
            "require_unique_timestamps"
        ),
        "quality_gate_require_monotonic_timestamps": section.get(
            "require_monotonic_timestamps"
        ),
    }


def _load_asset_training_split_defaults(asset_id: str) -> dict:
    config_path = Path(__file__).parent.parent / "config" / "data_sources.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    assets = payload.get("assets", [])
    if not isinstance(assets, list):
        return {}
    asset_cfg = next(
        (a for a in assets if str(a.get("symbol", "")).strip().upper() == asset_id),
        None,
    )
    if not isinstance(asset_cfg, dict):
        return {}
    return resolve_training_split(asset_cfg) or {}


def main() -> None:
    setup_logging(logging.INFO)
    args = parse_args()

    file_config = _load_json_config(args.config_json)
    asset_id = args.asset.strip().upper()

    config_features = file_config.get("features")
    features: list[str] | None = None
    if args.features:
        features = parse_feature_tokens(args.features)
    elif isinstance(config_features, str):
        features = parse_feature_tokens(config_features)
    elif isinstance(config_features, list):
        features = normalize_feature_tokens([str(c) for c in config_features])

    paths = load_data_paths()
    dataset_dir = paths["dataset_tft"]
    models_dir = Path(args.models_dir) if args.models_dir else paths["models"]

    dataset_repo = ParquetTFTDatasetRepository(output_dir=dataset_dir)
    model_repo = LocalTFTModelRepository(
        base_dir=models_dir,
        run_subdir=None if args.models_dir else "runs",
    )
    analytics_repo = ParquetAnalyticsRunRepository(output_dir=paths.get("analytics_silver", Path("data/analytics/silver")))
    trainer = PytorchForecastingTFTTrainer()

    use_case = TrainTFTModelUseCase(
        dataset_repository=dataset_repo,
        model_trainer=trainer,
        model_repository=model_repo,
        analytics_run_repository=analytics_repo,
    )

    training_config = dict(TFT_TRAINING_DEFAULTS)
    qg_defaults = _load_train_quality_gate_defaults()
    for k, v in qg_defaults.items():
        if v is not None:
            training_config[k] = v
    json_training = file_config.get("training_config")
    if isinstance(json_training, dict):
        for key, value in json_training.items():
            if key in TFT_TRAINING_DEFAULTS:
                training_config[key] = value
        if "derived_feature_groups" in json_training:
            training_config["derived_feature_groups"] = json_training[
                "derived_feature_groups"
            ]
    for key in TFT_TRAINING_DEFAULTS:
        if key in file_config:
            training_config[key] = file_config[key]
    if "derived_feature_groups" in file_config:
        training_config["derived_feature_groups"] = file_config["derived_feature_groups"]
    if "parent_sweep_id" in file_config and file_config["parent_sweep_id"] is not None:
        training_config["parent_sweep_id"] = str(file_config["parent_sweep_id"])

    overrides = {
        "max_encoder_length": args.max_encoder_length,
        "max_prediction_length": args.max_prediction_length,
        "evaluation_horizons": args.evaluation_horizons,
        "batch_size": args.batch_size,
        "max_epochs": args.max_epochs,
        "learning_rate": args.learning_rate,
        "hidden_size": args.hidden_size,
        "attention_head_size": args.attention_head_size,
        "dropout": args.dropout,
        "hidden_continuous_size": args.hidden_continuous_size,
        "seed": args.seed,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "prediction_mode": args.prediction_mode,
        "quantile_levels": args.quantile_levels,
        "warmup_policy": args.warmup_policy,
        "min_samples_train": args.min_samples_train,
        "min_samples_val": args.min_samples_val,
        "min_samples_test": args.min_samples_test,
        "quality_gate_max_nan_ratio_per_feature": args.quality_gate_max_nan_ratio_per_feature,
        "quality_gate_min_temporal_coverage_days": args.quality_gate_min_temporal_coverage_days,
        "quality_gate_require_unique_timestamps": args.quality_gate_require_unique_timestamps,
        "quality_gate_require_monotonic_timestamps": args.quality_gate_require_monotonic_timestamps,
        "store_split_timestamps_ref": args.store_split_timestamps_ref,
        "evaluate_train_split": args.evaluate_train_split,
        "compute_feature_importance": args.compute_feature_importance,
    }
    for key, value in overrides.items():
        if value is not None:
            training_config[key] = value
    parent_sweep_id = getattr(args, "parent_sweep_id", None)
    if parent_sweep_id is not None:
        training_config["parent_sweep_id"] = str(parent_sweep_id)
    validate_tft_training_config(training_config)

    training_config["entrypoint"] = "src.main_train_tft"
    training_config["cmdline"] = " ".join(sys.argv)

    split_config = dict(TFT_SPLIT_DEFAULTS)
    split_config.update(_load_asset_training_split_defaults(asset_id))
    json_split = file_config.get("split_config")
    if isinstance(json_split, dict):
        for key, value in json_split.items():
            if key in TFT_SPLIT_DEFAULTS:
                split_config[key] = value
    for key in TFT_SPLIT_DEFAULTS:
        if key in file_config:
            split_config[key] = file_config[key]

    split_overrides = {
        "train_start": args.train_start,
        "train_end": args.train_end,
        "val_start": args.val_start,
        "val_end": args.val_end,
        "test_start": args.test_start,
        "test_end": args.test_end,
    }
    for key, value in split_overrides.items():
        if value is not None:
            split_config[key] = value

    run_ablation = bool(file_config.get("run_ablation", False)) or args.run_ablation
    overwrite_on_collision = bool(file_config.get("overwrite_on_collision", False)) or bool(
        args.overwrite_on_collision
    )

    result = use_case.execute(
        asset_id,
        features=features,
        training_config=training_config,
        split_config=split_config,
        run_ablation=run_ablation,
        overwrite_on_collision=overwrite_on_collision,
    )

    logger.info(
        "TFT training completed",
        extra={
            "asset_id": result.asset_id,
            "version": result.version,
            "artifacts_dir": result.artifacts_dir,
            "metrics": result.metrics,
            "split_config": split_config,
        },
    )


if __name__ == "__main__":
    main()
