from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.adapters.parquet_tft_dataset_repository import ParquetTFTDatasetRepository
from src.adapters.pytorch_forecasting_tft_trainer import PytorchForecastingTFTTrainer
from src.adapters.local_tft_model_repository import LocalTFTModelRepository
from src.use_cases.train_tft_model_use_case import TrainTFTModelUseCase
from src.infrastructure.schemas.model_artifact_schema import (
    TFT_TRAINING_DEFAULTS,
    TFT_SPLIT_DEFAULTS,
    validate_tft_training_config,
)
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import load_data_paths

logger = logging.getLogger(__name__)


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


def main() -> None:
    setup_logging(logging.INFO)
    args = parse_args()

    file_config = _load_json_config(args.config_json)
    asset_id = args.asset.strip().upper()

    config_features = file_config.get("features")
    features: list[str] | None = None
    if args.features:
        features = [c.strip() for c in args.features.split(",") if c.strip()]
    elif isinstance(config_features, str):
        features = [c.strip() for c in config_features.split(",") if c.strip()]
    elif isinstance(config_features, list):
        features = [str(c).strip() for c in config_features if str(c).strip()]

    paths = load_data_paths()
    dataset_dir = paths["dataset_tft"]
    models_dir = paths["models"]

    dataset_repo = ParquetTFTDatasetRepository(output_dir=dataset_dir)
    model_repo = LocalTFTModelRepository(base_dir=models_dir)
    trainer = PytorchForecastingTFTTrainer()

    use_case = TrainTFTModelUseCase(
        dataset_repository=dataset_repo,
        model_trainer=trainer,
        model_repository=model_repo,
    )

    training_config = dict(TFT_TRAINING_DEFAULTS)
    json_training = file_config.get("training_config")
    if isinstance(json_training, dict):
        for key, value in json_training.items():
            if key in TFT_TRAINING_DEFAULTS:
                training_config[key] = value
    for key in TFT_TRAINING_DEFAULTS:
        if key in file_config:
            training_config[key] = file_config[key]

    overrides = {
        "max_encoder_length": args.max_encoder_length,
        "max_prediction_length": args.max_prediction_length,
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
    }
    for key, value in overrides.items():
        if value is not None:
            training_config[key] = value
    validate_tft_training_config(training_config)

    split_config = dict(TFT_SPLIT_DEFAULTS)
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

    result = use_case.execute(
        asset_id,
        features=features,
        training_config=training_config,
        split_config=split_config,
        run_ablation=run_ablation,
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
