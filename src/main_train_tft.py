from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.adapters.parquet_tft_dataset_repository import ParquetTFTDatasetRepository
from src.adapters.pytorch_forecasting_tft_trainer import PytorchForecastingTFTTrainer
from src.adapters.local_tft_model_repository import LocalTFTModelRepository
from src.use_cases.train_tft_model_use_case import TrainTFTModelUseCase
from src.infrastructure.schemas.model_artifact_schema import (
    TFT_TRAINING_DEFAULTS,
    TFT_SPLIT_DEFAULTS,
)
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import load_data_paths

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TFT model from dataset_tft")
    parser.add_argument("--asset", required=True, help="Asset symbol (e.g. AAPL)")
    parser.add_argument(
        "--features",
        type=str,
        required=False,
        help="Comma-separated list of feature columns to use",
    )
    parser.add_argument("--max-encoder-length", type=int)
    parser.add_argument("--max-prediction-length", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--attention-head-size", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--hidden-continuous-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-start", type=str, help="Train start date (yyyymmdd)")
    parser.add_argument("--train-end", type=str, help="Train end date (yyyymmdd)")
    parser.add_argument("--val-start", type=str, help="Validation start date (yyyymmdd)")
    parser.add_argument("--val-end", type=str, help="Validation end date (yyyymmdd)")
    parser.add_argument("--test-start", type=str, help="Test start date (yyyymmdd)")
    parser.add_argument("--test-end", type=str, help="Test end date (yyyymmdd)")
    return parser.parse_args()


def main() -> None:
    setup_logging(logging.INFO)
    args = parse_args()

    asset_id = args.asset.strip().upper()
    features = (
        [c.strip() for c in args.features.split(",") if c.strip()]
        if args.features
        else None
    )

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
    }
    for key, value in overrides.items():
        if value is not None:
            training_config[key] = value

    split_config = dict(TFT_SPLIT_DEFAULTS)
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

    result = use_case.execute(
        asset_id,
        features=features,
        training_config=training_config,
        split_config=split_config,
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
