from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.adapters.parquet_candle_repository import ParquetCandleRepository
from src.adapters.parquet_daily_sentiment_repository import (
    ParquetDailySentimentRepository,
)
from src.adapters.parquet_fundamental_repository import ParquetFundamentalRepository
from src.adapters.parquet_technical_indicator_repository import (
    ParquetTechnicalIndicatorRepository,
)
from src.adapters.parquet_tft_dataset_repository import ParquetTFTDatasetRepository
from src.domain.time.utc import parse_iso_utc
from src.use_cases.build_tft_dataset_use_case import BuildTFTDatasetUseCase
from src.domain.services.data_quality_reporter import DataQualityReporter
from src.domain.services.data_quality_profiles import get_profile
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import load_data_paths

logger = logging.getLogger(__name__)
load_dotenv()


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "data_sources.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a daily TFT training dataset from processed pipelines"
    )
    parser.add_argument("--asset", required=True, help="Asset symbol (e.g. AAPL)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing TFT dataset if it exists",
    )
    return parser.parse_args()


def main() -> None:
    setup_logging(logging.INFO)

    args = parse_args()
    asset_id = args.asset.strip().upper()
    overwrite = args.overwrite

    config = load_config()
    asset_cfg = next(
        (a for a in config.get("assets", []) if str(a.get("symbol", "")).upper() == asset_id),
        None,
    )
    if not asset_cfg:
        raise RuntimeError(f"Asset not found in config/data_sources.yaml: {asset_id}")

    start_date = parse_iso_utc(asset_cfg["start_date"])
    end_date = parse_iso_utc(asset_cfg["end_date"])
    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    paths = load_data_paths()

    raw_candles_base_dir = paths["raw_candles"]
    raw_candles_asset_dir = raw_candles_base_dir / asset_id
    if not raw_candles_asset_dir.exists():
        raise FileNotFoundError(
            f"No candle directory found for asset {asset_id}\n"
            f"Expected path: {raw_candles_asset_dir.resolve()}\n"
            "Run main_candles.py first."
        )

    indicators_asset_dir = paths["processed_technical_indicators"] / asset_id
    if not indicators_asset_dir.exists():
        raise FileNotFoundError(
            f"No technical indicators found for asset {asset_id}\n"
            f"Expected path: {indicators_asset_dir.resolve()}\n"
            "Run main_technical_indicators.py first."
        )

    processed_sentiment_daily_dir = paths["processed_sentiment_daily"]
    processed_fundamentals_dir = paths["processed_fundamentals"]
    dataset_tft_dir = paths["dataset_tft"]

    logger.info(
        "Starting TFT dataset build",
        extra={
            "asset": asset_id,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "candles_dir": str(raw_candles_asset_dir.resolve()),
            "indicators_dir": str(indicators_asset_dir.resolve()),
            "sentiment_daily_dir": str(processed_sentiment_daily_dir.resolve()),
            "fundamentals_dir": str(processed_fundamentals_dir.resolve()),
            "dataset_dir": str(dataset_tft_dir.resolve()),
            "overwrite": overwrite,
        },
    )

    candle_repository = ParquetCandleRepository(output_dir=raw_candles_base_dir)
    indicator_repository = ParquetTechnicalIndicatorRepository(
        output_dir=indicators_asset_dir
    )
    daily_sentiment_repository = ParquetDailySentimentRepository(
        output_dir=processed_sentiment_daily_dir
    )
    fundamental_repository = ParquetFundamentalRepository(
        output_dir=processed_fundamentals_dir
    )
    tft_dataset_repository = ParquetTFTDatasetRepository(output_dir=dataset_tft_dir)

    dataset_path = dataset_tft_dir / asset_id / f"dataset_tft_{asset_id}.parquet"
    if dataset_path.exists() and not overwrite:
        logger.info(
            "TFT dataset skipped (already exists). Use --overwrite to rebuild.",
            extra={"asset": asset_id, "path": str(dataset_path.resolve())},
        )
        profile = get_profile("dataset_tft")
        if not DataQualityReporter.report_exists(dataset_path.parent / "reports", profile.prefix):
            DataQualityReporter.report_from_parquet(dataset_path, **profile.to_kwargs())
        return

    use_case = BuildTFTDatasetUseCase(
        candle_repository=candle_repository,
        indicator_repository=indicator_repository,
        daily_sentiment_repository=daily_sentiment_repository,
        fundamental_repository=fundamental_repository,
        tft_dataset_repository=tft_dataset_repository,
    )

    result = use_case.execute(
        asset_id=asset_id,
        start_date=start_date,
        end_date=end_date,
    )
    if dataset_path.exists():
        profile = get_profile("dataset_tft")
        DataQualityReporter.report_from_parquet(dataset_path, **profile.to_kwargs())

    logger.info(
        "TFT dataset completed",
        extra={
            "asset": result.asset_id,
            "rows": result.rows,
            "start": result.start.isoformat(),
            "end": result.end.isoformat(),
            "nulls": result.nulls,
        },
    )


if __name__ == "__main__":
    main()
