from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.adapters.parquet_daily_sentiment_repository import (
    ParquetDailySentimentRepository,
)
from src.adapters.parquet_scored_news_repository import ParquetScoredNewsRepository
from src.domain.services.sentiment_aggregator import SentimentAggregator
from src.domain.time.utc import parse_iso_utc
from src.use_cases.sentiment_feature_engineering_use_case import (
    SentimentFeatureEngineeringUseCase,
)
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
        description="Generate daily sentiment features from scored news"
    )
    parser.add_argument("--asset", required=True, help="Asset symbol (e.g. AAPL)")
    return parser.parse_args()


def main() -> None:
    setup_logging(logging.INFO)

    args = parse_args()
    asset_id = args.asset.strip().upper()

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
    processed_news_dir = paths.get("processed_news_scored")
    if processed_news_dir is None:
        processed_news_dir = Path("data") / "processed" / "scored_news"
    processed_news_dir = Path(processed_news_dir)

    processed_sentiment_daily_dir = paths.get("processed_sentiment_daily")
    if processed_sentiment_daily_dir is None:
        processed_sentiment_daily_dir = Path("data") / "processed" / "sentiment_daily"
    processed_sentiment_daily_dir = Path(processed_sentiment_daily_dir)

    logger.info(
        "Starting sentiment features pipeline (daily aggregation)",
        extra={
            "asset": asset_id,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "processed_news_dir": str(processed_news_dir.resolve()),
            "processed_sentiment_daily_dir": str(
                processed_sentiment_daily_dir.resolve()
            ),
        },
    )

    scored_repository = ParquetScoredNewsRepository(output_dir=processed_news_dir)
    daily_sentiment_repository = ParquetDailySentimentRepository(
        output_dir=processed_sentiment_daily_dir
    )
    sentiment_aggregator = SentimentAggregator()

    use_case = SentimentFeatureEngineeringUseCase(
        scored_news_repository=scored_repository,
        sentiment_aggregator=sentiment_aggregator,
        daily_sentiment_repository=daily_sentiment_repository,
    )

    result = use_case.execute(
        asset_id=asset_id,
        start_date=start_date,
        end_date=end_date,
    )


    if result.aggregated == 0 and result.read > 0:
        logger.info(
            "Sentiment features skipped (no new daily aggregation)",
            extra={
                "asset": result.asset_id,
                "read": result.read,
                "start": result.start.isoformat(),
                "end": result.end.isoformat(),
            },
        )
    else:
        logger.info(
            "Daily sentiment features completed",
            extra={
                "asset": result.asset_id,
                "read": result.read,
                "aggregated": result.aggregated,
                "saved": result.saved,
            },
        )


if __name__ == "__main__":
    main()
