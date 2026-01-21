# src/main_sentiment.py
import argparse
import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.domain.time.utc import parse_iso_utc
from src.adapters.finnhub_news_fetcher import FinnhubNewsFetcher
from src.adapters.finbert_sentiment_model import FinBERTSentimentModel
from src.adapters.parquet_candle_repository import ParquetCandleRepository
from src.domain.services.sentiment_aggregator import SentimentAggregator
from src.use_cases.infer_sentiment_use_case import InferSentimentUseCase
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import load_data_paths

logger = logging.getLogger(__name__)

load_dotenv()


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "data_sources.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, help="Ex: AAPL")
    args = parser.parse_args()
    asset_id = args.asset

    config = load_config()

    sentiment_cfg = (config.get("data_sources") or {}).get("sentiment") or {}
    if not sentiment_cfg.get("enabled", False):
        logger.info("Sentiment pipeline disabled in config", extra={"asset": asset_id})
        return

    provider = sentiment_cfg.get("provider")
    if provider != "finnhub":
        raise ValueError(f"Unsupported sentiment provider: {provider}")

    asset_config = next((a for a in config.get("assets", []) if a["symbol"] == asset_id), None)
    if not asset_config:
        raise ValueError(f"Asset {asset_id} not found in config/data_sources.yaml")

    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("FINNHUB_API_KEY environment variable is not set")

    start_date = parse_iso_utc(asset_config["start_date"])
    end_date = parse_iso_utc(asset_config["end_date"])

    if start_date > end_date:
        raise ValueError("start_date must be <= end_date")

    logger.info(
        "Starting sentiment pipeline",
        extra={
            "asset": asset_id,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "provider": provider,
            "aggregation": sentiment_cfg.get("aggregation"),
        },
    )

    paths = load_data_paths()
    candles_base_dir = paths["raw_candles"]
    candles_base_dir.mkdir(parents=True, exist_ok=True)

    news_fetcher = FinnhubNewsFetcher(api_key=api_key)
    sentiment_model = FinBERTSentimentModel()
    sentiment_aggregator = SentimentAggregator()
    candle_repository = ParquetCandleRepository(output_dir=candles_base_dir)

    use_case = InferSentimentUseCase(
        news_fetcher=news_fetcher,
        sentiment_model=sentiment_model,
        sentiment_aggregator=sentiment_aggregator,
        candle_repository=candle_repository,
    )

    daily_sentiments = use_case.execute(
        asset_id=asset_id,
        start_date=start_date,
        end_date=end_date,
    )

    logger.info(
        "Sentiment enrichment completed",
        extra={"asset": asset_id, "days": len(daily_sentiments)},
    )


if __name__ == "__main__":
    main()
