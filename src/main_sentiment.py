from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.adapters.finbert_sentiment_model import FinBERTSentimentModel
from src.adapters.parquet_news_repository import ParquetNewsRepository
from src.adapters.parquet_scored_news_repository import ParquetScoredNewsRepository
from src.domain.time.utc import parse_iso_utc
from src.use_cases.infer_sentiment_use_case import InferSentimentUseCase
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
        description="Infer sentiment for raw news using FinBERT and persist to processed parquet"
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
    raw_news_dir = paths.get("news_dataset")
    if raw_news_dir is None:
        raw_news_dir = Path("data") / "raw" / "news"
    raw_news_dir = Path(raw_news_dir)

    processed_news_dir = paths.get("processed_news_scored")
    if processed_news_dir is None:
        processed_news_dir = Path("data") / "processed" / "scored_news"
    processed_news_dir = Path(processed_news_dir)

    logger.info(
        "Starting sentiment inference",
        extra={
            "asset": asset_id,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "raw_news_dir": str(raw_news_dir.resolve()),
            "processed_news_dir": str(processed_news_dir.resolve()),
        },
    )

    news_repository = ParquetNewsRepository(output_dir=raw_news_dir)
    scored_repository = ParquetScoredNewsRepository(output_dir=processed_news_dir)
    sentiment_model = FinBERTSentimentModel()

    infer_use_case = InferSentimentUseCase(
        news_repository=news_repository,
        sentiment_model=sentiment_model,
        scored_news_repository=scored_repository,
    )

    infer_result = infer_use_case.execute(
        asset_id=asset_id,
        start_date=start_date,
        end_date=end_date,
    )

    scored_path = processed_news_dir / asset_id / f"scored_news_{asset_id}.parquet"
    if scored_path.exists():
        profile = get_profile("scored_news")
        skipped = infer_result.scored == 0 and infer_result.read > 0
        reports_dir = scored_path.parent / "reports"
        if not skipped or not DataQualityReporter.report_exists(reports_dir, profile.prefix):
            DataQualityReporter.report_from_parquet(scored_path, **profile.to_kwargs())


    if infer_result.scored == 0 and infer_result.read > 0:
        logger.info(
            "Sentiment inference skipped (period already scored)",
            extra={
                "asset": infer_result.asset_id,
                "read": infer_result.read,
                "skipped": infer_result.skipped,
                "start": infer_result.start.isoformat(),
                "end": infer_result.end.isoformat(),
            },
        )
    else:
        logger.info(
            "Sentiment inference completed",
            extra={
                "asset": infer_result.asset_id,
                "read": infer_result.read,
                "skipped": infer_result.skipped,
                "scored": infer_result.scored,
                "saved": infer_result.saved,
            },
        )


if __name__ == "__main__":
    main()
