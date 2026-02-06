from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from src.domain.services.data_quality_profiles import get_profile
from src.domain.services.data_quality_reporter import DataQualityReporter
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import load_data_paths

logger = logging.getLogger(__name__)


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "data_sources.yaml"
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit datasets and generate data quality reports"
    )
    parser.add_argument("--asset", required=False, help="Asset symbol (e.g. AAPL)")
    return parser.parse_args()


def _audit_if_exists(path: Path, profile_name: str) -> None:
    if not path.exists():
        logger.warning(
            "Audit skipped (file not found)",
            extra={"path": str(path), "profile": profile_name},
        )
        return
    profile = get_profile(profile_name)
    DataQualityReporter.report_from_parquet(path, **profile.to_kwargs())


def main() -> None:
    setup_logging(logging.INFO)

    args = parse_args()
    config = load_config()

    assets = config.get("assets", [])
    if not isinstance(assets, list) or not assets:
        raise RuntimeError("No assets found in config/data_sources.yaml")

    if args.asset:
        wanted = args.asset.strip().upper()
        assets = [a for a in assets if str(a.get("symbol", "")).upper() == wanted]
        if not assets:
            raise RuntimeError(f"Asset not found in config: {wanted}")

    paths = load_data_paths()

    for asset in assets:
        symbol = str(asset["symbol"]).strip().upper()

        candles_path = paths["raw_candles"] / symbol / f"candles_{symbol}_1d.parquet"
        news_path = paths["news_dataset"] / symbol / f"news_{symbol}.parquet"
        scored_path = (
            paths["processed_news_scored"] / symbol / f"scored_news_{symbol}.parquet"
        )
        sentiment_path = (
            paths["processed_sentiment_daily"] / symbol / f"daily_sentiment_{symbol}.parquet"
        )
        indicators_path = (
            paths["processed_technical_indicators"] / symbol / f"technical_indicators_{symbol}.parquet"
        )
        fundamentals_path = (
            paths["processed_fundamentals"] / symbol / f"fundamentals_{symbol}.parquet"
        )
        dataset_path = (
            paths["dataset_tft"] / symbol / f"dataset_tft_{symbol}.parquet"
        )

        _audit_if_exists(candles_path, "candles")
        _audit_if_exists(news_path, "news_raw")
        _audit_if_exists(scored_path, "scored_news")
        _audit_if_exists(sentiment_path, "sentiment_daily")
        _audit_if_exists(indicators_path, "technical_indicators")
        _audit_if_exists(fundamentals_path, "fundamentals")
        _audit_if_exists(dataset_path, "dataset_tft")

    logger.info("Audit completed", extra={"assets": [a.get("symbol") for a in assets]})


if __name__ == "__main__":
    main()
