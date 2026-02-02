from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

from src.adapters.alpha_vantage_fundamental_fetcher import (
    AlphaVantageFundamentalFetcher,
)
from src.adapters.parquet_fundamental_repository import ParquetFundamentalRepository
from src.domain.time.utc import parse_iso_utc
from src.use_cases.fetch_fundamentals_use_case import FetchFundamentalsUseCase
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
        description="Fetch fundamentals (Alpha Vantage) and persist to parquet"
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

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not found in environment.")

    # ---------- Paths ----------
    paths = load_data_paths()
    processed_fundamentals_dir = paths.get("processed_fundamentals")
    if processed_fundamentals_dir is None:
        processed_fundamentals_dir = Path("data") / "processed" / "fundamentals"
    processed_fundamentals_dir = Path(processed_fundamentals_dir)

    logger.info(
        "Starting fundamentals pipeline",
        extra={
            "asset": asset_id,
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
            "output_dir": str(processed_fundamentals_dir.resolve()),
        },
    )

    fetcher = AlphaVantageFundamentalFetcher(api_key=api_key)
    repository = ParquetFundamentalRepository(output_dir=processed_fundamentals_dir)
    use_case = FetchFundamentalsUseCase(
        fundamental_fetcher=fetcher,
        fundamental_repository=repository,
    )

    result = use_case.execute(
        asset_id=asset_id,
        start_date=start_date,
        end_date=end_date,
    )

    logger.info(
        "Fundamentals pipeline completed",
        extra={
            "asset": result.asset_id,
            "fetched": result.fetched,
            "saved": result.saved,
            "report_types": result.report_types,
        },
    )


if __name__ == "__main__":
    main()
