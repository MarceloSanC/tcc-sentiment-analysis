from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import yaml
import pandas as pd
from dotenv import load_dotenv

from src.adapters.alpha_vantage_fundamental_fetcher import (
    AlphaVantageFundamentalFetcher,
)
from src.adapters.parquet_fundamental_repository import ParquetFundamentalRepository
from src.domain.time.utc import parse_iso_utc
from src.use_cases.fetch_fundamentals_use_case import FetchFundamentalsUseCase
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
        description="Fetch fundamentals (Alpha Vantage) and persist to parquet"
    )
    parser.add_argument("--asset", required=True, help="Asset symbol (e.g. AAPL)")
    return parser.parse_args()


def _is_period_covered(
    fundamentals_path: Path, start: datetime, end: datetime
) -> tuple[bool, datetime | None, datetime | None]:
    if not fundamentals_path.exists():
        return False, None, None
    try:
        df_existing = pd.read_parquet(fundamentals_path, columns=["fiscal_date_end"])
    except Exception:
        return False, None, None
    if df_existing.empty:
        return False, None, None
    ts = pd.to_datetime(df_existing["fiscal_date_end"], utc=True, errors="coerce")
    if ts.isna().all():
        return False, None, None
    ts_min = ts.min().date()
    ts_max = ts.max().date()
    covered = ts_min <= start.date() and ts_max >= end.date()
    return covered, ts.min(), ts.max()


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

    fundamentals_path = processed_fundamentals_dir / asset_id / f"fundamentals_{asset_id}.parquet"
    existing_rows = 0
    if fundamentals_path.exists():
        try:
            existing_rows = len(pd.read_parquet(fundamentals_path))
        except Exception:
            existing_rows = 0

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

    try:
        result = use_case.execute(
            asset_id=asset_id,
            start_date=start_date,
            end_date=end_date,
        )
    except RuntimeError as exc:
        msg = str(exc)
        if "Alpha Vantage Information" in msg or "rate limit" in msg.lower():
            covered, min_ts, max_ts = _is_period_covered(
                fundamentals_path, start_date, end_date
            )
            logger.warning(
                "Alpha Vantage rate limit encountered",
                extra={
                    "asset": asset_id,
                    "covered": covered,
                    "min_fiscal_date_end": min_ts.isoformat() if min_ts else None,
                    "max_fiscal_date_end": max_ts.isoformat() if max_ts else None,
                    "requested_start": start_date.isoformat(),
                    "requested_end": end_date.isoformat(),
                },
            )
            if covered:
                logger.warning(
                    "Alpha Vantage rate limit hit, but period already covered; skipping",
                    extra={
                        "asset": asset_id,
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat(),
                    },
                )
                if fundamentals_path.exists():
                    profile = get_profile("fundamentals")
                    reports_dir = fundamentals_path.parent / "reports"
                    if not DataQualityReporter.report_exists(reports_dir, profile.prefix):
                        DataQualityReporter.report_from_parquet(
                            fundamentals_path, **profile.to_kwargs()
                        )
                return

        raise

    total_rows = 0
    if fundamentals_path.exists():
        try:
            total_rows = len(pd.read_parquet(fundamentals_path))
        except Exception:
            total_rows = 0
    new_rows = max(total_rows - existing_rows, 0)

    if fundamentals_path.exists():
        profile = get_profile("fundamentals")
        skipped = result.fetched == 0 and result.saved == 0
        reports_dir = fundamentals_path.parent / "reports"
        if not skipped or not DataQualityReporter.report_exists(reports_dir, profile.prefix):
            DataQualityReporter.report_from_parquet(fundamentals_path, **profile.to_kwargs())

    logger.info(
        "Fundamentals pipeline completed",
        extra={
            "asset": result.asset_id,
            "fetched": result.fetched,
            "saved": result.saved,
            "report_types": result.report_types,
            "existing_rows": existing_rows,
            "total_rows": total_rows,
            "new_rows": new_rows,
        },
    )


if __name__ == "__main__":
    main()
