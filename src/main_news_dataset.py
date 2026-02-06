# src/main_news_dataset.py

from __future__ import annotations

import argparse
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
from pathlib import Path

import yaml
import pandas as pd
import pandas as pd

from src.domain.time.utc import parse_iso_utc
from src.adapters.alpha_vantage_news_fetcher import AlphaVantageNewsFetcher
from src.adapters.parquet_news_repository import ParquetNewsRepository
from src.use_cases.fetch_news_use_case import FetchNewsUseCase
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
        description="Build historical news dataset (offline) using Alpha Vantage -> Parquet"
    )
    parser.add_argument(
        "--asset",
        type=str,
        required=False,
        help="Asset symbol (e.g. AAPL). If omitted, runs all assets from config.",
    )
    return parser.parse_args()


def _get_news_dataset_config(cfg: dict) -> dict:
    """
    Expected config shape (data_sources.yaml):

    data_sources:
      news:
        build_enabled: true
        provider: "alpha_vantage"
        dataset_start: "2010-01-01T00:00:00Z"
        dataset_end: "2025-12-31T23:59:00Z"
        safety_margin: 950
    """
    news_cfg = cfg.get("data_sources", {}).get("news_dataset", {})
    if not isinstance(news_cfg, dict):
        return {}

    return news_cfg


def _is_period_covered(
    news_path: Path, start: datetime, end: datetime
) -> tuple[bool, datetime | None, datetime | None]:
    if not news_path.exists():
        return False, None, None
    try:
        df_existing = pd.read_parquet(news_path, columns=["published_at"])
    except Exception:
        return False, None, None
    if df_existing.empty:
        return False, None, None
    ts = pd.to_datetime(df_existing["published_at"], utc=True, errors="coerce")
    if ts.isna().all():
        return False, None, None
    ts_min = ts.min().date()
    ts_max = ts.max().date()

    # Tolerance: same-day coverage (ignore time component)
    covered = ts_max >= end.date()
    return covered, ts.min(), ts.max()


def main() -> None:
    setup_logging(logging.INFO)

    args = parse_args()
    config = load_config()

    news_cfg = _get_news_dataset_config(config)

    if not news_cfg.get("build_enabled", False):
        raise RuntimeError(
            "News dataset is disabled. Set data_sources.news_dataset.build_enabled=true "
            "in config/data_sources.yaml"
        )

    provider = news_cfg.get("provider")
    if provider != "alpha_vantage":
        raise RuntimeError(f"Unsupported news provider: {provider!r}")

    dataset_start = parse_iso_utc(news_cfg["dataset_start"])
    dataset_end = parse_iso_utc(news_cfg["dataset_end"])
    safety_margin = int(news_cfg.get("safety_margin", 950))

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not found in environment.")

    # ---------- Paths ----------
    # Esperado em config/data_paths.yaml:
    # data:
    #   raw:
    #     news: data/raw/news
    paths = load_data_paths()
    raw_news_dir = paths.get("raw_news")
    if raw_news_dir is None:
        # fallback para manter executável, mas o ideal é existir no YAML
        raw_news_dir = Path("data") / "raw" / "news"
    raw_news_dir = Path(raw_news_dir)
    raw_news_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Adapters ----------
    fetcher = AlphaVantageNewsFetcher(api_key=api_key)
    repository = ParquetNewsRepository(output_dir=raw_news_dir)

    # ---------- Use Case ----------
    use_case = FetchNewsUseCase(
        news_fetcher=fetcher,
        news_repository=repository,
        safety_margin=safety_margin,
    )

    # ---------- Assets selection ----------
    assets = config.get("assets", [])
    if not isinstance(assets, list) or not assets:
        raise RuntimeError("No assets found in config/data_sources.yaml")

    if args.asset:
        wanted = args.asset.strip().upper()
        assets = [a for a in assets if str(a.get("symbol", "")).upper() == wanted]
        if not assets:
            raise RuntimeError(f"Asset not found in config: {wanted}")

    logger.info(
        "Starting news dataset pipeline",
        extra={
            "assets": [a.get("symbol") for a in assets],
            "dataset_start": dataset_start.isoformat(),
            "dataset_end": dataset_end.isoformat(),
            "raw_news_dir": str(raw_news_dir.resolve()),
            "provider": provider,
            "safety_margin": safety_margin,
        },
    )

    for asset in assets:
        symbol = str(asset["symbol"]).strip().upper()
        news_path = raw_news_dir / symbol / f"news_{symbol}.parquet"
        existing_rows = 0
        if news_path.exists():
            try:
                existing_rows = len(pd.read_parquet(news_path))
            except Exception:
                existing_rows = 0

        try:
            result = use_case.execute(
                asset_id=symbol,
                start_date=dataset_start,
                end_date=dataset_end,
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "Alpha Vantage Information" in msg or "rate limit" in msg.lower():
                covered, min_ts, max_ts = _is_period_covered(
                    news_path, dataset_start, dataset_end
                )
                logger.warning(
                    "Alpha Vantage rate limit encountered",
                    extra={
                        "asset": symbol,
                        "covered": covered,
                        "min_published_at": min_ts.isoformat() if min_ts else None,
                        "max_published_at": max_ts.isoformat() if max_ts else None,
                        "requested_start": dataset_start.isoformat(),
                        "requested_end": dataset_end.isoformat(),
                    },
                )
                if covered:
                    logger.warning(
                        "Alpha Vantage rate limit hit, but period already covered; skipping",
                        extra={
                            "asset": symbol,
                            "start": dataset_start.isoformat(),
                            "end": dataset_end.isoformat(),
                        },
                    )
                    if news_path.exists():
                        profile = get_profile("news_raw")
                        reports_dir = news_path.parent / "reports"
                        if not DataQualityReporter.report_exists(reports_dir, profile.prefix):
                            DataQualityReporter.report_from_parquet(
                                news_path, **profile.to_kwargs()
                            )
                    continue

            raise

        total_rows = 0
        if news_path.exists():
            try:
                total_rows = len(pd.read_parquet(news_path))
            except Exception:
                total_rows = 0
        new_rows = max(total_rows - existing_rows, 0)

        if news_path.exists():
            profile = get_profile("news_raw")
            skipped = result.fetched == 0 and result.saved == 0
            reports_dir = news_path.parent / "reports"
            if not skipped or not DataQualityReporter.report_exists(reports_dir, profile.prefix):
                DataQualityReporter.report_from_parquet(news_path, **profile.to_kwargs())

        if result.fetched == 0 and result.saved == 0:
            logger.info(
                "News dataset skipped (already up to date)",
                extra={
                    "asset": symbol,
                    "start": result.start.isoformat(),
                    "end": result.end.isoformat(),
                    "last_cursor": result.last_cursor.isoformat(),
                },
            )
        else:
            logger.info(
                "News dataset completed",
                extra={
                    "asset": symbol,
                    "fetched": result.fetched,
                    "saved": result.saved,
                    "iterations": result.iterations,
                    "last_cursor": result.last_cursor.isoformat(),
                    "existing_rows": existing_rows,
                    "total_rows": total_rows,
                    "new_rows": new_rows,
                },
            )


if __name__ == "__main__":
    main()
