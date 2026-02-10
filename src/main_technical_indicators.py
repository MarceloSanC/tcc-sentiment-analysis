import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

from src.utils.path_resolver import load_data_paths
from src.utils.logging_config import setup_logging

from src.adapters.parquet_candle_repository import ParquetCandleRepository
from src.adapters.parquet_technical_indicator_repository import (
    ParquetTechnicalIndicatorRepository,
)
from src.adapters.technical_indicator_calculator import TechnicalIndicatorCalculator
from src.adapters.sklearn_indicator_normalizer import SklearnTechnicalIndicatorNormalizer
from src.use_cases.technical_indicator_engineering_use_case import (
    TechnicalIndicatorEngineeringUseCase,
)
from src.domain.services.data_quality_reporter import DataQualityReporter
from src.domain.services.data_quality_profiles import get_profile


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate technical indicators for a financial asset"
    )

    parser.add_argument(
        "--asset",
        type=str,
        required=True,
        help="Asset identifier (e.g. AAPL)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing indicator file if it exists",
    )

    return parser.parse_args()


def main() -> None:
    setup_logging(logging.INFO)
    args = parse_args()

    asset_id = args.asset
    overwrite = args.overwrite

    logger.info(
        "Starting technical indicators pipeline",
        extra={"asset": asset_id, "overwrite": overwrite},
    )

    # ---------- Paths ----------
    paths = load_data_paths()

    # Candles: data/raw/market/candles/{ASSET}
    raw_candles_base_dir = paths["raw_candles"]
    raw_candles_asset_dir = raw_candles_base_dir / asset_id

    if not raw_candles_asset_dir.exists():
        raise FileNotFoundError(
            f"No candle directory found for asset {asset_id}\n"
            f"Expected path: {raw_candles_asset_dir.resolve()}\n"
            "Run main_candles.py first."
        )

    # Indicators: data/processed/technical_indicators/{ASSET}
    processed_indicators_base_dir = paths["processed_technical_indicators"]
    processed_indicators_asset_dir = processed_indicators_base_dir / asset_id
    processed_indicators_asset_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Resolved data paths",
        extra={
            "candles_dir": str(raw_candles_asset_dir.resolve()),
            "indicators_dir": str(processed_indicators_asset_dir.resolve()),
        },
    )

    # ---------- Coverage check ----------
    indicators_path = processed_indicators_asset_dir / f"technical_indicators_{asset_id}.parquet"
    if indicators_path.exists() and not overwrite:
        try:
            df_existing = pd.read_parquet(indicators_path, columns=["timestamp"])
            ts = pd.to_datetime(df_existing["timestamp"], utc=True, errors="coerce")
            existing_start = ts.min()
            existing_end = ts.max()

            config_path = Path(__file__).parent.parent / "config" / "data_sources.yaml"
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            asset_cfg = next(
                (a for a in config.get("assets", []) if str(a.get("symbol", "")).upper() == asset_id),
                None,
            )
            if asset_cfg:
                requested_start = pd.to_datetime(asset_cfg["start_date"], utc=True, errors="coerce")
                requested_end = pd.to_datetime(asset_cfg["end_date"], utc=True, errors="coerce")
                if existing_start <= requested_start and existing_end >= requested_end:
                    logger.info(
                        "Technical indicators skipped (period already covered). Use --overwrite to rebuild.",
                        extra={
                            "asset": asset_id,
                            "existing_start": existing_start.isoformat(),
                            "existing_end": existing_end.isoformat(),
                            "requested_start": requested_start.isoformat(),
                            "requested_end": requested_end.isoformat(),
                        },
                    )
                    profile = get_profile("technical_indicators")
                    reports_dir = indicators_path.parent / "reports"
                    if not DataQualityReporter.report_exists(reports_dir, profile.prefix):
                        DataQualityReporter.report_from_parquet(
                            indicators_path, **profile.to_kwargs()
                        )
                    return
        except Exception:
            pass

    # ---------- Adapters ----------
    candle_repository = ParquetCandleRepository(
        output_dir=raw_candles_base_dir
    )

    indicator_repository = ParquetTechnicalIndicatorRepository(
        output_dir=processed_indicators_asset_dir,
        overwrite=overwrite,
    )

    indicator_calculator = TechnicalIndicatorCalculator()
    indicator_normalizer = SklearnTechnicalIndicatorNormalizer()

    # ---------- Use Case ----------
    use_case = TechnicalIndicatorEngineeringUseCase(
        candle_repository=candle_repository,
        indicator_calculator=indicator_calculator,
        indicator_repository=indicator_repository,
        indicator_normalizer=indicator_normalizer,
    )

    # ---------- Execute ----------
    try:
        indicator_sets = use_case.execute(asset_id)
    except FileExistsError:
        logger.info(
            "Technical indicators skipped (already exists). Use --overwrite to replace.",
            extra={
                "asset": asset_id,
                "output_dir": str(processed_indicators_asset_dir.resolve()),
            },
        )
        return

    indicators_path = processed_indicators_asset_dir / f"technical_indicators_{asset_id}.parquet"
    if indicators_path.exists():
        profile = get_profile("technical_indicators")
        skipped = len(indicator_sets) == 0
        reports_dir = indicators_path.parent / "reports"
        if not skipped or not DataQualityReporter.report_exists(reports_dir, profile.prefix):
            DataQualityReporter.report_from_parquet(indicators_path, **profile.to_kwargs())


    if len(indicator_sets) == 0:
        logger.info(
            "Technical indicators skipped (no data to process)",
            extra={
                "asset": asset_id,
                "output_dir": str(processed_indicators_asset_dir.resolve()),
            },
        )
    else:
        logger.info(
            "Technical indicators completed successfully",
            extra={
                "asset": asset_id,
                "rows": len(indicator_sets),
                "output_dir": str(processed_indicators_asset_dir.resolve()),
            },
        )


if __name__ == "__main__":
    main()
