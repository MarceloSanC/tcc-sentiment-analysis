import argparse
import logging
from pathlib import Path

from src.utils.path_resolver import load_data_paths

from src.adapters.parquet_candle_repository import ParquetCandleRepository
from src.adapters.parquet_technical_indicator_repository import (
    ParquetTechnicalIndicatorRepository,
)
from src.adapters.technical_indicator_calculator import TechnicalIndicatorCalculator
from src.adapters.sklearn_indicator_normalizer import SklearnTechnicalIndicatorNormalizer
from src.use_cases.technical_indicator_engineering_use_case import (
    TechnicalIndicatorEngineeringUseCase,
)


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
    indicator_sets = use_case.execute(asset_id)


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
