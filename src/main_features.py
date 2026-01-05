# src/main_features.py
import argparse
from pathlib import Path

from src.utils.path_resolver import load_data_paths

from src.adapters.parquet_candle_repository import ParquetCandleRepository
from src.adapters.parquet_feature_set_repository import ParquetFeatureSetRepository
from src.adapters.technical_indicator_calculator import TechnicalIndicatorCalculator
from src.adapters.sklearn_feature_normalizer import SklearnFeatureNormalizer
from src.use_cases.feature_engineering_use_case import FeatureEngineeringUseCase


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate technical features for a financial asset"
    )

    parser.add_argument(
        "--asset",
        type=str,
        required=True,
        help="Asset identifier (e.g. PETR4.SA, AAPL)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing feature file if it exists",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    asset_id = args.asset
    overwrite = args.overwrite

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

    # Features: data/processed/features/{ASSET}
    processed_features_base_dir = paths["processed_features"]
    processed_features_asset_dir = processed_features_base_dir / asset_id
    processed_features_asset_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Adapters ----------
    candle_repository = ParquetCandleRepository(
        output_dir=raw_candles_asset_dir
    )

    feature_repository = ParquetFeatureSetRepository(
        output_dir=processed_features_asset_dir,
        overwrite=overwrite,
    )

    feature_calculator = TechnicalIndicatorCalculator()
    feature_normalizer = SklearnFeatureNormalizer()

    # ---------- Use Case ----------
    use_case = FeatureEngineeringUseCase(
        candle_repository=candle_repository,
        feature_calculator=feature_calculator,
        feature_repository=feature_repository,
        feature_normalizer=feature_normalizer,
    )

    # ---------- Execute ----------
    feature_sets = use_case.execute(asset_id)

    print(
        f"[OK] Generated {len(feature_sets)} feature rows "
        f"for asset {asset_id}"
    )


if __name__ == "__main__":
    main()
