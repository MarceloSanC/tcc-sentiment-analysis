# src/main_candles.py
import argparse
import logging

from datetime import datetime
from pathlib import Path

import yaml

from src.utils.path_resolver import load_data_paths
from src.adapters.parquet_candle_repository import ParquetCandleRepository
from src.adapters.yfinance_candle_fetcher import YFinanceCandleFetcher
from src.use_cases.fetch_candles_use_case import FetchCandlesUseCase


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    config_path = Path(__file__).parent.parent / "config" / "data_sources.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, help="Ex: AAPL")
    args = parser.parse_args()

    asset_id = args.asset

    config = load_config()
    asset_config = next(
        (a for a in config["assets"] if a["symbol"] == asset_id), None
    )
    if not asset_config:
        raise ValueError(
            f"Ativo {asset_id} não encontrado em config/data_sources.yaml"
        )

    # ---------- Paths (ORQUESTRADOR decide estrutura) ----------
    paths = load_data_paths()

    # base: data/raw/market/candles
    candles_base_dir = paths["raw_candles"]

    # final: data/raw/market/candles/AAPL
    candles_asset_dir = candles_base_dir / asset_id
    candles_asset_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Adapters ----------
    fetcher = YFinanceCandleFetcher()

    repository = ParquetCandleRepository(
        output_dir=candles_asset_dir
    )

    use_case = FetchCandlesUseCase(fetcher, repository)

    # ---------- Execute ----------
    start = datetime.fromisoformat(asset_config["start_date"])
    end = datetime.fromisoformat(asset_config["end_date"])

    count = use_case.execute(asset_id, start, end)

    logger.info(
        "Pipeline de candles finalizado",
        extra={"asset": asset_id, "candles": count},
    )


if __name__ == "__main__":
    main()
