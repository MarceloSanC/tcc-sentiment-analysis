# src/main_candles.py
import argparse
import logging

from datetime import datetime
from pathlib import Path

import yaml

from src.adapters.parquet_data_repository import ParquetCandleRepository
from src.adapters.yfinance_data_fetcher import YFinanceDataFetcher
from src.use_cases.fetch_candles_use_case import FetchCandlesUseCase


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config():
    config_path = Path(__file__).parent.parent / "config" / "data_sources.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", required=True, help="Ex: PETR4.SA")
    args = parser.parse_args()

    config = load_config()
    asset_config = next(
        (a for a in config["assets"] if a["symbol"] == args.asset), None
    )
    if not asset_config:
        raise ValueError(
            f"Ativo {args.asset} não encontrado em config/data_sources.yaml"
        )

    # Setup
    fetcher = YFinanceDataFetcher()
    repo = ParquetCandleRepository(
        output_dir=config["data_sources"]["candles"]["output_dir"]
    )
    use_case = FetchCandlesUseCase(fetcher, repo)

    # Executar
    start = datetime.fromisoformat(asset_config["start_date"])
    end = datetime.fromisoformat(asset_config["end_date"])
    count = use_case.execute(args.asset, start, end)
    logger.info("Pipeline finalizado", extra={"symbol": args.asset, "candles": count})


if __name__ == "__main__":
    main()
