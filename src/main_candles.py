# src/main_candles.py
import argparse
from datetime import datetime
import yaml
from pathlib import Path
from src.adapters.yfinance_data_fetcher import YFinanceDataFetcher
from src.adapters.parquet_data_repository import ParquetDataRepository
from src.use_cases.fetch_candles_use_case import FetchCandlesUseCase

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
        (a for a in config["assets"] if a["symbol"] == args.asset),
        None
    )
    if not asset_config:
        raise ValueError(f"Ativo {args.asset} não encontrado em config/data_sources.yaml")

    # Setup
    fetcher = YFinanceDataFetcher()
    repo = ParquetDataRepository(output_dir=config["data_sources"]["candles"]["output_dir"])
    use_case = FetchCandlesUseCase(fetcher, repo)

    # Executar
    start = datetime.fromisoformat(asset_config["start_date"])
    end = datetime.fromisoformat(asset_config["end_date"])
    use_case.execute(args.asset, start, end)

if __name__ == "__main__":
    main()