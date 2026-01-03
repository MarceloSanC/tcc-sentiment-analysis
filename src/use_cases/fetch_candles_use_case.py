# src/use_cases/fetch_candles_use_case.py
from datetime import datetime

from src.interfaces.data_fetcher import DataFetcher
from interfaces.candle_repository import CandleRepository


class FetchCandlesUseCase:
    def __init__(self, data_fetcher: DataFetcher, data_repository: CandleRepository):
        self.data_fetcher = data_fetcher
        self.data_repository = data_repository

    def execute(self, symbol: str, start: datetime, end: datetime) -> int:
        candles = self.data_fetcher.fetch_candles(symbol, start, end)
        self.data_repository.save_candles(symbol, candles)
        return len(candles)
