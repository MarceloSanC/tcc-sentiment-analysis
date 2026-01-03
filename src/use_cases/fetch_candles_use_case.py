# src/use_cases/fetch_candles_use_case.py
from datetime import datetime

from interfaces.candle_fetcher import CandleFetcher
from interfaces.candle_repository import CandleRepository


class FetchCandlesUseCase:
    def __init__(self, candle_fetcher: CandleFetcher, candle_repository: CandleRepository):
        self.candle_fetcher = candle_fetcher
        self.candle_repository = candle_repository

    def execute(self, symbol: str, start: datetime, end: datetime) -> int:
        # NOTE:
        # This use case assumes a full re-fetch of candle data.
        # Persistence behavior (overwrite vs incremental) is delegated
        # to the CandleRepository implementation.
        candles = self.candle_fetcher.fetch_candles(symbol, start, end)
        self.candle_repository.save_candles(symbol, candles)
        return len(candles)
