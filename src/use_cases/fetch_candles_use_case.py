# src/use_cases/fetch_candles_use_case.py
from datetime import datetime
from typing import List
from src.entities.candle import Candle
from src.interfaces.data_fetcher import DataFetcher
from src.interfaces.data_repository import DataRepository

class FetchCandlesUseCase:
    def __init__(
        self,
        data_fetcher: DataFetcher,
        data_repository: DataRepository
    ):
        self.data_fetcher = data_fetcher
        self.data_repository = data_repository

    def execute(self, symbol: str, start: datetime, end: datetime) -> None:
        print(f"📥 Buscando candles de {symbol} ({start.date()} → {end.date()})...")
        candles = self.data_fetcher.fetch_candles(symbol, start, end)
        print(f"✅ {len(candles)} candles coletados")
        self.data_repository.save_candles(symbol, candles)