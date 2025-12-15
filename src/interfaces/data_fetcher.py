# src/interfaces/data_fetcher.py
from abc import ABC, abstractmethod
from datetime import datetime

from src.entities.candle import Candle


class DataFetcher(ABC):
    @abstractmethod
    def fetch_candles(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[Candle]:
        pass
