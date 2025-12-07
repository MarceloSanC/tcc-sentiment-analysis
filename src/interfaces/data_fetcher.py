# src/interfaces/data_fetcher.py
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List
from src.entities.candle import Candle

class DataFetcher(ABC):
    @abstractmethod
    def fetch_candles(
        self, symbol: str, start: datetime, end: datetime
    ) -> List[Candle]:
        pass