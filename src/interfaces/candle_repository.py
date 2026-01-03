# src/interfaces/data_repository.py
from abc import ABC, abstractmethod

from src.entities.candle import Candle


class CandleRepository(ABC):
    @abstractmethod
    def save_candles(self, symbol: str, candles: list[Candle]) -> None:
        pass
