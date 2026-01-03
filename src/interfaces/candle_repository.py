# src/interfaces/data_repository.py
from abc import ABC, abstractmethod

from src.entities.candle import Candle


class CandleRepository(ABC):
    @abstractmethod
    def load_candles(self, asset_id: str) -> list[Candle]:
        pass

    @abstractmethod
    def save_candles(self, asset_id: str, candles: list[Candle]) -> None:
        pass
