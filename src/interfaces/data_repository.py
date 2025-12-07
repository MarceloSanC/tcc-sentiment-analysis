# src/interfaces/data_repository.py
from abc import ABC, abstractmethod
from typing import List
from src.entities.candle import Candle

class DataRepository(ABC):
    @abstractmethod
    def save_candles(self, symbol: str, candles: List[Candle]) -> None:
        pass