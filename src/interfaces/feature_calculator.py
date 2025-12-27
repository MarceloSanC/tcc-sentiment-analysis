# src/interfaces/feature_calculator.py
from abc import ABC, abstractmethod
from typing import Iterable

from src.entities.candle import Candle
from src.entities.feature_set import FeatureSet


class FeatureCalculator(ABC):
    @abstractmethod
    def calculate(
        self,
        asset_id: str,
        candles: Iterable[Candle],
    ) -> list[FeatureSet]:
        """
        Calcula features a partir de candles e retorna entidades FeatureSet.
        """
        pass
