# src/interfaces/technical_indicator_calculator.py
from abc import ABC, abstractmethod
from typing import Iterable

from src.entities.candle import Candle
from src.entities.technical_indicator_set import TechnicalIndicatorSet


class TechnicalIndicatorCalculatorPort(ABC):
    @abstractmethod
    def calculate(
        self,
        asset_id: str,
        candles: Iterable[Candle],
    ) -> list[TechnicalIndicatorSet]:
        """
        Calcula indicadores técnicos a partir de candles.
        """
        ...
