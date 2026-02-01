# src/interfaces/technical_indicator_normalizer.py
from abc import ABC, abstractmethod
from typing import Iterable

from src.entities.technical_indicator_set import TechnicalIndicatorSet


class TechnicalIndicatorNormalizer(ABC):
    @abstractmethod
    def fit(self, indicators: Iterable[TechnicalIndicatorSet]) -> None:
        ...

    @abstractmethod
    def transform(
        self, indicators: Iterable[TechnicalIndicatorSet]
    ) -> list[TechnicalIndicatorSet]:
        ...
