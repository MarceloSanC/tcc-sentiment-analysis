# src/interfaces/technical_indicator_repository.py
from abc import ABC, abstractmethod

from src.entities.technical_indicator_set import TechnicalIndicatorSet


class TechnicalIndicatorRepository(ABC):
    """
    Interface para persistência de TechnicalIndicatorSet.
    Camada de domínio: NÃO conhece pandas, parquet, sklearn, etc.
    """

    @abstractmethod
    def save(self, asset_id: str, indicators: list[TechnicalIndicatorSet]) -> None:
        ...

    @abstractmethod
    def load(self, asset_id: str) -> list[TechnicalIndicatorSet]:
        ...
