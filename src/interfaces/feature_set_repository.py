# src/interfaces/feature_set_repository.py
from abc import ABC, abstractmethod

from src.entities.feature_set import FeatureSet


class FeatureSetRepository(ABC):
    """
    Interface para persistência de FeatureSets.
    Camada de domínio: NÃO conhece pandas, parquet, sklearn, etc.
    """

    @abstractmethod
    def save(self, asset_id: str, features: list[FeatureSet]) -> None:
        ...

    @abstractmethod
    def load(self, asset_id: str) -> list[FeatureSet]:
        ...
