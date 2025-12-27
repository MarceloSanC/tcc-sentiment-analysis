# src/interfaces/feature_normalizer.py
from abc import ABC, abstractmethod
from typing import Iterable

from src.entities.feature_set import FeatureSet


class FeatureNormalizer(ABC):
    @abstractmethod
    def fit(self, features: Iterable[FeatureSet]) -> None:
        """Ajusta estatísticas de normalização usando dados de treino."""
        pass

    @abstractmethod
    def transform(self, features: Iterable[FeatureSet]) -> list[FeatureSet]:
        """Aplica normalização usando estatísticas já ajustadas."""
        pass
