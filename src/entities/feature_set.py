# src/entities/feature_set.py
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping


@dataclass(frozen=True)
class FeatureSet:
    """
    Representa um snapshot de features calculadas para um ativo em um timestamp.
    Entidade de domínio: NÃO conhece pandas, numpy ou ML.
    """

    asset_id: str
    timestamp: datetime
    values: Mapping[str, float]

    def __post_init__(self):
        # TODO(validation): validar tipos e valores finitos
        if not self.features:
            raise ValueError("FeatureSet cannot be empty")