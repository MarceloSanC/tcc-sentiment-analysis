# src/adapters/sklearn_feature_normalizer.py
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.entities.feature_set import FeatureSet
from src.interfaces.feature_normalizer import FeatureNormalizer


class SklearnFeatureNormalizer(FeatureNormalizer):
    """
    Adapter responsável por normalizar features numéricas usando sklearn.
    Implementação atual: StandardScaler (z-score), ajustado por feature.
    """

    def __init__(self):
        # TODO(leakage): persistir scalers por asset_id
        # TODO(reprodutibilidade): permitir load/save dos scalers
        self.scalers: dict[str, StandardScaler] = {}

    def fit(self, features: list[FeatureSet]) -> None:
        """
        Ajusta um scaler independente para cada feature numérica.
        """
        # TODO(validation): validar número mínimo de amostras por feature
        # TODO(nans): definir política explícita para valores None
        values_by_feature = defaultdict(list)

        # Agrupar valores por nome da feature
        for fs in features:
            for name, value in fs.features.items():
                if value is not None:
                    values_by_feature[name].append(value)

        # Ajustar um scaler por feature
        for name, values in values_by_feature.items():
            # TODO(VALIDATION): Validar número mínimo de amostras antes do fit
            scaler = StandardScaler()
            scaler.fit(np.array(values).reshape(-1, 1))
            self.scalers[name] = scaler

    def transform(self, features: list[FeatureSet]) -> list[FeatureSet]:
        """
        Aplica normalização feature-a-feature, preservando entidades imutáveis.
        """
        # TODO(safety): garantir que fit foi chamado antes do transform
        normalized = []

        for fs in features:
            new_features = {}

            for name, value in fs.features.items():
                if value is None or name not in self.scalers:
                    new_features[name] = value
                else:
                    new_features[name] = float(
                        self.scalers[name].transform([[value]])[0][0]
                    )

            normalized.append(
                FeatureSet(
                    asset_id=fs.asset_id,
                    timestamp=fs.timestamp,
                    features=new_features,
                )
            )

        return normalized
