# src/adapters/sklearn_feature_normalizer.py
from collections import defaultdict

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.entities.technical_indicator_set import TechnicalIndicatorSet
from src.interfaces.technical_indicator_normalizer import TechnicalIndicatorNormalizer


class SklearnTechnicalIndicatorNormalizer(TechnicalIndicatorNormalizer):
    """
    Adapter responsável por normalizar features numéricas usando sklearn.
    Implementação atual: StandardScaler (z-score), ajustado por feature.
    """

    def __init__(self):
        # TODO(leakage): persistir scalers por asset_id
        # TODO(reprodutibilidade): permitir load/save dos scalers
        self.scalers: dict[str, StandardScaler] = {}

    def fit(self, indicators: list[TechnicalIndicatorSet]) -> None:
        """
        Ajusta um scaler independente para cada feature numérica.
        """
        # TODO(validation): validar número mínimo de amostras por feature
        # TODO(nans): definir política explícita para valores None
        values_by_feature = defaultdict(list)

        # Agrupar valores por nome da feature
        for item in indicators:
            for name, value in item.indicators.items():
                if value is not None:
                    values_by_feature[name].append(value)

        # Ajustar um scaler por feature
        for name, values in values_by_feature.items():
            # TODO(VALIDATION): Validar número mínimo de amostras antes do fit
            scaler = StandardScaler()
            scaler.fit(np.array(values).reshape(-1, 1))
            self.scalers[name] = scaler

    def transform(
        self, indicators: list[TechnicalIndicatorSet]
    ) -> list[TechnicalIndicatorSet]:
        """
        Aplica normalização feature-a-feature, preservando entidades imutáveis.
        """
        # TODO(safety): garantir que fit foi chamado antes do transform
        normalized = []

        for item in indicators:
            new_indicators = {}

            for name, value in item.indicators.items():
                if value is None or name not in self.scalers:
                    new_indicators[name] = value
                else:
                    new_indicators[name] = float(
                        self.scalers[name].transform([[value]])[0][0]
                    )

            normalized.append(
                TechnicalIndicatorSet(
                    asset_id=item.asset_id,
                    timestamp=item.timestamp,
                    indicators=new_indicators,
                )
            )

        return normalized



# =========================
# TODOs — melhorias futuras
# =========================

# TODO (Leakage):
# Persistir scalers por asset_id e por janela temporal
# para evitar data leakage entre treino, validação e inferência.

# TODO (Reproducibility):
# Implementar load/save dos scalers (ex: via joblib)
# para garantir reprodutibilidade entre execuções.

# TODO (Validation):
# Validar número mínimo de amostras por feature
# antes de permitir o ajuste do scaler.

# TODO (NaNs):
# Definir política explícita para valores None / NaN:
# drop | fill (mean/median) | flag binária.

# TODO (Safety):
# Garantir explicitamente que fit() foi chamado
# antes de permitir transform().
