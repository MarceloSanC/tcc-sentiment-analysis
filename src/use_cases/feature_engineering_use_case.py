# src/use_cases/feature_engineering_use_case.py
from typing import Iterable

from src.entities.candle import Candle
from src.entities.feature_set import FeatureSet
from interfaces.candle_repository import CandleRepository
from src.interfaces.feature_calculator import FeatureCalculator
from src.interfaces.feature_set_repository import FeatureSetRepository
from src.interfaces.feature_normalizer import FeatureNormalizer


class FeatureEngineeringUseCase:
    def __init__(
        self,
        candle_repository: CandleRepository,
        feature_calculator: FeatureCalculator,
        feature_repository: FeatureSetRepository,
        feature_normalizer: FeatureNormalizer | None = None,
    ):
        self.candle_repository = candle_repository
        self.feature_calculator = feature_calculator
        self.feature_repository = feature_repository
        self.feature_normalizer = feature_normalizer

    def execute(self, asset_id: str) -> list[FeatureSet]:
        """
        Orquestra o pipeline de feature engineering:
        - carrega candles
        - calcula features
        - retorna FeatureSets (persistência é responsabilidade externa)
        """

        # Load candles (domínio)
        candles: list[Candle] = self.candle_repository.load_candles(asset_id)

        if not candles:
            raise ValueError(f"Nenhum candle encontrado para {asset_id}")

        # Garantia temporal explícita
        candles = sorted(candles, key=lambda c: c.timestamp)

        # TODO: validar timestamp monotônico

        # Feature engineering
        feature_sets = self.feature_calculator.calculate(
            asset_id=asset_id,
            candles=candles,
        )

        if not feature_sets:
            raise ValueError("Feature calculator retornou vazio")

        # Normalização
        if self.feature_normalizer:
            self.feature_normalizer.fit(feature_sets)
            feature_sets = self.feature_normalizer.transform(feature_sets)

        # Persistência
        self.feature_repository.save(asset_id, feature_sets)

        # TODO: validar schema das features

        return feature_sets


# =========================
# TODOs — melhorias futuras
# =========================

# TODO (Validation):
# Validar monotonicidade estrita dos timestamps
# (sem duplicatas ou retrocessos temporais).

# TODO (Leakage):
# Ajustar normalizador apenas em janelas de treino
# e reutilizar parâmetros em validação/inferência

# TODO (Schema):
# Validar schema final das features antes da persistência
# para garantir consistência entre execuções e modelos.