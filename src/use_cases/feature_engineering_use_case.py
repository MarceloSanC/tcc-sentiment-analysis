# src/use_cases/feature_engineering_use_case.py
from typing import Iterable

from src.entities.candle import Candle
from src.entities.feature_set import FeatureSet
from interfaces.candle_repository import DataRepository
from src.interfaces.feature_calculator import FeatureCalculator
# TODO: criar FeatureSetRepository para persistência desacoplada


class FeatureEngineeringUseCase:
    def __init__(
        self,
        candle_repository: DataRepository,
        feature_calculator: FeatureCalculator,
        # TODO: injetar FeatureSetRepository
    ):
        self.candle_repository = candle_repository
        self.feature_calculator = feature_calculator

    def execute(self, asset_id: str) -> list[FeatureSet]:
        """
        Orquestra o pipeline de feature engineering:
        - carrega candles
        - calcula features
        - retorna FeatureSets (persistência é responsabilidade externa)
        """

        # 1. Load candles (domínio)
        candles: list[Candle] = self.candle_repository.load_candles(asset_id)

        if not candles:
            raise ValueError(f"Nenhum candle encontrado para {asset_id}")

        # 2. Garantia temporal explícita
        candles = sorted(candles, key=lambda c: c.timestamp)

        # TODO: validar timestamp monotônico

        # 3. Feature engineering
        feature_sets = self.feature_calculator.calculate(
            asset_id=asset_id,
            candles=candles,
        )

        # TODO: aplicar normalização (pipeline explícito)
        # TODO: validar schema das features
        # TODO: persistir FeatureSets via FeatureSetRepository

        return feature_sets
