# src/use_cases/feature_engineering_use_case.py
from typing import Iterable

from src.entities.candle import Candle
from src.entities.technical_indicator_set import TechnicalIndicatorSet
from interfaces.candle_repository import CandleRepository
from src.interfaces.technical_indicator_calculator import TechnicalIndicatorCalculatorPort
from src.interfaces.technical_indicator_repository import TechnicalIndicatorRepository
from src.interfaces.technical_indicator_normalizer import TechnicalIndicatorNormalizer


class TechnicalIndicatorEngineeringUseCase:
    def __init__(
        self,
        candle_repository: CandleRepository,
        indicator_calculator: TechnicalIndicatorCalculatorPort,
        indicator_repository: TechnicalIndicatorRepository,
        indicator_normalizer: TechnicalIndicatorNormalizer | None = None,
    ):
        self.candle_repository = candle_repository
        self.indicator_calculator = indicator_calculator
        self.indicator_repository = indicator_repository
        self.indicator_normalizer = indicator_normalizer

    def execute(self, asset_id: str) -> list[TechnicalIndicatorSet]:
        """
        Orquestra o pipeline de indicadores técnicos:
        - carrega candles
        - calcula indicadores técnicos
        - retorna TechnicalIndicatorSet (persistência é responsabilidade externa)
        """

        # Load candles (domínio)
        candles: list[Candle] = self.candle_repository.load_candles(asset_id)

        if not candles:
            raise ValueError(f"Nenhum candle encontrado para {asset_id}")

        # Garantia temporal explícita
        candles = sorted(candles, key=lambda c: c.timestamp)

        # TODO: validar timestamp monotônico

        # Indicadores técnicos
        indicator_sets = self.indicator_calculator.calculate(
            asset_id=asset_id,
            candles=candles,
        )

        if not indicator_sets:
            raise ValueError("Indicator calculator retornou vazio")

        # Normalização
        if self.indicator_normalizer:
            self.indicator_normalizer.fit(indicator_sets)
            indicator_sets = self.indicator_normalizer.transform(indicator_sets)

        # Persistência
        self.indicator_repository.save(asset_id, indicator_sets)

        # TODO: validar schema dos indicadores

        return indicator_sets


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
