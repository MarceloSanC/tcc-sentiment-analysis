from typing import Iterable

from src.entities.candle import Candle
from src.entities.technical_indicator_set import TechnicalIndicatorSet
from interfaces.candle_repository import CandleRepository
from src.interfaces.technical_indicator_calculator import TechnicalIndicatorCalculatorPort
from src.interfaces.technical_indicator_repository import TechnicalIndicatorRepository


class TechnicalIndicatorEngineeringUseCase:
    def __init__(
        self,
        candle_repository: CandleRepository,
        indicator_calculator: TechnicalIndicatorCalculatorPort,
        indicator_repository: TechnicalIndicatorRepository,
    ):
        self.candle_repository = candle_repository
        self.indicator_calculator = indicator_calculator
        self.indicator_repository = indicator_repository

    def execute(self, asset_id: str) -> list[TechnicalIndicatorSet]:
        """
        Orchestrates technical indicators pipeline:
        - loads candles
        - calculates technical indicators
        - persists raw indicators (no normalization at this stage)
        """

        candles: list[Candle] = self.candle_repository.load_candles(asset_id)
        if not candles:
            raise ValueError(f"Nenhum candle encontrado para {asset_id}")

        candles = sorted(candles, key=lambda c: c.timestamp)

        indicator_sets = self.indicator_calculator.calculate(
            asset_id=asset_id,
            candles=candles,
        )
        if not indicator_sets:
            raise ValueError("Indicator calculator retornou vazio")

        self.indicator_repository.save(asset_id, indicator_sets)
        return indicator_sets


# TODO(Validation): validate strictly monotonic timestamps.
# TODO(Leakage): keep normalization fit in training split only.
# TODO(Schema): validate final technical feature schema before persistence.
