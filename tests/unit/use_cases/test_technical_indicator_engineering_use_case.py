# tests/integration/test_feature_engineering_pipeline.py

from collections import defaultdict
from datetime import datetime, timedelta

from src.entities.candle import Candle
from src.entities.technical_indicator_set import TechnicalIndicatorSet
from src.use_cases.technical_indicator_engineering_use_case import (
    TechnicalIndicatorEngineeringUseCase,
)
from src.interfaces.candle_repository import CandleRepository
from src.interfaces.technical_indicator_calculator import TechnicalIndicatorCalculatorPort
from src.interfaces.technical_indicator_repository import TechnicalIndicatorRepository


# ---------- Fakes (test doubles) ----------

class InMemoryCandleRepository(CandleRepository):
    def __init__(self, initial_data: dict[str, list[Candle]] | None = None):
        self._storage: dict[str, list[Candle]] = defaultdict(list)

        if initial_data:
            for asset_id, candles in initial_data.items():
                self._storage[asset_id] = list(candles)

    def load_candles(self, asset_id: str) -> list[Candle]:
        return list(self._storage.get(asset_id, []))

    def save_candles(self, asset_id: str, candles: list[Candle]) -> None:
        # sobrescreve deliberadamente (sem append)
        self._storage[asset_id] = list(candles)

    def update_sentiment(self, asset_id: str, daily_sentiments) -> None:
        """
        Fake implementation.
        FeatureEngineeringUseCase não depende de sentimento.
        """
        return None

class DummyIndicatorCalculator(TechnicalIndicatorCalculatorPort):
    def calculate(
        self, asset_id: str, candles: list[Candle]
    ) -> list[TechnicalIndicatorSet]:
        return [
            TechnicalIndicatorSet(
                asset_id=asset_id,
                timestamp=c.timestamp,
                indicators={"close": c.close},
            )
            for c in candles
        ]

class InMemoryTechnicalIndicatorRepository(TechnicalIndicatorRepository):
    def __init__(self):
        self._storage: dict[str, list[TechnicalIndicatorSet]] = {}

    def save(self, asset_id: str, indicators: list[TechnicalIndicatorSet]) -> None:
        # overwrite deliberado (pipeline idempotente)
        self._storage[asset_id] = list(indicators)

    def load(self, asset_id: str) -> list[TechnicalIndicatorSet]:
        return list(self._storage.get(asset_id, []))

# ---------- Test ----------

def test_technical_indicator_pipeline_runs():
    asset_id = "AAPL"

    candles = [
        Candle(
            timestamp=datetime(2024, 1, 1) + timedelta(days=i),
            open=100 + i,
            high=101 + i,
            low=99 + i,
            close=100.5 + i,
            volume=1000 + i,
        )
        for i in range(5)
    ]

    candle_repo = InMemoryCandleRepository(
        initial_data={asset_id: candles}
    )

    indicator_calculator = DummyIndicatorCalculator()

    indicator_repository = InMemoryTechnicalIndicatorRepository()

    use_case = TechnicalIndicatorEngineeringUseCase(
        candle_repository=candle_repo,
        indicator_calculator=indicator_calculator,
        indicator_repository=indicator_repository,
    )

    indicator_sets = use_case.execute(asset_id)

    # -------- Assertions --------
    assert len(indicator_sets) == 5
    assert all(isinstance(fs, TechnicalIndicatorSet) for fs in indicator_sets)
    assert indicator_sets[0].asset_id == asset_id
    assert indicator_sets[0].indicators["close"] == candles[0].close

    persisted = indicator_repository.load(asset_id)

    assert len(persisted) == 5
    assert persisted[0].indicators["close"] == candles[0].close
