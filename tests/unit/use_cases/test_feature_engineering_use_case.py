# tests/integration/test_feature_engineering_pipeline.py

from collections import defaultdict
from datetime import datetime, timedelta

from src.entities.candle import Candle
from src.entities.feature_set import FeatureSet
from src.use_cases.feature_engineering_use_case import FeatureEngineeringUseCase
from src.interfaces.candle_repository import CandleRepository
from src.interfaces.feature_calculator import FeatureCalculator
from src.interfaces.feature_set_repository import FeatureSetRepository


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

class DummyFeatureCalculator(FeatureCalculator):
    def calculate(self, asset_id: str, candles: list[Candle]) -> list[FeatureSet]:
        return [
            FeatureSet(
                asset_id=asset_id,
                timestamp=c.timestamp,
                features={"close": c.close},
            )
            for c in candles
        ]

class InMemoryFeatureSetRepository(FeatureSetRepository):
    def __init__(self):
        self._storage: dict[str, list[FeatureSet]] = {}

    def save(self, asset_id: str, features: list[FeatureSet]) -> None:
        # overwrite deliberado (pipeline idempotente)
        self._storage[asset_id] = list(features)

    def load(self, asset_id: str) -> list[FeatureSet]:
        return list(self._storage.get(asset_id, []))

# ---------- Test ----------

def test_feature_engineering_pipeline_runs():
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

    feature_calculator = DummyFeatureCalculator()

    feature_repository = InMemoryFeatureSetRepository()

    use_case = FeatureEngineeringUseCase(
        candle_repository=candle_repo,
        feature_calculator=feature_calculator,
        feature_repository=feature_repository,
    )

    feature_sets = use_case.execute(asset_id)

    # -------- Assertions --------
    assert len(feature_sets) == 5
    assert all(isinstance(fs, FeatureSet) for fs in feature_sets)
    assert feature_sets[0].asset_id == asset_id
    assert feature_sets[0].features["close"] == candles[0].close

    persisted = feature_repository.load(asset_id)

    assert len(persisted) == 5
    assert persisted[0].features["close"] == candles[0].close
