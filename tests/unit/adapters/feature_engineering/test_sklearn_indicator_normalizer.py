from __future__ import annotations

from datetime import datetime, timedelta

from src.adapters.sklearn_indicator_normalizer import SklearnTechnicalIndicatorNormalizer
from src.entities.technical_indicator_set import TechnicalIndicatorSet


def test_normalizer_ignores_nan_in_fit_and_preserves_series() -> None:
    base = datetime(2024, 1, 1)
    indicators = [
        TechnicalIndicatorSet(
            asset_id="AAPL",
            timestamp=base,
            indicators={"ema_200": float("nan"), "candle_range": 2.0},
        ),
        TechnicalIndicatorSet(
            asset_id="AAPL",
            timestamp=base + timedelta(days=1),
            indicators={"ema_200": 10.0, "candle_range": 3.0},
        ),
        TechnicalIndicatorSet(
            asset_id="AAPL",
            timestamp=base + timedelta(days=2),
            indicators={"ema_200": 11.0, "candle_range": 4.0},
        ),
    ]

    normalizer = SklearnTechnicalIndicatorNormalizer()
    normalizer.fit(indicators)
    transformed = normalizer.transform(indicators)

    assert len(transformed) == 3
    assert "ema_200" in transformed[0].indicators
    assert transformed[0].indicators["ema_200"] != transformed[0].indicators["ema_200"]
    assert transformed[-1].indicators["ema_200"] == transformed[-1].indicators["ema_200"]
