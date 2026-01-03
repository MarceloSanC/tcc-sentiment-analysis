from datetime import datetime, timedelta

from src.adapters.technical_indicator_calculator import TechnicalIndicatorCalculator
from src.entities.candle import Candle


def test_indicator_calculation_adds_expected_features():
    candles = [
        Candle(
            timestamp=datetime(2024, 1, 1) + timedelta(days=i),
            open=10 + i,
            high=11 + i,
            low=9 + i,
            close=10 + i,
            volume=1000,
        )
        for i in range(250)
    ]

    calc = TechnicalIndicatorCalculator()

    result = calc.calculate(
        asset_id="AAPL",
        candles=candles,
    )

    assert len(result) == len(candles) - 200 + 1

    feature_keys = result[0].features.keys()

    assert "rsi_14" in feature_keys
    assert "macd" in feature_keys
    assert "ema_50" in feature_keys
    assert "volatility_20d" in feature_keys
