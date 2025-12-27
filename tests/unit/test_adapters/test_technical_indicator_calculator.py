# tests/unit/test_adapters/test_technical_indicator_calculator.py
import pandas as pd

from src.adapters.technical_indicator_calculator import TechnicalIndicatorCalculator


def test_indicator_calculation_adds_expected_columns():
    df = pd.DataFrame(
        {
            "open": [10, 11, 12, 13, 14] * 10,
            "high": [11, 12, 13, 14, 15] * 10,
            "low": [9, 10, 11, 12, 13] * 10,
            "close": [10, 11, 12, 13, 14] * 10,
            "volume": [100] * 50,
        }
    )

    calc = TechnicalIndicatorCalculator()
    result = calc.calculate(df)

    assert "rsi_14" in result.columns
    assert "macd" in result.columns
    assert "ema_50" in result.columns
