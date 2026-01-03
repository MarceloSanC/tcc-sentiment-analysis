# tests/integration/test_yfinance_fetcher.py
from datetime import datetime

import pytest

from src.adapters.yfinance_candle_fetcher import YFinanceCandleFetcher


@pytest.mark.integration
def test_yfinance_fetcher_real():
    fetcher = YFinanceCandleFetcher(max_retries=1)
    candles = fetcher.fetch_candles(
        "PETR4.SA", datetime(2024, 1, 1), datetime(2024, 1, 5)
    )
    assert candles
    assert all(c.close > 0 for c in candles)
    assert all(c.volume >= 0 for c in candles)

