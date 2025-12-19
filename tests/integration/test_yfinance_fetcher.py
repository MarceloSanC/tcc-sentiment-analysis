# tests/integration/test_yfinance_fetcher.py
from datetime import datetime

import pytest

from src.adapters.yfinance_data_fetcher import YFinanceDataFetcher


@pytest.mark.integration
def test_yfinance_fetcher_real():
    fetcher = YFinanceDataFetcher(max_retries=1)
    candles = fetcher.fetch_candles(
        "PETR4.SA", datetime(2024, 1, 1), datetime(2024, 1, 5)
    )
    assert len(candles) > 0
    assert candles[0].close > 0
    assert candles[0].volume > 0
