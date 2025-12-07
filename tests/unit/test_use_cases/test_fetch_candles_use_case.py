# tests/unit/test_use_cases/test_fetch_candles_use_case.py
from unittest.mock import Mock
from datetime import datetime
from src.entities.candle import Candle
from src.use_cases.fetch_candles_use_case import FetchCandlesUseCase

def test_fetch_candles_saves_data():
    # Arrange
    mock_fetcher = Mock()
    mock_repo = Mock()
    
    candles = [
        Candle(
            timestamp=datetime(2022, 1, 3),
            open=25.1, high=25.8, low=24.9, close=25.5, volume=1000000
        )
    ]
    mock_fetcher.fetch_candles.return_value = candles

    use_case = FetchCandlesUseCase(mock_fetcher, mock_repo)

    # Act
    use_case.execute("PETR4.SA", datetime(2022, 1, 1), datetime(2022, 1, 5))

    # Assert
    mock_fetcher.fetch_candles.assert_called_once_with(
        "PETR4.SA", datetime(2022, 1, 1), datetime(2022, 1, 5)
    )
    mock_repo.save_candles.assert_called_once_with("PETR4.SA", candles)