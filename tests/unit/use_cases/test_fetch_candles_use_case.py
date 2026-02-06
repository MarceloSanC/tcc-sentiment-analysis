# tests/unit/use_cases/test_fetch_candles_use_case.py

from datetime import datetime, timezone
from unittest.mock import Mock

from src.entities.candle import Candle
from src.use_cases.fetch_candles_use_case import FetchCandlesUseCase


def test_fetch_candles_saves_data():
    # Arrange
    mock_fetcher = Mock()
    mock_repo = Mock()

    candles = [
        Candle(
            timestamp=datetime(2022, 1, 3, tzinfo=timezone.utc),
            open=25.1,
            high=25.8,
            low=24.9,
            close=25.5,
            volume=1000000,
        )
    ]
    mock_fetcher.fetch_candles.return_value = candles
    mock_repo.load_candles.return_value = []  # existente vazio

    use_case = FetchCandlesUseCase(mock_fetcher, mock_repo)

    start = datetime(2022, 1, 1, tzinfo=timezone.utc)
    end = datetime(2022, 1, 5, tzinfo=timezone.utc)

    # Act
    fetched, existing = use_case.execute("AAPL", start, end)

    # Assert
    # contrato: end é normalizado para o próximo dia (boundary)
    mock_fetcher.fetch_candles.assert_called_once_with(
        "AAPL",
        start,
        datetime(2022, 1, 6, 0, 0, tzinfo=timezone.utc),
    )

    mock_repo.save_candles.assert_called_once_with("AAPL", candles)
    assert fetched == len(candles)
    assert existing == 0


def test_fetch_candles_normalizes_end_datetime_to_next_day_boundary():
    # Arrange
    mock_fetcher = Mock()
    mock_repo = Mock()
    mock_fetcher.fetch_candles.return_value = []  # evita len(Mock)
    mock_repo.load_candles.return_value = []  # evita len(Mock)

    use_case = FetchCandlesUseCase(mock_fetcher, mock_repo)

    start = datetime(2022, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2022, 1, 5, 15, 30, tzinfo=timezone.utc)

    # Act
    use_case.execute("AAPL", start, end)

    # Assert
    mock_fetcher.fetch_candles.assert_called_once_with(
        "AAPL",
        start,
        datetime(2022, 1, 6, 0, 0, tzinfo=timezone.utc),
    )
