# tests/integration/test_parquet_repository.py
from datetime import datetime
from pathlib import Path

from adapters.parquet_candle_repository import ParquetDataRepository
from src.entities.candle import Candle


def test_parquet_repository_saves_file(tmp_path: Path):
    repo = ParquetDataRepository(output_dir=tmp_path)

    candles = [
        Candle(
            timestamp=datetime(2024, 1, 1),
            open=10,
            high=12,
            low=9,
            close=11,
            volume=1000,
        )
    ]

    repo.save_candles("AAPL", candles)

    files = list(tmp_path.glob("*.parquet"))
    assert len(files) == 1
