# tests/unit/adapters/repositories/test_parquet_tft_dataset_repository.py

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.adapters.parquet_tft_dataset_repository import ParquetTFTDatasetRepository


def _dt_utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asset_id": "AAPL",
                "timestamp": _dt_utc(2024, 1, 1),
                "time_idx": 0,
                "day_of_week": 0,
                "month": 1,
                "target_return": 0.01,
                "close": 100.0,
            },
            {
                "asset_id": "AAPL",
                "timestamp": _dt_utc(2024, 1, 2),
                "time_idx": 1,
                "day_of_week": 1,
                "month": 1,
                "target_return": -0.02,
                "close": 98.0,
            },
        ]
    )


def test_save_and_load_dataset(tmp_path: Path) -> None:
    repo = ParquetTFTDatasetRepository(output_dir=tmp_path)

    df = _sample_df()
    repo.save("AAPL", df)

    loaded = repo.load("AAPL")
    assert len(loaded) == 2
    assert set(["asset_id", "timestamp", "time_idx", "target_return"]).issubset(
        loaded.columns
    )

    ts = pd.to_datetime(loaded["timestamp"], utc=True, errors="coerce")
    assert ts.isna().sum() == 0
    assert ts.is_monotonic_increasing


def test_raises_when_missing_base_columns(tmp_path: Path) -> None:
    repo = ParquetTFTDatasetRepository(output_dir=tmp_path)

    df = _sample_df().drop(columns=["target_return"])
    with pytest.raises(ValueError, match="Missing base columns"):
        repo.save("AAPL", df)


def test_output_dir_is_file_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("x", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        ParquetTFTDatasetRepository(output_dir=file_path)
