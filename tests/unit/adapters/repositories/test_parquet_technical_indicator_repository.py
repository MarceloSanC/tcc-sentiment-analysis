# tests/unit/test_adapters/test_parquet_feature_set_repository.py

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.adapters.parquet_technical_indicator_repository import (
    ParquetTechnicalIndicatorRepository,
)
from src.entities.technical_indicator_set import TechnicalIndicatorSet


def test_parquet_technical_indicator_repository_save_and_load(tmp_path: Path):
    repo = ParquetTechnicalIndicatorRepository(output_dir=tmp_path)

    asset_id = "AAPL"

    features = [
        TechnicalIndicatorSet(
            asset_id=asset_id,
            timestamp=datetime(2024, 1, 1) + timedelta(days=i),
            indicators={
                "rsi_14": 30.0 + i,
                "ema_50": 100.0 + i,
            },
        )
        for i in range(3)
    ]

    # Act
    repo.save(asset_id, features)
    loaded = repo.load(asset_id)

    # Assert
    assert len(loaded) == 3
    assert all(isinstance(fs, TechnicalIndicatorSet) for fs in loaded)

    # Ordenação temporal preservada
    assert loaded[0].timestamp < loaded[1].timestamp

    # Conteúdo preservado
    assert loaded[0].indicators["rsi_14"] == 30.0
    assert loaded[2].indicators["ema_50"] == 102.0

    # Parquet must be wide (one row per timestamp)
    filepath = tmp_path / f"technical_indicators_{asset_id}.parquet"
    df = pd.read_parquet(filepath)
    assert len(df) == 3
    assert set(["asset_id", "timestamp", "rsi_14", "ema_50"]).issubset(df.columns)
