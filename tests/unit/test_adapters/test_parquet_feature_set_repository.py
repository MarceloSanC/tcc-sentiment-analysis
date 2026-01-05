# tests/unit/test_adapters/test_parquet_feature_set_repository.py
from datetime import datetime, timedelta
from pathlib import Path

from src.adapters.parquet_feature_set_repository import ParquetFeatureSetRepository
from src.entities.feature_set import FeatureSet


def test_parquet_feature_set_repository_save_and_load(tmp_path: Path):
    repo = ParquetFeatureSetRepository(output_dir=tmp_path)

    asset_id = "AAPL"

    features = [
        FeatureSet(
            asset_id=asset_id,
            timestamp=datetime(2024, 1, 1) + timedelta(days=i),
            features={
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
    assert all(isinstance(fs, FeatureSet) for fs in loaded)

    # Ordenação temporal preservada
    assert loaded[0].timestamp < loaded[1].timestamp

    # Conteúdo preservado
    assert loaded[0].features["rsi_14"] == 30.0
    assert loaded[2].features["ema_50"] == 102.0