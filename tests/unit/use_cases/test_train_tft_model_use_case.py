from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from src.interfaces.model_trainer import ModelTrainer, TrainingResult
from src.interfaces.model_repository import ModelRepository
from src.interfaces.tft_dataset_repository import TFTDatasetRepository
from src.use_cases.train_tft_model_use_case import TrainTFTModelUseCase


class FakeDatasetRepository(TFTDatasetRepository):
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def save(self, asset_id: str, df: pd.DataFrame) -> None:
        raise NotImplementedError

    def load(self, asset_id: str) -> pd.DataFrame:
        return self._df


@dataclass
class FakeTrainer(ModelTrainer):
    seen_features: list[str] | None = None

    def train(
        self,
        df: pd.DataFrame,
        *,
        feature_cols: list[str],
        target_col: str,
        time_idx_col: str,
        group_col: str,
        known_real_cols: list[str],
        config: dict,
    ) -> TrainingResult:
        self.seen_features = feature_cols
        return TrainingResult(model=object(), metrics={"rmse": 1.0}, history=[])


@dataclass
class FakeModelRepo(ModelRepository):
    saved: bool = False

    def save_training_artifacts(
        self,
        asset_id: str,
        version: str,
        model,
        *,
        metrics: dict[str, float],
        history: list[dict[str, float]],
        features_used: list[str],
        training_window: dict[str, str],
        config: dict,
        plots: dict[str, str] | None = None,
    ) -> str:
        self.saved = True
        return "fake_dir"


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02"], utc=True
            ),
            "asset_id": ["AAPL", "AAPL"],
            "time_idx": [0, 1],
            "target_return": [0.1, 0.2],
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
            "day_of_week": [0, 1],
            "month": [1, 1],
        }
    )


def test_selects_requested_features() -> None:
    trainer = FakeTrainer()
    repo = FakeModelRepo()
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df()),
        model_trainer=trainer,
        model_repository=repo,
    )

    split_config = {
        "train_start": "20240101",
        "train_end": "20240101",
        "val_start": "20240102",
        "val_end": "20240102",
        "test_start": "20250101",
        "test_end": "20250102",
    }
    result = use_case.execute("AAPL", features=["feature_b"], split_config=split_config)

    assert trainer.seen_features == ["feature_b"]
    assert repo.saved is True
    assert result.metrics["rmse"] == 1.0


def test_raises_when_feature_missing() -> None:
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df()),
        model_trainer=FakeTrainer(),
        model_repository=FakeModelRepo(),
    )

    with pytest.raises(ValueError, match="Requested features not found"):
        use_case.execute("AAPL", features=["not_a_feature"])


def test_warns_on_leaky_split_ranges(caplog) -> None:
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df()),
        model_trainer=FakeTrainer(),
        model_repository=FakeModelRepo(),
    )

    split_config = {
        "train_start": "20240101",
        "train_end": "20240110",
        "val_start": "20240105",  # overlaps train
        "val_end": "20240120",
        "test_start": "20240201",
        "test_end": "20240210",
    }

    with pytest.raises(ValueError, match="data leakage risk"):
        use_case.execute("AAPL", features=["feature_a"], split_config=split_config)

    assert any("data leakage" in r.message for r in caplog.records)
