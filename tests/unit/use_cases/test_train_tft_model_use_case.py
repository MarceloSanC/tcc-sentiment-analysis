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
    calls: int = 0
    seen_train_df: pd.DataFrame | None = None
    seen_val_df: pd.DataFrame | None = None
    seen_test_df: pd.DataFrame | None = None

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        feature_cols: list[str],
        target_col: str,
        time_idx_col: str,
        group_col: str,
        known_real_cols: list[str],
        config: dict,
    ) -> TrainingResult:
        self.calls += 1
        self.seen_features = feature_cols
        self.seen_train_df = train_df.copy()
        self.seen_val_df = val_df.copy()
        self.seen_test_df = test_df.copy()
        return TrainingResult(
            model=object(),
            metrics={"rmse": 1.0},
            history=[],
            split_metrics={
                "train": {"rmse": 0.9, "mae": 0.7},
                "val": {"rmse": 1.0, "mae": 0.8},
                "test": {"rmse": 1.1, "mae": 0.9},
            },
        )


@dataclass
class FakeModelRepo(ModelRepository):
    saved: bool = False
    last_ablation_results: list[dict[str, float | str]] | None = None
    last_version: str | None = None
    last_dataset_parameters: dict | None = None

    def save_training_artifacts(
        self,
        asset_id: str,
        version: str,
        model,
        *,
        metrics: dict[str, float],
        history: list[dict[str, float]],
        split_metrics: dict[str, dict[str, float]],
        features_used: list[str],
        training_window: dict[str, str],
        split_window: dict[str, str],
        config: dict,
        feature_importance: list[dict[str, float | str]] | None = None,
        ablation_results: list[dict[str, float | str]] | None = None,
        checkpoint_path: str | None = None,
        dataset_parameters: dict | None = None,
        plots: dict[str, str] | None = None,
    ) -> str:
        self.saved = True
        self.last_ablation_results = ablation_results
        self.last_version = version
        self.last_dataset_parameters = dataset_parameters
        return "fake_dir"


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2025-01-02"], utc=True
            ),
            "asset_id": ["AAPL", "AAPL", "AAPL"],
            "time_idx": [0, 1, 2],
            "target_return": [0.1, 0.2, 0.3],
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [3.0, 4.0, 5.0],
            "day_of_week": [0, 1, 3],
            "month": [1, 1, 1],
        }
    )


def _df_ablation() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2025-01-02"], utc=True
            ),
            "asset_id": ["AAPL", "AAPL", "AAPL"],
            "time_idx": [0, 1, 2],
            "target_return": [0.1, 0.2, 0.3],
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "volume": [1000, 1100, 1200],
            "volatility_20d": [0.1, 0.2, 0.3],
            "sentiment_score": [0.1, -0.2, 0.0],
            "news_volume": [3, 0, 1],
            "sentiment_std": [0.2, 0.0, 0.1],
            "revenue": [100.0, 100.0, 100.0],
            "net_income": [10.0, 10.0, 10.0],
            "operating_cash_flow": [15.0, 15.0, 15.0],
            "total_shareholder_equity": [50.0, 50.0, 50.0],
            "total_liabilities": [25.0, 25.0, 25.0],
            "day_of_week": [0, 1, 3],
            "month": [1, 1, 1],
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
        "test_start": "20250102",
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


def test_runs_ablation_when_features_not_provided() -> None:
    trainer = FakeTrainer()
    repo = FakeModelRepo()
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df_ablation()),
        model_trainer=trainer,
        model_repository=repo,
    )

    split_config = {
        "train_start": "20240101",
        "train_end": "20240101",
        "val_start": "20240102",
        "val_end": "20240102",
        "test_start": "20250102",
        "test_end": "20250102",
    }

    use_case.execute("AAPL", split_config=split_config, run_ablation=True)

    # 1 run for primary training + 5 runs for ablation experiments
    assert trainer.calls == 6
    assert repo.last_ablation_results is not None
    assert len(repo.last_ablation_results) == 5


def test_resolves_group_tokens_for_features() -> None:
    trainer = FakeTrainer()
    repo = FakeModelRepo()
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df_ablation()),
        model_trainer=trainer,
        model_repository=repo,
    )
    split_config = {
        "train_start": "20240101",
        "train_end": "20240101",
        "val_start": "20240102",
        "val_end": "20240102",
        "test_start": "20250102",
        "test_end": "20250102",
    }

    use_case.execute(
        "AAPL",
        features=["BASELINE_FEATURES", "TECHNICAL_FEATURES"],
        split_config=split_config,
    )

    assert trainer.seen_features is not None
    assert "open" in trainer.seen_features
    assert "volatility_20d" in trainer.seen_features
    assert repo.last_version is not None
    assert repo.last_version.endswith("_BT")


def test_custom_feature_tokens_add_c_suffix() -> None:
    trainer = FakeTrainer()
    repo = FakeModelRepo()
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df_ablation()),
        model_trainer=trainer,
        model_repository=repo,
    )
    split_config = {
        "train_start": "20240101",
        "train_end": "20240101",
        "val_start": "20240102",
        "val_end": "20240102",
        "test_start": "20250102",
        "test_end": "20250102",
    }

    use_case.execute(
        "AAPL",
        features=["BASELINE_FEATURES", "sentiment_score"],
        split_config=split_config,
    )

    assert repo.last_version is not None
    assert repo.last_version.endswith("_BC")


def test_raises_when_split_window_is_empty() -> None:
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df()),
        model_trainer=FakeTrainer(),
        model_repository=FakeModelRepo(),
    )
    split_config = {
        "train_start": "20100101",
        "train_end": "20100131",
        "val_start": "20100201",
        "val_end": "20100228",
        "test_start": "20100301",
        "test_end": "20100331",
    }
    with pytest.raises(ValueError, match="empty dataset"):
        use_case.execute("AAPL", features=["feature_a"], split_config=split_config)


def test_raises_when_unknown_feature_token_is_provided() -> None:
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df()),
        model_trainer=FakeTrainer(),
        model_repository=FakeModelRepo(),
    )
    split_config = {
        "train_start": "20240101",
        "train_end": "20240101",
        "val_start": "20240102",
        "val_end": "20240102",
        "test_start": "20250102",
        "test_end": "20250102",
    }
    with pytest.raises(ValueError, match="Requested features not found"):
        use_case.execute("AAPL", features=["UNKNOWN_GROUP"], split_config=split_config)


def test_skips_ablation_when_explicit_features_are_provided() -> None:
    trainer = FakeTrainer()
    repo = FakeModelRepo()
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df_ablation()),
        model_trainer=trainer,
        model_repository=repo,
    )
    split_config = {
        "train_start": "20240101",
        "train_end": "20240101",
        "val_start": "20240102",
        "val_end": "20240102",
        "test_start": "20250102",
        "test_end": "20250102",
    }

    use_case.execute(
        "AAPL",
        features=["BASELINE_FEATURES", "TECHNICAL_FEATURES"],
        split_config=split_config,
        run_ablation=True,
    )

    assert trainer.calls == 1
    assert repo.last_ablation_results == []


def test_raises_when_default_features_not_available() -> None:
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df()),
        model_trainer=FakeTrainer(),
        model_repository=FakeModelRepo(),
    )
    split_config = {
        "train_start": "20240101",
        "train_end": "20240101",
        "val_start": "20240102",
        "val_end": "20240102",
        "test_start": "20250102",
        "test_end": "20250102",
    }
    with pytest.raises(ValueError, match="Default feature set is empty"):
        use_case.execute("AAPL", features=None, split_config=split_config)


def test_raises_when_features_list_is_empty() -> None:
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df_ablation()),
        model_trainer=FakeTrainer(),
        model_repository=FakeModelRepo(),
    )
    split_config = {
        "train_start": "20240101",
        "train_end": "20240101",
        "val_start": "20240102",
        "val_end": "20240102",
        "test_start": "20250102",
        "test_end": "20250102",
    }
    with pytest.raises(ValueError, match="No valid features"):
        use_case.execute("AAPL", features=[], split_config=split_config)


def test_applies_split_normalization_for_technical_features_and_persists_scalers() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2025-01-02"], utc=True
            ),
            "asset_id": ["AAPL", "AAPL", "AAPL"],
            "time_idx": [0, 1, 2],
            "target_return": [0.1, 0.2, 0.3],
            "open": [10.0, 11.0, 12.0],
            "high": [11.0, 12.0, 13.0],
            "low": [9.0, 10.0, 11.0],
            "close": [10.5, 11.5, 12.5],
            "volume": [1000, 1100, 1200],
            "volatility_20d": [10.0, 20.0, 30.0],
            "day_of_week": [0, 1, 3],
            "month": [1, 1, 1],
        }
    )
    trainer = FakeTrainer()
    repo = FakeModelRepo()
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(df),
        model_trainer=trainer,
        model_repository=repo,
    )
    split_config = {
        "train_start": "20240101",
        "train_end": "20240101",
        "val_start": "20240102",
        "val_end": "20240102",
        "test_start": "20250102",
        "test_end": "20250102",
    }

    use_case.execute(
        "AAPL",
        features=["BASELINE_FEATURES", "TECHNICAL_FEATURES"],
        split_config=split_config,
    )

    assert trainer.seen_train_df is not None
    assert trainer.seen_val_df is not None
    assert trainer.seen_test_df is not None
    assert float(trainer.seen_train_df["volatility_20d"].iloc[0]) == 0.0
    assert float(trainer.seen_val_df["volatility_20d"].iloc[0]) == 10.0
    assert float(trainer.seen_test_df["volatility_20d"].iloc[0]) == 20.0
    assert repo.last_dataset_parameters is not None
    assert "scalers" in repo.last_dataset_parameters
    assert "volatility_20d" in repo.last_dataset_parameters["scalers"]
