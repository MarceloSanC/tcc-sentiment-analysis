from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pytest

from src.interfaces.model_repository import ModelRepository
from src.interfaces.model_trainer import ModelTrainer, TrainingResult
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
            history=[{"epoch": 0.0, "train_loss": 0.9, "val_loss": 1.0, "epoch_time_seconds": 1.2, "best_epoch": 0.0, "stopped_epoch": 0.0, "early_stop_reason": "max_epochs"}],
            split_metrics={
                "train": {"rmse": 0.9, "mae": 0.7, "directional_accuracy": 0.6},
                "val": {"rmse": 1.0, "mae": 0.8, "directional_accuracy": 0.55},
                "test": {"rmse": 1.1, "mae": 0.9, "directional_accuracy": 0.52},
            },
            split_predictions={
                "train": {"y_true": [0.1], "y_pred": [0.1], "quantile_p10": [0.1], "quantile_p50": [0.1], "quantile_p90": [0.1], "horizon": [1.0]},
                "val": {"y_true": [0.2], "y_pred": [0.2], "quantile_p10": [0.2], "quantile_p50": [0.2], "quantile_p90": [0.2], "horizon": [1.0]},
                "test": {"y_true": [0.3], "y_pred": [0.3], "quantile_p10": [0.3], "quantile_p50": [0.3], "quantile_p90": [0.3], "horizon": [1.0]},
            },
        )


@dataclass
class FakeAnalyticsRunRepo:
    rows: list[dict] | None = None
    snapshots: list[dict] | None = None
    split_refs: list[dict] | None = None
    fact_configs: list[dict] | None = None
    split_metric_rows: list[dict] | None = None
    failures: list[dict] | None = None
    epoch_rows: list[dict] | None = None
    oos_rows: list[dict] | None = None
    model_artifacts_rows: list[dict] | None = None
    bridge_feature_rows: list[dict] | None = None
    inference_rows: list[dict] | None = None

    def upsert_dim_run(self, row: dict) -> None:
        if self.rows is None:
            self.rows = []
        self.rows.append(row)

    def append_fact_run_snapshot(self, row: dict, overwrite: bool = False) -> None:
        if self.snapshots is None:
            self.snapshots = []
        self.snapshots.append({**row, "_overwrite": overwrite})

    def append_fact_split_timestamps_ref(
        self,
        rows: list[dict],
        overwrite: bool = False,
    ) -> None:
        if self.split_refs is None:
            self.split_refs = []
        self.split_refs.extend(rows)

    def append_fact_config(self, row: dict, overwrite: bool = False) -> None:
        if self.fact_configs is None:
            self.fact_configs = []
        self.fact_configs.append({**row, "_overwrite": overwrite})

    def append_fact_split_metrics(self, rows: list[dict], overwrite: bool = False) -> None:
        if self.split_metric_rows is None:
            self.split_metric_rows = []
        self.split_metric_rows.extend(rows)

    def append_fact_epoch_metrics(self, rows: list[dict], overwrite: bool = False) -> None:
        if self.epoch_rows is None:
            self.epoch_rows = []
        self.epoch_rows.extend(rows)

    def append_fact_oos_predictions(self, rows: list[dict], overwrite: bool = False) -> None:
        if self.oos_rows is None:
            self.oos_rows = []
        self.oos_rows.extend(rows)

    def append_fact_model_artifacts(self, row: dict, overwrite: bool = False) -> None:
        if self.model_artifacts_rows is None:
            self.model_artifacts_rows = []
        self.model_artifacts_rows.append({**row, "_overwrite": overwrite})

    def append_bridge_run_features(self, rows: list[dict], overwrite: bool = False) -> None:
        if self.bridge_feature_rows is None:
            self.bridge_feature_rows = []
        self.bridge_feature_rows.extend(rows)

    def append_fact_inference_runs(self, row: dict, overwrite: bool = False) -> None:
        if self.inference_rows is None:
            self.inference_rows = []
        self.inference_rows.append({**row, "_overwrite": overwrite})

    def append_fact_failures(self, row: dict, overwrite: bool = False) -> None:
        if self.failures is None:
            self.failures = []
        self.failures.append({**row, "_overwrite": overwrite})



@dataclass
class FailingTrainer(ModelTrainer):
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
        raise RuntimeError("boom training")

class FakeModelRepo(ModelRepository):
    saved: bool = False
    last_ablation_results: list[dict[str, float | str]] | None = None
    last_version: str | None = None
    last_dataset_parameters: dict | None = None
    last_config: dict | None = None

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
        self.last_config = config
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
            "rsi_14": [30.0, 40.0, 50.0],
            "candle_body": [0.5, 0.5, 0.5],
            "macd_signal": [0.01, 0.02, 0.03],
            "ema_100": [10.0, 10.1, 10.2],
            "macd": [0.1, 0.2, 0.3],
            "ema_10": [10.2, 10.3, 10.4],
            "ema_200": [9.8, 9.9, 10.0],
            "ema_50": [10.1, 10.2, 10.3],
            "candle_range": [2.0, 2.0, 2.0],
            "sentiment_score": [0.1, -0.2, 0.0],
            "news_volume": [3, 0, 1],
            "sentiment_std": [0.2, 0.0, 0.1],
            "has_news": [1, 0, 1],
            "revenue": [100.0, 100.0, 100.0],
            "net_income": [10.0, 10.0, 10.0],
            "operating_cash_flow": [15.0, 15.0, 15.0],
            "total_shareholder_equity": [50.0, 50.0, 50.0],
            "total_liabilities": [25.0, 25.0, 25.0],
            "net_margin": [0.1, 0.1, 0.1],
            "leverage_ratio": [0.5, 0.5, 0.5],
            "cashflow_efficiency": [0.15, 0.15, 0.15],
            "revenue_yoy_growth": [0.0, 0.0, 0.0],
            "net_income_yoy_growth": [0.0, 0.0, 0.0],
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
    assert repo.last_config is not None
    assert repo.last_config["feature_list_ordered"] == trainer.seen_features
    assert isinstance(repo.last_config["feature_set_hash"], str)
    assert len(repo.last_config["feature_set_hash"]) == 64


def test_warmup_strict_fail_raises_for_window_feature() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2025-01-02"], utc=True
            ),
            "asset_id": ["AAPL", "AAPL", "AAPL"],
            "time_idx": [0, 1, 2],
            "target_return": [0.1, 0.2, 0.3],
            "ema_200": [None, 1.0, 2.0],
            "day_of_week": [0, 1, 3],
            "month": [1, 1, 1],
        }
    )
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(df),
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
    with pytest.raises(
        ValueError,
        match=r"policy=strict_fail.*first_valid_train_start=20240102",
    ):
        use_case.execute(
            "AAPL",
            features=["ema_200"],
            split_config=split_config,
            training_config={"warmup_policy": "strict_fail"},
        )


def test_warmup_drop_leading_adjusts_train_start_and_runs() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2025-01-02", "2026-01-02"], utc=True
            ),
            "asset_id": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "time_idx": [0, 1, 2, 3],
            "target_return": [0.1, 0.2, 0.3, 0.4],
            "ema_200": [None, 1.0, 2.0, 3.0],
            "day_of_week": [0, 1, 3, 4],
            "month": [1, 1, 1, 1],
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
        "train_end": "20240102",
        "val_start": "20250102",
        "val_end": "20250102",
        "test_start": "20260102",
        "test_end": "20260102",
    }
    result = use_case.execute(
        "AAPL",
        features=["ema_200"],
        split_config=split_config,
        training_config={"warmup_policy": "drop_leading"},
    )
    assert result.metrics["rmse"] == 1.0
    assert repo.saved is True
    assert repo.last_config is not None
    assert repo.last_config["warmup_policy"] == "drop_leading"
    assert repo.last_config["warmup_applied"] is True
    assert repo.last_config["effective_train_start"] == "20240102"
    assert repo.last_config["feature_list_ordered"] == ["ema_200"]


def test_warmup_drop_leading_revalidates_min_samples_and_fails_if_insufficient() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2025-01-02", "2026-01-02"], utc=True
            ),
            "asset_id": ["AAPL", "AAPL", "AAPL", "AAPL"],
            "time_idx": [0, 1, 2, 3],
            "target_return": [0.1, 0.2, 0.3, 0.4],
            "ema_200": [None, 1.0, 2.0, 3.0],
            "day_of_week": [0, 1, 3, 4],
            "month": [1, 1, 1, 1],
        }
    )
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(df),
        model_trainer=FakeTrainer(),
        model_repository=FakeModelRepo(),
    )
    split_config = {
        "train_start": "20240101",
        "train_end": "20240102",
        "val_start": "20250102",
        "val_end": "20250102",
        "test_start": "20260102",
        "test_end": "20260102",
    }
    with pytest.raises(ValueError, match="insufficient samples"):
        use_case.execute(
            "AAPL",
            features=["ema_200"],
            split_config=split_config,
            training_config={
                "warmup_policy": "drop_leading",
                "min_samples_train": 2,
            },
        )


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


def test_raises_when_group_features_are_missing_in_dataset() -> None:
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
    with pytest.raises(ValueError, match="not fully available"):
        use_case.execute("AAPL", features=["TECHNICAL_FEATURES"], split_config=split_config)


def test_derived_feature_groups_can_be_enabled_via_training_config() -> None:
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
        features=["BASELINE_FEATURES"],
        split_config=split_config,
        training_config={"derived_feature_groups": ["FUNDAMENTAL_DERIVED_FEATURES"]},
    )

    assert trainer.seen_features is not None
    assert "open" in trainer.seen_features
    assert "net_margin" in trainer.seen_features
    assert repo.last_version is not None
    assert repo.last_version.endswith("_BQ")


def test_raises_when_derived_feature_groups_has_invalid_type() -> None:
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
    with pytest.raises(ValueError, match="derived_feature_groups must be a list"):
        use_case.execute(
            "AAPL",
            features=["BASELINE_FEATURES"],
            split_config=split_config,
            training_config={"derived_feature_groups": "FUNDAMENTAL_DERIVED_FEATURES"},
        )


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


def test_quality_gate_rejects_duplicate_timestamps() -> None:
    df = _df_ablation().copy()
    df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(df),
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
    with pytest.raises(ValueError, match="duplicate timestamp"):
        use_case.execute(
            "AAPL",
            features=["BASELINE_FEATURES"],
            split_config=split_config,
            training_config={"quality_gate_require_unique_timestamps": True},
        )


def test_quality_gate_rejects_high_nan_ratio_per_feature() -> None:
    df = _df_ablation().copy()
    df["close"] = [None, None, 12.5]
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(df),
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
    with pytest.raises(ValueError, match="NaN ratio above threshold"):
        use_case.execute(
            "AAPL",
            features=["close"],
            split_config=split_config,
            training_config={"quality_gate_max_nan_ratio_per_feature": 0.5},
        )


def test_quality_gate_rejects_insufficient_temporal_coverage() -> None:
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
    with pytest.raises(ValueError, match="temporal coverage below minimum"):
        use_case.execute(
            "AAPL",
            features=["BASELINE_FEATURES"],
            split_config=split_config,
            training_config={"quality_gate_min_temporal_coverage_days": 800},
        )


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
            "rsi_14": [30.0, 40.0, 50.0],
            "candle_body": [0.5, 0.5, 0.5],
            "macd_signal": [0.01, 0.02, 0.03],
            "ema_100": [10.0, 10.1, 10.2],
            "macd": [0.1, 0.2, 0.3],
            "ema_10": [10.2, 10.3, 10.4],
            "ema_200": [9.8, 9.9, 10.0],
            "ema_50": [10.1, 10.2, 10.3],
            "candle_range": [2.0, 2.0, 2.0],
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



def test_persists_dim_run_identity_fields_when_analytics_repo_is_enabled() -> None:
    trainer = FakeTrainer()
    repo = FakeModelRepo()
    analytics = FakeAnalyticsRunRepo()
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df()),
        model_trainer=trainer,
        model_repository=repo,
        analytics_run_repository=analytics,
    )

    split_config = {
        "train_start": "20240101",
        "train_end": "20240101",
        "val_start": "20240102",
        "val_end": "20240102",
        "test_start": "20250102",
        "test_end": "20250102",
    }
    training_config = {
        "seed": 7,
        "parent_sweep_id": "sweep_x",
        "trial_number": 3,
        "fold": "wf_1",
        "pipeline_version": "0.1",
        "feature_set_name": "B",
    }

    use_case.execute("AAPL", features=["feature_a"], split_config=split_config, training_config=training_config)

    assert analytics.rows is not None
    assert len(analytics.rows) == 1
    row = analytics.rows[0]
    assert row["asset"] == "AAPL"
    assert row["parent_sweep_id"] == "sweep_x"
    assert row["trial_number"] == 3
    assert row["fold"] == "wf_1"
    assert row["seed"] == 7
    assert row["feature_set_name"] == "C"
    assert row["feature_set_hash"]
    assert row["model_version"]
    assert row["run_id"]
    assert analytics.snapshots is not None
    assert analytics.split_refs is None
    assert analytics.fact_configs is not None
    assert len(analytics.fact_configs) == 1
    fact_config = analytics.fact_configs[0]
    assert fact_config["run_id"] == row["run_id"]
    assert fact_config["prediction_mode"] == "quantile"
    assert fact_config["quantile_levels_json"]
    assert fact_config["evaluation_horizons_json"] == "[1]"
    assert analytics.split_metric_rows is not None
    assert {r["split"] for r in analytics.split_metric_rows} == {"train", "val", "test"}
    assert all(r["run_id"] == row["run_id"] for r in analytics.split_metric_rows)
    assert analytics.epoch_rows is not None
    assert len(analytics.epoch_rows) >= 1
    assert analytics.epoch_rows[0]["run_id"] == row["run_id"]
    assert "train_loss" in analytics.epoch_rows[0]
    assert "val_loss" in analytics.epoch_rows[0]
    assert analytics.oos_rows is not None
    assert len(analytics.oos_rows) >= 1
    oos = analytics.oos_rows[0]
    assert oos["run_id"] == row["run_id"]
    assert oos["model_version"] == row["model_version"]
    assert oos["asset"] == "AAPL"
    assert oos["feature_set_name"] == row["feature_set_name"]
    assert "y_true" in oos and "y_pred" in oos
    assert "quantile_p10" in oos and "quantile_p90" in oos
    assert analytics.model_artifacts_rows is not None
    assert len(analytics.model_artifacts_rows) == 1
    mar = analytics.model_artifacts_rows[0]
    assert mar["run_id"] == row["run_id"]
    assert mar["asset"] == "AAPL"
    assert "feature_importance_json" in mar
    assert "attention_summary_json" in mar
    assert analytics.bridge_feature_rows is not None
    assert len(analytics.bridge_feature_rows) >= 1


def test_persists_split_timestamps_when_flag_enabled() -> None:
    trainer = FakeTrainer()
    repo = FakeModelRepo()
    analytics = FakeAnalyticsRunRepo()
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df()),
        model_trainer=trainer,
        model_repository=repo,
        analytics_run_repository=analytics,
    )

    split_config = {
        "train_start": "20240101",
        "train_end": "20240101",
        "val_start": "20240102",
        "val_end": "20240102",
        "test_start": "20250102",
        "test_end": "20250102",
    }
    training_config = {
        "seed": 7,
        "feature_set_name": "B",
        "store_split_timestamps_ref": True,
    }

    use_case.execute(
        "AAPL",
        features=["feature_a"],
        split_config=split_config,
        training_config=training_config,
    )

    assert analytics.split_refs is not None
    assert len(analytics.split_refs) == 3
    assert analytics.rows is not None
    assert analytics.bridge_feature_rows[0]["run_id"] == analytics.rows[0]["run_id"]


def test_persists_failed_status_and_failure_fact_when_training_raises() -> None:
    repo = FakeModelRepo()
    analytics = FakeAnalyticsRunRepo()
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df()),
        model_trainer=FailingTrainer(),
        model_repository=repo,
        analytics_run_repository=analytics,
    )

    split_config = {
        "train_start": "20240101",
        "train_end": "20240101",
        "val_start": "20240102",
        "val_end": "20240102",
        "test_start": "20250102",
        "test_end": "20250102",
    }

    with pytest.raises(RuntimeError, match="boom training"):
        use_case.execute("AAPL", features=["feature_a"], split_config=split_config)

    assert analytics.rows is not None
    assert len(analytics.rows) == 1
    assert analytics.rows[0]["status"] == "failed"
    assert analytics.failures is not None
    assert len(analytics.failures) == 1
    assert analytics.failures[0]["stage"] == "train_execute"
    assert "boom training" in analytics.failures[0]["error_message"]


def test_persist_fact_oos_predictions_keeps_horizon_index_alignment_per_split() -> None:
    analytics = FakeAnalyticsRunRepo()
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df()),
        model_trainer=FakeTrainer(),
        model_repository=FakeModelRepo(),
        analytics_run_repository=analytics,
    )

    split_frames = {
        "val": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2025-01-10", "2025-01-11"], utc=True),
            }
        ),
        "test": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2025-02-10", "2025-02-11"], utc=True),
            }
        ),
    }
    split_predictions = {
        "val": {
            "horizons": [1, 7, 30],
            "y_true_matrix": [[101.0, 107.0, 130.0], [201.0, 207.0, 230.0]],
            "y_pred_matrix": [[111.0, 117.0, 140.0], [211.0, 217.0, 240.0]],
            "quantile_p10_matrix": [[91.0, 97.0, 120.0], [191.0, 197.0, 220.0]],
            "quantile_p50_matrix": [[111.0, 117.0, 140.0], [211.0, 217.0, 240.0]],
            "quantile_p90_matrix": [[131.0, 137.0, 160.0], [231.0, 237.0, 260.0]],
        },
        "test": {
            "horizons": [1, 7, 30],
            "y_true_matrix": [[301.0, 307.0, 330.0], [401.0, 407.0, 430.0]],
            "y_pred_matrix": [[311.0, 317.0, 340.0], [411.0, 417.0, 440.0]],
            "quantile_p10_matrix": [[291.0, 297.0, 320.0], [391.0, 397.0, 420.0]],
            "quantile_p50_matrix": [[311.0, 317.0, 340.0], [411.0, 417.0, 440.0]],
            "quantile_p90_matrix": [[331.0, 337.0, 360.0], [431.0, 437.0, 460.0]],
        },
    }

    use_case._persist_fact_oos_predictions(
        run_id="run_x",
        asset_id="AAPL",
        feature_set_name="B",
        model_version="20260407_010101_B",
        config_signature="sig_x",
        fold_name="wf_1",
        seed=7,
        split_frames=split_frames,
        split_predictions=split_predictions,
    )

    assert analytics.oos_rows is not None
    rows = analytics.oos_rows
    # 2 timestamps x 3 horizons x 2 splits
    assert len(rows) == 12

    val_h7 = [
        r for r in rows
        if r["split"] == "val" and int(r["horizon"]) == 7 and r["timestamp_utc"].startswith("2025-01-10")
    ]
    assert len(val_h7) == 1
    assert val_h7[0]["y_true"] == 107.0
    assert val_h7[0]["y_pred"] == 117.0
    assert val_h7[0]["quantile_p10"] == 97.0
    assert val_h7[0]["quantile_p50"] == 117.0
    assert val_h7[0]["quantile_p90"] == 137.0
    assert val_h7[0]["target_timestamp_utc"].startswith("2025-01-16")

    test_h30 = [
        r for r in rows
        if r["split"] == "test" and int(r["horizon"]) == 30 and r["timestamp_utc"].startswith("2025-02-10")
    ]
    assert len(test_h30) == 1
    assert test_h30[0]["y_true"] == 330.0
    assert test_h30[0]["y_pred"] == 340.0
    assert test_h30[0]["quantile_p10"] == 320.0
    assert test_h30[0]["quantile_p50"] == 340.0
    assert test_h30[0]["quantile_p90"] == 360.0
    assert test_h30[0]["target_timestamp_utc"].startswith("2025-03-11")


def test_persist_fact_oos_predictions_applies_quantile_guardrail_columns() -> None:
    analytics = FakeAnalyticsRunRepo()
    use_case = TrainTFTModelUseCase(
        dataset_repository=FakeDatasetRepository(_df()),
        model_trainer=FakeTrainer(),
        model_repository=FakeModelRepo(),
        analytics_run_repository=analytics,
    )

    split_frames = {
        "test": pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2025-02-10"], utc=True),
            }
        )
    }
    split_predictions = {
        "test": {
            "horizons": [1],
            "y_true_matrix": [[1.0]],
            "y_pred_matrix": [[1.1]],
            "quantile_p10_matrix": [[1.3]],
            "quantile_p50_matrix": [[1.1]],
            "quantile_p90_matrix": [[0.9]],
        }
    }

    use_case._persist_fact_oos_predictions(
        run_id="run_guardrail",
        asset_id="AAPL",
        feature_set_name="B",
        model_version="20260409_101010_B",
        config_signature="sig_guardrail",
        fold_name="wf_1",
        seed=7,
        split_frames=split_frames,
        split_predictions=split_predictions,
    )

    assert analytics.oos_rows is not None
    assert len(analytics.oos_rows) == 1
    row = analytics.oos_rows[0]
    assert row["quantile_p10"] == 1.3
    assert row["quantile_p50"] == 1.1
    assert row["quantile_p90"] == 0.9
    assert row["quantile_p10_post_guardrail"] == 0.9
    assert row["quantile_p50_post_guardrail"] == 1.1
    assert row["quantile_p90_post_guardrail"] == 1.3
    assert row["quantile_guardrail_applied"] == 1
