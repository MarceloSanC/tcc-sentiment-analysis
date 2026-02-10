from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging

import pandas as pd

from src.infrastructure.schemas.tft_dataset_parquet_schema import DEFAULT_TFT_FEATURES
from src.infrastructure.schemas.model_artifact_schema import TFT_SPLIT_DEFAULTS

from src.interfaces.tft_dataset_repository import TFTDatasetRepository
from src.interfaces.model_trainer import ModelTrainer, TrainingResult
from src.interfaces.model_repository import ModelRepository


@dataclass(frozen=True)
class TrainTFTModelResult:
    asset_id: str
    version: str
    metrics: dict[str, float]
    artifacts_dir: str


class TrainTFTModelUseCase:
    def __init__(
        self,
        dataset_repository: TFTDatasetRepository,
        model_trainer: ModelTrainer,
        model_repository: ModelRepository,
    ) -> None:
        self.dataset_repository = dataset_repository
        self.model_trainer = model_trainer
        self.model_repository = model_repository

    @staticmethod
    def _select_features(df: pd.DataFrame, features: list[str] | None) -> list[str]:
        base_exclude = {"timestamp", "asset_id", "target_return"}
        if features:
            missing = [c for c in features if c not in df.columns]
            if missing:
                raise ValueError(f"Requested features not found in dataset: {missing}")
            return features
        available = [c for c in DEFAULT_TFT_FEATURES if c in df.columns]
        if not available:
            raise ValueError(
                "Default feature set is empty or not found in dataset. "
                f"Expected any of: {DEFAULT_TFT_FEATURES}"
            )
        return [c for c in available if c not in base_exclude]

    @staticmethod
    def _parse_yyyymmdd(value: str) -> datetime:
        try:
            return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError(f"Invalid date format: {value}. Expected yyyymmdd.") from exc

    @staticmethod
    def _apply_time_split(
        df: pd.DataFrame,
        *,
        train_start: str,
        train_end: str,
        val_start: str,
        val_end: str,
        test_start: str,
        test_end: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ts = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
        t_start = TrainTFTModelUseCase._parse_yyyymmdd(train_start)
        t_end = TrainTFTModelUseCase._parse_yyyymmdd(train_end)
        v_start = TrainTFTModelUseCase._parse_yyyymmdd(val_start)
        v_end = TrainTFTModelUseCase._parse_yyyymmdd(val_end)
        te_start = TrainTFTModelUseCase._parse_yyyymmdd(test_start)
        te_end = TrainTFTModelUseCase._parse_yyyymmdd(test_end)

        if not (t_start <= t_end < v_start <= v_end < te_start <= te_end):
            logging.getLogger(__name__).warning(
                "Temporal split ranges indicate data leakage",
                extra={
                    "train_start": train_start,
                    "train_end": train_end,
                    "val_start": val_start,
                    "val_end": val_end,
                    "test_start": test_start,
                    "test_end": test_end,
                },
            )
            raise ValueError(
                "Temporal split ranges must be sequential and non-overlapping (data leakage risk)"
            )

        train_df = df[(ts >= t_start) & (ts <= t_end)].copy()
        val_df = df[(ts >= v_start) & (ts <= v_end)].copy()
        test_df = df[(ts >= te_start) & (ts <= te_end)].copy()
        return train_df, val_df, test_df

    def execute(
        self,
        asset_id: str,
        *,
        features: list[str] | None = None,
        training_config: dict | None = None,
        split_config: dict | None = None,
    ) -> TrainTFTModelResult:
        df = self.dataset_repository.load(asset_id)
        if df.empty:
            raise ValueError("Dataset is empty")

        if "time_idx" not in df.columns or "target_return" not in df.columns:
            raise ValueError("Dataset missing required columns for TFT training")

        feature_cols = self._select_features(df, features)

        known_real_cols = [c for c in ["time_idx", "day_of_week", "month"] if c in df.columns]

        split_cfg = dict(TFT_SPLIT_DEFAULTS)
        if split_config:
            split_cfg.update(split_config)
        train_df, val_df, test_df = self._apply_time_split(
            df,
            train_start=split_cfg["train_start"],
            train_end=split_cfg["train_end"],
            val_start=split_cfg["val_start"],
            val_end=split_cfg["val_end"],
            test_start=split_cfg["test_start"],
            test_end=split_cfg["test_end"],
        )

        if train_df.empty or val_df.empty:
            raise ValueError("Train/validation split resulted in empty dataset")

        training_cutoff = int(train_df["time_idx"].max())

        effective_config = dict(training_config or {})
        effective_config["training_cutoff_time_idx"] = training_cutoff

        training_result: TrainingResult = self.model_trainer.train(
            pd.concat([train_df, val_df], ignore_index=True),
            feature_cols=feature_cols,
            target_col="target_return",
            time_idx_col="time_idx",
            group_col="asset_id",
            known_real_cols=known_real_cols,
            config=effective_config,
        )

        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        training_window = {
            "start": df["timestamp"].min().isoformat(),
            "end": df["timestamp"].max().isoformat(),
        }

        artifacts_dir = self.model_repository.save_training_artifacts(
            asset_id,
            version,
            training_result.model,
            metrics=training_result.metrics,
            history=training_result.history,
            features_used=feature_cols,
            training_window=training_window,
            config=effective_config,
        )

        return TrainTFTModelResult(
            asset_id=asset_id,
            version=version,
            metrics=training_result.metrics,
            artifacts_dir=artifacts_dir,
        )
