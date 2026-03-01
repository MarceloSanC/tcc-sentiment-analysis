from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.domain.services.feature_warmup_inspector import FeatureWarmupInspector
from src.infrastructure.schemas.tft_dataset_parquet_schema import (
    BASELINE_FEATURES,
    DEFAULT_TFT_FEATURES,
    FUNDAMENTAL_FEATURES,
    SENTIMENT_FEATURES,
    TECHNICAL_FEATURES,
)
from src.infrastructure.schemas.model_artifact_schema import TFT_SPLIT_DEFAULTS
from src.infrastructure.schemas.model_artifact_schema import TFT_TRAINING_DEFAULTS

from src.interfaces.tft_dataset_repository import TFTDatasetRepository
from src.interfaces.model_trainer import ModelTrainer, TrainingResult
from src.interfaces.model_repository import ModelRepository

logger = logging.getLogger(__name__)


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
    def _resolve_features(
        df: pd.DataFrame, features: list[str] | None
    ) -> tuple[list[str], str]:
        token_groups: dict[str, tuple[str, list[str]]] = {
            "BASELINE_FEATURES": ("B", BASELINE_FEATURES),
            "TECHNICAL_FEATURES": ("T", TECHNICAL_FEATURES),
            "SENTIMENT_FEATURES": ("S", SENTIMENT_FEATURES),
            "FUNDAMENTAL_FEATURES": ("F", FUNDAMENTAL_FEATURES),
            "B": ("B", BASELINE_FEATURES),
            "T": ("T", TECHNICAL_FEATURES),
            "S": ("S", SENTIMENT_FEATURES),
            "F": ("F", FUNDAMENTAL_FEATURES),
        }
        allowed_order = ["B", "T", "S", "F", "C"]

        if features is None:
            selected = TrainTFTModelUseCase._select_features(df, None)
            letters: set[str] = set()
            if any(c in selected for c in BASELINE_FEATURES):
                letters.add("B")
            if any(c in selected for c in TECHNICAL_FEATURES):
                letters.add("T")
            if any(c in selected for c in SENTIMENT_FEATURES):
                letters.add("S")
            if any(c in selected for c in FUNDAMENTAL_FEATURES):
                letters.add("F")
            suffix = "".join([l for l in allowed_order if l in letters]) or "C"
            return selected, suffix

        selected: list[str] = []
        selected_set: set[str] = set()
        letters: set[str] = set()
        custom_requested = False
        missing: list[str] = []

        for raw_token in features:
            token = raw_token.strip()
            upper = token.upper()
            if upper in token_groups:
                letter, cols = token_groups[upper]
                letters.add(letter)
                for col in cols:
                    if col in df.columns and col not in selected_set:
                        selected.append(col)
                        selected_set.add(col)
                continue

            custom_requested = True
            col = token
            if col not in df.columns:
                lower_col = token.lower()
                if lower_col in df.columns:
                    col = lower_col
                else:
                    missing.append(token)
                    continue
            if col not in selected_set:
                selected.append(col)
                selected_set.add(col)

        if missing:
            raise ValueError(f"Requested features not found in dataset: {missing}")
        if not selected:
            raise ValueError("No valid features resolved from provided --features")

        if custom_requested:
            letters.add("C")
        suffix = "".join([l for l in allowed_order if l in letters]) or "C"
        return selected, suffix

    @staticmethod
    def _parse_yyyymmdd(value: str) -> datetime:
        try:
            return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError(f"Invalid date format: {value}. Expected yyyymmdd.") from exc

    @staticmethod
    def _available(df: pd.DataFrame, columns: list[str]) -> list[str]:
        return [c for c in columns if c in df.columns]

    @staticmethod
    def _apply_split_feature_normalization(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        feature_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, StandardScaler]]:
        technical_cols = [c for c in feature_cols if c in TECHNICAL_FEATURES]
        if not technical_cols:
            return train_df, val_df, test_df, {}

        train_norm = train_df.copy()
        val_norm = val_df.copy()
        test_norm = test_df.copy()
        scalers: dict[str, StandardScaler] = {}

        for col in technical_cols:
            train_series = pd.to_numeric(train_norm[col], errors="coerce")
            train_mask = np.isfinite(train_series.to_numpy())
            if not train_mask.any():
                continue

            scaler = StandardScaler()
            scaler.fit(train_series.to_numpy()[train_mask].reshape(-1, 1))
            scalers[col] = scaler

            for df_norm in (train_norm, val_norm, test_norm):
                values = pd.to_numeric(df_norm[col], errors="coerce")
                arr = values.to_numpy(dtype="float64")
                mask = np.isfinite(arr)
                if mask.any():
                    arr[mask] = scaler.transform(arr[mask].reshape(-1, 1)).reshape(-1)
                df_norm[col] = arr

        return train_norm, val_norm, test_norm, scalers

    def _run_ablation(
        self,
        *,
        df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        known_real_cols: list[str],
        split_cfg: dict,
        training_config: dict,
    ) -> list[dict[str, float | str]]:
        baseline = self._available(df, BASELINE_FEATURES)
        technical = self._available(df, TECHNICAL_FEATURES)
        sentiment = self._available(df, SENTIMENT_FEATURES)
        fundamentals = self._available(df, FUNDAMENTAL_FEATURES)
        experiments = {
            "baseline": baseline,
            "baseline_plus_technical": baseline + technical,
            "baseline_plus_sentiment": baseline + sentiment,
            "baseline_plus_fundamentals": baseline + fundamentals,
            "baseline_plus_technical_plus_sentiment_plus_fundamentals": baseline + technical + sentiment + fundamentals,
        }

        results: list[dict[str, float | str]] = []
        for name, feat_cols in experiments.items():
            feat_cols = list(dict.fromkeys(feat_cols))
            if not feat_cols:
                continue
            cfg = dict(training_config)
            cfg["split_config"] = split_cfg
            run = self.model_trainer.train(
                train_df,
                val_df,
                test_df,
                feature_cols=feat_cols,
                target_col="target_return",
                time_idx_col="time_idx",
                group_col="asset_id",
                known_real_cols=known_real_cols,
                config=cfg,
            )
            results.append(
                {
                    "experiment": name,
                    "n_features": float(len(feat_cols)),
                    "train_rmse": float(run.split_metrics.get("train", {}).get("rmse", float("nan"))),
                    "val_rmse": float(run.split_metrics.get("val", {}).get("rmse", float("nan"))),
                    "test_rmse": float(run.split_metrics.get("test", {}).get("rmse", float("nan"))),
                    "train_mae": float(run.split_metrics.get("train", {}).get("mae", float("nan"))),
                    "val_mae": float(run.split_metrics.get("val", {}).get("mae", float("nan"))),
                    "test_mae": float(run.split_metrics.get("test", {}).get("mae", float("nan"))),
                    "train_da": float(run.split_metrics.get("train", {}).get("directional_accuracy", float("nan"))),
                    "val_da": float(run.split_metrics.get("val", {}).get("directional_accuracy", float("nan"))),
                    "test_da": float(run.split_metrics.get("test", {}).get("directional_accuracy", float("nan"))),
                }
            )
        return results

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
        run_ablation: bool = False,
    ) -> TrainTFTModelResult:
        df = self.dataset_repository.load(asset_id)
        if df.empty:
            raise ValueError("Dataset is empty")

        if "time_idx" not in df.columns or "target_return" not in df.columns:
            raise ValueError("Dataset missing required columns for TFT training")

        feature_cols, feature_tag = self._resolve_features(df, features)

        known_real_cols = [c for c in ["time_idx", "day_of_week", "month"] if c in df.columns]

        split_cfg = dict(TFT_SPLIT_DEFAULTS)
        if split_config:
            split_cfg.update(split_config)
        warmup_segments = FeatureWarmupInspector.detect_leading_null_warmups(
            df,
            feature_cols,
            requested_start=split_cfg["train_start"],
            requested_end=split_cfg["test_end"],
        )
        for segment in warmup_segments:
            logger.warning(
                "Warm-up null segment detected. Requested period %s to %s contains %d leading null values for feature '%s' from %s to %s.",
                segment.requested_start,
                segment.requested_end,
                segment.num_null,
                segment.feature_name,
                segment.first_date_warmup,
                segment.last_date_warmup_null,
            )
        train_df, val_df, test_df = self._apply_time_split(
            df,
            train_start=split_cfg["train_start"],
            train_end=split_cfg["train_end"],
            val_start=split_cfg["val_start"],
            val_end=split_cfg["val_end"],
            test_start=split_cfg["test_start"],
            test_end=split_cfg["test_end"],
        )

        if train_df.empty or val_df.empty or test_df.empty:
            raise ValueError("Train/validation/test split resulted in empty dataset")

        train_df, val_df, test_df, split_scalers = self._apply_split_feature_normalization(
            train_df,
            val_df,
            test_df,
            feature_cols=feature_cols,
        )

        trainer_config = self._build_trainer_config(training_config)
        metadata_config = dict(trainer_config)
        metadata_config["split_config"] = split_cfg
        metadata_config["feature_set_tag"] = feature_tag
        metadata_config["feature_tokens"] = features or ["DEFAULT_TFT_FEATURES"]
        metadata_config["split_normalized_technical_features"] = sorted(split_scalers.keys())

        training_result: TrainingResult = self.model_trainer.train(
            train_df,
            val_df,
            test_df,
            feature_cols=feature_cols,
            target_col="target_return",
            time_idx_col="time_idx",
            group_col="asset_id",
            known_real_cols=known_real_cols,
            config=trainer_config,
        )
        ablation_results: list[dict[str, float | str]] = []
        if run_ablation and features is None:
            ablation_results = self._run_ablation(
                df=df,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                known_real_cols=known_real_cols,
                split_cfg=split_cfg,
                training_config=trainer_config,
            )

        version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{feature_tag}"
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
            split_metrics=training_result.split_metrics,
            features_used=feature_cols,
            training_window=training_window,
            split_window=split_cfg,
            config=metadata_config,
            feature_importance=training_result.feature_importance,
            ablation_results=ablation_results,
            checkpoint_path=training_result.checkpoint_path,
            dataset_parameters={
                **(training_result.dataset_parameters or {}),
                "scalers": {
                    **(
                        (training_result.dataset_parameters or {}).get("scalers", {})
                        if isinstance(training_result.dataset_parameters, dict)
                        else {}
                    ),
                    **split_scalers,
                },
            },
        )

        return TrainTFTModelResult(
            asset_id=asset_id,
            version=version,
            metrics=training_result.metrics,
            artifacts_dir=artifacts_dir,
        )
    @staticmethod
    def _build_trainer_config(training_config: dict | None) -> dict:
        config = dict(training_config or {})
        return {k: v for k, v in config.items() if k in TFT_TRAINING_DEFAULTS}
