from __future__ import annotations

import importlib.metadata
import json
import logging
import platform
import subprocess
import traceback

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from src.domain.services.dataset_quality_gate import (
    DatasetQualityGate,
    DatasetQualityGateConfig,
)
from src.domain.services.feature_warmup_inspector import FeatureWarmupInspector
from src.domain.services.multi_horizon_prediction_persister import (
    IncompletePredictionWindowError,
    MultiHorizonPredictionPersister,
    RunContext,
)
from src.domain.services.quantile_guardrail_service import QuantileGuardrailService
from src.infrastructure.schemas.analytics_store_schema import (
    ANALYTICS_SCHEMA_VERSION,
    compute_config_signature,
    compute_dataset_fingerprint,
    compute_feature_set_hash,
    compute_run_id,
    compute_split_fingerprint,
)
from src.infrastructure.schemas.feature_validation_schema import FEATURE_WARMUP_BARS
from src.infrastructure.schemas.model_artifact_schema import (
    TFT_SPLIT_DEFAULTS,
    TFT_TRAINING_DEFAULTS,
)
from src.infrastructure.schemas.tft_dataset_parquet_schema import (
    BASELINE_FEATURES,
    DEFAULT_TFT_FEATURES,
    FUNDAMENTAL_DERIVED_FEATURES,
    FUNDAMENTAL_FEATURES,
    MOMENTUM_LIQUIDITY_FEATURES,
    REGIME_FEATURES,
    SENTIMENT_DYNAMICS_FEATURES,
    SENTIMENT_FEATURES,
    TECHNICAL_FEATURES,
    VOLATILITY_ROBUST_FEATURES,
)
from src.interfaces.analytics_run_repository import AnalyticsRunRepository
from src.interfaces.model_repository import ModelRepository
from src.interfaces.model_trainer import ModelTrainer, TrainingResult
from src.interfaces.tft_dataset_repository import TFTDatasetRepository
from src.utils.path_policy import to_project_relative

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
        analytics_run_repository: AnalyticsRunRepository | None = None,
    ) -> None:
        self.dataset_repository = dataset_repository
        self.model_trainer = model_trainer
        self.model_repository = model_repository
        self.analytics_run_repository = analytics_run_repository

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
        token_groups["MOMENTUM_LIQUIDITY_FEATURES"] = ("D", MOMENTUM_LIQUIDITY_FEATURES)
        token_groups["D"] = ("D", MOMENTUM_LIQUIDITY_FEATURES)
        token_groups["VOLATILITY_ROBUST_FEATURES"] = ("V", VOLATILITY_ROBUST_FEATURES)
        token_groups["V"] = ("V", VOLATILITY_ROBUST_FEATURES)
        token_groups["REGIME_FEATURES"] = ("R", REGIME_FEATURES)
        token_groups["R"] = ("R", REGIME_FEATURES)
        token_groups["SENTIMENT_DYNAMICS_FEATURES"] = ("Y", SENTIMENT_DYNAMICS_FEATURES)
        token_groups["Y"] = ("Y", SENTIMENT_DYNAMICS_FEATURES)
        token_groups["FUNDAMENTAL_DERIVED_FEATURES"] = ("Q", FUNDAMENTAL_DERIVED_FEATURES)
        token_groups["Q"] = ("Q", FUNDAMENTAL_DERIVED_FEATURES)
        allowed_order = ["B", "T", "S", "F", "D", "V", "R", "Y", "Q", "C"]

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
            if any(c in selected for c in MOMENTUM_LIQUIDITY_FEATURES):
                letters.add("D")
            if any(c in selected for c in VOLATILITY_ROBUST_FEATURES):
                letters.add("V")
            if any(c in selected for c in REGIME_FEATURES):
                letters.add("R")
            if any(c in selected for c in SENTIMENT_DYNAMICS_FEATURES):
                letters.add("Y")
            if any(c in selected for c in FUNDAMENTAL_DERIVED_FEATURES):
                letters.add("Q")
            suffix = "".join([letter for letter in allowed_order if letter in letters]) or "C"
            return selected, suffix

        selected: list[str] = []
        selected_set: set[str] = set()
        letters: set[str] = set()
        custom_requested = False
        missing: list[str] = []
        missing_group_columns: dict[str, list[str]] = {}

        for raw_token in features:
            token = raw_token.strip()
            upper = token.upper()
            if upper in token_groups:
                letter, cols = token_groups[upper]
                letters.add(letter)
                group_missing = [col for col in cols if col not in df.columns]
                if group_missing:
                    missing_group_columns[upper] = group_missing
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
        if missing_group_columns:
            details = "; ".join(
                f"{group} missing {cols}"
                for group, cols in sorted(missing_group_columns.items())
            )
            raise ValueError(
                "Requested feature groups are not fully available in dataset: "
                f"{details}"
            )
        if not selected:
            raise ValueError("No valid features resolved from provided --features")

        if custom_requested:
            letters.add("C")
        suffix = "".join([letter for letter in allowed_order if letter in letters]) or "C"
        return selected, suffix

    @staticmethod
    def _parse_yyyymmdd(value: str) -> datetime:
        try:
            return datetime.strptime(value, "%Y%m%d").replace(tzinfo=UTC)
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
                # Ensure a writable buffer before in-place scaling.
                arr = np.array(values.to_numpy(dtype="float64"), copy=True)
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

    @staticmethod
    def _apply_warmup_policy_to_split(
        df: pd.DataFrame,
        *,
        feature_cols: list[str],
        split_cfg: dict,
        warmup_policy: str,
    ) -> tuple[dict, dict]:
        policy = str(warmup_policy).strip().lower()
        if policy not in {"strict_fail", "drop_leading"}:
            raise ValueError("warmup_policy must be one of ['strict_fail', 'drop_leading']")

        warmup_features = [
            c for c in feature_cols if int(FEATURE_WARMUP_BARS.get(c, 0)) > 0
        ]
        required_warmup_count = max((int(FEATURE_WARMUP_BARS.get(c, 0)) for c in warmup_features), default=0)
        if not warmup_features:
            return split_cfg, {
                "warmup_policy": policy,
                "required_warmup_count": required_warmup_count,
                "warmup_applied": False,
                "warmup_features": [],
                "effective_train_start": split_cfg["train_start"],
            }

        segments = FeatureWarmupInspector.detect_leading_null_warmups(
            df,
            warmup_features,
            requested_start=split_cfg["train_start"],
            requested_end=split_cfg["test_end"],
        )
        if not segments:
            return split_cfg, {
                "warmup_policy": policy,
                "required_warmup_count": required_warmup_count,
                "warmup_applied": False,
                "warmup_features": warmup_features,
                "effective_train_start": split_cfg["train_start"],
            }

        suggested = FeatureWarmupInspector.suggest_trimmed_start_yyyymmdd(
            df,
            warmup_features,
            requested_start=split_cfg["train_start"],
            requested_end=split_cfg["test_end"],
        )
        longest_warmup = max(
            (int(FEATURE_WARMUP_BARS.get(c, 0)) for c in warmup_features),
            default=0,
        )
        longest_warmup_features = sorted(
            [c for c in warmup_features if int(FEATURE_WARMUP_BARS.get(c, 0)) == longest_warmup]
        )

        if policy == "strict_fail":
            detail = ", ".join(f"{s.feature_name}:{s.num_null}" for s in segments)
            longest_info = (
                f"longest_warmup={longest_warmup}, longest_warmup_features={longest_warmup_features}"
            )
            first_valid = (
                f"first_valid_train_start={suggested}"
                if suggested is not None
                else "first_valid_train_start=unavailable"
            )
            raise ValueError(
                "Warm-up validation failed for selected features in requested training period "
                f"(policy=strict_fail). leading_nulls={detail}; {longest_info}; {first_valid}. "
                "Use warmup_policy=drop_leading to auto-adjust train_start."
            )

        # drop_leading
        if suggested is None:
            raise ValueError(
                "Warm-up validation failed: warm-up consumes the entire requested window."
            )
        new_split = dict(split_cfg)
        new_split["train_start"] = suggested
        return new_split, {
            "warmup_policy": policy,
            "required_warmup_count": required_warmup_count,
            "warmup_applied": suggested != split_cfg["train_start"],
            "warmup_features": warmup_features,
            "effective_train_start": suggested,
        }

    @staticmethod
    def _validate_min_split_samples(
        *,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        trainer_config: dict,
    ) -> None:
        min_train = int(trainer_config.get("min_samples_train", 1))
        min_val = int(trainer_config.get("min_samples_val", 1))
        min_test = int(trainer_config.get("min_samples_test", 1))

        counts = {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        }
        required = {
            "train": min_train,
            "val": min_val,
            "test": min_test,
        }
        insufficient = [
            f"{split}={counts[split]}<min={required[split]}"
            for split in ("train", "val", "test")
            if counts[split] < required[split]
        ]
        if insufficient:
            raise ValueError(
                "Train/validation/test split has insufficient samples after warmup/split validation: "
                + ", ".join(insufficient)
            )

    @staticmethod
    def _run_pretrain_quality_gate(
        *,
        df: pd.DataFrame,
        feature_cols: list[str],
        trainer_config: dict,
    ) -> None:
        cfg = DatasetQualityGateConfig(
            max_nan_ratio_per_feature=float(
                trainer_config.get("quality_gate_max_nan_ratio_per_feature", 1.0)
            ),
            min_temporal_coverage_days=int(
                trainer_config.get("quality_gate_min_temporal_coverage_days", 1)
            ),
            require_unique_timestamps=bool(
                trainer_config.get("quality_gate_require_unique_timestamps", True)
            ),
            require_monotonic_timestamps=bool(
                trainer_config.get("quality_gate_require_monotonic_timestamps", True)
            ),
        )
        DatasetQualityGate.validate(
            df=df,
            feature_cols=feature_cols,
            config=cfg,
            context="pre-train",
            warmup_counts=FEATURE_WARMUP_BARS,
        )

    @staticmethod
    def _collect_git_commit() -> str:
        try:
            out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            return out
        except Exception:
            return "unknown"

    @staticmethod
    def _collect_library_versions() -> dict[str, str]:
        packages = ["torch", "lightning", "pytorch-forecasting", "pandas"]
        out: dict[str, str] = {}
        for pkg in packages:
            try:
                out[pkg] = importlib.metadata.version(pkg)
            except Exception:
                out[pkg] = "unknown"
        return out

    @staticmethod
    def _collect_hardware_info() -> dict[str, str | int | float]:
        info: dict[str, str | int | float] = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        }
        try:
            import torch

            info["gpu_available"] = bool(torch.cuda.is_available())
            info["gpu_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
            if torch.cuda.is_available():
                info["gpu_name"] = str(torch.cuda.get_device_name(0))
        except Exception:
            info["gpu_available"] = False
            info["gpu_count"] = 0
        return info

    def _persist_dim_run(
        self,
        *,
        asset_id: str,
        version: str,
        artifacts_dir: str,
        trainer_config: dict,
        feature_cols: list[str],
        split_signature: str,
        config_signature: str,
        feature_set_hash: str,
        created_at_utc: str,
        status: str = "ok",
    ) -> tuple[str | None, dict | None]:
        if self.analytics_run_repository is None:
            return None, None

        parent_sweep_id = trainer_config.get("parent_sweep_id")
        trial_number = trainer_config.get("trial_number")
        fold_name = trainer_config.get("fold")
        seed = trainer_config.get("seed")
        pipeline_version = str(trainer_config.get("pipeline_version", "0.1"))

        run_id = compute_run_id(
            asset=asset_id,
            feature_set_hash=feature_set_hash,
            trial_number=trial_number if isinstance(trial_number, int) else None,
            fold=str(fold_name) if fold_name is not None else None,
            seed=seed if isinstance(seed, int) else None,
            model_version=version,
            config_signature=config_signature,
            split_signature=split_signature,
            pipeline_version=pipeline_version,
        )

        version_dir = Path(artifacts_dir)
        final_model = version_dir / "model_state.pt"
        best_ckpt = version_dir / "checkpoints" / "best.ckpt"

        row = {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": run_id,
            "execution_id": None,
            "parent_sweep_id": str(parent_sweep_id) if parent_sweep_id is not None else None,
            "trial_number": int(trial_number) if isinstance(trial_number, int) else None,
            "fold": str(fold_name) if fold_name is not None else None,
            "seed": int(seed) if isinstance(seed, int) else None,
            "asset": str(asset_id),
            "feature_set_name": str(trainer_config.get("feature_set_name", trainer_config.get("feature_set_tag", "CUSTOM"))),
            "feature_set_hash": feature_set_hash,
            "feature_list_ordered_json": pd.Series(feature_cols).to_json(orient="values"),
            "config_signature": config_signature,
            "split_fingerprint": split_signature,
            "model_version": version,
            "checkpoint_path_final": to_project_relative(final_model),
            "checkpoint_path_best": to_project_relative(best_ckpt),
            "git_commit": self._collect_git_commit(),
            "pipeline_version": pipeline_version,
            "library_versions_json": pd.Series(self._collect_library_versions()).to_json(),
            "hardware_info_json": pd.Series(self._collect_hardware_info()).to_json(),
            "status": str(status),
            "duration_total_seconds": float(trainer_config.get("duration_total_seconds", 0.0) or 0.0),
            "eta_recorded_seconds": float(trainer_config.get("eta_recorded_seconds", 0.0) or 0.0),
            "retries": int(trainer_config.get("retries", 0) or 0),
            "created_at_utc": created_at_utc,
        }
        self.analytics_run_repository.upsert_dim_run(row)
        return run_id, row

    def _persist_run_snapshot(
        self,
        *,
        run_id: str,
        asset_id: str,
        parent_sweep_id: str | None,
        split_cfg: dict,
        warmup_meta: dict,
        df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        split_signature: str,
        store_split_timestamps_ref: bool,
        overwrite: bool = False,
    ) -> None:
        if self.analytics_run_repository is None:
            return

        ts_all = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        ts_train = pd.to_datetime(train_df["timestamp"], utc=True, errors="coerce")
        ts_val = pd.to_datetime(val_df["timestamp"], utc=True, errors="coerce")
        ts_test = pd.to_datetime(test_df["timestamp"], utc=True, errors="coerce")

        fingerprint = compute_dataset_fingerprint(
            asset=asset_id,
            timestamp_min=str(ts_all.min().isoformat()),
            timestamp_max=str(ts_all.max().isoformat()),
            row_count=int(len(df)),
            close_sum=float(pd.to_numeric(df.get("close"), errors="coerce").sum() if "close" in df.columns else 0.0),
            volume_sum=float(pd.to_numeric(df.get("volume"), errors="coerce").sum() if "volume" in df.columns else 0.0),
            parquet_file_hash=sha256(("|".join(str(x) for x in ts_all.astype(str).tolist())).encode("utf-8")).hexdigest(),
        )

        row = {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": run_id,
            "asset": asset_id,
            "parent_sweep_id": parent_sweep_id,
            "dataset_start_utc": str(ts_all.min().isoformat()),
            "dataset_end_utc": str(ts_all.max().isoformat()),
            "train_start_utc": str(ts_train.min().isoformat()),
            "train_end_utc": str(ts_train.max().isoformat()),
            "val_start_utc": str(ts_val.min().isoformat()),
            "val_end_utc": str(ts_val.max().isoformat()),
            "test_start_utc": str(ts_test.min().isoformat()),
            "test_end_utc": str(ts_test.max().isoformat()),
            "warmup_policy": str(warmup_meta.get("warmup_policy", split_cfg.get("warmup_policy", "strict_fail"))),
            "required_warmup_count": int(warmup_meta.get("required_warmup_count", 0) or 0),
            "warmup_applied": str(bool(warmup_meta.get("warmup_applied", False))).lower(),
            "effective_train_start_utc": str(warmup_meta.get("effective_train_start") or split_cfg.get("train_start")),
            "n_samples_train": int(len(train_df)),
            "n_samples_val": int(len(val_df)),
            "n_samples_test": int(len(test_df)),
            "dataset_fingerprint": fingerprint,
            "split_fingerprint": split_signature,
        }
        self.analytics_run_repository.append_fact_run_snapshot(row, overwrite=overwrite)

        if store_split_timestamps_ref:
            split_rows = []
            for split_name, series in (("train", ts_train), ("val", ts_val), ("test", ts_test)):
                values = [str(x.isoformat()) for x in series]
                compact_json = pd.Series(values).to_json(orient="values")
                split_rows.append(
                    {
                        "schema_version": ANALYTICS_SCHEMA_VERSION,
                        "run_id": run_id,
                        "split": split_name,
                        "artifact_path": None,
                        "artifact_hash": sha256(compact_json.encode("utf-8")).hexdigest(),
                        "timestamps_compact_json": compact_json,
                    }
                )
            self.analytics_run_repository.append_fact_split_timestamps_ref(
                split_rows,
                overwrite=overwrite,
            )

    @staticmethod
    def _safe_json_dumps(value: object) -> str:
        def _normalize(v: object) -> object:
            if v is None or isinstance(v, (str, int, float, bool)):
                return v
            if isinstance(v, (datetime, pd.Timestamp)):
                return str(v)
            if isinstance(v, np.generic):
                return v.item()
            if isinstance(v, dict):
                return {str(k): _normalize(val) for k, val in v.items()}
            if isinstance(v, (list, tuple, set)):
                return [_normalize(item) for item in v]
            if hasattr(v, 'get_params') and callable(getattr(v, 'get_params')):
                try:
                    params = v.get_params(deep=False)
                except Exception:
                    params = {}
                return {
                    '__type__': type(v).__name__,
                    'params': _normalize(params),
                }
            return str(v)

        return json.dumps(_normalize(value), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _extract_loss_metadata(model: object, prediction_mode: str) -> tuple[str, list[float]]:
        loss = getattr(model, 'loss', None)
        loss_name = type(loss).__name__ if loss is not None else 'unknown'

        quantile_levels: list[float] = []
        raw_quantiles = getattr(loss, 'quantiles', None)
        if isinstance(raw_quantiles, (list, tuple)):
            for q in raw_quantiles:
                try:
                    quantile_levels.append(float(q))
                except Exception:
                    continue

        mode = str(prediction_mode).strip().lower()
        if mode == 'quantile' and not quantile_levels:
            quantile_levels = [0.1, 0.5, 0.9]
        return loss_name, quantile_levels

    @staticmethod
    def _infer_scaler_type(dataset_parameters: dict | None) -> str:
        if not isinstance(dataset_parameters, dict):
            return 'unknown'
        scalers = dataset_parameters.get('scalers')
        if not isinstance(scalers, dict) or not scalers:
            return 'none'
        scaler_names = sorted({type(s).__name__ for s in scalers.values() if s is not None})
        if not scaler_names:
            return 'none'
        if len(scaler_names) == 1:
            return scaler_names[0]
        return 'mixed:' + ','.join(scaler_names)

    def _persist_fact_oos_predictions(
        self,
        *,
        run_id: str,
        asset_id: str,
        feature_set_name: str,
        model_version: str,
        config_signature: str,
        fold_name: str | None,
        seed: int | None,
        max_encoder_length: int,
        split_frames: dict[str, pd.DataFrame],
        split_predictions: dict[str, dict[str, list[float]]],
        overwrite: bool = False,
    ) -> None:
        if self.analytics_run_repository is None:
            return
        if not split_predictions:
            return

        # Per ADR-0003 Opcao (a): sample i (from pytorch_forecasting dataloader)
        # has its encoder_end at split_df row (max_encoder_length - 1) + i,
        # which is the canonical decision_day for that sample.
        decision_start_offset = max(int(max_encoder_length) - 1, 0)

        rows: list[dict[str, object]] = []
        for split_name, pred in split_predictions.items():
            if split_name not in split_frames:
                continue
            split_df = split_frames[split_name].copy().sort_values("timestamp").reset_index(drop=True)
            if split_df.empty:
                continue
            dataset_timestamps = pd.to_datetime(split_df["timestamp"], utc=True).tolist()

            run_context = RunContext(
                schema_version=ANALYTICS_SCHEMA_VERSION,
                run_id=run_id,
                model_version=str(model_version),
                asset=str(asset_id),
                feature_set_name=str(feature_set_name),
                config_signature=str(config_signature),
                split=str(split_name),
                fold=str(fold_name) if fold_name is not None else "none",
                seed=int(seed) if seed is not None else 0,
            )

            if "y_true_matrix" in pred and "y_pred_matrix" in pred:
                horizons = [int(v) for v in pred.get("horizons", [])]
                y_true_m = pred.get("y_true_matrix", [])
                y_pred_m = pred.get("y_pred_matrix", [])
                q10_m = pred.get("quantile_p10_matrix", [])
                q50_m = pred.get("quantile_p50_matrix", [])
                q90_m = pred.get("quantile_p90_matrix", [])

                n = min(len(y_true_m), len(y_pred_m), len(q10_m), len(q50_m), len(q90_m))
                if n == 0 or not horizons:
                    continue

                for i in range(n):
                    decision_idx = decision_start_offset + i
                    if decision_idx >= len(dataset_timestamps):
                        break
                    for h_idx, h in enumerate(horizons):
                        if (
                            h_idx >= len(y_true_m[i]) or h_idx >= len(y_pred_m[i]) or
                            h_idx >= len(q10_m[i]) or h_idx >= len(q50_m[i]) or h_idx >= len(q90_m[i])
                        ):
                            continue
                        y_true = float(y_true_m[i][h_idx])
                        y_pred = float(y_pred_m[i][h_idx])
                        q10 = float(q10_m[i][h_idx])
                        q50 = float(q50_m[i][h_idx])
                        q90 = float(q90_m[i][h_idx])
                        guardrail = QuantileGuardrailService.enforce_monotonic_triplet(q10, q50, q90)
                        try:
                            record = MultiHorizonPredictionPersister.build_record(
                                decision_idx=decision_idx,
                                h=int(h),
                                y_true=y_true,
                                y_pred=y_pred,
                                q10=q10,
                                q50=q50,
                                q90=q90,
                                q10_post=guardrail.p10_post,
                                q50_post=guardrail.p50_post,
                                q90_post=guardrail.p90_post,
                                guardrail_applied=bool(guardrail.applied),
                                dataset_timestamps=dataset_timestamps,
                                run_context=run_context,
                            )
                        except IncompletePredictionWindowError:
                            continue
                        rows.append(record.to_dict())
                continue

            # Backward-compatible path for legacy 1-horizon details.
            y_true_list = [float(v) for v in pred.get("y_true", [])]
            y_pred_list = [float(v) for v in pred.get("y_pred", [])]
            q10_list = [float(v) for v in pred.get("quantile_p10", [])]
            q50_list = [float(v) for v in pred.get("quantile_p50", [])]
            q90_list = [float(v) for v in pred.get("quantile_p90", [])]
            h_list = [int(v) for v in pred.get("horizon", [])]
            n = min(len(y_true_list), len(y_pred_list), len(q10_list), len(q50_list), len(q90_list), len(h_list))
            if n == 0:
                continue
            for i in range(n):
                decision_idx = decision_start_offset + i
                if decision_idx >= len(dataset_timestamps):
                    break
                guardrail = QuantileGuardrailService.enforce_monotonic_triplet(
                    q10_list[i], q50_list[i], q90_list[i]
                )
                try:
                    record = MultiHorizonPredictionPersister.build_record(
                        decision_idx=decision_idx,
                        h=int(h_list[i]),
                        y_true=y_true_list[i],
                        y_pred=y_pred_list[i],
                        q10=q10_list[i],
                        q50=q50_list[i],
                        q90=q90_list[i],
                        q10_post=guardrail.p10_post,
                        q50_post=guardrail.p50_post,
                        q90_post=guardrail.p90_post,
                        guardrail_applied=bool(guardrail.applied),
                        dataset_timestamps=dataset_timestamps,
                        run_context=run_context,
                    )
                except IncompletePredictionWindowError:
                    continue
                rows.append(record.to_dict())

        if rows:
            self.analytics_run_repository.append_fact_oos_predictions(
                rows,
                overwrite=overwrite,
            )

    def _persist_fact_epoch_metrics(
        self,
        *,
        run_id: str,
        asset_id: str,
        parent_sweep_id: str | None,
        fold_name: str | None,
        history: list[dict[str, float]],
        overwrite: bool = False,
    ) -> None:
        if self.analytics_run_repository is None:
            return
        if not history:
            return

        rows: list[dict[str, object]] = []
        for idx, h in enumerate(history):
            train_loss = h.get("train_loss")
            val_loss = h.get("val_loss")
            if train_loss is None or val_loss is None:
                continue

            epoch_value = int(h.get("epoch", idx))
            row = {
                "schema_version": ANALYTICS_SCHEMA_VERSION,
                "run_id": run_id,
                "asset": str(asset_id),
                "parent_sweep_id": str(parent_sweep_id) if parent_sweep_id is not None else None,
                "fold": str(fold_name) if fold_name is not None else "none",
                "epoch": int(epoch_value),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
            if "epoch_time_seconds" in h:
                row["epoch_time_seconds"] = float(h["epoch_time_seconds"])
            if "best_epoch" in h:
                row["best_epoch"] = int(h["best_epoch"])
            if "stopped_epoch" in h:
                row["stopped_epoch"] = int(h["stopped_epoch"])
            if "early_stop_reason" in h:
                row["early_stop_reason"] = str(h["early_stop_reason"])
            rows.append(row)

        if rows:
            self.analytics_run_repository.append_fact_epoch_metrics(
                rows,
                overwrite=overwrite,
            )

    def _persist_fact_split_metrics(
        self,
        *,
        run_id: str,
        asset_id: str,
        parent_sweep_id: str | None,
        split_metrics: dict[str, dict[str, float]],
        split_counts: dict[str, int],
        overwrite: bool = False,
    ) -> None:
        if self.analytics_run_repository is None:
            return
        if not split_metrics:
            return

        rows: list[dict[str, object]] = []
        for split_name, metrics in split_metrics.items():
            rmse = metrics.get("rmse")
            mae = metrics.get("mae")
            da = metrics.get("directional_accuracy")
            if rmse is None or mae is None or da is None:
                continue
            rows.append(
                {
                    "schema_version": ANALYTICS_SCHEMA_VERSION,
                    "run_id": run_id,
                    "asset": str(asset_id),
                    "parent_sweep_id": str(parent_sweep_id) if parent_sweep_id is not None else None,
                    "split": str(split_name),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "mape": float(metrics.get("mape", 0.0)),
                    "smape": float(metrics.get("smape", 0.0)),
                    "directional_accuracy": float(da),
                    "n_samples": int(split_counts.get(split_name, 0)),
                }
            )

        if rows:
            self.analytics_run_repository.append_fact_split_metrics(
                rows,
                overwrite=overwrite,
            )

    def _persist_bridge_run_features(
        self,
        *,
        run_id: str,
        feature_cols: list[str],
        overwrite: bool = False,
    ) -> None:
        if self.analytics_run_repository is None:
            return
        if not feature_cols:
            return
        rows = [
            {
                "schema_version": ANALYTICS_SCHEMA_VERSION,
                "run_id": run_id,
                "feature_order": int(idx),
                "feature_name": str(name),
            }
            for idx, name in enumerate(feature_cols)
        ]
        self.analytics_run_repository.append_bridge_run_features(rows, overwrite=overwrite)

    def _persist_fact_config(
        self,
        *,
        run_id: str,
        asset_id: str,
        parent_sweep_id: str | None,
        trainer_config: dict,
        training_result: TrainingResult,
        dataset_parameters: dict | None,
        overwrite: bool = False,
    ) -> None:
        if self.analytics_run_repository is None:
            return

        prediction_mode = str(trainer_config.get('prediction_mode', 'quantile')).strip().lower()
        loss_name, quantile_levels = self._extract_loss_metadata(training_result.model, prediction_mode)

        row = {
            'schema_version': ANALYTICS_SCHEMA_VERSION,
            'run_id': run_id,
            'asset': str(asset_id),
            'parent_sweep_id': str(parent_sweep_id) if parent_sweep_id is not None else None,
            'prediction_mode': prediction_mode,
            'loss_name': loss_name,
            'quantile_levels_json': self._safe_json_dumps(quantile_levels),
            'evaluation_horizons_json': self._safe_json_dumps(
                trainer_config.get('evaluation_horizons', TFT_TRAINING_DEFAULTS.get('evaluation_horizons', [1]))
            ),
            'max_encoder_length': int(trainer_config.get('max_encoder_length', TFT_TRAINING_DEFAULTS['max_encoder_length'])),
            'max_prediction_length': int(trainer_config.get('max_prediction_length', TFT_TRAINING_DEFAULTS['max_prediction_length'])),
            'batch_size': int(trainer_config.get('batch_size', TFT_TRAINING_DEFAULTS['batch_size'])),
            'max_epochs': int(trainer_config.get('max_epochs', TFT_TRAINING_DEFAULTS['max_epochs'])),
            'learning_rate': float(trainer_config.get('learning_rate', TFT_TRAINING_DEFAULTS['learning_rate'])),
            'hidden_size': int(trainer_config.get('hidden_size', TFT_TRAINING_DEFAULTS['hidden_size'])),
            'attention_head_size': int(trainer_config.get('attention_head_size', TFT_TRAINING_DEFAULTS['attention_head_size'])),
            'dropout': float(trainer_config.get('dropout', TFT_TRAINING_DEFAULTS['dropout'])),
            'hidden_continuous_size': int(trainer_config.get('hidden_continuous_size', TFT_TRAINING_DEFAULTS['hidden_continuous_size'])),
            'early_stopping_patience': int(trainer_config.get('early_stopping_patience', TFT_TRAINING_DEFAULTS['early_stopping_patience'])),
            'early_stopping_min_delta': float(trainer_config.get('early_stopping_min_delta', TFT_TRAINING_DEFAULTS['early_stopping_min_delta'])),
            'scaler_type': self._infer_scaler_type(dataset_parameters),
            'training_config_json': self._safe_json_dumps(trainer_config),
            'dataset_parameters_json': self._safe_json_dumps(dataset_parameters or {}),
            'search_space_json': self._safe_json_dumps(trainer_config.get('search_space', {})),
            'objective_name': str(trainer_config.get('objective_metric', trainer_config.get('objective_name', 'val_loss'))),
            'objective_direction': str(trainer_config.get('objective_direction', 'minimize')),
        }
        self.analytics_run_repository.append_fact_config(row, overwrite=overwrite)

    def _persist_fact_model_artifacts(
        self,
        *,
        run_id: str,
        asset_id: str,
        version: str,
        artifacts_dir: str,
        training_result: TrainingResult,
        overwrite: bool = False,
    ) -> None:
        if self.analytics_run_repository is None:
            return

        version_dir = Path(artifacts_dir)
        checkpoint_final = version_dir / "model_state.pt"
        checkpoint_best = version_dir / "checkpoints" / "best.ckpt"
        config_path = version_dir / "config.json"
        scaler_path = version_dir / "scalers.pkl"
        encoder_path = version_dir / "dataset_parameters.pkl"
        history_path = version_dir / "history.csv"
        metrics_path = version_dir / "metrics.json"
        split_metrics_path = version_dir / "split_metrics.json"
        metadata_path = version_dir / "metadata.json"
        loss_curve_path = version_dir / "plots" / "loss_curve.png"

        fi_payload = training_result.feature_importance or []
        attention_summary = {
            "available": False,
            "source": "training_pipeline",
            "reason": "attention tensors are not exported by trainer yet",
        }
        logs_refs = {
            "history_csv": to_project_relative(history_path) if history_path.exists() else None,
            "metrics_json": to_project_relative(metrics_path) if metrics_path.exists() else None,
            "split_metrics_json": to_project_relative(split_metrics_path) if split_metrics_path.exists() else None,
            "metadata_json": to_project_relative(metadata_path) if metadata_path.exists() else None,
            "loss_curve_png": to_project_relative(loss_curve_path) if loss_curve_path.exists() else None,
        }
        if metadata_path.exists():
            metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if not isinstance(metadata_payload, dict):
                raise ValueError(f"metadata.json root must be an object: {metadata_path}")
            metadata_payload["training_run_id"] = run_id
            metadata_path.write_text(
                json.dumps(metadata_payload, indent=2),
                encoding="utf-8",
            )

        row = {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": run_id,
            "training_run_id": run_id,
            "asset": str(asset_id),
            "model_version": str(version),
            "checkpoint_path_final": to_project_relative(checkpoint_final),
            "checkpoint_path_best": (
                to_project_relative(checkpoint_best)
                if checkpoint_best.exists()
                else None
            ),
            "config_path": to_project_relative(config_path),
            "scaler_path": (
                to_project_relative(scaler_path) if scaler_path.exists() else None
            ),
            "encoder_path": (
                to_project_relative(encoder_path) if encoder_path.exists() else None
            ),
            "feature_importance_json": self._safe_json_dumps(fi_payload),
            "attention_summary_json": self._safe_json_dumps(attention_summary),
            "logs_ref_json": self._safe_json_dumps(logs_refs),
        }
        self.analytics_run_repository.append_fact_model_artifacts(row, overwrite=overwrite)

    def _append_fact_failure(
        self,
        *,
        run_id: str,
        asset_id: str,
        trainer_config: dict,
        stage: str,
        exc: Exception,
        overwrite: bool = False,
    ) -> None:
        if self.analytics_run_repository is None:
            return
        tb_text = traceback.format_exc()
        if not tb_text or tb_text == "NoneType: None\n":
            tb_text = str(exc)
        tb_excerpt = tb_text[-4000:]
        row = {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": run_id,
            "execution_id": None,
            "asset": str(asset_id),
            "failed_at_utc": datetime.now(UTC).isoformat(),
            "stage": str(stage),
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:1000],
            "trace_hash": sha256(tb_text.encode("utf-8")).hexdigest(),
            "traceback_excerpt": tb_excerpt,
            "entrypoint": str(trainer_config.get("entrypoint", "src.main_train_tft")),
            "cmdline": str(trainer_config.get("cmdline", "")),
            "stdout_truncated": str(trainer_config.get("stdout_truncated", ""))[:2000],
            "stderr_truncated": str(trainer_config.get("stderr_truncated", tb_excerpt))[:2000],
        }
        self.analytics_run_repository.append_fact_failures(row, overwrite=overwrite)

    def execute(
        self,
        asset_id: str,
        *,
        features: list[str] | None = None,
        training_config: dict | None = None,
        split_config: dict | None = None,
        run_ablation: bool = False,
        overwrite_on_collision: bool = False,
    ) -> TrainTFTModelResult:
        run_started_at = datetime.now(UTC)

        df = self.dataset_repository.load(asset_id)
        if df.empty:
            raise ValueError("Dataset is empty")

        if "time_idx" not in df.columns or "target_return" not in df.columns:
            raise ValueError("Dataset missing required columns for TFT training")

        trainer_config = self._build_trainer_config(training_config)
        derived_group_tokens = trainer_config.get("derived_feature_groups")
        requested_features = [] if features is None else list(features)
        if isinstance(derived_group_tokens, list):
            for token in derived_group_tokens:
                parsed = str(token).strip()
                if parsed and parsed not in requested_features:
                    requested_features.append(parsed)
        elif derived_group_tokens is not None:
            raise ValueError(
                "training_config.derived_feature_groups must be a list of feature group tokens"
            )

        feature_inputs = None if features is None and not requested_features else requested_features
        feature_cols, feature_tag = self._resolve_features(df, feature_inputs)
        feature_set_hash = compute_feature_set_hash(feature_cols)
        version = datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + f"_{feature_tag}"
        created_at_utc = run_started_at.isoformat()

        trainer_config.setdefault("prediction_mode", str(TFT_TRAINING_DEFAULTS.get("prediction_mode", "quantile")))
        trainer_config.setdefault("quantile_levels", TFT_TRAINING_DEFAULTS.get("quantile_levels", [0.1, 0.5, 0.9]))
        trainer_config.setdefault("evaluation_horizons", TFT_TRAINING_DEFAULTS.get("evaluation_horizons", [1, 7, 30]))
        max_prediction_length = int(
            trainer_config.get(
                "max_prediction_length",
                TFT_TRAINING_DEFAULTS.get("max_prediction_length", 1),
            )
        )
        raw_horizons = trainer_config.get("evaluation_horizons", [1])
        if isinstance(raw_horizons, list) and raw_horizons:
            normalized_horizons = sorted({max(1, int(h)) for h in raw_horizons})
        else:
            normalized_horizons = [1]
        trainer_config["evaluation_horizons"] = [
            h for h in normalized_horizons if h <= max_prediction_length
        ] or [1]

        self._run_pretrain_quality_gate(
            df=df,
            feature_cols=feature_cols,
            trainer_config=trainer_config,
        )

        known_real_cols = [c for c in ["time_idx", "day_of_week", "month"] if c in df.columns]
        warmup_policy = str(trainer_config.get("warmup_policy", "strict_fail")).strip().lower()

        split_cfg = dict(TFT_SPLIT_DEFAULTS)
        if split_config:
            split_cfg.update(split_config)
        split_cfg, warmup_meta = self._apply_warmup_policy_to_split(
            df,
            feature_cols=feature_cols,
            split_cfg=split_cfg,
            warmup_policy=warmup_policy,
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
        self._validate_min_split_samples(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            trainer_config=trainer_config,
        )

        split_signature = compute_split_fingerprint(
            train_timestamps=[str(ts.isoformat()) for ts in pd.to_datetime(train_df["timestamp"], utc=True)],
            val_timestamps=[str(ts.isoformat()) for ts in pd.to_datetime(val_df["timestamp"], utc=True)],
            test_timestamps=[str(ts.isoformat()) for ts in pd.to_datetime(test_df["timestamp"], utc=True)],
        )
        config_signature = compute_config_signature(trainer_config)

        train_df, val_df, test_df, split_scalers = self._apply_split_feature_normalization(
            train_df,
            val_df,
            test_df,
            feature_cols=feature_cols,
        )

        metadata_config = dict(trainer_config)
        metadata_config["split_config"] = split_cfg
        metadata_config["feature_set_tag"] = feature_tag
        metadata_config["feature_set_name"] = feature_tag
        metadata_config["feature_tokens"] = feature_inputs or ["DEFAULT_TFT_FEATURES"]
        metadata_config["feature_list_ordered"] = feature_cols
        metadata_config["feature_set_hash"] = feature_set_hash
        metadata_config["split_normalized_technical_features"] = sorted(split_scalers.keys())
        metadata_config["warmup_policy"] = warmup_meta["warmup_policy"]
        metadata_config["required_warmup_count"] = warmup_meta["required_warmup_count"]
        metadata_config["warmup_applied"] = warmup_meta["warmup_applied"]
        metadata_config["warmup_features"] = warmup_meta["warmup_features"]
        metadata_config["effective_train_start"] = warmup_meta["effective_train_start"]

        artifacts_dir = to_project_relative(Path("data/models") / asset_id / "failed_runs" / version)
        persisted_run_id: str | None = None
        persisted_dim_row: dict | None = None

        try:
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

            training_window = {
                "start": df["timestamp"].min().isoformat(),
                "end": df["timestamp"].max().isoformat(),
            }

            dataset_parameters_payload = {
                **(training_result.dataset_parameters or {}),
                "scalers": {
                    **(
                        (training_result.dataset_parameters or {}).get("scalers", {})
                        if isinstance(training_result.dataset_parameters, dict)
                        else {}
                    ),
                    **split_scalers,
                },
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
                dataset_parameters=dataset_parameters_payload,
            )

            metadata_config["duration_total_seconds"] = float((datetime.now(UTC) - run_started_at).total_seconds())

            persisted_run_id, persisted_dim_row = self._persist_dim_run(
                asset_id=asset_id,
                version=version,
                artifacts_dir=artifacts_dir,
                trainer_config=metadata_config,
                feature_cols=feature_cols,
                split_signature=split_signature,
                config_signature=config_signature,
                feature_set_hash=feature_set_hash,
                created_at_utc=created_at_utc,
                status="ok",
            )

            if persisted_run_id:
                try:
                    self._persist_run_snapshot(
                        run_id=persisted_run_id,
                        asset_id=asset_id,
                        parent_sweep_id=metadata_config.get("parent_sweep_id"),
                        split_cfg=split_cfg,
                        warmup_meta=warmup_meta,
                        df=df,
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                        split_signature=split_signature,
                        store_split_timestamps_ref=bool(
                            metadata_config.get("store_split_timestamps_ref", False)
                        ),
                        overwrite=overwrite_on_collision,
                    )
                    self._persist_fact_epoch_metrics(
                        run_id=persisted_run_id,
                        asset_id=asset_id,
                        parent_sweep_id=metadata_config.get("parent_sweep_id"),
                        fold_name=metadata_config.get("fold"),
                        history=training_result.history,
                        overwrite=overwrite_on_collision,
                    )
                    self._persist_fact_split_metrics(
                        run_id=persisted_run_id,
                        asset_id=asset_id,
                        parent_sweep_id=metadata_config.get("parent_sweep_id"),
                        split_metrics=training_result.split_metrics,
                        split_counts={
                            "train": int(len(train_df)),
                            "val": int(len(val_df)),
                            "test": int(len(test_df)),
                        },
                        overwrite=overwrite_on_collision,
                    )
                    self._persist_fact_oos_predictions(
                        run_id=persisted_run_id,
                        asset_id=asset_id,
                        feature_set_name=metadata_config.get("feature_set_name", feature_tag),
                        model_version=str(version),
                        config_signature=config_signature,
                        fold_name=metadata_config.get("fold"),
                        seed=metadata_config.get("seed") if isinstance(metadata_config.get("seed"), int) else None,
                        max_encoder_length=int(
                            metadata_config.get(
                                "max_encoder_length",
                                TFT_TRAINING_DEFAULTS["max_encoder_length"],
                            )
                        ),
                        split_frames={"train": train_df, "val": val_df, "test": test_df},
                        split_predictions=training_result.split_predictions,
                        overwrite=overwrite_on_collision,
                    )
                    self._persist_fact_config(
                        run_id=persisted_run_id,
                        asset_id=asset_id,
                        parent_sweep_id=metadata_config.get("parent_sweep_id"),
                        trainer_config=metadata_config,
                        training_result=training_result,
                        dataset_parameters=dataset_parameters_payload,
                        overwrite=overwrite_on_collision,
                    )
                    self._persist_fact_model_artifacts(
                        run_id=persisted_run_id,
                        asset_id=asset_id,
                        version=version,
                        artifacts_dir=artifacts_dir,
                        training_result=training_result,
                        overwrite=overwrite_on_collision,
                    )
                    self._persist_bridge_run_features(
                        run_id=persisted_run_id,
                        feature_cols=feature_cols,
                        overwrite=overwrite_on_collision,
                    )
                except Exception as persist_exc:
                    if persisted_dim_row is not None:
                        partial_row = dict(persisted_dim_row)
                        partial_row["status"] = "partial_failed"
                        self.analytics_run_repository.upsert_dim_run(partial_row)
                    self._append_fact_failure(
                        run_id=persisted_run_id,
                        asset_id=asset_id,
                        trainer_config=metadata_config,
                        stage="analytics_persist",
                        exc=persist_exc,
                        overwrite=overwrite_on_collision,
                    )
                    logger.exception("analytics persistence failed after successful training")

            return TrainTFTModelResult(
                asset_id=asset_id,
                version=version,
                metrics=training_result.metrics,
                artifacts_dir=artifacts_dir,
            )

        except Exception as exc:
            metadata_config["duration_total_seconds"] = float((datetime.now(UTC) - run_started_at).total_seconds())
            persisted_run_id, _ = self._persist_dim_run(
                asset_id=asset_id,
                version=version,
                artifacts_dir=artifacts_dir,
                trainer_config=metadata_config,
                feature_cols=feature_cols,
                split_signature=split_signature,
                config_signature=config_signature,
                feature_set_hash=feature_set_hash,
                created_at_utc=created_at_utc,
                status="failed",
            )
            if persisted_run_id:
                self._append_fact_failure(
                    run_id=persisted_run_id,
                    asset_id=asset_id,
                    trainer_config=metadata_config,
                    stage="train_execute",
                    exc=exc,
                    overwrite=overwrite_on_collision,
                )
            raise

    @staticmethod
    def _build_trainer_config(training_config: dict | None) -> dict:
        config = dict(training_config or {})
        allowed_extra = {"derived_feature_groups", "parent_sweep_id", "trial_number", "fold", "pipeline_version", "retries", "eta_recorded_seconds", "duration_total_seconds", "feature_set_name", "search_space", "objective_metric", "objective_name", "objective_direction", "entrypoint", "cmdline", "stdout_truncated", "stderr_truncated"}
        return {
            k: v
            for k, v in config.items()
            if k in TFT_TRAINING_DEFAULTS or k in allowed_extra
        }
