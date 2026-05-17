from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.domain.services.quantile_guardrail_service import QuantileGuardrailService
from src.infrastructure.schemas.analytics_store_schema import ANALYTICS_SCHEMA_VERSION
from src.interfaces.analytics_run_repository import AnalyticsRunRepository

logger = logging.getLogger(__name__)

# Stage 12 MVP: subset minimo do contrato BASELINES.md entregue pelo runner.
# Follow-up YELLOW (Stage 12-bis): random_walk, AR(1), EWMA-vol.
SUPPORTED_BASELINES = ("zero_return", "historical_mean_rolling", "historical_quantiles_rolling")

# Janela default para baselines rolling. Escolha justificada:
# - 30 dias para media: suaviza ruido diario sem capturar regime de medio prazo.
# - 252 dias para quantis: aproxima 1 ano de pregoes (anualidade financeira).
DEFAULT_MEAN_WINDOW = 30
DEFAULT_QUANTILE_WINDOW = 252

PredictionMode = Literal["point", "quantile"]


@dataclass(frozen=True)
class BaselineSpec:
    name: str
    prediction_mode: PredictionMode
    window: int | None


BASELINE_SPECS: dict[str, BaselineSpec] = {
    "zero_return": BaselineSpec(name="zero_return", prediction_mode="point", window=None),
    "historical_mean_rolling": BaselineSpec(
        name="historical_mean_rolling",
        prediction_mode="point",
        window=DEFAULT_MEAN_WINDOW,
    ),
    "historical_quantiles_rolling": BaselineSpec(
        name="historical_quantiles_rolling",
        prediction_mode="quantile",
        window=DEFAULT_QUANTILE_WINDOW,
    ),
}


@dataclass(frozen=True)
class RunBaselinesResult:
    asset: str
    parent_sweep_id: str
    baselines_persisted: list[str]
    run_ids: dict[str, str]
    n_rows_per_baseline: dict[str, int]


def _sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


class RunBaselinesUseCase:
    """
    Persist statistical baselines into fact_oos_predictions at the same grain as
    the TFT candidate, sharing `parent_sweep_id` to enable paired DM/MCS/win-rate
    builders in RefreshAnalyticsStoreUseCase to recognize baseline rows.

    Implements the BASELINES.md contract for Stage 12 (audit M7-Q2):
    persistence with traceable run_id, monotonic guardrail parity with TFT,
    no look-ahead, and identical (target_timestamp_utc, split, horizon) grain.
    """

    def __init__(
        self,
        *,
        analytics_run_repository: AnalyticsRunRepository,
        pipeline_version: str = "0.1",
    ) -> None:
        self.analytics_run_repository = analytics_run_repository
        self.pipeline_version = str(pipeline_version)

    @staticmethod
    def _normalize_asset(asset: str) -> str:
        return asset.split(".")[0].upper()

    @staticmethod
    def _compute_run_id(
        *,
        baseline_name: str,
        asset: str,
        parent_sweep_id: str,
        seed: int | None,
        window: int | None,
    ) -> str:
        payload = {
            "kind": "baseline",
            "baseline_name": baseline_name,
            "asset": asset,
            "parent_sweep_id": parent_sweep_id,
            "seed": seed,
            "window": window,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return _sha256_text(canonical)

    @staticmethod
    def _validate_splits(
        *,
        split_definitions: dict[str, tuple[Any, Any]],
    ) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        if not split_definitions:
            raise ValueError("split_definitions must contain at least one split")
        out: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for name, bounds in split_definitions.items():
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise ValueError(f"split '{name}' must be (start, end) tuple")
            start = pd.Timestamp(bounds[0])
            end = pd.Timestamp(bounds[1])
            if start.tzinfo is None:
                start = start.tz_localize("UTC")
            else:
                start = start.tz_convert("UTC")
            if end.tzinfo is None:
                end = end.tz_localize("UTC")
            else:
                end = end.tz_convert("UTC")
            if start > end:
                raise ValueError(f"split '{name}' has start > end")
            out[str(name)] = (start, end)
        return out

    @staticmethod
    def _load_dataset(dataset_path: Path) -> pd.DataFrame:
        df = pd.read_parquet(dataset_path)
        required = {"timestamp", "target_return"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"dataset at {dataset_path} missing required columns: {sorted(missing)}"
            )
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["target_return"] = pd.to_numeric(df["target_return"], errors="coerce")
        return df

    @staticmethod
    def _compute_prediction(
        *,
        baseline_name: str,
        history: np.ndarray,
        window: int | None,
    ) -> tuple[float, float, float, float] | None:
        """
        Compute (y_pred, q10, q50, q90) for a baseline given the *strictly past*
        target_return history (history[-1] is the value at decision_ts - 1).

        Returns None when the rolling window is incomplete (warmup), to ensure
        no silent zero/NaN predictions.
        """
        finite_history = history[np.isfinite(history)]
        if baseline_name == "zero_return":
            return 0.0, 0.0, 0.0, 0.0
        if window is None:
            raise ValueError(f"baseline '{baseline_name}' requires a window")
        if len(finite_history) < int(window):
            return None
        window_slice = finite_history[-int(window):]
        if baseline_name == "historical_mean_rolling":
            mean_val = float(np.mean(window_slice))
            return mean_val, mean_val, mean_val, mean_val
        if baseline_name == "historical_quantiles_rolling":
            q10 = float(np.percentile(window_slice, 10))
            q50 = float(np.percentile(window_slice, 50))
            q90 = float(np.percentile(window_slice, 90))
            return q50, q10, q50, q90
        raise ValueError(f"unsupported baseline: {baseline_name}")

    def _emit_oos_rows(
        self,
        *,
        df: pd.DataFrame,
        baseline_name: str,
        spec: BaselineSpec,
        run_id: str,
        asset: str,
        feature_set_name: str,
        model_version: str,
        config_signature: str,
        seed: int | None,
        split_definitions: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
        horizons: list[int],
    ) -> tuple[list[dict[str, Any]], int]:
        rows: list[dict[str, Any]] = []
        skipped_warmup = 0
        target_returns = df["target_return"].to_numpy(dtype="float64")
        timestamps = df["timestamp"].to_numpy()

        # Indices in chronological order; history at index i is strictly target_returns[:i].
        for split_name, (start, end) in split_definitions.items():
            ts_mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
            idxs = df.index[ts_mask].tolist()
            for i in idxs:
                decision_ts = pd.Timestamp(timestamps[i])
                history = target_returns[:i]
                prediction = self._compute_prediction(
                    baseline_name=baseline_name,
                    history=history,
                    window=spec.window,
                )
                if prediction is None:
                    skipped_warmup += 1
                    continue
                y_pred, q10, q50, q90 = prediction

                y_true_value = target_returns[i]
                if not np.isfinite(y_true_value):
                    # AGENT_CORE Non-Negotiable: skip rows with missing supervision.
                    continue

                for h in horizons:
                    h_int = int(h)
                    if h_int < 1:
                        continue
                    target_ts = decision_ts + pd.Timedelta(days=max(h_int - 1, 0))
                    if target_ts <= decision_ts and h_int > 1:
                        # Defensive: per the convention target_ts >= decision_ts for h>=2.
                        continue
                    guardrail = QuantileGuardrailService.enforce_monotonic_triplet(q10, q50, q90)
                    err = float(y_pred - y_true_value)
                    rows.append(
                        {
                            "schema_version": ANALYTICS_SCHEMA_VERSION,
                            "run_id": run_id,
                            "model_version": str(model_version),
                            "asset": str(asset),
                            "feature_set_name": str(feature_set_name),
                            "config_signature": str(config_signature),
                            "split": str(split_name),
                            "fold": "none",
                            "seed": int(seed) if seed is not None else 0,
                            "horizon": h_int,
                            "timestamp_utc": str(decision_ts.isoformat()),
                            "target_timestamp_utc": str(target_ts.isoformat()),
                            "y_true": float(y_true_value),
                            "y_pred": float(y_pred),
                            "error": err,
                            "abs_error": float(abs(err)),
                            "sq_error": float(err * err),
                            "quantile_p10": float(q10),
                            "quantile_p50": float(q50),
                            "quantile_p90": float(q90),
                            "quantile_p10_post_guardrail": guardrail.p10_post,
                            "quantile_p50_post_guardrail": guardrail.p50_post,
                            "quantile_p90_post_guardrail": guardrail.p90_post,
                            "quantile_guardrail_applied": int(guardrail.applied),
                            "year": int(target_ts.year),
                        }
                    )
        return rows, skipped_warmup

    def _persist_dim_run(
        self,
        *,
        run_id: str,
        asset: str,
        parent_sweep_id: str,
        baseline_name: str,
        feature_set_name: str,
        feature_set_hash: str,
        config_signature: str,
        split_signature: str,
        model_version: str,
        created_at_utc: str,
    ) -> None:
        row = {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": run_id,
            "execution_id": None,
            "parent_sweep_id": parent_sweep_id,
            "trial_number": None,
            "fold": None,
            "seed": None,
            "asset": asset,
            "feature_set_name": feature_set_name,
            "feature_set_hash": feature_set_hash,
            "feature_list_ordered_json": json.dumps([baseline_name]),
            "config_signature": config_signature,
            "split_fingerprint": split_signature,
            "model_version": model_version,
            "checkpoint_path_final": None,
            "checkpoint_path_best": None,
            "git_commit": None,
            "pipeline_version": self.pipeline_version,
            "library_versions_json": None,
            "hardware_info_json": None,
            "status": "ok",
            "duration_total_seconds": 0.0,
            "eta_recorded_seconds": 0.0,
            "retries": 0,
            "created_at_utc": created_at_utc,
        }
        self.analytics_run_repository.upsert_dim_run(row)

    def _persist_fact_run_snapshot(
        self,
        *,
        run_id: str,
        asset: str,
        parent_sweep_id: str,
        df: pd.DataFrame,
        split_definitions: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
        split_signature: str,
        overwrite: bool,
    ) -> None:
        ts_all = df["timestamp"]
        per_split: dict[str, pd.Series] = {}
        for name, (start, end) in split_definitions.items():
            mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
            per_split[name] = df.loc[mask, "timestamp"]

        def _bounds(series: pd.Series) -> tuple[str, str]:
            if series.empty:
                return ts_all.min().isoformat(), ts_all.min().isoformat()
            return series.min().isoformat(), series.max().isoformat()

        train_start, train_end = _bounds(per_split.get("train", pd.Series(dtype="datetime64[ns, UTC]")))
        val_start, val_end = _bounds(per_split.get("val", pd.Series(dtype="datetime64[ns, UTC]")))
        test_start, test_end = _bounds(per_split.get("test", pd.Series(dtype="datetime64[ns, UTC]")))

        fingerprint_payload = "|".join(
            [asset, str(len(df)), str(ts_all.min().isoformat()), str(ts_all.max().isoformat())]
        )

        row = {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": run_id,
            "asset": asset,
            "parent_sweep_id": parent_sweep_id,
            "dataset_start_utc": str(ts_all.min().isoformat()),
            "dataset_end_utc": str(ts_all.max().isoformat()),
            "train_start_utc": str(train_start),
            "train_end_utc": str(train_end),
            "val_start_utc": str(val_start),
            "val_end_utc": str(val_end),
            "test_start_utc": str(test_start),
            "test_end_utc": str(test_end),
            "warmup_policy": "drop_leading",
            "required_warmup_count": 0,
            "warmup_applied": "false",
            "effective_train_start_utc": str(train_start),
            "n_samples_train": int(len(per_split.get("train", []))),
            "n_samples_val": int(len(per_split.get("val", []))),
            "n_samples_test": int(len(per_split.get("test", []))),
            "dataset_fingerprint": _sha256_text(fingerprint_payload),
            "split_fingerprint": split_signature,
        }
        self.analytics_run_repository.append_fact_run_snapshot(row, overwrite=overwrite)

    def _persist_fact_config(
        self,
        *,
        run_id: str,
        asset: str,
        parent_sweep_id: str,
        spec: BaselineSpec,
        horizons: list[int],
        overwrite: bool,
    ) -> None:
        row = {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": run_id,
            "asset": asset,
            "parent_sweep_id": parent_sweep_id,
            # Critical: Stage 9 filters point baselines out of probabilistic metrics;
            # Stage 11 degeneracy gate skips point modes by design.
            "prediction_mode": spec.prediction_mode,
            "loss_name": f"baseline_{spec.name}",
            "quantile_levels_json": json.dumps([0.1, 0.5, 0.9]) if spec.prediction_mode == "quantile" else json.dumps([]),
            "evaluation_horizons_json": json.dumps(sorted({int(h) for h in horizons})),
            "max_encoder_length": int(spec.window or 0),
            "max_prediction_length": int(max(horizons) if horizons else 1),
            "batch_size": 0,
            "max_epochs": 0,
            "learning_rate": 0.0,
            "hidden_size": 0,
            "attention_head_size": 0,
            "dropout": 0.0,
            "hidden_continuous_size": 0,
            "early_stopping_patience": 0,
            "early_stopping_min_delta": 0.0,
            "scaler_type": "none",
            "training_config_json": json.dumps({"baseline": spec.name, "window": spec.window}),
            "dataset_parameters_json": "{}",
            "search_space_json": "{}",
            "objective_name": "mean_pinball" if spec.prediction_mode == "quantile" else "rmse",
            "objective_direction": "minimize",
        }
        self.analytics_run_repository.append_fact_config(row, overwrite=overwrite)

    def _persist_bridge_run_features(
        self,
        *,
        run_id: str,
        baseline_name: str,
        overwrite: bool,
    ) -> None:
        rows = [
            {
                "schema_version": ANALYTICS_SCHEMA_VERSION,
                "run_id": run_id,
                "feature_order": 0,
                "feature_name": f"baseline:{baseline_name}",
            }
        ]
        self.analytics_run_repository.append_bridge_run_features(rows, overwrite=overwrite)

    def execute(
        self,
        *,
        asset: str,
        dataset_path: Path | str,
        parent_sweep_id: str,
        split_definitions: dict[str, tuple[Any, Any]],
        horizons: list[int],
        baselines: list[str],
        seed: int | None = None,
        overwrite_on_collision: bool = False,
    ) -> RunBaselinesResult:
        if not parent_sweep_id or not str(parent_sweep_id).strip():
            raise ValueError(
                "parent_sweep_id is required to share grouping with TFT candidate runs"
            )
        asset_norm = self._normalize_asset(asset)
        ds_path = Path(dataset_path)
        if not ds_path.exists():
            raise ValueError(f"dataset not found: {ds_path}")

        unknown = [b for b in baselines if b not in BASELINE_SPECS]
        if unknown:
            raise ValueError(
                f"unsupported baselines: {sorted(unknown)}. "
                f"Stage 12 MVP supports: {sorted(BASELINE_SPECS.keys())}"
            )
        if not horizons:
            raise ValueError("horizons must contain at least one positive integer")
        horizons_sorted = sorted({int(h) for h in horizons if int(h) >= 1})
        if not horizons_sorted:
            raise ValueError("horizons must contain at least one positive integer")

        splits = self._validate_splits(split_definitions=split_definitions)
        df = self._load_dataset(ds_path)
        created_at_utc = datetime.now(UTC).isoformat()

        split_signature_payload = json.dumps(
            {
                k: [pd.Timestamp(v[0]).isoformat(), pd.Timestamp(v[1]).isoformat()]
                for k, v in sorted(splits.items())
            },
            sort_keys=True,
        )
        split_signature = _sha256_text(split_signature_payload)

        run_ids: dict[str, str] = {}
        n_rows: dict[str, int] = {}
        persisted: list[str] = []

        for baseline_name in baselines:
            spec = BASELINE_SPECS[baseline_name]
            feature_set_name = "baseline"
            feature_set_hash = _sha256_text(f"baseline|{baseline_name}|window={spec.window}")
            config_signature = _sha256_text(
                json.dumps(
                    {"baseline": baseline_name, "window": spec.window, "horizons": horizons_sorted},
                    sort_keys=True,
                )
            )
            model_version = f"baseline_{baseline_name}_v1"
            run_id = self._compute_run_id(
                baseline_name=baseline_name,
                asset=asset_norm,
                parent_sweep_id=str(parent_sweep_id),
                seed=seed,
                window=spec.window,
            )

            self._persist_dim_run(
                run_id=run_id,
                asset=asset_norm,
                parent_sweep_id=str(parent_sweep_id),
                baseline_name=baseline_name,
                feature_set_name=feature_set_name,
                feature_set_hash=feature_set_hash,
                config_signature=config_signature,
                split_signature=split_signature,
                model_version=model_version,
                created_at_utc=created_at_utc,
            )
            self._persist_fact_run_snapshot(
                run_id=run_id,
                asset=asset_norm,
                parent_sweep_id=str(parent_sweep_id),
                df=df,
                split_definitions=splits,
                split_signature=split_signature,
                overwrite=overwrite_on_collision,
            )
            self._persist_fact_config(
                run_id=run_id,
                asset=asset_norm,
                parent_sweep_id=str(parent_sweep_id),
                spec=spec,
                horizons=horizons_sorted,
                overwrite=overwrite_on_collision,
            )
            self._persist_bridge_run_features(
                run_id=run_id,
                baseline_name=baseline_name,
                overwrite=overwrite_on_collision,
            )

            rows, skipped_warmup = self._emit_oos_rows(
                df=df,
                baseline_name=baseline_name,
                spec=spec,
                run_id=run_id,
                asset=asset_norm,
                feature_set_name=feature_set_name,
                model_version=model_version,
                config_signature=config_signature,
                seed=seed,
                split_definitions=splits,
                horizons=horizons_sorted,
            )
            if rows:
                self.analytics_run_repository.append_fact_oos_predictions(
                    rows, overwrite=overwrite_on_collision
                )
            logger.info(
                "baseline persisted",
                extra={
                    "asset": asset_norm,
                    "parent_sweep_id": parent_sweep_id,
                    "baseline": baseline_name,
                    "run_id": run_id,
                    "n_rows": len(rows),
                    "skipped_warmup": skipped_warmup,
                },
            )
            run_ids[baseline_name] = run_id
            n_rows[baseline_name] = len(rows)
            persisted.append(baseline_name)

        return RunBaselinesResult(
            asset=asset_norm,
            parent_sweep_id=str(parent_sweep_id),
            baselines_persisted=persisted,
            run_ids=run_ids,
            n_rows_per_baseline=n_rows,
        )
