from __future__ import annotations

import logging

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from src.domain.services.quantile_guardrail_service import QuantileGuardrailService
from src.domain.time.utc import require_tz_aware, to_utc
from src.entities.tft_inference_record import TFTInferenceRecord
from src.infrastructure.schemas.analytics_store_schema import ANALYTICS_SCHEMA_VERSION
from src.infrastructure.schemas.feature_validation_schema import IMPLEMENTED_FEATURES
from src.interfaces.analytics_run_repository import AnalyticsRunRepository
from src.interfaces.tft_dataset_repository import TFTDatasetRepository
from src.interfaces.tft_inference_engine import TFTInferenceEngine
from src.interfaces.tft_inference_model_loader import TFTInferenceModelLoader
from src.interfaces.tft_inference_repository import TFTInferenceRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunTFTInferenceResult:
    asset_id: str
    model_version: str
    start: datetime
    end: datetime
    inferred: int
    skipped_existing: int
    attempted_upserts: int
    refreshed_dataset: bool


class RunTFTInferenceUseCase:
    def __init__(
        self,
        *,
        dataset_repository: TFTDatasetRepository,
        inference_repository: TFTInferenceRepository,
        model_loader: TFTInferenceModelLoader,
        inference_engine: TFTInferenceEngine,
        analytics_run_repository: AnalyticsRunRepository | None = None,
        refresh_dataset_fn: Callable[[str, datetime, datetime, datetime], None] | None = None,
    ) -> None:
        self.dataset_repository = dataset_repository
        self.inference_repository = inference_repository
        self.model_loader = model_loader
        self.inference_engine = inference_engine
        self.analytics_run_repository = analytics_run_repository
        self.refresh_dataset_fn = refresh_dataset_fn

    def _persist_fact_inference_run(
        self,
        *,
        asset: str,
        model_version: str,
        inference_run_id: str,
        start_utc: datetime,
        end_utc: datetime,
        overwrite: bool,
        batch_size: int,
        result_status: str,
        inferred_count: int,
        skipped_count: int,
        upserts_count: int,
        duration_seconds: float,
        overwrite_on_collision: bool = False,
    ) -> None:
        if self.analytics_run_repository is None:
            return
        row = {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": None,
            "inference_run_id": str(inference_run_id),
            "model_version": str(model_version),
            "asset": str(asset),
            "inference_start_utc": str(start_utc.isoformat()),
            "inference_end_utc": str(end_utc.isoformat()),
            "overwrite": str(bool(overwrite)).lower(),
            "batch_size": int(batch_size),
            "status": str(result_status),
            "inferred_count": int(inferred_count),
            "skipped_count": int(skipped_count),
            "upserts_count": int(upserts_count),
            "duration_seconds": float(duration_seconds),
        }
        self.analytics_run_repository.append_fact_inference_runs(
            row,
            overwrite=overwrite_on_collision,
        )

    def _persist_fact_inference_predictions(
        self,
        *,
        records: list[TFTInferenceRecord],
        asset: str,
        model_version: str,
        feature_set_name: str,
        features_used_csv: str,
        model_path: str,
        inference_run_id: str,
        overwrite_on_collision: bool = False,
    ) -> None:
        if self.analytics_run_repository is None or not records:
            return
        created_at_utc = datetime.now(UTC).isoformat()
        rows: list[dict[str, object]] = []
        for r in records:
            target_ts_utc = to_utc(r.target_timestamp or r.timestamp)
            decision_ts_utc = to_utc(r.decision_timestamp or r.timestamp)
            horizon = int(getattr(r, "horizon", 1) or 1)
            q10 = float(r.quantile_p10) if r.quantile_p10 is not None else None
            q50 = float(r.quantile_p50) if r.quantile_p50 is not None else None
            q90 = float(r.quantile_p90) if r.quantile_p90 is not None else None
            guardrail = QuantileGuardrailService.enforce_monotonic_triplet(q10, q50, q90)
            pred = float(r.prediction)
            rows.append(
                {
                    "schema_version": ANALYTICS_SCHEMA_VERSION,
                    "inference_run_id": str(inference_run_id),
                    "run_id": None,
                    "model_version": str(model_version),
                    "asset": str(asset),
                    "feature_set_name": str(feature_set_name),
                    "features_used_csv": str(features_used_csv),
                    "model_path": str(model_path),
                    "split": "inference",
                    "horizon": horizon,
                    "timestamp_utc": decision_ts_utc.isoformat(),
                    "target_timestamp_utc": target_ts_utc.isoformat(),
                    "y_true": None,
                    "y_pred": pred,
                    "error": None,
                    "abs_error": None,
                    "sq_error": None,
                    "quantile_p10": q10,
                    "quantile_p50": q50,
                    "quantile_p90": q90,
                    "quantile_p10_post_guardrail": guardrail.p10_post,
                    "quantile_p50_post_guardrail": guardrail.p50_post,
                    "quantile_p90_post_guardrail": guardrail.p90_post,
                    "quantile_guardrail_applied": int(guardrail.applied),
                    "year": int(target_ts_utc.year),
                    "created_at_utc": created_at_utc,
                }
            )
        self.analytics_run_repository.append_fact_inference_predictions(
            rows,
            overwrite=overwrite_on_collision,
        )

    def _persist_fact_feature_contrib_local(
        self,
        *,
        inference_slice: pd.DataFrame,
        records: list[TFTInferenceRecord],
        feature_cols: list[str],
        asset: str,
        model_version: str,
        feature_set_name: str,
        inference_run_id: str,
        top_k: int = 5,
        overwrite_on_collision: bool = False,
    ) -> None:
        if self.analytics_run_repository is None or not records or not feature_cols:
            return

        usable_cols = [c for c in feature_cols if c in inference_slice.columns]
        if not usable_cols:
            return

        frame = inference_slice.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"], keep="last")
        by_ts = frame.set_index("timestamp")

        created_at_utc = datetime.now(UTC).isoformat()
        rows: list[dict[str, object]] = []
        top_k = max(1, int(top_k))

        for rec in records:
            target_ts_utc = pd.Timestamp(to_utc(rec.target_timestamp or rec.timestamp))
            decision_ts_utc = pd.Timestamp(to_utc(rec.decision_timestamp or rec.timestamp))
            horizon = int(getattr(rec, "horizon", 1) or 1)
            if target_ts_utc not in by_ts.index:
                continue
            row = by_ts.loc[target_ts_utc]
            vals: list[tuple[str, float]] = []
            for feat in usable_cols:
                val = pd.to_numeric(pd.Series([row.get(feat)]), errors="coerce").iloc[0]
                vals.append((feat, float(val) if pd.notna(val) else 0.0))

            denom = float(sum(abs(v) for _, v in vals))
            if denom <= 0.0:
                denom = float(len(vals)) if vals else 1.0

            pred = float(rec.prediction)
            contribs: list[tuple[str, float]] = []
            for feat, val in vals:
                w = (abs(val) / denom) if denom > 0.0 else 0.0
                sign = 1.0 if val >= 0.0 else -1.0
                c = float(pred * w * sign)
                contribs.append((feat, c))

            contribs = sorted(contribs, key=lambda x: abs(x[1]), reverse=True)[:top_k]
            for rank, (feat, contrib) in enumerate(contribs, start=1):
                rows.append(
                    {
                        "schema_version": ANALYTICS_SCHEMA_VERSION,
                        "inference_run_id": str(inference_run_id),
                        "run_id": None,
                        "model_version": str(model_version),
                        "asset": str(asset),
                        "feature_set_name": str(feature_set_name),
                        "split": "inference",
                        "horizon": horizon,
                        "timestamp_utc": decision_ts_utc.isoformat(),
                        "target_timestamp_utc": target_ts_utc.isoformat(),
                        "feature_name": str(feat),
                        "feature_rank": int(rank),
                        "contribution": float(contrib),
                        "abs_contribution": float(abs(contrib)),
                        "contribution_sign": "positive" if contrib >= 0.0 else "negative",
                        "method": "local_magnitude_signed_v1",
                        "year": int(target_ts_utc.year),
                        "created_at_utc": created_at_utc,
                    }
                )

        if rows:
            self.analytics_run_repository.append_fact_feature_contrib_local(
                rows,
                overwrite=overwrite_on_collision,
            )

    @staticmethod
    def _normalize_asset(asset_id: str) -> str:
        return asset_id.split(".")[0].upper()

    @staticmethod
    def _apply_scalers(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
        if not scalers:
            return df
        out = df.copy()
        # Structural columns must remain untouched for TimeSeriesDataSet compatibility.
        protected_columns = {"time_idx", "timestamp", "asset_id", "target_return"}
        for col, scaler in scalers.items():
            if col in protected_columns:
                continue
            if col not in out.columns or scaler is None:
                continue
            values = pd.to_numeric(out[col], errors="coerce")
            arr = np.array(values.to_numpy(dtype="float64"), copy=True)
            mask = np.isfinite(arr)
            if mask.any():
                if hasattr(scaler, "feature_names_in_"):
                    valid_input = pd.DataFrame({col: arr[mask]})
                else:
                    valid_input = arr[mask].reshape(-1, 1)
                arr[mask] = scaler.transform(valid_input).reshape(-1)
            out[col] = arr
        return out

    @staticmethod
    def _compute_requested_window(
        *,
        asset_id: str,
        inference_repo: TFTInferenceRepository,
        start_date: datetime | None,
        end_date: datetime | None,
        default_end_date: datetime,
    ) -> tuple[datetime, datetime]:
        if end_date is None:
            end_utc = to_utc(default_end_date)
        else:
            require_tz_aware(end_date, "end_date")
            end_utc = to_utc(end_date)

        if start_date is not None:
            require_tz_aware(start_date, "start_date")
            start_utc = to_utc(start_date)
        else:
            latest = inference_repo.get_latest_timestamp(asset_id)
            if latest is None:
                raise ValueError(
                    "No inference history found for asset. "
                    "Provide an explicit start/end period."
                )
            start_utc = to_utc(latest) + timedelta(days=1)

        if start_utc > end_utc:
            raise ValueError("Requested period invalid: start_date must be <= end_date.")
        return start_utc, end_utc

    @staticmethod
    def _validate_feature_compatibility(
        *,
        dataset_df: pd.DataFrame,
        model_feature_cols: list[str],
    ) -> tuple[list[str], list[str]]:
        expected = [c for c in model_feature_cols if c.strip()]
        expected_set = set(expected)
        missing = sorted([c for c in expected if c not in dataset_df.columns])

        implemented_set = set(IMPLEMENTED_FEATURES) | expected_set
        dataset_feature_cols = sorted(
            [c for c in dataset_df.columns if c in implemented_set]
        )
        excess = sorted([c for c in dataset_feature_cols if c not in expected_set])
        return missing, excess

    @staticmethod
    def _has_contiguous_context(df: pd.DataFrame, *, target_idx: int, max_encoder_length: int) -> bool:
        start_idx = int(target_idx) - int(max_encoder_length)
        if start_idx < 0:
            return False
        if "time_idx" not in df.columns:
            return False
        window = df.iloc[start_idx : int(target_idx) + 1]["time_idx"]
        vals = pd.to_numeric(window, errors="coerce")
        if vals.isna().any() or len(vals) != (max_encoder_length + 1):
            return False
        diffs = vals.diff().iloc[1:]
        return bool((diffs == 1).all())

    @staticmethod
    def _compute_eligible_target_indexes(
        df: pd.DataFrame,
        *,
        target_indexes: list[int],
        max_encoder_length: int,
    ) -> tuple[list[int], int, int]:
        eligible: list[int] = []
        skipped_no_context = 0
        skipped_non_contiguous = 0
        for idx in target_indexes:
            if int(idx) < int(max_encoder_length):
                skipped_no_context += 1
                continue
            if not RunTFTInferenceUseCase._has_contiguous_context(
                df,
                target_idx=int(idx),
                max_encoder_length=int(max_encoder_length),
            ):
                skipped_non_contiguous += 1
                continue
            eligible.append(int(idx))
        return eligible, skipped_no_context, skipped_non_contiguous

    def execute(
        self,
        *,
        asset_id: str,
        model_path: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        overwrite: bool = False,
        batch_size: int = 64,
        default_end_date: datetime | None = None,
        strict_quantiles: bool = True,
        inference_mode: str = "rolling",
        overwrite_on_collision: bool = False,
    ) -> RunTFTInferenceResult:
        run_started_at = datetime.now(UTC)
        asset = self._normalize_asset(asset_id)
        default_end = default_end_date or datetime.now(UTC)
        require_tz_aware(default_end, "default_end_date")

        normalized_inference_mode = str(inference_mode or "rolling").strip().lower()
        if normalized_inference_mode not in {"rolling", "last_point"}:
            raise ValueError(
                "Invalid inference_mode. Expected one of: rolling, last_point. "
                f"Received: {inference_mode}"
            )
        silver_overwrite = bool(overwrite_on_collision) or bool(overwrite)

        model_bundle = self.model_loader.load(model_path)
        model_asset = self._normalize_asset(model_bundle.asset_id)
        if model_asset != asset:
            raise ValueError(
                "Model asset does not match requested asset: "
                f"model={model_asset} requested={asset}"
            )

        max_encoder_length = int(model_bundle.training_config.get("max_encoder_length", 0))
        max_prediction_length = int(model_bundle.training_config.get("max_prediction_length", 1))
        if max_encoder_length < 2:
            raise ValueError(
                "Invalid model artifact: max_encoder_length not found or invalid in training config."
            )
        if max_prediction_length < 1:
            raise ValueError(
                "Invalid model artifact: max_prediction_length not found or invalid in training config."
            )

        start_utc, end_utc = self._compute_requested_window(
            asset_id=asset,
            inference_repo=self.inference_repository,
            start_date=start_date,
            end_date=end_date,
            default_end_date=default_end,
        )

        dataset_df = self.dataset_repository.load(asset)
        if dataset_df.empty:
            raise ValueError(
                "dataset_tft is empty for asset; build dataset first before inference."
            )
        dataset_df = dataset_df.copy()
        dataset_df["timestamp"] = pd.to_datetime(dataset_df["timestamp"], utc=True, errors="raise")
        dataset_df = dataset_df.sort_values("timestamp").reset_index(drop=True)

        ds_min = dataset_df["timestamp"].min().to_pydatetime()
        ds_max = dataset_df["timestamp"].max().to_pydatetime()

        refreshed = False
        if to_utc(end_utc) > to_utc(ds_max):
            if self.refresh_dataset_fn is None:
                raise ValueError(
                    "Requested end_date exceeds dataset_tft coverage and auto-refresh is disabled. "
                    "Run fetch/build pipelines manually to extend dataset_tft coverage, or enable "
                    "auto-refresh and retry."
                )
            refresh_start = to_utc(ds_max) + timedelta(days=1)
            rebuild_start = to_utc(ds_min)
            if refresh_start <= end_utc:
                logger.info(
                    "Refreshing dataset_tft coverage before inference",
                    extra={
                        "asset": asset,
                        "refresh_start": refresh_start.isoformat(),
                        "refresh_end": end_utc.isoformat(),
                        "rebuild_start": rebuild_start.isoformat(),
                    },
                )
                self.refresh_dataset_fn(asset, refresh_start, end_utc, rebuild_start)
                refreshed = True

                dataset_df = self.dataset_repository.load(asset).copy()
                dataset_df["timestamp"] = pd.to_datetime(
                    dataset_df["timestamp"], utc=True, errors="raise"
                )
                dataset_df = dataset_df.sort_values("timestamp").reset_index(drop=True)
                ds_min = dataset_df["timestamp"].min().to_pydatetime()
                ds_max = dataset_df["timestamp"].max().to_pydatetime()

        if start_utc < to_utc(ds_min) or end_utc > to_utc(ds_max):
            raise ValueError(
                "Requested period is outside dataset_tft coverage after refresh: "
                f"dataset=[{to_utc(ds_min).date()}..{to_utc(ds_max).date()}], "
                f"requested=[{start_utc.date()}..{end_utc.date()}]."
            )

        target_mask = (dataset_df["timestamp"] >= pd.Timestamp(start_utc)) & (
            dataset_df["timestamp"] <= pd.Timestamp(end_utc)
        )
        target_idx = dataset_df.index[target_mask].tolist()
        total_requested_days = int(len(target_idx))
        if not target_idx:
            raise ValueError("No dataset_tft rows found inside requested inference period.")

        eligible_idx, skipped_no_context, skipped_non_contiguous = self._compute_eligible_target_indexes(
            dataset_df,
            target_indexes=[int(i) for i in target_idx],
            max_encoder_length=int(max_encoder_length),
        )
        if not eligible_idx:
            raise ValueError(
                "No eligible timestamps for inference in requested period after context validation. "
                f"requested_days={total_requested_days} skipped_no_context={skipped_no_context} "
                f"skipped_non_contiguous={skipped_non_contiguous}."
            )

        first_target_idx = int(eligible_idx[0])
        last_target_idx = int(eligible_idx[-1])
        requested_target_timestamps = {
            to_utc(ts).isoformat()
            for ts in dataset_df.loc[eligible_idx, "timestamp"].tolist()
        }

        logger.info(
            "Inference temporal eligibility computed",
            extra={
                "asset_id": asset,
                "model_version": model_bundle.version,
                "requested_days": total_requested_days,
                "eligible_days": len(eligible_idx),
                "skipped_no_context": skipped_no_context,
                "skipped_non_contiguous": skipped_non_contiguous,
                "max_encoder_length": int(max_encoder_length),
                "max_prediction_length": int(max_prediction_length),
                "inference_mode": normalized_inference_mode,
            },
        )

        required_cols = {"timestamp", "time_idx", "asset_id", "target_return"}
        missing = [c for c in required_cols if c not in dataset_df.columns]
        if missing:
            raise ValueError(f"dataset_tft missing required columns for inference: {sorted(missing)}")

        missing_model_features, excess_dataset_features = self._validate_feature_compatibility(
            dataset_df=dataset_df,
            model_feature_cols=model_bundle.feature_cols,
        )
        if missing_model_features:
            raise ValueError(
                "Inference feature compatibility validation failed: "
                f"missing_model_features={missing_model_features}; "
                f"excess_dataset_features={excess_dataset_features}."
            )
        if excess_dataset_features:
            logger.info(
                "Inference feature compatibility: dataset contains extra model feature columns not used by this model",
                extra={
                    "asset_id": asset,
                    "model_version": model_bundle.version,
                    "excess_dataset_features": excess_dataset_features,
                },
            )

        inference_slice = dataset_df.iloc[first_target_idx - max_encoder_length : last_target_idx + 1].copy()
        inference_slice = self._apply_scalers(inference_slice, model_bundle.scalers)

        run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        features_used_csv = ",".join(model_bundle.feature_cols)
        records = self.inference_engine.infer(
            model=model_bundle.model,
            dataset_df=inference_slice,
            asset_id=asset,
            model_version=model_bundle.version,
            model_path=str(model_bundle.model_dir),
            feature_set_name=model_bundle.feature_set_name,
            features_used_csv=features_used_csv,
            feature_cols=model_bundle.feature_cols,
            dataset_parameters=model_bundle.dataset_parameters,
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            batch_size=batch_size,
            run_id=run_id,
        )
        records = [
            r
            for r in records
            if to_utc(r.target_timestamp or r.timestamp).isoformat() in requested_target_timestamps
        ]

        if normalized_inference_mode == "last_point":
            latest_target = max(
                to_utc(r.target_timestamp or r.timestamp)
                for r in records
            )
            records = [
                r for r in records
                if to_utc(r.target_timestamp or r.timestamp) == latest_target
            ]
        if not records:
            raise ValueError(
                "Inference generated no rows for eligible timestamps in requested period."
            )

        produced_target_timestamps = {
            to_utc(r.target_timestamp or r.timestamp).isoformat()
            for r in records
        }
        skipped_no_model_output = max(
            0,
            int(len(requested_target_timestamps) - len(produced_target_timestamps)),
        )

        logger.info(
            "Inference rolling coverage before dedup",
            extra={
                "asset_id": asset,
                "model_version": model_bundle.version,
                "requested_days": total_requested_days,
                "eligible_days": len(eligible_idx),
                "inferred_days_raw": len(produced_target_timestamps),
                "skipped_no_context": skipped_no_context,
                "skipped_non_contiguous": skipped_non_contiguous,
                "skipped_no_model_output": skipped_no_model_output,
            },
        )

        prediction_mode = str(
            model_bundle.training_config.get("prediction_mode", "quantile")
        ).strip().lower()
        if strict_quantiles and prediction_mode == "quantile":
            missing_q = [
                r
                for r in records
                if r.quantile_p10 is None or r.quantile_p50 is None or r.quantile_p90 is None
            ]
            if missing_q:
                first_ts = to_utc(missing_q[0].target_timestamp or missing_q[0].timestamp).isoformat()
                raise ValueError(
                    "Strict quantile validation failed: model configured with "
                    "prediction_mode='quantile' but quantile outputs are missing. "
                    f"missing_rows={len(missing_q)} first_missing_timestamp={first_ts} "
                    f"asset={asset} model_version={model_bundle.version}"
                )

        skipped_existing = 0
        if not overwrite:
            existing_ts = self.inference_repository.list_inference_timestamps(
                asset,
                start_utc,
                end_utc,
                model_version=model_bundle.version,
            )
            existing_norm = {to_utc(ts) for ts in existing_ts}
            filtered: list[TFTInferenceRecord] = []
            for r in records:
                target_ts = to_utc(r.target_timestamp or r.timestamp)
                if target_ts in existing_norm:
                    skipped_existing += 1
                    continue
                filtered.append(r)
            records = filtered

        if not records:
            logger.info(
                "Inference skipped: no new eligible rows after dedup",
                extra={
                    "asset_id": asset,
                    "model_version": model_bundle.version,
                    "eligible_days": len(eligible_idx),
                    "requested_days": total_requested_days,
                    "skipped_existing": skipped_existing,
                    "overwrite": bool(overwrite),
                    "inference_mode": normalized_inference_mode,
                },
            )
            return RunTFTInferenceResult(
                asset_id=asset,
                model_version=model_bundle.version,
                start=start_utc,
                end=end_utc,
                inferred=skipped_existing,
                skipped_existing=skipped_existing,
                attempted_upserts=0,
                refreshed_dataset=refreshed,
            )

        logger.info(
            "Inference rolling coverage after dedup",
            extra={
                "asset_id": asset,
                "model_version": model_bundle.version,
                "eligible_days": len(eligible_idx),
                "inferred_days_final": len(records),
                "skipped_existing": skipped_existing,
                "overwrite": bool(overwrite),
                "inference_mode": normalized_inference_mode,
            },
        )

        attempted_upserts = self.inference_repository.upsert_records(asset, records)

        self._persist_fact_inference_predictions(
            records=records,
            asset=asset,
            model_version=model_bundle.version,
            feature_set_name=model_bundle.feature_set_name,
            features_used_csv=features_used_csv,
            model_path=str(model_bundle.model_dir),
            inference_run_id=run_id,
            overwrite_on_collision=silver_overwrite,
        )
        self._persist_fact_feature_contrib_local(
            inference_slice=inference_slice,
            records=records,
            feature_cols=model_bundle.feature_cols,
            asset=asset,
            model_version=model_bundle.version,
            feature_set_name=model_bundle.feature_set_name,
            inference_run_id=run_id,
            top_k=min(5, max(1, len(model_bundle.feature_cols))),
            overwrite_on_collision=silver_overwrite,
        )

        duration_seconds = float((datetime.now(UTC) - run_started_at).total_seconds())
        self._persist_fact_inference_run(
            asset=asset,
            model_version=model_bundle.version,
            inference_run_id=run_id,
            start_utc=start_utc,
            end_utc=end_utc,
            overwrite=overwrite,
            batch_size=batch_size,
            result_status="ok",
            inferred_count=len(records) + skipped_existing,
            skipped_count=skipped_existing,
            upserts_count=attempted_upserts,
            duration_seconds=duration_seconds,
            overwrite_on_collision=silver_overwrite,
        )

        return RunTFTInferenceResult(
            asset_id=asset,
            model_version=model_bundle.version,
            start=start_utc,
            end=end_utc,
            inferred=len(records) + skipped_existing,
            skipped_existing=skipped_existing,
            attempted_upserts=attempted_upserts,
            refreshed_dataset=refreshed,
        )
