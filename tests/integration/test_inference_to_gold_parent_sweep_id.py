from __future__ import annotations

import json

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.adapters.parquet_analytics_run_repository import ParquetAnalyticsRunRepository
from src.entities.tft_inference_record import TFTInferenceRecord
from src.infrastructure.schemas.analytics_store_schema import ANALYTICS_SCHEMA_VERSION
from src.interfaces.tft_inference_model_loader import LoadedTFTInferenceModel
from src.use_cases.refresh_analytics_store_use_case import RefreshAnalyticsStoreUseCase
from src.use_cases.run_tft_inference_use_case import RunTFTInferenceUseCase


class _DatasetRepo:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def load(self, asset_id: str) -> pd.DataFrame:
        return self.df.copy()


class _InferenceRepo:
    def __init__(self) -> None:
        self.rows: list[TFTInferenceRecord] = []

    def get_latest_timestamp(self, asset_id: str):
        return None

    def list_inference_timestamps(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
        *,
        model_version: str | None = None,
        feature_set_name: str | None = None,
    ):
        return set()

    def upsert_records(self, asset_id: str, records: list[TFTInferenceRecord]) -> int:
        self.rows.extend(records)
        return len(records)


class _MetadataModelLoader:
    def load(self, model_dir: str | Path) -> LoadedTFTInferenceModel:
        model_path = Path(model_dir)
        metadata = json.loads((model_path / "metadata.json").read_text(encoding="utf-8"))
        return LoadedTFTInferenceModel(
            asset_id=str(metadata["asset_id"]),
            version=str(metadata["version"]),
            model_dir=model_path,
            model=object(),
            feature_cols=["close"],
            feature_set_name="BASELINE_FEATURES",
            feature_tokens=["BASELINE_FEATURES"],
            training_config={
                "max_encoder_length": 3,
                "max_prediction_length": 1,
                "prediction_mode": "quantile",
            },
            training_run_id=str(metadata["training_run_id"]),
            scalers={},
            dataset_parameters={},
        )


class _Engine:
    def infer(
        self,
        *,
        model: Any,
        dataset_df: pd.DataFrame,
        asset_id: str,
        model_version: str,
        model_path: str,
        feature_set_name: str,
        features_used_csv: str,
        feature_cols: list[str],
        dataset_parameters: dict | None,
        max_encoder_length: int,
        max_prediction_length: int,
        batch_size: int,
        run_id: str,
    ) -> list[TFTInferenceRecord]:
        records: list[TFTInferenceRecord] = []
        for _, row in dataset_df.iterrows():
            ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
            records.append(
                TFTInferenceRecord(
                    asset_id=asset_id,
                    timestamp=ts,
                    decision_timestamp=ts,
                    target_timestamp=ts,
                    model_version=model_version,
                    model_path=model_path,
                    feature_set_name=feature_set_name,
                    features_used_csv=features_used_csv,
                    prediction=float(row["target_return"]),
                    quantile_p10=-0.1,
                    quantile_p50=0.0,
                    quantile_p90=0.1,
                    inference_run_id=run_id,
                    horizon=1,
                )
            )
        return records


def _dataset(start: datetime) -> pd.DataFrame:
    rows = []
    for i in range(8):
        rows.append(
            {
                "timestamp": start + timedelta(days=i),
                "time_idx": i,
                "asset_id": "AAPL",
                "target_return": 0.01 * i,
                "close": 100.0 + i,
            }
        )
    return pd.DataFrame(rows)


def _read_table(root: Path, table_name: str) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in sorted((root / table_name).rglob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@pytest.mark.integration
def test_inference_training_run_id_feeds_gold_parent_sweep_id(tmp_path: Path) -> None:
    silver_dir = tmp_path / "silver"
    model_dir = tmp_path / "models" / "AAPL" / "runs" / "20260303_120000_B"
    model_dir.mkdir(parents=True)
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "asset_id": "AAPL",
                "version": "20260303_120000_B",
                "training_run_id": "R1",
            }
        ),
        encoding="utf-8",
    )

    analytics_repo = ParquetAnalyticsRunRepository(silver_dir)
    analytics_repo.upsert_dim_run(
        {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": "R1",
            "execution_id": None,
            "parent_sweep_id": "S1",
            "trial_number": 1,
            "fold": "wf_1",
            "seed": 7,
            "asset": "AAPL",
            "feature_set_name": "BASELINE_FEATURES",
            "feature_set_hash": "feature_hash",
            "feature_list_ordered_json": '["close"]',
            "config_signature": "config_sig",
            "split_fingerprint": "split_sig",
            "model_version": "20260303_120000_B",
            "checkpoint_path_final": str(model_dir / "model_state.pt"),
            "checkpoint_path_best": None,
            "git_commit": "test",
            "pipeline_version": "0.1",
            "library_versions_json": "{}",
            "hardware_info_json": "{}",
            "status": "ok",
            "duration_total_seconds": 1.0,
            "eta_recorded_seconds": 0.0,
            "retries": 0,
            "created_at_utc": "2026-05-17T00:00:00+00:00",
        }
    )
    analytics_repo.append_fact_model_artifacts(
        {
            "schema_version": ANALYTICS_SCHEMA_VERSION,
            "run_id": "R1",
            "training_run_id": "R1",
            "asset": "AAPL",
            "model_version": "20260303_120000_B",
            "checkpoint_path_final": str(model_dir / "model_state.pt"),
            "checkpoint_path_best": None,
            "config_path": str(model_dir / "config.json"),
            "scaler_path": None,
            "encoder_path": None,
            "feature_importance_json": "[]",
            "attention_summary_json": "{}",
            "logs_ref_json": "{}",
        }
    )

    start = datetime(2026, 1, 1, tzinfo=UTC)
    use_case = RunTFTInferenceUseCase(
        dataset_repository=_DatasetRepo(_dataset(start)),
        inference_repository=_InferenceRepo(),
        model_loader=_MetadataModelLoader(),
        inference_engine=_Engine(),
        analytics_run_repository=analytics_repo,
    )
    use_case.execute(
        asset_id="AAPL",
        model_path=str(model_dir),
        start_date=start + timedelta(days=5),
        end_date=start + timedelta(days=5),
    )

    fact_feature_contrib_local = _read_table(silver_dir, "fact_feature_contrib_local")
    dim_run = _read_table(silver_dir, "dim_run")

    assert not fact_feature_contrib_local.empty
    assert fact_feature_contrib_local["run_id"].eq("R1").all()
    assert fact_feature_contrib_local["training_run_id"].eq("R1").all()

    gold = RefreshAnalyticsStoreUseCase._build_gold_feature_contrib_local_summary(
        fact_feature_contrib_local,
        dim_run,
    )

    assert not gold.empty
    assert gold["parent_sweep_id"].tolist() == ["S1"]
