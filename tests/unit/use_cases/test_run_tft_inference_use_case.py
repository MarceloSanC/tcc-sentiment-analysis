from __future__ import annotations

import logging

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.entities.tft_inference_record import TFTInferenceRecord
from src.interfaces.tft_inference_model_loader import LoadedTFTInferenceModel
from src.use_cases.run_tft_inference_use_case import RunTFTInferenceUseCase


class _FakeDatasetRepo:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()

    def load(self, asset_id: str) -> pd.DataFrame:
        return self.df.copy()


class _FakeInferenceRepo:
    def __init__(self) -> None:
        self.rows: list[TFTInferenceRecord] = []

    def get_latest_timestamp(self, asset_id: str):
        if not self.rows:
            return None
        return max(r.timestamp for r in self.rows)

    def list_inference_timestamps(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
        *,
        model_version: str | None = None,
        feature_set_name: str | None = None,
    ):
        out = []
        for r in self.rows:
            if not (start_date <= r.timestamp <= end_date):
                continue
            if model_version is not None and r.model_version != model_version:
                continue
            if feature_set_name is not None and r.feature_set_name != feature_set_name:
                continue
            out.append(r.timestamp)
        return set(out)

    def upsert_records(self, asset_id: str, records: list[TFTInferenceRecord]) -> int:
        self.rows.extend(records)
        return len(records)


class _FakeModelLoader:
    def __init__(self, asset_id: str = "AAPL") -> None:
        self.asset_id = asset_id

    def load(self, model_dir: str | Path) -> LoadedTFTInferenceModel:
        return LoadedTFTInferenceModel(
            asset_id=self.asset_id,
            version="20260302_010101_B",
            model_dir=Path(model_dir),
            model=object(),
            feature_cols=["close"],
            feature_set_name="BASELINE_FEATURES",
            feature_tokens=["BASELINE_FEATURES"],
            training_config={"max_encoder_length": 3, "max_prediction_length": 1},
            scalers={},
        )


class _FakeEngine:
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
        out: list[TFTInferenceRecord] = []
        for _, row in dataset_df.iterrows():
            ts = pd.Timestamp(row["timestamp"]).to_pydatetime()
            out.append(
                TFTInferenceRecord(
                    asset_id=asset_id,
                    timestamp=ts,
                    model_version=model_version,
                    model_path=model_path,
                    feature_set_name=feature_set_name,
                    features_used_csv=features_used_csv,
                    prediction=float(row["target_return"]),
                    quantile_p10=-0.1,
                    quantile_p50=0.0,
                    quantile_p90=0.1,
                    inference_run_id=run_id,
                )
            )
        return out


class _FakeEngineMissingQuantiles(_FakeEngine):
    def infer(self, **kwargs) -> list[TFTInferenceRecord]:
        base = super().infer(**kwargs)
        out: list[TFTInferenceRecord] = []
        for r in base:
            out.append(
                TFTInferenceRecord(
                    asset_id=r.asset_id,
                    timestamp=r.timestamp,
                    model_version=r.model_version,
                    model_path=r.model_path,
                    feature_set_name=r.feature_set_name,
                    features_used_csv=r.features_used_csv,
                    prediction=r.prediction,
                    quantile_p10=None,
                    quantile_p50=None,
                    quantile_p90=None,
                    inference_run_id=r.inference_run_id,
                )
            )
        return out


class _FakeAnalyticsRunRepo:
    def __init__(self) -> None:
        self.inference_rows: list[dict] = []
        self.inference_prediction_rows: list[dict] = []
        self.feature_contrib_rows: list[dict] = []

    def append_fact_inference_runs(self, row: dict, overwrite: bool = False) -> None:
        self.inference_rows.append({**row, "_overwrite": overwrite})

    def append_fact_inference_predictions(
        self,
        rows: list[dict],
        overwrite: bool = False,
    ) -> None:
        self.inference_prediction_rows.extend(rows)

    def append_fact_feature_contrib_local(
        self,
        rows: list[dict],
        overwrite: bool = False,
    ) -> None:
        self.feature_contrib_rows.extend(rows)


class _FakeScaler:
    def transform(self, x):
        arr = pd.DataFrame(x).to_numpy(dtype="float64")
        return arr + 1000.0


def _dataset(start: datetime, rows: int = 12) -> pd.DataFrame:
    data = []
    for i in range(rows):
        ts = start + timedelta(days=i)
        data.append(
            {
                "timestamp": ts,
                "time_idx": i,
                "asset_id": "AAPL",
                "target_return": 0.001 * i,
                "close": 100.0 + i,
                "day_of_week": i % 7,
                "month": 1,
            }
        )
    return pd.DataFrame(data)


def test_use_case_skips_existing_when_not_overwrite() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    dataset_repo = _FakeDatasetRepo(_dataset(start))
    inference_repo = _FakeInferenceRepo()
    loader = _FakeModelLoader(asset_id="AAPL")
    engine = _FakeEngine()

    existing_ts = start + timedelta(days=6)
    inference_repo.rows.append(
        TFTInferenceRecord(
            asset_id="AAPL",
            timestamp=existing_ts,
            model_version="20260302_010101_B",
            model_path="/tmp/model",
            feature_set_name="BASELINE_FEATURES",
            features_used_csv="close",
            prediction=0.0,
        )
    )

    use_case = RunTFTInferenceUseCase(
        dataset_repository=dataset_repo,
        inference_repository=inference_repo,
        model_loader=loader,
        inference_engine=engine,
    )

    result = use_case.execute(
        asset_id="AAPL",
        model_path="/tmp/model",
        start_date=start + timedelta(days=5),
        end_date=start + timedelta(days=7),
        overwrite=False,
    )

    assert result.inferred >= 3
    assert result.skipped_existing >= 1
    assert result.attempted_upserts >= 2


def test_use_case_requires_explicit_period_without_history() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    dataset_repo = _FakeDatasetRepo(_dataset(start))
    inference_repo = _FakeInferenceRepo()
    loader = _FakeModelLoader(asset_id="AAPL")
    engine = _FakeEngine()

    use_case = RunTFTInferenceUseCase(
        dataset_repository=dataset_repo,
        inference_repository=inference_repo,
        model_loader=loader,
        inference_engine=engine,
    )

    with pytest.raises(ValueError, match="No inference history found"):
        use_case.execute(
            asset_id="AAPL",
            model_path="/tmp/model",
            start_date=None,
            end_date=None,
        )


def test_use_case_triggers_refresh_when_end_exceeds_dataset() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    dataset_repo = _FakeDatasetRepo(_dataset(start, rows=8))
    inference_repo = _FakeInferenceRepo()
    loader = _FakeModelLoader(asset_id="AAPL")
    engine = _FakeEngine()

    called = {"value": False}

    def _refresh(asset_id: str, refresh_start: datetime, refresh_end: datetime, rebuild_start: datetime):
        called["value"] = True
        dataset_repo.df = _dataset(start, rows=12)

    use_case = RunTFTInferenceUseCase(
        dataset_repository=dataset_repo,
        inference_repository=inference_repo,
        model_loader=loader,
        inference_engine=engine,
        refresh_dataset_fn=_refresh,
    )

    result = use_case.execute(
        asset_id="AAPL",
        model_path="/tmp/model",
        start_date=start + timedelta(days=5),
        end_date=start + timedelta(days=9),
    )

    assert called["value"] is True
    assert result.refreshed_dataset is True
    assert result.attempted_upserts > 0


def test_use_case_fail_fast_when_end_exceeds_dataset_and_auto_refresh_disabled() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    dataset_repo = _FakeDatasetRepo(_dataset(start, rows=8))
    inference_repo = _FakeInferenceRepo()
    loader = _FakeModelLoader(asset_id="AAPL")
    engine = _FakeEngine()

    use_case = RunTFTInferenceUseCase(
        dataset_repository=dataset_repo,
        inference_repository=inference_repo,
        model_loader=loader,
        inference_engine=engine,
        refresh_dataset_fn=None,
    )

    with pytest.raises(ValueError, match="auto-refresh is disabled"):
        use_case.execute(
            asset_id="AAPL",
            model_path="/tmp/model",
            start_date=start + timedelta(days=5),
            end_date=start + timedelta(days=9),
        )


def test_inference_fails_with_clear_diagnostic_when_model_feature_is_missing() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    df = _dataset(start).drop(columns=["close"])
    dataset_repo = _FakeDatasetRepo(df)
    inference_repo = _FakeInferenceRepo()
    loader = _FakeModelLoader(asset_id="AAPL")
    engine = _FakeEngine()
    use_case = RunTFTInferenceUseCase(
        dataset_repository=dataset_repo,
        inference_repository=inference_repo,
        model_loader=loader,
        inference_engine=engine,
    )

    with pytest.raises(ValueError, match="missing_model_features=.*close"):
        use_case.execute(
            asset_id="AAPL",
            model_path="/tmp/model",
            start_date=start + timedelta(days=5),
            end_date=start + timedelta(days=7),
        )


def test_inference_logs_excess_dataset_features_but_runs(caplog) -> None:
    caplog.set_level(logging.INFO)
    start = datetime(2025, 1, 1, tzinfo=UTC)
    df = _dataset(start)
    df["open"] = df["close"] - 1.0  # feature implemented but unused by model
    dataset_repo = _FakeDatasetRepo(df)
    inference_repo = _FakeInferenceRepo()
    loader = _FakeModelLoader(asset_id="AAPL")
    engine = _FakeEngine()
    use_case = RunTFTInferenceUseCase(
        dataset_repository=dataset_repo,
        inference_repository=inference_repo,
        model_loader=loader,
        inference_engine=engine,
    )

    result = use_case.execute(
        asset_id="AAPL",
        model_path="/tmp/model",
        start_date=start + timedelta(days=5),
        end_date=start + timedelta(days=7),
    )

    assert result.attempted_upserts > 0
    assert any("extra model feature columns not used" in r.message for r in caplog.records)


def test_persists_fact_inference_runs_when_analytics_repo_is_configured() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    dataset_repo = _FakeDatasetRepo(_dataset(start))
    inference_repo = _FakeInferenceRepo()
    loader = _FakeModelLoader(asset_id="AAPL")
    engine = _FakeEngine()
    analytics_repo = _FakeAnalyticsRunRepo()
    use_case = RunTFTInferenceUseCase(
        dataset_repository=dataset_repo,
        inference_repository=inference_repo,
        model_loader=loader,
        inference_engine=engine,
        analytics_run_repository=analytics_repo,
    )

    result = use_case.execute(
        asset_id="AAPL",
        model_path="/tmp/model",
        start_date=start + timedelta(days=5),
        end_date=start + timedelta(days=7),
    )

    assert result.attempted_upserts > 0
    assert len(analytics_repo.inference_rows) == 1
    row = analytics_repo.inference_rows[0]
    assert row["asset"] == "AAPL"
    assert row["model_version"] == "20260302_010101_B"
    assert row["status"] == "ok"
    assert row["upserts_count"] == result.attempted_upserts
    assert len(analytics_repo.inference_prediction_rows) == result.attempted_upserts
    assert all(r["split"] == "inference" for r in analytics_repo.inference_prediction_rows)
    assert all(r["horizon"] == 1 for r in analytics_repo.inference_prediction_rows)
    assert all("quantile_p10_post_guardrail" in r for r in analytics_repo.inference_prediction_rows)
    assert all("quantile_guardrail_applied" in r for r in analytics_repo.inference_prediction_rows)
    assert len(analytics_repo.feature_contrib_rows) > 0
    assert all(r["method"] == "local_magnitude_signed_v1" for r in analytics_repo.feature_contrib_rows)
    assert all(r["feature_rank"] >= 1 for r in analytics_repo.feature_contrib_rows)


def test_apply_scalers_keeps_time_idx_unscaled() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    df = _dataset(start, rows=5)
    original_time_idx = df["time_idx"].copy()

    out = RunTFTInferenceUseCase._apply_scalers(
        df,
        {
            "time_idx": _FakeScaler(),  # must be ignored
            "close": _FakeScaler(),     # must be transformed
        },
    )

    assert out["time_idx"].tolist() == original_time_idx.tolist()
    assert (out["close"] > df["close"]).all()


def test_strict_quantiles_fails_when_prediction_mode_is_quantile_and_outputs_are_missing() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    dataset_repo = _FakeDatasetRepo(_dataset(start))
    inference_repo = _FakeInferenceRepo()
    loader = _FakeModelLoader(asset_id="AAPL")
    engine = _FakeEngineMissingQuantiles()
    use_case = RunTFTInferenceUseCase(
        dataset_repository=dataset_repo,
        inference_repository=inference_repo,
        model_loader=loader,
        inference_engine=engine,
    )

    with pytest.raises(ValueError, match="Strict quantile validation failed"):
        use_case.execute(
            asset_id="AAPL",
            model_path="/tmp/model",
            start_date=start + timedelta(days=5),
            end_date=start + timedelta(days=7),
            strict_quantiles=True,
        )


def test_use_case_short_period_one_eligible_day_persists_single_prediction() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    dataset_repo = _FakeDatasetRepo(_dataset(start))
    inference_repo = _FakeInferenceRepo()
    loader = _FakeModelLoader(asset_id="AAPL")
    engine = _FakeEngine()

    use_case = RunTFTInferenceUseCase(
        dataset_repository=dataset_repo,
        inference_repository=inference_repo,
        model_loader=loader,
        inference_engine=engine,
    )

    result = use_case.execute(
        asset_id="AAPL",
        model_path="/tmp/model",
        start_date=start + timedelta(days=5),
        end_date=start + timedelta(days=5),
        overwrite=True,
    )

    assert result.attempted_upserts == 1
    assert result.inferred == 1


def test_use_case_multiple_eligible_days_persists_all_days_in_window() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    dataset_repo = _FakeDatasetRepo(_dataset(start))
    inference_repo = _FakeInferenceRepo()
    loader = _FakeModelLoader(asset_id="AAPL")
    engine = _FakeEngine()

    use_case = RunTFTInferenceUseCase(
        dataset_repository=dataset_repo,
        inference_repository=inference_repo,
        model_loader=loader,
        inference_engine=engine,
    )

    result = use_case.execute(
        asset_id="AAPL",
        model_path="/tmp/model",
        start_date=start + timedelta(days=5),
        end_date=start + timedelta(days=9),
        overwrite=True,
    )

    # Eligible timestamps are exactly [d5..d9] (5 dias) for max_encoder_length=3.
    assert result.attempted_upserts == 5
    assert result.inferred == 5


def test_use_case_fails_when_requested_window_has_no_eligible_context() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    dataset_repo = _FakeDatasetRepo(_dataset(start))
    inference_repo = _FakeInferenceRepo()
    loader = _FakeModelLoader(asset_id="AAPL")
    engine = _FakeEngine()

    use_case = RunTFTInferenceUseCase(
        dataset_repository=dataset_repo,
        inference_repository=inference_repo,
        model_loader=loader,
        inference_engine=engine,
    )

    with pytest.raises(ValueError, match="No eligible timestamps"):
        use_case.execute(
            asset_id="AAPL",
            model_path="/tmp/model",
            start_date=start + timedelta(days=1),
            end_date=start + timedelta(days=1),
            overwrite=True,
        )


def test_use_case_overwrite_false_with_all_existing_does_not_create_orphan_inference_run() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    dataset_repo = _FakeDatasetRepo(_dataset(start))
    inference_repo = _FakeInferenceRepo()
    loader = _FakeModelLoader(asset_id="AAPL")
    engine = _FakeEngine()
    analytics_repo = _FakeAnalyticsRunRepo()

    # Preload exactly the requested eligible targets (d5, d6, d7)
    for d in [5, 6, 7]:
        ts = start + timedelta(days=d)
        inference_repo.rows.append(
            TFTInferenceRecord(
                asset_id="AAPL",
                timestamp=ts,
                model_version="20260302_010101_B",
                model_path="/tmp/model",
                feature_set_name="BASELINE_FEATURES",
                features_used_csv="close",
                prediction=0.0,
            )
        )

    use_case = RunTFTInferenceUseCase(
        dataset_repository=dataset_repo,
        inference_repository=inference_repo,
        model_loader=loader,
        inference_engine=engine,
        analytics_run_repository=analytics_repo,
    )

    result = use_case.execute(
        asset_id="AAPL",
        model_path="/tmp/model",
        start_date=start + timedelta(days=5),
        end_date=start + timedelta(days=7),
        overwrite=False,
    )

    assert result.attempted_upserts == 0
    assert result.skipped_existing == 3
    # No rows persisted in analytics when there are no new records.
    assert analytics_repo.inference_rows == []
    assert analytics_repo.inference_prediction_rows == []


def test_use_case_last_point_mode_keeps_only_latest_target_timestamp() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    dataset_repo = _FakeDatasetRepo(_dataset(start))
    inference_repo = _FakeInferenceRepo()
    loader = _FakeModelLoader(asset_id="AAPL")
    engine = _FakeEngine()

    use_case = RunTFTInferenceUseCase(
        dataset_repository=dataset_repo,
        inference_repository=inference_repo,
        model_loader=loader,
        inference_engine=engine,
    )

    result = use_case.execute(
        asset_id="AAPL",
        model_path="/tmp/model",
        start_date=start + timedelta(days=5),
        end_date=start + timedelta(days=9),
        overwrite=True,
        inference_mode="last_point",
    )

    assert result.attempted_upserts == 1
    assert len(inference_repo.rows) == 1
    assert inference_repo.rows[0].timestamp == start + timedelta(days=9)


def test_use_case_rejects_invalid_inference_mode() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    dataset_repo = _FakeDatasetRepo(_dataset(start))
    inference_repo = _FakeInferenceRepo()
    loader = _FakeModelLoader(asset_id="AAPL")
    engine = _FakeEngine()

    use_case = RunTFTInferenceUseCase(
        dataset_repository=dataset_repo,
        inference_repository=inference_repo,
        model_loader=loader,
        inference_engine=engine,
    )

    with pytest.raises(ValueError, match="Invalid inference_mode"):
        use_case.execute(
            asset_id="AAPL",
            model_path="/tmp/model",
            start_date=start + timedelta(days=5),
            end_date=start + timedelta(days=7),
            inference_mode="foo",
        )


def test_apply_scalers_keeps_target_return_timestamp_and_asset_id_unscaled() -> None:
    start = datetime(2025, 1, 1, tzinfo=UTC)
    df = _dataset(start, rows=5)
    original_target = df["target_return"].copy()
    original_timestamp = df["timestamp"].copy()
    original_asset = df["asset_id"].copy()

    out = RunTFTInferenceUseCase._apply_scalers(
        df,
        {
            "target_return": _FakeScaler(),  # must be ignored
            "timestamp": _FakeScaler(),      # must be ignored
            "asset_id": _FakeScaler(),       # must be ignored
            "close": _FakeScaler(),          # must be transformed
        },
    )

    assert out["target_return"].tolist() == original_target.tolist()
    assert out["timestamp"].tolist() == original_timestamp.tolist()
    assert out["asset_id"].tolist() == original_asset.tolist()
    assert (out["close"] > df["close"]).all()
