from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adapters.parquet_analytics_run_repository import ParquetAnalyticsRunRepository
from src.interfaces.analytics_run_repository import DuplicateKeyError
from src.use_cases.run_baselines_use_case import (
    BASELINE_SPECS,
    RunBaselinesUseCase,
)


def _write_dataset(path: Path, *, n_days: int = 300, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    returns = rng.normal(0.0, 0.01, size=n_days)
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["AAPL"] * n_days,
            "time_idx": np.arange(n_days, dtype="int64"),
            "target_return": returns,
        }
    )
    df.to_parquet(path, index=False)


def _split_definitions() -> dict[str, tuple[str, str]]:
    return {
        "train": ("2024-01-01", "2024-06-30"),
        "val": ("2024-07-01", "2024-08-31"),
        "test": ("2024-09-01", "2024-10-26"),
    }


def _load_oos(silver: Path) -> pd.DataFrame:
    base = silver / "fact_oos_predictions"
    if not base.exists():
        return pd.DataFrame()
    files = sorted(base.rglob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True) if files else pd.DataFrame()


def _load_dim(silver: Path) -> pd.DataFrame:
    base = silver / "dim_run"
    files = sorted(base.rglob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True) if files else pd.DataFrame()


def _load_config(silver: Path) -> pd.DataFrame:
    base = silver / "fact_config"
    files = sorted(base.rglob("*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True) if files else pd.DataFrame()


def test_zero_return_baseline_persists_with_correct_grain(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    ds = tmp_path / "dataset_tft_AAPL.parquet"
    _write_dataset(ds)

    repo = ParquetAnalyticsRunRepository(silver)
    use_case = RunBaselinesUseCase(analytics_run_repository=repo)

    result = use_case.execute(
        asset="AAPL",
        dataset_path=ds,
        parent_sweep_id="sw_test_round",
        split_definitions=_split_definitions(),
        horizons=[1, 2],
        baselines=["zero_return"],
    )

    assert result.baselines_persisted == ["zero_return"]
    oos = _load_oos(silver)
    assert not oos.empty
    assert (oos["y_pred"] == 0.0).all()
    assert (oos["run_id"] == result.run_ids["zero_return"]).all()

    cfg = _load_config(silver)
    assert (cfg["prediction_mode"] == "point").all()

    dim = _load_dim(silver)
    assert (dim["parent_sweep_id"] == "sw_test_round").all()
    assert (dim["feature_set_name"] == "baseline").all()
    assert dim["model_version"].iloc[0].startswith("baseline_zero_return_")


def test_historical_quantiles_rolling_produces_monotonic_quantiles(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    ds = tmp_path / "dataset_tft_AAPL.parquet"
    _write_dataset(ds, n_days=400, seed=1)

    repo = ParquetAnalyticsRunRepository(silver)
    use_case = RunBaselinesUseCase(analytics_run_repository=repo)
    use_case.execute(
        asset="AAPL",
        dataset_path=ds,
        parent_sweep_id="sw_q",
        split_definitions=_split_definitions(),
        horizons=[1],
        baselines=["historical_quantiles_rolling"],
    )

    oos = _load_oos(silver)
    assert not oos.empty
    assert (oos["quantile_p10_post_guardrail"] <= oos["quantile_p50_post_guardrail"]).all()
    assert (oos["quantile_p50_post_guardrail"] <= oos["quantile_p90_post_guardrail"]).all()

    cfg = _load_config(silver)
    assert (cfg["prediction_mode"] == "quantile").all()


def test_baseline_run_id_is_deterministic(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    ds = tmp_path / "ds.parquet"
    _write_dataset(ds)

    repo = ParquetAnalyticsRunRepository(silver)
    use_case = RunBaselinesUseCase(analytics_run_repository=repo)
    first = use_case.execute(
        asset="AAPL",
        dataset_path=ds,
        parent_sweep_id="sw1",
        split_definitions=_split_definitions(),
        horizons=[1],
        baselines=["zero_return"],
        seed=42,
        overwrite_on_collision=True,
    )
    second = use_case.execute(
        asset="AAPL",
        dataset_path=ds,
        parent_sweep_id="sw1",
        split_definitions=_split_definitions(),
        horizons=[1],
        baselines=["zero_return"],
        seed=42,
        overwrite_on_collision=True,
    )
    assert first.run_ids["zero_return"] == second.run_ids["zero_return"]


def test_baseline_overwrite_on_collision_respects_lei_2(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    ds = tmp_path / "ds.parquet"
    _write_dataset(ds)

    repo = ParquetAnalyticsRunRepository(silver)
    use_case = RunBaselinesUseCase(analytics_run_repository=repo)
    use_case.execute(
        asset="AAPL",
        dataset_path=ds,
        parent_sweep_id="sw_lei2",
        split_definitions=_split_definitions(),
        horizons=[1],
        baselines=["zero_return"],
    )
    with pytest.raises(DuplicateKeyError):
        use_case.execute(
            asset="AAPL",
            dataset_path=ds,
            parent_sweep_id="sw_lei2",
            split_definitions=_split_definitions(),
            horizons=[1],
            baselines=["zero_return"],
            overwrite_on_collision=False,
        )
    # With overwrite flag, second execution succeeds.
    use_case.execute(
        asset="AAPL",
        dataset_path=ds,
        parent_sweep_id="sw_lei2",
        split_definitions=_split_definitions(),
        horizons=[1],
        baselines=["zero_return"],
        overwrite_on_collision=True,
    )


def test_historical_mean_skips_rows_without_warmup_window(tmp_path: Path) -> None:
    # 50 days of data: rolling mean window=30 means the first 30 rows in the
    # train split have no warmup history and must be skipped (no silent zero).
    silver = tmp_path / "silver"
    ds = tmp_path / "ds.parquet"
    _write_dataset(ds, n_days=50)

    splits = {
        "train": ("2024-01-01", "2024-02-19"),  # day 1 to day 50 includes warmup
    }
    repo = ParquetAnalyticsRunRepository(silver)
    use_case = RunBaselinesUseCase(analytics_run_repository=repo)
    result = use_case.execute(
        asset="AAPL",
        dataset_path=ds,
        parent_sweep_id="sw_warmup",
        split_definitions=splits,
        horizons=[1],
        baselines=["historical_mean_rolling"],
    )
    oos = _load_oos(silver)
    # Total days = 50, but first 30 rows have history < 30 -> skipped (i counted
    # from 0; index i=30 has history target_returns[:30] of length 30 -> first
    # eligible). Therefore n_rows == 50 - 30 == 20.
    assert result.n_rows_per_baseline["historical_mean_rolling"] == 20
    assert len(oos) == 20


def test_baseline_no_lookahead_invariant(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    ds = tmp_path / "ds.parquet"
    _write_dataset(ds)

    repo = ParquetAnalyticsRunRepository(silver)
    use_case = RunBaselinesUseCase(analytics_run_repository=repo)
    use_case.execute(
        asset="AAPL",
        dataset_path=ds,
        parent_sweep_id="sw_nla",
        split_definitions=_split_definitions(),
        horizons=[1, 2, 3],
        baselines=["historical_quantiles_rolling"],
    )
    oos = _load_oos(silver)
    ts = pd.to_datetime(oos["timestamp_utc"], utc=True)
    tgt = pd.to_datetime(oos["target_timestamp_utc"], utc=True)
    # h=1 -> target_ts == ts; h>=2 -> target_ts > ts. Never target_ts < ts.
    assert (tgt >= ts).all()
    h_gt_1 = oos["horizon"].astype(int) > 1
    assert (tgt[h_gt_1] > ts[h_gt_1]).all()


def test_baseline_requires_parent_sweep_id(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    ds = tmp_path / "ds.parquet"
    _write_dataset(ds)
    repo = ParquetAnalyticsRunRepository(silver)
    use_case = RunBaselinesUseCase(analytics_run_repository=repo)
    with pytest.raises(ValueError, match="parent_sweep_id"):
        use_case.execute(
            asset="AAPL",
            dataset_path=ds,
            parent_sweep_id="",
            split_definitions=_split_definitions(),
            horizons=[1],
            baselines=["zero_return"],
        )


def test_baseline_rejects_unknown_baseline(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    ds = tmp_path / "ds.parquet"
    _write_dataset(ds)
    repo = ParquetAnalyticsRunRepository(silver)
    use_case = RunBaselinesUseCase(analytics_run_repository=repo)
    with pytest.raises(ValueError, match="unsupported baselines"):
        use_case.execute(
            asset="AAPL",
            dataset_path=ds,
            parent_sweep_id="sw1",
            split_definitions=_split_definitions(),
            horizons=[1],
            baselines=["arima_x"],
        )


def test_supported_baselines_specs_complete() -> None:
    # Guard: ensure MVP subset matches Stage 12 commitment.
    assert set(BASELINE_SPECS.keys()) == {
        "zero_return",
        "historical_mean_rolling",
        "historical_quantiles_rolling",
    }


def _write_deterministic_dataset(path: Path, *, n_days: int = 100) -> None:
    timestamps = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    returns = np.arange(1, n_days + 1, dtype="float64")
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "asset_id": ["AAPL"] * n_days,
            "time_idx": np.arange(n_days, dtype="int64"),
            "target_return": returns,
        }
    )
    df.to_parquet(path, index=False)


def test_baseline_window_override_changes_run_id_and_predictions(tmp_path: Path) -> None:
    silver_a = tmp_path / "silver_a"
    silver_b = tmp_path / "silver_b"
    ds = tmp_path / "ds.parquet"
    _write_deterministic_dataset(ds, n_days=100)

    common = {
        "asset": "AAPL",
        "dataset_path": ds,
        "parent_sweep_id": "sw_f4",
        "split_definitions": {"train": ("2024-01-01", "2024-04-09")},
        "horizons": [1],
        "baselines": ["historical_mean_rolling"],
    }
    # Default window=30.
    repo_a = ParquetAnalyticsRunRepository(silver_a)
    res_a = RunBaselinesUseCase(analytics_run_repository=repo_a).execute(**common)
    # Override window=10 via baseline_windows.
    repo_b = ParquetAnalyticsRunRepository(silver_b)
    res_b = RunBaselinesUseCase(analytics_run_repository=repo_b).execute(
        **common,
        baseline_windows={"historical_mean_rolling": 10},
    )

    # Different effective window -> different run_id.
    assert res_a.run_ids["historical_mean_rolling"] != res_b.run_ids["historical_mean_rolling"]

    # Default window=30 skips first 30 rows of warmup; override=10 skips first 10.
    n_a = res_a.n_rows_per_baseline["historical_mean_rolling"]
    n_b = res_b.n_rows_per_baseline["historical_mean_rolling"]
    assert n_b > n_a
    # Confirm predictions differ numerically: at i=30 (only point both windows cover),
    # default mean = mean(1..30) = 15.5; override window=10 mean = mean(21..30) = 25.5.
    oos_a = _load_oos(silver_a).sort_values("timestamp_utc").reset_index(drop=True)
    oos_b = _load_oos(silver_b).sort_values("timestamp_utc").reset_index(drop=True)
    a_first = float(oos_a.iloc[0]["y_pred"])
    # First eligible row in a (window=30) is index 30; mean of target_returns[0:30] = 15.5.
    assert a_first == pytest.approx(15.5)
    # First eligible row in b (window=10) is index 10; mean of target_returns[0:10] = 5.5.
    b_first = float(oos_b.iloc[0]["y_pred"])
    assert b_first == pytest.approx(5.5)


def test_baseline_window_override_same_value_keeps_run_id_deterministic(tmp_path: Path) -> None:
    silver_a = tmp_path / "silver_a"
    silver_b = tmp_path / "silver_b"
    ds = tmp_path / "ds.parquet"
    _write_deterministic_dataset(ds)

    common = {
        "asset": "AAPL",
        "dataset_path": ds,
        "parent_sweep_id": "sw_f4_det",
        "split_definitions": {"train": ("2024-01-01", "2024-04-09")},
        "horizons": [1],
        "baselines": ["historical_mean_rolling"],
        "baseline_windows": {"historical_mean_rolling": 15},
    }
    res_a = RunBaselinesUseCase(
        analytics_run_repository=ParquetAnalyticsRunRepository(silver_a)
    ).execute(**common)
    res_b = RunBaselinesUseCase(
        analytics_run_repository=ParquetAnalyticsRunRepository(silver_b)
    ).execute(**common)
    assert res_a.run_ids["historical_mean_rolling"] == res_b.run_ids["historical_mean_rolling"]


def test_baseline_window_override_invalid_window(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    ds = tmp_path / "ds.parquet"
    _write_deterministic_dataset(ds)
    use_case = RunBaselinesUseCase(
        analytics_run_repository=ParquetAnalyticsRunRepository(silver)
    )
    with pytest.raises(ValueError, match="window must be >= 1"):
        use_case.execute(
            asset="AAPL",
            dataset_path=ds,
            parent_sweep_id="sw_f4",
            split_definitions={"train": ("2024-01-01", "2024-04-09")},
            horizons=[1],
            baselines=["historical_mean_rolling"],
            baseline_windows={"historical_mean_rolling": 0},
        )


def test_baseline_window_override_rejected_on_pointless_baseline(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    ds = tmp_path / "ds.parquet"
    _write_deterministic_dataset(ds)
    use_case = RunBaselinesUseCase(
        analytics_run_repository=ParquetAnalyticsRunRepository(silver)
    )
    with pytest.raises(ValueError, match="zero_return does not accept window"):
        use_case.execute(
            asset="AAPL",
            dataset_path=ds,
            parent_sweep_id="sw_f4",
            split_definitions={"train": ("2024-01-01", "2024-04-09")},
            horizons=[1],
            baselines=["zero_return"],
            baseline_windows={"zero_return": 30},
        )
