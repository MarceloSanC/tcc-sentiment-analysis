from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.adapters.parquet_phase_b_tier_sidecar_writer import (
    ParquetPhaseBTierSidecarWriter,
)
from src.use_cases.compute_phase_b_tier_metrics_use_case import (
    ComputePhaseBTierMetricsUseCase,
)

BASELINES = {
    "baseline_zero_return_v1": 0.04,
    "baseline_historical_mean_rolling_v1": 0.05,
    "baseline_historical_quantiles_rolling_v1": 0.06,
}
POINT_BASELINES = {"baseline_zero_return_v1", "baseline_historical_mean_rolling_v1"}
FOLDS = {
    "wf_1": pd.Timestamp("2020-01-01", tz="UTC"),
    "wf_2": pd.Timestamp("2020-02-01", tz="UTC"),
    "wf_3": pd.Timestamp("2020-03-01", tz="UTC"),
}
SEEDS = [20260517, 20260518, 20260519, 20260520, 20260521]


def _quantile_row(
    run_id: str,
    split: str,
    horizon: int,
    target_ts: pd.Timestamp,
    width: float,
    *,
    fold: str,
    point: bool = False,
) -> dict[str, object]:
    q10, q50, q90 = (width, width, width) if point else (-width, 0.0, width)
    return {
        "run_id": run_id,
        "asset": "AAPL",
        "feature_set_name": "BTSF" if run_id.startswith("tft") else "baseline",
        "split": split,
        "fold": fold,
        "horizon": horizon,
        "target_timestamp_utc": target_ts,
        "y_true": 0.0,
        "quantile_p10_post_guardrail": q10,
        "quantile_p50_post_guardrail": q50,
        "quantile_p90_post_guardrail": q90,
    }


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _build_synthetic_store(tmp_path: Path) -> tuple[Path, Path, Path]:
    cohort = "phase_b_confirmatorio_test"
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    reports = tmp_path / "reports"
    dim_rows: list[dict[str, object]] = []
    snapshot_rows: list[dict[str, object]] = []
    oos_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []

    for fold_name, train_end in FOLDS.items():
        for seed in SEEDS:
            tft_run_id = f"tft_{fold_name}_{seed}"
            dim_rows.append(
                {
                    "run_id": tft_run_id,
                    "asset": "AAPL",
                    "parent_sweep_id": cohort,
                    "feature_set_name": "BTSF",
                    "model_version": "tft_phase_b",
                    "fold": None,
                    "seed": seed,
                    "status": "ok",
                }
            )
            snapshot_rows.append(
                {
                    "run_id": tft_run_id,
                    "asset": "AAPL",
                    "parent_sweep_id": cohort,
                    "train_end_utc": train_end,
                }
            )
            for split, offset in [("val", 10), ("test", 30)]:
                for horizon in [1, 7]:
                    for idx in range(12):
                        target_ts = train_end + pd.Timedelta(days=offset + idx + horizon)
                        oos_rows.append(
                            _quantile_row(
                                tft_run_id,
                                split,
                                horizon,
                                target_ts,
                                0.03,
                                fold="none",
                            )
                        )
            for horizon in [1, 7]:
                calibration_rows.append(
                    {
                        "run_id": tft_run_id,
                        "asset": "AAPL",
                        "feature_set_name": "BTSF",
                        "parent_sweep_id": cohort,
                        "split": "test",
                        "horizon": horizon,
                        "mean_pinball_post_guardrail": 0.002,
                    }
                )

            for baseline_model_version, width in BASELINES.items():
                baseline_run_id = f"{baseline_model_version}_{fold_name}_{seed}"
                dim_rows.append(
                    {
                        "run_id": baseline_run_id,
                        "asset": "AAPL",
                        "parent_sweep_id": cohort,
                        "feature_set_name": "baseline",
                        "model_version": baseline_model_version,
                        "fold": fold_name,
                        "seed": seed,
                        "status": "ok",
                    }
                )
                snapshot_rows.append(
                    {
                        "run_id": baseline_run_id,
                        "asset": "AAPL",
                        "parent_sweep_id": cohort,
                        "train_end_utc": train_end,
                    }
                )
                for split, offset in [("val", 10), ("test", 30)]:
                    for horizon in [1, 7]:
                        for idx in range(12):
                            target_ts = train_end + pd.Timedelta(days=offset + idx + horizon)
                            oos_rows.append(
                                _quantile_row(
                                    baseline_run_id,
                                    split,
                                    horizon,
                                    target_ts,
                                    width,
                                    fold=fold_name,
                                    point=baseline_model_version in POINT_BASELINES,
                                )
                            )
                for horizon in [1, 7]:
                    calibration_rows.append(
                        {
                            "run_id": baseline_run_id,
                            "asset": "AAPL",
                            "feature_set_name": "baseline",
                            "parent_sweep_id": cohort,
                            "split": "test",
                            "horizon": horizon,
                            "mean_pinball_post_guardrail": (
                                float("nan")
                                if baseline_model_version in POINT_BASELINES
                                else width / 20.0
                            ),
                        }
                    )

    _write_parquet(
        silver / "dim_run" / "asset=AAPL" / f"sweep_id={cohort}" / "dim_run.parquet",
        pd.DataFrame(dim_rows),
    )
    _write_parquet(
        silver / "fact_run_snapshot" / "asset=AAPL" / f"sweep_id={cohort}" / "fact_run_snapshot.parquet",
        pd.DataFrame(snapshot_rows),
    )
    _write_parquet(
        silver / "fact_oos_predictions" / "asset=AAPL" / "feature_set_name=all" / "year=2020" / "fact_oos_predictions.parquet",
        pd.DataFrame(oos_rows),
    )
    _write_parquet(
        gold / "gold_prediction_calibration.parquet",
        pd.DataFrame(calibration_rows),
    )
    return silver, gold, reports


def test_compute_phase_b_tier_metrics_use_case_writes_five_sidecars(tmp_path: Path) -> None:
    silver, gold, reports = _build_synthetic_store(tmp_path)
    writer = ParquetPhaseBTierSidecarWriter(reports)
    use_case = ComputePhaseBTierMetricsUseCase(
        silver_dir=silver,
        gold_dir=gold,
        sidecar_writer=writer,
    )

    result = use_case.execute(asset="AAPL", parent_sweep_id="phase_b_confirmatorio_test")

    assert set(result.sidecar_paths) == {
        "marginal_coverage",
        "dm_family_6",
        "dm_family_18_sensitivity",
        "delta_pinball",
        "tier_verdict",
    }
    for path in result.sidecar_paths.values():
        assert path.exists()

    base = reports / "cohort=phase_b_confirmatorio_test"
    marginal = pd.read_parquet(base / "phase_b_marginal_coverage.parquet")
    dm6 = pd.read_parquet(base / "phase_b_dm_family_6.parquet")
    dm18 = pd.read_parquet(base / "phase_b_dm_family_18_sensitivity.parquet")
    delta = pd.read_parquet(base / "phase_b_delta_pinball.parquet")
    verdict = pd.read_parquet(base / "phase_b_tier_verdict.parquet")

    assert len(marginal) == 60
    assert len(dm6) == 6
    assert len(dm18) == 18
    assert len(delta) == 6
    assert len(verdict) == 6
    assert verdict["tier"].isin(["tier_1", "tier_2", "refutado"]).all()
    assert set(verdict["hypothesis"]) == {"H1", "H2a", "H2b"}

    assert (dm6["pvalue_one_sided_less"] >= 0.0).all()
    assert (dm6["pvalue_one_sided_less"] <= 1.0).all()
    assert dm6["n_obs_effective"].gt(0).all()
    assert dm6["pvalue_one_sided_less"].notna().all()
    assert dm18["n_obs_effective"].gt(0).all()
    assert dm18["pvalue_one_sided_less"].notna().all()
    assert (dm6["pvalue_adj_holm"] >= dm6["pvalue_one_sided_less"]).all()
    ordered = dm6.sort_values("pvalue_one_sided_less")
    assert ordered["pvalue_adj_holm"].is_monotonic_increasing
    assert dm18["analysis_role"].eq("sensitivity_conservative").all()
    assert delta["delta_mean_pinball_rel"].notna().all()


def test_compute_phase_b_tier_metrics_use_case_integrity_gate_rejects_invalid_dm_and_delta() -> None:
    valid_dm = pd.DataFrame(
        {
            "n_obs_effective": [10],
            "dm_stat": [-2.0],
            "pvalue_one_sided_less": [0.02],
            "pvalue_adj_holm": [0.04],
        }
    )
    valid_delta = pd.DataFrame({"delta_mean_pinball_rel": [0.1]})

    zero_n_obs = valid_dm.copy()
    zero_n_obs["n_obs_effective"] = [0]
    with pytest.raises(ValueError, match="zero n_obs_effective"):
        ComputePhaseBTierMetricsUseCase._validate_pre_write_integrity(
            zero_n_obs,
            valid_dm,
            valid_delta,
        )

    nan_dm = valid_dm.copy()
    nan_dm["pvalue_one_sided_less"] = [float("nan")]
    with pytest.raises(ValueError, match="NaN statistics"):
        ComputePhaseBTierMetricsUseCase._validate_pre_write_integrity(
            nan_dm,
            valid_dm,
            valid_delta,
        )

    nan_delta = pd.DataFrame({"delta_mean_pinball_rel": [float("nan")]})
    with pytest.raises(ValueError, match="delta_pinball has NaN"):
        ComputePhaseBTierMetricsUseCase._validate_pre_write_integrity(
            valid_dm,
            valid_dm,
            nan_delta,
        )
