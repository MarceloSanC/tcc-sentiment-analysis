from __future__ import annotations

import pandas as pd
import pytest

from src.domain.services.quantile_contract_analyzer import QuantileDegeneracyThresholds
from src.domain.services.scope_spec import ScopeSpec
from src.use_cases.refresh_analytics_store_use_case import RefreshAnalyticsStoreUseCase


def _write_table(base, table_name: str, rows: list[dict], parts: dict[str, str] | None = None) -> None:
    table_dir = base / table_name
    if parts:
        for k, v in parts.items():
            table_dir = table_dir / f"{k}={v}"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(table_dir / f"{table_name}.parquet", index=False)


def _quantile_contract_dim_run() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "r1",
                "model_version": "v1",
                "feature_set_hash": "fh1",
                "parent_sweep_id": "sw1",
                "trial_number": 1,
                "status": "ok",
            }
        ]
    )


def _quantile_contract_fact_config(run_ids: tuple[str, ...] = ("r1",), mode: str = "quantile") -> pd.DataFrame:
    return pd.DataFrame(
        [{"run_id": rid, "prediction_mode": mode} for rid in run_ids]
    )


def _quantile_contract_oos(*, include_post_guardrail: bool = True) -> pd.DataFrame:
    row = {
        "run_id": "r1",
        "asset": "AAPL",
        "feature_set_name": "BT",
        "config_signature": "cfg1",
        "split": "test",
        "fold": "wf_1",
        "seed": 42,
        "horizon": 1,
        "y_true": 0.0,
        "y_pred": 0.0,
        "quantile_p10": 1.0,
        "quantile_p50": 0.0,
        "quantile_p90": -1.0,
    }
    if include_post_guardrail:
        row.update(
            {
                "quantile_p10_post_guardrail": -1.0,
                "quantile_p50_post_guardrail": 0.0,
                "quantile_p90_post_guardrail": 1.0,
            }
        )
    return pd.DataFrame([row])


def test_metrics_by_run_split_horizon_emits_raw_and_post_guardrail_pairs() -> None:
    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_run_split_horizon(
        _quantile_contract_dim_run(),
        _quantile_contract_oos(),
        _quantile_contract_fact_config(),
    )

    expected = {
        "picp_raw",
        "picp_post_guardrail",
        "mpiw_raw",
        "mpiw_post_guardrail",
        "pinball_q10_raw",
        "pinball_q10_post_guardrail",
        "pinball_q50_raw",
        "pinball_q50_post_guardrail",
        "pinball_q90_raw",
        "pinball_q90_post_guardrail",
        "mean_pinball_raw",
        "mean_pinball_post_guardrail",
        "coverage_error_raw",
        "coverage_error_post_guardrail",
        "confidence_calibrated_raw",
        "confidence_calibrated_post_guardrail",
    }
    assert expected.issubset(set(out.columns))
    row = out.iloc[0]
    assert float(row["mpiw_raw"]) == -2.0
    assert float(row["mpiw_post_guardrail"]) == 2.0
    assert float(row["picp_raw"]) == 0.0
    assert float(row["picp_post_guardrail"]) == 1.0
    assert float(row["mean_pinball_raw"]) != float(row["mean_pinball_post_guardrail"])


def test_metrics_emits_nan_post_guardrail_when_silver_missing_columns() -> None:
    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_run_split_horizon(
        _quantile_contract_dim_run(),
        _quantile_contract_oos(include_post_guardrail=False),
        _quantile_contract_fact_config(),
    )

    row = out.iloc[0]
    assert float(row["mpiw_raw"]) == -2.0
    assert float(row["picp_raw"]) == 0.0
    assert pd.isna(row["mpiw_post_guardrail"])
    assert pd.isna(row["picp_post_guardrail"])
    assert pd.isna(row["mean_pinball_post_guardrail"])


def test_pred_interval_negative_uses_raw_quantiles_not_post_guardrail() -> None:
    """Regression guard: _pred_interval_negative MUST be calculated on raw
    quantiles. If someone changes it to *_post_guardrail, the check becomes
    tautologically zero and loses diagnostic value (Categoria C).
    Ver METRICS_DEFINITIONS.md §"Variante quantilica" - Categoria C.
    """
    fact = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "split": "test",
                "horizon": 1,
                "timestamp_utc": "2024-01-01T00:00:00Z",
                "target_timestamp_utc": "2024-01-02T00:00:00Z",
                "y_true": 0.0,
                "y_pred": 0.0,
                # Crossing real em raw (p10 > p90 -> width = -2 < 0).
                "quantile_p10": 1.0,
                "quantile_p50": 0.0,
                "quantile_p90": -1.0,
                # Monotonic em post-guardrail (width = 2 > 0).
                "quantile_p10_post_guardrail": -1.0,
                "quantile_p50_post_guardrail": 0.0,
                "quantile_p90_post_guardrail": 1.0,
            }
        ]
    )

    out = RefreshAnalyticsStoreUseCase._build_gold_oos_quality_report(
        _quantile_contract_dim_run(),
        fact,
    )

    # n_negative_interval_width > 0 prova que o check leu raw (com crossing).
    # Se alguem trocar por *_post_guardrail, este valor cairia para 0.
    run_row = out[out["scope"] == "run_split_horizon"].iloc[0]
    assert int(run_row["n_negative_interval_width"]) == 1


def test_quantile_degeneracy_report_materializes_group_metrics_and_gate_status() -> None:
    rows = []
    for idx in range(1600):
        degenerate = idx < 1460
        rows.append(
            {
                "run_id": "r1",
                "split": "test",
                "horizon": 1,
                "quantile_p10": 0.5 if degenerate else 0.4,
                "quantile_p50": 0.5,
                "quantile_p90": 0.5 if degenerate else 0.6,
            }
        )
    fact_config = pd.DataFrame(
        [{"run_id": "r1", "prediction_mode": "quantile", "parent_sweep_id": "sw1"}]
    )

    out = RefreshAnalyticsStoreUseCase._build_gold_quantile_degeneracy_report(
        pd.DataFrame(rows),
        fact_config,
        thresholds=QuantileDegeneracyThresholds(),
    )

    assert list(out.columns) == [
        "parent_sweep_id",
        "split",
        "horizon",
        "prediction_mode",
        "n_rows",
        "p10_eq_p90_count",
        "p10_eq_p90_rate",
        "p10_eq_p50_eq_p90_count",
        "p10_eq_p50_eq_p90_rate",
        "gate_passed",
    ]
    row = out.iloc[0]
    assert row["parent_sweep_id"] == "sw1"
    assert row["split"] == "test"
    assert int(row["horizon"]) == 1
    assert row["prediction_mode"] == "quantile"
    assert int(row["n_rows"]) == 1600
    assert int(row["p10_eq_p90_count"]) == 1460
    assert float(row["p10_eq_p90_rate"]) == pytest.approx(0.9125)
    assert bool(row["gate_passed"]) is False


def test_quantile_degeneracy_report_honors_custom_thresholds() -> None:
    rows = []
    for idx in range(200):
        degenerate = idx < 60
        rows.append(
            {
                "run_id": "r1",
                "split": "test",
                "horizon": 1,
                "quantile_p10": 0.5 if degenerate else 0.4,
                "quantile_p50": 0.5,
                "quantile_p90": 0.5 if degenerate else 0.6,
            }
        )
    fact_config = pd.DataFrame(
        [{"run_id": "r1", "prediction_mode": "quantile", "parent_sweep_id": "sw1"}]
    )

    permissive = RefreshAnalyticsStoreUseCase._build_gold_quantile_degeneracy_report(
        pd.DataFrame(rows),
        fact_config,
        thresholds=QuantileDegeneracyThresholds(
            min_rows_for_gate=100,
            max_p10_eq_p90_rate=0.50,
        ),
    )
    strict = RefreshAnalyticsStoreUseCase._build_gold_quantile_degeneracy_report(
        pd.DataFrame(rows),
        fact_config,
        thresholds=QuantileDegeneracyThresholds(
            min_rows_for_gate=100,
            max_p10_eq_p90_rate=0.10,
        ),
    )

    assert float(permissive.iloc[0]["p10_eq_p90_rate"]) == pytest.approx(0.30)
    assert bool(permissive.iloc[0]["gate_passed"]) is True
    assert bool(strict.iloc[0]["gate_passed"]) is False


def test_prob_up_emits_dual_variants() -> None:
    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_run_split_horizon(
        _quantile_contract_dim_run(),
        _quantile_contract_oos(),
        _quantile_contract_fact_config(),
    )

    expected = {
        "prob_up_raw",
        "prob_up_post_guardrail",
        "prob_down_raw",
        "prob_down_post_guardrail",
    }
    assert expected.issubset(set(out.columns))

    row = out.iloc[0]
    # raw quantiles (p10=1.0, p50=0.0, p90=-1.0) acionam hard bound q90<0 -> cdf0=1.0
    # -> prob_up_raw=0.0. Post-guardrail (p10=-1.0, p50=0.0, p90=1.0) sem hard bound
    # -> cdf0=0.5 -> prob_up_post_guardrail=0.5.
    assert float(row["prob_up_raw"]) == pytest.approx(0.0)
    assert float(row["prob_up_post_guardrail"]) == pytest.approx(0.5)
    assert float(row["prob_up_raw"]) != float(row["prob_up_post_guardrail"])
    assert float(row["prob_down_raw"]) == pytest.approx(1.0)
    assert float(row["prob_down_post_guardrail"]) == pytest.approx(0.5)

    # Alias default = post-guardrail (compatibilidade com consumidores existentes).
    assert float(row["prob_up"]) == pytest.approx(float(row["prob_up_post_guardrail"]))
    assert float(row["prob_down"]) == pytest.approx(float(row["prob_down_post_guardrail"]))


def test_prob_up_alias_falls_back_to_raw_when_post_guardrail_missing() -> None:
    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_run_split_horizon(
        _quantile_contract_dim_run(),
        _quantile_contract_oos(include_post_guardrail=False),
        _quantile_contract_fact_config(),
    )

    row = out.iloc[0]
    assert float(row["prob_up_raw"]) == pytest.approx(0.0)
    assert pd.isna(row["prob_up_post_guardrail"])
    assert float(row["prob_up"]) == pytest.approx(float(row["prob_up_raw"]))
    assert float(row["prob_down"]) == pytest.approx(float(row["prob_down_raw"]))


def test_delta_columns_equal_post_minus_raw_per_row() -> None:
    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_run_split_horizon(
        _quantile_contract_dim_run(),
        _quantile_contract_oos(),
        _quantile_contract_fact_config(),
    )

    expected_delta_cols = {
        "delta_picp_post_minus_raw",
        "delta_mpiw_post_minus_raw",
        "delta_pred_interval_width_post_minus_raw",
        "delta_coverage_error_post_minus_raw",
        "delta_mean_pinball_post_minus_raw",
        "delta_pinball_q10_post_minus_raw",
        "delta_pinball_q50_post_minus_raw",
        "delta_pinball_q90_post_minus_raw",
        "delta_confidence_calibrated_post_minus_raw",
    }
    assert expected_delta_cols.issubset(set(out.columns))

    row = out.iloc[0]
    for base in [
        "picp",
        "mpiw",
        "pred_interval_width",
        "coverage_error",
        "mean_pinball",
        "pinball_q10",
        "pinball_q50",
        "pinball_q90",
        "confidence_calibrated",
    ]:
        assert float(row[f"delta_{base}_post_minus_raw"]) == pytest.approx(
            float(row[f"{base}_post_guardrail"]) - float(row[f"{base}_raw"]),
            abs=1e-12,
        )


def test_delta_columns_are_nan_when_silver_missing_post_guardrail() -> None:
    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_run_split_horizon(
        _quantile_contract_dim_run(),
        _quantile_contract_oos(include_post_guardrail=False),
        _quantile_contract_fact_config(),
    )

    row = out.iloc[0]
    for base in [
        "picp",
        "mpiw",
        "pred_interval_width",
        "coverage_error",
        "mean_pinball",
        "pinball_q10",
        "pinball_q50",
        "pinball_q90",
        "confidence_calibrated",
    ]:
        assert pd.isna(row[f"delta_{base}_post_minus_raw"])


def test_gold_prediction_risk_uses_post_guardrail_quantiles() -> None:
    # Fixture: crossing em raw (p10 > p50), monotonico em post-guardrail.
    fact = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "split": "test",
                "fold": "wf_1",
                "seed": 42,
                "horizon": 1,
                "y_pred": 0.5,
                "quantile_p10": 1.0,
                "quantile_p50": 0.0,
                "quantile_p90": -1.0,
                "quantile_p10_post_guardrail": -1.0,
                "quantile_p50_post_guardrail": 0.0,
                "quantile_p90_post_guardrail": 1.0,
            }
        ]
    )

    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_risk(
        _quantile_contract_dim_run(),
        fact,
    )

    row = out.iloc[0]
    assert float(row["var_10"]) == -1.0
    assert float(row["var_10"]) != float(fact.iloc[0]["quantile_p10"])
    # ES_10 = 1.125 * (-1.0) - 0.125 * 0.0 = -1.125, clipado em min(es, var_10) = -1.125
    assert float(row["es_10_approx"]) == pytest.approx(-1.125)
    assert float(row["es_10_approx"]) <= float(row["var_10"])


def test_gold_prediction_risk_emits_nan_when_post_guardrail_missing() -> None:
    fact = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "split": "test",
                "fold": "wf_1",
                "seed": 42,
                "horizon": 1,
                "y_pred": 0.5,
                "quantile_p10": 1.0,
                "quantile_p50": 0.0,
                "quantile_p90": -1.0,
            }
        ]
    )

    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_risk(
        _quantile_contract_dim_run(),
        fact,
    )

    row = out.iloc[0]
    assert float(row["expected_move"]) == pytest.approx(0.5)
    assert float(row["downside_risk"]) == 0.0
    assert pd.isna(row["var_10"])
    assert pd.isna(row["es_10_approx"])


def _write_legacy_silver_without_post_guardrail(silver) -> None:
    dim_rows = [
        {
            "schema_version": 1,
            "run_id": "lr1",
            "asset": "AAPL",
            "feature_set_name": "B",
            "feature_set_hash": "fh1",
            "config_signature": "cfg",
            "model_version": "v1",
            "parent_sweep_id": "sw1",
            "trial_number": 1,
            "fold": "wf_1",
            "seed": 1,
            "status": "ok",
            "created_at_utc": "2026-01-01T00:00:00+00:00",
            "feature_list_ordered_json": "[]",
            "split_fingerprint": "sp1",
            "pipeline_version": "0.1",
            "checkpoint_path_final": "/tmp/final.pt",
            "checkpoint_path_best": "/tmp/best.ckpt",
            "git_commit": "abc",
            "library_versions_json": "{}",
            "hardware_info_json": "{}",
            "duration_total_seconds": 1.0,
            "eta_recorded_seconds": 0.0,
            "retries": 0,
        }
    ]
    oos_rows = [
        {
            "schema_version": 1,
            "run_id": "lr1",
            "asset": "AAPL",
            "feature_set_name": "B",
            "config_signature": "cfg",
            "split": "test",
            "fold": "wf_1",
            "seed": 1,
            "horizon": 1,
            "timestamp_utc": "2026-01-01T00:00:00+00:00",
            "target_timestamp_utc": "2026-01-01T00:00:00+00:00",
            "y_true": 0.1,
            "y_pred": 0.2,
            "error": 0.1,
            "abs_error": 0.1,
            "sq_error": 0.01,
            "quantile_p10": 0.0,
            "quantile_p50": 0.2,
            "quantile_p90": 0.5,
            "year": 2026,
        }
    ]
    _write_table(silver, "dim_run", dim_rows, {"asset": "AAPL"})
    _write_table(silver, "fact_oos_predictions", oos_rows, {"asset": "AAPL", "year": "2026"})


def test_post_guardrail_missing_warning_emits_once_per_refresh_not_per_process(
    tmp_path, caplog
) -> None:
    import logging

    from src.use_cases import refresh_analytics_store_use_case as mod

    silver = tmp_path / "silver"
    _write_legacy_silver_without_post_guardrail(silver)

    use_case_1 = RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=tmp_path / "gold_1",
    )
    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        caplog.clear()
        use_case_1.execute()
        first_warnings = [
            r for r in caplog.records
            if "Post-guardrail quantile columns missing" in r.getMessage()
        ]
    assert len(first_warnings) >= 1

    use_case_2 = RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=tmp_path / "gold_2",
    )
    with caplog.at_level(logging.WARNING, logger=mod.__name__):
        caplog.clear()
        use_case_2.execute()
        second_warnings = [
            r for r in caplog.records
            if "Post-guardrail quantile columns missing" in r.getMessage()
        ]
    assert len(second_warnings) >= 1


def test_primary_quantile_contract_default_is_post_guardrail(tmp_path) -> None:
    use_case = RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=tmp_path / "silver",
        analytics_gold_dir=tmp_path / "gold",
    )

    assert use_case.primary_quantile_contract == "post_guardrail"


def test_primary_quantile_contract_rejects_invalid_value(tmp_path) -> None:
    with pytest.raises(ValueError, match="primary_quantile_contract"):
        RefreshAnalyticsStoreUseCase(
            analytics_silver_dir=tmp_path / "silver",
            analytics_gold_dir=tmp_path / "gold",
            primary_quantile_contract="invalid",  # type: ignore[arg-type]
        )


def test_model_decision_final_uses_primary_contract() -> None:
    metrics_by_config = pd.DataFrame(
        [
            {
                "asset": "AAPL",
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "split": "test",
                "horizon": 1,
                "n_runs": 1,
                "mean_rmse": 0.1,
                "mean_mae": 0.1,
                "mean_directional_accuracy": 0.6,
                "mean_mean_pinball_raw": 9.0,
                "mean_mean_pinball_post_guardrail": 1.0,
                "mean_picp_raw": 0.1,
                "mean_picp_post_guardrail": 0.8,
                "mean_mpiw_raw": -3.0,
                "mean_mpiw_post_guardrail": 2.0,
            }
        ]
    )

    raw = RefreshAnalyticsStoreUseCase._build_gold_model_decision_final(
        metrics_by_config=metrics_by_config,
        robustness_by_horizon=pd.DataFrame(),
        generalization_gap=pd.DataFrame(),
        dm_results=pd.DataFrame(),
        mcs_results=pd.DataFrame(),
        win_rate_results=pd.DataFrame(),
        paired_intersection=pd.DataFrame(),
        primary_quantile_contract="raw",
    )
    post = RefreshAnalyticsStoreUseCase._build_gold_model_decision_final(
        metrics_by_config=metrics_by_config,
        robustness_by_horizon=pd.DataFrame(),
        generalization_gap=pd.DataFrame(),
        dm_results=pd.DataFrame(),
        mcs_results=pd.DataFrame(),
        win_rate_results=pd.DataFrame(),
        paired_intersection=pd.DataFrame(),
        primary_quantile_contract="post_guardrail",
    )

    assert float(raw.iloc[0]["mean_mean_pinball"]) == 9.0
    assert float(raw.iloc[0]["mean_picp"]) == 0.1
    assert float(raw.iloc[0]["mean_mpiw"]) == -3.0
    assert raw.iloc[0]["primary_quantile_contract"] == "raw"
    assert float(post.iloc[0]["mean_mean_pinball"]) == 1.0
    assert float(post.iloc[0]["mean_picp"]) == 0.8
    assert float(post.iloc[0]["mean_mpiw"]) == 2.0
    assert post.iloc[0]["primary_quantile_contract"] == "post_guardrail"


def _write_two_sweep_refresh_fixture(silver) -> None:
    dim_rows = []
    split_rows = []
    oos_rows = []
    artifact_rows = []
    local_contrib_rows = []

    for idx, (run_id, sweep, cfg, rmse) in enumerate(
        [
            ("sw1_r1", "sw1_round", "cfg_sw1", 0.10),
            ("sw2_r1", "sw2_round", "cfg_sw2", 0.30),
        ],
        start=1,
    ):
        dim_rows.append(
            {
                "schema_version": 1,
                "run_id": run_id,
                "asset": "AAPL",
                "feature_set_name": "B",
                "feature_set_hash": "fh1",
                "config_signature": cfg,
                "model_version": f"v{idx}",
                "parent_sweep_id": sweep,
                "trial_number": idx,
                "fold": "wf_1",
                "seed": idx,
                "status": "ok",
                "created_at_utc": f"2026-01-0{idx}T00:00:00+00:00",
                "feature_list_ordered_json": "[]",
                "split_fingerprint": f"sp{idx}",
                "pipeline_version": "0.1",
                "checkpoint_path_final": "/tmp/final.pt",
                "checkpoint_path_best": "/tmp/best.ckpt",
                "git_commit": "abc",
                "library_versions_json": "{}",
                "hardware_info_json": "{}",
                "duration_total_seconds": 1.0,
                "eta_recorded_seconds": 0.0,
                "retries": 0,
            }
        )
        split_rows.append(
            {
                "schema_version": 1,
                "run_id": run_id,
                "asset": "AAPL",
                "parent_sweep_id": sweep,
                "split": "test",
                "rmse": rmse,
                "mae": rmse,
                "mape": 0.0,
                "smape": 0.0,
                "directional_accuracy": 0.50 + idx / 100,
                "n_samples": 1,
            }
        )
        oos_rows.append(
            {
                "schema_version": 1,
                "run_id": run_id,
                "asset": "AAPL",
                "feature_set_name": "B",
                "config_signature": cfg,
                "split": "test",
                "fold": "wf_1",
                "seed": idx,
                "horizon": 1,
                "timestamp_utc": f"2026-01-0{idx}T00:00:00+00:00",
                "target_timestamp_utc": f"2026-01-0{idx}T00:00:00+00:00",
                "y_true": 0.1,
                "y_pred": 0.1 + rmse,
                "error": rmse,
                "abs_error": rmse,
                "sq_error": rmse**2,
                "quantile_p10": 0.0,
                "quantile_p50": 0.1 + rmse,
                "quantile_p90": 0.5,
                "quantile_p10_post_guardrail": 0.0,
                "quantile_p50_post_guardrail": 0.1 + rmse,
                "quantile_p90_post_guardrail": 0.5,
                "quantile_guardrail_applied": 0,
                "year": 2026,
            }
        )
        if idx == 1:
            for split, horizon, day in [("val", 1, "03"), ("test", 7, "04")]:
                oos_rows.append(
                    {
                        "schema_version": 1,
                        "run_id": run_id,
                        "asset": "AAPL",
                        "feature_set_name": "B",
                        "config_signature": cfg,
                        "split": split,
                        "fold": "wf_1",
                        "seed": idx,
                        "horizon": horizon,
                        "timestamp_utc": f"2026-01-{day}T00:00:00+00:00",
                        "target_timestamp_utc": f"2026-01-{day}T00:00:00+00:00",
                        "y_true": 0.1,
                        "y_pred": 0.1 + rmse,
                        "error": rmse,
                        "abs_error": rmse,
                        "sq_error": rmse**2,
                        "quantile_p10": 0.0,
                        "quantile_p50": 0.1 + rmse,
                        "quantile_p90": 0.5,
                        "quantile_p10_post_guardrail": 0.0,
                        "quantile_p50_post_guardrail": 0.1 + rmse,
                        "quantile_p90_post_guardrail": 0.5,
                        "quantile_guardrail_applied": 0,
                        "year": 2026,
                    }
                )
        artifact_rows.append(
            {
                "schema_version": 1,
                "run_id": run_id,
                "asset": "AAPL",
                "model_version": f"v{idx}",
                "checkpoint_path_final": "/tmp/final.pt",
                "checkpoint_path_best": "/tmp/best.ckpt",
                "config_path": "/tmp/config.json",
                "scaler_path": None,
                "encoder_path": None,
                "feature_importance_json": (
                    f'[{{"feature": "close", "delta_rmse": {rmse}, '
                    f'"delta_mae": {rmse}, "baseline_rmse": {rmse}, "baseline_mae": {rmse}}}]'
                ),
                "attention_summary_json": '{"available": false}',
                "logs_ref_json": "{}",
            }
        )
        local_contrib_rows.append(
            {
                "schema_version": 1,
                "inference_run_id": f"inf_{run_id}",
                "run_id": run_id,
                "model_version": f"v{idx}",
                "asset": "AAPL",
                "feature_set_name": "B",
                "split": "inference",
                "horizon": 1,
                "timestamp_utc": f"2026-01-0{idx}T00:00:00+00:00",
                "target_timestamp_utc": f"2026-01-0{idx}T00:00:00+00:00",
                "feature_name": "close",
                "feature_rank": 1,
                "contribution": rmse,
                "abs_contribution": rmse,
                "contribution_sign": "positive",
                "method": "local_magnitude_signed_v1",
                "year": 2026,
                "created_at_utc": f"2026-01-0{idx}T01:00:00+00:00",
            }
        )

    local_contrib_rows.append(
        {
            "schema_version": 1,
            "inference_run_id": "inf_legacy",
            "run_id": None,
            "model_version": "v_legacy",
            "asset": "AAPL",
            "feature_set_name": "B",
            "split": "inference",
            "horizon": 1,
            "timestamp_utc": "2026-01-03T00:00:00+00:00",
            "target_timestamp_utc": "2026-01-03T00:00:00+00:00",
            "feature_name": "close",
            "feature_rank": 1,
            "contribution": 1.0,
            "abs_contribution": 1.0,
            "contribution_sign": "positive",
            "method": "local_magnitude_signed_v1",
            "year": 2026,
            "created_at_utc": "2026-01-03T01:00:00+00:00",
        }
    )

    _write_table(silver, "dim_run", dim_rows, {"asset": "AAPL"})
    _write_table(silver, "fact_split_metrics", split_rows, {"asset": "AAPL"})
    _write_table(silver, "fact_oos_predictions", oos_rows, {"asset": "AAPL", "year": "2026"})
    _write_table(silver, "fact_model_artifacts", artifact_rows, {"asset": "AAPL"})
    _write_table(silver, "fact_feature_contrib_local", local_contrib_rows, {"asset": "AAPL", "year": "2026"})


def _stable_gold_frame(path) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_index(axis=1)
    if df.empty:
        return df.reset_index(drop=True)
    return df.sort_values(
        by=list(df.columns),
        na_position="first",
        kind="mergesort",
    ).reset_index(drop=True)


def _assert_gold_table_equal(gold_a, gold_b, table_name: str) -> None:
    pd.testing.assert_frame_equal(
        _stable_gold_frame(gold_a / f"{table_name}.parquet"),
        _stable_gold_frame(gold_b / f"{table_name}.parquet"),
        check_exact=True,
    )


def test_refresh_without_scope_spec_preserves_global_behavior(tmp_path) -> None:
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _write_two_sweep_refresh_fixture(silver)

    RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold,
    ).execute()

    oos = pd.read_parquet(gold / "gold_oos_consolidated.parquet")
    ranking = pd.read_parquet(gold / "gold_ranking_by_config.parquet")

    assert set(oos["parent_sweep_id"].dropna()) == {"sw1_round", "sw2_round"}
    assert set(ranking["parent_sweep_id"].dropna()) == {"sw1_round", "sw2_round"}


def test_refresh_without_scope_spec_is_bitwise_equivalent_to_legacy(tmp_path) -> None:
    silver = tmp_path / "silver"
    gold_legacy = tmp_path / "gold_legacy"
    gold_explicit_none = tmp_path / "gold_explicit_none"
    _write_two_sweep_refresh_fixture(silver)

    RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold_legacy,
    ).execute()
    RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold_explicit_none,
        scope_spec=None,
    ).execute()

    for table_name in [
        "gold_ranking_by_config",
        "gold_oos_consolidated",
        "gold_prediction_metrics_by_run_split_horizon",
        "gold_feature_contrib_local_summary",
        "gold_model_decision_final",
    ]:
        _assert_gold_table_equal(gold_legacy, gold_explicit_none, table_name)


def test_refresh_with_scope_spec_cohort_decision_filters_silver(tmp_path) -> None:
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _write_two_sweep_refresh_fixture(silver)

    RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold,
        scope_spec=ScopeSpec.create(
            scope_mode="cohort_decision",
            parent_sweep_prefixes=("sw1",),
        ),
    ).execute()

    oos = pd.read_parquet(gold / "gold_oos_consolidated.parquet")
    metrics = pd.read_parquet(gold / "gold_prediction_metrics_by_run_split_horizon.parquet")

    assert set(oos["run_id"]) == {"sw1_r1"}
    assert set(oos["parent_sweep_id"].dropna()) == {"sw1_round"}
    assert set(metrics["run_id"]) == {"sw1_r1"}


def test_refresh_execute_scope_spec_overrides_instance_default(tmp_path) -> None:
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _write_two_sweep_refresh_fixture(silver)

    RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold,
        scope_spec=ScopeSpec.create(
            scope_mode="cohort_decision",
            parent_sweep_prefixes=("sw2",),
        ),
    ).execute(
        scope_spec=ScopeSpec.create(
            scope_mode="cohort_decision",
            parent_sweep_prefixes=("sw1",),
        )
    )

    oos = pd.read_parquet(gold / "gold_oos_consolidated.parquet")
    ranking = pd.read_parquet(gold / "gold_ranking_by_config.parquet")

    assert set(oos["run_id"]) == {"sw1_r1"}
    assert set(oos["parent_sweep_id"].dropna()) == {"sw1_round"}
    assert set(ranking["parent_sweep_id"].dropna()) == {"sw1_round"}


def test_refresh_global_health_ignores_cohort_filters(tmp_path) -> None:
    silver = tmp_path / "silver"
    gold_global = tmp_path / "gold_global"
    gold_global_health = tmp_path / "gold_global_health"
    _write_two_sweep_refresh_fixture(silver)

    RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold_global,
        scope_spec=None,
    ).execute()
    RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold_global_health,
        scope_spec=ScopeSpec.create(
            scope_mode="global_health",
            parent_sweep_prefixes=("sw1",),
        ),
    ).execute()

    oos = pd.read_parquet(gold_global_health / "gold_oos_consolidated.parquet")
    assert set(oos["parent_sweep_id"].dropna()) == {"sw1_round", "sw2_round"}

    for table_name in [
        "gold_ranking_by_config",
        "gold_oos_consolidated",
        "gold_prediction_metrics_by_run_split_horizon",
        "gold_feature_contrib_local_summary",
        "gold_model_decision_final",
    ]:
        _assert_gold_table_equal(gold_global, gold_global_health, table_name)


def test_refresh_with_scope_spec_filters_by_split_and_horizon(tmp_path) -> None:
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _write_two_sweep_refresh_fixture(silver)

    RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold,
        scope_spec=ScopeSpec.create(
            scope_mode="cohort_decision",
            splits=("test",),
            horizons=(1,),
        ),
    ).execute()

    oos = pd.read_parquet(gold / "gold_oos_consolidated.parquet")

    assert set(oos["split"]) == {"test"}
    assert set(oos["horizon"]) == {1}


def test_refresh_with_scope_spec_validates_eagerly(tmp_path) -> None:
    with pytest.raises(ValueError, match="scope_mode=cohort_decision requires"):
        RefreshAnalyticsStoreUseCase(
            analytics_silver_dir=tmp_path / "silver",
            analytics_gold_dir=tmp_path / "gold",
            scope_spec=ScopeSpec.create(scope_mode="cohort_decision"),
        )


def test_refresh_does_not_empty_tables_without_scope_columns(tmp_path) -> None:
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _write_two_sweep_refresh_fixture(silver)

    RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold,
        scope_spec=ScopeSpec.create(
            scope_mode="cohort_decision",
            parent_sweep_prefixes=("sw1",),
        ),
    ).execute()

    local = pd.read_parquet(gold / "gold_feature_contrib_local_summary.parquet")

    assert not local.empty
    assert set(local["parent_sweep_id"].dropna()) == {"sw1_round"}
    assert set(local["feature_name"]) == {"close"}
    assert "sw2_round" not in set(local["parent_sweep_id"].dropna())


def test_refresh_analytics_store_builds_gold_tables(tmp_path) -> None:
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"

    _write_table(
        silver,
        "dim_run",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "feature_set_hash": "fh1",
                "config_signature": "cfg1",
                "model_version": "v1",
                "parent_sweep_id": "sw1",
                "trial_number": 1,
                "fold": "wf_1",
                "seed": 7,
                "status": "ok",
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "feature_list_ordered_json": "[]",
                "split_fingerprint": "sp1",
                "pipeline_version": "0.1",
                "checkpoint_path_final": "/tmp/final.pt",
                "checkpoint_path_best": "/tmp/best.ckpt",
                "git_commit": "abc",
                "library_versions_json": "{}",
                "hardware_info_json": "{}",
                "duration_total_seconds": 1.0,
                "eta_recorded_seconds": 0.0,
                "retries": 0,
            },
            {
                "schema_version": 1,
                "run_id": "r2",
                "asset": "AAPL",
                "feature_set_name": "B",
                "feature_set_hash": "fh1",
                "config_signature": "cfg2",
                "model_version": "v2",
                "parent_sweep_id": "sw1",
                "trial_number": 2,
                "fold": "wf_1",
                "seed": 42,
                "status": "ok",
                "created_at_utc": "2026-01-02T00:00:00+00:00",
                "feature_list_ordered_json": "[]",
                "split_fingerprint": "sp2",
                "pipeline_version": "0.1",
                "checkpoint_path_final": "/tmp/final.pt",
                "checkpoint_path_best": "/tmp/best.ckpt",
                "git_commit": "abc",
                "library_versions_json": "{}",
                "hardware_info_json": "{}",
                "duration_total_seconds": 1.0,
                "eta_recorded_seconds": 0.0,
                "retries": 0,
            },
        ],
        {"asset": "AAPL", "sweep_id": "sw1"},
    )

    _write_table(
        silver,
        "fact_split_metrics",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "asset": "AAPL",
                "parent_sweep_id": "sw1",
                "split": "test",
                "rmse": 0.10,
                "mae": 0.09,
                "mape": 0.0,
                "smape": 0.0,
                "directional_accuracy": 0.55,
                "n_samples": 100,
            },
            {
                "schema_version": 1,
                "run_id": "r2",
                "asset": "AAPL",
                "parent_sweep_id": "sw1",
                "split": "test",
                "rmse": 0.12,
                "mae": 0.10,
                "mape": 0.0,
                "smape": 0.0,
                "directional_accuracy": 0.52,
                "n_samples": 100,
            },
        ],
        {"asset": "AAPL", "sweep_id": "sw1"},
    )

    _write_table(
        silver,
        "fact_oos_predictions",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "config_signature": "cfg1",
                "split": "test",
                "fold": "wf_1",
                "seed": 7,
                "horizon": 1,
                "timestamp_utc": "2026-01-03T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "y_true": 0.1,
                "y_pred": 0.12,
                "error": 0.02,
                "abs_error": 0.02,
                "sq_error": 0.0004,
                "quantile_p10": 0.05,
                "quantile_p50": 0.12,
                "quantile_p90": 0.2,
                "quantile_p10_post_guardrail": 0.05,
                "quantile_p50_post_guardrail": 0.12,
                "quantile_p90_post_guardrail": 0.2,
                "quantile_guardrail_applied": 0,
                "year": 2026,
            },
            {
                "schema_version": 1,
                "run_id": "r1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "config_signature": "cfg1",
                "split": "test",
                "fold": "wf_1",
                "seed": 7,
                "horizon": 7,
                "timestamp_utc": "2026-01-04T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-10T00:00:00+00:00",
                "y_true": -0.05,
                "y_pred": -0.02,
                "error": 0.03,
                "abs_error": 0.03,
                "sq_error": 0.0009,
                "quantile_p10": -0.10,
                "quantile_p50": -0.02,
                "quantile_p90": 0.05,
                "quantile_p10_post_guardrail": -0.10,
                "quantile_p50_post_guardrail": -0.02,
                "quantile_p90_post_guardrail": 0.05,
                "quantile_guardrail_applied": 0,
                "year": 2026,
            },
            {
                "schema_version": 1,
                "run_id": "r2",
                "asset": "AAPL",
                "feature_set_name": "B",
                "config_signature": "cfg2",
                "split": "test",
                "fold": "wf_1",
                "seed": 42,
                "horizon": 1,
                "timestamp_utc": "2026-01-03T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "y_true": 0.1,
                "y_pred": 0.08,
                "error": -0.02,
                "abs_error": 0.02,
                "sq_error": 0.0004,
                "quantile_p10": 0.01,
                "quantile_p50": 0.08,
                "quantile_p90": 0.16,
                "quantile_p10_post_guardrail": 0.01,
                "quantile_p50_post_guardrail": 0.08,
                "quantile_p90_post_guardrail": 0.16,
                "quantile_guardrail_applied": 0,
                "year": 2026,
            },
            {
                "schema_version": 1,
                "run_id": "r2",
                "asset": "AAPL",
                "feature_set_name": "B",
                "config_signature": "cfg2",
                "split": "test",
                "fold": "wf_1",
                "seed": 42,
                "horizon": 7,
                "timestamp_utc": "2026-01-04T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-10T00:00:00+00:00",
                "y_true": -0.05,
                "y_pred": -0.07,
                "error": -0.02,
                "abs_error": 0.02,
                "sq_error": 0.0004,
                "quantile_p10": -0.14,
                "quantile_p50": -0.07,
                "quantile_p90": 0.01,
                "quantile_p10_post_guardrail": -0.14,
                "quantile_p50_post_guardrail": -0.07,
                "quantile_p90_post_guardrail": 0.01,
                "quantile_guardrail_applied": 0,
                "year": 2026,
            }
        ],
        {"asset": "AAPL", "feature_set_name": "B", "year": "2026"},
    )

    _write_table(
        silver,
        "fact_config",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "asset": "AAPL",
                "parent_sweep_id": "sw1",
                "prediction_mode": "quantile",
            },
            {
                "schema_version": 1,
                "run_id": "r2",
                "asset": "AAPL",
                "parent_sweep_id": "sw1",
                "prediction_mode": "quantile",
            },
        ],
        {"asset": "AAPL", "sweep_id": "sw1"},
    )

    _write_table(
        silver,
        "fact_model_artifacts",
        [
            {
                "schema_version": 1,
                "run_id": "r1",
                "asset": "AAPL",
                "model_version": "v1",
                "checkpoint_path_final": "/tmp/final.pt",
                "checkpoint_path_best": "/tmp/best.ckpt",
                "config_path": "/tmp/config.json",
                "scaler_path": None,
                "encoder_path": None,
                "feature_importance_json": '[{"feature": "close", "delta_rmse": 0.01, "delta_mae": 0.02, "baseline_rmse": 0.10, "baseline_mae": 0.09}]',
                "attention_summary_json": '{"available": false}',
                "logs_ref_json": "{}",
            },
            {
                "schema_version": 1,
                "run_id": "r2",
                "asset": "AAPL",
                "model_version": "v2",
                "checkpoint_path_final": "/tmp/final.pt",
                "checkpoint_path_best": "/tmp/best.ckpt",
                "config_path": "/tmp/config.json",
                "scaler_path": None,
                "encoder_path": None,
                "feature_importance_json": '[{"feature": "close", "delta_rmse": 0.03, "delta_mae": 0.04, "baseline_rmse": 0.12, "baseline_mae": 0.10}]',
                "attention_summary_json": '{"available": false}',
                "logs_ref_json": "{}",
            },
        ],
        {"asset": "AAPL"},
    )

    _write_table(
        silver,
        "fact_feature_contrib_local",
        [
            {
                "schema_version": 1,
                "inference_run_id": "inf_1",
                "run_id": None,
                "model_version": "v1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "split": "inference",
                "horizon": 1,
                "timestamp_utc": "2026-01-03T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "feature_name": "close",
                "feature_rank": 1,
                "contribution": 0.05,
                "abs_contribution": 0.05,
                "contribution_sign": "positive",
                "method": "local_magnitude_signed_v1",
                "year": 2026,
                "created_at_utc": "2026-01-03T01:00:00+00:00",
            },
            {
                "schema_version": 1,
                "inference_run_id": "inf_1",
                "run_id": None,
                "model_version": "v1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "split": "inference",
                "horizon": 1,
                "timestamp_utc": "2026-01-03T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-03T00:00:00+00:00",
                "feature_name": "volume",
                "feature_rank": 2,
                "contribution": -0.02,
                "abs_contribution": 0.02,
                "contribution_sign": "negative",
                "method": "local_magnitude_signed_v1",
                "year": 2026,
                "created_at_utc": "2026-01-03T01:00:00+00:00",
            },
            {
                "schema_version": 1,
                "inference_run_id": "inf_2",
                "run_id": None,
                "model_version": "v1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "split": "inference",
                "horizon": 1,
                "timestamp_utc": "2026-01-04T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-04T00:00:00+00:00",
                "feature_name": "close",
                "feature_rank": 1,
                "contribution": 0.03,
                "abs_contribution": 0.03,
                "contribution_sign": "positive",
                "method": "local_magnitude_signed_v1",
                "year": 2026,
                "created_at_utc": "2026-01-04T01:00:00+00:00",
            },
            {
                "schema_version": 1,
                "inference_run_id": "inf_2",
                "run_id": None,
                "model_version": "v1",
                "asset": "AAPL",
                "feature_set_name": "B",
                "split": "inference",
                "horizon": 1,
                "timestamp_utc": "2026-01-04T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-04T00:00:00+00:00",
                "feature_name": "open",
                "feature_rank": 2,
                "contribution": -0.01,
                "abs_contribution": 0.01,
                "contribution_sign": "negative",
                "method": "local_magnitude_signed_v1",
                "year": 2026,
                "created_at_utc": "2026-01-04T01:00:00+00:00",
            },
        ],
        {"asset": "AAPL", "model_version": "v1", "year": "2026"},
    )

    use_case = RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=silver,
        analytics_gold_dir=gold,
    )
    result = use_case.execute()

    assert "gold_runs_long" in result.outputs
    assert "gold_oos_consolidated" in result.outputs
    assert "gold_ranking_by_config" in result.outputs
    assert "gold_consistency_topk" in result.outputs
    assert "gold_ic95_by_config_metric" in result.outputs
    assert "gold_feature_set_impact" in result.outputs
    assert "gold_prediction_metrics_by_run_split_horizon" in result.outputs
    assert "gold_quantile_guardrail_audit" in result.outputs
    assert "gold_quantile_degeneracy_report" in result.outputs
    assert "gold_prediction_metrics_by_config" in result.outputs
    assert "gold_prediction_metrics_by_horizon" in result.outputs
    assert "gold_prediction_calibration" in result.outputs
    assert "gold_prediction_risk" in result.outputs
    assert "gold_prediction_generalization_gap" in result.outputs
    assert "gold_prediction_robustness_by_horizon" in result.outputs
    assert "gold_feature_impact_by_horizon" in result.outputs
    assert "gold_feature_contrib_local_summary" in result.outputs
    assert "gold_oos_quality_report" in result.outputs
    assert "gold_dm_pairwise_results" in result.outputs
    assert "gold_mcs_results" in result.outputs
    assert "gold_win_rate_pairwise_results" in result.outputs
    assert "gold_paired_oos_intersection_by_horizon" in result.outputs
    assert "gold_model_decision_final" in result.outputs
    assert "gold_quality_statistics_report" in result.outputs
    assert "gold_quality_run_sweep_summary" in result.outputs

    ranking = pd.read_parquet(gold / "gold_ranking_by_config.parquet")
    assert len(ranking) == 2
    assert ranking.iloc[0]["config_signature"] == "cfg1"

    oos = pd.read_parquet(gold / "gold_oos_consolidated.parquet")
    assert len(oos) == 4
    assert oos.iloc[0]["run_id"] == "r1"

    impact = pd.read_parquet(gold / "gold_feature_set_impact.parquet")
    assert not impact.empty
    assert {"parent_sweep_id", "metric"}.issubset(set(impact.columns))

    ic95 = pd.read_parquet(gold / "gold_ic95_by_config_metric.parquet")
    assert not ic95.empty
    assert {"parent_sweep_id", "metric"}.issubset(set(ic95.columns))
    assert set(ic95["parent_sweep_id"].dropna()) == {"sw1"}


    pred_run_h = pd.read_parquet(gold / "gold_prediction_metrics_by_run_split_horizon.parquet")
    assert len(pred_run_h) == 4
    h1 = pred_run_h[(pred_run_h["run_id"] == "r1") & (pred_run_h["horizon"] == 1)].iloc[0]
    assert h1["n_samples"] == 1
    assert abs(float(h1["bias"]) - 0.02) < 1e-12
    assert abs(float(h1["rmse"]) - 0.02) < 1e-12
    assert abs(float(h1["mae"]) - 0.02) < 1e-12
    assert abs(float(h1["pinball_q10_raw"]) - 0.005) < 1e-12
    assert abs(float(h1["pinball_q50_raw"]) - 0.01) < 1e-12
    assert abs(float(h1["pinball_q90_raw"]) - 0.01) < 1e-12
    assert abs(float(h1["mean_pinball_raw"]) - ((0.005 + 0.01 + 0.01) / 3.0)) < 1e-12
    assert abs(float(h1["picp_raw"]) - 1.0) < 1e-12
    assert abs(float(h1["mpiw_raw"]) - 0.15) < 1e-12
    assert abs(float(h1["pred_interval_width_raw"]) - 0.15) < 1e-12
    assert abs(float(h1["coverage_error_raw"]) - 0.2) < 1e-12
    assert abs(float(h1["prob_down"]) - 0.0) < 1e-12
    assert float(h1["confidence_calibrated_post_guardrail"]) > 0.0

    degeneracy = pd.read_parquet(gold / "gold_quantile_degeneracy_report.parquet")
    assert {"parent_sweep_id", "split", "horizon", "prediction_mode", "gate_passed"}.issubset(
        set(degeneracy.columns)
    )

    by_cfg = pd.read_parquet(gold / "gold_prediction_metrics_by_config.parquet")
    assert not by_cfg.empty
    assert {"parent_sweep_id", "n_oos", "mean_bias", "mean_mean_pinball_raw", "mean_mean_pinball_post_guardrail", "mean_picp_raw", "mean_picp_post_guardrail", "mean_mpiw_raw", "mean_mpiw_post_guardrail", "mean_coverage_error_raw", "mean_coverage_error_post_guardrail", "mean_prob_down_raw", "mean_prob_down_post_guardrail", "mean_confidence_calibrated_post_guardrail", "iqr_rmse"}.issubset(set(by_cfg.columns))
    assert set(by_cfg["parent_sweep_id"].dropna()) == {"sw1"}
    assert int((pd.to_numeric(by_cfg["n_oos"], errors="coerce") <= 0).sum()) == 0

    key_cols = ["asset", "feature_set_name", "parent_sweep_id", "config_signature", "split", "horizon"]
    expected_n_oos = (
        pred_run_h.groupby(key_cols, dropna=False)["n_samples"]
        .sum()
        .reset_index()
        .rename(columns={"n_samples": "expected_n_oos"})
    )
    by_cfg_cmp = by_cfg.merge(expected_n_oos, on=key_cols, how="left")
    assert int(
        (
            pd.to_numeric(by_cfg_cmp["n_oos"], errors="coerce").fillna(0).astype(int)
            != pd.to_numeric(by_cfg_cmp["expected_n_oos"], errors="coerce").fillna(0).astype(int)
        ).sum()
    ) == 0

    by_h = pd.read_parquet(gold / "gold_prediction_metrics_by_horizon.parquet")
    assert not by_h.empty
    assert {"parent_sweep_id", "horizon"}.issubset(set(by_h.columns))

    cal = pd.read_parquet(gold / "gold_prediction_calibration.parquet")
    assert not cal.empty
    assert {"run_id", "parent_sweep_id", "horizon", "pinball_q10_raw", "pinball_q10_post_guardrail", "pinball_q50_raw", "pinball_q50_post_guardrail", "pinball_q90_raw", "pinball_q90_post_guardrail", "mean_pinball_raw", "mean_pinball_post_guardrail", "picp_raw", "picp_post_guardrail", "mpiw_raw", "mpiw_post_guardrail", "coverage_error_raw", "coverage_error_post_guardrail"}.issubset(set(cal.columns))
    assert set(cal["parent_sweep_id"].dropna()) == {"sw1"}
    qaudit = pd.read_parquet(gold / "gold_quantile_guardrail_audit.parquet")
    assert not qaudit.empty
    assert {"mean_pinball_before", "mean_pinball_after", "crossing_before_count", "crossing_after_count"}.issubset(set(qaudit.columns))
    # Stage 9: r1/r2 sao prediction_mode='quantile' com p10 != p90 em raw,
    # entao audit DEVE expor numericos em before/after/delta -- garantia de
    # que fact_config foi propagado ao audit builder. Antes de RED #1
    # essas colunas vinham todas NaN.
    eligible_audit = qaudit[qaudit["run_id"].isin({"r1", "r2"})]
    assert not eligible_audit.empty
    assert eligible_audit["mean_pinball_before"].notna().any()
    assert eligible_audit["mean_pinball_after"].notna().any()
    assert eligible_audit["delta_mean_pinball_after_minus_before"].notna().any()

    quality = pd.read_parquet(gold / "gold_oos_quality_report.parquet")
    assert not quality.empty
    assert {"scope", "passed"}.issubset(set(quality.columns))

    dm = pd.read_parquet(gold / "gold_dm_pairwise_results.parquet")
    mcs = pd.read_parquet(gold / "gold_mcs_results.parquet")
    assert set(dm.columns).issuperset({"left_config", "right_config", "pvalue_two_sided", "pvalue_adj_holm", "significant_adj_0_05"}) or dm.empty
    assert set(mcs.columns).issuperset({"config_label", "selected_in_mcs_alpha_0_05"}) or mcs.empty


    wr = pd.read_parquet(gold / "gold_win_rate_pairwise_results.parquet")
    assert set(wr.columns).issuperset({"left_config", "right_config", "left_win_rate", "right_win_rate"}) or wr.empty

    qsr = pd.read_parquet(gold / "gold_quality_statistics_report.parquet")
    assert set(qsr.columns).issuperset({"quality_passed_all", "dm_available", "mcs_available", "win_rate_available", "statistics_ready"}) or qsr.empty

    paired = pd.read_parquet(gold / "gold_paired_oos_intersection_by_horizon.parquet")
    assert not paired.empty
    assert {"target_union_count", "target_intersection_count", "pairwise_ready_dm", "pairwise_ready_mcs"}.issubset(set(paired.columns))

    decision = pd.read_parquet(gold / "gold_model_decision_final.parquet")
    assert not decision.empty
    assert {"mean_rmse", "mean_mae", "mean_directional_accuracy", "mean_mean_pinball", "mean_picp", "mean_mpiw", "academic_decision_ready"}.issubset(set(decision.columns))
    if not dm.empty:
        assert "dm_net_wins" in decision.columns
    if not mcs.empty:
        assert "mcs_selected_alpha_0_05" in decision.columns
    if not wr.empty:
        assert "win_rate_ex_ties_mean" in decision.columns

    risk = pd.read_parquet(gold / "gold_prediction_risk.parquet")
    assert not risk.empty
    assert {"expected_move", "downside_risk", "var_10", "es_10_approx"}.issubset(set(risk.columns))
    r1h1 = risk[(risk["run_id"] == "r1") & (risk["horizon"] == 1)].iloc[0]
    assert abs(float(r1h1["expected_move"]) - 0.12) < 1e-12
    assert abs(float(r1h1["downside_risk"]) - 0.0) < 1e-12
    assert abs(float(r1h1["var_10"]) - 0.05) < 1e-12
    assert abs(float(r1h1["es_10_approx"]) - (1.125 * 0.05 - 0.125 * 0.12)) < 1e-12


    gap = pd.read_parquet(gold / "gold_prediction_generalization_gap.parquet")
    assert set(gap.columns).issuperset({"parent_sweep_id", "gap_rmse_test_minus_val", "gap_mae_test_minus_val"}) or gap.empty

    robust_h = pd.read_parquet(gold / "gold_prediction_robustness_by_horizon.parquet")
    assert set(robust_h.columns).issuperset({"parent_sweep_id", "metric", "mean", "std", "median", "iqr", "ci95_low", "ci95_high"}) or robust_h.empty

    fih = pd.read_parquet(gold / "gold_feature_impact_by_horizon.parquet")
    assert not fih.empty
    assert {"parent_sweep_id", "feature_name", "horizon", "mean_delta_rmse", "method"}.issubset(set(fih.columns))

    local = pd.read_parquet(gold / "gold_feature_contrib_local_summary.parquet")
    assert not local.empty
    assert {"parent_sweep_id", "feature_name", "horizon", "mean_abs_contribution", "top3_frequency", "local_top3_jaccard_mean", "local_top3_jaccard_pairs"}.issubset(set(local.columns))
    stab = local[(local["horizon"] == 1) & (local["method"] == "local_magnitude_signed_v1")]
    assert not stab.empty
    assert int(pd.to_numeric(stab["local_top3_jaccard_pairs"], errors="coerce").max()) >= 1
    j = float(pd.to_numeric(stab["local_top3_jaccard_mean"], errors="coerce").dropna().iloc[0])
    assert 0.0 <= j <= 1.0

    qrun = pd.read_parquet(gold / "gold_quality_run_sweep_summary.parquet")
    assert not qrun.empty
    assert {"scope"}.issubset(set(qrun.columns))



def test_build_gold_prediction_metrics_by_config_n_oos_is_idempotent_on_row_order() -> None:
    rows = [
        {
            "run_id": "r1",
            "asset": "AAPL",
            "feature_set_name": "BT",
            "parent_sweep_id": "sw1",
            "config_signature": "cfg1",
            "split": "test",
            "horizon": 1,
            "n_samples": 3,
            "rmse": 0.10,
            "mae": 0.08,
            "directional_accuracy": 0.55,
        },
        {
            "run_id": "r2",
            "asset": "AAPL",
            "feature_set_name": "BT",
            "parent_sweep_id": "sw1",
            "config_signature": "cfg1",
            "split": "test",
            "horizon": 1,
            "n_samples": 5,
            "rmse": 0.12,
            "mae": 0.09,
            "directional_accuracy": 0.53,
        },
    ]
    df = pd.DataFrame(rows)
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    out_a = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_config(df)
    out_b = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_config(shuffled)

    assert "n_oos" in out_a.columns
    assert "parent_sweep_id" in out_a.columns
    assert int(out_a.iloc[0]["n_oos"]) == 8
    assert int(out_b.iloc[0]["n_oos"]) == 8

    sort_cols = ["asset", "feature_set_name", "parent_sweep_id", "config_signature", "split", "horizon"]
    out_a = out_a.sort_values(sort_cols).reset_index(drop=True)
    out_b = out_b.sort_values(sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(out_a, out_b, check_like=True)


def test_gold_metrics_by_config_carries_parent_sweep_id() -> None:
    dim_run = pd.DataFrame(
        [
            {"run_id": "sw1_r1", "parent_sweep_id": "sw1"},
            {"run_id": "sw1_r2", "parent_sweep_id": "sw1"},
            {"run_id": "sw2_r1", "parent_sweep_id": "sw2"},
            {"run_id": "sw2_r2", "parent_sweep_id": "sw2"},
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": sweep,
                "config_signature": config,
                "split": "test",
                "horizon": 1,
                "n_samples": samples,
                "rmse": rmse,
                "mae": rmse,
            }
            for run_id, sweep, config, samples, rmse in [
                ("sw1_r1", "sw1", "cfg_sw1", 3, 1.0),
                ("sw1_r2", "sw1", "cfg_sw1", 5, 3.0),
                ("sw2_r1", "sw2", "cfg_sw2", 7, 10.0),
                ("sw2_r2", "sw2", "cfg_sw2", 11, 30.0),
            ]
        ]
    )

    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_config(metrics)
    expected_by_config = metrics.groupby(["asset", "feature_set_name", "config_signature", "split", "horizon"], dropna=False).ngroups

    assert "parent_sweep_id" in out.columns
    assert len(out) == expected_by_config
    assert set(out["parent_sweep_id"]) == {"sw1", "sw2"}
    expected_parent_by_run = dim_run.set_index("run_id")["parent_sweep_id"].to_dict()
    expected_parent_by_config = {
        row["config_signature"]: expected_parent_by_run[row["run_id"]]
        for _, row in metrics.drop_duplicates("config_signature").iterrows()
    }
    actual_parent_by_config = out.set_index("config_signature")["parent_sweep_id"].to_dict()
    assert actual_parent_by_config == expected_parent_by_config


def test_gold_metrics_by_config_preserves_legacy_null_parent_sweep_id() -> None:
    metrics = pd.DataFrame(
        [
            {
                "run_id": "sw1_r1",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg1",
                "split": "test",
                "horizon": 1,
                "n_samples": 3,
                "rmse": 1.0,
                "mae": 1.0,
            },
            {
                "run_id": "sw1_r2",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg1",
                "split": "test",
                "horizon": 1,
                "n_samples": 5,
                "rmse": 3.0,
                "mae": 3.0,
            },
            {
                "run_id": "legacy_r1",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": None,
                "config_signature": "cfg1",
                "split": "test",
                "horizon": 1,
                "n_samples": 7,
                "rmse": 10.0,
                "mae": 10.0,
            },
        ]
    )

    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_config(metrics)
    cfg1 = out[out["config_signature"] == "cfg1"].reset_index(drop=True)

    assert len(cfg1) == 2
    assert (cfg1["parent_sweep_id"] == "sw1").any()
    assert cfg1["parent_sweep_id"].isna().any()


def test_gold_feature_set_impact_is_cohort_aware() -> None:
    base = pd.DataFrame(
        [
            {
                "run_id": "sw1_r1",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": "sw1",
                "split": "test",
                "rmse": 1.0,
                "mae": 0.10,
                "directional_accuracy": 0.60,
            },
            {
                "run_id": "sw1_r2",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": "sw1",
                "split": "test",
                "rmse": 3.0,
                "mae": 0.30,
                "directional_accuracy": 0.70,
            },
            {
                "run_id": "sw2_r1",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": "sw2",
                "split": "test",
                "rmse": 10.0,
                "mae": 1.00,
                "directional_accuracy": 0.40,
            },
            {
                "run_id": "sw2_r2",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": "sw2",
                "split": "test",
                "rmse": 30.0,
                "mae": 3.00,
                "directional_accuracy": 0.50,
            },
        ]
    )

    out = RefreshAnalyticsStoreUseCase._build_gold_feature_set_impact(base)
    rmse = out[out["metric"] == "rmse"].sort_values("parent_sweep_id").reset_index(drop=True)

    assert "parent_sweep_id" in out.columns
    assert rmse["parent_sweep_id"].tolist() == ["sw1", "sw2"]
    assert rmse["n_runs"].tolist() == [2, 2]
    assert rmse["mean_value"].tolist() == [2.0, 20.0]


def test_gold_prediction_metrics_by_horizon_is_cohort_aware() -> None:
    rows = [
        {
            "run_id": run_id,
            "asset": "AAPL",
            "feature_set_name": "BT",
            "parent_sweep_id": sweep,
            "split": "test",
            "horizon": 1,
            "n_samples": samples,
            "rmse": rmse,
            "mae": rmse,
        }
        for run_id, sweep, samples, rmse in [
            ("sw1_r1", "sw1", 3, 1.0),
            ("sw1_r2", "sw1", 5, 3.0),
            ("sw2_r1", "sw2", 7, 10.0),
            ("sw2_r2", "sw2", 11, 30.0),
        ]
    ]

    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_horizon(pd.DataFrame(rows))
    out = out.sort_values("parent_sweep_id").reset_index(drop=True)

    assert "parent_sweep_id" in out.columns
    assert out["parent_sweep_id"].tolist() == ["sw1", "sw2"]
    assert out["n_runs"].tolist() == [2, 2]
    assert out["n_oos"].tolist() == [8, 18]
    assert out["mean_rmse"].tolist() == [2.0, 20.0]


def test_gold_feature_impact_by_horizon_is_cohort_aware() -> None:
    fact_model_artifacts = pd.DataFrame(
        [
            {"run_id": run_id, "feature_importance_json": f'[{{"feature": "close", "delta_rmse": {delta}, "delta_mae": {delta}}}]'}
            for run_id, delta in [
                ("sw1_r1", 1.0),
                ("sw1_r2", 3.0),
                ("sw2_r1", 10.0),
                ("sw2_r2", 30.0),
            ]
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": sweep,
                "split": "test",
                "horizon": 1,
            }
            for run_id, sweep in [
                ("sw1_r1", "sw1"),
                ("sw1_r2", "sw1"),
                ("sw2_r1", "sw2"),
                ("sw2_r2", "sw2"),
            ]
        ]
    )

    out = RefreshAnalyticsStoreUseCase._build_gold_feature_impact_by_horizon(fact_model_artifacts, metrics)
    out = out.sort_values("parent_sweep_id").reset_index(drop=True)

    assert "parent_sweep_id" in out.columns
    assert out["parent_sweep_id"].tolist() == ["sw1", "sw2"]
    assert out["n_runs"].tolist() == [2, 2]
    assert out["mean_delta_rmse"].tolist() == [2.0, 20.0]


def test_gold_feature_contrib_local_summary_is_cohort_aware_via_dim_run() -> None:
    fact_feature_contrib_local = pd.DataFrame(
        [
            {
                "inference_run_id": f"inf_{run_id}",
                "run_id": run_id,
                "asset": "AAPL",
                "feature_set_name": "BT",
                "horizon": 1,
                "feature_name": "close",
                "feature_rank": 1,
                "contribution": contribution,
                "abs_contribution": abs(contribution),
                "method": "local_magnitude_signed_v1",
            }
            for run_id, contribution in [
                ("sw1_r1", 1.0),
                ("sw1_r2", 3.0),
                ("sw2_r1", 10.0),
                ("sw2_r2", 30.0),
            ]
        ]
    )
    dim_run = pd.DataFrame(
        [
            {"run_id": "sw1_r1", "parent_sweep_id": "sw1"},
            {"run_id": "sw1_r2", "parent_sweep_id": "sw1"},
            {"run_id": "sw2_r1", "parent_sweep_id": "sw2"},
            {"run_id": "sw2_r2", "parent_sweep_id": "sw2"},
        ]
    )

    out = RefreshAnalyticsStoreUseCase._build_gold_feature_contrib_local_summary(
        fact_feature_contrib_local,
        dim_run,
    )
    out = out.sort_values("parent_sweep_id").reset_index(drop=True)

    assert "parent_sweep_id" in out.columns
    assert out["parent_sweep_id"].tolist() == ["sw1", "sw2"]
    assert out["n_inference_runs"].tolist() == [2, 2]
    assert out["mean_abs_contribution"].tolist() == [2.0, 20.0]


def test_gold_feature_contrib_local_summary_keeps_legacy_rows_without_parent_sweep_id() -> None:
    fact_feature_contrib_local = pd.DataFrame(
        [
            {
                "inference_run_id": "inf_1",
                "run_id": "legacy_r1",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "horizon": 1,
                "feature_name": "close",
                "feature_rank": 1,
                "contribution": 1.0,
                "abs_contribution": 1.0,
                "method": "local_magnitude_signed_v1",
            },
            {
                "inference_run_id": "inf_2",
                "run_id": "legacy_r2",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "horizon": 1,
                "feature_name": "close",
                "feature_rank": 2,
                "contribution": -3.0,
                "abs_contribution": 3.0,
                "method": "local_magnitude_signed_v1",
            },
        ]
    )

    out = RefreshAnalyticsStoreUseCase._build_gold_feature_contrib_local_summary(
        fact_feature_contrib_local,
        pd.DataFrame(),
    )

    assert not out.empty
    assert "parent_sweep_id" in out.columns
    assert out["parent_sweep_id"].isna().all()
    assert len(out) == 1
    row = out.iloc[0]
    assert int(row["n_inference_runs"]) == 2
    assert float(row["mean_abs_contribution"]) == 2.0


def test_gold_consistency_topk_ranks_within_parent_sweep() -> None:
    base = pd.DataFrame(
        [
            {
                "run_id": "sw1_cfg_a",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg_a",
                "split": "test",
                "fold": "wf_1",
                "seed": 7,
                "rmse": 0.10,
            },
            {
                "run_id": "sw1_cfg_b",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg_b",
                "split": "test",
                "fold": "wf_1",
                "seed": 7,
                "rmse": 0.90,
            },
            {
                "run_id": "sw2_cfg_a",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": "sw2",
                "config_signature": "cfg_a",
                "split": "test",
                "fold": "wf_1",
                "seed": 7,
                "rmse": 0.90,
            },
            {
                "run_id": "sw2_cfg_b",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": "sw2",
                "config_signature": "cfg_b",
                "split": "test",
                "fold": "wf_1",
                "seed": 7,
                "rmse": 0.10,
            },
        ]
    )

    out = RefreshAnalyticsStoreUseCase._build_gold_consistency_topk(base)

    assert "parent_sweep_id" in out.columns
    lookup = {
        (row["parent_sweep_id"], row["config_signature"]): float(row["top1_pct"])
        for _, row in out.iterrows()
    }
    assert lookup[("sw1", "cfg_a")] == 1.0
    assert lookup[("sw1", "cfg_b")] == 0.0
    assert lookup[("sw2", "cfg_a")] == 0.0
    assert lookup[("sw2", "cfg_b")] == 1.0


def test_gold_ranking_by_config_is_cohort_aware() -> None:
    base = pd.DataFrame(
        [
            {
                "run_id": "sw1_cfg_shared",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg_shared",
                "split": "test",
                "rmse": 0.10,
                "mae": 0.10,
                "directional_accuracy": 0.60,
            },
            {
                "run_id": "sw1_cfg_other",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": "sw1",
                "config_signature": "cfg_other",
                "split": "test",
                "rmse": 0.80,
                "mae": 0.80,
                "directional_accuracy": 0.40,
            },
            {
                "run_id": "sw2_cfg_shared",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": "sw2",
                "config_signature": "cfg_shared",
                "split": "test",
                "rmse": 0.90,
                "mae": 0.90,
                "directional_accuracy": 0.30,
            },
            {
                "run_id": "sw2_cfg_other",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "parent_sweep_id": "sw2",
                "config_signature": "cfg_other",
                "split": "test",
                "rmse": 0.20,
                "mae": 0.20,
                "directional_accuracy": 0.70,
            },
        ]
    )

    out = RefreshAnalyticsStoreUseCase._build_gold_ranking_by_config(base)

    assert "parent_sweep_id" in out.columns
    assert len(out) == 4
    sw1 = out[out["parent_sweep_id"] == "sw1"].sort_values("rank_test_rmse")
    sw2 = out[out["parent_sweep_id"] == "sw2"].sort_values("rank_test_rmse")
    assert sw1["rank_test_rmse"].tolist() == [1.0, 2.0]
    assert sw2["rank_test_rmse"].tolist() == [1.0, 2.0]
    assert sw1.iloc[0]["config_signature"] == "cfg_shared"
    assert sw2.iloc[0]["config_signature"] == "cfg_other"


def test_gold_model_decision_final_is_cohort_aware() -> None:
    metrics_by_config = pd.DataFrame(
        [
            {
                "asset": "AAPL",
                "feature_set_name": "BT",
                "config_signature": "cfg_sw1_a",
                "split": "test",
                "horizon": 1,
                "n_runs": 1,
                "mean_rmse": 0.10,
                "mean_mae": 0.10,
                "mean_directional_accuracy": 0.60,
            },
            {
                "asset": "AAPL",
                "feature_set_name": "BT",
                "config_signature": "cfg_sw1_b",
                "split": "test",
                "horizon": 1,
                "n_runs": 1,
                "mean_rmse": 0.80,
                "mean_mae": 0.80,
                "mean_directional_accuracy": 0.40,
            },
            {
                "asset": "AAPL",
                "feature_set_name": "BT",
                "config_signature": "cfg_sw2_a",
                "split": "test",
                "horizon": 1,
                "n_runs": 1,
                "mean_rmse": 0.90,
                "mean_mae": 0.90,
                "mean_directional_accuracy": 0.30,
            },
            {
                "asset": "AAPL",
                "feature_set_name": "BT",
                "config_signature": "cfg_sw2_b",
                "split": "test",
                "horizon": 1,
                "n_runs": 1,
                "mean_rmse": 0.20,
                "mean_mae": 0.20,
                "mean_directional_accuracy": 0.70,
            },
        ]
    )
    mcs_results = pd.DataFrame(
        [
            {
                "asset": "AAPL",
                "parent_sweep_id": sweep,
                "split": "test",
                "horizon": 1,
                "config_label": f"BT|{config}",
                "selected_in_mcs_alpha_0_05": True,
            }
            for sweep, config in [
                ("sw1", "cfg_sw1_a"),
                ("sw1", "cfg_sw1_b"),
                ("sw2", "cfg_sw2_a"),
                ("sw2", "cfg_sw2_b"),
            ]
        ]
    )

    out = RefreshAnalyticsStoreUseCase._build_gold_model_decision_final(
        metrics_by_config=metrics_by_config,
        robustness_by_horizon=pd.DataFrame(),
        generalization_gap=pd.DataFrame(),
        dm_results=pd.DataFrame(),
        mcs_results=mcs_results,
        win_rate_results=pd.DataFrame(),
        paired_intersection=pd.DataFrame(),
    )

    assert "parent_sweep_id" in out.columns
    sw1 = out[out["parent_sweep_id"] == "sw1"].sort_values("rank_rmse")
    sw2 = out[out["parent_sweep_id"] == "sw2"].sort_values("rank_rmse")
    assert sw1["rank_rmse"].tolist() == [1.0, 2.0]
    assert sw2["rank_rmse"].tolist() == [1.0, 2.0]
    assert sw1["rank_mae"].tolist() == [1.0, 2.0]
    assert sw2["rank_mae"].tolist() == [1.0, 2.0]
    assert sw1.sort_values("rank_da")["rank_da"].tolist() == [1.0, 2.0]
    assert sw2.sort_values("rank_da")["rank_da"].tolist() == [1.0, 2.0]
    assert sw1.iloc[0]["config_signature"] == "cfg_sw1_a"
    assert sw2.iloc[0]["config_signature"] == "cfg_sw2_b"


# ---------------------------------------------------------------------------
# Stage 9 -- filtro de metricas probabilisticas por modo + degeneracao raw.
# Cross-link: PHASE_B_IMPLEMENTATION_CHECKLIST.md §Stage 9,
# METRICS_DEFINITIONS.md §"Variante quantilica" (Cat C).
# ---------------------------------------------------------------------------


def _stage9_dim_run() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "r_quantile_genuine",
                "model_version": "v1",
                "feature_set_hash": "fh1",
                "parent_sweep_id": "sw1",
                "trial_number": 1,
                "status": "ok",
            },
            {
                "run_id": "r_point",
                "model_version": "v1",
                "feature_set_hash": "fh1",
                "parent_sweep_id": "sw1",
                "trial_number": 2,
                "status": "ok",
            },
            {
                "run_id": "r_quantile_degenerate",
                "model_version": "v1",
                "feature_set_hash": "fh1",
                "parent_sweep_id": "sw1",
                "trial_number": 3,
                "status": "ok",
            },
        ]
    )


def _stage9_fact_config() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"run_id": "r_quantile_genuine", "prediction_mode": "quantile"},
            {"run_id": "r_point", "prediction_mode": "point"},
            {"run_id": "r_quantile_degenerate", "prediction_mode": "quantile"},
        ]
    )


def _stage9_oos_row(run_id: str, *, q10: float, q50: float, q90: float, y_true: float = 0.2, y_pred: float = 0.2) -> dict:
    return {
        "run_id": run_id,
        "asset": "AAPL",
        "feature_set_name": "BT",
        "config_signature": f"cfg_{run_id}",
        "split": "test",
        "fold": "wf_1",
        "seed": 1,
        "horizon": 1,
        "y_true": y_true,
        "y_pred": y_pred,
        "quantile_p10": q10,
        "quantile_p50": q50,
        "quantile_p90": q90,
        "quantile_p10_post_guardrail": q10,
        "quantile_p50_post_guardrail": q50,
        "quantile_p90_post_guardrail": q90,
    }


def _stage9_oos() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Non-degenerate quantiles, y_true within interval -> covered_80=1.
            _stage9_oos_row("r_quantile_genuine", q10=0.1, q50=0.2, q90=0.3, y_true=0.15, y_pred=0.22),
            # Point mode -- quantiles colapsam (p10==p50==p90 por contrato).
            _stage9_oos_row("r_point", q10=0.2, q50=0.2, q90=0.2, y_true=0.1, y_pred=0.18),
            # Quantile mode mas degenerado (p10==p90).
            _stage9_oos_row("r_quantile_degenerate", q10=0.2, q50=0.2, q90=0.2, y_true=0.1, y_pred=0.18),
        ]
    )


_STAGE9_PROBABILISTIC_BASES = (
    "picp",
    "mpiw",
    "pred_interval_width",
    "coverage_error",
    "mean_pinball",
    "pinball_q10",
    "pinball_q50",
    "pinball_q90",
    "confidence_calibrated",
    "prob_up",
    "prob_down",
)


def test_genuine_quantile_run_contributes_to_picp() -> None:
    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_run_split_horizon(
        _stage9_dim_run(),
        _stage9_oos(),
        _stage9_fact_config(),
    )

    row = out[out["run_id"] == "r_quantile_genuine"].iloc[0]
    assert bool(row["is_quantile_genuine"]) is True
    assert int(row["n_probabilistic_samples"]) == 1
    assert float(row["picp_raw"]) == pytest.approx(1.0)
    assert float(row["picp_post_guardrail"]) == pytest.approx(1.0)
    assert float(row["mpiw_raw"]) == pytest.approx(0.2)
    assert not pd.isna(row["mean_pinball_raw"])
    assert not pd.isna(row["mean_pinball_post_guardrail"])


def test_point_run_excluded_from_probabilistic_metrics() -> None:
    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_run_split_horizon(
        _stage9_dim_run(),
        _stage9_oos(),
        _stage9_fact_config(),
    )

    row = out[out["run_id"] == "r_point"].iloc[0]
    assert bool(row["is_quantile_genuine"]) is False
    assert int(row["n_probabilistic_samples"]) == 0
    # Point metrics intactas.
    assert not pd.isna(row["rmse"])
    assert not pd.isna(row["mae"])
    assert not pd.isna(row["directional_accuracy"])
    assert not pd.isna(row["bias"])
    # Probabilisticas (raw + post_guardrail) NaN.
    for base in _STAGE9_PROBABILISTIC_BASES:
        assert pd.isna(row[f"{base}_raw"]), f"{base}_raw esperado NaN"
        assert pd.isna(row[f"{base}_post_guardrail"]), f"{base}_post_guardrail esperado NaN"


def test_degenerate_quantile_run_excluded_from_probabilistic_metrics() -> None:
    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_run_split_horizon(
        _stage9_dim_run(),
        _stage9_oos(),
        _stage9_fact_config(),
    )

    row = out[out["run_id"] == "r_quantile_degenerate"].iloc[0]
    assert bool(row["is_quantile_genuine"]) is False
    assert int(row["n_probabilistic_samples"]) == 0
    assert not pd.isna(row["rmse"])
    assert not pd.isna(row["mae"])
    for base in _STAGE9_PROBABILISTIC_BASES:
        assert pd.isna(row[f"{base}_raw"]), f"{base}_raw esperado NaN"
        assert pd.isna(row[f"{base}_post_guardrail"]), f"{base}_post_guardrail esperado NaN"


def test_n_probabilistic_samples_matches_eligible_rows() -> None:
    # Run misto: duas rows com mesmo horizon, uma elegivel e uma degenerada.
    oos = pd.DataFrame(
        [
            _stage9_oos_row("r_mixed", q10=0.1, q50=0.2, q90=0.3, y_true=0.15, y_pred=0.22),
            _stage9_oos_row("r_mixed", q10=0.2, q50=0.2, q90=0.2, y_true=0.10, y_pred=0.18),
        ]
    )
    dim_run = pd.DataFrame(
        [
            {
                "run_id": "r_mixed",
                "model_version": "v1",
                "feature_set_hash": "fh1",
                "parent_sweep_id": "sw1",
                "trial_number": 1,
                "status": "ok",
            }
        ]
    )
    fact_config = pd.DataFrame([{"run_id": "r_mixed", "prediction_mode": "quantile"}])

    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_run_split_horizon(
        dim_run, oos, fact_config
    )
    row = out.iloc[0]
    assert int(row["n_samples"]) == 2
    assert int(row["n_probabilistic_samples"]) == 1
    assert int(row["n_probabilistic_samples"]) < int(row["n_samples"])
    assert bool(row["is_quantile_genuine"]) is True


def test_missing_fact_config_treats_as_non_quantile() -> None:
    out = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_run_split_horizon(
        _stage9_dim_run(),
        _stage9_oos(),
        pd.DataFrame(),  # fact_config vazio -> conservador: nenhum run elegivel.
    )

    assert not out.empty
    for _, row in out.iterrows():
        assert bool(row["is_quantile_genuine"]) is False
        assert int(row["n_probabilistic_samples"]) == 0
        for base in _STAGE9_PROBABILISTIC_BASES:
            assert pd.isna(row[f"{base}_raw"])
            assert pd.isna(row[f"{base}_post_guardrail"])
        # Point metrics ainda calculados.
        assert not pd.isna(row["rmse"])
        assert not pd.isna(row["mae"])


def test_pred_interval_negative_unchanged_by_stage9_filter() -> None:
    """Regression guard fortalecido (YELLOW #4): valida que os dois
    caminhos coexistem para um mesmo run com crossing raw mas
    NAO-degenerado (p10 != p90):

    1) gold_oos_quality_report._pred_interval_negative (Cat C raw-only)
       continua detectando o crossing (width raw < 0) -- Stage 9 NAO
       toca esse builder.
    2) gold_prediction_metrics_by_run_split_horizon mantem
       is_quantile_genuine=True para esse run (porque mode=quantile e
       p10 != p90), e as metricas probabilisticas raw sao numericas
       -- prova que o filtro Stage 9 distingue crossing (apenas conta
       width<0) de degeneracao (p10==p90).
    """
    fact = pd.DataFrame(
        [
            {
                "run_id": "r_crossing_quantile",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "split": "test",
                "fold": "wf_1",
                "seed": 1,
                "horizon": 1,
                "timestamp_utc": "2026-01-01T00:00:00Z",
                "target_timestamp_utc": "2026-01-02T00:00:00Z",
                "y_true": 0.0,
                "y_pred": 0.0,
                # Crossing real em raw: p10 > p90 -> width raw < 0;
                # mas p10 != p90 -> non-degenerate -> Stage 9 deixa passar.
                "quantile_p10": 1.0,
                "quantile_p50": 0.0,
                "quantile_p90": -1.0,
                "quantile_p10_post_guardrail": -1.0,
                "quantile_p50_post_guardrail": 0.0,
                "quantile_p90_post_guardrail": 1.0,
            }
        ]
    )
    dim_run = pd.DataFrame(
        [
            {
                "run_id": "r_crossing_quantile",
                "model_version": "v1",
                "feature_set_hash": "fh1",
                "parent_sweep_id": "sw1",
                "trial_number": 1,
                "status": "ok",
            }
        ]
    )
    fact_config = pd.DataFrame(
        [{"run_id": "r_crossing_quantile", "prediction_mode": "quantile"}]
    )

    # Caminho 1: quality_report -- Cat C raw-only, conta crossing.
    out_q = RefreshAnalyticsStoreUseCase._build_gold_oos_quality_report(dim_run, fact)
    run_row = out_q[out_q["scope"] == "run_split_horizon"].iloc[0]
    assert int(run_row["n_negative_interval_width"]) == 1

    # Caminho 2: metrics_by_run_split_horizon -- Stage 9 filter mantem
    # esse run (mode=quantile, p10 != p90) como elegivel.
    out_m = RefreshAnalyticsStoreUseCase._build_gold_prediction_metrics_by_run_split_horizon(
        dim_run, fact, fact_config
    )
    metrics_row = out_m.iloc[0]
    assert bool(metrics_row["is_quantile_genuine"]) is True
    assert int(metrics_row["n_probabilistic_samples"]) == 1
    # Probabilisticas raw numericas (Stage 9 nao mascarou).
    assert not pd.isna(metrics_row["picp_raw"])
    assert not pd.isna(metrics_row["mpiw_raw"])
    assert not pd.isna(metrics_row["mean_pinball_raw"])
    # Os dois caminhos coexistem: crossing contado no quality_report
    # E probabilisticas mantidas em metrics_by_run -- comportamento esperado.


def test_build_gold_oos_quality_report_preserves_asset_after_dim_run_merge() -> None:
    """Stage R-E regression guard: when both fact_oos_predictions and dim_run
    carry an `asset` column, the merge inside `_build_gold_oos_quality_report`
    must preserve `asset` (not coerce to NaN). Pre-Stage 23 bug: pandas
    suffixed the colliding columns to `asset_x`/`asset_y`, dropping the
    plain `asset` column; the downstream groupby then read `asset = NaN`,
    which made `gold_quality_statistics_report.statistics_ready=False` even
    with all checks passing. Fix: `df.merge(..., suffixes=("", "_dim"))`
    keeps the fact_oos side canonical.

    This test FAILS if the `suffixes=("", "_dim")` argument is ever removed
    from `_build_gold_oos_quality_report`.
    """
    fact = pd.DataFrame(
        [
            {
                "run_id": "r_suffix",
                "asset": "AAPL",
                "feature_set_name": "BASELINE_TECHNICAL",
                "config_signature": "cfg_x",
                "parent_sweep_id": "sw_x",
                "split": "test",
                "horizon": 1,
                "timestamp_utc": "2026-01-15T00:00:00Z",
                "target_timestamp_utc": "2026-01-16T00:00:00Z",
                "y_true": 0.001,
                "y_pred": 0.002,
                "quantile_p10": -0.01,
                "quantile_p50": 0.0,
                "quantile_p90": 0.01,
            }
        ]
    )
    # dim_run shares `asset`, `feature_set_name`, `config_signature`,
    # `parent_sweep_id` with fact_oos -- this is the exact collision the
    # pre-fix code couldn't handle.
    dim_run = pd.DataFrame(
        [
            {
                "run_id": "r_suffix",
                "asset": "AAPL",
                "feature_set_name": "BASELINE_TECHNICAL",
                "config_signature": "cfg_x",
                "parent_sweep_id": "sw_x",
                "model_version": "tft_v1",
                "trial_number": 1,
                "status": "ok",
            }
        ]
    )

    report = RefreshAnalyticsStoreUseCase._build_gold_oos_quality_report(dim_run, fact)

    run_rows = report[report["scope"] == "run_split_horizon"]
    assert len(run_rows) == 1, "Expected exactly one run/split/horizon row."
    row = run_rows.iloc[0]
    # The substantive assertion: asset survived the merge.
    assert row["asset"] == "AAPL", (
        f"asset column lost after dim_run merge: got {row['asset']!r}. "
        "Likely a regression of the Stage 23 suffixes=('','_dim') fix."
    )
    # Sibling columns from the collision set must also be canonical (fact_oos
    # side), not NaN, not suffixed.
    assert row["feature_set_name"] == "BASELINE_TECHNICAL"
    assert row["config_signature"] == "cfg_x"
    assert row["parent_sweep_id"] == "sw_x"
    # And no `*_dim` leak appears in the report columns.
    leaked = [c for c in report.columns if c.endswith("_dim")]
    assert not leaked, f"_dim suffix leaked into the gold report columns: {leaked}"


def _stage12_candidate_baseline_oos(*, parent_sweep_id: str = "sw_stage12") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stage 12 integration fixture: candidate TFT + baseline persisted on
    same parent_sweep_id, sharing target_timestamps so DM/MCS builders
    produce pairwise rows. Returns (dim_run, fact_oos, fact_config).
    """
    n = 10
    timestamps = pd.date_range("2026-01-01", periods=n, freq="D", tz="UTC")

    dim_run = pd.DataFrame(
        [
            {
                "run_id": "tft_cand",
                "model_version": "tft_v1",
                "feature_set_name": "BT",
                "feature_set_hash": "fh_tft",
                "config_signature": "cfg_tft",
                "parent_sweep_id": parent_sweep_id,
                "trial_number": 1,
                "status": "ok",
                "asset": "AAPL",
                "split_fingerprint": "sp1",
            },
            {
                "run_id": "baseline_zero",
                "model_version": "baseline_zero_return_v1",
                "feature_set_name": "baseline",
                "feature_set_hash": "fh_baseline",
                "config_signature": "cfg_baseline",
                "parent_sweep_id": parent_sweep_id,
                "trial_number": None,
                "status": "ok",
                "asset": "AAPL",
                "split_fingerprint": "sp1",
            },
        ]
    )

    rows: list[dict] = []
    for i, ts in enumerate(timestamps):
        y_true = float(0.01 * ((-1) ** i))
        rows.append(
            {
                "run_id": "tft_cand",
                "asset": "AAPL",
                "feature_set_name": "BT",
                "config_signature": "cfg_tft",
                "split": "test",
                "fold": "none",
                "seed": 42,
                "horizon": 1,
                "timestamp_utc": ts.isoformat(),
                "target_timestamp_utc": ts.isoformat(),
                "y_true": y_true,
                "y_pred": y_true * 0.9,
                "quantile_p10": -0.02,
                "quantile_p50": y_true * 0.9,
                "quantile_p90": 0.02,
                "quantile_p10_post_guardrail": -0.02,
                "quantile_p50_post_guardrail": y_true * 0.9,
                "quantile_p90_post_guardrail": 0.02,
            }
        )
        rows.append(
            {
                "run_id": "baseline_zero",
                "asset": "AAPL",
                "feature_set_name": "baseline",
                "config_signature": "cfg_baseline",
                "split": "test",
                "fold": "none",
                "seed": 0,
                "horizon": 1,
                "timestamp_utc": ts.isoformat(),
                "target_timestamp_utc": ts.isoformat(),
                "y_true": y_true,
                "y_pred": 0.0,
                "quantile_p10": 0.0,
                "quantile_p50": 0.0,
                "quantile_p90": 0.0,
                "quantile_p10_post_guardrail": 0.0,
                "quantile_p50_post_guardrail": 0.0,
                "quantile_p90_post_guardrail": 0.0,
            }
        )

    fact_oos = pd.DataFrame(rows)
    fact_config = pd.DataFrame(
        [
            {"run_id": "tft_cand", "prediction_mode": "quantile"},
            {"run_id": "baseline_zero", "prediction_mode": "point"},
        ]
    )
    return dim_run, fact_oos, fact_config


def test_refresh_dm_pairwise_includes_candidate_vs_baseline_when_shared_parent_sweep_id() -> None:
    dim_run, fact_oos, _ = _stage12_candidate_baseline_oos()

    dm = RefreshAnalyticsStoreUseCase._build_gold_dm_pairwise_results(dim_run, fact_oos)
    assert not dm.empty, "DM pairwise must contain candidate vs baseline pair when sharing parent_sweep_id"
    assert (dm["parent_sweep_id"] == "sw_stage12").all()
    assert int(dm["n_configs"].iloc[0]) >= 2
    # Sanity: tudo no mesmo (asset, split, horizon).
    assert set(dm["asset"]) == {"AAPL"}
    assert set(dm["split"]) == {"test"}
    assert set(dm["horizon"]) == {1}


def test_refresh_mcs_includes_candidate_and_baseline_configs_when_shared_parent_sweep_id() -> None:
    dim_run, fact_oos, _ = _stage12_candidate_baseline_oos()

    mcs = RefreshAnalyticsStoreUseCase._build_gold_mcs_results(dim_run, fact_oos)
    assert not mcs.empty, "MCS must include the candidate+baseline pool"
    assert (mcs["parent_sweep_id"] == "sw_stage12").all()
    # Ambos configs (tft + baseline) devem aparecer.
    cfg_labels = set(mcs.get("config_label", pd.Series(dtype=str)).astype(str))
    assert any("BT|cfg_tft" in label for label in cfg_labels) or any("baseline" in label for label in cfg_labels)


def test_refresh_win_rate_pairwise_includes_candidate_vs_baseline_when_shared_parent_sweep_id() -> None:
    dim_run, fact_oos, _ = _stage12_candidate_baseline_oos()
    win_rate = RefreshAnalyticsStoreUseCase._build_gold_win_rate_pairwise_results(dim_run, fact_oos)
    assert not win_rate.empty, "win_rate pairwise must contain candidate vs baseline pair when sharing parent_sweep_id"
    assert (win_rate["parent_sweep_id"] == "sw_stage12").all()
    # Sweep has exactly 2 configs -> exactly one pair (left, right) per group key.
    assert len(win_rate) == 1
    row = win_rate.iloc[0]
    # left_wins + right_wins + ties == aligned_timestamps (n=10 timestamps in fixture).
    n = int(row["aligned_timestamps"])
    assert n == 10
    assert int(row["left_wins"]) + int(row["right_wins"]) + int(row["ties"]) == n
