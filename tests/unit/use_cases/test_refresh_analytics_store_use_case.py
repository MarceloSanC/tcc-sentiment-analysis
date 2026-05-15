from __future__ import annotations

import pandas as pd

from src.use_cases.refresh_analytics_store_use_case import RefreshAnalyticsStoreUseCase


def _write_table(base, table_name: str, rows: list[dict], parts: dict[str, str] | None = None) -> None:
    table_dir = base / table_name
    if parts:
        for k, v in parts.items():
            table_dir = table_dir / f"{k}={v}"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(table_dir / f"{table_name}.parquet", index=False)


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


    pred_run_h = pd.read_parquet(gold / "gold_prediction_metrics_by_run_split_horizon.parquet")
    assert len(pred_run_h) == 4
    h1 = pred_run_h[(pred_run_h["run_id"] == "r1") & (pred_run_h["horizon"] == 1)].iloc[0]
    assert h1["n_samples"] == 1
    assert abs(float(h1["bias"]) - 0.02) < 1e-12
    assert abs(float(h1["rmse"]) - 0.02) < 1e-12
    assert abs(float(h1["mae"]) - 0.02) < 1e-12
    assert abs(float(h1["pinball_q10"]) - 0.005) < 1e-12
    assert abs(float(h1["pinball_q50"]) - 0.01) < 1e-12
    assert abs(float(h1["pinball_q90"]) - 0.01) < 1e-12
    assert abs(float(h1["mean_pinball"]) - ((0.005 + 0.01 + 0.01) / 3.0)) < 1e-12
    assert abs(float(h1["picp"]) - 1.0) < 1e-12
    assert abs(float(h1["mpiw"]) - 0.15) < 1e-12
    assert abs(float(h1["pred_interval_width"]) - 0.15) < 1e-12
    assert abs(float(h1["coverage_error"]) - 0.2) < 1e-12
    assert abs(float(h1["prob_down"]) - 0.0) < 1e-12
    assert float(h1["confidence_calibrated"]) > 0.0

    by_cfg = pd.read_parquet(gold / "gold_prediction_metrics_by_config.parquet")
    assert not by_cfg.empty
    assert {"n_oos", "mean_bias", "mean_mean_pinball", "mean_picp", "mean_mpiw", "mean_coverage_error", "mean_prob_down", "mean_confidence_calibrated", "iqr_rmse"}.issubset(set(by_cfg.columns))
    assert int((pd.to_numeric(by_cfg["n_oos"], errors="coerce") <= 0).sum()) == 0

    key_cols = ["asset", "feature_set_name", "config_signature", "split", "horizon"]
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
    assert {"run_id", "horizon", "pinball_q10", "pinball_q50", "pinball_q90", "mean_pinball", "picp", "mpiw", "coverage_error"}.issubset(set(cal.columns))
    qaudit = pd.read_parquet(gold / "gold_quantile_guardrail_audit.parquet")
    assert not qaudit.empty
    assert {"mean_pinball_before", "mean_pinball_after", "crossing_before_count", "crossing_after_count"}.issubset(set(qaudit.columns))

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
    assert set(gap.columns).issuperset({"gap_rmse_test_minus_val", "gap_mae_test_minus_val"}) or gap.empty

    robust_h = pd.read_parquet(gold / "gold_prediction_robustness_by_horizon.parquet")
    assert set(robust_h.columns).issuperset({"metric", "mean", "std", "median", "iqr", "ci95_low", "ci95_high"}) or robust_h.empty

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
    assert int(out_a.iloc[0]["n_oos"]) == 8
    assert int(out_b.iloc[0]["n_oos"]) == 8

    sort_cols = ["asset", "feature_set_name", "config_signature", "split", "horizon"]
    out_a = out_a.sort_values(sort_cols).reset_index(drop=True)
    out_b = out_b.sort_values(sort_cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(out_a, out_b, check_like=True)


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
