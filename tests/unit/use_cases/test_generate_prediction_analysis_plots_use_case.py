from __future__ import annotations

import json

from pathlib import Path

import pandas as pd
import pytest

from src.use_cases.generate_prediction_analysis_plots_use_case import (
    GeneratePredictionAnalysisPlotsUseCase,
)


def _write_gold(base: Path, name: str, rows: list[dict]) -> None:
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(base / f"{name}.parquet", index=False)


def _write_partitioned(base: Path, table_name: str, rows: list[dict], parts: dict[str, str]) -> None:
    table_dir = base / table_name
    for k, v in parts.items():
        table_dir = table_dir / f"{k}={v}"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(table_dir / f"{table_name}.parquet", index=False)


def test_generate_prediction_analysis_plots_use_case_generates_all_outputs(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    silver = tmp_path / "silver"
    out = tmp_path / "out"

    _write_gold(
        gold,
        "gold_prediction_metrics_by_config",
        [
            {
                "asset": "AAPL",
                "split": "test",
                "horizon": 1,
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "mean_rmse": 0.1,
                "mean_mae": 0.08,
                "mean_directional_accuracy": 0.55,
                "mean_mean_pinball": 0.03,
                "mean_mpiw": 0.2,
                "mean_picp": 0.85,
            },
            {
                "asset": "AAPL",
                "split": "test",
                "horizon": 7,
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "mean_rmse": 0.12,
                "mean_mae": 0.09,
                "mean_directional_accuracy": 0.53,
                "mean_mean_pinball": 0.035,
                "mean_mpiw": 0.25,
                "mean_picp": 0.87,
            },
        ],
    )

    _write_gold(
        gold,
        "gold_prediction_metrics_by_run_split_horizon",
        [
            {
                "asset": "AAPL",
                "run_id": "r1",
                "split": "test",
                "horizon": 1,
                "fold": "wf_1",
                "seed": 42,
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "rmse": 0.1,
            },
            {
                "asset": "AAPL",
                "run_id": "r2",
                "split": "test",
                "horizon": 7,
                "fold": "wf_1",
                "seed": 42,
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "rmse": 0.12,
            },
        ],
    )

    _write_gold(
        gold,
        "gold_dm_pairwise_results",
        [
            {
                "asset": "AAPL",
                "horizon": 1,
                "left_config": "BT|cfg1",
                "right_config": "BT|cfg2",
                "pvalue_adj_holm": 0.03,
            }
        ],
    )

    _write_gold(
        gold,
        "gold_prediction_calibration",
        [
            {
                "asset": "AAPL",
                "split": "test",
                "horizon": 1,
                "coverage_nominal": 0.8,
                "picp": 0.78,
            }
        ],
    )

    _write_gold(
        gold,
        "gold_oos_consolidated",
        [
            {
                "asset": "AAPL",
                "run_id": "r1",
                "split": "test",
                "horizon": 1,
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "timestamp_utc": "2026-01-10T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-13T00:00:00+00:00",
                "y_true": 0.02,
                "y_pred": 0.015,
                "quantile_p10": -0.01,
                "quantile_p90": 0.03,
                "abs_error": 0.005,
            }
        ],
    )

    _write_gold(
        gold,
        "gold_model_decision_final",
        [
            {
                "asset": "AAPL",
                "horizon": 1,
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "rank_rmse": 1,
            }
        ],
    )

    _write_gold(
        gold,
        "gold_feature_impact_by_horizon",
        [
            {
                "asset": "AAPL",
                "parent_sweep_id": "sw1",
                "split": "test",
                "horizon": 1,
                "feature_name": "close",
                "mean_delta_rmse": 0.02,
            }
        ],
    )

    _write_partitioned(
        silver,
        "fact_feature_contrib_local",
        [
            {
                "asset": "AAPL",
                "run_id": "r1",
                "horizon": 1,
                "timestamp_utc": "2026-01-10T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-13T00:00:00+00:00",
                "feature_name": "close",
                "contribution": 0.03,
                "abs_contribution": 0.03,
            },
            {
                "asset": "AAPL",
                "run_id": "r1",
                "horizon": 1,
                "timestamp_utc": "2026-01-10T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-13T00:00:00+00:00",
                "feature_name": "volume",
                "contribution": -0.01,
                "abs_contribution": 0.01,
            },
        ],
        {"asset": "AAPL"},
    )

    result = GeneratePredictionAnalysisPlotsUseCase(
        analytics_gold_dir=gold,
        analytics_silver_dir=silver,
        output_dir=out,
    ).execute(asset="AAPL", top_n_configs=5, top_k_features=5, max_timeseries_points=40)

    expected_keys = {
        "fig_heatmap_metrics_by_horizon",
        "fig_boxplot_error_by_fold_seed",
        "fig_dm_pvalue_matrix",
        "fig_calibration_curve",
        "fig_interval_width_vs_coverage",
        "fig_oos_timeseries_examples",
        "fig_feature_importance_global",
        "fig_feature_importance_global_all_features",
        "fig_feature_contrib_local_cases",
        "manifest",
    }
    assert expected_keys.issubset(set(result.outputs.keys()))

    for key in expected_keys - {"manifest"}:
        assert (out / f"{key}.png").exists()

    manifest_path = out / "prediction_analysis_plots_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["asset"] == "AAPL"
    assert set(payload["outputs"].keys()) == expected_keys - {"manifest"}



def test_generate_prediction_analysis_plots_use_case_scope_csv_filters_candidates(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    silver = tmp_path / "silver"
    out = tmp_path / "out"
    scope_csv = tmp_path / "scope.csv"

    # Two configs in gold; scope CSV keeps only cfg_keep.
    _write_gold(
        gold,
        "gold_prediction_metrics_by_config",
        [
            {
                "asset": "AAPL",
                "split": "test",
                "horizon": 1,
                "feature_set_name": "BT",
                "config_signature": "cfg_keep",
                "mean_rmse": 0.10,
                "mean_mae": 0.08,
                "mean_directional_accuracy": 0.55,
                "mean_mean_pinball": 0.03,
                "mean_mpiw": 0.2,
                "mean_picp": 0.85,
            },
            {
                "asset": "AAPL",
                "split": "test",
                "horizon": 1,
                "feature_set_name": "BT",
                "config_signature": "cfg_drop",
                "mean_rmse": 0.12,
                "mean_mae": 0.09,
                "mean_directional_accuracy": 0.52,
                "mean_mean_pinball": 0.04,
                "mean_mpiw": 0.25,
                "mean_picp": 0.80,
            },
        ],
    )
    _write_gold(
        gold,
        "gold_prediction_metrics_by_run_split_horizon",
        [
            {
                "asset": "AAPL",
                "run_id": "r_keep",
                "split": "test",
                "horizon": 1,
                "fold": "wf_1",
                "seed": 42,
                "feature_set_name": "BT",
                "config_signature": "cfg_keep",
                "rmse": 0.1,
            },
            {
                "asset": "AAPL",
                "run_id": "r_drop",
                "split": "test",
                "horizon": 1,
                "fold": "wf_1",
                "seed": 42,
                "feature_set_name": "BT",
                "config_signature": "cfg_drop",
                "rmse": 0.12,
            },
        ],
    )
    _write_gold(
        gold,
        "gold_dm_pairwise_results",
        [
            {
                "asset": "AAPL",
                "horizon": 1,
                "left_config": "BT|cfg_keep",
                "right_config": "BT|cfg_drop",
                "pvalue_adj_holm": 0.03,
            }
        ],
    )
    _write_gold(
        gold,
        "gold_prediction_calibration",
        [
            {
                "asset": "AAPL",
                "split": "test",
                "horizon": 1,
                "coverage_nominal": 0.8,
                "picp": 0.78,
            }
        ],
    )
    _write_gold(
        gold,
        "gold_oos_consolidated",
        [
            {
                "asset": "AAPL",
                "run_id": "r_keep",
                "split": "test",
                "horizon": 1,
                "feature_set_name": "BT",
                "config_signature": "cfg_keep",
                "timestamp_utc": "2026-01-10T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-13T00:00:00+00:00",
                "y_true": 0.02,
                "y_pred": 0.015,
                "quantile_p10": -0.01,
                "quantile_p90": 0.03,
                "abs_error": 0.005,
            }
        ],
    )
    _write_gold(
        gold,
        "gold_model_decision_final",
        [
            {
                "asset": "AAPL",
                "horizon": 1,
                "feature_set_name": "BT",
                "config_signature": "cfg_keep",
                "rank_rmse": 1,
            }
        ],
    )
    _write_gold(
        gold,
        "gold_feature_impact_by_horizon",
        [
            {
                "asset": "AAPL",
                "parent_sweep_id": "0_2_2_explicit",
                "split": "test",
                "horizon": 1,
                "feature_name": "close",
                "mean_delta_rmse": 0.02,
            }
        ],
    )
    _write_partitioned(
        silver,
        "fact_feature_contrib_local",
        [
            {
                "asset": "AAPL",
                "run_id": "r_keep",
                "horizon": 1,
                "timestamp_utc": "2026-01-10T00:00:00+00:00",
                "target_timestamp_utc": "2026-01-13T00:00:00+00:00",
                "feature_name": "close",
                "contribution": 0.03,
                "abs_contribution": 0.03,
            }
        ],
        {"asset": "AAPL"},
    )
    _write_partitioned(
        silver,
        "dim_run",
        [
            {
                "run_id": "r_keep",
                "feature_set_name": "BT",
                "config_signature": "cfg_keep",
                "model_version": "v_keep",
                "parent_sweep_id": "0_2_2_explicit",
            },
            {
                "run_id": "r_drop",
                "feature_set_name": "BT",
                "config_signature": "cfg_drop",
                "model_version": "v_drop",
                "parent_sweep_id": "0_1_5_optuna",
            },
        ],
        {"asset": "AAPL"},
    )

    pd.DataFrame([{"run_id": "r_keep"}]).to_csv(scope_csv, index=False)

    result = GeneratePredictionAnalysisPlotsUseCase(
        analytics_gold_dir=gold,
        analytics_silver_dir=silver,
        output_dir=out,
    ).execute(asset="AAPL", scope_csv_path=scope_csv)

    payload = json.loads((out / "prediction_analysis_plots_manifest.json").read_text(encoding="utf-8"))
    assert payload["scope_csv_path"] == str(scope_csv)
    assert payload["scope_selected_labels"] == ["BT|cfg_keep"]
    assert "fig_heatmap_metrics_by_horizon" in result.outputs


def test_filter_scope_preserves_parent_sweep_grain_for_stage5_gold_tables() -> None:
    df = pd.DataFrame(
        [
            {
                "asset": "AAPL",
                "parent_sweep_id": "sw_keep",
                "split": "test",
                "horizon": 1,
                "feature_name": "close",
                "mean_delta_rmse": 0.02,
            },
            {
                "asset": "AAPL",
                "parent_sweep_id": "sw_drop",
                "split": "test",
                "horizon": 1,
                "feature_name": "close",
                "mean_delta_rmse": 0.50,
            },
        ]
    )
    scope = {
        "pairs": set(),
        "labels": set(),
        "run_ids": set(),
        "model_versions": set(),
        "parent_sweep_ids": {"sw_keep"},
    }

    out = GeneratePredictionAnalysisPlotsUseCase._filter_df_by_scope(df, scope)

    assert out["parent_sweep_id"].tolist() == ["sw_keep"]
    assert out["mean_delta_rmse"].tolist() == [0.02]


def test_generate_prediction_analysis_plots_use_case_scope_prefix_uses_checkpoint_path_fallback(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    uc = GeneratePredictionAnalysisPlotsUseCase(
        analytics_gold_dir=tmp_path / "gold",
        analytics_silver_dir=silver,
        output_dir=tmp_path / "out",
    )

    _write_partitioned(
        silver,
        "dim_run",
        [
            {
                "run_id": "r1",
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "model_version": "v1",
                "parent_sweep_id": None,
                "checkpoint_path_best": "data/models/AAPL/sweeps/0_2_2_explicit_x/folds/wf_1/models/AAPL/x/checkpoints/best.ckpt",
            },
            {
                "run_id": "r2",
                "feature_set_name": "BT",
                "config_signature": "cfg2",
                "model_version": "v2",
                "parent_sweep_id": None,
                "checkpoint_path_best": "data/models/AAPL/sweeps/0_1_5_optuna_x/folds/wf_1/models/AAPL/y/checkpoints/best.ckpt",
            },
        ],
        {"asset": "AAPL"},
    )

    scope = uc._build_scope_selection(scope_csv_path=None, scope_sweep_prefixes=["0_2_2_"])
    assert scope is not None
    labels = scope.get("labels", set())
    assert "BT|cfg1" in labels
    assert "BT|cfg2" not in labels



def test_generate_prediction_analysis_plots_use_case_scope_requested_without_match_raises(tmp_path: Path) -> None:
    gold = tmp_path / "gold"
    silver = tmp_path / "silver"
    out = tmp_path / "out"

    _write_gold(
        gold,
        "gold_prediction_metrics_by_config",
        [
            {
                "asset": "AAPL",
                "split": "test",
                "horizon": 1,
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "mean_rmse": 0.1,
                "mean_mae": 0.08,
                "mean_directional_accuracy": 0.55,
                "mean_mean_pinball": 0.03,
                "mean_mpiw": 0.2,
                "mean_picp": 0.85,
            }
        ],
    )
    _write_partitioned(
        silver,
        "dim_run",
        [
            {
                "run_id": "r1",
                "feature_set_name": "BT",
                "config_signature": "cfg1",
                "model_version": "v1",
                "parent_sweep_id": "0_1_5_optuna",
            }
        ],
        {"asset": "AAPL"},
    )

    uc = GeneratePredictionAnalysisPlotsUseCase(
        analytics_gold_dir=gold,
        analytics_silver_dir=silver,
        output_dir=out,
    )
    with pytest.raises(ValueError, match="Plot scope resolved no candidates"):
        uc.execute(asset="AAPL", scope_sweep_prefixes=["0_2_2_"])


def test_generate_prediction_analysis_plots_use_case_scope_csv_and_prefix_intersection(tmp_path: Path) -> None:
    silver = tmp_path / "silver"
    scope_csv = tmp_path / "scope.csv"
    uc = GeneratePredictionAnalysisPlotsUseCase(
        analytics_gold_dir=tmp_path / "gold",
        analytics_silver_dir=silver,
        output_dir=tmp_path / "out",
    )

    _write_partitioned(
        silver,
        "dim_run",
        [
            {
                "run_id": "r_keep",
                "feature_set_name": "BT",
                "config_signature": "cfg_keep",
                "model_version": "v_keep",
                "parent_sweep_id": "0_2_2_explicit",
            },
            {
                "run_id": "r_drop",
                "feature_set_name": "BT",
                "config_signature": "cfg_drop",
                "model_version": "v_drop",
                "parent_sweep_id": "0_1_5_optuna",
            },
        ],
        {"asset": "AAPL"},
    )
    pd.DataFrame([
        {"run_id": "r_keep"},
        {"run_id": "r_drop"},
    ]).to_csv(scope_csv, index=False)

    scope = uc._build_scope_selection(
        scope_csv_path=scope_csv,
        scope_sweep_prefixes=["0_2_2_"],
    )
    assert scope is not None
    labels = scope.get("labels", set())
    assert labels == {"BT|cfg_keep"}


def test_filter_df_by_scope_filters_pairwise_tables_by_both_sides() -> None:
    df = pd.DataFrame([
        {"left_config": "BT|cfg1", "right_config": "BT|cfg2", "horizon": 1},
        {"left_config": "BT|cfg1", "right_config": "BT|cfg3", "horizon": 1},
        {"left_config": "BT|cfg3", "right_config": "BT|cfg4", "horizon": 1},
    ])
    scope = {"labels": {"BT|cfg1", "BT|cfg2"}}
    out = GeneratePredictionAnalysisPlotsUseCase._filter_df_by_scope(df, scope)
    assert len(out) == 1
    assert out.iloc[0]["left_config"] == "BT|cfg1"
    assert out.iloc[0]["right_config"] == "BT|cfg2"
