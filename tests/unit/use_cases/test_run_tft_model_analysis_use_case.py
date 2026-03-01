from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import csv

import pytest

from src.use_cases.run_tft_model_analysis_use_case import RunTFTModelAnalysisUseCase


def test_param_ranges_empty_dict_is_preserved_for_optuna_usage() -> None:
    use_case = RunTFTModelAnalysisUseCase(
        train_runner=_FakeTrainRunner(),
        base_training_config={
            "max_encoder_length": 60,
            "max_prediction_length": 1,
            "batch_size": 64,
            "max_epochs": 20,
            "learning_rate": 5e-4,
            "hidden_size": 32,
            "attention_head_size": 2,
            "dropout": 0.1,
            "hidden_continuous_size": 8,
            "seed": 42,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0,
        },
        param_ranges={},
        generate_comparison_plots=False,
    )

    assert use_case.param_ranges == {}


class _FakeTrainRunner:
    def __init__(self, fail_after: int | None = None) -> None:
        self.index = 0
        self.fail_after = fail_after

    def run(
        self,
        *,
        asset: str,
        features: str | None,
        config: dict[str, Any],
        split_config: dict[str, str] | None,
        models_asset_dir: Path,
    ) -> tuple[str | None, dict[str, Any] | None]:
        self.index += 1
        if self.fail_after is not None and self.index > self.fail_after:
            raise RuntimeError("forced failure")

        version = f"run_{self.index:03d}"
        run_dir = models_asset_dir / version
        run_dir.mkdir(parents=True, exist_ok=True)
        rmse = 0.02 + (self.index * 0.001)
        metadata = {
            "version": version,
            "created_at": "2026-02-17T00:00:00+00:00",
            "features_used": ["open", "close"],
            "training_config": config,
            "split_metrics": {
                "train": {"rmse": rmse + 0.01, "mae": rmse + 0.005},
                "val": {"rmse": rmse + 0.005, "mae": rmse + 0.003},
                "test": {"rmse": rmse, "mae": rmse - 0.002},
            },
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return version, metadata


class _CountingTrainRunner:
    def __init__(self) -> None:
        self.calls = 0

    def run(
        self,
        *,
        asset: str,
        features: str | None,
        config: dict[str, Any],
        split_config: dict[str, str] | None,
        models_asset_dir: Path,
    ) -> tuple[str | None, dict[str, Any] | None]:
        self.calls += 1
        version = f"new_{self.calls:03d}"
        run_dir = models_asset_dir / version
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "version": version,
            "split_metrics": {
                "train": {"rmse": 0.03, "mae": 0.02},
                "val": {"rmse": 0.02, "mae": 0.01},
                "test": {"rmse": 0.021, "mae": 0.011},
            },
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return version, metadata


def _write_valid_model_artifacts(version_dir: Path, *, val_rmse: float, val_mae: float, test_rmse: float, test_mae: float) -> None:
    import torch

    version_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"layer.weight": torch.tensor([1.0])}, version_dir / "model_state.pt")
    (version_dir / "metrics.json").write_text(json.dumps({"rmse": test_rmse, "mae": test_mae}), encoding="utf-8")
    (version_dir / "split_metrics.json").write_text(
        json.dumps(
            {
                "train": {"rmse": val_rmse + 0.01, "mae": val_mae + 0.01},
                "val": {"rmse": val_rmse, "mae": val_mae},
                "test": {"rmse": test_rmse, "mae": test_mae},
            }
        ),
        encoding="utf-8",
    )
    (version_dir / "history.csv").write_text("epoch,train_loss,val_loss\n0,0.1,0.2\n", encoding="utf-8")
    (version_dir / "features.json").write_text(json.dumps({"features_used": ["open", "close"]}), encoding="utf-8")
    (version_dir / "config.json").write_text(json.dumps({"seed": 7}), encoding="utf-8")
    (version_dir / "metadata.json").write_text(
        json.dumps(
            {
                "version": version_dir.name,
                "split_metrics": {
                    "val": {"rmse": val_rmse, "mae": val_mae},
                    "test": {"rmse": test_rmse, "mae": test_mae},
                },
            }
        ),
        encoding="utf-8",
    )


def test_model_analysis_use_case_generates_sweep_artifacts(tmp_path: Path) -> None:
    use_case = RunTFTModelAnalysisUseCase(
        train_runner=_FakeTrainRunner(),
        base_training_config={
            "max_encoder_length": 60,
            "max_prediction_length": 1,
            "batch_size": 64,
            "max_epochs": 20,
            "learning_rate": 5e-4,
            "hidden_size": 32,
            "attention_head_size": 2,
            "dropout": 0.1,
            "hidden_continuous_size": 8,
            "seed": 42,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0,
        },
        param_ranges={"max_encoder_length": [30, 60]},
        generate_comparison_plots=False,
    )

    result = use_case.execute(
        asset="AAPL",
        models_asset_dir=tmp_path / "models" / "AAPL",
        max_runs=2,
        output_subdir="sweep_test",
        analysis_config={"test_mode": True},
    )

    sweep_dir = Path(result.sweep_dir)
    assert (sweep_dir / "summary.json").exists()
    assert (sweep_dir / "sweep_runs.csv").exists()
    assert (sweep_dir / "sweep_runs.json").exists()
    assert (sweep_dir / "analysis_config.json").exists()
    default_fold_dir = sweep_dir / "folds" / "default"
    assert (default_fold_dir / "all_models_ranked.csv").exists()
    assert (default_fold_dir / "config_ranking.csv").exists()
    sweep_runs = (default_fold_dir / "sweep_runs.csv").read_text(encoding="utf-8")
    assert "val_rmse" in sweep_runs
    assert "val_mae" in sweep_runs
    assert result.runs_ok == 6


def test_model_analysis_stops_on_first_error_when_continue_false(tmp_path: Path) -> None:
    use_case = RunTFTModelAnalysisUseCase(
        train_runner=_FakeTrainRunner(fail_after=1),
        base_training_config={
            "max_encoder_length": 60,
            "max_prediction_length": 1,
            "batch_size": 64,
            "max_epochs": 20,
            "learning_rate": 5e-4,
            "hidden_size": 32,
            "attention_head_size": 2,
            "dropout": 0.1,
            "hidden_continuous_size": 8,
            "seed": 42,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0,
        },
        param_ranges={"max_encoder_length": [30, 60, 90]},
        generate_comparison_plots=False,
    )

    result = use_case.execute(
        asset="AAPL",
        models_asset_dir=tmp_path / "models" / "AAPL",
        continue_on_error=False,
        output_subdir="sweep_fail_fast",
    )
    assert result.runs_failed == 1


class _RobustnessTrainRunner:
    def run(
        self,
        *,
        asset: str,
        features: str | None,
        config: dict[str, Any],
        split_config: dict[str, str] | None,
        models_asset_dir: Path,
    ) -> tuple[str | None, dict[str, Any] | None]:
        seed = int(config["seed"])
        max_epochs = int(config["max_epochs"])
        version = f"seed{seed}_epochs{max_epochs}"
        run_dir = models_asset_dir / version
        run_dir.mkdir(parents=True, exist_ok=True)

        if max_epochs == 30:
            val_rmse = 0.5
        else:
            val_by_seed = {42: 0.5, 7: 0.4, 123: 0.6}
            val_rmse = val_by_seed.get(seed, 0.5)

        metadata = {
            "version": version,
            "created_at": "2026-02-17T00:00:00+00:00",
            "features_used": ["open", "close"],
            "training_config": config,
            "split_metrics": {
                "train": {"rmse": val_rmse + 0.01, "mae": val_rmse + 0.005},
                "val": {"rmse": val_rmse, "mae": val_rmse},
                "test": {"rmse": val_rmse + 0.001, "mae": val_rmse + 0.001},
            },
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        return version, metadata


def test_model_analysis_top_5_uses_mean_and_std_val_rmse(tmp_path: Path) -> None:
    use_case = RunTFTModelAnalysisUseCase(
        train_runner=_RobustnessTrainRunner(),
        base_training_config={
            "max_encoder_length": 60,
            "max_prediction_length": 1,
            "batch_size": 64,
            "max_epochs": 20,
            "learning_rate": 5e-4,
            "hidden_size": 32,
            "attention_head_size": 2,
            "dropout": 0.1,
            "hidden_continuous_size": 8,
            "seed": 42,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0,
        },
        param_ranges={"max_epochs": [20, 30]},
        generate_comparison_plots=False,
    )

    result = use_case.execute(
        asset="AAPL",
        models_asset_dir=tmp_path / "models" / "AAPL",
        output_subdir="sweep_robustness",
    )

    assert result.top_5_runs
    best = result.top_5_runs[0]
    assert str(best["run_label"]).startswith("max_epochs=30")
    assert best["mean_val_rmse"] == 0.5
    assert best["std_val_rmse"] == 0.0


def test_model_analysis_adds_ci_columns_when_enabled(tmp_path: Path) -> None:
    use_case = RunTFTModelAnalysisUseCase(
        train_runner=_RobustnessTrainRunner(),
        base_training_config={
            "max_encoder_length": 60,
            "max_prediction_length": 1,
            "batch_size": 64,
            "max_epochs": 20,
            "learning_rate": 5e-4,
            "hidden_size": 32,
            "attention_head_size": 2,
            "dropout": 0.1,
            "hidden_continuous_size": 8,
            "seed": 42,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0,
        },
        param_ranges={"max_epochs": [20, 30]},
        compute_confidence_interval=True,
        generate_comparison_plots=False,
    )
    result = use_case.execute(
        asset="AAPL",
        models_asset_dir=tmp_path / "models" / "AAPL",
        output_subdir="sweep_ci",
        analysis_config={"compute_confidence_interval": True},
    )

    ranking = (Path(result.sweep_dir) / "folds" / "default" / "config_ranking.csv").read_text(
        encoding="utf-8"
    )
    assert "ci95_val_rmse_low" in ranking
    assert "ci95_val_rmse_high" in ranking


def test_model_analysis_walk_forward_runs_all_folds(tmp_path: Path) -> None:
    use_case = RunTFTModelAnalysisUseCase(
        train_runner=_FakeTrainRunner(),
        base_training_config={
            "max_encoder_length": 60,
            "max_prediction_length": 1,
            "batch_size": 64,
            "max_epochs": 20,
            "learning_rate": 5e-4,
            "hidden_size": 32,
            "attention_head_size": 2,
            "dropout": 0.1,
            "hidden_continuous_size": 8,
            "seed": 42,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0,
        },
        param_ranges={"max_encoder_length": [120]},
        replica_seeds=[7, 42],
        walk_forward_config={
            "enabled": True,
            "folds": [
                {
                    "name": "wf_1",
                    "train_start": "20100101",
                    "train_end": "20181231",
                    "val_start": "20190101",
                    "val_end": "20191231",
                    "test_start": "20200101",
                    "test_end": "20201231",
                },
                {
                    "name": "wf_2",
                    "train_start": "20100101",
                    "train_end": "20191231",
                    "val_start": "20200101",
                    "val_end": "20201231",
                    "test_start": "20210101",
                    "test_end": "20211231",
                },
            ],
        },
        generate_comparison_plots=False,
    )

    result = use_case.execute(
        asset="AAPL",
        models_asset_dir=tmp_path / "models" / "AAPL",
        output_subdir="sweep_walk_forward",
    )

    sweep_dir = Path(result.sweep_dir)
    runs_csv = (sweep_dir / "sweep_runs.csv").read_text(encoding="utf-8")
    summary = json.loads((sweep_dir / "summary.json").read_text(encoding="utf-8"))
    assert result.runs_ok == 8
    assert "fold_name" in runs_csv
    assert summary["walk_forward"]["enabled"] is True
    assert summary["walk_forward"]["folds"] == ["wf_1", "wf_2"]


def test_merge_tests_rejects_non_allowed_config_change(tmp_path: Path) -> None:
    runner = _CountingTrainRunner()
    use_case = RunTFTModelAnalysisUseCase(
        train_runner=runner,
        base_training_config={
            "max_encoder_length": 60,
            "max_prediction_length": 1,
            "batch_size": 64,
            "max_epochs": 20,
            "learning_rate": 5e-4,
            "hidden_size": 32,
            "attention_head_size": 2,
            "dropout": 0.1,
            "hidden_continuous_size": 8,
            "seed": 42,
            "early_stopping_patience": 5,
            "early_stopping_min_delta": 0.0,
        },
        param_ranges={"max_encoder_length": [60, 90]},
        replica_seeds=[7],
        generate_comparison_plots=False,
    )

    sweep_dir = tmp_path / "models" / "AAPL" / "sweeps" / "merge_guard"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    existing_cfg = {
        "features": "BASELINE_FEATURES",
        "continue_on_error": True,
        "output_subdir": "merge_guard",
        "compute_confidence_interval": True,
        "replica_seeds": [7],
        "walk_forward": {"enabled": False, "folds": []},
        "training_config": {"hidden_size": 32},
        "split_config": {
            "train_start": "20100101",
            "train_end": "20201231",
            "val_start": "20210101",
            "val_end": "20221231",
            "test_start": "20230101",
            "test_end": "20241231",
        },
        "param_ranges": {"max_encoder_length": [60, 90]},
    }
    (sweep_dir / "analysis_config.json").write_text(json.dumps(existing_cfg), encoding="utf-8")

    new_cfg = dict(existing_cfg)
    new_cfg["training_config"] = {"hidden_size": 64}
    new_cfg["merge_tests"] = True

    with pytest.raises(ValueError) as exc:
        use_case.execute(
            asset="AAPL",
            models_asset_dir=tmp_path / "models" / "AAPL",
            output_subdir="merge_guard",
            merge_tests=True,
            analysis_config=new_cfg,
        )
    assert "training_config.hidden_size" in str(exc.value)
    assert runner.calls == 0


def test_merge_tests_skips_existing_valid_run(tmp_path: Path) -> None:
    runner = _CountingTrainRunner()
    base_cfg = {
        "max_encoder_length": 60,
        "max_prediction_length": 1,
        "batch_size": 64,
        "max_epochs": 20,
        "learning_rate": 5e-4,
        "hidden_size": 32,
        "attention_head_size": 2,
        "dropout": 0.1,
        "hidden_continuous_size": 8,
        "seed": 42,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.0,
    }
    use_case = RunTFTModelAnalysisUseCase(
        train_runner=runner,
        base_training_config=base_cfg,
        param_ranges={"max_encoder_length": [60]},
        replica_seeds=[7],
        generate_comparison_plots=False,
    )

    models_asset_dir = tmp_path / "models" / "AAPL"
    sweep_dir = models_asset_dir / "sweeps" / "merge_skip"
    fold_dir = sweep_dir / "folds" / "default"
    models_dir = fold_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    version = "run_existing_001"
    _write_valid_model_artifacts(
        models_dir / version,
        val_rmse=0.02,
        val_mae=0.01,
        test_rmse=0.021,
        test_mae=0.011,
    )

    cfg_with_seed = dict(base_cfg)
    cfg_with_seed["seed"] = 7
    config_signature = use_case._config_signature(cfg_with_seed)
    row = {
        "fold_name": "default",
        "run_label": "baseline|seed=7|fold=default",
        "varied_param": "",
        "varied_value": "",
        "config_signature": config_signature,
        "version": version,
        "status": "ok",
        "error": "",
        "val_rmse": "0.02",
        "val_mae": "0.01",
        "test_rmse": "0.021",
        "test_mae": "0.011",
    }
    with (sweep_dir / "sweep_runs.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    analysis_cfg = {
        "features": "BASELINE_FEATURES",
        "continue_on_error": True,
        "merge_tests": True,
        "output_subdir": "merge_skip",
        "compute_confidence_interval": True,
        "replica_seeds": [7],
        "walk_forward": {"enabled": False, "folds": []},
        "training_config": base_cfg,
        "split_config": {
            "train_start": "20100101",
            "train_end": "20201231",
            "val_start": "20210101",
            "val_end": "20221231",
            "test_start": "20230101",
            "test_end": "20241231",
        },
        "param_ranges": {"max_encoder_length": [60]},
    }
    (sweep_dir / "analysis_config.json").write_text(json.dumps(analysis_cfg), encoding="utf-8")

    result = use_case.execute(
        asset="AAPL",
        models_asset_dir=models_asset_dir,
        output_subdir="merge_skip",
        merge_tests=True,
        analysis_config=analysis_cfg,
    )

    assert runner.calls == 0
    assert result.runs_ok == 1
    runs = (sweep_dir / "sweep_runs.csv").read_text(encoding="utf-8")
    assert "baseline|seed=7|fold=default" in runs
