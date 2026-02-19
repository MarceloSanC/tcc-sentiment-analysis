from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.use_cases.run_tft_model_analysis_use_case import RunTFTModelAnalysisUseCase


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
    )

    result = use_case.execute(
        asset="AAPL",
        models_asset_dir=tmp_path / "models" / "AAPL",
        max_runs=2,
        output_subdir="sweep_test",
        analysis_config={"test_mode": True},
    )

    sweep_dir = Path(result.sweep_dir)
    assert (sweep_dir / "sweep_runs.csv").exists()
    assert (sweep_dir / "sweep_runs.json").exists()
    assert (sweep_dir / "summary.json").exists()
    assert (sweep_dir / "all_models_ranked.csv").exists()
    assert (sweep_dir / "config_ranking.csv").exists()
    assert (sweep_dir / "analysis_config.json").exists()
    sweep_runs = (sweep_dir / "sweep_runs.csv").read_text(encoding="utf-8")
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
    )
    result = use_case.execute(
        asset="AAPL",
        models_asset_dir=tmp_path / "models" / "AAPL",
        output_subdir="sweep_ci",
        analysis_config={"compute_confidence_interval": True},
    )

    ranking = (Path(result.sweep_dir) / "config_ranking.csv").read_text(encoding="utf-8")
    assert "ci95_val_rmse_low" in ranking
    assert "ci95_val_rmse_high" in ranking
