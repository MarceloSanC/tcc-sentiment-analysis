from __future__ import annotations

import json
import pickle
import sys
import types
from pathlib import Path

from src.adapters.local_tft_model_repository import LocalTFTModelRepository


class _FakeModel:
    def state_dict(self):
        return {"w": 1}


class _FakeScaler:
    pass


def test_save_training_artifacts_persists_complete_bundle(monkeypatch, tmp_path: Path) -> None:
    fake_torch = types.ModuleType("torch")

    def _save(obj, path):
        Path(path).write_bytes(b"pt")

    fake_torch.save = _save
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    checkpoint_src = tmp_path / "source_best.ckpt"
    checkpoint_src.write_bytes(b"ckpt")

    repo = LocalTFTModelRepository(base_dir=tmp_path / "models")
    out_dir = Path(
        repo.save_training_artifacts(
            "AAPL",
            "20260211_120000_BT",
            _FakeModel(),
            metrics={"rmse": 1.0},
            history=[{"train_loss": 0.5, "val_loss": 0.4}],
            split_metrics={
                "train": {"rmse": 0.9, "mae": 0.7},
                "val": {"rmse": 1.0, "mae": 0.8},
                "test": {"rmse": 1.1, "mae": 0.9},
            },
            features_used=["open", "close"],
            training_window={"start": "2010-01-01", "end": "2025-12-31"},
            split_window={
                "train_start": "20100101",
                "train_end": "20221231",
                "val_start": "20230101",
                "val_end": "20241231",
                "test_start": "20250101",
                "test_end": "20251231",
            },
            config={"max_epochs": 10, "feature_set_tag": "BT"},
            feature_importance=[
                {"feature": "open", "delta_rmse": 0.02},
                {"feature": "close", "delta_rmse": 0.01},
            ],
            ablation_results=[
                {"experiment": "baseline", "test_rmse": 1.1},
                {"experiment": "baseline_plus_technical", "test_rmse": 1.0},
            ],
            checkpoint_path=str(checkpoint_src),
            dataset_parameters={"n_rows": 100},
        )
    )

    assert (out_dir / "model_state.pt").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "split_metrics.json").exists()
    assert (out_dir / "history.csv").exists()
    assert (out_dir / "features.json").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "dataset_parameters.pkl").exists()
    assert (out_dir / "checkpoints" / "best.ckpt").exists()
    assert (out_dir / "analysis" / "feature_importance.csv").exists()
    assert (out_dir / "analysis" / "ablation_results.csv").exists()

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["asset_id"] == "AAPL"
    assert metadata["version"] == "20260211_120000_BT"
    assert metadata["training_config"]["feature_set_tag"] == "BT"
    assert "split_metrics" in metadata
    assert "best_checkpoint" in metadata
    assert "analysis_artifacts" in metadata
    assert "feature_importance_csv" in metadata["analysis_artifacts"]
    assert "ablation_results_csv" in metadata["analysis_artifacts"]

    with (out_dir / "dataset_parameters.pkl").open("rb") as fp:
        params = pickle.load(fp)
    assert params == {"n_rows": 100}


def test_save_training_artifacts_persists_scalers_and_metadata(monkeypatch, tmp_path: Path) -> None:
    fake_torch = types.ModuleType("torch")

    def _save(obj, path):
        Path(path).write_bytes(b"pt")

    fake_torch.save = _save
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    repo = LocalTFTModelRepository(base_dir=tmp_path / "models")
    out_dir = Path(
        repo.save_training_artifacts(
            "AAPL",
            "20260211_130000_BT",
            _FakeModel(),
            metrics={"rmse": 1.0},
            history=[],
            split_metrics={
                "train": {"rmse": 0.9, "mae": 0.7},
                "val": {"rmse": 1.0, "mae": 0.8},
                "test": {"rmse": 1.1, "mae": 0.9},
            },
            features_used=["open", "close"],
            training_window={"start": "2010-01-01", "end": "2025-12-31"},
            split_window={
                "train_start": "20100101",
                "train_end": "20221231",
                "val_start": "20230101",
                "val_end": "20241231",
                "test_start": "20250101",
                "test_end": "20251231",
            },
            config={"max_epochs": 10, "feature_set_tag": "BT"},
            dataset_parameters={
                "scalers": {
                    "close": _FakeScaler(),
                    "volume": _FakeScaler(),
                }
            },
        )
    )

    assert (out_dir / "scalers.pkl").exists()
    with (out_dir / "scalers.pkl").open("rb") as fp:
        scalers = pickle.load(fp)
    assert set(scalers.keys()) == {"close", "volume"}

    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["scaler_type"] == ["_FakeScaler"]
    assert "scaler_artifacts" in metadata
    assert "scalers_pkl" in metadata["scaler_artifacts"]
