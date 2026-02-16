from __future__ import annotations

import sys
import types
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.adapters.pytorch_forecasting_tft_trainer import PytorchForecastingTFTTrainer


class _FakeTensor:
    def __init__(self, arr):
        self.arr = np.array(arr)

    @property
    def ndim(self):
        return self.arr.ndim

    @property
    def shape(self):
        return self.arr.shape

    def __getitem__(self, item):
        return _FakeTensor(self.arr[item])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.arr

    def item(self):
        return float(self.arr.reshape(-1)[0])


def _install_fake_training_modules(
    monkeypatch,
    tmp_path: Path,
    *,
    torch_load_raises: bool = False,
    load_from_checkpoint_raises: bool = False,
    predict_return_mode: str = "tensor",
):
    fake_torch = types.ModuleType("torch")

    def _cat(tensors, dim=0):
        arrays = [t.arr for t in tensors]
        return _FakeTensor(np.concatenate(arrays, axis=dim))

    def _save(obj, path):
        Path(path).write_bytes(b"pt")

    def _load(path, map_location=None, weights_only=None):
        if torch_load_raises:
            raise RuntimeError("corrupted checkpoint")
        return {"state_dict": {"w": 999}}

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _no_grad():
        return _NoGrad()

    def _as_tensor(x):
        return _FakeTensor(x)

    fake_torch.cat = _cat
    fake_torch.save = _save
    fake_torch.load = _load
    fake_torch.no_grad = _no_grad
    fake_torch.as_tensor = _as_tensor

    fake_pf = types.ModuleType("pytorch_forecasting")
    fake_pf_metrics = types.ModuleType("pytorch_forecasting.metrics")
    fake_pl = types.ModuleType("pytorch_lightning")
    fake_pl_callbacks = types.ModuleType("pytorch_lightning.callbacks")

    class _FakeDataset:
        def __init__(self, df, **kwargs):
            self.df = df.copy()
            self.kwargs = kwargs

        @classmethod
        def from_dataset(cls, training, df, predict=True, stop_randomization=True):
            return cls(df, **training.kwargs)

        def to_dataloader(self, train, batch_size, num_workers):
            y = self.df["target_return"].to_numpy().reshape(-1, 1)
            return [(_FakeTensor(y), (_FakeTensor(y), None))]

        def get_parameters(self):
            return {"n_rows": int(len(self.df))}

    class _FakeTFT:
        load_from_checkpoint_calls = 0

        def __init__(self):
            self._state = {"w": 1}

        @classmethod
        def from_dataset(cls, dataset, **kwargs):
            return cls()

        @classmethod
        def load_from_checkpoint(cls, path):
            cls.load_from_checkpoint_calls += 1
            if load_from_checkpoint_raises:
                raise RuntimeError("fallback load failed")
            return cls()

        def predict(self, dataloader, mode="prediction"):
            actual = np.concatenate([y[0].arr for _, y in dataloader], axis=0)
            offset = 0.9 if self._state.get("w") == 999 else 0.1
            if predict_return_mode == "prediction_obj":
                Prediction = namedtuple("Prediction", ["output", "x"])
                return Prediction(output=_FakeTensor(actual + offset), x={})
            if predict_return_mode == "list":
                return [_FakeTensor(actual + offset)]
            if predict_return_mode == "empty_list":
                return []
            return _FakeTensor(actual + offset)

        def __call__(self, x):
            offset = 0.9 if self._state.get("w") == 999 else 0.1
            return {"prediction": _FakeTensor(x.arr + offset)}

        def state_dict(self):
            return self._state

        def load_state_dict(self, state_dict):
            self._state = dict(state_dict)

    class _FakeQuantileLoss:
        pass

    class _FakeCallback:
        pass

    class _FakeModelCheckpoint:
        def __init__(self, dirpath, filename, monitor, mode, save_top_k):
            self.best_model_path = str(Path(dirpath) / f"{filename}.ckpt")

    class _FakeEarlyStopping:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _FakeTrainer:
        def __init__(self, max_epochs, enable_checkpointing, logger, callbacks):
            self.callbacks = callbacks
            self.callback_metrics = {
                "train_loss": _FakeTensor([0.5]),
                "val_loss": _FakeTensor([0.4]),
            }

        def fit(self, model, train_dataloader, val_dataloader):
            for cb in self.callbacks:
                if hasattr(cb, "on_validation_epoch_end"):
                    cb.on_validation_epoch_end(self, model)

    def _seed_everything(seed, workers=True):
        return None

    fake_pf.TimeSeriesDataSet = _FakeDataset
    fake_pf.TemporalFusionTransformer = _FakeTFT
    fake_pf_metrics.QuantileLoss = _FakeQuantileLoss
    fake_pl.Trainer = _FakeTrainer
    fake_pl.seed_everything = _seed_everything
    fake_pl_callbacks.Callback = _FakeCallback
    fake_pl_callbacks.ModelCheckpoint = _FakeModelCheckpoint
    fake_pl_callbacks.EarlyStopping = _FakeEarlyStopping

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pytorch_forecasting", fake_pf)
    monkeypatch.setitem(sys.modules, "pytorch_forecasting.metrics", fake_pf_metrics)
    monkeypatch.setitem(sys.modules, "pytorch_lightning", fake_pl)
    monkeypatch.setitem(sys.modules, "pytorch_lightning.callbacks", fake_pl_callbacks)
    return {"FakeTFT": _FakeTFT}


def test_trainer_flow_split_metrics_and_feature_importance(monkeypatch, tmp_path: Path) -> None:
    _install_fake_training_modules(monkeypatch, tmp_path)

    df_train = pd.DataFrame(
        {
            "asset_id": ["AAPL", "AAPL"],
            "time_idx": [0, 1],
            "target_return": [0.1, 0.2],
            "close": [10.0, 11.0],
            "volume": [1000, 1100],
            "day_of_week": [0, 1],
            "month": [1, 1],
        }
    )
    df_val = df_train.copy()
    df_test = df_train.copy()

    trainer = PytorchForecastingTFTTrainer()
    result = trainer.train(
        df_train,
        df_val,
        df_test,
        feature_cols=["close", "volume"],
        target_col="target_return",
        time_idx_col="time_idx",
        group_col="asset_id",
        known_real_cols=["time_idx", "day_of_week", "month"],
        config={},
    )

    assert set(result.split_metrics.keys()) == {"train", "val", "test"}
    assert "rmse" in result.split_metrics["test"]
    assert "mae" in result.split_metrics["test"]
    assert result.metrics == result.split_metrics["val"]
    assert result.history
    assert result.checkpoint_path is not None
    assert result.dataset_parameters is not None
    assert len(result.feature_importance) == 2
    assert result.feature_importance[0]["delta_rmse"] >= result.feature_importance[1]["delta_rmse"]


def test_trainer_reproducibility_with_fixed_seed(monkeypatch, tmp_path: Path) -> None:
    _install_fake_training_modules(monkeypatch, tmp_path)

    df_train = pd.DataFrame(
        {
            "asset_id": ["AAPL", "AAPL"],
            "time_idx": [0, 1],
            "target_return": [0.1, 0.2],
            "close": [10.0, 11.0],
            "volume": [1000, 1100],
            "day_of_week": [0, 1],
            "month": [1, 1],
        }
    )
    df_val = df_train.copy()
    df_test = df_train.copy()

    trainer = PytorchForecastingTFTTrainer()
    cfg = {"seed": 123, "max_epochs": 1}

    run1 = trainer.train(
        df_train,
        df_val,
        df_test,
        feature_cols=["close", "volume"],
        target_col="target_return",
        time_idx_col="time_idx",
        group_col="asset_id",
        known_real_cols=["time_idx", "day_of_week", "month"],
        config=cfg,
    )
    run2 = trainer.train(
        df_train,
        df_val,
        df_test,
        feature_cols=["close", "volume"],
        target_col="target_return",
        time_idx_col="time_idx",
        group_col="asset_id",
        known_real_cols=["time_idx", "day_of_week", "month"],
        config=cfg,
    )

    assert run1.metrics == run2.metrics
    assert run1.split_metrics == run2.split_metrics


def test_trainer_restores_best_checkpoint_state_dict(monkeypatch, tmp_path: Path) -> None:
    refs = _install_fake_training_modules(monkeypatch, tmp_path)

    df = pd.DataFrame(
        {
            "asset_id": ["AAPL", "AAPL"],
            "time_idx": [0, 1],
            "target_return": [0.1, 0.2],
            "close": [10.0, 11.0],
            "volume": [1000, 1100],
            "day_of_week": [0, 1],
            "month": [1, 1],
        }
    )

    trainer = PytorchForecastingTFTTrainer()
    result = trainer.train(
        df,
        df.copy(),
        df.copy(),
        feature_cols=["close", "volume"],
        target_col="target_return",
        time_idx_col="time_idx",
        group_col="asset_id",
        known_real_cols=["time_idx", "day_of_week", "month"],
        config={},
    )

    # offset=0.9 after loading checkpoint state_dict {"w": 999}
    assert result.split_metrics["val"]["rmse"] > 0.5
    assert refs["FakeTFT"].load_from_checkpoint_calls == 0


def test_trainer_raises_explicit_error_when_checkpoint_restore_fails(monkeypatch, tmp_path: Path) -> None:
    _install_fake_training_modules(
        monkeypatch,
        tmp_path,
        torch_load_raises=True,
        load_from_checkpoint_raises=True,
    )

    df = pd.DataFrame(
        {
            "asset_id": ["AAPL", "AAPL"],
            "time_idx": [0, 1],
            "target_return": [0.1, 0.2],
            "close": [10.0, 11.0],
            "volume": [1000, 1100],
            "day_of_week": [0, 1],
            "month": [1, 1],
        }
    )

    trainer = PytorchForecastingTFTTrainer()
    with pytest.raises(RuntimeError, match="Failed to restore best checkpoint"):
        trainer.train(
            df,
            df.copy(),
            df.copy(),
            feature_cols=["close", "volume"],
            target_col="target_return",
            time_idx_col="time_idx",
            group_col="asset_id",
            known_real_cols=["time_idx", "day_of_week", "month"],
            config={},
        )


def test_trainer_supports_predict_returning_list_of_tensors(monkeypatch, tmp_path: Path) -> None:
    _install_fake_training_modules(monkeypatch, tmp_path, predict_return_mode="list")

    df = pd.DataFrame(
        {
            "asset_id": ["AAPL", "AAPL"],
            "time_idx": [0, 1],
            "target_return": [0.1, 0.2],
            "close": [10.0, 11.0],
            "volume": [1000, 1100],
            "day_of_week": [0, 1],
            "month": [1, 1],
        }
    )

    trainer = PytorchForecastingTFTTrainer()
    result = trainer.train(
        df,
        df.copy(),
        df.copy(),
        feature_cols=["close", "volume"],
        target_col="target_return",
        time_idx_col="time_idx",
        group_col="asset_id",
        known_real_cols=["time_idx", "day_of_week", "month"],
        config={},
    )

    assert "val" in result.split_metrics
    assert result.split_metrics["val"]["rmse"] >= 0.0


def test_trainer_supports_predict_returning_prediction_object(
    monkeypatch, tmp_path: Path
) -> None:
    _install_fake_training_modules(monkeypatch, tmp_path, predict_return_mode="prediction_obj")

    df = pd.DataFrame(
        {
            "asset_id": ["AAPL", "AAPL"],
            "time_idx": [0, 1],
            "target_return": [0.1, 0.2],
            "close": [10.0, 11.0],
            "volume": [1000, 1100],
            "day_of_week": [0, 1],
            "month": [1, 1],
        }
    )

    trainer = PytorchForecastingTFTTrainer()
    result = trainer.train(
        df,
        df.copy(),
        df.copy(),
        feature_cols=["close", "volume"],
        target_col="target_return",
        time_idx_col="time_idx",
        group_col="asset_id",
        known_real_cols=["time_idx", "day_of_week", "month"],
        config={},
    )

    assert "test" in result.split_metrics
    assert result.split_metrics["test"]["mae"] >= 0.0


def test_trainer_falls_back_to_manual_forward_for_empty_predict_output(
    monkeypatch, tmp_path: Path
) -> None:
    _install_fake_training_modules(monkeypatch, tmp_path, predict_return_mode="empty_list")

    df = pd.DataFrame(
        {
            "asset_id": ["AAPL", "AAPL"],
            "time_idx": [0, 1],
            "target_return": [0.1, 0.2],
            "close": [10.0, 11.0],
            "volume": [1000, 1100],
            "day_of_week": [0, 1],
            "month": [1, 1],
        }
    )

    trainer = PytorchForecastingTFTTrainer()
    result = trainer.train(
        df,
        df.copy(),
        df.copy(),
        feature_cols=["close", "volume"],
        target_col="target_return",
        time_idx_col="time_idx",
        group_col="asset_id",
        known_real_cols=["time_idx", "day_of_week", "month"],
        config={},
    )
    assert "val" in result.split_metrics
    assert result.split_metrics["val"]["rmse"] >= 0.0
