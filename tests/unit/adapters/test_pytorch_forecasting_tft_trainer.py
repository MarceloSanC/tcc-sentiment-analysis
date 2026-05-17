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


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorchModule:
    @staticmethod
    def no_grad():
        return _FakeNoGrad()


class _FakeQuantileForwardModel:
    def __init__(self, prediction):
        self.prediction = prediction
        self.loss = types.SimpleNamespace(quantiles=[0.1, 0.5, 0.9])

    def __call__(self, x):
        return {"prediction": _FakeTensor(self.prediction)}


def _manual_forward_case(prediction, *, actual_horizon: int = 3):
    actuals = np.array(
        [
            [10.0 + h for h in range(actual_horizon)],
            [20.0 + h for h in range(actual_horizon)],
        ]
    )
    return PytorchForecastingTFTTrainer._manual_forward_quantiles_and_actuals(
        best_model=_FakeQuantileForwardModel(prediction),
        dataloader=[(_FakeTensor(np.zeros((2, 3))), (_FakeTensor(actuals), None))],
        torch_module=_FakeTorchModule,
    )


def _install_fake_training_modules(
    monkeypatch,
    tmp_path: Path,
    *,
    torch_load_raises: bool = False,
    load_from_checkpoint_raises: bool = False,
    predict_return_mode: str = "tensor",
    raw_quantiles_available: bool = True,
    forward_quantiles_available: bool = False,
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
    fake_lightning = types.ModuleType("lightning")
    fake_lightning_pytorch = types.ModuleType("lightning.pytorch")
    fake_lightning_pytorch_callbacks = types.ModuleType("lightning.pytorch.callbacks")

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
            self.loss = _FakeQuantileLoss([0.1, 0.5, 0.9])

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
            if mode == "raw":
                if not raw_quantiles_available:
                    return []
                # Shape: [batch, horizon=1, quantiles=3]
                q10 = actual + (offset - 0.1)
                q50 = actual + offset
                q90 = actual + (offset + 0.1)
                qcube = np.stack([q10, q50, q90], axis=2)
                return _FakeTensor(qcube)
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
            if forward_quantiles_available:
                actual = x.arr
                q10 = actual + (offset - 0.1)
                q50 = actual + offset
                q90 = actual + (offset + 0.1)
                qcube = np.stack([q10, q50, q90], axis=2)
                return {"prediction": _FakeTensor(qcube)}
            return {"prediction": _FakeTensor(x.arr + offset)}

        def state_dict(self):
            return self._state

        def load_state_dict(self, state_dict):
            self._state = dict(state_dict)

    class _FakeQuantileLoss:
        def __init__(self, quantiles=None):
            self.quantiles = list(quantiles or [0.1, 0.5, 0.9])

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

    # Mirror the same fake API under the modern lightning.pytorch namespace.
    fake_lightning_pytorch.Trainer = _FakeTrainer
    fake_lightning_pytorch.seed_everything = _seed_everything
    fake_lightning_pytorch_callbacks.Callback = _FakeCallback
    fake_lightning_pytorch_callbacks.ModelCheckpoint = _FakeModelCheckpoint
    fake_lightning_pytorch_callbacks.EarlyStopping = _FakeEarlyStopping
    fake_lightning.pytorch = fake_lightning_pytorch

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pytorch_forecasting", fake_pf)
    monkeypatch.setitem(sys.modules, "pytorch_forecasting.metrics", fake_pf_metrics)
    monkeypatch.setitem(sys.modules, "pytorch_lightning", fake_pl)
    monkeypatch.setitem(sys.modules, "pytorch_lightning.callbacks", fake_pl_callbacks)
    monkeypatch.setitem(sys.modules, "lightning", fake_lightning)
    monkeypatch.setitem(sys.modules, "lightning.pytorch", fake_lightning_pytorch)
    monkeypatch.setitem(sys.modules, "lightning.pytorch.callbacks", fake_lightning_pytorch_callbacks)
    return {"FakeTFT": _FakeTFT}


def test_manual_forward_correct_indexing_layout_batch_horizon_quantile_H_gt_1() -> None:
    prediction = np.array(
        [
            [[0.1, 0.5, 0.9], [0.2, 0.6, 1.0], [0.3, 0.7, 1.1]],
            [[1.1, 1.5, 1.9], [1.2, 1.6, 2.0], [1.3, 1.7, 2.1]],
        ]
    )

    q10, q50, q90, actuals = _manual_forward_case(prediction)

    assert q10 is not None
    assert q50 is not None
    assert q90 is not None
    assert actuals is not None
    assert q10.shape == (2, 3)
    assert q50.shape == (2, 3)
    assert q90.shape == (2, 3)
    assert actuals.shape == (2, 3)
    assert q10[0, 0] == pytest.approx(0.1)
    assert q50[0, 0] == pytest.approx(0.5)
    assert q90[0, 0] == pytest.approx(0.9)
    assert q10[0, 1] == pytest.approx(0.2)
    assert q50[0, 1] == pytest.approx(0.6)
    assert q90[0, 1] == pytest.approx(1.0)


def test_manual_forward_correct_indexing_layout_batch_quantile_horizon_H_gt_1() -> None:
    prediction = np.array(
        [
            [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
            [[1.1, 1.2, 1.3, 1.4], [1.5, 1.6, 1.7, 1.8], [1.9, 2.0, 2.1, 2.2]],
        ]
    )

    q10, q50, q90, actuals = _manual_forward_case(prediction, actual_horizon=4)

    assert q10 is not None
    assert q50 is not None
    assert q90 is not None
    assert actuals is not None
    assert q10.shape == (2, 4)
    assert q50.shape == (2, 4)
    assert q90.shape == (2, 4)
    assert actuals.shape == (2, 4)
    assert q10[0, 0] == pytest.approx(0.1)
    assert q50[0, 0] == pytest.approx(0.5)
    assert q90[0, 0] == pytest.approx(0.9)
    assert q10[0, 1] == pytest.approx(0.2)
    assert q50[0, 1] == pytest.approx(0.6)
    assert q90[0, 1] == pytest.approx(1.0)


def test_manual_forward_documents_ambiguous_shape_when_H_eq_Q() -> None:
    prediction = np.array(
        [
            [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7], [0.9, 1.0, 1.1]],
            [[1.1, 1.2, 1.3], [1.5, 1.6, 1.7], [1.9, 2.0, 2.1]],
        ]
    )

    q10, q50, q90, _ = _manual_forward_case(prediction)

    assert q10 is not None
    assert q50 is not None
    assert q90 is not None
    assert q10[0].tolist() == pytest.approx([0.1, 0.5, 0.9])
    assert q50[0].tolist() == pytest.approx([0.2, 0.6, 1.0])
    assert q90[0].tolist() == pytest.approx([0.3, 0.7, 1.1])


def test_manual_forward_returns_none_for_unrecognized_shape() -> None:
    q10, q50, q90, actuals = _manual_forward_case(np.zeros((2, 4, 5)))

    assert q10 is None
    assert q50 is None
    assert q90 is None
    assert actuals is None


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
        config={"compute_feature_importance": True},
    )

    assert set(result.split_metrics.keys()) == {"val", "test"}
    assert "rmse" in result.split_metrics["test"]
    assert "mae" in result.split_metrics["test"]
    assert result.metrics == result.split_metrics["val"]
    assert result.history
    assert result.checkpoint_path is not None
    assert result.dataset_parameters is not None
    assert len(result.feature_importance) == 2
    assert result.feature_importance[0]["delta_rmse"] >= result.feature_importance[1]["delta_rmse"]


def test_trainer_skips_feature_importance_by_default(monkeypatch, tmp_path: Path) -> None:
    _install_fake_training_modules(monkeypatch, tmp_path)

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

    assert result.feature_importance == []


def test_trainer_can_evaluate_train_split_when_enabled(monkeypatch, tmp_path: Path) -> None:
    _install_fake_training_modules(monkeypatch, tmp_path)

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
        config={"evaluate_train_split": True},
    )

    assert set(result.split_metrics.keys()) == {"train", "val", "test"}


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




def test_trainer_uses_forward_quantile_fallback_when_raw_is_unavailable(monkeypatch, tmp_path: Path) -> None:
    _install_fake_training_modules(
        monkeypatch,
        tmp_path,
        raw_quantiles_available=False,
        forward_quantiles_available=True,
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
    result = trainer.train(
        df,
        df.copy(),
        df.copy(),
        feature_cols=["close", "volume"],
        target_col="target_return",
        time_idx_col="time_idx",
        group_col="asset_id",
        known_real_cols=["time_idx", "day_of_week", "month"],
        config={"prediction_mode": "quantile"},
    )
    assert "val" in result.split_predictions
    assert len(result.split_predictions["val"]["quantile_p10_matrix"]) > 0
def test_trainer_raises_when_quantile_outputs_are_unavailable(monkeypatch, tmp_path: Path) -> None:
    _install_fake_training_modules(monkeypatch, tmp_path, raw_quantiles_available=False)

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
    with pytest.raises(RuntimeError, match="Quantile extraction failed"):
        trainer.train(
            df,
            df.copy(),
            df.copy(),
            feature_cols=["close", "volume"],
            target_col="target_return",
            time_idx_col="time_idx",
            group_col="asset_id",
            known_real_cols=["time_idx", "day_of_week", "month"],
            config={"prediction_mode": "quantile"},
        )
