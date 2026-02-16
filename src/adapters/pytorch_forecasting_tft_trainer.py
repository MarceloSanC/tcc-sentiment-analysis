from __future__ import annotations

from dataclasses import dataclass
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.interfaces.model_trainer import ModelTrainer, TrainingResult


@dataclass(frozen=True)
class TFTTrainingConfig:
    max_encoder_length: int = 60
    max_prediction_length: int = 1
    batch_size: int = 64
    max_epochs: int = 20
    learning_rate: float = 1e-3
    hidden_size: int = 16
    attention_head_size: int = 2
    dropout: float = 0.1
    hidden_continuous_size: int = 8
    seed: int = 42
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0


class PytorchForecastingTFTTrainer(ModelTrainer):
    """
    Temporal Fusion Transformer trainer using pytorch-forecasting.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _resolve_config(config: dict) -> TFTTrainingConfig:
        return TFTTrainingConfig(**{**TFTTrainingConfig().__dict__, **config})

    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        *,
        feature_cols: list[str],
        target_col: str,
        time_idx_col: str,
        group_col: str,
        known_real_cols: list[str],
        config: dict,
    ) -> TrainingResult:
        try:
            import torch
            from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
            from pytorch_forecasting.metrics import QuantileLoss
            try:
                from lightning.pytorch import Trainer, seed_everything
                from lightning.pytorch.callbacks import (
                    Callback,
                    EarlyStopping,
                    ModelCheckpoint,
                )
            except Exception:
                from pytorch_lightning import Trainer, seed_everything
                from pytorch_lightning.callbacks import (
                    Callback,
                    EarlyStopping,
                    ModelCheckpoint,
                )
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pytorch-forecasting is required for TFT training. "
                "Install dependencies before running training."
            ) from exc

        cfg = self._resolve_config(config)
        seed_everything(cfg.seed, workers=True)

        train_df = train_df.copy().sort_values(time_idx_col).reset_index(drop=True)
        val_df = val_df.copy().sort_values(time_idx_col).reset_index(drop=True)
        test_df = test_df.copy().sort_values(time_idx_col).reset_index(drop=True)

        class HistoryCallback(Callback):
            def __init__(self) -> None:
                self.history: list[dict[str, float]] = []

            def on_validation_epoch_end(self, trainer: Trainer, pl_module: Any) -> None:
                metrics = {}
                for k, v in trainer.callback_metrics.items():
                    if k in {"train_loss", "val_loss"}:
                        try:
                            metrics[k] = float(v.detach().cpu().item())
                        except Exception:
                            pass
                if metrics:
                    self.history.append(metrics)

        class EpochMetricsPrinter(Callback):
            def on_validation_epoch_end(self, trainer: Trainer, pl_module: Any) -> None:
                m = trainer.callback_metrics
                train_loss = m.get("train_loss")
                val_loss = m.get("val_loss")
                try:
                    train_loss_value = (
                        float(train_loss.detach().cpu().item())
                        if train_loss is not None
                        else float("nan")
                    )
                except Exception:
                    train_loss_value = float("nan")
                try:
                    val_loss_value = (
                        float(val_loss.detach().cpu().item())
                        if val_loss is not None
                        else float("nan")
                    )
                except Exception:
                    val_loss_value = float("nan")
                epoch = getattr(trainer, "current_epoch", -1)
                tqdm.write(
                    f"[epoch={epoch}] "
                    f"train_loss={train_loss_value:.6f} "
                    f"val_loss={val_loss_value:.6f}"
                )

        history_cb = HistoryCallback()
        metrics_printer_cb = EpochMetricsPrinter()

        training = TimeSeriesDataSet(
            train_df,
            time_idx=time_idx_col,
            target=target_col,
            group_ids=[group_col],
            max_encoder_length=cfg.max_encoder_length,
            max_prediction_length=cfg.max_prediction_length,
            time_varying_known_reals=known_real_cols,
            time_varying_unknown_reals=feature_cols,
        )
        validation = TimeSeriesDataSet.from_dataset(
            training, val_df, predict=False, stop_randomization=True
        )
        testing = TimeSeriesDataSet.from_dataset(
            training, test_df, predict=False, stop_randomization=True
        )

        train_dataloader = training.to_dataloader(train=True, batch_size=cfg.batch_size, num_workers=0)
        train_eval_dataloader = training.to_dataloader(
            train=False, batch_size=cfg.batch_size, num_workers=0
        )
        val_dataloader = validation.to_dataloader(train=False, batch_size=cfg.batch_size, num_workers=0)
        test_dataloader = testing.to_dataloader(train=False, batch_size=cfg.batch_size, num_workers=0)

        model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=cfg.learning_rate,
            hidden_size=cfg.hidden_size,
            attention_head_size=cfg.attention_head_size,
            dropout=cfg.dropout,
            hidden_continuous_size=cfg.hidden_continuous_size,
            loss=QuantileLoss(),
        )

        checkpoint_dir = tempfile.mkdtemp(prefix="tft_ckpt_")
        checkpoint_cb = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        )
        early_stopping_cb = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=cfg.early_stopping_patience,
            min_delta=cfg.early_stopping_min_delta,
        )

        trainer = Trainer(
            max_epochs=cfg.max_epochs,
            enable_checkpointing=True,
            logger=False,
            callbacks=[history_cb, metrics_printer_cb, checkpoint_cb, early_stopping_cb],
        )
        trainer.fit(model, train_dataloader, val_dataloader)

        best_model = model
        best_checkpoint_path = checkpoint_cb.best_model_path or None
        if best_checkpoint_path:
            try:
                # PyTorch 2.6 defaults torch.load(..., weights_only=True), which
                # can fail for TFT checkpoints containing custom objects.
                try:
                    checkpoint = torch.load(
                        best_checkpoint_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                except TypeError:
                    checkpoint = torch.load(best_checkpoint_path, map_location="cpu")

                state_dict = checkpoint.get("state_dict", checkpoint)
                best_model.load_state_dict(state_dict)
            except Exception as load_exc:
                # Fallback for compatibility with mocked/unit environments.
                try:
                    best_model = TemporalFusionTransformer.load_from_checkpoint(best_checkpoint_path)
                except Exception as checkpoint_exc:
                    raise RuntimeError(
                        f"Failed to restore best checkpoint: {best_checkpoint_path}"
                    ) from checkpoint_exc

        predict_kwargs = {
            "trainer_kwargs": {
                "logger": False,
                "enable_progress_bar": False,
                "enable_model_summary": False,
            }
        }

        def _manual_forward_arrays(dataloader, split_name: str):
            pred_batches = []
            actual_batches = []
            for x, y in iter(dataloader):
                with torch.no_grad():
                    out = best_model(x)
                if isinstance(out, dict):
                    pred = out.get("prediction", out.get("output", out))
                elif hasattr(out, "prediction"):
                    pred = out.prediction
                elif hasattr(out, "output"):
                    pred = out.output
                else:
                    pred = out

                if not hasattr(pred, "detach"):
                    pred = torch.as_tensor(pred)
                if pred.ndim == 3:
                    pred = pred[:, :, pred.shape[2] // 2]
                if pred.ndim > 1:
                    pred = pred[:, 0]

                act = y[0]
                if act.ndim > 1:
                    act = act[:, 0]

                pred_batches.append(pred.detach().cpu())
                actual_batches.append(act.detach().cpu())

            if len(pred_batches) == 0:
                raise RuntimeError(
                    f"Manual forward produced no batches for split '{split_name}'."
                )

            preds_np = torch.cat(pred_batches, dim=0).numpy()
            actuals_np = torch.cat(actual_batches, dim=0).numpy()
            return preds_np, actuals_np

        def _predict_and_metrics(split_name: str, dataloader) -> dict[str, float]:
            try:
                preds = best_model.predict(dataloader, mode="prediction", **predict_kwargs)
            except TypeError:
                preds = best_model.predict(dataloader, mode="prediction")
            actuals = torch.cat([y[0] for _, y in iter(dataloader)], dim=0)


            def _to_numpy(x):
                # pytorch-forecasting Prediction/Output objects expose fields like
                # `.prediction` or `.output`; handle them before tuple/list fallback.
                if hasattr(x, "prediction"):
                    return _to_numpy(getattr(x, "prediction"))
                if hasattr(x, "output"):
                    return _to_numpy(getattr(x, "output"))
                if isinstance(x, dict):
                    if "prediction" in x:
                        return _to_numpy(x["prediction"])
                    if "output" in x:
                        return _to_numpy(x["output"])
                if hasattr(x, "detach"):
                    return x.detach().cpu().numpy()
                if isinstance(x, (list, tuple)):
                    parts = [_to_numpy(item) for item in x]
                    if len(parts) == 0:
                        raise RuntimeError(
                            f"Model predict returned empty output for split '{split_name}' evaluation."
                        )
                    return np.concatenate(parts, axis=0)
                return np.asarray(x)

            try:
                preds_np = _to_numpy(preds)
                actuals_np = _to_numpy(actuals)
            except RuntimeError as err:
                if "empty output" not in str(err):
                    raise
                preds_np, actuals_np = _manual_forward_arrays(dataloader, split_name)

            if preds_np.ndim > 1:
                preds_np = preds_np[:, 0]
            if actuals_np.ndim > 1:
                actuals_np = actuals_np[:, 0]
            if preds_np.shape[0] != actuals_np.shape[0]:
                raise RuntimeError(
                    "Model predict output length does not match target length "
                    f"(preds={preds_np.shape[0]}, actuals={actuals_np.shape[0]})."
                )

            return {
                "rmse": float(np.sqrt(np.mean((preds_np - actuals_np) ** 2))),
                "mae": float(np.mean(np.abs(preds_np - actuals_np))),
            }

        split_metrics = {
            "train": _predict_and_metrics("train", train_eval_dataloader),
            "val": _predict_and_metrics("val", val_dataloader),
            "test": _predict_and_metrics("test", test_dataloader),
        }
        baseline_test = split_metrics["test"]
        rng = np.random.default_rng(cfg.seed)
        feature_importance: list[dict[str, float | str]] = []
        for feature in feature_cols:
            if feature not in test_df.columns:
                continue
            permuted_test = test_df.copy()
            permuted_test[feature] = rng.permutation(permuted_test[feature].to_numpy())
            permuted_dataset = TimeSeriesDataSet.from_dataset(
                training, permuted_test, predict=True, stop_randomization=True
            )
            permuted_loader = permuted_dataset.to_dataloader(
                train=False, batch_size=cfg.batch_size, num_workers=0
            )
            permuted_metrics = _predict_and_metrics(f"test_permuted_{feature}", permuted_loader)
            feature_importance.append(
                {
                    "feature": feature,
                    "baseline_rmse": float(baseline_test["rmse"]),
                    "permuted_rmse": float(permuted_metrics["rmse"]),
                    "delta_rmse": float(permuted_metrics["rmse"] - baseline_test["rmse"]),
                    "baseline_mae": float(baseline_test["mae"]),
                    "permuted_mae": float(permuted_metrics["mae"]),
                    "delta_mae": float(permuted_metrics["mae"] - baseline_test["mae"]),
                }
            )
        feature_importance.sort(key=lambda x: x["delta_rmse"], reverse=True)

        metrics = split_metrics["val"]

        return TrainingResult(
            model=best_model,
            metrics=metrics,
            history=history_cb.history,
            split_metrics=split_metrics,
            feature_importance=feature_importance,
            checkpoint_path=best_checkpoint_path,
            dataset_parameters=training.get_parameters(),
        )
