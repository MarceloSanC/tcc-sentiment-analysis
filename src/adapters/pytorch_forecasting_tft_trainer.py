from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

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
        df: pd.DataFrame,
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
            from pytorch_lightning import Trainer, seed_everything
            from pytorch_lightning.callbacks import Callback
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pytorch-forecasting is required for TFT training. "
                "Install dependencies before running training."
            ) from exc

        cfg = self._resolve_config(config)
        seed_everything(cfg.seed, workers=True)

        df = df.copy()
        df = df.sort_values(time_idx_col).reset_index(drop=True)

        training_cutoff = df[time_idx_col].max() - cfg.max_prediction_length
        if "training_cutoff_time_idx" in config and config["training_cutoff_time_idx"] is not None:
            training_cutoff = int(config["training_cutoff_time_idx"])

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

        history_cb = HistoryCallback()

        training = TimeSeriesDataSet(
            df[df[time_idx_col] <= training_cutoff],
            time_idx=time_idx_col,
            target=target_col,
            group_ids=[group_col],
            max_encoder_length=cfg.max_encoder_length,
            max_prediction_length=cfg.max_prediction_length,
            time_varying_known_reals=known_real_cols,
            time_varying_unknown_reals=feature_cols,
        )
        validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

        train_dataloader = training.to_dataloader(train=True, batch_size=cfg.batch_size, num_workers=0)
        val_dataloader = validation.to_dataloader(train=False, batch_size=cfg.batch_size, num_workers=0)

        model = TemporalFusionTransformer.from_dataset(
            training,
            learning_rate=cfg.learning_rate,
            hidden_size=cfg.hidden_size,
            attention_head_size=cfg.attention_head_size,
            dropout=cfg.dropout,
            hidden_continuous_size=cfg.hidden_continuous_size,
            loss=QuantileLoss(),
        )

        trainer = Trainer(
            max_epochs=cfg.max_epochs,
            enable_checkpointing=False,
            logger=False,
            callbacks=[history_cb],
        )
        trainer.fit(model, train_dataloader, val_dataloader)

        preds = model.predict(val_dataloader, mode="prediction")
        actuals = torch.cat([y[0] for _, y in iter(val_dataloader)], dim=0)
        preds = preds.detach().cpu().numpy()
        actuals = actuals.detach().cpu().numpy()
        if preds.ndim > 1:
            preds = preds[:, 0]

        metrics = {
            "rmse": float(np.sqrt(np.mean((preds - actuals) ** 2))),
            "mae": float(np.mean(np.abs(preds - actuals))),
        }

        return TrainingResult(model=model, metrics=metrics, history=history_cb.history)
