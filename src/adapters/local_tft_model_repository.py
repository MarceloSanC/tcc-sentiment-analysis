from __future__ import annotations

import json
import pickle
import shutil
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import pandas as pd

from src.interfaces.model_repository import ModelRepository


class LocalTFTModelRepository(ModelRepository):
    """
    Save TFT model artifacts to local filesystem.

    Layout:
      data/models/tft/{ASSET}/{VERSION}/
        model_state.pt
        metrics.json
        history.csv
        features.json
        config.json
        metadata.json
        plots/*.png
    """

    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_training_artifacts(
        self,
        asset_id: str,
        version: str,
        model: Any,
        *,
        metrics: dict[str, float],
        history: list[dict[str, float]],
        split_metrics: dict[str, dict[str, float]],
        features_used: list[str],
        training_window: dict[str, str],
        split_window: dict[str, str],
        config: dict,
        feature_importance: list[dict[str, float | str]] | None = None,
        ablation_results: list[dict[str, float | str]] | None = None,
        checkpoint_path: str | None = None,
        dataset_parameters: dict[str, Any] | None = None,
        plots: dict[str, str] | None = None,
    ) -> str:
        import torch

        asset_dir = self.base_dir / asset_id
        version_dir = asset_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        model_path = version_dir / "model_state.pt"
        torch.save(model.state_dict(), model_path)

        metrics_path = version_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        split_metrics_path = version_dir / "split_metrics.json"
        split_metrics_path.write_text(
            json.dumps(split_metrics, indent=2), encoding="utf-8"
        )

        history_path = version_dir / "history.csv"
        if history:
            pd.DataFrame(history).to_csv(history_path, index=False)
        else:
            history_path.write_text("", encoding="utf-8")

        features_path = version_dir / "features.json"
        features_path.write_text(
            json.dumps({"features_used": features_used}, indent=2), encoding="utf-8"
        )

        config_path = version_dir / "config.json"
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        if dataset_parameters:
            dataset_params_path = version_dir / "dataset_parameters.pkl"
            with dataset_params_path.open("wb") as fp:
                pickle.dump(dataset_parameters, fp)

        checkpoint_out = None
        if checkpoint_path:
            src = Path(checkpoint_path)
            if src.exists() and src.is_file():
                ckpt_dir = version_dir / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                dst = ckpt_dir / "best.ckpt"
                shutil.copy2(src, dst)
                checkpoint_out = str(dst.resolve())

        plots_out: dict[str, str] = {}
        analysis_out: dict[str, str] = {}
        if history:
            try:
                import matplotlib.pyplot as plt

                df_hist = pd.DataFrame(history)
                fig = plt.figure()
                if "train_loss" in df_hist:
                    plt.plot(df_hist["train_loss"], label="train_loss")
                if "val_loss" in df_hist:
                    plt.plot(df_hist["val_loss"], label="val_loss")
                plt.legend()
                plt.title("Training Loss")
                plt.xlabel("epoch")
                plt.ylabel("loss")
                plots_dir = version_dir / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                loss_path = plots_dir / "loss_curve.png"
                fig.savefig(loss_path, dpi=120, bbox_inches="tight")
                plt.close(fig)
                plots_out["loss_curve"] = str(loss_path.resolve())
            except Exception:
                plots_out = {}

        analysis_dir = version_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        if feature_importance:
            fi_path = analysis_dir / "feature_importance.csv"
            pd.DataFrame(feature_importance).to_csv(fi_path, index=False)
            analysis_out["feature_importance_csv"] = str(fi_path.resolve())

        if ablation_results:
            ablation_path = analysis_dir / "ablation_results.csv"
            ablation_df = pd.DataFrame(ablation_results)
            ablation_df.to_csv(ablation_path, index=False)
            analysis_out["ablation_results_csv"] = str(ablation_path.resolve())
            try:
                import matplotlib.pyplot as plt

                if "test_rmse" in ablation_df.columns and "experiment" in ablation_df.columns:
                    fig = plt.figure()
                    ablation_df.plot(
                        x="experiment",
                        y="test_rmse",
                        kind="bar",
                        legend=False,
                        ax=plt.gca(),
                        title="Ablation Test RMSE",
                    )
                    plt.xlabel("experiment")
                    plt.ylabel("rmse")
                    plt.tight_layout()
                    ablation_plot = analysis_dir / "ablation_comparison.png"
                    fig.savefig(ablation_plot, dpi=120, bbox_inches="tight")
                    plt.close(fig)
                    analysis_out["ablation_comparison_plot"] = str(ablation_plot.resolve())
            except Exception:
                pass

        metadata = {
            "model_type": "TemporalFusionTransformer",
            "asset_id": asset_id,
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "training_window": training_window,
            "split_window": split_window,
            "features_used": features_used,
            "metrics": metrics,
            "split_metrics": split_metrics,
            "training_config": config,
        }
        if checkpoint_out:
            metadata["best_checkpoint"] = checkpoint_out
        if analysis_out:
            metadata["analysis_artifacts"] = analysis_out
        merged_plots = {}
        if plots:
            merged_plots.update(plots)
        if plots_out:
            merged_plots.update(plots_out)
        if merged_plots:
            metadata["plots"] = merged_plots
        metadata_path = version_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        return str(version_dir)
