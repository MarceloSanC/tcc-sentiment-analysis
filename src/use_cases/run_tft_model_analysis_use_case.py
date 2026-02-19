from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.domain.services.tft_sweep_experiment_builder import (
    SweepExperiment,
    build_one_at_a_time_experiments,
)

logger = logging.getLogger(__name__)


DEFAULT_SWEEP_PARAM_RANGES: dict[str, list[Any]] = {
    "max_encoder_length": [120],
    "batch_size": [32, 64, 128],
    "max_epochs": [20, 30, 50],
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
    "hidden_size": [16, 32, 64],
    "attention_head_size": [1, 2, 4],
    "dropout": [0.05, 0.1, 0.2, 0.3],
    "hidden_continuous_size": [4, 8, 16],
    "early_stopping_patience": [3, 5, 10],
    "early_stopping_min_delta": [0.0, 1e-5, 1e-4],
}


@dataclass
class AnalysisRunRecord:
    run_label: str
    varied_param: str | None
    varied_value: Any | None
    config_signature: str
    version: str | None
    status: str
    error: str | None = None
    val_rmse: float | None = None
    val_mae: float | None = None
    test_rmse: float | None = None
    test_mae: float | None = None


@dataclass(frozen=True)
class RunTFTModelAnalysisResult:
    asset: str
    sweep_name: str
    sweep_dir: str
    runs_ok: int
    runs_failed: int
    top_5_runs: list[dict[str, Any]]


class RunTFTModelAnalysisUseCase:
    def __init__(
        self,
        *,
        train_runner: Any,
        base_training_config: dict[str, Any],
        param_ranges: dict[str, list[Any]] | None = None,
        replica_seeds: list[int] | None = None,
        split_config: dict[str, str] | None = None,
        compute_confidence_interval: bool = False,
    ) -> None:
        self.train_runner = train_runner
        self.base_training_config = dict(base_training_config)
        self.param_ranges = param_ranges or DEFAULT_SWEEP_PARAM_RANGES
        self.replica_seeds = replica_seeds or [7, 42, 123]
        self.split_config = dict(split_config or {})
        self.compute_confidence_interval = bool(compute_confidence_interval)

    @staticmethod
    def _load_metadata(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _config_signature(config: dict[str, Any]) -> str:
        normalized = {k: v for k, v in config.items() if k != "seed"}
        return json.dumps(normalized, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _collect_all_models(models_asset_dir: Path) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for run_dir in sorted([p for p in models_asset_dir.iterdir() if p.is_dir()]):
            meta_path = run_dir / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                meta = RunTFTModelAnalysisUseCase._load_metadata(meta_path)
            except Exception:
                continue
            split_metrics = meta.get("split_metrics", {})
            test_metrics = split_metrics.get("test", {})
            val_metrics = split_metrics.get("val", {})
            train_metrics = split_metrics.get("train", {})
            training_cfg = meta.get("training_config", {})
            rows.append(
                {
                    "version": meta.get("version", run_dir.name),
                    "created_at": meta.get("created_at"),
                    "feature_set_tag": training_cfg.get("feature_set_tag"),
                    "features_count": len(meta.get("features_used", [])),
                    "test_rmse": test_metrics.get("rmse"),
                    "test_mae": test_metrics.get("mae"),
                    "val_rmse": val_metrics.get("rmse"),
                    "val_mae": val_metrics.get("mae"),
                    "train_rmse": train_metrics.get("rmse"),
                    "train_mae": train_metrics.get("mae"),
                    "learning_rate": training_cfg.get("learning_rate"),
                    "hidden_size": training_cfg.get("hidden_size"),
                    "max_encoder_length": training_cfg.get("max_encoder_length"),
                    "batch_size": training_cfg.get("batch_size"),
                    "dropout": training_cfg.get("dropout"),
                    "seed": training_cfg.get("seed"),
                }
            )
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["test_rmse", "test_mae"], ascending=[True, True]).reset_index(
                drop=True
            )
        return df

    @staticmethod
    def _build_param_impact(
        run_df: pd.DataFrame,
        baseline_test_rmse: float,
        baseline_test_mae: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if run_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        varied = run_df[run_df["varied_param"].notna()].copy()
        if varied.empty:
            return pd.DataFrame(), pd.DataFrame()
        varied["delta_test_rmse_vs_baseline"] = varied["test_rmse"] - baseline_test_rmse
        varied["delta_test_mae_vs_baseline"] = varied["test_mae"] - baseline_test_mae
        summary = (
            varied.groupby("varied_param", dropna=True)
            .agg(
                runs=("run_label", "count"),
                best_test_rmse=("test_rmse", "min"),
                avg_test_rmse=("test_rmse", "mean"),
                median_test_rmse=("test_rmse", "median"),
                best_delta_rmse_vs_baseline=("delta_test_rmse_vs_baseline", "min"),
                worst_delta_rmse_vs_baseline=("delta_test_rmse_vs_baseline", "max"),
                avg_delta_rmse_vs_baseline=("delta_test_rmse_vs_baseline", "mean"),
                avg_delta_mae_vs_baseline=("delta_test_mae_vs_baseline", "mean"),
            )
            .reset_index()
            .sort_values("best_delta_rmse_vs_baseline", ascending=True)
        )
        return varied.sort_values("test_rmse", ascending=True), summary

    @staticmethod
    def _build_config_ranking(
        run_df: pd.DataFrame,
        *,
        compute_confidence_interval: bool,
    ) -> pd.DataFrame:
        if run_df.empty:
            return pd.DataFrame()
        ok = run_df[run_df["status"] == "ok"].copy()
        if ok.empty:
            return pd.DataFrame()
        ok["val_rmse"] = pd.to_numeric(ok["val_rmse"], errors="coerce")
        ok["val_mae"] = pd.to_numeric(ok["val_mae"], errors="coerce")
        ok["test_rmse"] = pd.to_numeric(ok["test_rmse"], errors="coerce")
        ok["test_mae"] = pd.to_numeric(ok["test_mae"], errors="coerce")

        candidates = ok[ok["val_rmse"].notna()].copy()
        if candidates.empty:
            candidates = ok

        ranking = (
            candidates.groupby("config_signature", dropna=False)
            .agg(
                n_runs=("run_label", "count"),
                run_label=("run_label", "first"),
                varied_param=("varied_param", "first"),
                varied_value=("varied_value", "first"),
                mean_val_rmse=("val_rmse", "mean"),
                std_val_rmse=("val_rmse", "std"),
                mean_val_mae=("val_mae", "mean"),
                std_val_mae=("val_mae", "std"),
                mean_test_rmse=("test_rmse", "mean"),
                std_test_rmse=("test_rmse", "std"),
                mean_test_mae=("test_mae", "mean"),
                std_test_mae=("test_mae", "std"),
            )
            .reset_index()
        )
        for col in ["std_val_rmse", "std_val_mae", "std_test_rmse", "std_test_mae"]:
            ranking[col] = ranking[col].fillna(0.0)
        ranking["robust_score"] = ranking["mean_val_rmse"] + ranking["std_val_rmse"]
        if compute_confidence_interval:
            z = 1.96
            n = ranking["n_runs"].astype(float).replace(0.0, np.nan)
            ranking["ci95_val_rmse_low"] = ranking["mean_val_rmse"] - z * (
                ranking["std_val_rmse"] / np.sqrt(n)
            )
            ranking["ci95_val_rmse_high"] = ranking["mean_val_rmse"] + z * (
                ranking["std_val_rmse"] / np.sqrt(n)
            )
            ranking["ci95_test_rmse_low"] = ranking["mean_test_rmse"] - z * (
                ranking["std_test_rmse"] / np.sqrt(n)
            )
            ranking["ci95_test_rmse_high"] = ranking["mean_test_rmse"] + z * (
                ranking["std_test_rmse"] / np.sqrt(n)
            )
        return ranking.sort_values(
            ["mean_val_rmse", "std_val_rmse", "mean_val_mae", "mean_test_rmse"],
            ascending=[True, True, True, True],
        ).reset_index(drop=True)

    def execute(
        self,
        *,
        asset: str,
        models_asset_dir: Path,
        features: str | None = None,
        continue_on_error: bool = False,
        max_runs: int | None = None,
        output_subdir: str | None = None,
        analysis_config: dict[str, Any] | None = None,
    ) -> RunTFTModelAnalysisResult:
        asset = asset.strip().upper()
        models_asset_dir.mkdir(parents=True, exist_ok=True)
        sweep_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sweep_name = output_subdir or f"sweep_{sweep_ts}"
        sweep_dir = models_asset_dir / "sweeps" / sweep_name
        sweep_dir.mkdir(parents=True, exist_ok=True)
        sweep_models_dir = sweep_dir / "models"
        sweep_models_dir.mkdir(parents=True, exist_ok=True)
        sweep_models_asset_staging_dir = sweep_models_dir / asset
        sweep_models_asset_staging_dir.mkdir(parents=True, exist_ok=True)
        if analysis_config is not None:
            (sweep_dir / "analysis_config.json").write_text(
                json.dumps(analysis_config, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        experiments = build_one_at_a_time_experiments(
            base_config=self.base_training_config,
            param_ranges=self.param_ranges,
        )
        if max_runs is not None:
            experiments = experiments[: max(1, max_runs)]

        logger.info(
            "Starting TFT model analysis",
            extra={
                "asset": asset,
                "runs": len(experiments),
                "features": features or "(default from main_train_tft)",
                "sweep_dir": str(sweep_dir),
            },
        )

        run_records: list[AnalysisRunRecord] = []

        for idx, exp in enumerate(experiments, start=1):
            for seed in self.replica_seeds:
                logger.info(
                    "Analysis run started",
                    extra={
                        "run_index": idx,
                        "total_runs": len(experiments),
                        "run_label": exp.run_label,
                        "seed": seed,
                    },
                )
                cfg_with_seed = dict(exp.config)
                cfg_with_seed["seed"] = seed
                run_label = f"{exp.run_label}|seed={seed}"
                try:
                    version, metadata = self.train_runner.run(
                        asset=asset,
                        features=features,
                        config=cfg_with_seed,
                        split_config=self.split_config or None,
                        models_asset_dir=sweep_models_asset_staging_dir,
                    )
                    if version is None or metadata is None:
                        run_records.append(
                            AnalysisRunRecord(
                                run_label=run_label,
                                varied_param=exp.varied_param,
                                varied_value=exp.varied_value,
                                config_signature=self._config_signature(cfg_with_seed),
                                version=version,
                                status="failed",
                                error="Missing generated model version or metadata.json",
                            )
                        )
                        if not continue_on_error:
                            break
                        continue

                    val_metrics = metadata.get("split_metrics", {}).get("val", {})
                    test_metrics = metadata.get("split_metrics", {}).get("test", {})
                    val_rmse = float(val_metrics.get("rmse"))
                    val_mae = float(val_metrics.get("mae"))
                    test_rmse = float(test_metrics.get("rmse"))
                    test_mae = float(test_metrics.get("mae"))
                    run_records.append(
                        AnalysisRunRecord(
                            run_label=run_label,
                            varied_param=exp.varied_param,
                            varied_value=exp.varied_value,
                            config_signature=self._config_signature(cfg_with_seed),
                            version=version,
                            status="ok",
                            val_rmse=val_rmse,
                            val_mae=val_mae,
                            test_rmse=test_rmse,
                            test_mae=test_mae,
                        )
                    )
                    staged_version_dir = sweep_models_asset_staging_dir / version
                    flat_version_dir = sweep_models_dir / version
                    if staged_version_dir.exists():
                        if flat_version_dir.exists():
                            shutil.rmtree(flat_version_dir)
                        shutil.move(str(staged_version_dir), str(flat_version_dir))
                except Exception as exc:
                    run_records.append(
                        AnalysisRunRecord(
                            run_label=run_label,
                            varied_param=exp.varied_param,
                            varied_value=exp.varied_value,
                            config_signature=self._config_signature(cfg_with_seed),
                            version=None,
                            status="failed",
                            error=str(exc),
                        )
                    )
                    logger.exception(
                        "Analysis run failed", extra={"run_label": exp.run_label, "seed": seed}
                    )
                    if not continue_on_error:
                        break
            if run_records and run_records[-1].status == "failed" and not continue_on_error:
                break

        run_df = pd.DataFrame([asdict(r) for r in run_records])
        run_df.to_csv(sweep_dir / "sweep_runs.csv", index=False)
        (sweep_dir / "sweep_runs.json").write_text(
            json.dumps(run_df.to_dict(orient="records"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        all_models_df = self._collect_all_models(sweep_models_dir)
        all_models_df.to_csv(sweep_dir / "all_models_ranked.csv", index=False)

        baseline_rmse: float | None = None
        baseline_mae: float | None = None
        if not run_df.empty:
            baseline_rows = run_df[
                (run_df["status"] == "ok") & (run_df["varied_param"].isna())
            ].copy()
            if not baseline_rows.empty:
                baseline_rmse = float(pd.to_numeric(baseline_rows["test_rmse"], errors="coerce").mean())
                baseline_mae = float(pd.to_numeric(baseline_rows["test_mae"], errors="coerce").mean())

        if baseline_rmse is not None and baseline_mae is not None:
            impact_detail, impact_summary = self._build_param_impact(
                run_df[run_df["status"] == "ok"].copy(),
                baseline_rmse,
                baseline_mae,
            )
            impact_detail.to_csv(sweep_dir / "param_impact_detail.csv", index=False)
            impact_summary.to_csv(sweep_dir / "param_impact_summary.csv", index=False)
        config_ranking = self._build_config_ranking(
            run_df,
            compute_confidence_interval=self.compute_confidence_interval,
        )
        config_ranking.to_csv(sweep_dir / "config_ranking.csv", index=False)
        top_5_runs = config_ranking.head(5).to_dict(orient="records")

        summary = {
            "asset": asset,
            "sweep_name": sweep_name,
            "features": features or "(default from main_train_tft)",
            "runs_total": int(len(run_df)),
            "runs_ok": int((run_df["status"] == "ok").sum()) if not run_df.empty else 0,
            "runs_failed": int((run_df["status"] == "failed").sum()) if not run_df.empty else 0,
            "baseline_test_rmse": baseline_rmse,
            "baseline_test_mae": baseline_mae,
            "replica_seeds": self.replica_seeds,
            "split_config": self.split_config or None,
            "compute_confidence_interval": self.compute_confidence_interval,
            "top_5_runs": top_5_runs,
            "artifacts": {
                "sweep_runs_csv": str(sweep_dir / "sweep_runs.csv"),
                "sweep_models_dir": str(sweep_models_dir),
                "all_models_ranked_csv": str(sweep_dir / "all_models_ranked.csv"),
                "config_ranking_csv": str(sweep_dir / "config_ranking.csv"),
                "param_impact_detail_csv": str(sweep_dir / "param_impact_detail.csv"),
                "param_impact_summary_csv": str(sweep_dir / "param_impact_summary.csv"),
                "analysis_config_json": str(sweep_dir / "analysis_config.json")
                if analysis_config is not None
                else None,
            },
        }
        (sweep_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info(
            "TFT model analysis completed",
            extra={
                "asset": asset,
                "sweep_dir": str(sweep_dir),
                "runs_ok": summary["runs_ok"],
                "runs_failed": summary["runs_failed"],
                "top_5_runs": summary["top_5_runs"],
            },
        )
        return RunTFTModelAnalysisResult(
            asset=asset,
            sweep_name=sweep_name,
            sweep_dir=str(sweep_dir),
            runs_ok=summary["runs_ok"],
            runs_failed=summary["runs_failed"],
            top_5_runs=top_5_runs,
        )
