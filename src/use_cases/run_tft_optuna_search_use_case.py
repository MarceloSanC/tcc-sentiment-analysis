from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.use_cases.run_tft_model_analysis_use_case import RunTFTModelAnalysisUseCase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunTFTOptunaSearchResult:
    asset: str
    output_dir: str
    study_name: str
    best_value: float | None
    best_params: dict[str, Any]
    top_k_configs: list[dict[str, Any]]


class RunTFTOptunaSearchUseCase:
    def __init__(
        self,
        *,
        train_runner: Any,
        base_training_config: dict[str, Any],
        split_config: dict[str, str] | None,
        walk_forward_config: dict[str, Any] | None,
        replica_seeds: list[int] | None,
        continue_on_error: bool,
        objective_metric: str = "robust_score",
        objective_lambda: float = 1.0,
    ) -> None:
        self.train_runner = train_runner
        self.base_training_config = dict(base_training_config)
        self.split_config = dict(split_config or {})
        self.walk_forward_config = dict(walk_forward_config or {})
        self.replica_seeds = list(replica_seeds or [7, 42, 123])
        self.continue_on_error = bool(continue_on_error)
        self.objective_metric = str(objective_metric or "robust_score")
        self.objective_lambda = float(objective_lambda)

    @staticmethod
    def _feature_label(features: str | None) -> str:
        if not features:
            return "default"
        clean = "".join(ch if ch.isalnum() or ch in {"_", "-", "+", ","} else "_" for ch in features)
        return clean.strip("_") or "default"

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def _suggest_param(self, trial: Any, name: str, spec: Any) -> Any:
        if isinstance(spec, list):
            return trial.suggest_categorical(name, spec)
        if not isinstance(spec, dict):
            raise ValueError(f"Invalid search_space spec for {name}: expected list or object")

        ptype = str(spec.get("type", "")).strip().lower()
        if ptype == "int":
            low = int(spec["low"])
            high = int(spec["high"])
            step = int(spec.get("step", 1))
            log = bool(spec.get("log", False))
            return trial.suggest_int(name, low=low, high=high, step=step, log=log)
        if ptype == "float":
            low = float(spec["low"])
            high = float(spec["high"])
            step_raw = spec.get("step")
            step = None if step_raw is None else float(step_raw)
            log = bool(spec.get("log", False))
            return trial.suggest_float(name, low=low, high=high, step=step, log=log)
        if ptype == "categorical":
            choices = spec.get("choices")
            if not isinstance(choices, list) or not choices:
                raise ValueError(f"Invalid categorical choices for {name}")
            return trial.suggest_categorical(name, choices)

        raise ValueError(f"Unsupported search_space type for {name}: {ptype}")

    def _objective_from_summary(
        self,
        *,
        top_run: dict[str, Any] | None,
    ) -> float:
        if not top_run:
            return float("inf")

        if self.objective_metric == "robust_score":
            value = self._to_float(top_run.get("robust_score"))
            if value is not None:
                return value
            mean_val = self._to_float(top_run.get("mean_val_rmse"))
            std_val = self._to_float(top_run.get("std_val_rmse"))
            if mean_val is not None and std_val is not None:
                return mean_val + (self.objective_lambda * std_val)
            return float("inf")

        if self.objective_metric == "mean_val_rmse":
            value = self._to_float(top_run.get("mean_val_rmse"))
            return value if value is not None else float("inf")

        if self.objective_metric == "mean_test_rmse":
            value = self._to_float(top_run.get("mean_test_rmse"))
            return value if value is not None else float("inf")

        if self.objective_metric == "joint_val_test_rmse":
            mean_val = self._to_float(top_run.get("mean_val_rmse"))
            mean_test = self._to_float(top_run.get("mean_test_rmse"))
            if mean_val is None or mean_test is None:
                return float("inf")
            return (mean_val + mean_test) / 2.0

        raise ValueError(
            "Unsupported objective_metric. "
            "Use one of: robust_score, mean_val_rmse, mean_test_rmse, joint_val_test_rmse"
        )

    def execute(
        self,
        *,
        asset: str,
        models_asset_dir: Path,
        features: str | None,
        search_space: dict[str, Any],
        n_trials: int,
        top_k: int,
        output_subdir: str,
        study_name: str,
        timeout_seconds: int | None,
        sampler_seed: int | None,
    ) -> RunTFTOptunaSearchResult:
        try:
            import optuna
            from optuna.samplers import TPESampler
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Optuna is required. Install it with: pip install optuna"
            ) from exc

        if not search_space:
            raise ValueError("search_space cannot be empty")
        if n_trials <= 0:
            raise ValueError("n_trials must be > 0")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        asset = asset.strip().upper()
        feature_label = self._feature_label(features)

        root_optuna_dir = models_asset_dir / "optuna" / output_subdir
        if features is not None:
            run_output_dir = root_optuna_dir / feature_label
            effective_study_name = f"{study_name}__{feature_label}"
        else:
            run_output_dir = root_optuna_dir
            effective_study_name = study_name
        run_output_dir.mkdir(parents=True, exist_ok=True)

        storage_path = (run_output_dir / "optuna_study.db").resolve()
        storage = f"sqlite:///{str(storage_path).replace('\\', '/')}"

        trials_log: list[dict[str, Any]] = []

        def objective(trial: Any) -> float:
            trial_cfg = dict(self.base_training_config)
            for param_name, spec in search_space.items():
                trial_cfg[param_name] = self._suggest_param(trial, param_name, spec)

            if int(trial_cfg.get("max_prediction_length", 1)) > int(trial_cfg.get("max_encoder_length", 1)):
                raise optuna.exceptions.TrialPruned(
                    "Invalid config: max_prediction_length > max_encoder_length"
                )

            trial_use_case = RunTFTModelAnalysisUseCase(
                train_runner=self.train_runner,
                base_training_config=trial_cfg,
                param_ranges={},
                replica_seeds=self.replica_seeds,
                split_config=self.split_config,
                compute_confidence_interval=False,
                walk_forward_config=self.walk_forward_config,
                generate_comparison_plots=False,
            )

            trial_index = int(getattr(trial, "number", 0))
            trial_sweep_name = f"trial_{trial_index:04d}"
            analysis_cfg = {
                "runner": "optuna",
                "features": features,
                "study_name": effective_study_name,
                "trial_number": trial_index,
                "objective_metric": self.objective_metric,
                "objective_lambda": self.objective_lambda,
                "training_config": trial_cfg,
                "split_config": self.split_config,
                "walk_forward": self.walk_forward_config,
                "replica_seeds": self.replica_seeds,
                "search_space": search_space,
            }

            result = trial_use_case.execute(
                asset=asset,
                models_asset_dir=run_output_dir,
                features=features,
                continue_on_error=self.continue_on_error,
                merge_tests=False,
                max_runs=None,
                output_subdir=trial_sweep_name,
                analysis_config=analysis_cfg,
            )

            top_run = result.top_5_runs[0] if result.top_5_runs else None
            objective_value = self._objective_from_summary(top_run=top_run)

            trial.set_user_attr("sweep_dir", result.sweep_dir)
            trial.set_user_attr("top_run", top_run or {})
            trials_log.append(
                {
                    "trial_number": trial_index,
                    "objective_value": objective_value,
                    "status": "ok" if result.runs_failed == 0 else "partial_failed",
                    "runs_ok": result.runs_ok,
                    "runs_failed": result.runs_failed,
                    "sweep_dir": result.sweep_dir,
                    "top_run": top_run or {},
                    "params": dict(trial_cfg),
                }
            )

            logger.info(
                "Optuna trial completed",
                extra={
                    "trial_number": trial_index,
                    "objective_value": objective_value,
                    "sweep_dir": result.sweep_dir,
                    "runs_failed": result.runs_failed,
                },
            )
            return objective_value

        study = optuna.create_study(
            study_name=effective_study_name,
            storage=storage,
            load_if_exists=True,
            sampler=TPESampler(seed=sampler_seed),
            direction="minimize",
        )

        study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

        completed = [t for t in study.trials if t.value is not None]
        completed_sorted = sorted(completed, key=lambda t: float(t.value))

        top_entries: list[dict[str, Any]] = []
        for t in completed_sorted[:top_k]:
            trial_cfg = dict(self.base_training_config)
            trial_cfg.update(dict(t.params))
            top_entries.append(
                {
                    "rank": len(top_entries) + 1,
                    "trial_number": int(t.number),
                    "objective_value": float(t.value),
                    "params": dict(t.params),
                    "training_config": trial_cfg,
                    "sweep_dir": t.user_attrs.get("sweep_dir"),
                    "top_run": t.user_attrs.get("top_run", {}),
                }
            )

        best_trial = completed_sorted[0] if completed_sorted else None
        best_value = float(best_trial.value) if best_trial is not None else None
        best_params = dict(best_trial.params) if best_trial is not None else {}

        generated_at = datetime.now(timezone.utc).isoformat()
        summary_payload = {
            "asset": asset,
            "features": features,
            "study_name": effective_study_name,
            "generated_at": generated_at,
            "objective_metric": self.objective_metric,
            "objective_lambda": self.objective_lambda,
            "n_trials_requested": int(n_trials),
            "n_trials_completed": int(len(completed_sorted)),
            "top_k": int(top_k),
            "best_value": best_value,
            "best_params": best_params,
            "replica_seeds": self.replica_seeds,
            "split_config": self.split_config,
            "walk_forward": self.walk_forward_config,
            "storage": storage,
            "artifacts": {
                "study_db": str(storage_path),
                "trials_csv": str((run_output_dir / "optuna_trials.csv").resolve()),
                "trials_json": str((run_output_dir / "optuna_trials.json").resolve()),
                "top_k_json": str((run_output_dir / "optuna_top_k_configs.json").resolve()),
                "best_json": str((run_output_dir / "optuna_best_trial.json").resolve()),
                "summary_json": str((run_output_dir / "optuna_summary.json").resolve()),
            },
        }

        trials_df = pd.DataFrame([
            {
                "trial_number": int(t.number),
                "objective_value": float(t.value) if t.value is not None else None,
                "state": str(t.state),
                "params": json.dumps(t.params, ensure_ascii=False, sort_keys=True),
                "sweep_dir": t.user_attrs.get("sweep_dir"),
            }
            for t in study.trials
        ])
        trials_df.to_csv(run_output_dir / "optuna_trials.csv", index=False)
        (run_output_dir / "optuna_trials.json").write_text(
            json.dumps(trials_log, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (run_output_dir / "optuna_top_k_configs.json").write_text(
            json.dumps(top_entries, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (run_output_dir / "optuna_best_trial.json").write_text(
            json.dumps(
                {
                    "study_name": effective_study_name,
                    "best_value": best_value,
                    "best_params": best_params,
                    "best_training_config": ({**self.base_training_config, **best_params} if best_params else dict(self.base_training_config)),
                    "best_sweep_dir": best_trial.user_attrs.get("sweep_dir") if best_trial is not None else None,
                    "best_top_run": best_trial.user_attrs.get("top_run", {}) if best_trial is not None else {},
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (run_output_dir / "optuna_summary.json").write_text(
            json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        logger.info(
            "Optuna search completed",
            extra={
                "asset": asset,
                "study_name": effective_study_name,
                "output_dir": str(run_output_dir),
                "best_value": best_value,
                "top_k": len(top_entries),
            },
        )

        return RunTFTOptunaSearchResult(
            asset=asset,
            output_dir=str(run_output_dir),
            study_name=effective_study_name,
            best_value=best_value,
            best_params=best_params,
            top_k_configs=top_entries,
        )
