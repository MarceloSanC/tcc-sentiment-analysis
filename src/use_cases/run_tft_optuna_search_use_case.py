from __future__ import annotations

import json
import logging
import math
import shutil

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

import pandas as pd

from src.domain.services.sweep_eta_estimator import SweepEtaEstimator
from src.infrastructure.schemas.model_artifact_schema import TFT_TRAINING_DEFAULTS
from src.use_cases.run_tft_model_analysis_use_case import RunTFTModelAnalysisUseCase
from src.use_cases.test_pipeline_common import validate_train_runner_contract
from src.utils.matplotlib_backend import ensure_non_interactive_matplotlib_backend
from src.utils.path_policy import to_project_relative

logger = logging.getLogger(__name__)

_PINK = "\033[95m"
_RESET = "\033[0m"


@dataclass(frozen=True)
class RunTFTOptunaSearchResult:
    asset: str
    output_dir: str
    study_name: str
    best_value: float | None
    best_params: dict[str, Any]
    top_k_configs: list[dict[str, Any]]


class RunTFTOptunaSearchUseCase:
    VALID_OBJECTIVE_METRICS: ClassVar[tuple[str, ...]] = ("robust_score", "mean_val_rmse")

    def __init__(
        self,
        *,
        train_runner: Any,
        base_training_config: dict[str, Any],
        split_config: dict[str, str] | None,
        walk_forward_config: dict[str, Any] | None,
        replica_seeds: list[int] | None,
        continue_on_error: bool,
        merge_tests: bool = False,
        objective_metric: Literal["robust_score", "mean_val_rmse"] = "robust_score",
        objective_lambda: float = 1.0,
    ) -> None:
        validate_train_runner_contract(train_runner)
        self.train_runner = train_runner
        self.base_training_config = dict(base_training_config)
        self.split_config = dict(split_config or {})
        self.walk_forward_config = dict(walk_forward_config or {})
        self.replica_seeds = list(replica_seeds or [7, 42, 123])
        self.continue_on_error = bool(continue_on_error)
        self.merge_tests = bool(merge_tests)
        normalized_metric = str(objective_metric or "robust_score")
        # Stage 13: fail loud at __init__ instead of waiting until the first trial
        # runs _objective_from_summary — prevents starting a study under a
        # leakage-prone metric (M1-Q4).
        if normalized_metric not in self.VALID_OBJECTIVE_METRICS:
            raise ValueError(
                f"objective_metric={normalized_metric!r} invalido. "
                f"Opcoes validas: {self.VALID_OBJECTIVE_METRICS}. "
                "Stage 13: mean_test_rmse e joint_val_test_rmse removidos "
                "(leakage metodologico — usar test set como criterio de HPO "
                "viola separacao val/test; ver A_code_audit.md §M1-Q4)."
            )
        self.objective_metric = normalized_metric
        self.objective_lambda = float(objective_lambda)

    @staticmethod
    def _normalize_training_config_for_merge(cfg: dict[str, Any] | None) -> dict[str, Any]:
        """
        Normalize training configs for merge compatibility checks.
        Fills missing keys with current defaults so older summaries remain
        merge-compatible after additive config fields are introduced.
        """
        normalized = dict(TFT_TRAINING_DEFAULTS)
        if isinstance(cfg, dict):
            normalized.update(cfg)
        return normalized

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

    @staticmethod
    def _is_completed_trial(trial: Any) -> bool:
        try:
            state_name = str(getattr(getattr(trial, "state", None), "name", "")).upper()
        except Exception:
            state_name = ""
        if state_name != "COMPLETE":
            return False
        value = RunTFTOptunaSearchUseCase._to_float(getattr(trial, "value", None))
        return value is not None and math.isfinite(value)

    @classmethod
    def _count_completed_trials(cls, study: Any) -> int:
        return int(sum(1 for t in getattr(study, "trials", []) if cls._is_completed_trial(t)))

    @staticmethod
    def _cleanup_latest_incomplete_trial_artifacts(run_output_dir: Path, study: Any) -> int | None:
        incomplete_numbers: list[int] = []
        for t in getattr(study, "trials", []):
            try:
                state_name = str(getattr(getattr(t, "state", None), "name", "")).upper()
            except Exception:
                state_name = ""
            if state_name == "COMPLETE":
                continue
            try:
                incomplete_numbers.append(int(getattr(t, "number", -1)))
            except Exception:
                continue

        if not incomplete_numbers:
            return None

        last_incomplete = max(incomplete_numbers)
        sweep_dir = run_output_dir / "sweeps" / f"trial_{last_incomplete:04d}"
        if sweep_dir.exists() and sweep_dir.is_dir():
            shutil.rmtree(sweep_dir, ignore_errors=True)
        return last_incomplete

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

        # Defense in depth: unreachable if __init__ validation is intact.
        raise ValueError(
            "Unsupported objective_metric. "
            f"Use one of: {self.VALID_OBJECTIVE_METRICS}"
        )

    @staticmethod
    def _save_val_test_metrics_plot(
        *,
        run_output_dir: Path,
        entries: list[dict[str, Any]],
        filename: str,
        plot_title_suffix: str,
        x_axis_label: str,
    ) -> str | None:
        try:
            ensure_non_interactive_matplotlib_backend()
            import matplotlib.pyplot as plt
        except Exception:
            return None

        if not entries:
            return None

        x = list(range(len(entries)))
        labels = [
            f"R{int(entry.get('rank', idx + 1))} | T{int(entry.get('trial_number', idx))}"
            for idx, entry in enumerate(entries)
        ]

        metric_specs = [
            ("RMSE", "mean_val_rmse", "std_val_rmse", "mean_test_rmse", "std_test_rmse"),
            ("MAE", "mean_val_mae", "std_val_mae", "mean_test_mae", "std_test_mae"),
            ("DA", "mean_val_da", "std_val_da", "mean_test_da", "std_test_da"),
        ]

        fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
        if not isinstance(axes, (list, tuple)):
            axes = list(axes)

        has_any_metric = False
        for ax, (title, val_mean_key, val_std_key, test_mean_key, test_std_key) in zip(axes, metric_specs):
            val_mean: list[float] = []
            val_std: list[float] = []
            test_mean: list[float] = []
            test_std: list[float] = []

            for entry in entries:
                top_run = entry.get("top_run", {}) or {}
                val_m = top_run.get(val_mean_key)
                val_s = top_run.get(val_std_key)
                test_m = top_run.get(test_mean_key)
                test_s = top_run.get(test_std_key)

                val_mean.append(float(val_m) if val_m is not None else math.nan)
                val_std.append(float(val_s) if val_s is not None else 0.0)
                test_mean.append(float(test_m) if test_m is not None else math.nan)
                test_std.append(float(test_s) if test_s is not None else 0.0)

            if any(not math.isnan(v) for v in val_mean):
                has_any_metric = True
                val_low = [m - s if not math.isnan(m) else math.nan for m, s in zip(val_mean, val_std)]
                val_h = [2.0 * s if not math.isnan(m) else math.nan for m, s in zip(val_mean, val_std)]
                val_x = [xi - 0.12 for xi in x]
                ax.plot(x, val_mean, marker="o", linewidth=1.8, color="#1f77b4", label="val mean")
                ax.bar(
                    val_x,
                    val_h,
                    bottom=val_low,
                    width=0.22,
                    color="#1f77b4",
                    edgecolor="#1f77b4",
                    linewidth=0.8,
                    alpha=0.16,
                    label="val +- std (box)",
                )

            if any(not math.isnan(v) for v in test_mean):
                has_any_metric = True
                test_low = [m - s if not math.isnan(m) else math.nan for m, s in zip(test_mean, test_std)]
                test_h = [2.0 * s if not math.isnan(m) else math.nan for m, s in zip(test_mean, test_std)]
                test_x = [xi + 0.12 for xi in x]
                ax.plot(x, test_mean, marker="s", linewidth=1.8, color="#ff7f0e", label="test mean")
                ax.bar(
                    test_x,
                    test_h,
                    bottom=test_low,
                    width=0.22,
                    color="#ff7f0e",
                    edgecolor="#ff7f0e",
                    linewidth=0.8,
                    alpha=0.16,
                    label="test +- std (box)",
                )

            ax.set_title(f"{title} - {plot_title_suffix}")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.legend(loc="best")

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(labels, rotation=40, ha="right")
        axes[-1].set_xlabel(x_axis_label)

        fig.tight_layout()
        plot_path = run_output_dir / filename
        if has_any_metric:
            fig.savefig(plot_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            return to_project_relative(plot_path)

        plt.close(fig)
        return None

    @staticmethod
    def _existing_summary(path: Path) -> dict[str, Any] | None:
        if not path.exists() or not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _contains_value(values: list[Any], target: Any) -> bool:
        for candidate in values:
            if candidate == target:
                return True
            try:
                if float(candidate) == float(target):
                    return True
            except Exception:
                continue
        return False

    @classmethod
    def _validate_merge_context(
        cls,
        *,
        existing_summary: dict[str, Any],
        asset: str,
        features: str | None,
        study_name: str,
        objective_metric: str,
        objective_lambda: float,
        search_space: dict[str, Any],
        base_training_config: dict[str, Any],
        split_config: dict[str, str],
        walk_forward: dict[str, Any],
        replica_seeds: list[int],
    ) -> None:
        if str(existing_summary.get("asset") or "").strip().upper() != asset:
            raise ValueError("merge_tests requires same asset as previous Optuna run.")
        if (existing_summary.get("features") or None) != (features or None):
            raise ValueError("merge_tests requires same features as previous Optuna run.")
        if str(existing_summary.get("study_name") or "").strip() != str(study_name).strip():
            raise ValueError("merge_tests requires same study_name.")
        if str(existing_summary.get("objective_metric") or "").strip() != str(objective_metric).strip():
            raise ValueError("merge_tests requires same objective_metric.")
        try:
            old_lambda = float(existing_summary.get("objective_lambda"))
        except Exception:
            old_lambda = None
        if old_lambda is None or float(old_lambda) != float(objective_lambda):
            raise ValueError("merge_tests requires same objective_lambda.")
        if dict(existing_summary.get("split_config") or {}) != dict(split_config or {}):
            raise ValueError("merge_tests requires same split_config.")
        if dict(existing_summary.get("walk_forward") or {}) != dict(walk_forward or {}):
            raise ValueError("merge_tests requires same walk_forward config.")
        if dict(existing_summary.get("best_params") or {}) is not None:
            # best_params may change naturally; intentionally ignored.
            pass
        if dict(existing_summary.get("search_space") or search_space) != dict(search_space):
            # Older summaries may not store search_space. If present, enforce equality.
            if "search_space" in existing_summary:
                raise ValueError("merge_tests requires same search_space.")
        existing_base_cfg = cls._normalize_training_config_for_merge(
            dict(existing_summary.get("base_training_config") or {})
        )
        current_base_cfg = cls._normalize_training_config_for_merge(
            dict(base_training_config or {})
        )
        if existing_base_cfg != current_base_cfg:
            if "base_training_config" in existing_summary:
                raise ValueError("merge_tests requires same base training_config.")

        old_seeds_raw = existing_summary.get("replica_seeds") or []
        old_seeds = list(old_seeds_raw) if isinstance(old_seeds_raw, list) else []
        for old_seed in old_seeds:
            if not cls._contains_value(replica_seeds, old_seed):
                raise ValueError(
                    "merge_tests cannot remove old replica_seeds values. "
                    f"Missing seed: {old_seed}"
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

        if self.merge_tests:
            prev_summary = self._existing_summary(run_output_dir / "optuna_summary.json")
            if prev_summary is not None:
                self._validate_merge_context(
                    existing_summary=prev_summary,
                    asset=asset,
                    features=features,
                    study_name=effective_study_name,
                    objective_metric=self.objective_metric,
                    objective_lambda=self.objective_lambda,
                    search_space=search_space,
                    base_training_config=self.base_training_config,
                    split_config=self.split_config,
                    walk_forward=self.walk_forward_config,
                    replica_seeds=self.replica_seeds,
                )

        storage_path = run_output_dir / "optuna_study.db"
        storage_rel = to_project_relative(storage_path) or str(storage_path)
        storage = f"sqlite:///{storage_rel.replace('\\', '/')}"

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
                "schema_version": "1.0",
                "test_type": "optuna",
                "run_origin": "optuna_trial",
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
                "merge_tests": self.merge_tests,
            }

            result = trial_use_case.execute(
                asset=asset,
                models_asset_dir=run_output_dir,
                features=features,
                continue_on_error=self.continue_on_error,
                merge_tests=self.merge_tests,
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

        requested_total_trials = int(n_trials)
        if self.merge_tests:
            completed_before_merge = self._count_completed_trials(study)
            remaining_trials = max(0, requested_total_trials - completed_before_merge)
            cleaned_trial = self._cleanup_latest_incomplete_trial_artifacts(run_output_dir, study)
            if cleaned_trial is not None:
                logger.warning(
                    "Removed latest incomplete trial artifacts before merge resume",
                    extra={
                        "study_name": effective_study_name,
                        "trial_number": cleaned_trial,
                        "sweep_dir": str(run_output_dir / "sweeps" / f"trial_{cleaned_trial:04d}"),
                    },
                )
            logger.info(
                "Merge resume computed remaining trials",
                extra={
                    "study_name": effective_study_name,
                    "requested_total_trials": requested_total_trials,
                    "completed_trials_before_resume": completed_before_merge,
                    "remaining_trials_to_run": remaining_trials,
                },
            )
            n_trials_to_run = remaining_trials
        else:
            n_trials_to_run = requested_total_trials

        finished_before = len([t for t in study.trials if t.state.is_finished()])
        eta_estimator = SweepEtaEstimator(total_trials=int(n_trials_to_run))

        def on_trial_complete(study_obj: Any, trial_obj: Any) -> None:
            duration_seconds = None
            trial_duration = getattr(trial_obj, "duration", None)
            if trial_duration is not None:
                try:
                    duration_seconds = float(trial_duration.total_seconds())
                except Exception:
                    duration_seconds = None
            eta_estimator.add_sample(duration_seconds)

            finished_now = len([t for t in study_obj.trials if t.state.is_finished()])
            completed_current = max(0, finished_now - finished_before)
            avg_seconds = eta_estimator.avg_seconds
            eta_seconds = eta_estimator.estimate_remaining_seconds(completed_current)
            progress_pct = (
                (completed_current / float(n_trials_to_run)) * 100.0
                if n_trials_to_run > 0
                else 100.0
            )
            message = f"{_PINK}Optuna sweep progress{_RESET}"
            logger.info(
                message,
                extra={
                    "study_name": effective_study_name,
                    "features": features or "(default)",
                    "completed_trials": completed_current,
                    "total_trials": int(n_trials_to_run),
                    "progress_pct": round(progress_pct, 2),
                    "avg_trial_seconds": round(avg_seconds, 2) if avg_seconds is not None else None,
                    "eta_sweep": SweepEtaEstimator.format_seconds(eta_seconds),
                    "trial_number": int(getattr(trial_obj, "number", -1)),
                },
            )

        if n_trials_to_run > 0:
            study.optimize(
                objective,
                n_trials=n_trials_to_run,
                timeout=timeout_seconds,
                callbacks=[on_trial_complete],
            )
        else:
            logger.info(
                "No remaining trials to run for merge resume",
                extra={
                    "study_name": effective_study_name,
                    "requested_total_trials": requested_total_trials,
                },
            )

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

        all_entries: list[dict[str, Any]] = []
        for idx, t in enumerate(completed_sorted, start=1):
            trial_cfg = dict(self.base_training_config)
            trial_cfg.update(dict(t.params))
            all_entries.append(
                {
                    "rank": idx,
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
        top_k_metrics_plot_path = self._save_val_test_metrics_plot(
            run_output_dir=run_output_dir,
            entries=top_entries,
            filename="optuna_top_k_val_test_metrics.png",
            plot_title_suffix="Top-k Optuna Trials",
            x_axis_label="Top-k ranking by objective value",
        )
        all_trials_metrics_plot_path = self._save_val_test_metrics_plot(
            run_output_dir=run_output_dir,
            entries=all_entries,
            filename="optuna_all_trials_val_test_metrics.png",
            plot_title_suffix="All Optuna Trials",
            x_axis_label="Trial ranking by objective value",
        )

        generated_at = datetime.now(UTC).isoformat()
        summary_payload = {
            "asset": asset,
            "features": features,
            "study_name": effective_study_name,
            "generated_at": generated_at,
            "objective_metric": self.objective_metric,
            "objective_lambda": self.objective_lambda,
            "n_trials_requested": int(requested_total_trials),
            "n_trials_completed": int(len(completed_sorted)),
            "top_k": int(top_k),
            "best_value": best_value,
            "best_params": best_params,
            "replica_seeds": self.replica_seeds,
            "split_config": self.split_config,
            "walk_forward": self.walk_forward_config,
            "search_space": search_space,
            "base_training_config": self.base_training_config,
            "merge_tests": self.merge_tests,
            "storage": storage,
            "artifacts": {
                "study_db": to_project_relative(storage_path),
                "trials_csv": to_project_relative(run_output_dir / "optuna_trials.csv"),
                "trials_json": to_project_relative(run_output_dir / "optuna_trials.json"),
                "top_k_json": to_project_relative(run_output_dir / "optuna_top_k_configs.json"),
                "best_json": to_project_relative(run_output_dir / "optuna_best_trial.json"),
                "summary_json": to_project_relative(run_output_dir / "optuna_summary.json"),
                "top_k_val_test_metrics_plot_png": top_k_metrics_plot_path,
                "all_trials_val_test_metrics_plot_png": all_trials_metrics_plot_path,
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
