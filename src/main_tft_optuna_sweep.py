from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from src.adapters.cli_tft_train_runner import CLITFTTrainRunner
from src.infrastructure.schemas.model_artifact_schema import (
    TFT_SPLIT_DEFAULTS,
    TFT_TRAINING_DEFAULTS,
)
from src.use_cases.run_tft_optuna_search_use_case import RunTFTOptunaSearchUseCase
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import load_data_paths

logger = logging.getLogger(__name__)
DEFAULT_OPTUNA_CONFIG_PATH = Path("config/optuna/default_tft_optuna_sweep.json")


def _parse_csv_str(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_int(value: str | None) -> list[int]:
    return [int(item) for item in _parse_csv_str(value)]


def _load_json_config(path: str | None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_OPTUNA_CONFIG_PATH
    if not config_path.exists() or not config_path.is_file():
        raise ValueError(f"Config JSON not found: {config_path}")
    try:
        content = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file: {config_path}") from exc
    if not isinstance(content, dict):
        raise ValueError("Config JSON root must be an object")
    return content


def _default_optuna_config() -> dict[str, Any]:
    return {
        "features": None,
        "feature_sets": [],
        "continue_on_error": True,
        "output_subdir": "optuna_tft_search",
        "study_name": "tft_optuna",
        "n_trials": 30,
        "top_k": 5,
        "timeout_seconds": None,
        "sampler_seed": 42,
        "objective_metric": "robust_score",
        "objective_lambda": 1.0,
        "replica_seeds": [7, 42, 123],
        "walk_forward": {"enabled": False, "folds": []},
        "training_config": dict(TFT_TRAINING_DEFAULTS),
        "split_config": dict(TFT_SPLIT_DEFAULTS),
        "search_space": {
            "max_encoder_length": {"type": "int", "low": 2, "high": 120, "step": 1},
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-3, "log": True},
            "hidden_size": {"type": "categorical", "choices": [8, 16, 32, 64]},
            "dropout": {"type": "float", "low": 0.05, "high": 0.3, "step": 0.05},
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Optuna-based HPO for TFT and persist top-k candidate configs "
            "to run later in sweep analysis pipeline."
        )
    )
    parser.add_argument("--asset", required=True, help="Asset symbol. Example: AAPL")
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help=(
            "Optional path to Optuna config JSON. "
            "If omitted, uses config/optuna/default_tft_optuna_sweep.json."
        ),
    )
    parser.add_argument("--features", type=str, default=None, help="Feature token string.")
    parser.add_argument(
        "--feature-sets",
        type=str,
        default=None,
        help="Optional comma-separated feature-set entries to run independently.",
    )
    parser.add_argument("--n-trials", type=int, default=None, help="Override number of trials.")
    parser.add_argument("--top-k", type=int, default=None, help="Override number of top configs to save.")
    parser.add_argument("--timeout-seconds", type=int, default=None, help="Optional timeout for study.optimize.")
    parser.add_argument("--sampler-seed", type=int, default=None, help="Optional TPESampler seed.")
    parser.add_argument(
        "--replica-seeds",
        type=str,
        default=None,
        help="Comma-separated seed list for repeated runs, e.g. 7,42,123",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue when one run fails inside a trial.",
    )
    parser.add_argument("--output-subdir", type=str, default=None, help="Optional optuna output subdir.")
    parser.add_argument("--study-name", type=str, default=None, help="Optional study name.")
    parser.add_argument(
        "--objective-metric",
        type=str,
        default=None,
        choices=["robust_score", "mean_val_rmse", "mean_test_rmse", "joint_val_test_rmse"],
        help="Objective metric for Optuna minimization.",
    )
    parser.add_argument(
        "--objective-lambda",
        type=float,
        default=None,
        help="Lambda used when objective_metric=robust_score fallback formula.",
    )
    return parser.parse_args()


def _resolve_effective_config(args: argparse.Namespace) -> dict[str, Any]:
    effective = _default_optuna_config()
    file_config = _load_json_config(args.config_json)

    if isinstance(file_config.get("features"), str):
        effective["features"] = file_config["features"].strip() or None
    if isinstance(file_config.get("feature_sets"), list):
        effective["feature_sets"] = [str(v).strip() for v in file_config["feature_sets"] if str(v).strip()]
    if isinstance(file_config.get("continue_on_error"), bool):
        effective["continue_on_error"] = file_config["continue_on_error"]
    if isinstance(file_config.get("output_subdir"), str):
        effective["output_subdir"] = file_config["output_subdir"]
    if isinstance(file_config.get("study_name"), str):
        effective["study_name"] = file_config["study_name"]
    if isinstance(file_config.get("n_trials"), int):
        effective["n_trials"] = file_config["n_trials"]
    if isinstance(file_config.get("top_k"), int):
        effective["top_k"] = file_config["top_k"]
    if isinstance(file_config.get("timeout_seconds"), int):
        effective["timeout_seconds"] = file_config["timeout_seconds"]
    if isinstance(file_config.get("sampler_seed"), int):
        effective["sampler_seed"] = file_config["sampler_seed"]
    if isinstance(file_config.get("objective_metric"), str):
        effective["objective_metric"] = file_config["objective_metric"]
    if isinstance(file_config.get("objective_lambda"), (int, float)):
        effective["objective_lambda"] = float(file_config["objective_lambda"])
    if isinstance(file_config.get("replica_seeds"), list):
        effective["replica_seeds"] = [int(v) for v in file_config["replica_seeds"]]
    if isinstance(file_config.get("walk_forward"), dict):
        effective["walk_forward"] = dict(file_config["walk_forward"])
    if isinstance(file_config.get("training_config"), dict):
        merged_training = dict(effective["training_config"])
        merged_training.update(file_config["training_config"])
        effective["training_config"] = merged_training
    if isinstance(file_config.get("split_config"), dict):
        merged_split = dict(effective["split_config"])
        merged_split.update(file_config["split_config"])
        effective["split_config"] = merged_split
    if isinstance(file_config.get("search_space"), dict):
        effective["search_space"] = dict(file_config["search_space"])

    if args.features is not None:
        effective["features"] = args.features.strip() or None
    if args.feature_sets is not None:
        effective["feature_sets"] = _parse_csv_str(args.feature_sets)
    if args.continue_on_error:
        effective["continue_on_error"] = True
    if args.output_subdir is not None:
        effective["output_subdir"] = args.output_subdir
    if args.study_name is not None:
        effective["study_name"] = args.study_name
    if args.n_trials is not None:
        effective["n_trials"] = args.n_trials
    if args.top_k is not None:
        effective["top_k"] = args.top_k
    if args.timeout_seconds is not None:
        effective["timeout_seconds"] = args.timeout_seconds
    if args.sampler_seed is not None:
        effective["sampler_seed"] = args.sampler_seed
    if args.replica_seeds is not None:
        effective["replica_seeds"] = _parse_csv_int(args.replica_seeds)
    if args.objective_metric is not None:
        effective["objective_metric"] = args.objective_metric
    if args.objective_lambda is not None:
        effective["objective_lambda"] = float(args.objective_lambda)

    return effective


def _sanitize_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    slug = slug.strip("_")
    return slug or "features"


def main() -> None:
    setup_logging(logging.INFO)
    args = parse_args()
    asset = args.asset.strip().upper()
    effective_cfg = _resolve_effective_config(args)

    paths = load_data_paths()
    models_asset_dir = Path(paths["models"]) / asset

    use_case = RunTFTOptunaSearchUseCase(
        train_runner=CLITFTTrainRunner(),
        base_training_config=effective_cfg["training_config"],
        split_config=effective_cfg["split_config"],
        walk_forward_config=effective_cfg.get("walk_forward"),
        replica_seeds=effective_cfg["replica_seeds"],
        continue_on_error=effective_cfg["continue_on_error"],
        objective_metric=effective_cfg.get("objective_metric", "robust_score"),
        objective_lambda=float(effective_cfg.get("objective_lambda", 1.0)),
    )

    explicit_features = effective_cfg["features"]
    feature_sets = effective_cfg.get("feature_sets") or []
    if explicit_features:
        feature_runs = [explicit_features]
    elif feature_sets:
        feature_runs = feature_sets
    else:
        feature_runs = [None]

    base_output_subdir = str(effective_cfg.get("output_subdir") or "optuna_tft_search")
    base_study_name = str(effective_cfg.get("study_name") or "tft_optuna")

    for idx, feature_entry in enumerate(feature_runs, start=1):
        if len(feature_runs) > 1:
            label = _sanitize_slug(feature_entry or "default")
            run_output_subdir = f"{base_output_subdir}__{label}"
            run_study_name = f"{base_study_name}__{label}"
        else:
            run_output_subdir = base_output_subdir
            run_study_name = base_study_name

        result = use_case.execute(
            asset=asset,
            models_asset_dir=models_asset_dir,
            features=feature_entry,
            search_space=effective_cfg["search_space"],
            n_trials=int(effective_cfg["n_trials"]),
            top_k=int(effective_cfg["top_k"]),
            output_subdir=run_output_subdir,
            study_name=run_study_name,
            timeout_seconds=effective_cfg.get("timeout_seconds"),
            sampler_seed=effective_cfg.get("sampler_seed"),
        )
        logger.info(
            "Optuna HPO finished",
            extra={
                "asset": result.asset,
                "features": feature_entry or "(default)",
                "study_name": result.study_name,
                "output_dir": result.output_dir,
                "best_value": result.best_value,
                "best_params": result.best_params,
                "top_k": len(result.top_k_configs),
            },
        )


if __name__ == "__main__":
    main()
