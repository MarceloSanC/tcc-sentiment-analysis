from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import re

from src.adapters.cli_tft_train_runner import CLITFTTrainRunner
from src.infrastructure.schemas.model_artifact_schema import (
    TFT_SPLIT_DEFAULTS,
    TFT_TRAINING_DEFAULTS,
)
from src.main_generate_sweep_plots import generate_for_sweep
from src.use_cases.run_tft_model_analysis_use_case import (
    DEFAULT_SWEEP_PARAM_RANGES,
    RunTFTModelAnalysisUseCase,
)
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import load_data_paths

logger = logging.getLogger(__name__)
DEFAULT_ANALYSIS_CONFIG_PATH = Path("config/model_analysis.default.json")


def _parse_csv_str(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_int(value: str | None) -> list[int]:
    return [int(item) for item in _parse_csv_str(value)]


def _load_json_config(path: str | None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_ANALYSIS_CONFIG_PATH
    if not config_path.exists() or not config_path.is_file():
        raise ValueError(f"Config JSON not found: {config_path}")
    try:
        content = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file: {config_path}") from exc
    if not isinstance(content, dict):
        raise ValueError("Config JSON root must be an object")
    return content


def _default_analysis_config() -> dict[str, Any]:
    return {
        "features": None,
        "feature_sets": [],
        "continue_on_error": False,
        "merge_tests": False,
        "max_runs": None,
        "output_subdir": None,
        "compute_confidence_interval": False,
        "generate_comparison_plots": True,
        "replica_seeds": [7, 42, 123],
        "walk_forward": {"enabled": False, "folds": []},
        "training_config": dict(TFT_TRAINING_DEFAULTS),
        "split_config": dict(TFT_SPLIT_DEFAULTS),
        "param_ranges": dict(DEFAULT_SWEEP_PARAM_RANGES),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one-at-a-time hyperparameter analysis for TFT and build comparison "
            "artifacts from metadata.json files."
        )
    )
    parser.add_argument("--asset", required=True, help="Asset symbol. Example: AAPL")
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help=(
            "Optional path to analysis config JSON. "
            "If omitted, uses config/model_analysis.default.json."
        ),
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help=(
            "Feature tokens/columns forwarded to main_train_tft "
            "(comma-separated). If omitted, main_train_tft uses its default feature set."
        ),
    )
    parser.add_argument(
        "--feature-sets",
        type=str,
        default=None,
        help=(
            "Optional comma-separated feature-set entries to run independently. "
            "Example: BASELINE_FEATURES,BASELINE_FEATURES+TECHNICAL_FEATURES"
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue analysis when one run fails.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap for number of runs (useful for quick testing).",
    )
    parser.add_argument(
        "--merge-tests",
        action="store_true",
        help=(
            "Merge new runs into existing sweep_runs.csv when using the same --output-subdir, "
            "then regenerate summaries/plots from the unified records."
        ),
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="Optional custom subdirectory name under data/models/{ASSET}/sweeps.",
    )
    parser.add_argument(
        "--replica-seeds",
        type=str,
        default=None,
        help="Comma-separated seed list for repeated runs, e.g. 7,42,123",
    )
    parser.add_argument(
        "--compute-confidence-interval",
        action="store_true",
        help="Compute 95%% CI columns in config_ranking.csv.",
    )
    parser.add_argument(
        "--only-params",
        type=str,
        default=None,
        help="Optional comma-separated list of params to keep in param_ranges.",
    )
    return parser.parse_args()


def _resolve_effective_config(args: argparse.Namespace) -> dict[str, Any]:
    effective = _default_analysis_config()
    file_config = _load_json_config(args.config_json)

    if isinstance(file_config.get("features"), str):
        effective["features"] = file_config["features"].strip() or None
    if isinstance(file_config.get("feature_sets"), list):
        effective["feature_sets"] = [str(v).strip() for v in file_config["feature_sets"] if str(v).strip()]
    if isinstance(file_config.get("continue_on_error"), bool):
        effective["continue_on_error"] = file_config["continue_on_error"]
    if isinstance(file_config.get("merge_tests"), bool):
        effective["merge_tests"] = file_config["merge_tests"]
    if isinstance(file_config.get("max_runs"), int):
        effective["max_runs"] = file_config["max_runs"]
    if isinstance(file_config.get("output_subdir"), str):
        effective["output_subdir"] = file_config["output_subdir"]
    if isinstance(file_config.get("compute_confidence_interval"), bool):
        effective["compute_confidence_interval"] = file_config["compute_confidence_interval"]
    if isinstance(file_config.get("generate_comparison_plots"), bool):
        effective["generate_comparison_plots"] = file_config["generate_comparison_plots"]
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
    if isinstance(file_config.get("param_ranges"), dict):
        sanitized: dict[str, list[Any]] = {}
        for key, values in file_config["param_ranges"].items():
            if isinstance(values, list):
                sanitized[str(key)] = list(values)
        if sanitized:
            effective["param_ranges"] = sanitized

    if args.features is not None:
        effective["features"] = args.features.strip() or None
    if args.feature_sets is not None:
        effective["feature_sets"] = _parse_csv_str(args.feature_sets)
    if args.continue_on_error:
        effective["continue_on_error"] = True
    if args.merge_tests:
        effective["merge_tests"] = True
    if args.max_runs is not None:
        effective["max_runs"] = args.max_runs
    if args.output_subdir is not None:
        effective["output_subdir"] = args.output_subdir
    if args.compute_confidence_interval:
        effective["compute_confidence_interval"] = True
    if args.replica_seeds is not None:
        effective["replica_seeds"] = _parse_csv_int(args.replica_seeds)

    only_params = _parse_csv_str(args.only_params)
    if only_params:
        effective["param_ranges"] = {
            k: v for k, v in effective["param_ranges"].items() if k in set(only_params)
        }
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

    use_case = RunTFTModelAnalysisUseCase(
        train_runner=CLITFTTrainRunner(),
        base_training_config=effective_cfg["training_config"],
        param_ranges=effective_cfg["param_ranges"],
        replica_seeds=effective_cfg["replica_seeds"],
        split_config=effective_cfg["split_config"],
        compute_confidence_interval=effective_cfg["compute_confidence_interval"],
        walk_forward_config=effective_cfg.get("walk_forward"),
        generate_comparison_plots=effective_cfg.get("generate_comparison_plots", True),
    )
    explicit_features = effective_cfg["features"]
    feature_sets = effective_cfg.get("feature_sets") or []
    if explicit_features:
        feature_runs = [explicit_features]
    elif feature_sets:
        feature_runs = feature_sets
    else:
        feature_runs = [None]

    base_output_subdir = effective_cfg.get("output_subdir")
    for idx, feature_entry in enumerate(feature_runs, start=1):
        run_cfg = dict(effective_cfg)
        run_cfg["features"] = feature_entry
        if len(feature_runs) > 1:
            label = _sanitize_slug(feature_entry or "default")
            if base_output_subdir:
                run_output_subdir = f"{base_output_subdir}__{label}"
            else:
                run_output_subdir = f"sweep_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{idx:02d}_{label}"
        else:
            run_output_subdir = base_output_subdir

        result = use_case.execute(
            asset=asset,
            models_asset_dir=models_asset_dir,
            features=feature_entry,
            continue_on_error=effective_cfg["continue_on_error"],
            merge_tests=effective_cfg["merge_tests"],
            max_runs=effective_cfg["max_runs"],
            output_subdir=run_output_subdir,
            analysis_config=run_cfg,
        )
        generated_artifacts = generate_for_sweep(Path(result.sweep_dir))
        logger.info(
            "Model analysis finished",
            extra={
                "asset": result.asset,
                "features": feature_entry or "(default)",
                "sweep_dir": result.sweep_dir,
                "runs_ok": result.runs_ok,
                "runs_failed": result.runs_failed,
                "top_5_runs": result.top_5_runs,
                "artifacts_generated": len(generated_artifacts),
            },
        )


if __name__ == "__main__":
    main()
