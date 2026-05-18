"""Canonical entrypoint for the baselines test pipeline (Stage F.0.2).

Sibling of `src.main_tft_test_pipeline`: reads the same sweep JSON config
and dispatches per-fold/per-seed/per-baseline runs via
`RunBaselinesTestPipelineUseCase`. Use this for smoke and Phase B; reserve
`src.main_run_baselines` (F.0.1) for ad-hoc debug.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from pathlib import Path
from typing import Any

from src.adapters.parquet_analytics_run_repository import (
    ParquetAnalyticsRunRepository,
)
from src.use_cases.run_baselines_test_pipeline_use_case import (
    RunBaselinesTestPipelineUseCase,
)
from src.use_cases.run_baselines_use_case import RunBaselinesUseCase
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import load_data_paths

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baselines test pipeline (sibling of main_tft_test_pipeline). "
            "Reads same JSON; dispatches per-fold/per-seed/per-baseline."
        )
    )
    parser.add_argument("--asset", required=True, help="Asset symbol. Example: AAPL")
    parser.add_argument(
        "--config-json",
        type=str,
        required=True,
        help="Path to the unified sweep JSON config (same file used by TFT pipeline).",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help=(
            "Optional override for the dataset_tft parquet path. Default: "
            "data/processed/dataset_tft/<ASSET>/dataset_tft_<ASSET>.parquet"
        ),
    )
    parser.add_argument(
        "--overwrite-on-collision",
        action="store_true",
        help="Pass-through to RunBaselinesUseCase (Lei 2 explicit flag).",
    )
    return parser.parse_args()


def _load_json_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists() or not config_path.is_file():
        raise ValueError(f"Config JSON not found: {config_path}")
    try:
        content = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file: {config_path}") from exc
    if not isinstance(content, dict):
        raise ValueError("Config JSON root must be an object")
    return content


def _resolve_dataset_path(asset: str, override: str | None) -> Path:
    if override:
        return Path(override)
    return Path("data/processed/dataset_tft") / asset / f"dataset_tft_{asset}.parquet"


def main() -> int:
    setup_logging(logging.INFO)
    args = parse_args()
    asset = args.asset.strip().upper()

    config = _load_json_config(args.config_json)
    if args.overwrite_on_collision:
        config["overwrite_on_collision"] = True

    dataset_path = _resolve_dataset_path(asset, args.dataset_path)
    if not dataset_path.exists():
        logger.error("dataset not found", extra={"path": str(dataset_path)})
        return 2

    paths = load_data_paths()
    repository = ParquetAnalyticsRunRepository(output_dir=paths["analytics_silver"])
    baselines_runner = RunBaselinesUseCase(analytics_run_repository=repository)
    use_case = RunBaselinesTestPipelineUseCase(baselines_runner=baselines_runner)

    result = use_case.execute(
        asset=asset,
        config=config,
        dataset_path=dataset_path,
    )
    if result.warnings:
        for w in result.warnings:
            logger.warning(w)

    logger.info(
        "Baselines test pipeline finished",
        extra={
            "asset": asset,
            "parent_sweep_id": result.parent_sweep_id,
            "folds_processed": result.folds_processed,
            "seeds_processed": result.seeds_processed,
            "baselines_persisted_total": result.baselines_persisted_total,
        },
    )
    if result.parent_sweep_id is None:
        # No-op (no baselines section); exit 0 with informational warning only.
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
