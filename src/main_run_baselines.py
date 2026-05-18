"""Standalone CLI for invoking the Stage 12 baseline runner (F.0.1).

This entrypoint is the **debug-mode** path; the canonical path for smoke and
Phase B is `src.main_baselines_test_pipeline` (F.0.2), which reads the same
sweep JSON used by `src.main_tft_test_pipeline` and aligns warmup
automatically. Use this debug CLI when you need ad-hoc invocation outside
the sweep JSON contract — e.g. one-off reproductions or comparing baselines
against a candidate already persisted under a known `parent_sweep_id`.
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
from src.use_cases.run_baselines_use_case import (
    BASELINE_SPECS,
    RunBaselinesUseCase,
)
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import load_data_paths

logger = logging.getLogger(__name__)


def _parse_csv_baselines(value: str) -> list[str]:
    items = [v.strip() for v in str(value).split(",") if v.strip()]
    if not items:
        raise argparse.ArgumentTypeError(
            "--baselines must contain at least one comma-separated name"
        )
    unknown = sorted(set(items) - set(BASELINE_SPECS.keys()))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unsupported baselines: {unknown}. "
            f"Stage 12 MVP supports: {sorted(BASELINE_SPECS.keys())}"
        )
    return items


def _parse_horizons(value: str) -> list[int]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "--evaluation-horizons must be a JSON list (e.g. '[1, 7]')"
        ) from exc
    if not isinstance(parsed, list) or not parsed:
        raise argparse.ArgumentTypeError(
            "--evaluation-horizons must be a non-empty JSON list"
        )
    out: list[int] = []
    for h in parsed:
        if not isinstance(h, (int, float)):
            raise argparse.ArgumentTypeError("horizon values must be numeric")
        hv = int(h)
        if hv < 1:
            raise argparse.ArgumentTypeError("horizon values must be >= 1")
        out.append(hv)
    return out


def _parse_baseline_windows(value: str | None) -> dict[str, int] | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "--baseline-windows must be a JSON object "
            "(e.g. '{\"historical_mean_rolling\": 30}')"
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("--baseline-windows must be a JSON object")
    out: dict[str, int] = {}
    for k, v in parsed.items():
        if not isinstance(v, (int, float)):
            raise argparse.ArgumentTypeError(
                f"--baseline-windows[{k}] must be numeric"
            )
        out[str(k)] = int(v)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Debug-mode CLI for Stage 12 baseline runner. "
            "For smoke and Phase B, use main_baselines_test_pipeline."
        )
    )
    parser.add_argument("--asset", required=True, help="Asset symbol. Example: AAPL")
    parser.add_argument(
        "--parent-sweep-id",
        required=True,
        help="parent_sweep_id shared with the TFT candidate runs (Stage 12 contract).",
    )
    parser.add_argument(
        "--baselines",
        type=_parse_csv_baselines,
        default=list(BASELINE_SPECS.keys()),
        help=(
            "Comma-separated baseline names. "
            f"Supported: {','.join(sorted(BASELINE_SPECS.keys()))}. "
            "Default: all supported."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help=(
            "Path to the asset's dataset_tft parquet. "
            "Default: data/processed/dataset_tft/<ASSET>/dataset_tft_<ASSET>.parquet"
        ),
    )
    parser.add_argument(
        "--evaluation-horizons",
        type=_parse_horizons,
        default=[1, 7],
        help="Horizons to evaluate as JSON list. Default: [1, 7].",
    )
    parser.add_argument(
        "--train-start", required=True, help="Train start (YYYY-MM-DD or YYYYMMDD)."
    )
    parser.add_argument(
        "--train-end", required=True, help="Train end (YYYY-MM-DD or YYYYMMDD)."
    )
    parser.add_argument(
        "--val-start", required=True, help="Val start (YYYY-MM-DD or YYYYMMDD)."
    )
    parser.add_argument(
        "--val-end", required=True, help="Val end (YYYY-MM-DD or YYYYMMDD)."
    )
    parser.add_argument(
        "--test-start", required=True, help="Test start (YYYY-MM-DD or YYYYMMDD)."
    )
    parser.add_argument(
        "--test-end", required=True, help="Test end (YYYY-MM-DD or YYYYMMDD)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed registered in dim_run for traceability. Default: 42.",
    )
    parser.add_argument(
        "--baseline-windows",
        type=str,
        default=None,
        help=(
            "Optional JSON object overriding default rolling windows. "
            'Example: \'{"historical_mean_rolling": 30, "historical_quantiles_rolling": 252}\'.'
        ),
    )
    parser.add_argument(
        "--overwrite-on-collision",
        action="store_true",
        help="Overwrite silver rows if run_id collides (Lei 2 explicit flag).",
    )
    return parser.parse_args()


def _resolve_dataset_path(asset: str, override: str | None) -> Path:
    if override:
        return Path(override)
    return Path("data/processed/dataset_tft") / asset / f"dataset_tft_{asset}.parquet"


def main() -> int:
    setup_logging(logging.INFO)
    args = parse_args()
    asset = args.asset.strip().upper()

    dataset_path = _resolve_dataset_path(asset, args.dataset_path)
    if not dataset_path.exists():
        logger.error("dataset not found", extra={"path": str(dataset_path)})
        return 2

    paths = load_data_paths()
    repository = ParquetAnalyticsRunRepository(output_dir=paths["analytics_silver"])
    use_case = RunBaselinesUseCase(analytics_run_repository=repository)

    baseline_windows = _parse_baseline_windows(args.baseline_windows)

    split_definitions: dict[str, tuple[Any, Any]] = {
        "train": (args.train_start, args.train_end),
        "val": (args.val_start, args.val_end),
        "test": (args.test_start, args.test_end),
    }

    result = use_case.execute(
        asset=asset,
        dataset_path=dataset_path,
        parent_sweep_id=str(args.parent_sweep_id),
        split_definitions=split_definitions,
        horizons=list(args.evaluation_horizons),
        baselines=list(args.baselines),
        seed=int(args.seed),
        overwrite_on_collision=bool(args.overwrite_on_collision),
        baseline_windows=baseline_windows,
    )
    logger.info(
        "Baselines persisted",
        extra={
            "asset": result.asset,
            "parent_sweep_id": result.parent_sweep_id,
            "baselines_persisted": result.baselines_persisted,
            "n_rows_per_baseline": result.n_rows_per_baseline,
            "run_ids": result.run_ids,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
