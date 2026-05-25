from __future__ import annotations

import argparse
import logging
import sys

from pathlib import Path

import pandas as pd

from src.adapters.parquet_phase_b_tier_sidecar_writer import (
    ParquetPhaseBTierSidecarWriter,
)
from src.use_cases.compute_phase_b_tier_metrics_use_case import (
    ComputePhaseBTierMetricsUseCase,
)
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Phase B E5 tier sidecars from existing analytics silver/gold. "
            "Does not refresh gold or interpret scientific conclusions."
        )
    )
    parser.add_argument("--asset", required=True, help="Asset symbol. Example: AAPL")
    parser.add_argument(
        "--parent-sweep-id",
        required=True,
        help="Confirmatory cohort id. Example: phase_b_confirmatorio_20260524",
    )
    parser.add_argument(
        "--silver-dir",
        default="data/analytics/silver",
        help="Analytics silver directory. Default: data/analytics/silver",
    )
    parser.add_argument(
        "--gold-dir",
        default="data/analytics/gold",
        help="Analytics gold directory. Default: data/analytics/gold",
    )
    parser.add_argument(
        "--output-dir",
        default="data/analytics/reports/phase_b",
        help="Base output directory for Phase B sidecars.",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging(logging.INFO)
    args = parse_args()
    asset = str(args.asset).strip().upper()
    parent_sweep_id = str(args.parent_sweep_id).strip()
    writer = ParquetPhaseBTierSidecarWriter(Path(args.output_dir))
    use_case = ComputePhaseBTierMetricsUseCase(
        silver_dir=Path(args.silver_dir),
        gold_dir=Path(args.gold_dir),
        sidecar_writer=writer,
    )
    result = use_case.execute(asset=asset, parent_sweep_id=parent_sweep_id)

    row_counts: dict[str, int] = {}
    for name, path in result.sidecar_paths.items():
        row_counts[name] = int(len(pd.read_parquet(path)))
    tier_path = result.sidecar_paths["tier_verdict"]
    tier_counts = (
        pd.read_parquet(tier_path)["tier"].astype(str).value_counts().sort_index().to_dict()
    )
    logger.info(
        "Phase B tier metrics completed",
        extra={
            "asset": asset,
            "parent_sweep_id": parent_sweep_id,
            "row_counts": row_counts,
            "tier_counts": tier_counts,
            "sidecar_paths": {k: str(v) for k, v in result.sidecar_paths.items()},
        },
    )
    for name, path in result.sidecar_paths.items():
        print(f"{name}: {path} rows={row_counts[name]}")
    print(f"tier_counts: {tier_counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

