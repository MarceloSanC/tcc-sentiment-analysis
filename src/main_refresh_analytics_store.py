from __future__ import annotations

import argparse
import logging

from src.domain.services.scope_spec import ScopeSpec
from src.use_cases.generate_prediction_analysis_plots_use_case import (
    GeneratePredictionAnalysisPlotsUseCase,
)
from src.use_cases.refresh_analytics_store_use_case import RefreshAnalyticsStoreUseCase
from src.use_cases.validate_analytics_quality_use_case import (
    ValidateAnalyticsQualityUseCase,
)
from src.utils.logging_config import setup_logging
from src.utils.path_resolver import load_data_paths

logger = logging.getLogger(__name__)


def _parse_csv_values(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _parse_csv_int_values(value: str | None) -> list[int] | None:
    if not value:
        return None
    values: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh analytics gold tables from analytics silver tables.")
    parser.add_argument(
        "--fail-on-quality",
        action="store_true",
        help="Fail process with non-zero exit code when analytics quality checks fail.",
    )
    parser.add_argument("--min-samples-train", type=int, default=1)
    parser.add_argument("--min-samples-val", type=int, default=1)
    parser.add_argument("--min-samples-test", type=int, default=1)
    parser.add_argument(
        "--skip-prediction-plots",
        action="store_true",
        help="Skip generation of section 9.2/9.3 prediction analysis plots after gold refresh.",
    )
    parser.add_argument(
        "--primary-quantile-contract",
        choices=("raw", "post_guardrail"),
        default="post_guardrail",
        help="Primary quantile contract used for decision gold tables and prediction plots.",
    )
    parser.add_argument(
        "--plots-asset",
        type=str,
        default=None,
        help="Optional asset filter for plot generation (example: AAPL).",
    )
    parser.add_argument(
        "--plots-output-dir",
        type=str,
        default=None,
        help=(
            "Optional output directory for prediction analysis plots. "
            "Default: data/analytics/reports/prediction_analysis_plots[/asset=<ASSET>]"
        ),
    )
    parser.add_argument(
        "--plots-scope-csv",
        type=str,
        default=None,
        help=(
            "Optional CSV path to scope plot generation to a frozen candidate set "
            "(supports columns run_id and/or feature_set_name+config_signature and/or config_label)."
        ),
    )
    parser.add_argument(
        "--plots-scope-sweep-prefixes",
        type=str,
        default=None,
        help=(
            "Optional comma-separated sweep id prefixes used to scope plot generation "
            "(example: 0_2_2_)."
        ),
    )
    parser.add_argument(
        "--scope-mode",
        choices=("global_health", "cohort_decision"),
        default=None,
        help="Optional analytics refresh scope mode.",
    )
    parser.add_argument(
        "--scope-sweep-prefixes",
        type=str,
        default=None,
        help="Optional comma-separated parent_sweep_id prefixes used to scope the gold refresh.",
    )
    parser.add_argument(
        "--scope-splits",
        type=str,
        default=None,
        help="Optional comma-separated split filter used to scope the gold refresh (example: val,test).",
    )
    parser.add_argument(
        "--scope-horizons",
        type=str,
        default=None,
        help="Optional comma-separated horizon filter used to scope the gold refresh (example: 1,7,30).",
    )

    parser.add_argument(
        "--block-a-scope-sweep-prefixes",
        type=str,
        default=None,
        help=(
            "Optional comma-separated parent_sweep_id prefixes used to scope Quantile Block A "
            "acceptance checks (example: 0_2_3_)."
        ),
    )
    parser.add_argument(
        "--block-a-splits",
        type=str,
        default=None,
        help="Optional comma-separated split filter for Block A (example: val,test).",
    )
    parser.add_argument(
        "--block-a-horizons",
        type=str,
        default=None,
        help="Optional comma-separated horizon filter for Block A (example: 1,7,30).",
    )
    parser.add_argument(
        "--block-a-max-crossing-bruto-rate",
        type=float,
        default=0.001,
        help="Block A threshold for crossing bruto rate (default: 0.001 = 0.10%).",
    )
    parser.add_argument(
        "--block-a-max-negative-interval-width-count",
        type=int,
        default=0,
        help="Block A threshold for negative interval width count (default: 0).",
    )
    parser.add_argument(
        "--block-a-max-crossing-post-guardrail-rate",
        type=float,
        default=0.0,
        help="Block A threshold for crossing rate after guardrail (default: 0.0).",
    )
    parser.add_argument(
        "--block-a-require-post-guardrail",
        action="store_true",
        help="Require post-guardrail quantile columns for Block A acceptance.",
    )
    parser.add_argument(
        "--degeneracy-min-rows-for-gate",
        type=int,
        default=1000,
        help="Minimum group rows before the quantile degeneracy gate becomes blocking (default: 1000).",
    )
    parser.add_argument(
        "--degeneracy-max-p10-eq-p90-rate",
        type=float,
        default=0.05,
        help="Maximum allowed raw p10==p90 rate for quantile groups (default: 0.05 = 5%).",
    )
    args = parser.parse_args()
    scope_flags_used = bool(
        args.scope_mode
        or args.scope_sweep_prefixes
        or args.scope_splits
        or args.scope_horizons
    )
    if scope_flags_used and not args.scope_mode:
        parser.error("--scope-mode is required when using refresh scope filters.")
    return args


def main() -> None:
    setup_logging(logging.INFO)
    args = parse_args()
    paths = load_data_paths()
    primary_quantile_contract = getattr(args, "primary_quantile_contract", "post_guardrail")

    scope_spec = None
    if args.scope_mode or args.scope_sweep_prefixes or args.scope_splits or args.scope_horizons:
        scope_spec = ScopeSpec.create(
            scope_mode=args.scope_mode,
            parent_sweep_prefixes=_parse_csv_values(args.scope_sweep_prefixes),
            splits=_parse_csv_values(args.scope_splits),
            horizons=_parse_csv_int_values(args.scope_horizons),
        )

    use_case = RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=paths["analytics_silver"],
        analytics_gold_dir=paths["analytics_gold"],
        scope_spec=scope_spec,
        primary_quantile_contract=primary_quantile_contract,
    )
    logger.info(
        "Analytics gold refresh scope resolved",
        extra={
            "scope_spec": (
                ValidateAnalyticsQualityUseCase._scope_detail(use_case.scope_spec)
                if use_case.scope_spec is not None
                else None
            ),
        },
    )
    result = use_case.execute()
    logger.info(
        "Analytics gold refresh completed",
        extra={
            "gold_dir": result.gold_dir,
            "outputs": result.outputs,
            "primary_quantile_contract": primary_quantile_contract,
        },
    )

    block_a_scope_prefixes = None
    if args.block_a_scope_sweep_prefixes:
        block_a_scope_prefixes = [
            part.strip()
            for part in str(args.block_a_scope_sweep_prefixes).split(",")
            if part.strip()
        ]

    block_a_splits = None
    if args.block_a_splits:
        block_a_splits = [
            part.strip()
            for part in str(args.block_a_splits).split(",")
            if part.strip()
        ]

    block_a_horizons = None
    if args.block_a_horizons:
        block_a_horizons = []
        for part in str(args.block_a_horizons).split(","):
            part = part.strip()
            if not part:
                continue
            block_a_horizons.append(int(part))

    quality_result = ValidateAnalyticsQualityUseCase(
        analytics_silver_dir=paths["analytics_silver"],
        analytics_gold_dir=paths["analytics_gold"],
        min_samples_train=args.min_samples_train,
        min_samples_val=args.min_samples_val,
        min_samples_test=args.min_samples_test,
        block_a_parent_sweep_prefixes=block_a_scope_prefixes,
        block_a_splits=block_a_splits,
        block_a_horizons=block_a_horizons,
        block_a_max_crossing_bruto_rate=args.block_a_max_crossing_bruto_rate,
        block_a_max_negative_interval_width_count=args.block_a_max_negative_interval_width_count,
        block_a_max_crossing_post_guardrail_rate=args.block_a_max_crossing_post_guardrail_rate,
        block_a_require_post_guardrail=args.block_a_require_post_guardrail,
        degeneracy_min_rows_for_gate=getattr(args, "degeneracy_min_rows_for_gate", 1000),
        degeneracy_max_p10_eq_p90_rate=getattr(args, "degeneracy_max_p10_eq_p90_rate", 0.05),
    ).execute()
    failed_checks = [c for c in quality_result.checks if not bool(c["passed"])]
    logger.info(
        "Analytics quality validation completed",
        extra={
            "passed": quality_result.passed,
            "failed_checks": failed_checks,
            "total_checks": len(quality_result.checks),
            "block_a_scope_sweep_prefixes": args.block_a_scope_sweep_prefixes,
            "block_a_splits": args.block_a_splits,
            "block_a_horizons": args.block_a_horizons,
            "degeneracy_min_rows_for_gate": getattr(args, "degeneracy_min_rows_for_gate", 1000),
            "degeneracy_max_p10_eq_p90_rate": getattr(args, "degeneracy_max_p10_eq_p90_rate", 0.05),
        },
    )

    if not args.skip_prediction_plots:
        plots_output_dir = args.plots_output_dir
        if plots_output_dir is None:
            base = "data/analytics/reports/prediction_analysis_plots"
            plots_output_dir = (
                f"{base}/asset={args.plots_asset.upper()}"
                if args.plots_asset
                else base
            )
        scope_prefixes = None
        if args.plots_scope_sweep_prefixes:
            scope_prefixes = [
                part.strip()
                for part in str(args.plots_scope_sweep_prefixes).split(",")
                if part.strip()
            ]

        plots_result = GeneratePredictionAnalysisPlotsUseCase(
            analytics_silver_dir=paths["analytics_silver"],
            analytics_gold_dir=paths["analytics_gold"],
            output_dir=plots_output_dir,
            primary_quantile_contract=primary_quantile_contract,
        ).execute(
            asset=args.plots_asset,
            scope_csv_path=args.plots_scope_csv,
            scope_sweep_prefixes=scope_prefixes,
        )
        logger.info(
            "Prediction analysis plots generated",
            extra={
                "asset": args.plots_asset,
                "plots_scope_csv": args.plots_scope_csv,
                "plots_scope_sweep_prefixes": args.plots_scope_sweep_prefixes,
                "primary_quantile_contract": primary_quantile_contract,
                "output_dir": plots_result.output_dir,
                "outputs": plots_result.outputs,
            },
        )

    if args.fail_on_quality and not quality_result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
