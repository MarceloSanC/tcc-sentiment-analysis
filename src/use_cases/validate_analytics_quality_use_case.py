"""Analytics quality validation use case.

Thin orchestrator over `src.domain.services.quality_checks.QualityCheckRegistry`
(ADR-0004). Loads silver/gold tables, applies scope filtering, builds a
snapshot, then iterates the registry. The 27 concrete checks live in
`src/domain/services/quality_checks/{cardinality,alignment,calibration,
contracts}.py`; see `build_default_registry()` for the canonical order.
"""
from __future__ import annotations

import json
import warnings

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.domain.services.quality_checks import (
    AnalyticsSnapshot,
    QualityCheckRegistry,
    build_default_registry,
)
from src.domain.services.quality_checks.base import scope_table
from src.domain.services.quantile_contract_analyzer import (
    QuantileBlockAThresholds,
    QuantileDegeneracyThresholds,
)
from src.domain.services.scope_spec import ScopeSpec, validate_scope_spec


@dataclass(frozen=True)
class AnalyticsQualityResult:
    passed: bool
    checks: list[dict[str, object]]


_SILVER_TABLES: tuple[str, ...] = (
    "dim_run",
    "fact_run_snapshot",
    "fact_config",
    "fact_split_metrics",
    "fact_epoch_metrics",
    "fact_oos_predictions",
    "fact_model_artifacts",
    "fact_inference_runs",
    "fact_inference_predictions",
    "fact_feature_contrib_local",
    "fact_failures",
    "bridge_run_features",
    "fact_split_timestamps_ref",
)

_GOLD_TABLES: tuple[str, ...] = (
    "gold_dm_pairwise_results",
    "gold_mcs_results",
    "gold_quality_statistics_report",
    "gold_prediction_metrics_by_run_split_horizon",
    "gold_prediction_metrics_by_config",
)


class ValidateAnalyticsQualityUseCase:
    def __init__(
        self,
        *,
        analytics_silver_dir: str | Path,
        analytics_gold_dir: str | Path | None = None,
        scope_spec: ScopeSpec | None = None,
        min_samples_train: int = 1,
        min_samples_val: int = 1,
        min_samples_test: int = 1,
        block_a_parent_sweep_prefixes: list[str] | None = None,
        block_a_splits: list[str] | None = None,
        block_a_horizons: list[int] | None = None,
        block_a_max_crossing_bruto_rate: float = 0.001,
        block_a_max_negative_interval_width_count: int = 0,
        block_a_max_crossing_post_guardrail_rate: float = 0.0,
        block_a_require_post_guardrail: bool = False,
        degeneracy_min_rows_for_gate: int = 1000,
        degeneracy_max_p10_eq_p90_rate: float = 0.05,
        registry: QualityCheckRegistry | None = None,
    ) -> None:
        self.analytics_silver_dir = Path(analytics_silver_dir)
        self.analytics_gold_dir = Path(analytics_gold_dir) if analytics_gold_dir is not None else None
        self.scope_spec = scope_spec
        self.min_samples_train = int(min_samples_train)
        self.min_samples_val = int(min_samples_val)
        self.min_samples_test = int(min_samples_test)

        self.block_a_parent_sweep_prefixes = [
            str(v).strip() for v in (block_a_parent_sweep_prefixes or []) if str(v).strip()
        ]
        self.block_a_splits = [str(v).strip() for v in (block_a_splits or []) if str(v).strip()]
        self.block_a_horizons = [int(v) for v in (block_a_horizons or [])]
        self.block_a_thresholds = QuantileBlockAThresholds(
            max_crossing_bruto_rate=float(block_a_max_crossing_bruto_rate),
            max_negative_interval_width_count=int(block_a_max_negative_interval_width_count),
            max_crossing_post_guardrail_rate=float(block_a_max_crossing_post_guardrail_rate),
            require_post_guardrail=bool(block_a_require_post_guardrail),
        )
        self.degeneracy_thresholds = QuantileDegeneracyThresholds(
            min_rows_for_gate=int(degeneracy_min_rows_for_gate),
            max_p10_eq_p90_rate=float(degeneracy_max_p10_eq_p90_rate),
        )

        self._registry: QualityCheckRegistry = registry or build_default_registry(
            min_samples_train=self.min_samples_train,
            min_samples_val=self.min_samples_val,
            min_samples_test=self.min_samples_test,
            block_a_thresholds=self.block_a_thresholds,
            degeneracy_thresholds=self.degeneracy_thresholds,
        )

    @staticmethod
    def _load_partitioned_table(base_dir: Path, table_name: str) -> pd.DataFrame:
        # Supports both partitioned tables (<base>/<table>/**/*.parquet)
        # and flat parquet files (<base>/<table>.parquet), used by gold outputs.
        flat_file = base_dir / f"{table_name}.parquet"
        if flat_file.exists():
            return pd.read_parquet(flat_file)

        table_dir = base_dir / table_name
        if not table_dir.exists():
            return pd.DataFrame()
        files = sorted(table_dir.rglob("*.parquet"))
        if not files:
            return pd.DataFrame()
        frames = [pd.read_parquet(fp) for fp in files]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _resolve_scope_spec(self) -> ScopeSpec:
        legacy_scope_used = bool(
            self.block_a_parent_sweep_prefixes or self.block_a_splits or self.block_a_horizons
        )
        if legacy_scope_used:
            warnings.warn(
                "block_a_parent_sweep_prefixes/block_a_splits/block_a_horizons are deprecated; "
                "use scope_spec=ScopeSpec(...) instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.scope_spec is not None:
            return validate_scope_spec(self.scope_spec)

        if legacy_scope_used:
            return validate_scope_spec(
                ScopeSpec.create(
                    scope_mode="cohort_decision",
                    parent_sweep_prefixes=self.block_a_parent_sweep_prefixes,
                    splits=self.block_a_splits,
                    horizons=self.block_a_horizons,
                )
            )

        return validate_scope_spec(ScopeSpec.create(scope_mode="global_health"))

    @staticmethod
    def _scope_detail(scope_spec: ScopeSpec) -> str:
        return (
            "scope_mode="
            + str(scope_spec.scope_mode)
            + ", scope_parent_sweep_prefixes="
            + str(list(scope_spec.parent_sweep_prefixes))
            + ", scope_splits="
            + str(list(scope_spec.splits))
            + ", scope_horizons="
            + str(list(scope_spec.horizons))
        )

    @staticmethod
    def _build_scoped_run_ids(
        *,
        dim_run: pd.DataFrame,
        fact_oos_predictions: pd.DataFrame,
        scope_spec: ScopeSpec,
    ) -> set[str]:
        if dim_run.empty or "run_id" not in dim_run.columns:
            return set()

        dim_scoped = scope_table(
            dim_run,
            scope_spec=ScopeSpec.create(
                scope_mode=scope_spec.scope_mode,
                parent_sweep_prefixes=scope_spec.parent_sweep_prefixes,
            ),
            run_ids=None,
        )
        run_ids = set(dim_scoped["run_id"].dropna().astype(str).tolist())
        if not run_ids:
            return set()

        if not (scope_spec.splits or scope_spec.horizons):
            return run_ids

        if fact_oos_predictions.empty or "run_id" not in fact_oos_predictions.columns:
            return set()

        oos_scoped = scope_table(
            fact_oos_predictions,
            scope_spec=ScopeSpec.create(
                scope_mode=scope_spec.scope_mode,
                splits=scope_spec.splits,
                horizons=scope_spec.horizons,
            ),
            run_ids=run_ids,
        )
        if oos_scoped.empty:
            return set()
        return set(oos_scoped["run_id"].dropna().astype(str).tolist())

    @staticmethod
    def _build_expected_horizons_by_run(fact_config: pd.DataFrame) -> dict[str, list[int]]:
        expected_by_run: dict[str, list[int]] = {}
        if fact_config.empty or "run_id" not in fact_config.columns:
            return expected_by_run
        cfg_cols = [
            c for c in ["run_id", "evaluation_horizons_json", "max_prediction_length"]
            if c in fact_config.columns
        ]
        cfg_df = fact_config[cfg_cols].dropna(subset=["run_id"]) if cfg_cols else pd.DataFrame()
        for _, cfg_row in cfg_df.iterrows():
            run_key = str(cfg_row["run_id"])
            parsed: list[int] = []
            raw_h = cfg_row.get("evaluation_horizons_json")
            if isinstance(raw_h, str) and raw_h.strip():
                try:
                    obj = json.loads(raw_h)
                    if isinstance(obj, list):
                        parsed = [
                            max(1, int(v)) for v in obj if isinstance(v, (int, float))
                        ]
                except json.JSONDecodeError:
                    parsed = []
            if not parsed:
                mpl = pd.to_numeric(cfg_row.get("max_prediction_length"), errors="coerce")
                parsed = [1] if pd.isna(mpl) or int(mpl) < 1 else [1]
            expected_by_run[run_key] = sorted(set(parsed))
        return expected_by_run

    def _load_snapshot(self) -> AnalyticsSnapshot:
        """Load silver + gold tables, apply scope filters, return a snapshot
        ready for the registry. Bit-identical to the monolith pre-Stage R-21:
        same scope filtering, same gold load pattern, same
        `expected_horizons_by_run` derivation.
        """
        silver_tables: dict[str, pd.DataFrame] = {
            name: self._load_partitioned_table(self.analytics_silver_dir, name)
            for name in _SILVER_TABLES
        }

        scope_spec = self._resolve_scope_spec()
        scoped_run_ids = self._build_scoped_run_ids(
            dim_run=silver_tables["dim_run"],
            fact_oos_predictions=silver_tables["fact_oos_predictions"],
            scope_spec=scope_spec,
        )
        run_id_scope_filter: set[str] | None = (
            scoped_run_ids if scope_spec.has_cohort_filters() else None
        )

        scoped_silver = {
            name: scope_table(df, scope_spec=scope_spec, run_ids=run_id_scope_filter)
            for name, df in silver_tables.items()
        }

        gold_tables: dict[str, pd.DataFrame] = {}
        if self.analytics_gold_dir is not None:
            gold_tables = {
                name: self._load_partitioned_table(self.analytics_gold_dir, name)
                for name in _GOLD_TABLES
            }

        expected_horizons = self._build_expected_horizons_by_run(
            scoped_silver["fact_config"]
        )

        return AnalyticsSnapshot(
            tables=scoped_silver,
            gold_tables=gold_tables,
            scope_spec=scope_spec,
            scoped_run_ids=run_id_scope_filter,
            expected_horizons_by_run=expected_horizons,
            analytics_gold_dir=self.analytics_gold_dir,
        )

    def execute(self) -> AnalyticsQualityResult:
        snapshot = self._load_snapshot()
        scope_detail = self._scope_detail(snapshot.scope_spec)  # type: ignore[arg-type]

        applicable = self._registry.applicable(snapshot.scope_spec)
        results = [check.run(snapshot) for check in applicable]

        checks: list[dict[str, object]] = []
        for result in results:
            checks.append(
                {
                    "check": result.name,
                    "passed": bool(result.passed),
                    "detail": f"{result.detail} | {scope_detail}",
                }
            )

        passed = all(bool(item["passed"]) for item in checks)
        return AnalyticsQualityResult(passed=passed, checks=checks)
