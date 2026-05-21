"""Analytics gold refresh use case.

Thin orchestrator over `src.domain.services.gold_builders.GoldBuildersRegistry`
(ADR-0005). Loads silver + dim tables once, applies scope filtering,
then iterates the registry. The 25 concrete gold builders live in
`src/domain/services/gold_builders/{ranking,descriptive,quantile,
pairwise,confidence}.py`; see `build_default_registry()` for the
canonical order.

Stage R-22 migration: the previous god-object (~2.58k LOC of inline
`_build_gold_*` static methods) was extracted into per-cluster
`GoldBuilder` classes. The orchestrator's role is now to load the
silver snapshot, instantiate `BuildContext`, enforce `requires` /
`requires_gold` at registry-iteration time, persist each builder's
output, and accumulate `ctx.gold_outputs` so Tier 2/3 builders can
consume upstream gold without re-reading parquets.
"""
from __future__ import annotations

import logging

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from src.domain.services.gold_builders import (
    BuildContext,
    GoldBuilder,
    GoldBuilderRequirementError,
    GoldBuilderSnapshot,
    GoldBuildersRegistry,
    build_default_registry,
)
from src.domain.services.gold_builders.quantile import (
    PredictionMetricsByRunSplitHorizonGoldBuilder,
)
from src.domain.services.quantile_contract_analyzer import (
    QuantileDegeneracyThresholds,
)
from src.domain.services.scope_spec import (
    ScopeSpec,
    filter_dataframe_by_scope,
    validate_scope_spec,
)
from src.utils.path_policy import to_project_relative

logger = logging.getLogger(__name__)


_SILVER_TABLES: tuple[str, ...] = (
    "dim_run",
    "fact_split_metrics",
    "fact_oos_predictions",
    "fact_config",
    "fact_model_artifacts",
    "fact_feature_contrib_local",
)


@dataclass(frozen=True)
class RefreshAnalyticsStoreResult:
    gold_dir: str
    outputs: dict[str, str]


class RefreshAnalyticsStoreUseCase:
    def __init__(
        self,
        *,
        analytics_silver_dir: str | Path,
        analytics_gold_dir: str | Path,
        scope_spec: ScopeSpec | None = None,
        primary_quantile_contract: Literal["raw", "post_guardrail"] = "post_guardrail",
        degeneracy_thresholds: QuantileDegeneracyThresholds | None = None,
        registry: GoldBuildersRegistry | None = None,
    ) -> None:
        if primary_quantile_contract not in {"raw", "post_guardrail"}:
            raise ValueError(
                "primary_quantile_contract must be one of: raw, post_guardrail"
            )
        self.analytics_silver_dir = Path(analytics_silver_dir)
        self.analytics_gold_dir = Path(analytics_gold_dir)
        self.analytics_gold_dir.mkdir(parents=True, exist_ok=True)
        self.scope_spec = validate_scope_spec(scope_spec) if scope_spec is not None else None
        self.primary_quantile_contract = primary_quantile_contract
        self.degeneracy_thresholds = (
            degeneracy_thresholds
            if degeneracy_thresholds is not None
            else QuantileDegeneracyThresholds()
        )
        self._registry: GoldBuildersRegistry = registry or build_default_registry(
            degeneracy_thresholds=self.degeneracy_thresholds,
        )

    # ------------------------------------------------------------------
    # Silver loading / scope filtering helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scope_loaded_table(
        df: pd.DataFrame,
        *,
        scope_spec: ScopeSpec | None,
        scoped_run_ids: set[str] | None,
    ) -> pd.DataFrame:
        if df.empty or scope_spec is None or not scope_spec.has_cohort_filters():
            return df.copy()

        out = df.copy()
        applied_scope = False

        if scoped_run_ids is not None and "run_id" in out.columns:
            out = out[out["run_id"].astype(str).isin(scoped_run_ids)].copy()
            applied_scope = True

        table_scope = ScopeSpec.create(
            scope_mode=scope_spec.scope_mode,
            parent_sweep_prefixes=scope_spec.parent_sweep_prefixes
            if "parent_sweep_id" in out.columns
            else None,
            splits=scope_spec.splits if "split" in out.columns else None,
            horizons=scope_spec.horizons if "horizon" in out.columns else None,
        )
        if table_scope.has_cohort_filters():
            out = filter_dataframe_by_scope(out, scope_spec=table_scope)
            applied_scope = True

        return out if applied_scope else df.copy()

    def _load_partitioned_table(
        self,
        base_dir: Path,
        table_name: str,
        *,
        scope_spec: ScopeSpec | None = None,
        scoped_run_ids: set[str] | None = None,
    ) -> pd.DataFrame:
        table_dir = base_dir / table_name
        if not table_dir.exists():
            return pd.DataFrame()
        files = sorted(table_dir.rglob("*.parquet"))
        if not files:
            return pd.DataFrame()
        frames = [pd.read_parquet(fp) for fp in files]
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        return self._scope_loaded_table(
            df,
            scope_spec=scope_spec,
            scoped_run_ids=scoped_run_ids,
        )

    @staticmethod
    def _build_scoped_run_ids(
        *,
        dim_run: pd.DataFrame,
        fact_oos_predictions: pd.DataFrame,
        scope_spec: ScopeSpec | None,
    ) -> set[str] | None:
        if scope_spec is None or not scope_spec.has_cohort_filters():
            return None
        if dim_run.empty or "run_id" not in dim_run.columns:
            return set()

        dim_scope = ScopeSpec.create(
            scope_mode=scope_spec.scope_mode,
            parent_sweep_prefixes=scope_spec.parent_sweep_prefixes,
        )
        dim_scoped = RefreshAnalyticsStoreUseCase._scope_loaded_table(
            dim_run,
            scope_spec=dim_scope if dim_scope.has_cohort_filters() else None,
            scoped_run_ids=None,
        )
        run_ids = set(dim_scoped["run_id"].dropna().astype(str).tolist())
        if not run_ids or not (scope_spec.splits or scope_spec.horizons):
            return run_ids

        if fact_oos_predictions.empty or "run_id" not in fact_oos_predictions.columns:
            return set()

        oos_scope = ScopeSpec.create(
            scope_mode=scope_spec.scope_mode,
            splits=scope_spec.splits,
            horizons=scope_spec.horizons,
        )
        oos_scoped = RefreshAnalyticsStoreUseCase._scope_loaded_table(
            fact_oos_predictions,
            scope_spec=oos_scope if oos_scope.has_cohort_filters() else None,
            scoped_run_ids=run_ids,
        )
        if oos_scoped.empty:
            return set()
        return set(oos_scoped["run_id"].dropna().astype(str).tolist())

    def _load_silver_dim_snapshot(
        self, effective_scope: ScopeSpec | None
    ) -> GoldBuilderSnapshot:
        """Load all silver/dim tables, apply scope filtering, return a
        single snapshot that gold builders can read from. Mirrors the
        legacy execute() preamble (lines 2379-2422 of the pre-R-22
        monolith) so the migrated builders see byte-identical inputs.
        """
        dim_run = self._load_partitioned_table(self.analytics_silver_dir, "dim_run")
        fact_oos_predictions_unscoped = self._load_partitioned_table(
            self.analytics_silver_dir, "fact_oos_predictions"
        )
        scoped_run_ids = self._build_scoped_run_ids(
            dim_run=dim_run,
            fact_oos_predictions=fact_oos_predictions_unscoped,
            scope_spec=effective_scope,
        )
        dim_run = self._scope_loaded_table(
            dim_run,
            scope_spec=effective_scope,
            scoped_run_ids=scoped_run_ids,
        )
        fact_oos_predictions = self._scope_loaded_table(
            fact_oos_predictions_unscoped,
            scope_spec=effective_scope,
            scoped_run_ids=scoped_run_ids,
        )
        tables: dict[str, pd.DataFrame] = {
            "dim_run": dim_run,
            "fact_oos_predictions": fact_oos_predictions,
        }
        for name in _SILVER_TABLES:
            if name in tables:
                continue
            tables[name] = self._load_partitioned_table(
                self.analytics_silver_dir,
                name,
                scope_spec=effective_scope,
                scoped_run_ids=scoped_run_ids,
            )
        return GoldBuilderSnapshot(tables=tables)

    # ------------------------------------------------------------------
    # Builder execution
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_write(df: pd.DataFrame, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return str(to_project_relative(path))

    @staticmethod
    def _check_requirements(
        builder: GoldBuilder, snapshot: GoldBuilderSnapshot, ctx: BuildContext
    ) -> None:
        missing_silver = [t for t in builder.requires if t not in snapshot.tables]
        if missing_silver:
            raise GoldBuilderRequirementError(
                f"{builder.output_table} requires {builder.requires}, "
                f"snapshot missing tables: {missing_silver}"
            )
        missing_gold = [t for t in builder.requires_gold if t not in ctx.gold_outputs]
        if missing_gold:
            raise GoldBuilderRequirementError(
                f"{builder.output_table} requires_gold {builder.requires_gold}, "
                f"missing: {missing_gold}. Likely a registration order bug."
            )

    def execute(self, scope_spec: ScopeSpec | None = None) -> RefreshAnalyticsStoreResult:
        # Reset per-refresh: aceite original do Stage 8.1 e "warning unico
        # por refresh", nao "por processo". Sem este reset, multiplos
        # refreshes consecutivos no mesmo processo silenciariam
        # diagnostico de silver legado permanentemente.
        PredictionMetricsByRunSplitHorizonGoldBuilder._POST_GUARDRAIL_MISSING_WARNING_EMITTED = False

        effective_scope = (
            validate_scope_spec(scope_spec) if scope_spec is not None else self.scope_spec
        )

        snapshot = self._load_silver_dim_snapshot(effective_scope)
        ctx = BuildContext(
            scope_spec=effective_scope,
            primary_quantile_contract=self.primary_quantile_contract,
        )

        logger.info(
            "Analytics primary quantile contract resolved",
            extra={"primary_quantile_contract": self.primary_quantile_contract},
        )

        outputs: dict[str, str] = {}
        for builder in self._registry.applicable(effective_scope):
            self._check_requirements(builder, snapshot, ctx)
            df = builder.build(snapshot, ctx)
            path = self.analytics_gold_dir / f"{builder.output_table}.parquet"
            outputs[builder.output_table] = self._safe_write(df, path)
            ctx.gold_outputs[builder.output_table] = df

        return RefreshAnalyticsStoreResult(
            gold_dir=str(to_project_relative(self.analytics_gold_dir)),
            outputs=outputs,
        )
