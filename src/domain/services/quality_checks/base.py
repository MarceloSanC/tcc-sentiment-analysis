from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.domain.services.scope_spec import ScopeSpec


def scope_table(
    df: pd.DataFrame,
    *,
    scope_spec: ScopeSpec,
    run_ids: set[str] | None,
) -> pd.DataFrame:
    """Filter `df` to the rows matching `scope_spec` (and optional run_id
    allow-list). Mirrors the private `_scope_table` used by the legacy
    monolithic `ValidateAnalyticsQualityUseCase` so checks emit
    bit-identical scoped data.
    """
    if df.empty:
        return df.copy()
    out = df.copy()

    if run_ids is not None and "run_id" in out.columns:
        out = out[out["run_id"].astype(str).isin(run_ids)].copy()

    if scope_spec.parent_sweep_prefixes and "parent_sweep_id" in out.columns:
        prefixes = scope_spec.parent_sweep_prefixes
        out = out[
            out["parent_sweep_id"].astype(str).apply(
                lambda value: any(value.startswith(prefix) for prefix in prefixes)
            )
        ].copy()

    if scope_spec.splits and "split" in out.columns:
        out = out[out["split"].astype(str).isin(set(scope_spec.splits))].copy()

    if scope_spec.horizons and "horizon" in out.columns:
        horizon = pd.to_numeric(out["horizon"], errors="coerce")
        out = out[horizon.isin(set(scope_spec.horizons))].copy()

    return out


@dataclass(frozen=True)
class CheckResult:
    """Outcome of running a single quality check."""

    name: str
    passed: bool
    detail: str


@dataclass
class AnalyticsSnapshot:
    """Snapshot of analytics tables loaded for quality validation.

    Holds DataFrames keyed by table name (silver in `tables`, gold in
    `gold_tables`) plus the resolved scope and precomputed helpers shared
    across checks. The implicit schema each check used to navigate ad-hoc
    in the monolithic use case now lives here.

    Renamed to `QualityCheckSnapshot` in Stage R-22.0.bis (after R-21 ships
    its bit-identical regression) to disambiguate from the `AnalyticsSnapshot`
    in `src/domain/services/gold_builders/base.py`.
    """

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    gold_tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    scope_spec: ScopeSpec | None = None
    scoped_run_ids: set[str] | None = None
    expected_horizons_by_run: dict[str, list[int]] = field(default_factory=dict)
    analytics_gold_dir: Path | None = None

    def has(self, table: str) -> bool:
        df = self.tables.get(table)
        return df is not None and not df.empty

    def get(self, table: str) -> pd.DataFrame:
        df = self.tables.get(table)
        if df is None:
            return pd.DataFrame()
        return df

    def has_gold(self, table: str) -> bool:
        df = self.gold_tables.get(table)
        return df is not None and not df.empty

    def get_gold(self, table: str) -> pd.DataFrame:
        df = self.gold_tables.get(table)
        if df is None:
            return pd.DataFrame()
        return df

    def has_gold_dir(self) -> bool:
        """Whether the use case was configured with an `analytics_gold_dir`.

        Distinguishes "gold validation disabled" (no dir) from "gold dir
        configured but tables empty" (which still emits substantive
        failures rather than the `skipped(no_gold_dir)` detail).
        """
        return self.analytics_gold_dir is not None

    def scope_gold(self, table: str, *, with_run_id_filter: bool) -> pd.DataFrame:
        """Return `table` from `gold_tables` scoped via `scope_spec`.

        `with_run_id_filter=True` applies `scoped_run_ids` (matches the
        monolith for `gold_prediction_metrics_by_run_split_horizon` inside
        `gold_confidence_calibrated_by_horizon`). The same gold table is
        loaded WITHOUT a run_id filter for the n_oos / DM/MCS checks; pass
        `with_run_id_filter=False` there.
        """
        if self.scope_spec is None:
            return self.get_gold(table)
        run_ids = self.scoped_run_ids if with_run_id_filter else None
        return scope_table(self.get_gold(table), scope_spec=self.scope_spec, run_ids=run_ids)


class QualityCheck(ABC):
    """A single named quality check.

    Subclasses set `name` (class attribute) and implement `applies_when`
    (filter by scope) and `run` (compute (passed, detail)).

    All existing checks (Stage R-21 migration) return `True` from
    `applies_when` and emit a `skipped(...)` detail internally when not
    applicable to the active scope — preserves bit-identical output with
    the legacy monolithic use case. Future checks may genuinely want
    `applies_when=False` to be omitted from the emitted list entirely.
    """

    name: str = ""

    def applies_when(self, scope: ScopeSpec | None) -> bool:  # noqa: ARG002
        """True if this check should execute given the current scope.

        Defaults to True for migrated checks; bit-identical contract per
        ADR-0004 §Consequences requires every monolith-emitted check to
        also appear in the registry output.
        """
        return True

    @abstractmethod
    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        """Execute the check; return a CheckResult."""


class QualityCheckRegistry:
    """Ordered registry of QualityCheck instances.

    Order of registration is preserved across `applicable(...)` calls so
    the resulting `checks` list keeps a stable shape (bit-identical
    contract per ADR-0004 §Consequences).
    """

    def __init__(self) -> None:
        self._checks: list[QualityCheck] = []

    def register(self, check: QualityCheck) -> None:
        self._checks.append(check)

    def applicable(self, scope: ScopeSpec | None) -> list[QualityCheck]:
        return [c for c in self._checks if c.applies_when(scope)]

    def __iter__(self) -> Iterator[QualityCheck]:
        return iter(self._checks)

    def __len__(self) -> int:
        return len(self._checks)
