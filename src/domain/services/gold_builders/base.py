from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from src.domain.services.scope_spec import ScopeSpec

PrimaryQuantileContract = Literal["raw", "post_guardrail"]


@dataclass
class BuildContext:
    """Runtime context shared across all gold builders for a refresh.

    Holds resolved scope, primary quantile contract, and any runtime
    parameters that builders need beyond the silver/dim snapshot.

    `gold_outputs` accumulates emitted gold DataFrames in topological
    order so Tier 2/3 builders can read upstream gold via
    `ctx.gold_outputs["gold_X"]` instead of re-reading parquet files
    (per R-22.2.bis Opcao A+B hibrida).
    """

    scope_spec: ScopeSpec | None = None
    primary_quantile_contract: PrimaryQuantileContract = "post_guardrail"
    gold_outputs: dict[str, pd.DataFrame] = field(default_factory=dict)
    extras: dict[str, object] = field(default_factory=dict)


@dataclass
class GoldBuilderSnapshot:
    """Snapshot of silver + dim tables loaded for gold refresh.

    Builders should never re-read parquets; everything comes from this
    snapshot. Empty DataFrames are returned when a table is absent so
    builders can check `snapshot.has(...)` cheaply.

    Renamed from `AnalyticsSnapshot` in Stage R-22.0.bis to disambiguate
    from the `QualityCheckSnapshot`/`AnalyticsSnapshot` in
    `src/domain/services/quality_checks/base.py` (different shape, same
    name was confusing). `AnalyticsSnapshot` alias preserved for the
    test suite already imported under the old name.
    """

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def has(self, table: str) -> bool:
        df = self.tables.get(table)
        return df is not None and not df.empty

    def get(self, table: str) -> pd.DataFrame:
        df = self.tables.get(table)
        if df is None:
            return pd.DataFrame()
        return df


AnalyticsSnapshot = GoldBuilderSnapshot


class GoldBuilder(ABC):
    """A single gold table builder.

    Subclasses set `output_table` (the parquet name emitted), `requires`
    (silver/dim tables consumed) and optionally `requires_gold` (other
    gold tables consumed via `ctx.gold_outputs`). They implement `build`
    returning the gold DataFrame.

    `requires_gold` is enforced by the orchestrator the same way
    `requires` is: any missing dependency raises
    `GoldBuilderRequirementError` so dependency-ordering bugs surface
    immediately rather than producing silently-empty Tier 2/3 outputs
    (per R-22.2.bis).
    """

    output_table: str = ""
    requires: tuple[str, ...] = ()
    requires_gold: tuple[str, ...] = ()

    def applies_when(self, scope: ScopeSpec | None) -> bool:
        """Whether this builder runs under the current scope. Default: True."""
        return True

    @abstractmethod
    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        """Compute the gold DataFrame for this builder's output_table."""


class GoldBuildersRegistry:
    """Ordered registry of GoldBuilder instances.

    Order of registration is preserved across `applicable(...)` so the
    orchestrator emits parquets in a deterministic sequence (helpful for
    byte-identical regression diffing per ADR-0005 §Consequences) AND so
    Tier 2/3 builders see their `requires_gold` upstream outputs already
    in `ctx.gold_outputs`. A topology test guards the order at
    `tests/unit/domain/services/gold_builders/test_topology.py`.
    """

    def __init__(self) -> None:
        self._builders: list[GoldBuilder] = []

    def register(self, builder: GoldBuilder) -> None:
        self._builders.append(builder)

    def applicable(self, scope: ScopeSpec | None) -> list[GoldBuilder]:
        return [b for b in self._builders if b.applies_when(scope)]

    def __iter__(self) -> Iterator[GoldBuilder]:
        return iter(self._builders)

    def __len__(self) -> int:
        return len(self._builders)


class GoldBuilderRequirementError(RuntimeError):
    """Raised by the orchestrator when a builder's `requires` (silver/dim)
    or `requires_gold` is not satisfied by the current snapshot/ctx.

    Fail-fast: indicates either a bug in registration order (Tier 2/3
    declared before its Tier 1 upstream) or a snapshot missing a silver
    table the builder needs. Either way the right answer is "stop
    emitting parquets and surface the bug" rather than producing a
    silently-empty gold output.
    """


def normalize_parent_sweep_id_for_merge(
    df: pd.DataFrame, column: str = "parent_sweep_id"
) -> pd.DataFrame:
    """Canonicalize `parent_sweep_id` for safe merges across builders.

    Mirrors the monolith's `_normalize_parent_sweep_id_for_merge`
    helper: strips whitespace, drops `nan`/`none`/`null`/`<na>` sentinels
    to `None`, and removes a trailing `.0` produced when pandas
    upgrades an integer-like column to float during a merge. Multiple
    builders rely on this normalization so duplicates do not creep in.
    """
    if df.empty or column not in df.columns:
        return df

    out = df.copy()

    def _norm(value: object) -> object:
        if pd.isna(value):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "<na>"}:
            return None
        if text.endswith(".0") and text[:-2].isdigit():
            return text[:-2]
        return text

    out[column] = out[column].map(_norm).astype("object")
    return out
