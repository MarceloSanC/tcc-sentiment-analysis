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
    """

    scope_spec: ScopeSpec | None = None
    primary_quantile_contract: PrimaryQuantileContract = "post_guardrail"
    extras: dict[str, object] = field(default_factory=dict)


@dataclass
class AnalyticsSnapshot:
    """Snapshot of silver + dim tables loaded for gold refresh.

    Builders should never re-read parquets; everything comes from this
    snapshot. Empty DataFrames are returned when a table is absent so
    builders can check `snapshot.has(...)` cheaply.
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


class GoldBuilder(ABC):
    """A single gold table builder.

    Subclasses set `output_table` (the parquet name emitted) and
    `requires` (silver/dim tables consumed). They implement `build`
    returning the gold DataFrame.
    """

    output_table: str = ""
    requires: tuple[str, ...] = ()

    def applies_when(self, scope: ScopeSpec | None) -> bool:
        """Whether this builder runs under the current scope. Default: True."""
        return True

    @abstractmethod
    def build(self, snapshot: AnalyticsSnapshot, ctx: BuildContext) -> pd.DataFrame:
        """Compute the gold DataFrame for this builder's output_table."""


class GoldBuildersRegistry:
    """Ordered registry of GoldBuilder instances.

    Order of registration is preserved across `applicable(...)` so the
    orchestrator emits parquets in a deterministic sequence (helpful for
    byte-identical regression diffing per ADR-0005 §Consequences).
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
