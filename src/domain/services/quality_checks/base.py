from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field

import pandas as pd

from src.domain.services.scope_spec import ScopeSpec


@dataclass(frozen=True)
class CheckResult:
    """Outcome of running a single quality check."""

    name: str
    passed: bool
    detail: str


@dataclass
class AnalyticsSnapshot:
    """Snapshot of analytics tables loaded for quality validation.

    Holds DataFrames keyed by table name plus the resolved scope. The
    implicit schema that each check used to navigate ad-hoc lives here.
    """

    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    scope_spec: ScopeSpec | None = None
    scoped_run_ids: set[str] | None = None

    def has(self, table: str) -> bool:
        df = self.tables.get(table)
        return df is not None and not df.empty

    def get(self, table: str) -> pd.DataFrame:
        df = self.tables.get(table)
        if df is None:
            return pd.DataFrame()
        return df


class QualityCheck(ABC):
    """A single named quality check.

    Subclasses set `name` (class attribute) and implement `applies_when`
    (filter by scope) and `run` (compute (passed, detail)).
    """

    name: str = ""

    @abstractmethod
    def applies_when(self, scope: ScopeSpec | None) -> bool:
        """True if this check should execute given the current scope."""

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
