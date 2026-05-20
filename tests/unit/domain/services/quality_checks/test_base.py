from __future__ import annotations

import pandas as pd
import pytest

from src.domain.services.quality_checks import (
    AnalyticsSnapshot,
    CheckResult,
    QualityCheck,
    QualityCheckRegistry,
)
from src.domain.services.scope_spec import ScopeSpec


class _AlwaysPassCheck(QualityCheck):
    name = "always_pass"

    def applies_when(self, scope: ScopeSpec | None) -> bool:
        return True

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        return CheckResult(name=self.name, passed=True, detail="ok")


class _ConditionalCheck(QualityCheck):
    name = "needs_dim_run"

    def applies_when(self, scope: ScopeSpec | None) -> bool:
        return True

    def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
        if not snapshot.has("dim_run"):
            return CheckResult(name=self.name, passed=False, detail="missing dim_run")
        return CheckResult(name=self.name, passed=True, detail="ok")


def test_registry_preserves_registration_order() -> None:
    reg = QualityCheckRegistry()
    a, b = _AlwaysPassCheck(), _ConditionalCheck()
    reg.register(a)
    reg.register(b)
    assert list(reg) == [a, b]
    assert len(reg) == 2


def test_registry_applicable_filters_by_applies_when() -> None:
    class _OnlyCohortCheck(QualityCheck):
        name = "cohort_only"

        def applies_when(self, scope: ScopeSpec | None) -> bool:
            return scope is not None and scope.scope_mode == "cohort_decision"

        def run(self, snapshot: AnalyticsSnapshot) -> CheckResult:
            return CheckResult(name=self.name, passed=True, detail="ok")

    reg = QualityCheckRegistry()
    reg.register(_AlwaysPassCheck())
    reg.register(_OnlyCohortCheck())

    cohort = ScopeSpec(scope_mode="cohort_decision")
    global_ = ScopeSpec(scope_mode="global_health")

    assert [c.name for c in reg.applicable(cohort)] == ["always_pass", "cohort_only"]
    assert [c.name for c in reg.applicable(global_)] == ["always_pass"]


def test_check_result_immutable() -> None:
    r = CheckResult(name="x", passed=True, detail="ok")
    with pytest.raises((AttributeError, Exception)):
        r.passed = False  # type: ignore[misc]


def test_snapshot_has_and_get() -> None:
    snap = AnalyticsSnapshot(
        tables={
            "dim_run": pd.DataFrame({"run_id": ["r1"]}),
            "empty_table": pd.DataFrame(),
        }
    )
    assert snap.has("dim_run") is True
    assert snap.has("empty_table") is False
    assert snap.has("absent_table") is False
    assert len(snap.get("dim_run")) == 1
    assert len(snap.get("absent_table")) == 0
