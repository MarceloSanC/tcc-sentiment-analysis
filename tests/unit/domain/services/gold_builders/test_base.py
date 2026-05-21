from __future__ import annotations

import pandas as pd

from src.domain.services.gold_builders import (
    AnalyticsSnapshot,
    BuildContext,
    GoldBuilder,
    GoldBuildersRegistry,
)
from src.domain.services.scope_spec import ScopeSpec


class _FakeBuilder(GoldBuilder):
    output_table = "gold_fake"
    requires = ("dim_run",)

    def build(self, snapshot: AnalyticsSnapshot, ctx: BuildContext) -> pd.DataFrame:
        return pd.DataFrame({"x": [1]})


class _ScopedBuilder(GoldBuilder):
    output_table = "gold_scoped"
    requires = ("fact_oos_predictions",)

    def applies_when(self, scope: ScopeSpec | None) -> bool:
        return scope is not None and scope.scope_mode == "cohort_decision"

    def build(self, snapshot: AnalyticsSnapshot, ctx: BuildContext) -> pd.DataFrame:
        return pd.DataFrame({"y": [2]})


def test_registry_preserves_registration_order() -> None:
    reg = GoldBuildersRegistry()
    a, b = _FakeBuilder(), _ScopedBuilder()
    reg.register(a)
    reg.register(b)
    assert list(reg) == [a, b]
    assert len(reg) == 2


def test_registry_applicable_filters_by_scope() -> None:
    reg = GoldBuildersRegistry()
    reg.register(_FakeBuilder())
    reg.register(_ScopedBuilder())
    cohort = ScopeSpec(scope_mode="cohort_decision")
    global_ = ScopeSpec(scope_mode="global_health")
    assert [b.output_table for b in reg.applicable(cohort)] == ["gold_fake", "gold_scoped"]
    assert [b.output_table for b in reg.applicable(global_)] == ["gold_fake"]


def test_snapshot_has_and_get() -> None:
    snap = AnalyticsSnapshot(tables={"dim_run": pd.DataFrame({"r": [1]})})
    assert snap.has("dim_run") is True
    assert snap.has("absent") is False
    assert len(snap.get("dim_run")) == 1
    assert len(snap.get("absent")) == 0


def test_build_context_defaults() -> None:
    ctx = BuildContext()
    assert ctx.primary_quantile_contract == "post_guardrail"
    assert ctx.scope_spec is None
    assert ctx.extras == {}


def test_builder_default_applies_when_is_true() -> None:
    fake = _FakeBuilder()
    assert fake.applies_when(None) is True
    assert fake.applies_when(ScopeSpec(scope_mode="global_health")) is True
