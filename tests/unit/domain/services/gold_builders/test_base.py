from __future__ import annotations

import pandas as pd
import pytest

from src.domain.services.gold_builders import (
    AnalyticsSnapshot,
    BuildContext,
    GoldBuilder,
    GoldBuilderRequirementError,
    GoldBuilderSnapshot,
    GoldBuildersRegistry,
    normalize_parent_sweep_id_for_merge,
)
from src.domain.services.scope_spec import ScopeSpec


class _FakeBuilder(GoldBuilder):
    output_table = "gold_fake"
    requires = ("dim_run",)

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
        return pd.DataFrame({"x": [1]})


class _ScopedBuilder(GoldBuilder):
    output_table = "gold_scoped"
    requires = ("fact_oos_predictions",)

    def applies_when(self, scope: ScopeSpec | None) -> bool:
        return scope is not None and scope.scope_mode == "cohort_decision"

    def build(self, snapshot: GoldBuilderSnapshot, ctx: BuildContext) -> pd.DataFrame:
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
    snap = GoldBuilderSnapshot(tables={"dim_run": pd.DataFrame({"r": [1]})})
    assert snap.has("dim_run") is True
    assert snap.has("absent") is False
    assert len(snap.get("dim_run")) == 1
    assert len(snap.get("absent")) == 0


def test_analytics_snapshot_alias_round_trips() -> None:
    assert AnalyticsSnapshot is GoldBuilderSnapshot
    snap = AnalyticsSnapshot(tables={"dim_run": pd.DataFrame({"r": [1]})})
    assert isinstance(snap, GoldBuilderSnapshot)
    assert snap.has("dim_run") is True


def test_build_context_defaults() -> None:
    ctx = BuildContext()
    assert ctx.primary_quantile_contract == "post_guardrail"
    assert ctx.scope_spec is None
    assert ctx.gold_outputs == {}
    assert ctx.extras == {}


def test_build_context_gold_outputs_is_mutable() -> None:
    ctx = BuildContext()
    ctx.gold_outputs["gold_runs_long"] = pd.DataFrame({"run_id": ["a"]})
    assert "gold_runs_long" in ctx.gold_outputs
    assert len(ctx.gold_outputs["gold_runs_long"]) == 1


def test_builder_default_applies_when_is_true() -> None:
    fake = _FakeBuilder()
    assert fake.applies_when(None) is True
    assert fake.applies_when(ScopeSpec(scope_mode="global_health")) is True


def test_builder_default_requires_gold_is_empty() -> None:
    fake = _FakeBuilder()
    assert fake.requires_gold == ()


def test_gold_builder_requirement_error_is_runtime_error() -> None:
    err = GoldBuilderRequirementError("missing dim_run")
    assert isinstance(err, RuntimeError)
    with pytest.raises(GoldBuilderRequirementError, match="missing"):
        raise err


def test_normalize_parent_sweep_id_strips_trailing_dot_zero() -> None:
    df = pd.DataFrame({"parent_sweep_id": ["123.0", "ok", None]})
    out = normalize_parent_sweep_id_for_merge(df)
    result = out["parent_sweep_id"].tolist()
    assert len(result) == 3
    assert result[0] == "123"
    assert result[1] == "ok"
    assert pd.isna(result[2])


def test_normalize_parent_sweep_id_drops_sentinel_strings() -> None:
    df = pd.DataFrame({"parent_sweep_id": ["nan", "<NA>", "null", "abc"]})
    out = normalize_parent_sweep_id_for_merge(df)
    result = out["parent_sweep_id"].tolist()
    assert len(result) == 4
    assert pd.isna(result[0])
    assert pd.isna(result[1])
    assert pd.isna(result[2])
    assert result[3] == "abc"


def test_normalize_parent_sweep_id_noop_when_column_missing() -> None:
    df = pd.DataFrame({"other": [1, 2]})
    out = normalize_parent_sweep_id_for_merge(df)
    assert "parent_sweep_id" not in out.columns
    pd.testing.assert_frame_equal(out, df)
