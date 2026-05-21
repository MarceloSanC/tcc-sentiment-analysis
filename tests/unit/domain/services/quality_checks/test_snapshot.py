"""Tests for the QualityCheck snapshot helpers (Stage R-21.5)."""
from __future__ import annotations

import pandas as pd

from src.domain.services.quality_checks.base import (
    AnalyticsSnapshot,
    scope_table,
)
from src.domain.services.scope_spec import ScopeSpec


def test_snapshot_has_and_get_silver() -> None:
    snap = AnalyticsSnapshot(
        tables={
            "dim_run": pd.DataFrame({"run_id": ["r1", "r2"]}),
            "empty": pd.DataFrame(),
        }
    )
    assert snap.has("dim_run") is True
    assert snap.has("empty") is False
    assert snap.has("absent") is False
    assert len(snap.get("dim_run")) == 2
    assert len(snap.get("absent")) == 0


def test_snapshot_has_gold_and_get_gold() -> None:
    snap = AnalyticsSnapshot(
        gold_tables={
            "gold_a": pd.DataFrame({"run_id": ["r1"]}),
            "gold_empty": pd.DataFrame(),
        }
    )
    assert snap.has_gold("gold_a") is True
    assert snap.has_gold("gold_empty") is False
    assert snap.has_gold("absent") is False
    assert len(snap.get_gold("gold_a")) == 1


def test_has_gold_dir_true_when_configured() -> None:
    from pathlib import Path
    snap = AnalyticsSnapshot(analytics_gold_dir=Path("/tmp/gold"))
    assert snap.has_gold_dir() is True

    snap2 = AnalyticsSnapshot()
    assert snap2.has_gold_dir() is False


def test_scope_gold_filters_by_run_id_when_requested() -> None:
    df = pd.DataFrame(
        {
            "run_id": ["r1", "r2", "r3"],
            "parent_sweep_id": ["sw_a", "sw_b", "sw_a"],
            "split": ["test", "test", "test"],
            "horizon": [1, 1, 7],
            "value": [10, 20, 30],
        }
    )
    snap = AnalyticsSnapshot(
        gold_tables={"gold_x": df},
        scope_spec=ScopeSpec(scope_mode="cohort_decision"),
        scoped_run_ids={"r1", "r3"},
    )
    out_with = snap.scope_gold("gold_x", with_run_id_filter=True)
    assert set(out_with["run_id"].tolist()) == {"r1", "r3"}
    out_without = snap.scope_gold("gold_x", with_run_id_filter=False)
    assert set(out_without["run_id"].tolist()) == {"r1", "r2", "r3"}


def test_scope_gold_applies_parent_sweep_prefix() -> None:
    df = pd.DataFrame(
        {
            "run_id": ["r1", "r2"],
            "parent_sweep_id": ["phase_a_001", "phase_b_001"],
            "split": ["test", "test"],
            "horizon": [1, 1],
        }
    )
    snap = AnalyticsSnapshot(
        gold_tables={"gold_x": df},
        scope_spec=ScopeSpec(
            scope_mode="cohort_decision",
            parent_sweep_prefixes=("phase_a_",),
        ),
        scoped_run_ids=None,
    )
    out = snap.scope_gold("gold_x", with_run_id_filter=False)
    assert set(out["parent_sweep_id"].tolist()) == {"phase_a_001"}


def test_scope_table_no_op_when_no_filters_set() -> None:
    df = pd.DataFrame({"run_id": ["r1", "r2"], "value": [1, 2]})
    scope = ScopeSpec(scope_mode="global_health")
    out = scope_table(df, scope_spec=scope, run_ids=None)
    pd.testing.assert_frame_equal(out, df)


def test_scope_table_filters_by_split_and_horizon() -> None:
    df = pd.DataFrame(
        {
            "split": ["train", "val", "test"],
            "horizon": [1, 7, 30],
            "value": [10, 20, 30],
        }
    )
    scope = ScopeSpec(
        scope_mode="cohort_decision",
        splits=("test",),
        horizons=(1, 7),
    )
    out = scope_table(df, scope_spec=scope, run_ids=None)
    assert len(out) == 0  # split=test AND horizon in (1,7); row split=test has horizon=30


def test_scope_table_passes_run_id_set_through() -> None:
    df = pd.DataFrame({"run_id": ["r1", "r2", "r3"]})
    scope = ScopeSpec(scope_mode="global_health")
    out = scope_table(df, scope_spec=scope, run_ids={"r2"})
    assert out["run_id"].tolist() == ["r2"]
