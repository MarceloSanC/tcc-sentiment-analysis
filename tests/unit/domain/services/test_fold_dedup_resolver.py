from __future__ import annotations

import pandas as pd
import pytest

from src.domain.services.fold_dedup_resolver import (
    RunInFoldCandidate,
    select_operationally_latest_fold,
)


def _candidate(
    fold_name: str,
    train_end: str | pd.Timestamp,
    *,
    run_id: str | None = None,
) -> RunInFoldCandidate:
    return RunInFoldCandidate(
        run_id=run_id or f"run-{fold_name}",
        fold_name=fold_name,
        train_end_utc=pd.Timestamp(train_end, tz="UTC"),
        model_version="model",
        seed=1,
    )


def test_selects_single_eligible_fold() -> None:
    selected = select_operationally_latest_fold(
        [_candidate("wf_1", "2020-01-01")],
        pd.Timestamp("2020-01-10", tz="UTC"),
        horizon=1,
    )
    assert selected is not None
    assert selected.fold_name == "wf_1"


def test_selects_latest_train_end_before_forecast_origin() -> None:
    selected = select_operationally_latest_fold(
        [
            _candidate("wf_1", "2020-01-01"),
            _candidate("wf_2", "2020-01-05"),
        ],
        pd.Timestamp("2020-01-10", tz="UTC"),
        horizon=2,
    )
    assert selected is not None
    assert selected.fold_name == "wf_2"


def test_excludes_train_end_equal_to_forecast_origin() -> None:
    selected = select_operationally_latest_fold(
        [
            _candidate("wf_1", "2020-01-01"),
            _candidate("wf_2", "2020-01-08"),
        ],
        pd.Timestamp("2020-01-10", tz="UTC"),
        horizon=2,
    )
    assert selected is not None
    assert selected.fold_name == "wf_1"


def test_returns_none_when_no_fold_is_eligible() -> None:
    assert (
        select_operationally_latest_fold(
            [_candidate("wf_1", "2020-01-09")],
            pd.Timestamp("2020-01-10", tz="UTC"),
            horizon=1,
        )
        is None
    )


def test_tiebreak_same_train_end_by_fold_name_then_run_id() -> None:
    selected = select_operationally_latest_fold(
        [
            _candidate("wf_b", "2020-01-01", run_id="b"),
            _candidate("wf_a", "2020-01-01", run_id="z"),
            _candidate("wf_a", "2020-01-01", run_id="a"),
        ],
        pd.Timestamp("2020-01-10", tz="UTC"),
        horizon=1,
    )
    assert selected is not None
    assert selected.fold_name == "wf_a"
    assert selected.run_id == "a"


def test_ignores_nat_train_end_and_rejects_invalid_horizon() -> None:
    selected = select_operationally_latest_fold(
        [
            RunInFoldCandidate("bad", "wf_bad", pd.NaT, "model", None),
            _candidate("wf_1", "2020-01-01"),
        ],
        pd.Timestamp("2020-01-10"),
        horizon=1,
    )
    assert selected is not None
    assert selected.fold_name == "wf_1"
    with pytest.raises(ValueError, match="horizon"):
        select_operationally_latest_fold([], pd.Timestamp("2020-01-10"), horizon=0)

