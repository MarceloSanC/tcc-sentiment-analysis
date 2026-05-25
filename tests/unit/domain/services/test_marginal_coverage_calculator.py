from __future__ import annotations

import pandas as pd
import pytest

from src.domain.services.marginal_coverage_calculator import compute_marginal_coverage


def _coverage_fixture(n: int = 100) -> pd.DataFrame:
    y = list(range(n))
    return pd.DataFrame(
        {
            "run_id": ["run_a"] * n,
            "split": ["test"] * n,
            "horizon": [1] * n,
            "y_true": y,
            "quantile_p10_post_guardrail": [9.5] * n,
            "quantile_p50_post_guardrail": [49.5] * n,
            "quantile_p90_post_guardrail": [89.5] * n,
        }
    )


def test_computes_expected_marginal_coverage_and_picp() -> None:
    out = compute_marginal_coverage(_coverage_fixture(), {"run_a"}, [0.1, 0.5, 0.9])
    row = out.iloc[0]
    assert row["coverage_q10"] == 0.10
    assert row["coverage_q50"] == 0.50
    assert row["coverage_q90"] == 0.90
    assert row["picp_q10_q90"] == 0.80
    assert row["coverage_error_picp"] == 0.0
    assert row["mpiw"] == 80.0


def test_filters_run_ids_before_grouping() -> None:
    df = pd.concat(
        [_coverage_fixture(), _coverage_fixture().assign(run_id="run_b")],
        ignore_index=True,
    )
    out = compute_marginal_coverage(df, {"run_b"}, [10, 50, 90])
    assert out["run_id"].tolist() == ["run_b"]
    assert int(out.iloc[0]["n_obs"]) == 100


def test_groups_by_run_split_and_horizon() -> None:
    df = pd.concat(
        [
            _coverage_fixture(),
            _coverage_fixture().assign(split="val", horizon=7),
        ],
        ignore_index=True,
    )
    out = compute_marginal_coverage(df, {"run_a"}, [0.1, 0.5, 0.9])
    assert set(zip(out["split"], out["horizon"], strict=True)) == {("test", 1), ("val", 7)}


def test_drops_rows_with_nan_supervision_or_quantiles() -> None:
    df = _coverage_fixture(5)
    df.loc[0, "y_true"] = None
    df.loc[1, "quantile_p50_post_guardrail"] = None
    out = compute_marginal_coverage(df, {"run_a"}, [0.1, 0.5, 0.9])
    assert int(out.iloc[0]["n_obs"]) == 3


def test_returns_empty_frame_when_run_ids_do_not_match() -> None:
    out = compute_marginal_coverage(_coverage_fixture(), {"missing"}, [0.1, 0.5, 0.9])
    assert out.empty
    assert {"coverage_q10", "coverage_q50", "coverage_q90"}.issubset(out.columns)


def test_validates_required_columns_and_quantile_levels() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        compute_marginal_coverage(pd.DataFrame({"run_id": ["run_a"]}), {"run_a"}, [0.1])
    with pytest.raises(ValueError, match="unsupported quantile"):
        compute_marginal_coverage(_coverage_fixture(), {"run_a"}, [0.2])

