from __future__ import annotations

import pandas as pd
import pytest

from src.domain.services.feature_warmup_inspector import FeatureWarmupInspector


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_detects_leading_null_warmup_for_multiple_features() -> None:
    df = _df(
        [
            {"timestamp": "2010-01-06", "f1": 1.0, "f2": 4.0},
            {"timestamp": "2010-01-04", "f1": None, "f2": None},
            {"timestamp": "2010-01-05", "f1": None, "f2": None},
        ]
    )

    out = FeatureWarmupInspector.detect_leading_null_warmups(
        df,
        ["f1", "f2"],
        requested_start="20100101",
        requested_end="20100131",
    )

    assert len(out) == 2
    by_feature = {x.feature_name: x for x in out}
    assert by_feature["f1"].num_null == 2
    assert by_feature["f1"].first_date_warmup == "2010-01-04"
    assert by_feature["f1"].last_date_warmup_null == "2010-01-05"
    assert by_feature["f2"].num_null == 2


def test_does_not_flag_non_leading_nulls() -> None:
    df = _df(
        [
            {"timestamp": "2010-01-04", "f1": 1.0},
            {"timestamp": "2010-01-05", "f1": None},
            {"timestamp": "2010-01-06", "f1": 2.0},
        ]
    )

    out = FeatureWarmupInspector.detect_leading_null_warmups(
        df,
        ["f1"],
        requested_start="20100101",
        requested_end="20100131",
    )

    assert out == []


def test_respects_requested_period_boundaries() -> None:
    df = _df(
        [
            {"timestamp": "2010-01-04", "f1": None},
            {"timestamp": "2010-01-05", "f1": None},
            {"timestamp": "2010-01-06", "f1": 3.0},
        ]
    )

    out = FeatureWarmupInspector.detect_leading_null_warmups(
        df,
        ["f1"],
        requested_start="20100105",
        requested_end="20100131",
    )

    assert len(out) == 1
    assert out[0].num_null == 1
    assert out[0].first_date_warmup == "2010-01-05"
    assert out[0].last_date_warmup_null == "2010-01-05"


def test_ignores_missing_timestamp_or_empty_data() -> None:
    assert (
        FeatureWarmupInspector.detect_leading_null_warmups(
            pd.DataFrame(), ["f1"], requested_start="20100101", requested_end="20100131"
        )
        == []
    )

    no_ts = pd.DataFrame({"f1": [None, 1.0]})
    assert (
        FeatureWarmupInspector.detect_leading_null_warmups(
            no_ts, ["f1"], requested_start="20100101", requested_end="20100131"
        )
        == []
    )


def test_raises_on_invalid_date_format_to_avoid_silent_validation_bug() -> None:
    df = _df([{"timestamp": "2010-01-04", "f1": None}])
    with pytest.raises(ValueError):
        FeatureWarmupInspector.detect_leading_null_warmups(
            df, ["f1"], requested_start="2010-01-01", requested_end="20100131"
        )

