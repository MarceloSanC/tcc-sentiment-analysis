# tests/unit/domain/services/test_data_quality_reporter.py

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.domain.services.data_quality_reporter import DataQualityReporter
from src.domain.services.data_quality_profiles import get_profile


def _dt_utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _profile_kwargs(profile) -> dict:
    data = profile.to_kwargs()
    data.pop("prefix", None)
    return data


def test_generate_report_with_dates_and_gaps() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                _dt_utc(2024, 1, 1),
                _dt_utc(2024, 1, 2),
                _dt_utc(2024, 1, 4),
            ],
            "close": [100.0, 101.0, 102.0],
        }
    )

    report = DataQualityReporter.generate_report(
        df,
        date_col="timestamp",
        key_cols=["timestamp"],
        business_days=False,
    )

    assert report["rows"] == 3
    assert report["date_min"] == "2024-01-01"
    assert report["date_max"] == "2024-01-04"
    assert report["missing_days_count"] == 1  # 2024-01-03
    assert report["gaps_count"] == 1
    assert report["invalid_date_rows"] == 0
    assert report["duplicate_keys"] == 0


def test_validation_and_comparison_rules() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [_dt_utc(2024, 1, 1), _dt_utc(2024, 1, 2)],
            "high": [10.0, 5.0],
            "low": [9.0, 6.0],  # second row violates high>=low
            "sentiment_score": [0.5, 2.0],  # second row out of range
        }
    )

    report = DataQualityReporter.generate_report(
        df,
        date_col="timestamp",
        key_cols=["timestamp"],
        value_ranges={"sentiment_score": (-1.0, 1.0)},
        comparison_rules=[
            {"name": "high>=low", "left": "high", "right": "low", "op": ">="}
        ],
        business_days=False,
    )

    assert report["out_of_range"]["sentiment_score"] == 1
    assert report["comparison_rules"]["high>=low"] == 1
    assert report["invalid_rows_total"] >= 1


def test_report_from_parquet_writes_versioned_file(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "timestamp": [_dt_utc(2024, 1, 1), _dt_utc(2024, 1, 2)],
            "close": [100.0, 101.0],
        }
    )
    parquet_path = tmp_path / "dataset_tft_AAPL.parquet"
    df.to_parquet(parquet_path, index=False)

    profile = get_profile("dataset_tft")
    report_path = DataQualityReporter.report_from_parquet(parquet_path, **profile.to_kwargs())

    assert report_path.exists()
    assert report_path.parent.name == "reports"

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["source_path"].endswith("dataset_tft_AAPL.parquet")
    assert "file_hash_sha256" in payload


def test_report_exists(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    assert DataQualityReporter.report_exists(reports_dir, "candles") is False
    (reports_dir / "candles_report_20240101_000000.json").write_text("{}", encoding="utf-8")
    assert DataQualityReporter.report_exists(reports_dir, "candles") is True


def test_profile_candles_validation_rules() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [_dt_utc(2024, 1, 1), _dt_utc(2024, 1, 2)],
            "open": [10.0, 10.0],
            "high": [9.0, 11.0],  # first row violates high>=open/close/low
            "low": [11.0, 9.0],  # first row violates low<=open/close
            "close": [10.0, 10.0],
            "volume": [100, -1],  # second row violates nonnegative
        }
    )
    profile = get_profile("candles")
    report = DataQualityReporter.generate_report(df, **_profile_kwargs(profile))
    assert report["comparison_rules"]["high>=low"] == 1
    assert report["comparison_rules"]["high>=open"] == 1
    assert report["comparison_rules"]["high>=close"] == 1
    assert report["comparison_rules"]["low<=open"] == 1
    assert report["comparison_rules"]["low<=close"] == 1
    assert report["out_of_range"]["volume"] == 1


def test_profile_scored_news_rules() -> None:
    df = pd.DataFrame(
        {
            "article_id": ["a1", "a2"],
            "published_at": [_dt_utc(2024, 1, 1), _dt_utc(2024, 1, 2)],
            "sentiment_score": [0.2, 2.0],  # second out of range
            "model_name": ["m", None],  # second invalid
            "confidence": [0.5, 1.2],  # second invalid
        }
    )
    profile = get_profile("scored_news")
    report = DataQualityReporter.generate_report(df, **_profile_kwargs(profile))
    assert report["out_of_range"]["sentiment_score"] == 1
    assert report["validation_rules"]["model_name_not_null"] == 1
    assert report["validation_rules"]["confidence_range"] == 1


def test_profile_sentiment_daily_rules() -> None:
    df = pd.DataFrame(
        {
            "day": [_dt_utc(2024, 1, 1), _dt_utc(2024, 1, 2)],
            "sentiment_score": [0.1, -0.2],
            "n_articles": [2, -1],
            "sentiment_std": [0.1, -0.5],
        }
    )
    profile = get_profile("sentiment_daily")
    report = DataQualityReporter.generate_report(df, **_profile_kwargs(profile))
    assert report["validation_rules"]["news_volume_nonneg"] == 1
    assert report["validation_rules"]["sentiment_std_nonneg"] == 1


def test_profile_fundamentals_rules() -> None:
    df = pd.DataFrame(
        {
            "report_type": ["annual", "invalid"],
            "fiscal_date_end": [_dt_utc(2023, 12, 31), _dt_utc(2024, 3, 31)],
        }
    )
    profile = get_profile("fundamentals")
    report = DataQualityReporter.generate_report(df, **_profile_kwargs(profile))
    assert report["validation_rules"]["report_type_in_set"] == 1


def test_profile_dataset_tft_rules() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [_dt_utc(2024, 1, 1), _dt_utc(2024, 1, 2)],
            "time_idx": [0, -1],
            "close": [100.0, -1.0],
            "volume": [100.0, -10.0],
            "sentiment_score": [0.5, 2.0],
            "news_volume": [1, -5],
        }
    )
    profile = get_profile("dataset_tft")
    report = DataQualityReporter.generate_report(df, **_profile_kwargs(profile))
    assert report["validation_rules"]["time_idx_nonneg"] == 1
    assert report["validation_rules"]["news_volume_nonneg"] == 1
    assert report["out_of_range"]["close"] == 1
    assert report["out_of_range"]["volume"] == 1
    assert report["out_of_range"]["sentiment_score"] == 1


def test_report_hash_changes_when_file_changes(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "timestamp": [_dt_utc(2024, 1, 1), _dt_utc(2024, 1, 2)],
            "close": [100.0, 101.0],
        }
    )
    parquet_path = tmp_path / "candles_AAPL_1d.parquet"
    df.to_parquet(parquet_path, index=False)

    profile = get_profile("candles")
    report_path = DataQualityReporter.report_from_parquet(parquet_path, **profile.to_kwargs())
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    first_hash = payload["file_hash_sha256"]

    df.loc[1, "close"] = 105.0
    df.to_parquet(parquet_path, index=False)

    report_path_2 = DataQualityReporter.report_from_parquet(parquet_path, **profile.to_kwargs())
    payload_2 = json.loads(report_path_2.read_text(encoding="utf-8"))
    assert payload_2["file_hash_sha256"] != first_hash


def test_report_from_parquet_handles_invalid_file(tmp_path: Path) -> None:
    bad_path = tmp_path / "invalid.parquet"
    bad_path.write_text("not parquet", encoding="utf-8")

    profile = get_profile("candles")
    out = DataQualityReporter.report_from_parquet(bad_path, **profile.to_kwargs())
    assert out == bad_path


def test_business_days_flag_changes_missing_days() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                _dt_utc(2024, 1, 5),  # Fri
                _dt_utc(2024, 1, 8),  # Mon
            ]
        }
    )
    profile = get_profile("candles")

    report_bd = DataQualityReporter.generate_report(
        df, date_col="timestamp", key_cols=["timestamp"], business_days=True
    )
    report_all = DataQualityReporter.generate_report(
        df, date_col="timestamp", key_cols=["timestamp"], business_days=False
    )

    assert report_bd["missing_days_count"] != report_all["missing_days_count"]


def test_expected_dtypes_mismatch() -> None:
    df = pd.DataFrame({"timestamp": [_dt_utc(2024, 1, 1)], "close": [100.0]})
    report = DataQualityReporter.generate_report(
        df,
        date_col="timestamp",
        expected_dtypes={"close": "int64"},
        business_days=False,
    )
    assert "close" in report["dtype_mismatches"]


def test_missing_key_cols_detected() -> None:
    df = pd.DataFrame({"timestamp": [_dt_utc(2024, 1, 1)]})
    report = DataQualityReporter.generate_report(
        df,
        date_col="timestamp",
        key_cols=["timestamp", "article_id"],
        business_days=False,
    )
    assert "article_id" in report["missing_key_cols"]


def test_multiple_rows_per_day_counts() -> None:
    df = pd.DataFrame(
        {
            "timestamp": [
                _dt_utc(2024, 1, 1),
                _dt_utc(2024, 1, 1),
                _dt_utc(2024, 1, 2),
            ]
        }
    )
    report = DataQualityReporter.generate_report(
        df, date_col="timestamp", key_cols=["timestamp"], business_days=False
    )
    assert report["per_day_counts"]["2024-01-01"] == 2
    assert report["per_day_counts"]["2024-01-02"] == 1


def test_validation_rules_count_nans() -> None:
    df = pd.DataFrame({"value": [1.0, None, 2.0]})
    report = DataQualityReporter.generate_report(
        df,
        validation_rules=[{"name": "value_not_null", "column": "value", "op": "not_null"}],
        business_days=False,
    )
    assert report["validation_rules"]["value_not_null"] == 1


def test_comparison_rules_with_nan_counts_invalid() -> None:
    df = pd.DataFrame(
        {
            "high": [10.0, None],
            "low": [9.0, 8.0],
        }
    )
    report = DataQualityReporter.generate_report(
        df,
        comparison_rules=[{"name": "high>=low", "left": "high", "right": "low", "op": ">="}],
        business_days=False,
    )
    # second row should be counted as invalid
    assert report["comparison_rules"]["high>=low"] == 1


def test_out_of_range_missing_column_is_ignored() -> None:
    df = pd.DataFrame({"value": [1.0, 2.0]})
    report = DataQualityReporter.generate_report(
        df,
        value_ranges={"missing_col": (0.0, 1.0)},
        business_days=False,
    )
    assert report["out_of_range"] == {}
