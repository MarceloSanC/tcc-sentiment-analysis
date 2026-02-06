from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DataQualityProfile:
    prefix: str
    date_col: str | None = None
    key_cols: list[str] | None = None
    expected_dtypes: dict[str, str] | None = None
    value_ranges: dict[str, tuple[float, float]] | None = None
    validation_rules: list[dict[str, Any]] | None = None
    comparison_rules: list[dict[str, Any]] | None = None
    business_days: bool = True

    def to_kwargs(self) -> dict[str, Any]:
        return {
            "prefix": self.prefix,
            "date_col": self.date_col,
            "key_cols": self.key_cols,
            "expected_dtypes": self.expected_dtypes,
            "value_ranges": self.value_ranges,
            "validation_rules": self.validation_rules,
            "comparison_rules": self.comparison_rules,
            "business_days": self.business_days,
        }


def get_profile(dataset: str) -> DataQualityProfile:
    dataset = dataset.lower()

    if dataset == "candles":
        return DataQualityProfile(
            prefix="candles",
            date_col="timestamp",
            key_cols=["timestamp"],
            value_ranges={
                "open": (0.0, float("inf")),
                "high": (0.0, float("inf")),
                "low": (0.0, float("inf")),
                "close": (0.0, float("inf")),
                "volume": (0.0, float("inf")),
            },
            comparison_rules=[
                {"name": "high>=low", "left": "high", "right": "low", "op": ">="},
                {"name": "high>=open", "left": "high", "right": "open", "op": ">="},
                {"name": "high>=close", "left": "high", "right": "close", "op": ">="},
                {"name": "low<=open", "left": "low", "right": "open", "op": "<="},
                {"name": "low<=close", "left": "low", "right": "close", "op": "<="},
            ],
            business_days=True,
        )

    if dataset == "news_raw":
        return DataQualityProfile(
            prefix="news_raw",
            date_col="published_at",
            key_cols=["article_id"],
            validation_rules=[
                {"name": "url_not_null", "column": "url", "op": "not_null"},
                {"name": "source_not_null", "column": "source", "op": "not_null"},
            ],
            business_days=False,
        )

    if dataset == "scored_news":
        return DataQualityProfile(
            prefix="scored_news",
            date_col="published_at",
            key_cols=["article_id"],
            value_ranges={"sentiment_score": (-1.0, 1.0)},
            validation_rules=[
                {"name": "confidence_range", "column": "confidence", "op": "between", "min": 0.0, "max": 1.0},
                {"name": "model_name_not_null", "column": "model_name", "op": "not_null"},
            ],
            business_days=False,
        )

    if dataset == "sentiment_daily":
        return DataQualityProfile(
            prefix="sentiment_daily",
            date_col="day",
            key_cols=["day"],
            value_ranges={"sentiment_score": (-1.0, 1.0)},
            validation_rules=[
                {"name": "news_volume_nonneg", "column": "n_articles", "op": "nonnegative"},
                {"name": "sentiment_std_nonneg", "column": "sentiment_std", "op": "nonnegative"},
            ],
            business_days=True,
        )

    if dataset == "technical_indicators":
        return DataQualityProfile(
            prefix="technical_indicators",
            date_col="timestamp",
            key_cols=["timestamp"],
            business_days=True,
        )

    if dataset == "fundamentals":
        return DataQualityProfile(
            prefix="fundamentals",
            date_col="fiscal_date_end",
            key_cols=["report_type", "fiscal_date_end"],
            validation_rules=[
                {
                    "name": "report_type_in_set",
                    "column": "report_type",
                    "op": "in",
                    "values": ["annual", "quarterly"],
                }
            ],
            business_days=False,
        )

    if dataset == "dataset_tft":
        return DataQualityProfile(
            prefix="dataset_tft",
            date_col="timestamp",
            key_cols=["timestamp"],
            value_ranges={
                "close": (0.0, float("inf")),
                "volume": (0.0, float("inf")),
                "sentiment_score": (-1.0, 1.0),
            },
            validation_rules=[
                {"name": "time_idx_nonneg", "column": "time_idx", "op": "nonnegative"},
                {"name": "news_volume_nonneg", "column": "news_volume", "op": "nonnegative"},
            ],
            business_days=True,
        )

    raise ValueError(f"Unknown data quality profile: {dataset}")
