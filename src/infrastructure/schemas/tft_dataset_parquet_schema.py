from __future__ import annotations

from typing import Dict, List

TFT_DATASET_BASE_COLUMNS: List[str] = [
    "asset_id",
    "timestamp",
    "time_idx",
    "day_of_week",
    "month",
    "target_return",
]

TFT_DATASET_DTYPES: Dict[str, str] = {
    "asset_id": "string",
    "time_idx": "int64",
    "day_of_week": "int64",
    "month": "int64",
    "target_return": "float64",
}

BASELINE_FEATURES: List[str] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
]

TECHNICAL_FEATURES: List[str] = [
    "volatility_20d",
    "rsi_14",
    "candle_body",
    "macd_signal",
    "ema_100",
    "macd",
    "ema_10",
    "ema_200",
    "ema_50",
    "candle_range",
]

SENTIMENT_FEATURES: List[str] = [
    "sentiment_score",
    "news_volume",
    "sentiment_std",
]

FUNDAMENTAL_FEATURES: List[str] = [
    "revenue",
    "net_income",
    "operating_cash_flow",
    "total_shareholder_equity",
    "total_liabilities",
]

DEFAULT_TFT_FEATURES: List[str] = [
    *BASELINE_FEATURES
]
