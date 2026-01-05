# src/infrastructure/schemas/tft_dataset_schema.py

TFT_DATASET_INDEX = [
    "asset_id",
    "time_idx",
]

TFT_STATIC_FEATURES = [
    "sector",
    "market_cap",
]

TFT_KNOWN_FEATURES = [
    "day_of_week",
    "month",
]

TFT_OBSERVED_FEATURES = [
    "close",
    "volume",
    "rsi",
    "sentiment_score",
]

TFT_TARGET = "target_return"
