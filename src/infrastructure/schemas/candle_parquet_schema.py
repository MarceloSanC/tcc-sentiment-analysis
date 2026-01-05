# src/infrastructure/schemas/candle_parquet_schema.py

CANDLE_PARQUET_COLUMNS = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
}

CANDLE_PARQUET_DTYPES = {
    "open": "float32",
    "high": "float32",
    "low": "float32",
    "close": "float32",
    "volume": "int64",
}
