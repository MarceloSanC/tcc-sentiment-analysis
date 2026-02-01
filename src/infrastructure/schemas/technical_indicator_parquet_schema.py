# src/infrastructure/schemas/feature_set_parquet_schema.py

TECHNICAL_INDICATOR_BASE_COLUMNS = {
    "asset_id",
    "timestamp",
}

TECHNICAL_INDICATOR_DTYPES = {
    "asset_id": "string",
}

TECHNICAL_INDICATOR_INDEX = ["asset_id", "timestamp"]
