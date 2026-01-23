# src/infrastructure/schemas/feature_set_parquet_schema.py

FEATURE_SET_BASE_COLUMNS = {
    "asset_id",
    "timestamp",
}

FEATURE_SET_DTYPES = {
    "asset_id": "string",
}

FEATURE_SET_INDEX = ["asset_id", "timestamp"]
