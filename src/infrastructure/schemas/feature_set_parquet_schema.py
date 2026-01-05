# src/infrastructure/schemas/feature_set_parquet_schema.py

FEATURE_SET_BASE_COLUMNS = {
    "asset_id",
    "timestamp",
    "feature_name",
    "feature_value",
}

FEATURE_SET_DTYPES = {
    "asset_id": "string",
    "feature_name": "string",
    "feature_value": "float32",
}

FEATURE_SET_INDEX = ["asset_id", "timestamp"]
