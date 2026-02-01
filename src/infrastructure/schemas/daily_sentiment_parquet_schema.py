from __future__ import annotations

from typing import Dict, List

DAILY_SENTIMENT_PARQUET_COLUMNS: List[str] = [
    "asset_id",
    "day",
    "sentiment_score",
    "n_articles",
    "sentiment_std",
]

# pandas dtypes for stable parquet writing
# NOTE: day handled separately as datetime64[ns, UTC]
DAILY_SENTIMENT_PARQUET_DTYPES: Dict[str, str] = {
    "asset_id": "string",
    "sentiment_score": "float64",
    "n_articles": "int64",
    "sentiment_std": "Float64",
}
