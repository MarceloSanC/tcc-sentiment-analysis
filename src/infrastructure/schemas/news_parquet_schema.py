# src/infrastructure/schemas/news_parquet_schema.py

from __future__ import annotations

from typing import Dict, Set

NEWS_PARQUET_COLUMNS: Set[str] = {
    "asset_id",
    "article_id",
    "published_at",
    "headline",
    "summary",
    "source",
    "url",
    "language",
}

# pandas dtypes for stable parquet writing
# NOTE: published_at handled separately as datetime64[ns, UTC]
NEWS_PARQUET_DTYPES: Dict[str, str] = {
    "asset_id": "string",
    "article_id": "string",
    "headline": "string",
    "summary": "string",
    "source": "string",
    "url": "string",
    "language": "string",
}
