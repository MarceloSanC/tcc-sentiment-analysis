# src/adapters/parquet_news_repository.py

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.domain.time.utc import require_tz_aware, to_utc
from src.entities.news_article import NewsArticle
from src.infrastructure.schemas.news_parquet_schema import (
    NEWS_PARQUET_COLUMNS,
    NEWS_PARQUET_DTYPES,
)
from src.interfaces.news_repository import NewsRepository

logger = logging.getLogger(__name__)


class ParquetNewsRepository(NewsRepository):
    """
    Parquet-based repository for NewsArticle.

    Storage layout:
      data/raw/news/AAPL/news_AAPL.parquet

    Temporal contract:
    - All datetimes in/out are timezone-aware UTC.
    - published_at stored as datetime64[ns, UTC] in parquet.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

        # Fail fast: path exists but is not a directory
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise NotADirectoryError(
                f"News output_dir is not a directory: {self.output_dir.resolve()}"
            )

        # Create if missing
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ParquetNewsRepository initialized",
            extra={"output_dir": str(self.output_dir.resolve())},
        )

    @staticmethod
    def _normalize_symbol(asset_id: str) -> str:
        return asset_id.split(".")[0].upper()

    def _asset_dir(self, asset_id: str) -> Path:
        symbol = self._normalize_symbol(asset_id)
        return self.output_dir / symbol

    def _filepath(self, asset_id: str) -> Path:
        symbol = self._normalize_symbol(asset_id)
        return self._asset_dir(symbol) / f"news_{symbol}.parquet"

    @staticmethod
    def _ensure_article_ids(articles: Iterable[NewsArticle]) -> list[NewsArticle]:
        """
        Enforce a stable article_id for dedup/upsert.
        Policy:
        - If article_id is missing but url exists, article_id := url
        - If both missing, raise (data quality issue)
        """
        normalized: list[NewsArticle] = []
        for a in articles:
            article_id = a.article_id or a.url
            if not article_id:
                raise ValueError(
                    "NewsArticle must have article_id or url (used as stable id)."
                )
            normalized.append(
                NewsArticle(
                    article_id=str(article_id),
                    asset_id=a.asset_id,
                    published_at=a.published_at,
                    headline=a.headline,
                    summary=a.summary,
                    source=a.source,
                    url=a.url,
                    language=a.language or "en",
                )
            )
        return normalized

    def get_latest_published_at(self, asset_id: str) -> datetime | None:
        filepath = self._filepath(asset_id)
        if not filepath.exists():
            return None

        df = pd.read_parquet(filepath, columns=["published_at"])
        if df.empty:
            return None

        published = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        if published.isna().all():
            raise ValueError(
                f"Could not parse published_at in parquet for {asset_id}: {filepath}"
            )

        latest = published.max()
        # pandas Timestamp -> python datetime (tz-aware)
        return latest.to_pydatetime()

    def upsert_news_batch(self, articles: list[NewsArticle]) -> None:
        if not articles:
            raise ValueError("No news articles to upsert")

        # Enforce single-asset batch (simplifies layout & correctness)
        asset = articles[0].asset_id
        if any(a.asset_id != asset for a in articles):
            raise ValueError("All articles in a batch must share the same asset_id")

        articles = self._ensure_article_ids(articles)

        # Validate + normalize UTC
        rows: list[dict] = []
        for a in articles:
            require_tz_aware(a.published_at, "published_at")
            published_at_utc = to_utc(a.published_at)

            rows.append(
                {
                    "asset_id": self._normalize_symbol(a.asset_id),
                    "article_id": str(a.article_id),
                    "published_at": published_at_utc,
                    "headline": a.headline,
                    "summary": a.summary,
                    "source": a.source,
                    "url": a.url,
                    "language": a.language or "en",
                }
            )

        df_new = pd.DataFrame(rows)
        df_new = df_new[list(NEWS_PARQUET_COLUMNS)]

        # Normalize dtypes (especially published_at UTC)
        df_new["published_at"] = pd.to_datetime(df_new["published_at"], utc=True, errors="raise")
        for col, dtype in NEWS_PARQUET_DTYPES.items():
            if col == "published_at":
                continue
            if col in df_new.columns:
                df_new[col] = df_new[col].astype(dtype)

        # Ensure target dirs exist
        filepath = self._filepath(asset)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.exists():
            df_old = pd.read_parquet(filepath)
            if not df_old.empty:
                # Ensure published_at UTC
                df_old["published_at"] = pd.to_datetime(df_old["published_at"], utc=True, errors="coerce")
                df = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df = df_new
        else:
            df = df_new

        # Upsert semantics: dedup by article_id, keep last occurrence
        df = df.drop_duplicates(subset=["article_id"], keep="last")
        df = df.sort_values("published_at").reset_index(drop=True)

        # Basic schema guard
        missing = set(NEWS_PARQUET_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for news parquet: {sorted(missing)}")

        df.to_parquet(filepath, index=False)

        logger.info(
            "News upserted",
            extra={
                "asset_id": self._normalize_symbol(asset),
                "saved_rows": len(df_new),
                "total_rows": len(df),
                "path": str(filepath.resolve()),
            },
        )

    def list_news(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[NewsArticle]:
        require_tz_aware(start_date, "start_date")
        require_tz_aware(end_date, "end_date")

        start_utc = to_utc(start_date)
        end_utc = to_utc(end_date)

        if start_utc > end_utc:
            raise ValueError("start_date must be <= end_date")

        filepath = self._filepath(asset_id)
        if not filepath.exists():
            return []

        df = pd.read_parquet(filepath)
        if df.empty:
            return []

        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        if df["published_at"].isna().any():
            raise ValueError(f"Invalid published_at values found in {filepath}")

        # Inclusive filter
        mask = (df["published_at"] >= pd.Timestamp(start_utc)) & (
            df["published_at"] <= pd.Timestamp(end_utc)
        )
        df = df.loc[mask].sort_values("published_at")

        out: list[NewsArticle] = []
        for _, r in df.iterrows():
            out.append(
                NewsArticle(
                    asset_id=str(r.get("asset_id") or self._normalize_symbol(asset_id)),
                    article_id=str(r.get("article_id")) if r.get("article_id") is not None else None,
                    published_at=pd.Timestamp(r["published_at"]).to_pydatetime(),
                    headline=str(r.get("headline") or ""),
                    summary=str(r.get("summary") or ""),
                    source=str(r.get("source") or ""),
                    url=str(r.get("url")) if r.get("url") not in (None, "") else None,
                    language=str(r.get("language") or "en"),
                )
            )

        return out
