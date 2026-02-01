from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.domain.time.utc import require_tz_aware, to_utc
from src.entities.scored_news_article import ScoredNewsArticle
from src.infrastructure.schemas.scored_news_parquet_schema import (
    SCORED_NEWS_PARQUET_COLUMNS,
    SCORED_NEWS_PARQUET_DTYPES,
)
from src.interfaces.scored_news_repository import ScoredNewsRepository

logger = logging.getLogger(__name__)


class ParquetScoredNewsRepository(ScoredNewsRepository):
    """
    Parquet-based repository for ScoredNewsArticle.

    Storage layout:
      data/processed/scored_news/AAPL/scored_news_AAPL.parquet

    Temporal contract:
    - All datetimes in/out are timezone-aware UTC.
    - published_at stored as datetime64[ns, UTC] in parquet.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

        # Fail fast: path exists but is not a directory
        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise NotADirectoryError(
                f"Scored news output_dir is not a directory: {self.output_dir.resolve()}"
            )

        # Create if missing
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ParquetScoredNewsRepository initialized",
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
        return self._asset_dir(symbol) / f"scored_news_{symbol}.parquet"

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
        return latest.to_pydatetime() 

    def upsert_scored_news_batch(self, articles: list[ScoredNewsArticle]) -> None:
        if not articles:
            raise ValueError("No scored news articles to upsert")

        asset = articles[0].asset_id
        if any(a.asset_id != asset for a in articles):
            raise ValueError("All articles in a batch must share the same asset_id")

        rows: list[dict] = []
        for a in articles:
            require_tz_aware(a.published_at, "published_at")
            published_at_utc = to_utc(a.published_at)

            rows.append(
                {
                    "asset_id": self._normalize_symbol(a.asset_id),
                    "article_id": str(a.article_id),
                    "published_at": published_at_utc,
                    "sentiment_score": float(a.sentiment_score),
                    "confidence": float(a.confidence) if a.confidence is not None else None,
                    "model_name": a.model_name,
                }
            )

        df_new = pd.DataFrame(rows)
        df_new = df_new[SCORED_NEWS_PARQUET_COLUMNS]

        df_new["published_at"] = pd.to_datetime(
            df_new["published_at"], utc=True, errors="raise"
        )
        for col, dtype in SCORED_NEWS_PARQUET_DTYPES.items():
            if col in df_new.columns:
                df_new[col] = df_new[col].astype(dtype)

        filepath = self._filepath(asset)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.exists():
            df_old = pd.read_parquet(filepath)
            if not df_old.empty:
                df_old["published_at"] = pd.to_datetime(
                    df_old["published_at"], utc=True, errors="coerce"
                )
                df = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df = df_new
        else:
            df = df_new

        df = df.drop_duplicates(subset=["article_id"], keep="last")
        df = df.sort_values("published_at").reset_index(drop=True)

        missing = set(SCORED_NEWS_PARQUET_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for scored news parquet: {sorted(missing)}")

        df.to_parquet(filepath, index=False)

        logger.debug(
            "Scored news upserted",
            extra={
                "asset_id": self._normalize_symbol(asset),
                "saved_rows": len(df_new),
                "total_rows": len(df),
                "path": str(filepath.resolve()),
            },
        )

    def list_scored_news(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[ScoredNewsArticle]:
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

        mask = (df["published_at"] >= pd.Timestamp(start_utc)) & (
            df["published_at"] <= pd.Timestamp(end_utc)
        )
        df = df.loc[mask].sort_values("published_at")

        out: list[ScoredNewsArticle] = []
        for _, r in df.iterrows():
            confidence = r.get("confidence")
            model_name = r.get("model_name")
            if pd.isna(confidence):
                confidence = None
            if pd.isna(model_name) or model_name == "":
                model_name = None

            out.append(
                ScoredNewsArticle(
                    asset_id=str(r.get("asset_id") or self._normalize_symbol(asset_id)),
                    article_id=str(r.get("article_id")),
                    published_at=pd.Timestamp(r["published_at"]).to_pydatetime(),
                    sentiment_score=float(r.get("sentiment_score")),
                    confidence=float(confidence) if confidence is not None else None,
                    model_name=str(model_name) if model_name is not None else None,
                )
            )

        return out

    def list_article_ids(self, asset_id: str) -> set[str]:
        filepath = self._filepath(asset_id)
        if not filepath.exists():
            return set()

        df = pd.read_parquet(filepath, columns=["article_id"])
        if df.empty:
            return set()

        ids = set()
        for v in df["article_id"].dropna().tolist():
            if str(v).strip():
                ids.add(str(v))
        return ids
