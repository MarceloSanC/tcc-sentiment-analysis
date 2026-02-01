from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from src.domain.time.utc import require_tz_aware, to_utc
from src.entities.news_article import NewsArticle
from src.entities.scored_news_article import ScoredNewsArticle
from src.interfaces.news_repository import NewsRepository
from src.interfaces.scored_news_repository import ScoredNewsRepository
from src.interfaces.sentiment_model import SentimentModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InferSentimentResult:
    asset_id: str
    read: int
    skipped: int
    scored: int
    saved: int
    start: datetime
    end: datetime


class InferSentimentUseCase:
    """
    Infers sentiment for raw news articles and persists them into the processed dataset.

    Flow:
      NewsRepository -> NewsArticle
      SentimentModel -> ScoredNewsArticle
      ScoredNewsRepository -> Parquet
    """

    def __init__(
        self,
        news_repository: NewsRepository,
        sentiment_model: SentimentModel,
        scored_news_repository: ScoredNewsRepository,
        batch_size: int = 32,
    ) -> None:
        self.news_repository = news_repository
        self.sentiment_model = sentiment_model
        self.scored_news_repository = scored_news_repository
        self.batch_size = int(batch_size)

    @staticmethod
    def _chunk(items: list[NewsArticle], size: int) -> Iterable[list[NewsArticle]]:
        for i in range(0, len(items), size):
            yield items[i : i + size]

    @staticmethod
    def _require_article_ids(articles: Iterable[NewsArticle]) -> None:
        for a in articles:
            if a.article_id is None or not str(a.article_id).strip():
                raise ValueError(
                    "NewsArticle.article_id is required for scoring (stable id)."
                )

    def execute(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> InferSentimentResult:
        require_tz_aware(start_date, "start_date")
        require_tz_aware(end_date, "end_date")

        start_utc = to_utc(start_date)
        end_utc = to_utc(end_date)
        if start_utc > end_utc:
            raise ValueError("start_date must be <= end_date")

        articles = self.news_repository.list_news(asset_id, start_utc, end_utc)
        if not articles:
            return InferSentimentResult(
                asset_id=asset_id,
                read=0,
                skipped=0,
                scored=0,
                saved=0,
                start=start_utc,
                end=end_utc,
            )

        articles = sorted(articles, key=lambda a: a.published_at)
        self._require_article_ids(articles)

        scored_ids = self.scored_news_repository.list_article_ids(asset_id)
        candidates = [a for a in articles if str(a.article_id) not in scored_ids]

        skipped = len(articles) - len(candidates)

        if not candidates:
            logger.info(
                "Sentiment inference skipped (all articles already scored)",
                extra={
                    "asset_id": asset_id,
                    "read": len(articles),
                    "skipped": skipped,
                    "start": start_utc.isoformat(),
                    "end": end_utc.isoformat(),
                },
            )
            return InferSentimentResult(
                asset_id=asset_id,
                read=len(articles),
                skipped=skipped,
                scored=0,
                saved=0,
                start=start_utc,
                end=end_utc,
            )

        scored_total = 0
        saved_total = 0
        total_candidates = len(candidates)

        for batch in self._chunk(candidates, self.batch_size):
            scored_batch: list[ScoredNewsArticle] = self.sentiment_model.infer(batch)
            scored_total += len(scored_batch)

            if scored_batch:
                self.scored_news_repository.upsert_scored_news_batch(scored_batch)
                saved_total += len(scored_batch)

            logger.info(
                f"Scored news progress {len(scored_batch)}/{total_candidates}",
                extra={
                    "asset_id": asset_id,
                    "scored": scored_total,
                    "total": total_candidates,
                },
            )

        logger.info(
            "Scored news dataset",
            extra={
                "asset_id": asset_id,
                "read": len(articles),
                "skipped": skipped,
                "scored": scored_total,
                "saved": saved_total,
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat(),
            },
        )

        return InferSentimentResult(
            asset_id=asset_id,
            read=len(articles),
            skipped=skipped,
            scored=scored_total,
            saved=saved_total,
            start=start_utc,
            end=end_utc,
        )
