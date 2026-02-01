from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from src.domain.services.sentiment_aggregator import SentimentAggregator
from src.domain.time.utc import require_tz_aware, to_utc
from src.interfaces.daily_sentiment_repository import DailySentimentRepository
from src.interfaces.scored_news_repository import ScoredNewsRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SentimentFeatureEngineeringResult:
    asset_id: str
    read: int
    aggregated: int
    saved: int
    start: datetime
    end: datetime


class SentimentFeatureEngineeringUseCase:
    """
    Aggregates scored news into daily sentiment and persists to processed dataset.

    Flow:
      ScoredNewsRepository -> ScoredNewsArticle
      SentimentAggregator -> DailySentiment
      DailySentimentRepository -> Parquet
    """

    def __init__(
        self,
        scored_news_repository: ScoredNewsRepository,
        sentiment_aggregator: SentimentAggregator,
        daily_sentiment_repository: DailySentimentRepository,
    ) -> None:
        self.scored_news_repository = scored_news_repository
        self.sentiment_aggregator = sentiment_aggregator
        self.daily_sentiment_repository = daily_sentiment_repository

    def execute(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> SentimentFeatureEngineeringResult:
        require_tz_aware(start_date, "start_date")
        require_tz_aware(end_date, "end_date")

        start_utc = to_utc(start_date)
        end_utc = to_utc(end_date)
        if start_utc > end_utc:
            raise ValueError("start_date must be <= end_date")

        scored = self.scored_news_repository.list_scored_news(
            asset_id=asset_id,
            start_date=start_utc,
            end_date=end_utc,
        )

        if not scored:
            return SentimentFeatureEngineeringResult(
                asset_id=asset_id,
                read=0,
                aggregated=0,
                saved=0,
                start=start_utc,
                end=end_utc,
            )

        daily = self.sentiment_aggregator.aggregate_daily(asset_id, scored)

        if daily:
            self.daily_sentiment_repository.upsert_daily_sentiment_batch(daily)

        logger.info(
            "Daily sentiment aggregated",
            extra={
                "asset_id": asset_id,
                "read": len(scored),
                "aggregated": len(daily),
                "saved": len(daily),
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat(),
            },
        )

        return SentimentFeatureEngineeringResult(
            asset_id=asset_id,
            read=len(scored),
            aggregated=len(daily),
            saved=len(daily),
            start=start_utc,
            end=end_utc,
        )
