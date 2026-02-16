from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.domain.services.sentiment_aggregator import SentimentAggregator
from src.domain.time.trading_calendar import normalize_to_trading_day
from src.domain.time.utc import require_tz_aware, to_utc
from src.entities.daily_sentiment import DailySentiment
from src.entities.scored_news_article import ScoredNewsArticle
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

        daily = self.sentiment_aggregator.aggregate_daily(asset_id, scored)
        daily = self._fill_missing_days_with_zero_news(
            asset_id=asset_id,
            daily=daily,
            start_utc=start_utc,
            end_utc=end_utc,
        )
        self._validate_daily_causality(
            scored=scored,
            daily=daily,
            start_utc=start_utc,
            end_utc=end_utc,
        )

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

    @staticmethod
    def _validate_daily_causality(
        *,
        scored: list[ScoredNewsArticle],
        daily: list[DailySentiment],
        start_utc: datetime,
        end_utc: datetime,
    ) -> None:
        """
        Explicit causality guard:
        - only scored news inside requested window can feed daily sentiment;
        - each daily row must map to at least one source news day;
        - n_articles in daily aggregate must match source count of that day.
        """
        start_day = start_utc.date()
        end_day = end_utc.date()

        day_counts: dict = {}
        for article in scored:
            article_day = normalize_to_trading_day(article.published_at)
            if article_day < start_day or article_day > end_day:
                raise ValueError(
                    "Causality violation: scored news outside requested date window"
                )
            day_counts[article_day] = day_counts.get(article_day, 0) + 1

        for item in daily:
            if item.day < start_day or item.day > end_day:
                raise ValueError(
                    "Causality violation: daily sentiment day outside requested date window"
                )

            if item.day in day_counts:
                if item.n_articles != day_counts[item.day]:
                    raise ValueError(
                        "Causality violation: n_articles does not match source news count"
                    )
                continue

            if item.n_articles != 0:
                raise ValueError(
                    "Causality violation: synthetic day without source news must have n_articles=0"
                )
            if item.sentiment_score != 0.0:
                raise ValueError(
                    "Causality violation: synthetic day without source news must have sentiment_score=0.0"
                )

        daily_days = {item.day for item in daily}
        missing_source_days = sorted(day for day in day_counts if day not in daily_days)
        if missing_source_days:
            raise ValueError(
                "Causality violation: source news day missing from daily sentiment aggregate"
            )

    @staticmethod
    def _fill_missing_days_with_zero_news(
        *,
        asset_id: str,
        daily: list[DailySentiment],
        start_utc: datetime,
        end_utc: datetime,
    ) -> list[DailySentiment]:
        by_day = {item.day: item for item in daily}

        all_days: list = []
        current = start_utc.date()
        end_day = end_utc.date()
        while current <= end_day:
            all_days.append(current)
            current += timedelta(days=1)

        filled: list[DailySentiment] = []
        for day in all_days:
            existing = by_day.get(day)
            if existing is not None:
                filled.append(existing)
                continue

            filled.append(
                DailySentiment(
                    asset_id=asset_id,
                    day=day,
                    sentiment_score=0.0,
                    n_articles=0,
                    sentiment_std=0.0,
                )
            )
        return filled
