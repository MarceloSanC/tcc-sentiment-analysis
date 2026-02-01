from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from src.entities.daily_sentiment import DailySentiment


class DailySentimentRepository(ABC):
    @abstractmethod
    def upsert_daily_sentiment_batch(
        self,
        daily_sentiments: list[DailySentiment],
    ) -> None:
        """Insert or update daily sentiment (dedup by day)."""
        ...

    @abstractmethod
    def list_daily_sentiment(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[DailySentiment]:
        """List daily sentiment in the given interval."""
        ...
