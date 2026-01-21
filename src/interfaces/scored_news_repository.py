from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from src.entities.scored_news_article import ScoredNewsArticle


class ScoredNewsRepository(ABC):
    @abstractmethod
    def get_latest_published_at(self, asset_id: str) -> datetime | None:
        """Latest published_at persisted for the asset."""
        ...

    @abstractmethod
    def upsert_scored_news_batch(self, articles: list[ScoredNewsArticle]) -> None:
        """Insert or update scored news (dedup by article_id)."""
        ...

    @abstractmethod
    def list_scored_news(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[ScoredNewsArticle]:
        """List scored news in the given interval."""
        ...

    @abstractmethod
    def list_article_ids(self, asset_id: str) -> set[str]:
        """Return scored article ids for skip logic."""
        ...
