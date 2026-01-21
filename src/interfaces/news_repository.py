# src/interfaces/news_repository.py
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from src.entities.news_article import NewsArticle


class NewsRepository(ABC):
    @abstractmethod
    def get_latest_published_at(self, asset_id: str) -> datetime | None:
        """Maior published_at persistido para o ativo."""
        ...

    @abstractmethod
    def upsert_news_batch(self, articles: list[NewsArticle]) -> None:
        """Insere/atualiza (dedup por article_id)."""
        ...

    @abstractmethod
    def list_news(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[NewsArticle]:
        """Lê notícias persistidas no intervalo."""
        ...
