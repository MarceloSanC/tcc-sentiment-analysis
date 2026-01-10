# src/interfaces/news_repository.py
from abc import ABC, abstractmethod
from datetime import datetime

from src.entities.news import News


class NewsRepository(ABC):
    @abstractmethod
    def get_latest_news_date(self, ticker: str) -> datetime | None:
        ...

    @abstractmethod
    def get_unprocessed_news(self, ticker: str) -> list[News]:
        """Retorna notícias com sentiment == None."""
        ...

    @abstractmethod
    def save_news_batch(self, news_list: list[News]) -> None:
        """Salva ou atualiza notícias (incluindo preenchimento de sentiment/confidence)."""
        ...
