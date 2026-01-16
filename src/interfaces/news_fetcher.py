# src/interfaces/news_fetcher.py
from abc import ABC, abstractmethod
from datetime import datetime

from src.entities.news_article import NewsArticle


class NewsFetcher(ABC):
    """
    Interface para obtenção de notícias de fontes externas.
    Abstrai chamadas a serviços terceiros (ex: Finnhub, Twitter).
    Implementações concretas lidam com protocolos, autenticação e parsing de respostas.
    """

    @abstractmethod
    def fetch_company_news(
        self, ticker: str, start_date: datetime, end_date: datetime
    ) -> list[NewsArticle]:
        """Busca notícias da fonte externa (ex: Finnhub)."""
        ...
