# src/interfaces/sentiment_model.py

from abc import ABC, abstractmethod
from typing import List

from src.entities.news_article import NewsArticle
from src.entities.scored_news_article import ScoredNewsArticle


class SentimentModel(ABC):
    @abstractmethod
    def infer(
        self,
        articles: List[NewsArticle],
    ) -> List[ScoredNewsArticle]:
        """
        Infere sentimento quantitativo para cada notícia.
        """
        ...