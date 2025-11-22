from abc import ABC, abstractmethod

from src.entities.news import News


class SentimentModel(ABC):
    @abstractmethod
    def predict(self, text: str) -> News:
        """Dado um texto, retorna uma notícia inferida com sentimento e confiança."""
        pass
