from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SentimentLabel(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class News:
    """Representa uma notícia em qualquer estágio: bruta ou inferida."""

    ticker: str
    published_at: datetime
    title: str
    source: str
    url: str
    sentiment: SentimentLabel | None = None
    confidence: float | None = None
