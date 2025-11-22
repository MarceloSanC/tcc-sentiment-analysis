from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.entities.news import InferredNews, SentimentLabel
from src.interfaces.sentiment_model import SentimentModel


class FinBERTSentimentModel(SentimentModel):
    def __init__(self, model_name: str = "yiyanghkust/finbert-tone"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model,
            tokenizer=self.tokenizer,
            return_all_scores=False,
        )

    def predict(self, text: str) -> InferredNews:
        # Garantir que o texto seja string válida
        if not isinstance(text, str) or not text.strip():
            text = "."

        result = self.pipeline(text)[0]  # {"label": "positive", "score": 0.95}

        # Normalizar label
        label_map = {
            "positive": SentimentLabel.POSITIVE,
            "negative": SentimentLabel.NEGATIVE,
            "neutral": SentimentLabel.NEUTRAL,
        }

        sentiment = label_map.get(result["label"].lower(), SentimentLabel.NEUTRAL)
        confidence = float(result["score"])

        # Retorna InferredNews *sem* ticker/data — isso será preenchido pelo Use Case
        return InferredNews(
            ticker="",  # será sobrescrito
            published_at=None,  # será sobrescrito
            title="",  # será sobrescrito
            source="",
            url="",
            sentiment=sentiment,
            confidence=confidence,
        )
