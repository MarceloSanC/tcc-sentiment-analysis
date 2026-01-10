# src/adapters/finbert_sentiment_model.py

from __future__ import annotations

from typing import Iterable, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

from src.entities.news_article import NewsArticle
from src.entities.scored_news_article import ScoredNewsArticle
from src.interfaces.sentiment_model import SentimentModel


class FinBERTSentimentModel(SentimentModel):
    """
    Adapter responsável por inferência de sentimento financeiro
    utilizando FinBERT.

    Entrada:
        - NewsArticle (domínio)

    Saída:
        - ScoredNewsArticle com score normalizado ∈ [-1.0, +1.0]

    Encapsula completamente HuggingFace / transformers.
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def infer(
        self,
        articles: List[NewsArticle],
    ) -> List[ScoredNewsArticle]:
        """
        Infere sentimento para uma lista de NewsArticle.

        Estratégia:
            score = P(positive) - P(negative)
        """

        if not articles:
            return []

        texts = [a.content or a.title for a in articles]

        scores = self._score_texts(texts)

        if len(scores) != len(articles):
            raise RuntimeError(
                "Sentiment inference produced inconsistent output size"
            )

        scored: list[ScoredNewsArticle] = []

        for article, score in zip(articles, scores):
            scored.append(
                ScoredNewsArticle(
                    article_id=article.article_id,
                    asset_id=article.asset_id,
                    published_at=article.published_at,
                    sentiment_score=float(score),
                    model_name=self.model_name,
                )
            )

        return scored

    def _score_texts(self, texts: Iterable[str]) -> List[float]:
        texts = list(texts)

        scores: list[float] = []

        for batch in self._batch(texts):
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            logits = outputs.logits

            probs = torch.softmax(logits, dim=1).cpu().numpy()

            # FinBERT labels: [negative, neutral, positive]
            batch_scores = probs[:, 2] - probs[:, 0]

            scores.extend(batch_scores.tolist())

        return scores

    def _batch(self, texts: List[str]) -> Iterable[List[str]]:
        for i in range(0, len(texts), self.batch_size):
            yield texts[i : i + self.batch_size]


# =========================
# TODOs — melhorias futuras
# =========================

# TODO (CleanArch):
# Centralizar política de seleção de texto
# para inferência de sentimento em método da entidade
# ou Value Object dedicado (ex: SentimentInputText)

# TODO(feature-engineering):
# Retornar distribuição completa (neg / neu / pos)
# e permitir múltiplas estratégias de agregação
# (ex: weighted_pos, entropy, polarity_strength)

# TODO(normalization & leakage):
# Calibrar scores por ativo usando z-score rolling
# para evitar viés cross-asset

# TODO(performance):
# Implementar cache de inferência por hash de texto
# (útil quando múltiplos ativos compartilham notícias)

# TODO(performance):
# Suporte a inferência em lote via DataLoader
# com pin_memory e prefetching

# TODO(stat-validation):
# Validar estabilidade temporal do score
# (ex: rolling mean / drift detection)
