# src/adapters/finbert_sentiment_model.py

from __future__ import annotations

from typing import Iterable, List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.domain.time.utc import require_tz_aware
from src.entities.news_article import NewsArticle
from src.entities.scored_news_article import ScoredNewsArticle
from src.interfaces.sentiment_model import SentimentModel


class FinBERTSentimentModel(SentimentModel):
    """
    Financial sentiment inference using FinBERT.

    Input:
      - NewsArticle (domain)

    Output:
      - ScoredNewsArticle with sentiment_score ∈ [-1.0, +1.0]

    Score strategy:
      score = P(positive) - P(negative)
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)

        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            self.device
        )
        self.model.eval()

    @torch.no_grad()
    def infer(self, articles: List[NewsArticle]) -> List[ScoredNewsArticle]:
        if not articles:
            return []

        # Guard rails for the new "score raw dataset" pipeline:
        # - We rely on article_id for idempotent persistence in processed parquet.
        # - Time must be tz-aware for downstream joins/aggregation.
        for a in articles:
            if a.article_id is None or not str(a.article_id).strip():
                raise ValueError(
                    "NewsArticle.article_id is required for scoring (used as stable id)."
                )
            require_tz_aware(a.published_at, "published_at")

        texts = [self._build_text(a) for a in articles]
        scores = self._score_texts(texts)

        if len(scores) != len(articles):
            raise RuntimeError("Sentiment inference produced inconsistent output size")

        scored: list[ScoredNewsArticle] = []
        for article, score in zip(articles, scores):
            s = float(score)
            scored.append(
                ScoredNewsArticle(
                    article_id=str(article.article_id),
                    asset_id=article.asset_id,
                    published_at=article.published_at,
                    sentiment_score=s,
                    confidence=abs(s),
                    model_name=self.model_name,
                )
            )

        return scored

    @staticmethod
    def _build_text(article: NewsArticle) -> str:
        # Minimal policy: concat headline + summary.
        # If both are empty/whitespace, fallback to a neutral token.
        headline = (article.headline or "").strip()
        summary = (article.summary or "").strip()
        text = " ".join(p for p in (headline, summary) if p)
        return text if text else " "

    def _score_texts(self, texts: Iterable[str]) -> List[float]:
        batch_texts = list(texts)
        scores: list[float] = []

        for batch in self._batch(batch_texts):
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

            # FinBERT labels: [negative, neutral, positive]
            batch_scores = (probs[:, 2] - probs[:, 0]).detach().cpu().tolist()
            scores.extend(batch_scores)

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

# TODO(feature-engineering):
# Permitir que modelos retornem confidence explícita
# independente do sentiment_score (ex: entropy-based confidence)

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
