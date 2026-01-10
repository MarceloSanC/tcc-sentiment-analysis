# tests/unit/entities/test_scored_news_article.py

import pytest
from datetime import datetime, timezone

from src.entities.scored_news_article import ScoredNewsArticle


class TestScoredNewsArticle:
    """
    Testes unitários para a entidade de domínio ScoredNewsArticle.

    Foco:
    - Invariantes de domínio
    - Validações explícitas
    - Comportamento determinístico
    """

    def test_create_valid_article_with_minimal_fields(self):
        """
        Deve permitir criação válida quando todos os campos obrigatórios
        estão corretos e metadados opcionais são omitidos.
        """
        article = ScoredNewsArticle(
            article_id="news_001",
            asset_id="AAPL",
            published_at=datetime(2024, 1, 10, tzinfo=timezone.utc),
            sentiment_score=0.42,
        )

        assert article.article_id == "news_001"
        assert article.asset_id == "AAPL"
        assert article.sentiment_score == 0.42
        assert article.confidence is None
        assert article.model_name is None

    @pytest.mark.parametrize("score", [-1.0, 0.0, 1.0])
    def test_sentiment_score_boundary_values_are_allowed(self, score: float):
        """
        Valores de fronteira do intervalo [-1, +1] devem ser aceitos.
        """
        article = ScoredNewsArticle(
            article_id="news_boundary",
            asset_id="MSFT",
            published_at=datetime.now(tz=timezone.utc),
            sentiment_score=score,
        )

        assert article.sentiment_score == score

    @pytest.mark.parametrize("score", [-1.1, 1.1, 2.0, -5.0])
    def test_sentiment_score_out_of_range_raises_error(self, score: float):
        """
        Qualquer score fora do intervalo [-1, +1] deve falhar.
        """
        with pytest.raises(ValueError, match="sentiment_score must be in the range"):
            ScoredNewsArticle(
                article_id="news_invalid",
                asset_id="GOOGL",
                published_at=datetime.now(tz=timezone.utc),
                sentiment_score=score,
            )

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_valid_confidence_values_are_allowed(self, confidence: float):
        """
        Confidence, quando presente, deve estar no intervalo [0, 1].
        """
        article = ScoredNewsArticle(
            article_id="news_confidence",
            asset_id="NVDA",
            published_at=datetime.now(tz=timezone.utc),
            sentiment_score=0.1,
            confidence=confidence,
        )

        assert article.confidence == confidence

    @pytest.mark.parametrize("confidence", [-0.1, 1.1, 2.0])
    def test_invalid_confidence_raises_error(self, confidence: float):
        """
        Confidence fora do intervalo permitido deve gerar erro.
        """
        with pytest.raises(ValueError, match="confidence must be in the range"):
            ScoredNewsArticle(
                article_id="news_invalid_confidence",
                asset_id="TSLA",
                published_at=datetime.now(tz=timezone.utc),
                sentiment_score=0.2,
                confidence=confidence,
            )

    def test_entity_is_immutable(self):
        """
        Entidades de domínio devem ser imutáveis.
        """
        article = ScoredNewsArticle(
            article_id="news_immutable",
            asset_id="AMZN",
            published_at=datetime.now(tz=timezone.utc),
            sentiment_score=0.3,
        )

        with pytest.raises(Exception):
            article.sentiment_score = 0.9


# =========================
# TODOs — melhorias futuras
# =========================

# TODO(modeling):
# Testar comportamento ao introduzir metadado `language`
# e validar compatibilidade com pipelines multilíngues.

# TODO(feature-engineering):
# Testar uso de `confidence` como peso
# quando integrado ao SentimentAggregator.

# TODO(stat-validation):
# Introduzir teste de consistência temporal
# quando houver associação explícita com candles.

# TODO(reproducibility):
# Adicionar teste de unicidade e rastreabilidade
# via content_hash quando o campo for introduzido.
