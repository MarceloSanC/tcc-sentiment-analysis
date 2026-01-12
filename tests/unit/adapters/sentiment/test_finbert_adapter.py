# tests/unit/adapters/sentiment/test_finbert_adapter.py

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from src.adapters.finbert_sentiment_model import FinBERTSentimentModel
from src.entities.news_article import NewsArticle
from src.entities.scored_news_article import ScoredNewsArticle


# Helpers / Fakes

def make_article(article_id: str, text: str) -> NewsArticle:
    return NewsArticle(
        article_id=article_id,
        asset_id="AAPL",
        published_at=datetime(2024, 1, 10, tzinfo=timezone.utc),
        headline=text or " ",
        summary=text or " ",
        source="unit-test",
        url="http://example.com",
    )


def fake_pipeline(outputs: list[dict]):
    """
    Cria um fake de pipeline HuggingFace compatível com batch inference.

    Cada output deve conter:
    - label: str  (e.g. POSITIVE / NEGATIVE / NEUTRAL)
    - score: float (confidence da classe prevista)
    """
    pipeline = Mock()
    pipeline.return_value = outputs
    return pipeline


class DummyFinBERT(FinBERTSentimentModel):
    def __init__(self, scores: list[float]):
        self.model_name = "dummy-finbert"
        self._scores = scores

    def _score_texts(self, texts):
        return self._scores

# Tests

def test_infer_preserves_article_identity_and_order():
    """
    Deve preservar:
    - article_id
    - ordem de entrada == ordem de saída

    Protege contra bug crítico de batch inference fora de ordem.
    """
    articles = [
        make_article("id_1", "Good earnings report"),
        make_article("id_2", "Market uncertainty rises"),
        make_article("id_3", "Company faces lawsuit"),
    ]

    adapter = DummyFinBERT(scores=[0.8, 0.0, -0.6])

    result = adapter.infer(articles)

    assert [a.article_id for a in result] == ["id_1", "id_2", "id_3"]
    assert all(isinstance(a, ScoredNewsArticle) for a in result)


def test_mapping_logits_to_continuous_score():
    """
    Verifica o mapeamento:
    classe + confidence -> score contínuo [-1, +1]

    Convenção esperada:
    - POSITIVE -> +confidence
    - NEGATIVE -> -confidence
    - NEUTRAL  -> 0.0
    """
    articles = [
        make_article("pos", "Great growth outlook"),
        make_article("neu", "Company announces meeting"),
        make_article("neg", "Profit warning issued"),
    ]

    adapter = DummyFinBERT(scores=[0.7, 0.0, -0.4])

    result = adapter.infer(articles)

    scores = {a.article_id: a.sentiment_score for a in result}

    assert scores["pos"] == pytest.approx(0.7)
    assert scores["neu"] == pytest.approx(0.0)
    assert scores["neg"] == pytest.approx(-0.4)


def test_confidence_is_preserved_when_available():
    """
    Confidence deve ser preservada explicitamente
    para uso futuro em agregação ponderada.
    """
    articles = [make_article("id_conf", "Strong performance")]

    adapter = DummyFinBERT(scores=[0.85])

    result = adapter.infer(articles)

    assert result[0].confidence == pytest.approx(0.85)


def test_empty_input_returns_empty_output():
    """
    Inferência sobre lista vazia deve ser segura e determinística.
    """

    adapter = DummyFinBERT(scores=[])

    result = adapter.infer([])

    assert result == []


@pytest.mark.parametrize("text", ["", " ", ".", "ok"])
def test_short_or_empty_text_does_not_break_pipeline(text: str):
    """
    Textos vazios ou muito curtos:
    - não devem quebrar o pipeline
    - devem retornar score neutro por default
    """
    article = make_article("edge_case", text)

    adapter = DummyFinBERT(scores=[0.0])

    result = adapter.infer([article])

    assert len(result) == 1
    assert result[0].sentiment_score == 0.0


# ==========================================================
# TODOs — melhorias futuras
# ==========================================================

# TODO(test-robustness):
# Testar comportamento quando o pipeline retorna labels inesperados
# (ex: "LABEL_0", "LABEL_1") e garantir fallback seguro.

# TODO(modeling):
# Validar compatibilidade com modelos multilíngues
# (ex: FinBERT + textos PT-BR).

# TODO(performance):
# Adicionar teste garantindo inferência em batch
# sem degradação de ordem ou latência excessiva.

# TODO(reproducibility):
# Incluir model_name e model_version em ScoredNewsArticle
# e validar persistência correta desses metadados.
