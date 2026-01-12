# tests/unit/entities/test_daily_sentiment.py

from datetime import date, datetime

import pytest

from src.entities.daily_sentiment import DailySentiment


def test_daily_sentiment_valid_creation():
    """
    Deve criar a entidade corretamente quando todos os invariantes
    de domínio são respeitados.
    """
    sentiment = DailySentiment(
        asset_id="AAPL",
        day=date(2024, 1, 1),
        sentiment_score=0.25,
        n_articles=3,
        sentiment_std=0.1,
    )

    assert sentiment.asset_id == "AAPL"
    assert sentiment.day == date(2024, 1, 1)
    assert sentiment.sentiment_score == 0.25
    assert sentiment.n_articles == 3
    assert sentiment.sentiment_std == 0.1


@pytest.mark.parametrize("score", [-1.0, 0.0, 1.0])
def test_sentiment_score_boundary_values_are_allowed(score):
    """
    Scores exatamente nos limites do intervalo [-1, +1]
    devem ser aceitos.
    """
    DailySentiment(
        asset_id="AAPL",
        day=date(2024, 1, 1),
        sentiment_score=score,
        n_articles=1,
    )


@pytest.mark.parametrize("score", [-1.01, 1.01, 2.0])
def test_sentiment_score_out_of_range_raises(score):
    """
    Scores fora do intervalo permitido devem falhar explicitamente.
    """
    with pytest.raises(ValueError):
        DailySentiment(
            asset_id="AAPL",
            day=date(2024, 1, 1),
            sentiment_score=score,
            n_articles=1,
        )


def test_day_must_be_date_without_time():
    """
    Garante que a entidade usa apenas date (sem hora),
    evitando vazamento temporal.
    """
    with pytest.raises(TypeError):
        DailySentiment(
            asset_id="AAPL",
            day=datetime(2024, 1, 1, 10, 30),  # type: ignore
            sentiment_score=0.1,
            n_articles=1,
        )


def test_asset_id_is_required_and_cannot_be_empty():
    """
    asset_id é obrigatório para integridade do domínio.
    """
    with pytest.raises(TypeError):
        DailySentiment(  # type: ignore
            day=date(2024, 1, 1),
            sentiment_score=0.1,
            n_articles=1,
        )


@pytest.mark.parametrize("n_articles", [0, -1, -10])
def test_n_articles_must_be_positive(n_articles):
    """
    n_articles deve ser inteiro positivo.
    """
    with pytest.raises(ValueError):
        DailySentiment(
            asset_id="AAPL",
            day=date(2024, 1, 1),
            sentiment_score=0.1,
            n_articles=n_articles,
        )


def test_sentiment_std_cannot_be_negative():
    """
    Desvio padrão negativo não faz sentido estatístico.
    """
    with pytest.raises(ValueError):
        DailySentiment(
            asset_id="AAPL",
            day=date(2024, 1, 1),
            sentiment_score=0.1,
            n_articles=2,
            sentiment_std=-0.01,
        )


def test_daily_sentiment_is_immutable():
    """
    A entidade deve ser imutável para garantir
    rastreabilidade e reprodutibilidade.
    """
    sentiment = DailySentiment(
        asset_id="AAPL",
        day=date(2024, 1, 1),
        sentiment_score=0.2,
        n_articles=1,
    )

    with pytest.raises(Exception):
        sentiment.sentiment_score = 0.9  # type: ignore

# =========================
# TODOs — melhorias futuras
# =========================

# TODO(reproducibility):
# Avaliar se versões futuras devem permitir cópia controlada
# via método `.with_updates()` em vez de mutação direta.
