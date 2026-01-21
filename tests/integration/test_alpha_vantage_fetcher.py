# tests/integration/test_alpha_vantage_fetcher.py

from __future__ import annotations

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
import pytest

from src.adapters.alpha_vantage_news_fetcher import AlphaVantageNewsFetcher

load_dotenv()

def _dt_utc(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    not os.getenv("ALPHAVANTAGE_API_KEY"),
    reason="Requires ALPHAVANTAGE_API_KEY in environment",
)
def test_alpha_vantage_fetcher_real_smoke():
    """
    Real API smoke test.

    Goals (integration-level):
    - Auth + HTTP OK
    - Response shape contains 'feed'
    - Parse time_published into tz-aware UTC datetimes
    - Returned objects satisfy NewsArticle invariants
    """
    api_key = os.environ["ALPHAVANTAGE_API_KEY"]

    fetcher = AlphaVantageNewsFetcher(api_key=api_key, timeout_seconds=60)

    # janela curta para reduzir payload e risco de rate limit
    start = _dt_utc(2025, 1, 1, 0, 0)
    end = _dt_utc(2025, 1, 7, 23, 59)

    articles = fetcher.fetch_company_news("AAPL", start, end)

    # Não garante sempre >0 (pode variar), mas garante contrato do retorno
    assert isinstance(articles, list)

    for a in articles[:10]:  # limita validação
        assert a.asset_id == "AAPL"
        assert a.published_at.tzinfo is not None
        assert isinstance(a.headline, str)
        assert isinstance(a.summary, str)
        assert isinstance(a.source, str) and a.source.strip()
        if a.url is not None:
            assert a.url.startswith(("http://", "https://"))
