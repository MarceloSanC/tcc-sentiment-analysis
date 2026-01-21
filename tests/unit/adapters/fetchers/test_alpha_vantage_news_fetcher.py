# tests/unit/adapters/test_alpha_vantage_news_fetcher.py

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import pytest

from src.adapters.alpha_vantage_news_fetcher import AlphaVantageNewsFetcher


class _FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.last_url: Optional[str] = None
        self.last_params: Optional[Dict[str, Any]] = None
        self.last_headers: Optional[Dict[str, Any]] = None
        self.last_timeout: Optional[int] = None

    def get(self, url: str, params: Dict[str, Any], headers: Dict[str, Any], timeout: int):
        self.last_url = url
        self.last_params = params
        self.last_headers = headers
        self.last_timeout = timeout
        return _FakeResponse(self.payload, status_code=self.status_code)


def _dt_utc(y, m, d, hh=0, mm=0, ss=0):
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


def test_parse_time_published_supports_minute_and_second_formats():
    f = AlphaVantageNewsFetcher(api_key="k", session=_FakeSession({"feed": []}))

    assert f._parse_time_published("20240105T1150") == _dt_utc(2024, 1, 5, 11, 50)
    assert f._parse_time_published("20240123T094238") == _dt_utc(2024, 1, 23, 9, 42, 38)

    with pytest.raises(ValueError):
        f._parse_time_published("2024-01-01T00:00")  # formato errado


def test_fetch_company_news_builds_expected_request_and_parses_articles():
    payload = {
        "items": 2,
        "feed": [
            {
                "time_published": "20100105T1150",
                "title": "Apple launches something",
                "summary": "Details about the launch",
                "source": "Reuters",
                "url": "https://example.com/a",
            },
            {
                "time_published": "20100105T1200",
                "title": "Apple follow-up",
                "summary": "More details",
                "source": "Bloomberg",
                "url": "https://example.com/b",
            },
        ],
    }

    session = _FakeSession(payload)
    fetcher = AlphaVantageNewsFetcher(
        api_key="demo",
        session=session,
        timeout_seconds=12,
        user_agent="unit-test/1.0",
    )

    start = _dt_utc(2010, 1, 1, 0, 0)
    end = _dt_utc(2010, 1, 31, 23, 59)

    articles = fetcher.fetch_company_news("AAPL", start, end)

    # request contract
    assert session.last_url == AlphaVantageNewsFetcher.BASE_URL
    assert session.last_timeout == 12
    assert session.last_headers is not None
    assert session.last_headers["Accept"] == "application/json"
    assert session.last_headers["User-Agent"] == "unit-test/1.0"

    assert session.last_params is not None
    assert session.last_params["function"] == "NEWS_SENTIMENT"
    assert session.last_params["tickers"] == "AAPL"
    assert session.last_params["sort"] == "EARLIEST"
    assert session.last_params["limit"] == 1000
    assert session.last_params["apikey"] == "demo"
    # YYYYMMDDTHHMM (UTC)
    assert session.last_params["time_from"] == "20100101T0000"
    assert session.last_params["time_to"] == "20100131T2359"

    # parsed domain entities
    assert len(articles) == 2
    assert all(a.asset_id == "AAPL" for a in articles)
    assert all(a.published_at.tzinfo is not None for a in articles)

    assert articles[0].published_at == _dt_utc(2010, 1, 5, 11, 50)
    assert articles[0].headline == "Apple launches something"
    assert articles[0].summary == "Details about the launch"
    assert articles[0].source == "Reuters"
    assert articles[0].url == "https://example.com/a"
    assert articles[0].article_id == "https://example.com/a"
    assert articles[0].language == "en"


def test_fetch_company_news_requires_tz_aware_dates():
    session = _FakeSession({"feed": []})
    fetcher = AlphaVantageNewsFetcher(api_key="k", session=session)

    with pytest.raises(ValueError, match="timezone-aware"):
        fetcher.fetch_company_news(
            "AAPL",
            datetime(2010, 1, 1),  # naive
            _dt_utc(2010, 1, 2),
        )

    with pytest.raises(ValueError, match="timezone-aware"):
        fetcher.fetch_company_news(
            "AAPL",
            _dt_utc(2010, 1, 1),
            datetime(2010, 1, 2),  # naive
        )


def test_fetch_company_news_rejects_start_after_end():
    session = _FakeSession({"feed": []})
    fetcher = AlphaVantageNewsFetcher(api_key="k", session=session)

    with pytest.raises(ValueError, match="start_date must be <="):
        fetcher.fetch_company_news(
            "AAPL",
            _dt_utc(2010, 1, 3),
            _dt_utc(2010, 1, 2),
        )


def test_fetch_company_news_raises_on_rate_limit_note():
    session = _FakeSession({"Note": "Thank you for using Alpha Vantage! ..."})
    fetcher = AlphaVantageNewsFetcher(api_key="k", session=session)

    with pytest.raises(RuntimeError, match="rate limit"):
        fetcher.fetch_company_news("AAPL", _dt_utc(2010, 1, 1), _dt_utc(2010, 1, 2))


def test_fetch_company_news_raises_on_information_message():
    session = _FakeSession({"Information": "The parameter apikey is invalid"})
    fetcher = AlphaVantageNewsFetcher(api_key="k", session=session)

    with pytest.raises(RuntimeError, match="Information"):
        fetcher.fetch_company_news("AAPL", _dt_utc(2010, 1, 1), _dt_utc(2010, 1, 2))


def test_fetch_company_news_ignores_items_with_invalid_or_missing_time_published():
    payload = {
        "feed": [
            {
                "time_published": "INVALID",
                "title": "bad date",
                "summary": "x",
                "source": "s",
                "url": "https://example.com/bad",
            },
            {
                # missing time_published
                "title": "missing date",
                "summary": "x",
                "source": "s",
                "url": "https://example.com/missing",
            },
            {
                "time_published": "20100105T1150",
                "title": "ok",
                "summary": "ok",
                "source": "Reuters",
                "url": "https://example.com/ok",
            },
        ]
    }

    session = _FakeSession(payload)
    fetcher = AlphaVantageNewsFetcher(api_key="k", session=session)

    articles = fetcher.fetch_company_news("AAPL", _dt_utc(2010, 1, 1), _dt_utc(2010, 1, 10))
    assert len(articles) == 1
    assert articles[0].url == "https://example.com/ok"


def test_fetch_company_news_fallbacks_to_space_when_headline_and_summary_empty():
    payload = {
        "feed": [
            {
                "time_published": "20100105T1150",
                "title": "   ",
                "summary": "",
                "source": "",
                "url": None,
            }
        ]
    }

    session = _FakeSession(payload)
    fetcher = AlphaVantageNewsFetcher(api_key="k", session=session)

    articles = fetcher.fetch_company_news("AAPL", _dt_utc(2010, 1, 1), _dt_utc(2010, 1, 10))
    assert len(articles) == 1
    assert articles[0].headline == " "
    assert articles[0].summary == " "
    # source fallback
    assert articles[0].source == "alpha_vantage"
    # article_id fallback (no url)
    assert isinstance(articles[0].article_id, str) and articles[0].article_id
