# src/adapters/alpha_vantage_news_fetcher.py
from __future__ import annotations

import logging
import re
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional

import requests

from domain.time.utc import require_tz_aware
from src.entities.news_article import NewsArticle
from src.interfaces.news_fetcher import NewsFetcher

logger = logging.getLogger(__name__)

_TIME_PUBLISHED_RE = re.compile(r"^\d{8}T\d{4}(\d{2})?$")  # YYYYMMDDTHHMM[SS]

class AlphaVantageNewsFetcher(NewsFetcher):
    """
    Adapter responsável por buscar notícias da API Alpha Vantage (NEWS_SENTIMENT)
    e convertê-las para entidades de domínio (NewsArticle).

    Contrato temporal:
    - start_date/end_date devem ser timezone-aware
    - published_at gerado sempre timezone-aware em UTC
    """

    BASE_URL = "https://www.alphavantage.co/query"
    _MIN_INTERVAL = 1.1  # segundos (1 req/s)

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 30,
        session: Optional[requests.Session] = None,
        user_agent: str = "tcc-sentiment-analysis/1.0",
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._session = session or requests.Session()
        self._user_agent = user_agent
        self._last_request_ts: float | None = None
        self._lock = threading.Lock()

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()

            if self._last_request_ts is not None:
                elapsed = now - self._last_request_ts
                sleep_for = self._MIN_INTERVAL - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

            self._last_request_ts = time.monotonic()

    @staticmethod
    def _parse_time_published(value: str) -> datetime:
        """
        Parse Alpha Vantage time_published:
        - YYYYMMDDTHHMM
        - YYYYMMDDTHHMMSS
        Returns timezone-aware UTC datetime.
        """
        v = (value or "").strip()

        if not _TIME_PUBLISHED_RE.match(v):
            raise ValueError(f"Unsupported time_published format: {value!r}")

        if len(v) == 13:  # YYYYMMDDTHHMM
            dt = datetime.strptime(v, "%Y%m%dT%H%M")
            return dt.replace(tzinfo=timezone.utc)

        if len(v) == 15:  # YYYYMMDDTHHMMSS
            dt = datetime.strptime(v, "%Y%m%dT%H%M%S")
            return dt.replace(tzinfo=timezone.utc)

        # (por segurança, embora regex + len já cubram)
        raise ValueError(f"Unsupported time_published length: {value!r}")

    def fetch_company_news(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
    ) -> List[NewsArticle]:
        require_tz_aware(start_date, "start_date")
        require_tz_aware(end_date, "end_date")

        start_utc = start_date.astimezone(timezone.utc)
        end_utc = end_date.astimezone(timezone.utc)

        if start_utc > end_utc:
            raise ValueError("start_date must be <= end_date")

        # Alpha Vantage usa time_from/time_to em formato: YYYYMMDDTHHMM
        time_from = start_utc.strftime("%Y%m%dT%H%M")
        time_to = end_utc.strftime("%Y%m%dT%H%M")

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": time_from,
            "time_to": time_to,
            "sort": "EARLIEST",  # ajuda a auditar cobertura
            "limit": 1000,
            "apikey": self.api_key,
        }

        headers = {
            "Accept": "application/json",
            "User-Agent": self._user_agent,
        }
        
        self._throttle()
        response = self._session.get(
            self.BASE_URL,
            params=params,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Unexpected Alpha Vantage response format: expected dict")

        # Rate limit / mensagens
        if "Note" in data:
            raise RuntimeError(f"Alpha Vantage rate limit hit: {data['Note']}")
        if "Information" in data:
            raise RuntimeError(f"Alpha Vantage Information: {data['Information']}")

        feed = data.get("feed")
        if feed is None:
            raise ValueError(f"Unexpected response shape. Keys: {list(data.keys())}")
        if not isinstance(feed, list):
            raise ValueError("Unexpected Alpha Vantage response: 'feed' is not a list")

        articles: List[NewsArticle] = []

        for item in feed:
            if not isinstance(item, dict):
                continue

            time_published = item.get("time_published")
            if not time_published:
                continue

            try:
                published_at = self._parse_time_published(str(time_published))
            except ValueError:
                # se vier inválido, ignora item para não quebrar pipeline
                continue

            headline = (item.get("title") or "").strip()
            summary = (item.get("summary") or "").strip()
            source = (item.get("source") or "").strip()
            url = item.get("url")

            # fallback: evita strings vazias
            if not headline and not summary:
                headline = " "
                summary = " "

            # ID estável: url > time+title
            article_id = str(url or f"{time_published}:{headline[:80]}")

            articles.append(
                NewsArticle(
                    article_id=article_id,
                    asset_id=ticker,
                    published_at=published_at,
                    headline=headline,
                    summary=summary,
                    source=source or "alpha_vantage",
                    url=url,
                    language="en",
                )
            )

        logger.info(
            "Alpha Vantage fetched %d news for %s | period=%s -> %s",
            len(articles),
            ticker,
            time_from,
            time_to,
            extra={
                "asset": ticker,
                "time_from": time_from,
                "time_to": time_to,
                "limit": params["limit"],
            },
        )

        return articles
