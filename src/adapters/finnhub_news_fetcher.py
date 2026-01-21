# src/adapters/finnhub_news_fetcher.py

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

import requests

from domain.time.utc import require_tz_aware
from src.entities.news_article import NewsArticle
from src.interfaces.news_fetcher import NewsFetcher

logger = logging.getLogger(__name__)


class FinnhubNewsFetcher(NewsFetcher):
    """
    Adapter responsável por buscar notícias financeiras da API Finnhub
    e convertê-las para entidades de domínio (NewsArticle).

    Contrato temporal:
    - start_date/end_date devem ser timezone-aware
    - published_at gerado sempre timezone-aware em UTC
    """

    BASE_URL = "https://finnhub.io/api/v1/company-news"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 10,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._session = session or requests.Session()

    def fetch_company_news(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
        limit: Optional[int] = None,
    ) -> List[NewsArticle]:
        """
        Busca notícias relacionadas a um ativo em um intervalo de datas.

        Args:
            asset_id: ticker do ativo (ex: AAPL)
            start_date: data inicial (timezone-aware)
            end_date: data final (timezone-aware)
            limit: número máximo de notícias (opcional)

        Returns:
            Lista de NewsArticle (domínio)
        """
        require_tz_aware(start_date, "start_date")
        require_tz_aware(end_date, "end_date")

        start_utc = start_date.astimezone(timezone.utc)
        end_utc = end_date.astimezone(timezone.utc)

        if start_utc > end_utc:
            raise ValueError("start_date must be <= end_date")

        params = {
            "symbol": asset_id,
            "from": start_utc.date().isoformat(),
            "to": end_utc.date().isoformat(),
            "token": self.api_key,
        }

        logger.info(
            "Fetching news from Finnhub",
            extra={
                "asset": asset_id,
                "from": params["from"],
                "to": params["to"],
                "limit": limit,
            },
        )

        response = self._session.get(
            self.BASE_URL,
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        raw_articles = response.json()
        if not isinstance(raw_articles, list):
            raise ValueError("Unexpected Finnhub response format: expected list")

        if limit is not None:
            raw_articles = raw_articles[: max(0, int(limit))]

        articles: List[NewsArticle] = []

        for raw in raw_articles:
            ts = raw.get("datetime")
            if ts is None:
                continue

            published_at = datetime.fromtimestamp(int(ts), tz=timezone.utc)

            headline = (raw.get("headline") or "").strip()
            summary = (raw.get("summary") or "").strip()
            source = (raw.get("source") or "").strip()

            # fallback para reduzir risco de strings vazias
            if not headline and not summary:
                headline = " "
                summary = " "

            article_id = str(raw.get("id") or ts)

            articles.append(
                NewsArticle(
                    article_id=article_id,
                    asset_id=asset_id,
                    published_at=published_at,
                    headline=headline,
                    summary=summary,
                    source=source or "finnhub",
                    url=raw.get("url"),
                    language="en",
                )
            )

        logger.info(
            "Finnhub news fetched",
            extra={"asset": asset_id, "count": len(articles)},
        )

        return articles


# =========================
# TODOs — melhorias futuras
# =========================

# TODO(data-pipeline):
# Suportar persistência incremental de candles
# (append ou upsert por timestamp)

# TODO(data-pipeline):
# Implementar deduplicação temporal
# (manter último candle por timestamp)

# TODO(architecture):
# Expor política explícita de persistência:
# overwrite | append | upsert

# TODO(stat-validation):
# Validar gaps temporais excessivos
# (ex: dias úteis faltantes)

# TODO(reproducibility):
# Versionar datasets de candles persistidos
# (ex: hash do arquivo + metadata JSON)