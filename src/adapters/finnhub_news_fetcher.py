# src/adapters/finnhub_news_fetcher.py

from datetime import datetime, timezone
from typing import List, Optional
import requests

from src.entities.news_article import NewsArticle


class FinnhubNewsFetcher:
    """
    Adapter responsável por buscar notícias financeiras da API Finnhub
    e convertê-las para entidades de domínio (NewsArticle).
    """

    BASE_URL = "https://finnhub.io/api/v1/company-news"

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 10,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

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
            asset_id: ticker do ativo (ex: PETR4.SA)
            start_date: data inicial (timezone-aware)
            end_date: data final (timezone-aware)
            limit: número máximo de notícias (opcional)

        Returns:
            Lista de NewsArticle (domínio)
        """

        params = {
            "symbol": asset_id,
            "from": start_date.date().isoformat(),
            "to": end_date.date().isoformat(),
            "token": self.api_key,
        }

        response = requests.get(
            self.BASE_URL,
            params=params,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        raw_articles = response.json()

        articles: List[NewsArticle] = []

        for raw in raw_articles[:limit]:
            published_at = datetime.fromtimestamp(
                raw["datetime"],
                tz=timezone.utc,
            )

            articles.append(
                NewsArticle(
                    article_id=str(raw.get("id", raw["datetime"])),
                    asset_id=asset_id,
                    published_at=published_at,
                    title=raw.get("headline", ""),
                    content=raw.get("summary", ""),
                    source=raw.get("source"),
                    url=raw.get("url"),
                )
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