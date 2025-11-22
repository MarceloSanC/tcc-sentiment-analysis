from datetime import datetime

import requests

from src.entities.news import News
from src.interfaces.news_fetcher import NewsFetcher


class FinnhubNewsFetcher(NewsFetcher):
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_news(
        self, ticker: str, start_date: datetime, end_date: datetime
    ) -> list[News]:
        if ticker.startswith(("BINANCE:", "COINBASE:")):
            url = "https://finnhub.io/api/v1/crypto-news"
            params = {"token": self.api_key}
        elif ticker == "SP500":
            url = "https://finnhub.io/api/v1/news"
            params = {"category": "general", "token": self.api_key}
        else:
            url = "https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": ticker,
                "from": start_date.strftime("%Y-%m-%d"),
                "to": end_date.strftime("%Y-%m-%d"),
                "token": self.api_key,
            }

        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Erro ao buscar {ticker}: {response.status_code}")
            return (
                []
            )  # Em produção, poderia lançar uma exceção customizada (ex: NewsFetchError)

        try:
            data = response.json()
        except ValueError:
            print(f"Erro ao decodificar JSON para {ticker}.")
            return []

        if not isinstance(data, list):
            return []

        news_list = []
        for item in data:
            try:
                published_at = datetime.fromtimestamp(item["datetime"])
                news = News(
                    ticker=ticker,
                    published_at=published_at,
                    title=item["headline"],
                    source=item["source"],
                    url=item["url"],
                )
                news_list.append(news)
            except (KeyError, ValueError):
                continue  # ignora itens malformados

        return news_list
