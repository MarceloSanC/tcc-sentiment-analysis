from datetime import datetime, timedelta

from src.interfaces.news_fetcher import NewsFetcher
from src.interfaces.news_repository import NewsRepository


class FetchNewsUseCase:
    def __init__(
        self,
        news_fetcher: NewsFetcher,
        news_repository: NewsRepository,
        days_back: int = 60,
    ):
        self.news_fetcher: NewsFetcher = news_fetcher
        self.news_repository: NewsRepository = news_repository
        self.days_back: int = days_back

    def execute(self, tickers: list[str]) -> None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_back)

        for ticker in tickers:
            print(
                f"\nBuscando notícias de {ticker}, de {start_date.date()} a {end_date.date()}"
            )

            # Verifica última data salva
            last_saved = self.news_repository.get_latest_news_date(ticker)
            effective_start = max(start_date, last_saved) if last_saved else start_date

            # Busca somente o necessário
            news_list = self.news_fetcher.fetch_company_news(ticker, effective_start, end_date)

            if news_list:
                self.news_repository.save_news_batch(news_list)
                print(f"✅ {len(news_list)} novas notícias salvas para {ticker}.")
            else:
                print(f"Nenhuma nova notícia para {ticker}.")
