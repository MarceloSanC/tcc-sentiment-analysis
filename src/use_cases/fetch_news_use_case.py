# src/use_cases/fetch_news_use_case.py
import logging
from datetime import datetime, timedelta, timezone

from src.interfaces.news_fetcher import NewsFetcher
from src.interfaces.news_repository import NewsRepository

logger = logging.getLogger(__name__)


class FetchNewsUseCase:
    def __init__(
        self,
        news_fetcher: NewsFetcher,
        news_repository: NewsRepository,
        days_back: int = 60,
    ) -> None:
        self.news_fetcher = news_fetcher
        self.news_repository = news_repository
        self.days_back = days_back

    def execute(self, tickers: list[str]) -> None:
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=self.days_back)

        for ticker in tickers:
            logger.info(
                "Fetching news window",
                extra={
                    "asset": ticker,
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
            )

            last_saved = self.news_repository.get_latest_news_date(ticker)
            effective_start = max(start_date, last_saved) if last_saved else start_date

            news_list = self.news_fetcher.fetch_company_news(
                ticker, effective_start, end_date
            )

            if news_list:
                self.news_repository.save_news_batch(news_list)
                logger.info(
                    "News saved",
                    extra={"asset": ticker, "count": len(news_list)},
                )
            else:
                logger.info("No new news", extra={"asset": ticker})
