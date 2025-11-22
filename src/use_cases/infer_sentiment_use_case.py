
from src.entities.news import News
from src.interfaces.news_repository import NewsRepository
from src.interfaces.sentiment_model import SentimentModel


class InferSentimentUseCase:
    def __init__(
        self, news_repository: NewsRepository, sentiment_model: SentimentModel
    ):
        self.news_repository = news_repository
        self.sentiment_model = sentiment_model

    def execute(self, tickers: list[str]) -> None:
        for ticker in tickers:
            print(f"\nInferindo sentimento para notícias de {ticker}...")

            # Obtém notícias brutas (não inferidas)
            raw_news_list = self.news_repository.get_unprocessed_news(ticker)

            if not raw_news_list:
                print(f"Nenhuma notícia bruta encontrada para {ticker}.")
                continue

            inferred_list = []
            for raw in raw_news_list:
                try:
                    inferred = self.sentiment_model.predict(raw.title)
                    # Garante que o ticker e metadata sejam preservados
                    inferred_news = News(
                        ticker=raw.ticker,
                        published_at=raw.published_at,
                        title=raw.title,
                        source=raw.source,
                        url=raw.url,
                        sentiment=inferred.sentiment,
                        confidence=inferred.confidence,
                    )
                    inferred_list.append(inferred_news)
                except Exception as e:
                    print(f"❌ Erro ao processar notícia '{raw.title[:50]}...': {e}")

            if inferred_list:
                self.news_repository.save_news_batch(inferred_list)
                print(
                    f"✅ {len(inferred_list)} notícias inferidas e salvas para {ticker}."
                )
            else:
                print(f"⚠️ Nenhuma notícia foi inferida com sucesso para {ticker}.")
