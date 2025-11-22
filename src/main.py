from src.adapters.finbert_sentiment_model import FinBERTSentimentModel
from src.adapters.finnhub_news_fetcher import FinnhubNewsFetcher
from src.adapters.sqlite_news_repository import SQLiteNewsRepository
from src.use_cases.fetch_news_use_case import FetchNewsUseCase
from src.use_cases.infer_sentiment_use_case import InferSentimentUseCase

TICKERS = ["AAPL", "MSFT", "TSLA", "GOOGL", "NVDA"]
API_KEY = "d0ls2p9r01qpni3125ngd0ls2p9r01qpni3125o0"
DB_PATH = "data/tcc_sentiment.db"


def main():
    # Dependências
    fetcher = FinnhubNewsFetcher(api_key=API_KEY)
    repository = SQLiteNewsRepository(db_path=DB_PATH)
    sentiment_model = FinBERTSentimentModel()

    # Use Cases
    fetch_use_case = FetchNewsUseCase(fetcher, repository, days_back=60)
    infer_use_case = InferSentimentUseCase(repository, sentiment_model)

    # Execução
    print("📥 Etapa 1: Buscando notícias...")
    fetch_use_case.execute(TICKERS)

    print("\n🧠 Etapa 2: Inferindo sentimento...")
    infer_use_case.execute(TICKERS)

    print("\n✅ Pipeline concluído.")


if __name__ == "__main__":
    main()
