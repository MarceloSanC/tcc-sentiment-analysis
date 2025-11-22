import sqlite3

from datetime import datetime
from pathlib import Path

from src.entities.news import News
from src.interfaces.news_repository import NewsRepository


class SQLiteNewsRepository(NewsRepository):
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    title TEXT NOT NULL,
                    source TEXT,
                    url TEXT UNIQUE,
                    sentiment TEXT CHECK(sentiment IN ('Positive','Negative','Neutral')),
                    confidence REAL
                )
            """
            )

    def get_latest_news_date(self, ticker: str) -> datetime | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT MAX(published_at) FROM raw_news WHERE ticker = ?", (ticker,)
            )
            result = cursor.fetchone()[0]
            if result:
                return datetime.fromisoformat(result)
            return None

    def save_news_batch(self, news_list: list[News]) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for news in news_list:
                cursor.execute(
                    """
                    INSERT INTO news
                    (ticker, published_at, title, source, url, sentiment, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        sentiment = excluded.sentiment,
                        confidence = excluded.confidence
                """,
                    (
                        news.ticker,
                        news.published_at.isoformat(),
                        news.title,
                        news.source,
                        news.url,
                        news.sentiment.value if news.sentiment else None,
                        news.confidence,
                    ),
                )
            conn.commit()

    def get_unprocessed_news(self, ticker: str) -> list[News]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT ticker, published_at, title, source, url, sentiment, confidence
                FROM news
                WHERE ticker = ? AND sentiment IS NULL
            """,
                (ticker,),
            )
            rows = cursor.fetchall()
            return [
                News(
                    ticker=r[0],
                    published_at=datetime.fromisoformat(r[1]),
                    title=r[2],
                    source=r[3],
                    url=r[4],
                    sentiment=None,  # garantido por WHERE
                    confidence=None,
                )
                for r in rows
            ]
