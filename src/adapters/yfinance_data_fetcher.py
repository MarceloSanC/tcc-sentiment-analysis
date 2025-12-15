# src/adapters/yfinance_data_fetcher.py
import time

from datetime import datetime

import yfinance as yf

from src.entities.candle import Candle
from src.interfaces.data_fetcher import DataFetcher


class YFinanceDataFetcher(DataFetcher):
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch_candles(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[Candle]:
        for attempt in range(self.max_retries + 1):
            try:
                # YFinance espera strings no formato YYYY-MM-DD
                df = yf.download(
                    symbol,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval="1d",
                    progress=False,
                    timeout=10,
                )
                if df.empty:
                    raise ValueError(f"No data returned for {symbol}")

                # Validar schema mínimo
                required_cols = {"Open", "High", "Low", "Close", "Volume"}
                if not required_cols.issubset(df.columns):
                    raise ValueError(f"Missing columns in response: {df.columns}")

                candles = []
                for idx, row in df.iterrows():
                    # idx é Timestamp → converter para datetime
                    ts = idx.to_pydatetime()
                    candle = Candle(
                        timestamp=ts,
                        open=float(row["Open"]),
                        high=float(row["High"]),
                        low=float(row["Low"]),
                        close=float(row["Close"]),
                        volume=int(row["Volume"]),
                    )
                    candles.append(candle)
                return candles

            except Exception as e:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))  # backoff exponencial
                    continue
                raise RuntimeError(
                    f"Failed to fetch {symbol} after {self.max_retries} retries: {e}"
                ) from e
