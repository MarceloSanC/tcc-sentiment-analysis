# src/adapters/yfinance_data_fetcher.py
import time
import logging
from datetime import datetime

import yfinance as yf

from src.entities.candle import Candle
from src.interfaces.candle_fetcher import CandleFetcher


logger = logging.getLogger(__name__) 

class YFinanceCandleFetcher(CandleFetcher):
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch_candles(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[Candle]:
        
        logger.info(
            "Fetching candles",
            extra={
                "symbol": symbol,
                "start": start.isoformat(),
                "end": end.isoformat(),
            },
        )

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

                # Normalizar MultiIndex
                if isinstance(df.columns, type(df.columns)) and df.columns.nlevels > 1:
                    df.columns = df.columns.get_level_values(0)

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

                logger.info(
                    "Candles fetched successfully",
                    extra={
                        "symbol": symbol,
                        "count": len(candles),
                    },
                )

                return candles

            except Exception as e:
                logger.warning(
                    "Fetch attempt failed",
                    extra={
                        "symbol": symbol,
                        "attempt": attempt + 1,
                        "error": str(e),
                    },
                )

                if attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2**attempt))
                    continue

                logger.error(
                    "Fetch failed after retries",
                    extra={"symbol": symbol},
                    exc_info=True,
                )

                raise RuntimeError(
                    f"Failed to fetch {symbol} after {self.max_retries} retries: {e}"
                ) from e
