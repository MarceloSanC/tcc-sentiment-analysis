# src/adapters/yfinance_candle_fetcher.py

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone, time

import yfinance as yf

from domain.time.utc import require_tz_aware, to_utc
from src.entities.candle import Candle
from src.interfaces.candle_fetcher import CandleFetcher

logger = logging.getLogger(__name__)


class YFinanceCandleFetcher(CandleFetcher):
    """
    Adapter responsável por buscar candles via yfinance.

    Contrato temporal:
    - start/end devem ser timezone-aware
    - Candle.timestamp retornado sempre timezone-aware em UTC
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0) -> None:
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch_candles(self, symbol: str, start: datetime, end: datetime) -> list[Candle]:
        require_tz_aware(start, "start")
        require_tz_aware(end, "end")

        start_utc = to_utc(start)
        end_utc = to_utc(end)

        if start_utc > end_utc:
            raise ValueError("start must be <= end")

        logger.info(
            "Fetching candles",
            extra={
                "symbol": symbol,
                "start": start_utc.date().isoformat(),
                "end": end_utc.date().isoformat(),
            },
        )

        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                df = yf.download(
                    symbol,
                    start=start_utc.strftime("%Y-%m-%d"),
                    end=end_utc.strftime("%Y-%m-%d"),
                    interval="1d",
                    progress=False,
                    timeout=10,
                    auto_adjust=False,
                )

                if df is None or df.empty:
                    raise ValueError(f"No data returned for {symbol}")

                # Normalizar MultiIndex (quando yfinance retorna colunas com níveis)
                if getattr(df.columns, "nlevels", 1) > 1:
                    df.columns = df.columns.get_level_values(0)

                required_cols = {"Open", "High", "Low", "Close", "Volume"}
                missing = required_cols - set(df.columns)
                if missing:
                    raise ValueError(f"Missing columns in response: {sorted(missing)}")

                candles: list[Candle] = []

                for idx, row in df.iterrows():
                    ts = idx.to_pydatetime()

                    # idx pode vir naive dependendo do ambiente; padroniza para UTC
                    if ts.tzinfo is None:
                        ts = datetime.combine(ts.date(), time(0, 0), tzinfo=timezone.utc)
                    else:
                        # Converte pra UTC e normaliza para 00:00 UTC do dia (opcional, mas consistente p/ joins)
                        ts_utc = ts.astimezone(timezone.utc)
                        ts = datetime.combine(ts_utc.date(), time(0, 0), tzinfo=timezone.utc)

                    candles.append(
                        Candle(
                            timestamp=ts,
                            open=float(row["Open"]),
                            high=float(row["High"]),
                            low=float(row["Low"]),
                            close=float(row["Close"]),
                            volume=int(row["Volume"]),
                        )
                    )

                logger.info(
                    "Candles fetched successfully",
                    extra={"symbol": symbol, "count": len(candles)},
                )

                return candles

            except Exception as e:
                last_error = e
                logger.warning(
                    "Fetch attempt failed",
                    extra={"symbol": symbol, "attempt": attempt + 1, "error": str(e)},
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
            f"Failed to fetch {symbol} after {self.max_retries} retries: {last_error}"
        ) from last_error
