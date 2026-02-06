# src/use_cases/fetch_candles_use_case.py
from __future__ import annotations

from datetime import datetime, timedelta, time, timezone

from domain.time.utc import require_tz_aware, to_utc
from src.interfaces.candle_fetcher import CandleFetcher
from src.interfaces.candle_repository import CandleRepository


class FetchCandlesUseCase:
    def __init__(
        self,
        candle_fetcher: CandleFetcher,
        candle_repository: CandleRepository,
    ) -> None:
        self.candle_fetcher = candle_fetcher
        self.candle_repository = candle_repository

    @staticmethod
    def _inclusive_end_for_daily(end_utc: datetime) -> datetime:
        """
        yfinance 'end' tende a ser exclusive. Para candles 1d,
        tornamos o intervalo inclusivo avançando 1 dia.
        """
        # se vier como "dia" (00:00) ou sem hora relevante, torna inclusivo
        if end_utc.time() == time(0, 0):
            return end_utc + timedelta(days=1)
        # mesmo se vier com hora, incluir o dia do end
        return datetime.combine(end_utc.date() + timedelta(days=1), time(0, 0), tzinfo=timezone.utc)

    def execute(self, symbol: str, start: datetime, end: datetime) -> tuple[int, int]:
        require_tz_aware(start, "start")
        require_tz_aware(end, "end")

        start_utc = to_utc(start)
        end_utc = to_utc(end)

        if start_utc > end_utc:
            raise ValueError("start must be <= end")

        end_inclusive = self._inclusive_end_for_daily(end_utc)

        candles = self.candle_fetcher.fetch_candles(symbol, start_utc, end_inclusive)

        existing = 0
        try:
            existing = len(self.candle_repository.load_candles(symbol))
        except FileNotFoundError:
            existing = 0

        self.candle_repository.save_candles(symbol, candles)
        return len(candles), existing
