# src/domain/time/trading_calendar.py
from datetime import datetime, date, timezone


def normalize_to_trading_day(ts: datetime) -> date:
    """
    Normalize a timestamp to a trading day (UTC-based).

    Domain rule:
    - All temporal joins between news and candles
      must be performed at date granularity in UTC.
    - This avoids time leakage and timezone ambiguity.
    """
    if ts.tzinfo is None:
        raise ValueError(
            "Timestamp must be timezone-aware for trading day normalization"
        )

    return ts.astimezone(timezone.utc).date()
