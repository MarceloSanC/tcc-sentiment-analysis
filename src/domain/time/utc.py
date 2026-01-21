# src/domain/time/utc.py

from __future__ import annotations

from datetime import datetime, timezone


def require_tz_aware(dt: datetime, name: str = "datetime") -> None:
    """
    Enforce timezone-aware datetime.

    Use in core/use-cases/adapters where naive datetimes are forbidden.
    """
    if dt.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware (UTC enforced)")


def to_utc(dt: datetime) -> datetime:
    """
    Convert a timezone-aware datetime to UTC.

    Raises:
        ValueError if dt is naive.
    """
    require_tz_aware(dt, "dt")
    return dt.astimezone(timezone.utc)


def ensure_utc(dt: datetime) -> datetime:
    """
    Ensure a datetime is timezone-aware in UTC.

    - If naive: assumes UTC.
    - If aware: converts to UTC.

    Intended for entry points (config/CLI parsing), NOT for domain/adapters.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_iso_utc(value: str) -> datetime:
    """
    Parse ISO date/datetime string and return timezone-aware UTC datetime.

    Accepts:
      - YYYY-MM-DD
      - YYYY-MM-DDTHH:MM:SS
      - With or without timezone
      - Trailing 'Z'

    Examples:
      "2025-12-31"
      "2025-12-31T15:30:00"
      "2025-12-31T15:30:00Z"
      "2025-12-31T15:30:00+00:00"
    """
    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"

    dt = datetime.fromisoformat(v)
    return ensure_utc(dt)
