from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass(frozen=True, slots=True)
class FundamentalReport:
    """
    Domain entity representing a fundamentals snapshot for a period.

    Invariants:
    - asset_id non-empty
    - fiscal_date_end is a date (no time)
    - report_type in {"annual", "quarterly"}
    - numeric fields are floats if provided
    """

    asset_id: str
    fiscal_date_end: date
    report_type: str

    revenue: Optional[float]
    net_income: Optional[float]
    operating_cash_flow: Optional[float]
    total_shareholder_equity: Optional[float]
    total_liabilities: Optional[float]

    reported_date: Optional[date] = None
    source: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.asset_id, str) or not self.asset_id.strip():
            raise ValueError("asset_id must be a non-empty string")

        if isinstance(self.fiscal_date_end, datetime):
            raise TypeError("fiscal_date_end must be a date without time")
        if not isinstance(self.fiscal_date_end, date):
            raise TypeError("fiscal_date_end must be a date instance")

        if self.report_type not in {"annual", "quarterly"}:
            raise ValueError("report_type must be 'annual' or 'quarterly'")

        for name, value in (
            ("revenue", self.revenue),
            ("net_income", self.net_income),
            ("operating_cash_flow", self.operating_cash_flow),
            ("total_shareholder_equity", self.total_shareholder_equity),
            ("total_liabilities", self.total_liabilities),
        ):
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric when provided")

        if self.reported_date is not None:
            if isinstance(self.reported_date, datetime):
                raise TypeError("reported_date must be a date without time")
            if not isinstance(self.reported_date, date):
                raise TypeError("reported_date must be a date instance")
