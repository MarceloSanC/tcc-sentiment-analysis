# tests/unit/use_cases/test_fetch_fundamentals_use_case.py

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from src.entities.fundamental_report import FundamentalReport
from src.interfaces.fundamental_fetcher import FundamentalFetcher
from src.interfaces.fundamental_repository import FundamentalRepository
from src.use_cases.fetch_fundamentals_use_case import FetchFundamentalsUseCase


def _dt_utc(y: int, m: int, d: int, hh: int = 0, mm: int = 0, ss: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


class FakeFundamentalFetcher(FundamentalFetcher):
    def __init__(self, reports: list[FundamentalReport]) -> None:
        self._reports = reports

    def fetch_fundamentals(self, asset_id: str) -> list[FundamentalReport]:
        return list(self._reports)


class FakeFundamentalRepository(FundamentalRepository):
    def __init__(self) -> None:
        self.saved: list[FundamentalReport] = []

    def get_latest_fiscal_date(self, asset_id: str, report_type: str | None = None):
        return None

    def upsert_reports(self, reports: list[FundamentalReport]) -> None:
        self.saved.extend(reports)

    def list_reports(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
        report_type: str | None = None,
        include_latest_before_start: bool = False,
    ):
        return []


def _report(asset_id: str, fiscal: date, report_type: str) -> FundamentalReport:
    return FundamentalReport(
        asset_id=asset_id,
        fiscal_date_end=fiscal,
        report_type=report_type,
        revenue=100.0,
        net_income=10.0,
        operating_cash_flow=12.0,
        total_shareholder_equity=50.0,
        total_liabilities=40.0,
    )


def test_filters_reports_by_date_and_type() -> None:
    reports = [
        _report("AAPL", date(2023, 12, 31), "annual"),
        _report("AAPL", date(2024, 3, 31), "quarterly"),
        _report("AAPL", date(2024, 12, 31), "annual"),
    ]

    fetcher = FakeFundamentalFetcher(reports)
    repo = FakeFundamentalRepository()

    use_case = FetchFundamentalsUseCase(fetcher, repo)

    result = use_case.execute(
        asset_id="AAPL",
        start_date=_dt_utc(2024, 1, 1),
        end_date=_dt_utc(2024, 12, 31),
        report_types=["annual"],
    )

    assert result.saved == 1
    assert repo.saved[0].report_type == "annual"
    assert repo.saved[0].fiscal_date_end == date(2024, 12, 31)


def test_short_circuit_when_no_reports_in_interval() -> None:
    reports = [
        _report("AAPL", date(2023, 12, 31), "annual"),
    ]

    fetcher = FakeFundamentalFetcher(reports)
    repo = FakeFundamentalRepository()

    use_case = FetchFundamentalsUseCase(fetcher, repo)

    result = use_case.execute(
        asset_id="AAPL",
        start_date=_dt_utc(2024, 1, 1),
        end_date=_dt_utc(2024, 12, 31),
    )

    assert result.saved == 0
    assert repo.saved == []


def test_raises_on_invalid_date_range() -> None:
    fetcher = FakeFundamentalFetcher([])
    repo = FakeFundamentalRepository()

    use_case = FetchFundamentalsUseCase(fetcher, repo)

    with pytest.raises(ValueError, match="start_date must be <= end_date"):
        use_case.execute(
            asset_id="AAPL",
            start_date=_dt_utc(2024, 1, 2),
            end_date=_dt_utc(2024, 1, 1),
        )
