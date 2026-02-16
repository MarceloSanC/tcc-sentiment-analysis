from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, date

from src.entities.fundamental_report import FundamentalReport


class FundamentalRepository(ABC):
    @abstractmethod
    def get_latest_fiscal_date(
        self, asset_id: str, report_type: str | None = None
    ) -> date | None:
        """Latest fiscal_date_end persisted for the asset."""
        ...

    @abstractmethod
    def upsert_reports(self, reports: list[FundamentalReport]) -> None:
        """Insert or update fundamentals (dedup by report_type + fiscal_date_end)."""
        ...

    @abstractmethod
    def list_reports(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
        report_type: str | None = None,
        include_latest_before_start: bool = True,
    ) -> list[FundamentalReport]:
        """List fundamentals in the given interval."""
        ...
