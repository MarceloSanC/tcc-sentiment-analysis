from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from src.domain.time.utc import require_tz_aware, to_utc
from src.entities.fundamental_report import FundamentalReport
from src.interfaces.fundamental_fetcher import FundamentalFetcher
from src.interfaces.fundamental_repository import FundamentalRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FetchFundamentalsResult:
    asset_id: str
    fetched: int
    saved: int
    start: datetime
    end: datetime
    report_types: list[str]


class FetchFundamentalsUseCase:
    """
    Fetch and persist fundamentals for an asset.
    """

    def __init__(
        self,
        fundamental_fetcher: FundamentalFetcher,
        fundamental_repository: FundamentalRepository,
    ) -> None:
        self.fundamental_fetcher = fundamental_fetcher
        self.fundamental_repository = fundamental_repository

    def execute(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
        *,
        report_types: list[str] | None = None,
    ) -> FetchFundamentalsResult:
        require_tz_aware(start_date, "start_date")
        require_tz_aware(end_date, "end_date")

        start_utc = to_utc(start_date)
        end_utc = to_utc(end_date)
        if start_utc > end_utc:
            raise ValueError("start_date must be <= end_date")

        report_types = report_types or ["annual", "quarterly"]

        reports = self.fundamental_fetcher.fetch_fundamentals(asset_id)
        filtered: list[FundamentalReport] = []

        for r in reports:
            if r.report_type not in report_types:
                continue
            if r.fiscal_date_end < start_utc.date() or r.fiscal_date_end > end_utc.date():
                continue
            filtered.append(r)

        if not filtered:
            logger.info(
                "Fundamentals fetch skipped (no reports in interval)",
                extra={
                    "asset": asset_id,
                    "start": start_utc.isoformat(),
                    "end": end_utc.isoformat(),
                    "report_types": report_types,
                },
            )
            return FetchFundamentalsResult(
                asset_id=asset_id,
                fetched=0,
                saved=0,
                start=start_utc,
                end=end_utc,
                report_types=report_types,
            )

        self.fundamental_repository.upsert_reports(filtered)

        logger.info(
            "Fundamentals fetched",
            extra={
                "asset": asset_id,
                "fetched": len(reports),
                "saved": len(filtered),
                "start": start_utc.isoformat(),
                "end": end_utc.isoformat(),
                "report_types": report_types,
            },
        )

        return FetchFundamentalsResult(
            asset_id=asset_id,
            fetched=len(reports),
            saved=len(filtered),
            start=start_utc,
            end=end_utc,
            report_types=report_types,
        )
