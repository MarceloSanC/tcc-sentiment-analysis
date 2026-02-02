from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime
from typing import Any, Optional

import requests

from src.entities.fundamental_report import FundamentalReport
from src.interfaces.fundamental_fetcher import FundamentalFetcher

logger = logging.getLogger(__name__)


class AlphaVantageFundamentalFetcher(FundamentalFetcher):
    """
    Fetch fundamentals using Alpha Vantage (annual + quarterly).

    Endpoints:
      - INCOME_STATEMENT
      - BALANCE_SHEET
      - CASH_FLOW
      - EARNINGS
    """

    BASE_URL = "https://www.alphavantage.co/query"
    _MIN_INTERVAL = 12.5  # seconds (Alpha Vantage free tier)

    def __init__(
        self,
        api_key: str,
        timeout_seconds: int = 30,
        session: Optional[requests.Session] = None,
        user_agent: str = "tcc-sentiment-analysis/1.0",
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._session = session or requests.Session()
        self._user_agent = user_agent
        self._last_request_ts: float | None = None
        self._lock = threading.Lock()

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()

            if self._last_request_ts is not None:
                elapsed = now - self._last_request_ts
                sleep_for = self._MIN_INTERVAL - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

            self._last_request_ts = time.monotonic()

    def _get(self, function: str, symbol: str) -> dict[str, Any]:
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.api_key,
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": self._user_agent,
        }

        self._throttle()
        response = self._session.get(
            self.BASE_URL,
            params=params,
            headers=headers,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, dict):
            raise ValueError(f"Unexpected response format for {function}")

        if "Note" in data:
            raise RuntimeError(f"Alpha Vantage rate limit hit: {data['Note']}")
        if "Information" in data:
            raise RuntimeError(f"Alpha Vantage Information: {data['Information']}")

        return data

    @staticmethod
    def _to_date(value: str | None) -> date | None:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return None

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value in (None, "", "None", "null", "NaN"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _merge_reports(
        base: dict[tuple[str, date], dict[str, Any]],
        report_type: str,
        records: list[dict[str, Any]],
        field_map: dict[str, str],
    ) -> None:
        for item in records:
            fiscal_end = AlphaVantageFundamentalFetcher._to_date(
                item.get("fiscalDateEnding")
            )
            if fiscal_end is None:
                continue
            key = (report_type, fiscal_end)
            entry = base.setdefault(
                key,
                {
                    "fiscal_date_end": fiscal_end,
                    "report_type": report_type,
                },
            )
            for src_key, dst_key in field_map.items():
                entry[dst_key] = AlphaVantageFundamentalFetcher._to_float(
                    item.get(src_key)
                )

    def fetch_fundamentals(self, asset_id: str) -> list[FundamentalReport]:
        income = self._get("INCOME_STATEMENT", asset_id)
        balance = self._get("BALANCE_SHEET", asset_id)
        cash_flow = self._get("CASH_FLOW", asset_id)
        earnings = self._get("EARNINGS", asset_id)

        merged: dict[tuple[str, date], dict[str, Any]] = {}

        income_map = {
            "totalRevenue": "revenue",
            "netIncome": "net_income",
        }
        balance_map = {
            "totalShareholderEquity": "total_shareholder_equity",
            "totalLiabilities": "total_liabilities",
        }
        cash_flow_map = {
            "operatingCashflow": "operating_cash_flow",
        }

        for report_type, key in (("annual", "annualReports"), ("quarterly", "quarterlyReports")):
            self._merge_reports(merged, report_type, income.get(key, []) or [], income_map)
            self._merge_reports(merged, report_type, balance.get(key, []) or [], balance_map)
            self._merge_reports(merged, report_type, cash_flow.get(key, []) or [], cash_flow_map)

        # reported_date from earnings endpoint (if present)
        for report_type, key in (("annual", "annualEarnings"), ("quarterly", "quarterlyEarnings")):
            for item in earnings.get(key, []) or []:
                fiscal_end = self._to_date(item.get("fiscalDateEnding"))
                if fiscal_end is None:
                    continue
                entry = merged.setdefault(
                    (report_type, fiscal_end),
                    {
                        "fiscal_date_end": fiscal_end,
                        "report_type": report_type,
                    },
                )
                reported_date = self._to_date(item.get("reportedDate"))
                if reported_date is not None:
                    entry["reported_date"] = reported_date

        reports: list[FundamentalReport] = []
        for (report_type, fiscal_end), values in merged.items():
            reports.append(
                FundamentalReport(
                    asset_id=asset_id,
                    fiscal_date_end=fiscal_end,
                    report_type=report_type,
                    revenue=values.get("revenue"),
                    net_income=values.get("net_income"),
                    operating_cash_flow=values.get("operating_cash_flow"),
                    total_shareholder_equity=values.get("total_shareholder_equity"),
                    total_liabilities=values.get("total_liabilities"),
                    reported_date=values.get("reported_date"),
                    source="alpha_vantage",
                )
            )

        logger.info(
            "Alpha Vantage fundamentals fetched",
            extra={
                "asset": asset_id,
                "annual": len([r for r in reports if r.report_type == "annual"]),
                "quarterly": len([r for r in reports if r.report_type == "quarterly"]),
            },
        )

        return reports
