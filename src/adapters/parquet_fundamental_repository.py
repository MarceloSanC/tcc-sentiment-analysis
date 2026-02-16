from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.domain.time.utc import require_tz_aware, to_utc
from src.entities.fundamental_report import FundamentalReport
from src.infrastructure.schemas.fundamental_parquet_schema import (
    FUNDAMENTAL_PARQUET_COLUMNS,
    FUNDAMENTAL_PARQUET_DTYPES,
)
from src.interfaces.fundamental_repository import FundamentalRepository

logger = logging.getLogger(__name__)


class ParquetFundamentalRepository(FundamentalRepository):
    """
    Parquet-based repository for FundamentalReport.

    Storage layout:
      data/processed/fundamentals/AAPL/fundamentals_AAPL.parquet
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

        if self.output_dir.exists() and not self.output_dir.is_dir():
            raise NotADirectoryError(
                f"Fundamentals output_dir is not a directory: {self.output_dir.resolve()}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "ParquetFundamentalRepository initialized",
            extra={"output_dir": str(self.output_dir.resolve())},
        )

    @staticmethod
    def _normalize_symbol(asset_id: str) -> str:
        return asset_id.split(".")[0].upper()

    def _asset_dir(self, asset_id: str) -> Path:
        symbol = self._normalize_symbol(asset_id)
        return self.output_dir / symbol

    def _filepath(self, asset_id: str) -> Path:
        symbol = self._normalize_symbol(asset_id)
        return self._asset_dir(symbol) / f"fundamentals_{symbol}.parquet"

    def get_latest_fiscal_date(
        self, asset_id: str, report_type: str | None = None
    ) -> date | None:
        filepath = self._filepath(asset_id)
        if not filepath.exists():
            return None

        df = pd.read_parquet(filepath, columns=["fiscal_date_end", "report_type"])
        if df.empty:
            return None

        if report_type:
            df = df[df["report_type"] == report_type]
            if df.empty:
                return None

        dates = pd.to_datetime(df["fiscal_date_end"], utc=True, errors="coerce")
        if dates.isna().all():
            raise ValueError(f"Invalid fiscal_date_end in {filepath}")

        latest = dates.max()
        return latest.date()

    def upsert_reports(self, reports: list[FundamentalReport]) -> None:
        if not reports:
            raise ValueError("No fundamental reports to upsert")

        asset = reports[0].asset_id
        if any(r.asset_id != asset for r in reports):
            raise ValueError("All reports in a batch must share the same asset_id")

        rows: list[dict] = []
        for r in reports:
            rows.append(
                {
                    "asset_id": self._normalize_symbol(r.asset_id),
                    "report_type": r.report_type,
                    "fiscal_date_end": pd.Timestamp(r.fiscal_date_end, tz="UTC"),
                    "reported_date": pd.Timestamp(r.reported_date, tz="UTC")
                    if r.reported_date is not None
                    else None,
                    "revenue": float(r.revenue) if r.revenue is not None else None,
                    "net_income": float(r.net_income) if r.net_income is not None else None,
                    "operating_cash_flow": float(r.operating_cash_flow)
                    if r.operating_cash_flow is not None
                    else None,
                    "total_shareholder_equity": float(r.total_shareholder_equity)
                    if r.total_shareholder_equity is not None
                    else None,
                    "total_liabilities": float(r.total_liabilities)
                    if r.total_liabilities is not None
                    else None,
                    "source": r.source,
                }
            )

        df_new = pd.DataFrame(rows)
        df_new = df_new[FUNDAMENTAL_PARQUET_COLUMNS]

        df_new["fiscal_date_end"] = pd.to_datetime(
            df_new["fiscal_date_end"], utc=True, errors="raise"
        )
        if "reported_date" in df_new.columns:
            df_new["reported_date"] = pd.to_datetime(
                df_new["reported_date"], utc=True, errors="coerce"
            )
        for col, dtype in FUNDAMENTAL_PARQUET_DTYPES.items():
            if col in df_new.columns:
                df_new[col] = df_new[col].astype(dtype)

        filepath = self._filepath(asset)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.exists():
            df_old = pd.read_parquet(filepath)
            if not df_old.empty:
                df_old["fiscal_date_end"] = pd.to_datetime(
                    df_old["fiscal_date_end"], utc=True, errors="coerce"
                )
                df = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df = df_new
        else:
            df = df_new

        df = df.drop_duplicates(
            subset=["report_type", "fiscal_date_end"], keep="last"
        )
        df = df.sort_values(["report_type", "fiscal_date_end"]).reset_index(drop=True)

        missing = set(FUNDAMENTAL_PARQUET_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns for fundamentals parquet: {sorted(missing)}")

        df.to_parquet(filepath, index=False)

        logger.info(
            "Fundamentals upserted",
            extra={
                "asset_id": self._normalize_symbol(asset),
                "saved_rows": len(df_new),
                "total_rows": len(df),
                "path": str(filepath.resolve()),
            },
        )

    def list_reports(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
        report_type: str | None = None,
        include_latest_before_start: bool = True,
    ) -> list[FundamentalReport]:
        require_tz_aware(start_date, "start_date")
        require_tz_aware(end_date, "end_date")

        start_utc = to_utc(start_date)
        end_utc = to_utc(end_date)
        if start_utc > end_utc:
            raise ValueError("start_date must be <= end_date")

        filepath = self._filepath(asset_id)
        if not filepath.exists():
            return []

        df = pd.read_parquet(filepath)
        if df.empty:
            return []

        if report_type:
            df = df[df["report_type"] == report_type]
            if df.empty:
                return []

        df["fiscal_date_end"] = pd.to_datetime(df["fiscal_date_end"], utc=True, errors="coerce")
        if df["fiscal_date_end"].isna().any():
            raise ValueError(f"Invalid fiscal_date_end values found in {filepath}")
        if "reported_date" in df.columns:
            df["reported_date"] = pd.to_datetime(df["reported_date"], utc=True, errors="coerce")
        else:
            df["reported_date"] = pd.NaT

        effective_date = df["reported_date"].fillna(df["fiscal_date_end"] + pd.Timedelta(days=45))
        df = df.assign(_effective_date=effective_date)

        start_day = pd.Timestamp(start_utc.date(), tz="UTC")
        end_day = pd.Timestamp(end_utc.date(), tz="UTC")
        mask = (df["_effective_date"] >= start_day) & (df["_effective_date"] <= end_day)
        selected = df.loc[mask]
        if include_latest_before_start:
            prior = df.loc[df["_effective_date"] < start_day]
            if not prior.empty:
                selected = pd.concat(
                    [selected, prior.nlargest(1, "_effective_date")],
                    ignore_index=True,
                )
        df = selected.drop_duplicates(
            subset=["report_type", "fiscal_date_end"], keep="last"
        ).sort_values(["report_type", "fiscal_date_end"])

        out: list[FundamentalReport] = []
        for _, r in df.iterrows():
            fiscal_date_end = r.get("fiscal_date_end")
            if isinstance(fiscal_date_end, pd.Timestamp):
                fiscal_date_end = fiscal_date_end.date()

            reported_date = r.get("reported_date")
            if isinstance(reported_date, pd.Timestamp):
                reported_date = reported_date.date()
            if pd.isna(reported_date):
                reported_date = None

            source_val = r.get("source")
            if pd.isna(source_val):
                source_val = None

            out.append(
                FundamentalReport(
                    asset_id=str(r.get("asset_id") or self._normalize_symbol(asset_id)),
                    fiscal_date_end=fiscal_date_end,
                    report_type=str(r.get("report_type")),
                    revenue=float(r.get("revenue")) if pd.notna(r.get("revenue")) else None,
                    net_income=float(r.get("net_income")) if pd.notna(r.get("net_income")) else None,
                    operating_cash_flow=float(r.get("operating_cash_flow"))
                    if pd.notna(r.get("operating_cash_flow"))
                    else None,
                    total_shareholder_equity=float(r.get("total_shareholder_equity"))
                    if pd.notna(r.get("total_shareholder_equity"))
                    else None,
                    total_liabilities=float(r.get("total_liabilities"))
                    if pd.notna(r.get("total_liabilities"))
                    else None,
                    reported_date=reported_date,
                    source=str(source_val) if source_val else None,
                )
            )

        return out
