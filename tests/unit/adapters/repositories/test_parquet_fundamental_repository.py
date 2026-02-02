# tests/unit/adapters/repositories/test_parquet_fundamental_repository.py

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.adapters.parquet_fundamental_repository import ParquetFundamentalRepository
from src.entities.fundamental_report import FundamentalReport
from src.infrastructure.schemas.fundamental_parquet_schema import (
    FUNDAMENTAL_PARQUET_COLUMNS,
    FUNDAMENTAL_PARQUET_DTYPES,
)


def _dt_utc(y: int, m: int, d: int, hh: int = 0, mm: int = 0, ss: int = 0) -> datetime:
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


@pytest.fixture
def repo(tmp_path: Path) -> ParquetFundamentalRepository:
    return ParquetFundamentalRepository(output_dir=tmp_path)


def test_repository_raises_if_output_dir_is_not_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir.txt"
    file_path.write_text("not a dir")

    with pytest.raises(NotADirectoryError):
        ParquetFundamentalRepository(output_dir=file_path)


def test_upsert_persists_parquet_with_schema_and_dtypes(
    repo: ParquetFundamentalRepository, tmp_path: Path
) -> None:
    reports = [
        FundamentalReport(
            asset_id="AAPL",
            fiscal_date_end=date(2024, 12, 31),
            report_type="annual",
            revenue=100.0,
            net_income=10.0,
            operating_cash_flow=12.0,
            total_shareholder_equity=50.0,
            total_liabilities=40.0,
            reported_date=date(2025, 2, 1),
            source="alpha_vantage",
        ),
        FundamentalReport(
            asset_id="AAPL",
            fiscal_date_end=date(2024, 9, 30),
            report_type="quarterly",
            revenue=25.0,
            net_income=2.0,
            operating_cash_flow=3.0,
            total_shareholder_equity=49.0,
            total_liabilities=41.0,
            reported_date=None,
            source=None,
        ),
    ]

    repo.upsert_reports(reports)

    parquet_path = tmp_path / "AAPL" / "fundamentals_AAPL.parquet"
    assert parquet_path.exists()

    df = pd.read_parquet(parquet_path)
    assert list(df.columns) == FUNDAMENTAL_PARQUET_COLUMNS

    fiscal = pd.to_datetime(df["fiscal_date_end"], utc=True, errors="raise")
    assert str(fiscal.dt.tz) == "UTC"

    reported = pd.to_datetime(df["reported_date"], utc=True, errors="coerce")
    assert str(reported.dt.tz) == "UTC"

    assert str(df["revenue"].dtype) == FUNDAMENTAL_PARQUET_DTYPES["revenue"]
    assert str(df["net_income"].dtype) == FUNDAMENTAL_PARQUET_DTYPES["net_income"]
    assert str(df["operating_cash_flow"].dtype) == FUNDAMENTAL_PARQUET_DTYPES["operating_cash_flow"]
    assert str(df["total_shareholder_equity"].dtype) == FUNDAMENTAL_PARQUET_DTYPES["total_shareholder_equity"]
    assert str(df["total_liabilities"].dtype) == FUNDAMENTAL_PARQUET_DTYPES["total_liabilities"]

    assert pd.isna(df.loc[df["report_type"] == "quarterly", "reported_date"]).any()


def test_upsert_deduplicates_by_report_type_and_fiscal_date(
    repo: ParquetFundamentalRepository, tmp_path: Path
) -> None:
    r1 = FundamentalReport(
        asset_id="AAPL",
        fiscal_date_end=date(2024, 12, 31),
        report_type="annual",
        revenue=100.0,
        net_income=10.0,
        operating_cash_flow=12.0,
        total_shareholder_equity=50.0,
        total_liabilities=40.0,
        reported_date=date(2025, 2, 1),
        source="alpha_vantage",
    )
    r1_updated = FundamentalReport(
        asset_id="AAPL",
        fiscal_date_end=date(2024, 12, 31),
        report_type="annual",
        revenue=110.0,
        net_income=11.0,
        operating_cash_flow=13.0,
        total_shareholder_equity=51.0,
        total_liabilities=41.0,
        reported_date=date(2025, 2, 5),
        source="alpha_vantage",
    )

    repo.upsert_reports([r1])
    repo.upsert_reports([r1_updated])

    parquet_path = tmp_path / "AAPL" / "fundamentals_AAPL.parquet"
    df = pd.read_parquet(parquet_path)

    assert len(df) == 1
    assert df.loc[0, "revenue"] == pytest.approx(110.0)
    reported = pd.to_datetime(df.loc[0, "reported_date"], utc=True, errors="raise")
    assert reported.date().isoformat() == "2025-02-05"


def test_list_reports_filters_inclusive(repo: ParquetFundamentalRepository) -> None:
    repo.upsert_reports(
        [
            FundamentalReport(
                asset_id="AAPL",
                fiscal_date_end=date(2023, 12, 31),
                report_type="annual",
                revenue=90.0,
                net_income=9.0,
                operating_cash_flow=10.0,
                total_shareholder_equity=45.0,
                total_liabilities=35.0,
            ),
            FundamentalReport(
                asset_id="AAPL",
                fiscal_date_end=date(2024, 12, 31),
                report_type="annual",
                revenue=100.0,
                net_income=10.0,
                operating_cash_flow=12.0,
                total_shareholder_equity=50.0,
                total_liabilities=40.0,
            ),
        ]
    )

    out = repo.list_reports("AAPL", _dt_utc(2024, 1, 1), _dt_utc(2024, 12, 31))

    assert len(out) == 1
    assert out[0].fiscal_date_end == date(2024, 12, 31)
