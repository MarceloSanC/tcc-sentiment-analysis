# tests/unit/use_cases/test_build_tft_dataset_use_case.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.entities.candle import Candle
from src.entities.daily_sentiment import DailySentiment
from src.entities.fundamental_report import FundamentalReport
from src.entities.technical_indicator_set import TechnicalIndicatorSet
from src.interfaces.candle_repository import CandleRepository
from src.interfaces.daily_sentiment_repository import DailySentimentRepository
from src.interfaces.fundamental_repository import FundamentalRepository
from src.interfaces.technical_indicator_repository import TechnicalIndicatorRepository
from src.interfaces.tft_dataset_repository import TFTDatasetRepository
from src.use_cases.build_tft_dataset_use_case import BuildTFTDatasetUseCase


def _dt_utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


class FakeCandleRepository(CandleRepository):
    def __init__(self, candles: list[Candle]) -> None:
        self._candles = candles

    def load_candles(self, asset_id: str) -> list[Candle]:
        return self._candles

    def save_candles(self, asset_id: str, candles: list[Candle]) -> None:
        raise NotImplementedError

    def update_sentiment(self, asset_id: str, daily_sentiments) -> None:
        raise NotImplementedError


class FakeTechnicalIndicatorRepository(TechnicalIndicatorRepository):
    def __init__(self, indicators: list[TechnicalIndicatorSet]) -> None:
        self._indicators = indicators

    def save(self, asset_id: str, indicators: list[TechnicalIndicatorSet]) -> None:
        raise NotImplementedError

    def load(self, asset_id: str) -> list[TechnicalIndicatorSet]:
        return self._indicators


class FakeDailySentimentRepository(DailySentimentRepository):
    def __init__(self, daily: list[DailySentiment]) -> None:
        self._daily = daily

    def upsert_daily_sentiment_batch(self, daily_sentiments) -> None:
        raise NotImplementedError

    def list_daily_sentiment(self, asset_id: str, start_date: datetime, end_date: datetime):
        return self._daily


class FakeFundamentalRepository(FundamentalRepository):
    def __init__(self, reports: list[FundamentalReport]) -> None:
        self._reports = reports

    def get_latest_fiscal_date(self, asset_id: str, report_type: str | None = None):
        raise NotImplementedError

    def upsert_reports(self, reports: list[FundamentalReport]) -> None:
        raise NotImplementedError

    def list_reports(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
        report_type: str | None = None,
        include_latest_before_start: bool = False,
    ) -> list[FundamentalReport]:
        start_day = start_date.date()
        end_day = end_date.date()

        selected: list[FundamentalReport] = []
        prior: list[tuple[date, FundamentalReport]] = []
        for report in self._reports:
            if report_type and report.report_type != report_type:
                continue
            effective_day = report.reported_date or (report.fiscal_date_end + timedelta(days=45))
            if start_day <= effective_day <= end_day:
                selected.append(report)
            elif effective_day < start_day:
                prior.append((effective_day, report))

        if include_latest_before_start and prior:
            selected.append(max(prior, key=lambda x: x[0])[1])
        return selected


@dataclass
class FakeTFTDatasetRepository(TFTDatasetRepository):
    output_dir: Path
    saved: pd.DataFrame | None = None

    def save(self, asset_id: str, df: pd.DataFrame) -> None:
        self.saved = df

    def load(self, asset_id: str) -> pd.DataFrame:
        raise NotImplementedError


def _candles() -> list[Candle]:
    return [
        Candle(timestamp=_dt_utc(2024, 1, 1), open=100, high=101, low=99, close=100, volume=10),
        Candle(timestamp=_dt_utc(2024, 1, 2), open=100, high=102, low=99, close=101, volume=11),
        Candle(timestamp=_dt_utc(2024, 1, 3), open=101, high=103, low=100, close=102, volume=12),
    ]


def _indicators(asset_id: str) -> list[TechnicalIndicatorSet]:
    return [
        TechnicalIndicatorSet(
            asset_id=asset_id,
            timestamp=_dt_utc(2024, 1, 1),
            indicators={"rsi_14": 30.0},
        ),
        TechnicalIndicatorSet(
            asset_id=asset_id,
            timestamp=_dt_utc(2024, 1, 2),
            indicators={"rsi_14": 31.0},
        ),
        TechnicalIndicatorSet(
            asset_id=asset_id,
            timestamp=_dt_utc(2024, 1, 3),
            indicators={"rsi_14": 32.0},
        ),
    ]


def _daily_sentiment(asset_id: str) -> list[DailySentiment]:
    return [
        DailySentiment(
            asset_id=asset_id,
            day=date(2024, 1, 1),
            sentiment_score=0.2,
            n_articles=2,
            sentiment_std=0.1,
        ),
        DailySentiment(
            asset_id=asset_id,
            day=date(2024, 1, 2),
            sentiment_score=-0.1,
            n_articles=1,
            sentiment_std=0.2,
        ),
    ]


def _fundamentals(asset_id: str) -> list[FundamentalReport]:
    return [
        FundamentalReport(
            asset_id=asset_id,
            fiscal_date_end=date(2023, 12, 31),
            report_type="annual",
            revenue=1000.0,
            net_income=200.0,
            operating_cash_flow=150.0,
            total_shareholder_equity=300.0,
            total_liabilities=400.0,
            reported_date=date(2024, 1, 1),
            source="mock",
        )
    ]


def test_builds_dataset_and_writes_reports(tmp_path: Path) -> None:
    asset_id = "AAPL"

    use_case = BuildTFTDatasetUseCase(
        candle_repository=FakeCandleRepository(_candles()),
        indicator_repository=FakeTechnicalIndicatorRepository(_indicators(asset_id)),
        daily_sentiment_repository=FakeDailySentimentRepository(_daily_sentiment(asset_id)),
        fundamental_repository=FakeFundamentalRepository(_fundamentals(asset_id)),
        tft_dataset_repository=FakeTFTDatasetRepository(output_dir=tmp_path),
    )

    result = use_case.execute(asset_id, _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 3))

    assert result.rows == 2  # last row dropped for target_return


def test_news_volume_defaults_to_zero_when_missing(tmp_path: Path) -> None:
    asset_id = "AAPL"
    candles = [
        Candle(timestamp=_dt_utc(2024, 1, 1), open=100, high=101, low=99, close=100, volume=10),
        Candle(timestamp=_dt_utc(2024, 1, 2), open=100, high=102, low=99, close=101, volume=11),
        Candle(timestamp=_dt_utc(2024, 1, 3), open=101, high=103, low=100, close=102, volume=12),
        Candle(timestamp=_dt_utc(2024, 1, 4), open=102, high=104, low=101, close=103, volume=13),
    ]
    indicators = [
        TechnicalIndicatorSet(
            asset_id=asset_id,
            timestamp=_dt_utc(2024, 1, 1),
            indicators={"rsi_14": 30.0},
        ),
        TechnicalIndicatorSet(
            asset_id=asset_id,
            timestamp=_dt_utc(2024, 1, 2),
            indicators={"rsi_14": 31.0},
        ),
        TechnicalIndicatorSet(
            asset_id=asset_id,
            timestamp=_dt_utc(2024, 1, 3),
            indicators={"rsi_14": 32.0},
        ),
        TechnicalIndicatorSet(
            asset_id=asset_id,
            timestamp=_dt_utc(2024, 1, 4),
            indicators={"rsi_14": 33.0},
        ),
    ]
    sentiments = [
        DailySentiment(
            asset_id=asset_id,
            day=date(2024, 1, 1),
            sentiment_score=0.2,
            n_articles=2,
            sentiment_std=0.1,
        ),
    ]
    repo = FakeTFTDatasetRepository(output_dir=tmp_path)
    use_case = BuildTFTDatasetUseCase(
        candle_repository=FakeCandleRepository(candles),
        indicator_repository=FakeTechnicalIndicatorRepository(indicators),
        daily_sentiment_repository=FakeDailySentimentRepository(sentiments),
        fundamental_repository=FakeFundamentalRepository([]),
        tft_dataset_repository=repo,
    )

    use_case.execute(asset_id, _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 4))

    assert repo.saved is not None
    missing_day = _dt_utc(2024, 1, 2)
    value = repo.saved.loc[repo.saved["timestamp"] == missing_day, "news_volume"].item()
    assert value == 0


def test_raises_on_invalid_date_range(tmp_path: Path) -> None:
    use_case = BuildTFTDatasetUseCase(
        candle_repository=FakeCandleRepository(_candles()),
        indicator_repository=FakeTechnicalIndicatorRepository(_indicators("AAPL")),
        daily_sentiment_repository=FakeDailySentimentRepository([]),
        fundamental_repository=FakeFundamentalRepository([]),
        tft_dataset_repository=FakeTFTDatasetRepository(output_dir=tmp_path),
    )

    with pytest.raises(ValueError, match="start_date must be <= end_date"):
        use_case.execute("AAPL", _dt_utc(2024, 1, 2), _dt_utc(2024, 1, 1))


def test_raises_when_insufficient_rows_for_target(tmp_path: Path) -> None:
    one_candle = [
        Candle(timestamp=_dt_utc(2024, 1, 1), open=100, high=101, low=99, close=100, volume=10),
    ]
    indicators = [
        TechnicalIndicatorSet(
            asset_id="AAPL",
            timestamp=_dt_utc(2024, 1, 1),
            indicators={"rsi_14": 30.0},
        )
    ]

    use_case = BuildTFTDatasetUseCase(
        candle_repository=FakeCandleRepository(one_candle),
        indicator_repository=FakeTechnicalIndicatorRepository(indicators),
        daily_sentiment_repository=FakeDailySentimentRepository([]),
        fundamental_repository=FakeFundamentalRepository([]),
        tft_dataset_repository=FakeTFTDatasetRepository(output_dir=tmp_path),
    )

    with pytest.raises(ValueError, match="Not enough rows to compute target_return"):
        use_case.execute("AAPL", _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 1))


def test_fundamentals_forward_fill_uses_latest_before_start(tmp_path: Path) -> None:
    asset_id = "AAPL"
    repo = FakeTFTDatasetRepository(output_dir=tmp_path)
    fundamentals = [
        FundamentalReport(
            asset_id=asset_id,
            fiscal_date_end=date(2023, 12, 31),
            report_type="annual",
            revenue=999.0,
            net_income=111.0,
            operating_cash_flow=222.0,
            total_shareholder_equity=333.0,
            total_liabilities=444.0,
            reported_date=date(2023, 12, 20),
            source="mock",
        )
    ]

    use_case = BuildTFTDatasetUseCase(
        candle_repository=FakeCandleRepository(_candles()),
        indicator_repository=FakeTechnicalIndicatorRepository(_indicators(asset_id)),
        daily_sentiment_repository=FakeDailySentimentRepository(_daily_sentiment(asset_id)),
        fundamental_repository=FakeFundamentalRepository(fundamentals),
        tft_dataset_repository=repo,
    )

    use_case.execute(asset_id, _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 3))

    assert repo.saved is not None
    assert repo.saved["revenue"].iloc[0] == pytest.approx(999.0)
