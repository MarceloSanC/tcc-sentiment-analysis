from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.domain.time.utc import require_tz_aware, to_utc
from src.entities.candle import Candle
from src.entities.technical_indicator_set import TechnicalIndicatorSet
from src.entities.daily_sentiment import DailySentiment
from src.entities.fundamental_report import FundamentalReport
from src.interfaces.candle_repository import CandleRepository
from src.interfaces.technical_indicator_repository import TechnicalIndicatorRepository
from src.interfaces.daily_sentiment_repository import DailySentimentRepository
from src.interfaces.fundamental_repository import FundamentalRepository
from src.interfaces.tft_dataset_repository import TFTDatasetRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildTFTDatasetResult:
    asset_id: str
    rows: int
    start: datetime
    end: datetime
    nulls: int


class BuildTFTDatasetUseCase:
    """
    Assemble daily dataset for TFT from:
      candles + technical indicators + daily sentiment + fundamentals.
    """

    def __init__(
        self,
        candle_repository: CandleRepository,
        indicator_repository: TechnicalIndicatorRepository,
        daily_sentiment_repository: DailySentimentRepository,
        fundamental_repository: FundamentalRepository,
        tft_dataset_repository: TFTDatasetRepository,
        report_dir_name: str = "reports",
    ) -> None:
        self.candle_repository = candle_repository
        self.indicator_repository = indicator_repository
        self.daily_sentiment_repository = daily_sentiment_repository
        self.fundamental_repository = fundamental_repository
        self.tft_dataset_repository = tft_dataset_repository
        self.report_dir_name = report_dir_name

    @staticmethod
    def _candles_to_df(candles: list[Candle]) -> pd.DataFrame:
        rows = []
        for c in candles:
            rows.append(
                {
                    "timestamp": to_utc(c.timestamp),
                    "open": float(c.open),
                    "high": float(c.high),
                    "low": float(c.low),
                    "close": float(c.close),
                    "volume": float(c.volume),
                }
            )
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    @staticmethod
    def _indicators_to_df(indicators: list[TechnicalIndicatorSet]) -> pd.DataFrame:
        rows = []
        for item in indicators:
            row = {"timestamp": to_utc(item.timestamp)}
            row.update(item.indicators)
            rows.append(row)
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
        return df

    @staticmethod
    def _sentiment_to_df(sentiments: list[DailySentiment]) -> pd.DataFrame:
        rows = []
        for s in sentiments:
            rows.append(
                {
                    "date": pd.Timestamp(s.day, tz="UTC"),
                    "sentiment_score": float(s.sentiment_score),
                    "news_volume": int(s.n_articles),
                    "sentiment_std": float(s.sentiment_std) if s.sentiment_std is not None else None,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _fundamentals_to_df(reports: list[FundamentalReport]) -> pd.DataFrame:
        rows = []
        for r in reports:
            reported_date = r.reported_date
            if reported_date is None:
                reported_date = r.fiscal_date_end + timedelta(days=45)
            rows.append(
                {
                    "effective_date": pd.Timestamp(reported_date, tz="UTC"),
                    "revenue": r.revenue,
                    "net_income": r.net_income,
                    "operating_cash_flow": r.operating_cash_flow,
                    "total_shareholder_equity": r.total_shareholder_equity,
                    "total_liabilities": r.total_liabilities,
                }
            )
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["effective_date"] = pd.to_datetime(df["effective_date"], utc=True, errors="raise")
        df = df.sort_values("effective_date").reset_index(drop=True)
        return df

    def execute(self, asset_id: str, start_date: datetime, end_date: datetime) -> BuildTFTDatasetResult:
        require_tz_aware(start_date, "start_date")
        require_tz_aware(end_date, "end_date")

        start_utc = to_utc(start_date)
        end_utc = to_utc(end_date)
        if start_utc > end_utc:
            raise ValueError("start_date must be <= end_date")

        candles = self.candle_repository.load_candles(asset_id)
        if not candles:
            raise ValueError(f"No candles found for {asset_id}")
        candles_df = self._candles_to_df(candles)

        indicators = self.indicator_repository.load(asset_id)
        if not indicators:
            raise ValueError(f"No technical indicators found for {asset_id}")
        indicators_df = self._indicators_to_df(indicators)

        daily_sentiments = self.daily_sentiment_repository.list_daily_sentiment(
            asset_id,
            start_utc,
            end_utc,
        )
        if not daily_sentiments:
            logger.warning(
                "No daily sentiment found for period",
                extra={
                    "asset_id": asset_id,
                    "start": start_utc.isoformat(),
                    "end": end_utc.isoformat(),
                },
            )
        sentiment_df = self._sentiment_to_df(daily_sentiments)

        fundamentals = self.fundamental_repository.list_reports(
            asset_id,
            start_utc,
            end_utc,
        )
        if not fundamentals:
            logger.warning(
                "No fundamentals found for period",
                extra={
                    "asset_id": asset_id,
                    "start": start_utc.isoformat(),
                    "end": end_utc.isoformat(),
                },
            )
        fundamentals_df = self._fundamentals_to_df(fundamentals)

        base = candles_df.copy()
        base["asset_id"] = asset_id
        base["date"] = base["timestamp"].dt.normalize()
        base = base[(base["timestamp"] >= start_utc) & (base["timestamp"] <= end_utc)]

        # Merge indicators on timestamp
        df = base.merge(indicators_df, on="timestamp", how="left")

        # Merge sentiment on date
        if not sentiment_df.empty:
            df = df.merge(sentiment_df, on="date", how="left")
        else:
            df["sentiment_score"] = pd.NA
            df["news_volume"] = pd.NA
            df["sentiment_std"] = pd.NA

        # Merge fundamentals with as-of join
        if not fundamentals_df.empty:
            df = df.sort_values("date")
            fundamentals_df = fundamentals_df.sort_values("effective_date")
            df = pd.merge_asof(
                df,
                fundamentals_df,
                left_on="date",
                right_on="effective_date",
                direction="backward",
            )
            df = df.drop(columns=["effective_date"])
        else:
            for col in [
                "revenue",
                "net_income",
                "operating_cash_flow",
                "total_shareholder_equity",
                "total_liabilities",
            ]:
                df[col] = pd.NA

        # Time features
        df["day_of_week"] = df["timestamp"].dt.dayofweek.astype("int64")
        df["month"] = df["timestamp"].dt.month.astype("int64")

        # Target: next-day log-return
        df = df.sort_values("timestamp").reset_index(drop=True)
        if df["timestamp"].duplicated().any():
            raise ValueError("Duplicate timestamps found while building TFT dataset")
        if not df["timestamp"].is_monotonic_increasing:
            raise ValueError("Timestamps are not monotonic in TFT dataset")

        df["target_return"] = np.log(df["close"].shift(-1) / df["close"])
        df = df.dropna(subset=["target_return"]).reset_index(drop=True)
        if df.empty:
            raise ValueError("Not enough rows to compute target_return")

        # Drop helper columns not used by the model
        if "date" in df.columns:
            df = df.drop(columns=["date"])

        # time_idx
        df["time_idx"] = range(len(df))

        self.tft_dataset_repository.save(asset_id, df)

        nulls = int(df.isna().sum().sum())

        logger.info(
            "TFT dataset built",
            extra={
                "asset_id": asset_id,
                "rows": len(df),
                "cols": len(df.columns),
                "start": df["timestamp"].min().isoformat(),
                "end": df["timestamp"].max().isoformat(),
                "nulls": nulls,
            },
        )

        return BuildTFTDatasetResult(
            asset_id=asset_id,
            rows=len(df),
            start=df["timestamp"].min().to_pydatetime(),
            end=df["timestamp"].max().to_pydatetime(),
            nulls=nulls,
        )
