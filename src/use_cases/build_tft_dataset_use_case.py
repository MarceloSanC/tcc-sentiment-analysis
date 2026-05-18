from __future__ import annotations

import logging

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from src.domain.services.dataset_quality_gate import (
    DatasetQualityGate,
    DatasetQualityGateConfig,
)
from src.domain.time.trading_calendar import (
    TradingDayPolicy,
    trading_day_from_timestamp,
)
from src.domain.time.utc import require_tz_aware, to_utc
from src.entities.candle import Candle
from src.entities.daily_sentiment import DailySentiment
from src.entities.fundamental_report import FundamentalReport
from src.entities.technical_indicator_set import TechnicalIndicatorSet
from src.infrastructure.schemas.feature_validation_schema import FEATURE_WARMUP_BARS
from src.interfaces.candle_repository import CandleRepository
from src.interfaces.daily_sentiment_repository import DailySentimentRepository
from src.interfaces.fundamental_repository import FundamentalRepository
from src.interfaces.technical_indicator_repository import TechnicalIndicatorRepository
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
        trading_day_policy: TradingDayPolicy | None = None,
        quality_gate_config: DatasetQualityGateConfig | None = None,
    ) -> None:
        self.candle_repository = candle_repository
        self.indicator_repository = indicator_repository
        self.daily_sentiment_repository = daily_sentiment_repository
        self.fundamental_repository = fundamental_repository
        self.tft_dataset_repository = tft_dataset_repository
        self.report_dir_name = report_dir_name
        self.trading_day_policy = trading_day_policy or TradingDayPolicy()
        self.quality_gate_config = quality_gate_config or DatasetQualityGateConfig()

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

    def _fundamentals_to_df(self, reports: list[FundamentalReport]) -> pd.DataFrame:
        rows = []
        for r in reports:
            reported_date = r.reported_date
            if reported_date is None:
                reported_date = r.fiscal_date_end + timedelta(days=45)
            effective_ts = datetime.combine(
                reported_date,
                self.trading_day_policy.close_hour,
                tzinfo=UTC,
            )
            effective_day = trading_day_from_timestamp(
                effective_ts,
                self.trading_day_policy,
            )
            rows.append(
                {
                    "effective_date": pd.Timestamp(effective_day, tz="UTC"),
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

    @staticmethod
    def _add_phase_a_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = pd.to_numeric(out["close"], errors="coerce")
        volume = pd.to_numeric(out["volume"], errors="coerce")
        open_ = pd.to_numeric(out["open"], errors="coerce") if "open" in out.columns else pd.Series(np.nan, index=out.index)
        high = pd.to_numeric(out["high"], errors="coerce") if "high" in out.columns else pd.Series(np.nan, index=out.index)
        low = pd.to_numeric(out["low"], errors="coerce") if "low" in out.columns else pd.Series(np.nan, index=out.index)

        out["log_return_1d"] = np.log(close / close.shift(1))
        out["log_return_5d"] = np.log(close / close.shift(5))
        out["log_return_21d"] = np.log(close / close.shift(21))

        out["momentum_5d"] = close / close.shift(5) - 1.0
        out["momentum_21d"] = close / close.shift(21) - 1.0
        out["momentum_63d"] = close / close.shift(63) - 1.0
        out["reversal_1d"] = -close.pct_change(1)
        out["reversal_5d"] = -out["momentum_5d"]
        rolling_max_63 = close.rolling(63, min_periods=63).max()
        out["drawdown_lookback"] = close / rolling_max_63 - 1.0

        ret_1d = close.pct_change(1).abs()
        out["amihud_illiquidity_proxy"] = np.where(volume > 0, ret_1d / volume, np.nan)

        trailing_mean = volume.shift(1).rolling(20, min_periods=20).mean()
        trailing_std = volume.shift(1).rolling(20, min_periods=20).std(ddof=0)
        out["volume_zscore"] = np.where(
            trailing_std > 0,
            (volume - trailing_mean) / trailing_std,
            np.nan,
        )
        out["volume_spike_flag"] = (pd.to_numeric(out["volume_zscore"], errors="coerce") > 3.0).astype("int64")

        if {"open", "high", "low", "close"}.issubset(out.columns):
            log_hl = pd.Series(
                np.where((high > 0) & (low > 0), np.log(high / low), np.nan),
                index=out.index,
                dtype="float64",
            )
            park_var = (log_hl.pow(2).rolling(20, min_periods=20).mean()) / (4.0 * np.log(2.0))
            out["volatility_parkinson"] = np.sqrt(park_var.clip(lower=0.0))

            log_co = pd.Series(
                np.where((close > 0) & (open_ > 0), np.log(close / open_), np.nan),
                index=out.index,
                dtype="float64",
            )
            gk_term = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
            gk_var = gk_term.rolling(20, min_periods=20).mean()
            out["volatility_garman_klass"] = np.sqrt(gk_var.clip(lower=0.0))
        else:
            out["volatility_parkinson"] = np.nan
            out["volatility_garman_klass"] = np.nan

        downside = pd.Series(np.minimum(close.pct_change(1), 0.0), index=out.index, dtype="float64")
        out["downside_semivolatility"] = np.sqrt(
            downside.pow(2).rolling(20, min_periods=20).mean().clip(lower=0.0)
        )

        if "volatility_20d" in out.columns:
            vol20 = pd.to_numeric(out["volatility_20d"], errors="coerce")
        else:
            vol20 = close.pct_change().rolling(20, min_periods=20).std()
        out["vol_of_vol"] = vol20.rolling(20, min_periods=20).std(ddof=0)

        # Regimes (causal: thresholds use shifted trailing windows)
        q33 = vol20.shift(1).rolling(63, min_periods=63).quantile(1.0 / 3.0)
        q66 = vol20.shift(1).rolling(63, min_periods=63).quantile(2.0 / 3.0)
        vol_regime = pd.Series(np.nan, index=out.index, dtype="float64")
        valid_v = vol20.notna() & q33.notna() & q66.notna()
        vol_regime.loc[valid_v & (vol20 <= q33)] = 0
        vol_regime.loc[valid_v & (vol20 > q33) & (vol20 <= q66)] = 1
        vol_regime.loc[valid_v & (vol20 > q66)] = 2
        out["volatility_regime"] = vol_regime

        ema10 = pd.to_numeric(out["ema_10"], errors="coerce") if "ema_10" in out.columns else pd.Series(np.nan, index=out.index)
        ema50 = pd.to_numeric(out["ema_50"], errors="coerce") if "ema_50" in out.columns else pd.Series(np.nan, index=out.index)
        spread = ema10 - ema50
        deadband = spread.shift(1).rolling(63, min_periods=63).std(ddof=0) * 0.10
        trend_regime = pd.Series(np.nan, index=out.index, dtype="float64")
        valid_t = spread.notna() & deadband.notna()
        trend_regime.loc[valid_t & (spread > deadband)] = 1
        trend_regime.loc[valid_t & (spread < -deadband)] = -1
        trend_regime.loc[valid_t & (spread.abs() <= deadband)] = 0
        out["trend_regime"] = trend_regime

        ret1 = close.pct_change(1)
        tail_q10 = ret1.shift(1).rolling(63, min_periods=63).quantile(0.10)
        stress = pd.Series(np.nan, index=out.index, dtype="float64")
        valid_s = ret1.notna() & tail_q10.notna()
        stress.loc[valid_s] = (ret1.loc[valid_s] <= tail_q10.loc[valid_s]).astype("int64")
        out["stress_tail_return_flag"] = stress
        return out

    @staticmethod
    def _add_sentiment_dynamic_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        sentiment = pd.to_numeric(out.get("sentiment_score"), errors="coerce")
        volume = pd.to_numeric(out.get("volume"), errors="coerce")
        vol20 = pd.to_numeric(out.get("volatility_20d"), errors="coerce")

        out["sentiment_lag_1"] = sentiment.shift(1)
        out["sentiment_lag_3"] = sentiment.shift(3)
        out["sentiment_lag_5"] = sentiment.shift(5)
        out["sentiment_ema"] = sentiment.ewm(span=10, adjust=False).mean()
        baseline = sentiment.shift(1).rolling(5, min_periods=5).mean()
        out["sentiment_surprise"] = sentiment - baseline
        out["sentiment_x_volatility"] = sentiment * vol20
        out["sentiment_x_volume"] = sentiment * volume
        return out

    @staticmethod
    def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        num = pd.to_numeric(numerator, errors="coerce")
        den = pd.to_numeric(denominator, errors="coerce")
        out = pd.Series(np.nan, index=num.index, dtype="float64")
        valid = den.notna() & (den != 0) & num.notna()
        out.loc[valid] = (num.loc[valid] / den.loc[valid]).astype("float64")
        return out

    @staticmethod
    def _add_fundamental_derived_features(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        def _series_or_nan(col: str) -> pd.Series:
            if col in out.columns:
                return pd.to_numeric(out[col], errors="coerce")
            return pd.Series(np.nan, index=out.index, dtype="float64")

        revenue = _series_or_nan("revenue")
        net_income = _series_or_nan("net_income")
        operating_cf = _series_or_nan("operating_cash_flow")
        equity = _series_or_nan("total_shareholder_equity")
        liabilities = _series_or_nan("total_liabilities")

        out["net_margin"] = BuildTFTDatasetUseCase._safe_ratio(net_income, revenue)
        out["leverage_ratio"] = BuildTFTDatasetUseCase._safe_ratio(liabilities, equity)
        out["cashflow_efficiency"] = BuildTFTDatasetUseCase._safe_ratio(operating_cf, revenue)

        out["revenue_yoy_growth"] = revenue.pct_change(252, fill_method=None)
        out["net_income_yoy_growth"] = net_income.pct_change(252, fill_method=None)
        return out

    @staticmethod
    def _validate_feature_anti_leakage(df: pd.DataFrame) -> None:
        # Same-timestamp deterministic derivations
        if {"high", "low", "candle_range"}.issubset(df.columns):
            expected = pd.to_numeric(df["high"], errors="coerce") - pd.to_numeric(df["low"], errors="coerce")
            actual = pd.to_numeric(df["candle_range"], errors="coerce")
            mask = expected.notna() & actual.notna()
            if mask.any() and not np.allclose(expected[mask], actual[mask], atol=1e-12):
                raise ValueError("Anti-leakage validation failed: candle_range must equal high-low at the same timestamp")

        if {"open", "close", "candle_body"}.issubset(df.columns):
            expected = (pd.to_numeric(df["close"], errors="coerce") - pd.to_numeric(df["open"], errors="coerce")).abs()
            actual = pd.to_numeric(df["candle_body"], errors="coerce")
            mask = expected.notna() & actual.notna()
            if mask.any() and not np.allclose(expected[mask], actual[mask], atol=1e-12):
                raise ValueError("Anti-leakage validation failed: candle_body must equal abs(close-open) at the same timestamp")

        if {"news_volume", "has_news"}.issubset(df.columns):
            expected = (pd.to_numeric(df["news_volume"], errors="coerce").fillna(0) > 0).astype("int64")
            actual = pd.to_numeric(df["has_news"], errors="coerce").fillna(0).astype("int64")
            if not expected.equals(actual):
                raise ValueError("Anti-leakage validation failed: has_news must be derived from news_volume > 0")

        if {"volume_zscore", "volume_spike_flag"}.issubset(df.columns):
            expected = (pd.to_numeric(df["volume_zscore"], errors="coerce") > 3.0).fillna(False).astype("int64")
            actual = pd.to_numeric(df["volume_spike_flag"], errors="coerce").fillna(0).astype("int64")
            if not expected.equals(actual):
                raise ValueError("Anti-leakage validation failed: volume_spike_flag must be derived from volume_zscore > 3.0")

        for col in [
            "volatility_parkinson",
            "volatility_garman_klass",
            "downside_semivolatility",
            "vol_of_vol",
        ]:
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            if (series.dropna() < 0).any():
                raise ValueError(f"Anti-leakage validation failed: {col} must be non-negative")

        if {"volatility_regime"}.issubset(df.columns):
            allowed = {0, 1, 2}
            vals = set(pd.to_numeric(df["volatility_regime"], errors="coerce").dropna().astype("int64").tolist())
            if not vals.issubset(allowed):
                raise ValueError("Anti-leakage validation failed: volatility_regime must be in {0,1,2}")

        if {"trend_regime"}.issubset(df.columns):
            allowed = {-1, 0, 1}
            vals = set(pd.to_numeric(df["trend_regime"], errors="coerce").dropna().astype("int64").tolist())
            if not vals.issubset(allowed):
                raise ValueError("Anti-leakage validation failed: trend_regime must be in {-1,0,1}")

        if {"stress_tail_return_flag"}.issubset(df.columns):
            allowed = {0, 1}
            vals = set(pd.to_numeric(df["stress_tail_return_flag"], errors="coerce").dropna().astype("int64").tolist())
            if not vals.issubset(allowed):
                raise ValueError("Anti-leakage validation failed: stress_tail_return_flag must be in {0,1}")

        if {"sentiment_lag_1", "sentiment_score"}.issubset(df.columns):
            expected = pd.to_numeric(df["sentiment_score"], errors="coerce").shift(1)
            actual = pd.to_numeric(df["sentiment_lag_1"], errors="coerce")
            mask = expected.notna() & actual.notna()
            if mask.any() and not np.allclose(expected[mask], actual[mask], atol=1e-12):
                raise ValueError("Anti-leakage validation failed: sentiment_lag_1 must equal shifted sentiment_score")

        if {"sentiment_lag_3", "sentiment_score"}.issubset(df.columns):
            expected = pd.to_numeric(df["sentiment_score"], errors="coerce").shift(3)
            actual = pd.to_numeric(df["sentiment_lag_3"], errors="coerce")
            mask = expected.notna() & actual.notna()
            if mask.any() and not np.allclose(expected[mask], actual[mask], atol=1e-12):
                raise ValueError("Anti-leakage validation failed: sentiment_lag_3 must equal shifted sentiment_score")

        if {"sentiment_lag_5", "sentiment_score"}.issubset(df.columns):
            expected = pd.to_numeric(df["sentiment_score"], errors="coerce").shift(5)
            actual = pd.to_numeric(df["sentiment_lag_5"], errors="coerce")
            mask = expected.notna() & actual.notna()
            if mask.any() and not np.allclose(expected[mask], actual[mask], atol=1e-12):
                raise ValueError("Anti-leakage validation failed: sentiment_lag_5 must equal shifted sentiment_score")

        if {"sentiment_x_volume", "sentiment_score", "volume"}.issubset(df.columns):
            expected = pd.to_numeric(df["sentiment_score"], errors="coerce") * pd.to_numeric(df["volume"], errors="coerce")
            actual = pd.to_numeric(df["sentiment_x_volume"], errors="coerce")
            mask = expected.notna() & actual.notna()
            if mask.any() and not np.allclose(expected[mask], actual[mask], atol=1e-12):
                raise ValueError("Anti-leakage validation failed: sentiment_x_volume must equal sentiment_score * volume")

        if {"net_margin", "net_income", "revenue"}.issubset(df.columns):
            expected = BuildTFTDatasetUseCase._safe_ratio(df["net_income"], df["revenue"])
            actual = pd.to_numeric(df["net_margin"], errors="coerce")
            mask = expected.notna() & actual.notna()
            if mask.any() and not np.allclose(expected[mask], actual[mask], atol=1e-12):
                raise ValueError("Anti-leakage validation failed: net_margin must equal net_income/revenue")

        if {"leverage_ratio", "total_liabilities", "total_shareholder_equity"}.issubset(df.columns):
            expected = BuildTFTDatasetUseCase._safe_ratio(
                df["total_liabilities"], df["total_shareholder_equity"]
            )
            actual = pd.to_numeric(df["leverage_ratio"], errors="coerce")
            mask = expected.notna() & actual.notna()
            if mask.any() and not np.allclose(expected[mask], actual[mask], atol=1e-12):
                raise ValueError(
                    "Anti-leakage validation failed: leverage_ratio must equal total_liabilities/total_shareholder_equity"
                )

        if {"cashflow_efficiency", "operating_cash_flow", "revenue"}.issubset(df.columns):
            expected = BuildTFTDatasetUseCase._safe_ratio(df["operating_cash_flow"], df["revenue"])
            actual = pd.to_numeric(df["cashflow_efficiency"], errors="coerce")
            mask = expected.notna() & actual.notna()
            if mask.any() and not np.allclose(expected[mask], actual[mask], atol=1e-12):
                raise ValueError(
                    "Anti-leakage validation failed: cashflow_efficiency must equal operating_cash_flow/revenue"
                )

        # As-of guard for fundamentals
        fundamental_cols = [
            "revenue",
            "net_income",
            "operating_cash_flow",
            "total_shareholder_equity",
            "total_liabilities",
        ]
        if "effective_date" in df.columns and "date" in df.columns:
            has_fund = df[fundamental_cols].notna().any(axis=1) if all(c in df.columns for c in fundamental_cols) else pd.Series([False] * len(df))
            if has_fund.any():
                date_left = pd.to_datetime(df.loc[has_fund, "date"], utc=True, errors="coerce")
                date_eff = pd.to_datetime(df.loc[has_fund, "effective_date"], utc=True, errors="coerce")
                bad = date_eff > date_left
                if bool(bad.fillna(False).any()):
                    raise ValueError("Anti-leakage validation failed: fundamental effective_date cannot be after the sample date")

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
            include_latest_before_start=True,
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
        base["date"] = (
            pd.to_datetime(base["timestamp"], utc=True, errors="raise")
            .apply(lambda ts: trading_day_from_timestamp(ts.to_pydatetime(), self.trading_day_policy))
        )
        base["date"] = pd.to_datetime(base["date"], utc=True, errors="raise").dt.normalize()
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
        df["news_volume"] = (
            pd.to_numeric(df["news_volume"], errors="coerce")
            .fillna(0)
            .astype("int64")
        )
        df["has_news"] = (df["news_volume"] > 0).astype("int64")
        df["sentiment_score"] = (
            pd.to_numeric(df["sentiment_score"], errors="coerce")
            .fillna(0.0)
            .astype("float64")
        )
        df["sentiment_std"] = (
            pd.to_numeric(df["sentiment_std"], errors="coerce")
            .fillna(0.0)
            .astype("float64")
        )
        df = self._add_phase_a_derived_features(df)
        df = self._add_sentiment_dynamic_features(df)

        # Merge fundamentals with as-of join. We rename `effective_date` to
        # `fundamentals_effective_date` only at the merge boundary so the
        # column persists in the final dataset for source-level auditability
        # (Stage 14 / audit M3-Q4) while the internal name in `fundamentals_df`
        # construction stays untouched.
        if not fundamentals_df.empty:
            df = df.sort_values("date")
            fundamentals_df = fundamentals_df.rename(
                columns={"effective_date": "fundamentals_effective_date"}
            ).sort_values("fundamentals_effective_date")
            df = pd.merge_asof(
                df,
                fundamentals_df,
                left_on="date",
                right_on="fundamentals_effective_date",
                direction="backward",
            )
            if "fundamentals_effective_date" in df.columns:
                mask = df["fundamentals_effective_date"].notna()
                if mask.any() and (
                    df.loc[mask, "fundamentals_effective_date"]
                    > df.loc[mask, "date"]
                ).any():
                    raise ValueError(
                        "Anti-leakage violation: fundamentals_effective_date "
                        "posterior to sample date (Stage 14 / A_code_audit.md "
                        "§M3-Q4)."
                    )
        else:
            for col in [
                "revenue",
                "net_income",
                "operating_cash_flow",
                "total_shareholder_equity",
                "total_liabilities",
            ]:
                df[col] = pd.NA

        # Fundamental derived features must be computed after as-of merge so
        # base fundamental columns are available in the aligned daily frame.
        df = self._add_fundamental_derived_features(df)

        self._validate_feature_anti_leakage(df)

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

        model_feature_cols = [
            c
            for c in df.columns
            if c
            not in {
                "asset_id",
                "timestamp",
                "time_idx",
                "day_of_week",
                "month",
                "target_return",
            }
        ]
        DatasetQualityGate.validate(
            df=df,
            feature_cols=model_feature_cols,
            config=self.quality_gate_config,
            context="build_dataset",
            warmup_counts=FEATURE_WARMUP_BARS,
        )

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
