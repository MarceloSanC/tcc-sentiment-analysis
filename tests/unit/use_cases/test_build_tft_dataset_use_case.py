# tests/unit/use_cases/test_build_tft_dataset_use_case.py

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.domain.services.dataset_quality_gate import DatasetQualityGateConfig
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
    return datetime(y, m, d, tzinfo=UTC)


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
    has_news = repo.saved.loc[repo.saved["timestamp"] == missing_day, "has_news"].item()
    assert has_news == 0
    sentiment_score = repo.saved.loc[repo.saved["timestamp"] == missing_day, "sentiment_score"].item()
    assert sentiment_score == pytest.approx(0.0)
    sentiment_std = repo.saved.loc[repo.saved["timestamp"] == missing_day, "sentiment_std"].item()
    assert sentiment_std == pytest.approx(0.0)


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


def test_fundamental_derived_features_are_computed_after_fundamental_merge(tmp_path: Path) -> None:
    asset_id = "AAPL"
    repo = FakeTFTDatasetRepository(output_dir=tmp_path)
    use_case = BuildTFTDatasetUseCase(
        candle_repository=FakeCandleRepository(_candles()),
        indicator_repository=FakeTechnicalIndicatorRepository(_indicators(asset_id)),
        daily_sentiment_repository=FakeDailySentimentRepository(_daily_sentiment(asset_id)),
        fundamental_repository=FakeFundamentalRepository(_fundamentals(asset_id)),
        tft_dataset_repository=repo,
    )

    use_case.execute(asset_id, _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 3))
    assert repo.saved is not None
    df = repo.saved

    # Fundamentals are available in this window; derived ratios must be populated.
    assert df["revenue"].notna().any()
    assert df["net_margin"].notna().any()
    assert df["leverage_ratio"].notna().any()
    assert df["cashflow_efficiency"].notna().any()


def test_anti_leakage_validator_rejects_invalid_candle_derivations() -> None:
    df = pd.DataFrame(
        {
            "high": [11.0],
            "low": [10.0],
            "candle_range": [2.0],  # should be 1.0
            "open": [10.0],
            "close": [10.5],
            "candle_body": [0.5],
            "news_volume": [1],
            "has_news": [1],
        }
    )
    with pytest.raises(ValueError, match="candle_range must equal high-low"):
        BuildTFTDatasetUseCase._validate_feature_anti_leakage(df)


def test_anti_leakage_validator_rejects_future_fundamental_effective_date() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"], utc=True),
            "effective_date": pd.to_datetime(["2024-01-02"], utc=True),
            "revenue": [100.0],
            "net_income": [10.0],
            "operating_cash_flow": [9.0],
            "total_shareholder_equity": [50.0],
            "total_liabilities": [40.0],
        }
    )
    with pytest.raises(ValueError, match="fundamental effective_date cannot be after"):
        BuildTFTDatasetUseCase._validate_feature_anti_leakage(df)


def test_phase_a_derived_features_are_generated_causally() -> None:
    n = 80
    df = pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.0 + i for i in range(n)],
            "volume": [1000.0 + (i % 10) * 10 for i in range(n)],
            "sentiment_score": [(-1.0) ** i * 0.1 for i in range(n)],
        }
    )

    out = BuildTFTDatasetUseCase._add_phase_a_derived_features(df)
    out = BuildTFTDatasetUseCase._add_sentiment_dynamic_features(out)
    out = BuildTFTDatasetUseCase._add_fundamental_derived_features(out)

    for col in [
        "log_return_1d",
        "log_return_5d",
        "log_return_21d",
        "momentum_5d",
        "momentum_21d",
        "momentum_63d",
        "reversal_1d",
        "reversal_5d",
        "drawdown_lookback",
        "amihud_illiquidity_proxy",
        "volume_zscore",
        "volume_spike_flag",
        "volatility_parkinson",
        "volatility_garman_klass",
        "downside_semivolatility",
        "vol_of_vol",
        "volatility_regime",
        "trend_regime",
        "stress_tail_return_flag",
        "sentiment_lag_1",
        "sentiment_lag_3",
        "sentiment_lag_5",
        "sentiment_ema",
        "sentiment_surprise",
        "sentiment_x_volatility",
        "sentiment_x_volume",
        "net_margin",
        "leverage_ratio",
        "cashflow_efficiency",
        "revenue_yoy_growth",
        "net_income_yoy_growth",
    ]:
        assert col in out.columns

    # Warmup behavior: early rows must be NaN for long-window features.
    assert pd.isna(out.loc[20, "momentum_63d"])
    assert not pd.isna(out.loc[70, "momentum_63d"])
    assert pd.isna(out.loc[0, "log_return_1d"])
    assert pd.isna(out.loc[4, "log_return_5d"])
    assert pd.isna(out.loc[61, "drawdown_lookback"])
    assert out.loc[70, "drawdown_lookback"] <= 0.0
    assert pd.isna(out.loc[18, "volatility_parkinson"])
    assert not pd.isna(out.loc[40, "volatility_parkinson"])
    assert (pd.to_numeric(out["volatility_parkinson"], errors="coerce").dropna() >= 0).all()
    assert (pd.to_numeric(out["volatility_garman_klass"], errors="coerce").dropna() >= 0).all()
    assert (pd.to_numeric(out["downside_semivolatility"], errors="coerce").dropna() >= 0).all()
    assert (pd.to_numeric(out["vol_of_vol"], errors="coerce").dropna() >= 0).all()
    vol_reg = pd.to_numeric(out["volatility_regime"], errors="coerce").dropna().astype("int64")
    if not vol_reg.empty:
        assert set(vol_reg.tolist()).issubset({0, 1, 2})
    tr_reg = pd.to_numeric(out["trend_regime"], errors="coerce").dropna().astype("int64")
    if not tr_reg.empty:
        assert set(tr_reg.tolist()).issubset({-1, 0, 1})
    stress = pd.to_numeric(out["stress_tail_return_flag"], errors="coerce").dropna().astype("int64")
    if not stress.empty:
        assert set(stress.tolist()).issubset({0, 1})
    # Sentiment dynamics
    s = pd.to_numeric(out["sentiment_score"], errors="coerce")
    s_l1 = pd.to_numeric(out["sentiment_lag_1"], errors="coerce")
    s_l3 = pd.to_numeric(out["sentiment_lag_3"], errors="coerce")
    s_l5 = pd.to_numeric(out["sentiment_lag_5"], errors="coerce")
    assert pd.isna(s_l1.iloc[0])
    assert pd.isna(s_l3.iloc[2])
    assert pd.isna(s_l5.iloc[4])
    assert s_l1.iloc[10] == pytest.approx(s.iloc[9])
    assert s_l3.iloc[10] == pytest.approx(s.iloc[7])
    assert s_l5.iloc[10] == pytest.approx(s.iloc[5])
    sxv = pd.to_numeric(out["sentiment_x_volume"], errors="coerce")
    vol = pd.to_numeric(out["volume"], errors="coerce")
    mask = s.notna() & vol.notna() & sxv.notna()
    if mask.any():
        assert np.allclose(sxv[mask], (s * vol)[mask], atol=1e-12)
    # Fundamental derived ratios
    n_yoy = 320
    rev = pd.Series([100.0 + i for i in range(n_yoy)], dtype="float64")
    ni = pd.Series([10.0 + 0.1 * i for i in range(n_yoy)], dtype="float64")
    ocf = pd.Series([20.0 + 0.2 * i for i in range(n_yoy)], dtype="float64")
    eq = pd.Series([50.0 + 0.5 * i for i in range(n_yoy)], dtype="float64")
    liab = pd.Series([30.0 + 0.3 * i for i in range(n_yoy)], dtype="float64")
    base_f = pd.DataFrame(
        {
            "revenue": rev,
            "net_income": ni,
            "operating_cash_flow": ocf,
            "total_shareholder_equity": eq,
            "total_liabilities": liab,
        }
    )
    f_out = BuildTFTDatasetUseCase._add_fundamental_derived_features(base_f)
    assert np.isclose(f_out.loc[10, "net_margin"], ni.iloc[10] / rev.iloc[10])
    assert np.isclose(f_out.loc[10, "leverage_ratio"], liab.iloc[10] / eq.iloc[10])
    assert np.isclose(f_out.loc[10, "cashflow_efficiency"], ocf.iloc[10] / rev.iloc[10])
    assert pd.isna(f_out.loc[200, "revenue_yoy_growth"])
    assert not pd.isna(f_out.loc[260, "revenue_yoy_growth"])

    # Reversal definitions
    assert out.loc[10, "reversal_5d"] == pytest.approx(-out.loc[10, "momentum_5d"])

    # Spike flag must match zscore threshold rule.
    expected_flag = (pd.to_numeric(out["volume_zscore"], errors="coerce") > 3.0).fillna(False).astype("int64")
    assert expected_flag.equals(out["volume_spike_flag"].astype("int64"))


def test_build_dataset_is_deterministic_for_same_input(tmp_path: Path) -> None:
    asset_id = "AAPL"
    repo = FakeTFTDatasetRepository(output_dir=tmp_path)
    use_case = BuildTFTDatasetUseCase(
        candle_repository=FakeCandleRepository(_candles()),
        indicator_repository=FakeTechnicalIndicatorRepository(_indicators(asset_id)),
        daily_sentiment_repository=FakeDailySentimentRepository(_daily_sentiment(asset_id)),
        fundamental_repository=FakeFundamentalRepository(_fundamentals(asset_id)),
        tft_dataset_repository=repo,
    )
    use_case.execute(asset_id, _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 3))
    first = repo.saved.copy()
    use_case.execute(asset_id, _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 3))
    second = repo.saved.copy()
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_build_dataset_schema_contract_basic(tmp_path: Path) -> None:
    asset_id = "AAPL"
    repo = FakeTFTDatasetRepository(output_dir=tmp_path)
    use_case = BuildTFTDatasetUseCase(
        candle_repository=FakeCandleRepository(_candles()),
        indicator_repository=FakeTechnicalIndicatorRepository(_indicators(asset_id)),
        daily_sentiment_repository=FakeDailySentimentRepository(_daily_sentiment(asset_id)),
        fundamental_repository=FakeFundamentalRepository(_fundamentals(asset_id)),
        tft_dataset_repository=repo,
    )
    use_case.execute(asset_id, _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 3))
    assert repo.saved is not None
    df = repo.saved
    required = {
        "timestamp",
        "asset_id",
        "time_idx",
        "target_return",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    assert required.issubset(df.columns)
    assert df["timestamp"].is_monotonic_increasing
    assert not df["timestamp"].duplicated().any()
    assert pd.api.types.is_integer_dtype(df["time_idx"])
    for col in ["open", "high", "low", "close", "volume", "target_return"]:
        assert pd.api.types.is_numeric_dtype(df[col])


def test_dataset_preserves_fundamentals_effective_date_column(tmp_path: Path) -> None:
    asset_id = "AAPL"
    repo = FakeTFTDatasetRepository(output_dir=tmp_path)
    use_case = BuildTFTDatasetUseCase(
        candle_repository=FakeCandleRepository(_candles()),
        indicator_repository=FakeTechnicalIndicatorRepository(_indicators(asset_id)),
        daily_sentiment_repository=FakeDailySentimentRepository(_daily_sentiment(asset_id)),
        fundamental_repository=FakeFundamentalRepository(_fundamentals(asset_id)),
        tft_dataset_repository=repo,
    )

    use_case.execute(asset_id, _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 3))

    assert repo.saved is not None
    df = repo.saved
    assert "fundamentals_effective_date" in df.columns
    assert df["fundamentals_effective_date"].notna().any()


def test_fundamentals_effective_date_never_after_sample_date(tmp_path: Path) -> None:
    asset_id = "AAPL"
    repo = FakeTFTDatasetRepository(output_dir=tmp_path)
    use_case = BuildTFTDatasetUseCase(
        candle_repository=FakeCandleRepository(_candles()),
        indicator_repository=FakeTechnicalIndicatorRepository(_indicators(asset_id)),
        daily_sentiment_repository=FakeDailySentimentRepository(_daily_sentiment(asset_id)),
        fundamental_repository=FakeFundamentalRepository(_fundamentals(asset_id)),
        tft_dataset_repository=repo,
    )

    use_case.execute(asset_id, _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 3))
    assert repo.saved is not None
    df = repo.saved
    mask = df["fundamentals_effective_date"].notna()
    timestamps = pd.to_datetime(df.loc[mask, "timestamp"], utc=True).dt.normalize()
    eff = pd.to_datetime(df.loc[mask, "fundamentals_effective_date"], utc=True)
    assert (eff <= timestamps).all()


def test_dataset_does_not_have_effective_date_column(tmp_path: Path) -> None:
    asset_id = "AAPL"
    repo = FakeTFTDatasetRepository(output_dir=tmp_path)
    use_case = BuildTFTDatasetUseCase(
        candle_repository=FakeCandleRepository(_candles()),
        indicator_repository=FakeTechnicalIndicatorRepository(_indicators(asset_id)),
        daily_sentiment_repository=FakeDailySentimentRepository(_daily_sentiment(asset_id)),
        fundamental_repository=FakeFundamentalRepository(_fundamentals(asset_id)),
        tft_dataset_repository=repo,
    )

    use_case.execute(asset_id, _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 3))
    assert repo.saved is not None
    assert "effective_date" not in repo.saved.columns


def test_build_dataset_quality_gate_rejects_nan_ratio_above_threshold(tmp_path: Path) -> None:
    asset_id = "AAPL"
    indicators = _indicators(asset_id)
    # Non-warmup feature with NaN in middle row must still fail quality gate.
    mid = len(indicators) // 2
    indicators[mid].indicators["candle_body"] = float("nan")
    repo = FakeTFTDatasetRepository(output_dir=tmp_path)
    use_case = BuildTFTDatasetUseCase(
        candle_repository=FakeCandleRepository(_candles()),
        indicator_repository=FakeTechnicalIndicatorRepository(indicators),
        daily_sentiment_repository=FakeDailySentimentRepository(_daily_sentiment(asset_id)),
        fundamental_repository=FakeFundamentalRepository(_fundamentals(asset_id)),
        tft_dataset_repository=repo,
        quality_gate_config=DatasetQualityGateConfig(
            max_nan_ratio_per_feature=0.0,
            min_temporal_coverage_days=1,
            require_unique_timestamps=True,
            require_monotonic_timestamps=True,
        ),
    )

    with pytest.raises(ValueError, match="build_dataset quality gate failed: feature NaN ratio"):
        use_case.execute(asset_id, _dt_utc(2024, 1, 1), _dt_utc(2024, 1, 3))


def test_fundamentals_effective_date_applies_45d_fallback_when_reported_date_missing(
    tmp_path: Path,
) -> None:
    asset_id = "AAPL"
    # fiscal_date_end Fri 2023-12-01 + 45 days = Mon 2024-01-15 (weekday, no roll).
    fiscal_end = date(2023, 12, 1)
    expected_fallback = fiscal_end + timedelta(days=45)

    candles = [
        Candle(
            timestamp=_dt_utc(2024, 1, day),
            open=100, high=101, low=99, close=100 + day, volume=10 + day,
        )
        for day in (15, 16, 17)
    ]
    indicators = [
        TechnicalIndicatorSet(
            asset_id=asset_id,
            timestamp=_dt_utc(2024, 1, day),
            indicators={"rsi_14": 30.0 + day},
        )
        for day in (15, 16, 17)
    ]
    fundamentals = [
        FundamentalReport(
            asset_id=asset_id,
            fiscal_date_end=fiscal_end,
            report_type="annual",
            revenue=1000.0,
            net_income=200.0,
            operating_cash_flow=150.0,
            total_shareholder_equity=300.0,
            total_liabilities=400.0,
            reported_date=None,  # triggers +45d fallback (use case :117-118)
            source="mock",
        )
    ]

    repo = FakeTFTDatasetRepository(output_dir=tmp_path)
    use_case = BuildTFTDatasetUseCase(
        candle_repository=FakeCandleRepository(candles),
        indicator_repository=FakeTechnicalIndicatorRepository(indicators),
        daily_sentiment_repository=FakeDailySentimentRepository([]),
        fundamental_repository=FakeFundamentalRepository(fundamentals),
        tft_dataset_repository=repo,
    )

    use_case.execute(asset_id, _dt_utc(2024, 1, 15), _dt_utc(2024, 1, 17))

    assert repo.saved is not None
    df = repo.saved
    eff = pd.to_datetime(df["fundamentals_effective_date"], utc=True).dt.date
    # At least one row must carry the +45d fallback effective_date exactly
    # (fiscal_end + 45d lands on a weekday; no business-day roll applied).
    assert (eff == expected_fallback).any()
