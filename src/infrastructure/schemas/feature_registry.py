from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    group: str
    source_cols: tuple[str, ...]
    formula_desc: str
    anti_leakage_tag: str
    warmup_count: int = 0
    null_policy: str = "allow"
    dtype: str = "float64"
    enabled_by_default: bool = True


FEATURE_REGISTRY: dict[str, FeatureSpec] = {
    # Baseline
    "open": FeatureSpec(
        name="open",
        group="baseline",
        source_cols=("open",),
        formula_desc="Observed open price at timestamp t",
        anti_leakage_tag="point_in_time_ohlcv",
    ),
    "high": FeatureSpec(
        name="high",
        group="baseline",
        source_cols=("high",),
        formula_desc="Observed high price at timestamp t",
        anti_leakage_tag="point_in_time_ohlcv",
    ),
    "low": FeatureSpec(
        name="low",
        group="baseline",
        source_cols=("low",),
        formula_desc="Observed low price at timestamp t",
        anti_leakage_tag="point_in_time_ohlcv",
    ),
    "close": FeatureSpec(
        name="close",
        group="baseline",
        source_cols=("close",),
        formula_desc="Observed close price at timestamp t",
        anti_leakage_tag="point_in_time_ohlcv",
    ),
    "volume": FeatureSpec(
        name="volume",
        group="baseline",
        source_cols=("volume",),
        formula_desc="Observed traded volume at timestamp t",
        anti_leakage_tag="point_in_time_ohlcv",
    ),
    # Technical
    "volatility_20d": FeatureSpec(
        name="volatility_20d",
        group="technical",
        source_cols=("close",),
        formula_desc="Rolling std of close returns over 20 periods",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=20,
    ),
    "rsi_14": FeatureSpec(
        name="rsi_14",
        group="technical",
        source_cols=("close",),
        formula_desc="Relative Strength Index over 14 periods",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=14,
    ),
    "candle_body": FeatureSpec(
        name="candle_body",
        group="technical",
        source_cols=("open", "close"),
        formula_desc="Absolute candle body abs(close-open) at t",
        anti_leakage_tag="same_timestamp_ohlc_derived",
    ),
    "macd_signal": FeatureSpec(
        name="macd_signal",
        group="technical",
        source_cols=("close",),
        formula_desc="Signal line from MACD(12,26,9)",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=35,
    ),
    "ema_100": FeatureSpec(
        name="ema_100",
        group="technical",
        source_cols=("close",),
        formula_desc="Exponential Moving Average over 100 periods",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=100,
    ),
    "macd": FeatureSpec(
        name="macd",
        group="technical",
        source_cols=("close",),
        formula_desc="MACD line from EMA(12)-EMA(26)",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=26,
    ),
    "ema_10": FeatureSpec(
        name="ema_10",
        group="technical",
        source_cols=("close",),
        formula_desc="Exponential Moving Average over 10 periods",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=10,
    ),
    "ema_200": FeatureSpec(
        name="ema_200",
        group="technical",
        source_cols=("close",),
        formula_desc="Exponential Moving Average over 200 periods",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=200,
    ),
    "ema_50": FeatureSpec(
        name="ema_50",
        group="technical",
        source_cols=("close",),
        formula_desc="Exponential Moving Average over 50 periods",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=50,
    ),
    "candle_range": FeatureSpec(
        name="candle_range",
        group="technical",
        source_cols=("high", "low"),
        formula_desc="Candle range high-low at t",
        anti_leakage_tag="same_timestamp_ohlc_derived",
    ),
    # Sentiment
    "sentiment_score": FeatureSpec(
        name="sentiment_score",
        group="sentiment",
        source_cols=("scored_news",),
        formula_desc="Daily aggregate sentiment score for effective day",
        anti_leakage_tag="publication_cutoff_asof",
    ),
    "news_volume": FeatureSpec(
        name="news_volume",
        group="sentiment",
        source_cols=("scored_news",),
        formula_desc="Daily count of news articles for effective day",
        anti_leakage_tag="publication_cutoff_asof",
    ),
    "sentiment_std": FeatureSpec(
        name="sentiment_std",
        group="sentiment",
        source_cols=("scored_news",),
        formula_desc="Daily std of article sentiment scores",
        anti_leakage_tag="publication_cutoff_asof",
    ),
    "has_news": FeatureSpec(
        name="has_news",
        group="sentiment",
        source_cols=("news_volume",),
        formula_desc="Binary flag 1(news_volume>0) else 0",
        anti_leakage_tag="publication_cutoff_asof",
        dtype="int64",
    ),
    # Fundamental
    "revenue": FeatureSpec(
        name="revenue",
        group="fundamental",
        source_cols=("fundamental_reports",),
        formula_desc="Reported revenue merged with as-of join by fundamentals_effective_date",
        anti_leakage_tag="reported_date_asof",
    ),
    "net_income": FeatureSpec(
        name="net_income",
        group="fundamental",
        source_cols=("fundamental_reports",),
        formula_desc="Reported net income merged with as-of join by fundamentals_effective_date",
        anti_leakage_tag="reported_date_asof",
    ),
    "operating_cash_flow": FeatureSpec(
        name="operating_cash_flow",
        group="fundamental",
        source_cols=("fundamental_reports",),
        formula_desc="Reported operating cash flow merged with as-of join by fundamentals_effective_date",
        anti_leakage_tag="reported_date_asof",
    ),
    "total_shareholder_equity": FeatureSpec(
        name="total_shareholder_equity",
        group="fundamental",
        source_cols=("fundamental_reports",),
        formula_desc="Reported shareholder equity merged with as-of join by fundamentals_effective_date",
        anti_leakage_tag="reported_date_asof",
    ),
    "total_liabilities": FeatureSpec(
        name="total_liabilities",
        group="fundamental",
        source_cols=("fundamental_reports",),
        formula_desc="Reported liabilities merged with as-of join by fundamentals_effective_date",
        anti_leakage_tag="reported_date_asof",
    ),
    # Derived (from existing sources only)
    "log_return_5d": FeatureSpec(
        name="log_return_5d",
        group="derived",
        source_cols=("close",),
        formula_desc="log(close_t / close_t-5)",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=5,
    ),
    "log_return_1d": FeatureSpec(
        name="log_return_1d",
        group="derived",
        source_cols=("close",),
        formula_desc="log(close_t / close_t-1)",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=1,
    ),
    "log_return_21d": FeatureSpec(
        name="log_return_21d",
        group="derived",
        source_cols=("close",),
        formula_desc="log(close_t / close_t-21)",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=21,
    ),
    "momentum_5d": FeatureSpec(
        name="momentum_5d",
        group="derived",
        source_cols=("close",),
        formula_desc="close_t / close_t-5 - 1",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=5,
    ),
    "momentum_21d": FeatureSpec(
        name="momentum_21d",
        group="derived",
        source_cols=("close",),
        formula_desc="close_t / close_t-21 - 1",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=21,
    ),
    "momentum_63d": FeatureSpec(
        name="momentum_63d",
        group="derived",
        source_cols=("close",),
        formula_desc="close_t / close_t-63 - 1",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=63,
    ),
    "reversal_1d": FeatureSpec(
        name="reversal_1d",
        group="derived",
        source_cols=("close",),
        formula_desc="-1 * return_1d",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=1,
    ),
    "reversal_5d": FeatureSpec(
        name="reversal_5d",
        group="derived",
        source_cols=("close",),
        formula_desc="-1 * momentum_5d",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=5,
    ),
    "drawdown_lookback": FeatureSpec(
        name="drawdown_lookback",
        group="derived",
        source_cols=("close",),
        formula_desc="close_t / rolling_max(close,63)_t - 1",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=63,
    ),
    "amihud_illiquidity_proxy": FeatureSpec(
        name="amihud_illiquidity_proxy",
        group="derived",
        source_cols=("close", "volume"),
        formula_desc="abs(return_1d) / volume_t",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=1,
    ),
    "volume_zscore": FeatureSpec(
        name="volume_zscore",
        group="derived",
        source_cols=("volume",),
        formula_desc="(volume_t - mean(volume_t-20..t-1)) / std(volume_t-20..t-1)",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=20,
    ),
    "volume_spike_flag": FeatureSpec(
        name="volume_spike_flag",
        group="derived",
        source_cols=("volume",),
        formula_desc="1 if volume_zscore > 3 else 0",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=20,
        dtype="int64",
    ),
    "volatility_parkinson": FeatureSpec(
        name="volatility_parkinson",
        group="derived",
        source_cols=("high", "low"),
        formula_desc="sqrt(rolling_mean((ln(high/low)^2),20)/(4*ln(2)))",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=20,
    ),
    "volatility_garman_klass": FeatureSpec(
        name="volatility_garman_klass",
        group="derived",
        source_cols=("open", "high", "low", "close"),
        formula_desc="sqrt(rolling_mean(0.5*ln(h/l)^2-(2ln2-1)*ln(c/o)^2,20))",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=20,
    ),
    "downside_semivolatility": FeatureSpec(
        name="downside_semivolatility",
        group="derived",
        source_cols=("close",),
        formula_desc="sqrt(rolling_mean(min(return_1d,0)^2,20))",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=20,
    ),
    "vol_of_vol": FeatureSpec(
        name="vol_of_vol",
        group="derived",
        source_cols=("volatility_20d",),
        formula_desc="rolling_std(volatility_20d,20)",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=40,
    ),
    "volatility_regime": FeatureSpec(
        name="volatility_regime",
        group="derived",
        source_cols=("volatility_20d",),
        formula_desc="Regime 0/1/2 from trailing terciles of volatility_20d (window 63, shifted)",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=63,
        dtype="int64",
    ),
    "trend_regime": FeatureSpec(
        name="trend_regime",
        group="derived",
        source_cols=("ema_10", "ema_50"),
        formula_desc="Regime -1/0/1 from EMA spread with trailing deadband (window 63, shifted)",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=63,
        dtype="int64",
    ),
    "stress_tail_return_flag": FeatureSpec(
        name="stress_tail_return_flag",
        group="derived",
        source_cols=("close",),
        formula_desc="Flag 1 when return_1d <= trailing q10(return_1d) over 63, shifted",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=63,
        dtype="int64",
    ),
    "sentiment_lag_1": FeatureSpec(
        name="sentiment_lag_1",
        group="derived",
        source_cols=("sentiment_score",),
        formula_desc="sentiment_score shifted by 1",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=1,
    ),
    "sentiment_lag_3": FeatureSpec(
        name="sentiment_lag_3",
        group="derived",
        source_cols=("sentiment_score",),
        formula_desc="sentiment_score shifted by 3",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=3,
    ),
    "sentiment_lag_5": FeatureSpec(
        name="sentiment_lag_5",
        group="derived",
        source_cols=("sentiment_score",),
        formula_desc="sentiment_score shifted by 5",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=5,
    ),
    "sentiment_ema": FeatureSpec(
        name="sentiment_ema",
        group="derived",
        source_cols=("sentiment_score",),
        formula_desc="EMA(span=10) over sentiment_score",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=1,
    ),
    "sentiment_surprise": FeatureSpec(
        name="sentiment_surprise",
        group="derived",
        source_cols=("sentiment_score",),
        formula_desc="sentiment_score - trailing_mean(sentiment_score,5, shifted)",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=5,
    ),
    "sentiment_x_volatility": FeatureSpec(
        name="sentiment_x_volatility",
        group="derived",
        source_cols=("sentiment_score", "volatility_20d"),
        formula_desc="sentiment_score * volatility_20d",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=20,
    ),
    "sentiment_x_volume": FeatureSpec(
        name="sentiment_x_volume",
        group="derived",
        source_cols=("sentiment_score", "volume"),
        formula_desc="sentiment_score * volume",
        anti_leakage_tag="trailing_window_causal",
        warmup_count=0,
    ),
    "net_margin": FeatureSpec(
        name="net_margin",
        group="derived",
        source_cols=("net_income", "revenue"),
        formula_desc="net_income / revenue",
        anti_leakage_tag="reported_date_asof",
        warmup_count=0,
    ),
    "leverage_ratio": FeatureSpec(
        name="leverage_ratio",
        group="derived",
        source_cols=("total_liabilities", "total_shareholder_equity"),
        formula_desc="total_liabilities / total_shareholder_equity",
        anti_leakage_tag="reported_date_asof",
        warmup_count=0,
    ),
    "cashflow_efficiency": FeatureSpec(
        name="cashflow_efficiency",
        group="derived",
        source_cols=("operating_cash_flow", "revenue"),
        formula_desc="operating_cash_flow / revenue",
        anti_leakage_tag="reported_date_asof",
        warmup_count=0,
    ),
    "revenue_yoy_growth": FeatureSpec(
        name="revenue_yoy_growth",
        group="derived",
        source_cols=("revenue",),
        formula_desc="pct_change(revenue, 252) on as-of merged daily series",
        anti_leakage_tag="reported_date_asof",
        warmup_count=252,
    ),
    "net_income_yoy_growth": FeatureSpec(
        name="net_income_yoy_growth",
        group="derived",
        source_cols=("net_income",),
        formula_desc="pct_change(net_income, 252) on as-of merged daily series",
        anti_leakage_tag="reported_date_asof",
        warmup_count=252,
    ),
}


def get_feature_spec(feature_name: str) -> FeatureSpec:
    return FEATURE_REGISTRY[feature_name]


def list_feature_specs(*, group: str | None = None, enabled_only: bool = False) -> list[FeatureSpec]:
    specs = list(FEATURE_REGISTRY.values())
    if group:
        specs = [s for s in specs if s.group == group]
    if enabled_only:
        specs = [s for s in specs if s.enabled_by_default]
    return specs


def feature_registry_hash() -> str:
    entries = []
    for name in sorted(FEATURE_REGISTRY.keys()):
        s = FEATURE_REGISTRY[name]
        entries.append(
            "|".join(
                [
                    s.name,
                    s.group,
                    ",".join(s.source_cols),
                    s.formula_desc,
                    s.anti_leakage_tag,
                    str(s.warmup_count),
                    s.null_policy,
                    s.dtype,
                    str(s.enabled_by_default),
                ]
            )
        )
    payload = "\n".join(entries).encode("utf-8")
    return sha256(payload).hexdigest()
