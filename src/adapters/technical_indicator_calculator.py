# src/adapters/technical_indicator_calculator.py
import pandas as pd
import pandas_ta as ta

from src.entities.candle import Candle
from src.entities.technical_indicator_set import TechnicalIndicatorSet
from src.interfaces.technical_indicator_calculator import TechnicalIndicatorCalculatorPort

from src.infrastructure.schemas.technical_indicators_schema import (
    TECHNICAL_INDICATORS,
)


class TechnicalIndicatorCalculator(TechnicalIndicatorCalculatorPort):
    """
    Adapter responsável por calcular indicadores técnicos
    a partir de candles OHLCV.

    Utiliza bibliotecas externas (pandas, pandas-ta),
    portanto pertence à camada de adapters.
    """

    def calculate(
        self,
        asset_id: str,
        candles: list[Candle],
    ) -> list[TechnicalIndicatorSet]:

        # 1. Converter para DataFrame (infra detail)
        df = pd.DataFrame(
            [
                {
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                }
                for c in candles
            ]
        ).sort_values("timestamp")

        # 2. Indicadores técnicos
        df["rsi_14"] = ta.rsi(df["close"], length=14)

        macd = ta.macd(df["close"])
        df["macd"] = macd["MACD_12_26_9"]
        df["macd_signal"] = macd["MACDs_12_26_9"]

        for length in (10, 50, 100, 200):
            df[f"ema_{length}"] = ta.ema(df["close"], length=length)

        df["volatility_20d"] = df["close"].pct_change().rolling(20).std()
        df["candle_range"] = df["high"] - df["low"]
        df["candle_body"] = (df["close"] - df["open"]).abs()

        df = df.dropna()

        missing = TECHNICAL_INDICATORS - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Missing technical indicators: {missing}"
            )

        # 3. Converter para entidades TechnicalIndicatorSet
        indicators = []
        for _, row in df.iterrows():
            indicators.append(
                TechnicalIndicatorSet(
                    asset_id=asset_id,
                    timestamp=row["timestamp"],
                    indicators={
                        name: row[name]
                        for name in TECHNICAL_INDICATORS
                    },
                )
            )

        return indicators


# =========================
# TODOs — melhorias futuras
# =========================

# TODO(architecture):
# Extrair conversão Candle -> DataFrame
# para um utilitário compartilhado de adapters

# TODO(stat-validation):
# Validar ordenação temporal explícita dos candles
# antes do cálculo de indicadores

# TODO(feature-engineering):
# Tornar lista de indicadores configurável
# via parâmetros ou arquivo de configuração

# TODO(normalization & leakage):
# Validar janelas de cálculo dos indicadores
# versus horizonte do target para evitar leakage

# TODO(feature-engineering):
# Separar indicadores por categoria:
# trend, momentum, volatility, volume

# TODO(data-quality):
# Definir política explícita para NaNs:
# drop | fill | flag_missing

# TODO(performance):
# Avaliar vetorização adicional
# ou uso de numba para indicadores customizados
