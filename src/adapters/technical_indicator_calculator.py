# src/adapters/technical_indicator_calculator.py
import pandas as pd
import pandas_ta as ta

from src.entities.candle import Candle
from src.entities.feature_set import FeatureSet
from src.interfaces.feature_calculator import FeatureCalculator


class TechnicalIndicatorCalculator(FeatureCalculator):
    def calculate(
        self,
        asset_id: str,
        candles: list[Candle],
    ) -> list[FeatureSet]:
        
        # TODO(clean-arch): extrair conversão Candle -> DataFrame para adapter utilitário
        # TODO(test): validar se candles estão ordenados temporalmente

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

        # TODO(feature-eng): tornar indicadores configuráveis via parâmetros
        # TODO(leakage): validar janelas de cálculo vs horizonte do target

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

        # TODO(nans): definir política explícita de NaNs (drop, fill, flag)
        df = df.dropna()

        # 3. Converter para entidades FeatureSet
        features = []
        for _, row in df.iterrows():
            features.append(
                FeatureSet(
                    asset_id=asset_id,
                    timestamp=row["timestamp"],
                    features={
                        "rsi_14": row["rsi_14"],
                        "macd": row["macd"],
                        "macd_signal": row["macd_signal"],
                        "ema_10": row["ema_10"],
                        "ema_50": row["ema_50"],
                        "ema_100": row["ema_100"],
                        "ema_200": row["ema_200"],
                        "volatility_20d": row["volatility_20d"],
                        "candle_range": row["candle_range"],
                        "candle_body": row["candle_body"],
                    },
                )
            )

        return features
