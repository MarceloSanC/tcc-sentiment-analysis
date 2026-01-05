# src/infrastructure/schemas/technical_indicators_schema.py

TECHNICAL_INDICATORS = {
    # Momentum
    "rsi_14",

    # Trend
    "ema_10",
    "ema_50",
    "ema_100",
    "ema_200",

    # MACD
    "macd",
    "macd_signal",

    # Volatility
    "volatility_20d",

    # Price action
    "candle_range",
    "candle_body",
}

# Opcional: metadata para observabilidade futura
TECHNICAL_INDICATOR_METADATA = {
    "rsi_14": {
        "type": "momentum",
        "window": 14,
        "source": "close",
    },
    "ema_10": {
        "type": "trend",
        "window": 10,
        "source": "close",
    },
    "ema_50": {
        "type": "trend",
        "window": 50,
        "source": "close",
    },
    "ema_100": {
        "type": "trend",
        "window": 100,
        "source": "close",
    },
    "ema_200": {
        "type": "trend",
        "window": 200,
        "source": "close",
    },
    "macd": {
        "type": "momentum",
        "source": "close",
    },
    "macd_signal": {
        "type": "momentum",
        "source": "close",
    },
    "volatility_20d": {
        "type": "volatility",
        "window": 20,
        "source": "close",
    },
    "candle_range": {
        "type": "price_action",
        "source": "high-low",
    },
    "candle_body": {
        "type": "price_action",
        "source": "open-close",
    },
}
