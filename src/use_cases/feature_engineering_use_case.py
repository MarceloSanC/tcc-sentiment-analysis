# src/use_cases/feature_engineering_use_case.py
from pathlib import Path
import pandas as pd

from src.adapters.technical_indicator_calculator import (
    TechnicalIndicatorCalculator,
)
from src.entities.feature_set import FeatureSet


class FeatureEngineeringUseCase:
    def __init__(
        self,
        input_dir: str = "data/raw",
        output_dir: str = "data/processed",
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.indicator_calculator = TechnicalIndicatorCalculator()

    def execute(self, symbol: str) -> FeatureSet:
        clean_symbol = symbol.split(".")[0].upper()
        input_path = self.input_dir / f"candles_{clean_symbol}_1d.parquet"

        if not input_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

        # 1. Load
        df = pd.read_parquet(input_path)

        # 2. Sort + basic checks
        df = df.sort_values("timestamp").reset_index(drop=True)

        # 3. Indicators
        df_features = self.indicator_calculator.calculate(df)

        # 4. Drop NaNs iniciais (indicadores)
        df_features = df_features.dropna().reset_index(drop=True)

        # 5. Persist
        output_path = self.output_dir / f"features_{clean_symbol}.parquet"
        df_features.to_parquet(output_path, index=False)

        return FeatureSet(df=df_features)
