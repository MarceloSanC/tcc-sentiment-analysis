from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class TFTDatasetRepository(ABC):
    @abstractmethod
    def save(self, asset_id: str, df: pd.DataFrame) -> None:
        """Persist TFT dataset for the asset."""
        ...

    @abstractmethod
    def load(self, asset_id: str) -> pd.DataFrame:
        """Load TFT dataset for the asset."""
        ...
