from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelRepository(ABC):
    @abstractmethod
    def save_training_artifacts(
        self,
        asset_id: str,
        version: str,
        model: Any,
        *,
        metrics: dict[str, float],
        history: list[dict[str, float]],
        features_used: list[str],
        training_window: dict[str, str],
        config: dict,
        plots: dict[str, str] | None = None,
    ) -> str:
        raise NotImplementedError
