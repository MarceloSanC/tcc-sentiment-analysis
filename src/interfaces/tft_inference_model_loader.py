from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LoadedTFTInferenceModel:
    asset_id: str
    version: str
    model_dir: Path
    model: Any
    feature_cols: list[str]
    feature_set_name: str
    feature_tokens: list[str]
    training_config: dict[str, Any]
    training_run_id: str | None = None
    scalers: dict[str, Any] = field(default_factory=dict)
    dataset_parameters: dict[str, Any] = field(default_factory=dict)


class TFTInferenceModelLoader(ABC):
    @abstractmethod
    def load(self, model_dir: str | Path) -> LoadedTFTInferenceModel:
        """Load a trained TFT model and all artifacts required for inference."""
        ...
