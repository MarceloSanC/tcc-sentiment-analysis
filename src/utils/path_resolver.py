# src/utils/path_resolver.py
import os
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

def load_data_paths() -> dict:
    with open("config/data_paths.yaml") as f:
        paths = yaml.safe_load(f)

    root_env = os.getenv("DATA_ROOT")
    root = Path(root_env) if root_env else None

    logger.info("Resolved data paths | root=%s", root)

    def resolve(p: str) -> Path:
        return root / Path(p) if root else Path(p)

    return {
        "raw_candles": resolve(paths["data"]["raw"]["candles"]),
        "processed_features": resolve(paths["data"]["processed"]["features"]),
        "dataset_tft": resolve(paths["data"]["processed"]["dataset_tft"]),
        "models": resolve(paths["data"]["models"]["tft"]),
    }
