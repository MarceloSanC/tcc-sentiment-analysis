from __future__ import annotations

from typing import Dict, List

TFT_DATASET_BASE_COLUMNS: List[str] = [
    "asset_id",
    "timestamp",
    "time_idx",
    "day_of_week",
    "month",
    "target_return",
]

TFT_DATASET_DTYPES: Dict[str, str] = {
    "asset_id": "string",
    "time_idx": "int64",
    "day_of_week": "int64",
    "month": "int64",
    "target_return": "float64",
}
