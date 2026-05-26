from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FoldPeriod:
    name: str
    train_end_utc: pd.Timestamp


@dataclass(frozen=True)
class RunInFoldCandidate:
    run_id: str
    fold_name: str
    train_end_utc: pd.Timestamp
    model_version: str
    seed: int | None


def _to_utc_timestamp(value: Any) -> pd.Timestamp | None:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def select_operationally_latest_fold(
    candidates: list[RunInFoldCandidate],
    target_ts: pd.Timestamp,
    horizon: int,
) -> RunInFoldCandidate | None:
    if int(horizon) < 1:
        raise ValueError("horizon must be >= 1")

    target_utc = _to_utc_timestamp(target_ts)
    if target_utc is None:
        return None
    forecast_origin = target_utc - pd.Timedelta(days=int(horizon))

    eligible: list[RunInFoldCandidate] = []
    for candidate in candidates:
        train_end = _to_utc_timestamp(candidate.train_end_utc)
        if train_end is None:
            continue
        if train_end < forecast_origin:
            eligible.append(
                RunInFoldCandidate(
                    run_id=str(candidate.run_id),
                    fold_name=str(candidate.fold_name),
                    train_end_utc=train_end,
                    model_version=str(candidate.model_version),
                    seed=candidate.seed,
                )
            )

    if not eligible:
        return None

    return sorted(
        eligible,
        key=lambda c: (-c.train_end_utc.value, c.fold_name, c.run_id),
    )[0]

