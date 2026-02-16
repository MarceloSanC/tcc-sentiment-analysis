from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd


@dataclass(frozen=True)
class WarmupNullSegment:
    feature_name: str
    num_null: int
    first_date_warmup: str
    last_date_warmup_null: str
    requested_start: str
    requested_end: str


class FeatureWarmupInspector:
    @staticmethod
    def _parse_yyyymmdd(value: str) -> datetime:
        return datetime.strptime(value, "%Y%m%d").replace(tzinfo=timezone.utc)

    @staticmethod
    def detect_leading_null_warmups(
        df: pd.DataFrame,
        feature_cols: list[str],
        *,
        requested_start: str,
        requested_end: str,
        timestamp_col: str = "timestamp",
    ) -> list[WarmupNullSegment]:
        if df.empty or timestamp_col not in df.columns:
            return []

        ts = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
        start_utc = FeatureWarmupInspector._parse_yyyymmdd(requested_start)
        end_utc = FeatureWarmupInspector._parse_yyyymmdd(requested_end)

        scoped = df.copy()
        scoped["_ts"] = ts
        scoped = scoped[(scoped["_ts"] >= start_utc) & (scoped["_ts"] <= end_utc)]
        if scoped.empty:
            return []
        scoped = scoped.sort_values("_ts").reset_index(drop=True)

        requested_period_start = start_utc.strftime("%Y-%m-%d")
        requested_period_end = end_utc.strftime("%Y-%m-%d")
        segments: list[WarmupNullSegment] = []

        for feature_name in feature_cols:
            if feature_name not in scoped.columns:
                continue
            series = scoped[feature_name]
            if series.empty or not series.isna().iloc[0]:
                continue

            warmup_len = 0
            for is_null in series.isna().tolist():
                if is_null:
                    warmup_len += 1
                else:
                    break
            if warmup_len <= 0:
                continue

            first_warmup_date = scoped.loc[0, "_ts"].strftime("%Y-%m-%d")
            last_warmup_date = scoped.loc[warmup_len - 1, "_ts"].strftime("%Y-%m-%d")

            segments.append(
                WarmupNullSegment(
                    feature_name=feature_name,
                    num_null=warmup_len,
                    first_date_warmup=first_warmup_date,
                    last_date_warmup_null=last_warmup_date,
                    requested_start=requested_period_start,
                    requested_end=requested_period_end,
                )
            )

        return segments

