"""Gold builders modularization (ADR-0005).

26 gold builders previously living inside
`RefreshAnalyticsStoreUseCase` (~2.568 LOC) extracted into per-category
classes implementing `GoldBuilder.build(snapshot, ctx) -> pd.DataFrame`.
"""
from __future__ import annotations

from src.domain.services.gold_builders.base import (
    AnalyticsSnapshot,
    BuildContext,
    GoldBuilder,
    GoldBuildersRegistry,
)

__all__ = [
    "AnalyticsSnapshot",
    "BuildContext",
    "GoldBuilder",
    "GoldBuildersRegistry",
]
