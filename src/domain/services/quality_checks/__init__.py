"""Quality check registry for analytics validation.

Per ADR-0004, quality checks live as small `QualityCheck` classes
registered in a `QualityCheckRegistry`. The orchestrator (use case)
loads the data snapshot once, then iterates the registry per scope.
"""
from __future__ import annotations

from src.domain.services.quality_checks.base import (
    AnalyticsSnapshot,
    CheckResult,
    QualityCheck,
    QualityCheckRegistry,
)

__all__ = [
    "AnalyticsSnapshot",
    "CheckResult",
    "QualityCheck",
    "QualityCheckRegistry",
]
