from __future__ import annotations

from abc import ABC, abstractmethod

from src.entities.fundamental_report import FundamentalReport


class FundamentalFetcher(ABC):
    """
    Interface for fetching fundamentals from external sources.
    """

    @abstractmethod
    def fetch_fundamentals(self, asset_id: str) -> list[FundamentalReport]:
        """Fetch fundamentals for an asset (annual + quarterly)."""
        ...
