from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pandas as pd


class PhaseBTierSidecarWriter(Protocol):
    def write_marginal_coverage(self, df: pd.DataFrame, cohort_id: str) -> Path: ...

    def write_dm_family_6(self, df: pd.DataFrame, cohort_id: str) -> Path: ...

    def write_dm_family_18_sensitivity(
        self,
        df: pd.DataFrame,
        cohort_id: str,
    ) -> Path: ...

    def write_delta_pinball(self, df: pd.DataFrame, cohort_id: str) -> Path: ...

    def write_tier_verdict(self, df: pd.DataFrame, cohort_id: str) -> Path: ...

