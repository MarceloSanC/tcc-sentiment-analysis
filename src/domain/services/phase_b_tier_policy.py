from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseBTierPolicy:
    tier_1_coverage_q10_band: tuple[float, float] = (0.07, 0.13)
    tier_1_coverage_q50_band: tuple[float, float] = (0.45, 0.55)
    tier_1_coverage_q90_band: tuple[float, float] = (0.87, 0.93)
    tier_1_picp_error_max: float = 0.02
    tier_1_delta_pinball_rel_min: float = 0.03
    tier_1_dm_pvalue_max: float = 0.05

    tier_2_coverage_q10_band: tuple[float, float] = (0.05, 0.15)
    tier_2_coverage_q50_band: tuple[float, float] = (0.40, 0.60)
    tier_2_coverage_q90_band: tuple[float, float] = (0.85, 0.95)
    tier_2_picp_error_max: float = 0.05
    tier_2_delta_pinball_rel_min: float = 0.0
    tier_2_dm_pvalue_max: float = 0.10

    h2a_primary_baseline: str = "baseline_zero_return_v1"
    confirmatory_horizons: tuple[int, int] = (1, 7)
    baseline_model_versions: tuple[str, str, str] = (
        "baseline_zero_return_v1",
        "baseline_historical_mean_rolling_v1",
        "baseline_historical_quantiles_rolling_v1",
    )
    nominal_picp_q10_q90: float = 0.80


def default_phase_b_policy() -> PhaseBTierPolicy:
    return PhaseBTierPolicy()

