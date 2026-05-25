from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Literal

from src.domain.services.phase_b_tier_policy import PhaseBTierPolicy

TierName = Literal["tier_1", "tier_2", "refutado"]


@dataclass(frozen=True)
class TierVerdict:
    hypothesis: str
    horizon: int
    tier: TierName
    criteria_passed: dict[str, bool]
    numerical_inputs: dict[str, float]
    justification: str


def _finite(value: float) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _in_band(value: float, band: tuple[float, float]) -> bool:
    return _finite(value) and float(band[0]) <= float(value) <= float(band[1])


def _positive_mpiw(mpiw: float) -> bool:
    return _finite(mpiw) and float(mpiw) > 0.0


def _tier_from_criteria(
    hypothesis: str,
    horizon: int,
    tier1: dict[str, bool],
    tier2: dict[str, bool],
    numerical_inputs: dict[str, float],
) -> TierVerdict:
    if all(tier1.values()):
        return TierVerdict(
            hypothesis=hypothesis,
            horizon=int(horizon),
            tier="tier_1",
            criteria_passed=tier1,
            numerical_inputs=numerical_inputs,
            justification="tier_1 criteria satisfied mechanically",
        )
    if all(tier2.values()):
        return TierVerdict(
            hypothesis=hypothesis,
            horizon=int(horizon),
            tier="tier_2",
            criteria_passed=tier2,
            numerical_inputs=numerical_inputs,
            justification="tier_2 criteria satisfied mechanically",
        )
    merged = {f"tier1_{k}": v for k, v in tier1.items()}
    merged.update({f"tier2_{k}": v for k, v in tier2.items()})
    return TierVerdict(
        hypothesis=hypothesis,
        horizon=int(horizon),
        tier="refutado",
        criteria_passed=merged,
        numerical_inputs=numerical_inputs,
        justification="neither tier_1 nor tier_2 criteria satisfied mechanically",
    )


def classify_h1(
    coverage_q10: float,
    coverage_q50: float,
    coverage_q90: float,
    picp_error: float,
    mpiw: float,
    policy: PhaseBTierPolicy,
    *,
    horizon: int = 0,
) -> TierVerdict:
    tier1 = {
        "coverage_q10_band": _in_band(coverage_q10, policy.tier_1_coverage_q10_band),
        "coverage_q50_band": _in_band(coverage_q50, policy.tier_1_coverage_q50_band),
        "coverage_q90_band": _in_band(coverage_q90, policy.tier_1_coverage_q90_band),
        "picp_error_abs": _finite(picp_error)
        and abs(float(picp_error)) <= policy.tier_1_picp_error_max,
        "mpiw_positive": _positive_mpiw(mpiw),
    }
    tier2 = {
        "coverage_q10_band": _in_band(coverage_q10, policy.tier_2_coverage_q10_band),
        "coverage_q50_band": _in_band(coverage_q50, policy.tier_2_coverage_q50_band),
        "coverage_q90_band": _in_band(coverage_q90, policy.tier_2_coverage_q90_band),
        "picp_error_abs": _finite(picp_error)
        and abs(float(picp_error)) <= policy.tier_2_picp_error_max,
        "mpiw_positive": _positive_mpiw(mpiw),
    }
    return _tier_from_criteria(
        "H1",
        horizon,
        tier1,
        tier2,
        {
            "coverage_q10": float(coverage_q10),
            "coverage_q50": float(coverage_q50),
            "coverage_q90": float(coverage_q90),
            "picp_error": float(picp_error),
            "mpiw": float(mpiw),
        },
    )


def classify_h2a(
    dm_pvalue_adj_holm_zero_return: float,
    delta_pinball_rel_zero_return: float,
    mpiw: float,
    policy: PhaseBTierPolicy,
    *,
    horizon: int = 0,
) -> TierVerdict:
    tier1 = {
        "dm_pvalue_adj_holm": _finite(dm_pvalue_adj_holm_zero_return)
        and float(dm_pvalue_adj_holm_zero_return) < policy.tier_1_dm_pvalue_max,
        "delta_pinball_rel": _finite(delta_pinball_rel_zero_return)
        and float(delta_pinball_rel_zero_return) >= policy.tier_1_delta_pinball_rel_min,
        "mpiw_positive": _positive_mpiw(mpiw),
    }
    tier2 = {
        "dm_pvalue_adj_holm": _finite(dm_pvalue_adj_holm_zero_return)
        and float(dm_pvalue_adj_holm_zero_return) < policy.tier_2_dm_pvalue_max,
        "delta_pinball_rel": _finite(delta_pinball_rel_zero_return)
        and float(delta_pinball_rel_zero_return) >= policy.tier_2_delta_pinball_rel_min,
        "mpiw_positive": _positive_mpiw(mpiw),
    }
    return _tier_from_criteria(
        "H2a",
        horizon,
        tier1,
        tier2,
        {
            "dm_pvalue_adj_holm_zero_return": float(dm_pvalue_adj_holm_zero_return),
            "delta_pinball_rel_zero_return": float(delta_pinball_rel_zero_return),
            "mpiw": float(mpiw),
        },
    )


def classify_h2b(
    dm_pvalues_adj_holm_by_baseline: dict[str, float],
    delta_pinball_rel_by_baseline: dict[str, float],
    mpiw: float,
    policy: PhaseBTierPolicy,
    *,
    horizon: int = 0,
) -> TierVerdict:
    baselines = sorted(set(dm_pvalues_adj_holm_by_baseline) | set(delta_pinball_rel_by_baseline))
    tier1 = {"mpiw_positive": _positive_mpiw(mpiw)}
    tier2 = {"mpiw_positive": _positive_mpiw(mpiw)}
    numerical: dict[str, float] = {"mpiw": float(mpiw)}
    for baseline in baselines:
        pvalue = dm_pvalues_adj_holm_by_baseline.get(baseline, math.nan)
        delta = delta_pinball_rel_by_baseline.get(baseline, math.nan)
        tier1[f"dm_pvalue_adj_holm_{baseline}"] = _finite(pvalue) and (
            float(pvalue) < policy.tier_1_dm_pvalue_max
        )
        tier1[f"delta_pinball_rel_{baseline}"] = _finite(delta) and (
            float(delta) >= policy.tier_1_delta_pinball_rel_min
        )
        tier2[f"dm_pvalue_adj_holm_{baseline}"] = _finite(pvalue) and (
            float(pvalue) < policy.tier_2_dm_pvalue_max
        )
        tier2[f"delta_pinball_rel_{baseline}"] = _finite(delta) and (
            float(delta) >= policy.tier_2_delta_pinball_rel_min
        )
        numerical[f"dm_pvalue_adj_holm_{baseline}"] = float(pvalue)
        numerical[f"delta_pinball_rel_{baseline}"] = float(delta)
    return _tier_from_criteria("H2b", horizon, tier1, tier2, numerical)

