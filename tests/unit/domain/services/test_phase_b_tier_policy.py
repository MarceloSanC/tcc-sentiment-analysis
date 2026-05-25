from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.domain.services.phase_b_tier_policy import (
    PhaseBTierPolicy,
    default_phase_b_policy,
)


def test_default_policy_is_frozen() -> None:
    policy = default_phase_b_policy()
    with pytest.raises(FrozenInstanceError):
        policy.tier_1_dm_pvalue_max = 0.99  # type: ignore[misc]


def test_tier_1_bands_match_preregistration() -> None:
    policy = default_phase_b_policy()
    assert policy.tier_1_coverage_q10_band == (0.07, 0.13)
    assert policy.tier_1_coverage_q50_band == (0.45, 0.55)
    assert policy.tier_1_coverage_q90_band == (0.87, 0.93)
    assert policy.tier_1_picp_error_max == 0.02


def test_tier_2_bands_are_more_permissive() -> None:
    policy = default_phase_b_policy()
    assert policy.tier_2_coverage_q10_band == (0.05, 0.15)
    assert policy.tier_2_coverage_q50_band == (0.40, 0.60)
    assert policy.tier_2_coverage_q90_band == (0.85, 0.95)
    assert policy.tier_2_picp_error_max == 0.05


def test_dm_and_effect_size_thresholds_are_versioned_constants() -> None:
    policy = default_phase_b_policy()
    assert policy.tier_1_dm_pvalue_max == 0.05
    assert policy.tier_2_dm_pvalue_max == 0.10
    assert policy.tier_1_delta_pinball_rel_min == 0.03
    assert policy.tier_2_delta_pinball_rel_min == 0.0


def test_baseline_identity_uses_dim_run_model_version() -> None:
    policy = default_phase_b_policy()
    assert policy.h2a_primary_baseline == "baseline_zero_return_v1"
    assert policy.baseline_model_versions == (
        "baseline_zero_return_v1",
        "baseline_historical_mean_rolling_v1",
        "baseline_historical_quantiles_rolling_v1",
    )


def test_default_policy_returns_independent_instances() -> None:
    assert default_phase_b_policy() == PhaseBTierPolicy()
    assert default_phase_b_policy() is not default_phase_b_policy()

