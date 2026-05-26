from __future__ import annotations

import math

from src.domain.services.phase_b_tier_policy import default_phase_b_policy
from src.domain.services.tier_classifier_per_hypothesis import (
    classify_h1,
    classify_h2a,
    classify_h2b,
)


def test_h1_tier_1_accepts_borderline_registered_bands() -> None:
    verdict = classify_h1(0.07, 0.45, 0.93, 0.02, 0.01, default_phase_b_policy(), horizon=1)
    assert verdict.hypothesis == "H1"
    assert verdict.horizon == 1
    assert verdict.tier == "tier_1"


def test_h1_tier_2_when_only_permissive_bands_pass() -> None:
    verdict = classify_h1(0.05, 0.40, 0.95, 0.05, 0.01, default_phase_b_policy())
    assert verdict.tier == "tier_2"


def test_h1_refutado_when_mpiw_is_not_positive_or_nan() -> None:
    assert classify_h1(0.10, 0.50, 0.90, 0.0, 0.0, default_phase_b_policy()).tier == "refutado"
    assert (
        classify_h1(0.10, math.nan, 0.90, 0.0, 0.1, default_phase_b_policy()).tier
        == "refutado"
    )


def test_h2a_tier_1_uses_primary_zero_return_inputs() -> None:
    verdict = classify_h2a(0.049, 0.03, 0.01, default_phase_b_policy(), horizon=7)
    assert verdict.hypothesis == "H2a"
    assert verdict.horizon == 7
    assert verdict.tier == "tier_1"


def test_h2a_tier_2_and_refutado_truth_table() -> None:
    assert classify_h2a(0.099, 0.0, 0.01, default_phase_b_policy()).tier == "tier_2"
    assert classify_h2a(0.20, 0.04, 0.01, default_phase_b_policy()).tier == "refutado"
    assert classify_h2a(0.04, -0.01, 0.01, default_phase_b_policy()).tier == "refutado"


def test_h2b_requires_all_baselines_to_pass() -> None:
    policy = default_phase_b_policy()
    all_pass = classify_h2b(
        {"a": 0.01, "b": 0.02, "c": 0.03},
        {"a": 0.05, "b": 0.04, "c": 0.03},
        0.01,
        policy,
    )
    one_fail = classify_h2b(
        {"a": 0.01, "b": 0.20, "c": 0.03},
        {"a": 0.05, "b": 0.04, "c": 0.03},
        0.01,
        policy,
    )
    assert all_pass.tier == "tier_1"
    assert one_fail.tier == "refutado"


def test_h2b_tier_2_and_nan_handling() -> None:
    policy = default_phase_b_policy()
    assert classify_h2b({"a": 0.09}, {"a": 0.0}, 0.1, policy).tier == "tier_2"
    assert classify_h2b({"a": math.nan}, {"a": 0.1}, 0.1, policy).tier == "refutado"

