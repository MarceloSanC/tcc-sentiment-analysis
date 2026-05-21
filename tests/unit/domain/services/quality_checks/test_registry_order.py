"""Tests for the production registry: order + completeness + naming.

These tests are the structural half of the Stage R-21 bit-identical
contract per ADR-0004 §Consequences. The empirical (data-driven) half
is `tests/integration/test_quality_registry_bit_identical_archive.py`,
which runs the use case against the archive silver and compares to
fixtures captured from the monolithic implementation.
"""
from __future__ import annotations

import pytest

from src.domain.services.quality_checks import (
    QualityCheck,
    build_default_registry,
)
from src.domain.services.scope_spec import ScopeSpec

# Canonical order the monolithic ValidateAnalyticsQualityUseCase.execute()
# emitted checks in. Reordering anything here without updating the smoke
# regression baseline breaks the bit-identical contract.
EXPECTED_CHECK_NAMES: tuple[str, ...] = (
    "required_tables_presence",
    "inference_predictions_continuity",
    "feature_contrib_local_continuity",
    "run_id_execution_consistency",
    "referential_integrity",
    "required_metrics_nan",
    "temporal_consistency",
    "oos_unique_key",
    "oos_numeric_types",
    "oos_horizon_coverage",
    "oos_supervised_nulls",
    "oos_interval_width_non_negative",
    "oos_quantile_order",
    "oos_quantile_block_a_acceptance",
    "block_quantile_degeneracy_gate",
    "oos_pairwise_target_alignment",
    "inference_predictions_unique_key",
    "inference_predictions_numeric_types",
    "inference_predictions_quantile_order",
    "cardinality_config_fold_seed",
    "min_samples_by_split",
    "dm_mcs_persisted_executable",
    "gold_confidence_calibrated_by_horizon",
    "gold_metrics_by_config_n_oos_contract",
    "baselines_share_parent_sweep_id_with_candidates",
    "tft_baselines_timestamp_subset_alignment",
    "official_contract_quantile_attention",
)


def test_default_registry_has_exactly_27_checks() -> None:
    reg = build_default_registry()
    assert len(reg) == 27, (
        f"Expected 27 checks in default registry, got {len(reg)}. "
        "If you intentionally added a check, also update EXPECTED_CHECK_NAMES "
        "and the bit-identical archive fixtures."
    )


def test_default_registry_names_in_canonical_order() -> None:
    reg = build_default_registry()
    names = [c.name for c in reg]
    assert names == list(EXPECTED_CHECK_NAMES), (
        "Check registration order diverged from the monolith. "
        f"Got: {names}; expected: {list(EXPECTED_CHECK_NAMES)}"
    )


def test_default_registry_check_names_are_unique() -> None:
    reg = build_default_registry()
    names = [c.name for c in reg]
    assert len(set(names)) == len(names), f"Duplicate check names: {names}"


def test_default_registry_no_check_has_empty_name() -> None:
    reg = build_default_registry()
    bad = [type(c).__name__ for c in reg if not c.name]
    assert not bad, f"Check class(es) with empty name: {bad}"


@pytest.mark.parametrize("scope_mode", ["cohort_decision", "global_health"])
def test_all_default_checks_apply_under_both_scope_modes(scope_mode: str) -> None:
    """Bit-identical contract: every check the monolith emitted must also
    be emitted by the registry. The two checks documented as
    skip-under-global_health (tft_baselines_alignment +
    baselines_share_parent_sweep) emit `skipped(not_cohort_decision)` as
    their `run` detail — they DO appear in the output.
    """
    reg = build_default_registry()
    scope = ScopeSpec(scope_mode=scope_mode)
    applicable = reg.applicable(scope)
    assert [c.name for c in applicable] == list(EXPECTED_CHECK_NAMES)


def test_default_registry_returns_quality_check_instances() -> None:
    reg = build_default_registry()
    for check in reg:
        assert isinstance(check, QualityCheck), f"{type(check)} is not a QualityCheck"
