"""Bit-identical regression smoke for the Stage R-21 refactor.

Runs `ValidateAnalyticsQualityUseCase.execute()` against the read-only
`data/analytics_archive_pre_phase_b/silver` (851M of real F.1 smoke
data) in both `scope_mode` values, and compares the resulting checks
list against fixtures captured from the legacy monolithic
implementation BEFORE the registry migration shipped.

Why this fixture-vs-current comparison is the right regression guard:
the monolith no longer lives in the repo post-Stage R-21, so the only
way to lock down bit-identical behavior end-to-end is to compare
against fixtures generated from the pre-merge state. If a future change
diverges any check's `passed`/`detail`, this test fails with the exact
divergence highlighted.

Skipped (not xfail) when the archive directory is absent — keeps CI
green on contributor checkouts that legitimately lack the 851M archive.
"""
from __future__ import annotations

import json

from pathlib import Path

import pytest

from src.domain.services.scope_spec import ScopeSpec
from src.use_cases.validate_analytics_quality_use_case import (
    ValidateAnalyticsQualityUseCase,
)

_ARCHIVE_ROOT = Path("data/analytics_archive_pre_phase_b")
_SILVER_DIR = _ARCHIVE_ROOT / "silver"
_GOLD_DIR = _ARCHIVE_ROOT / "gold"
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "quality_checks"


def _archive_available() -> bool:
    return _SILVER_DIR.exists() and any(_SILVER_DIR.iterdir())


@pytest.mark.skipif(
    not _archive_available(),
    reason="data/analytics_archive_pre_phase_b/silver not present on this checkout",
)
@pytest.mark.parametrize(
    "scope_mode,fixture_name",
    [
        ("cohort_decision", "archive_silver_cohort_decision.json"),
        ("global_health", "archive_silver_global_health.json"),
    ],
)
def test_validate_quality_against_archive_matches_monolith_fixture(
    scope_mode: str, fixture_name: str
) -> None:
    """The 27-check output must match the pre-Stage R-21 monolith byte-byte.

    Fixtures live in `tests/integration/fixtures/quality_checks/` and were
    captured from the pre-merge state via:

        # On main (pre-R-21):
        python -c "
        import json
        from src.use_cases.validate_analytics_quality_use_case \\
            import ValidateAnalyticsQualityUseCase
        from src.domain.services.scope_spec import ScopeSpec
        for mode in ['cohort_decision', 'global_health']:
            uc = ValidateAnalyticsQualityUseCase(
                analytics_silver_dir='data/analytics_archive_pre_phase_b/silver',
                analytics_gold_dir='data/analytics_archive_pre_phase_b/gold',
                scope_spec=ScopeSpec.create(
                    scope_mode=mode,
                    parent_sweep_prefixes=['phase_a_'] if mode == 'cohort_decision' else (),
                ),
            )
            result = uc.execute()
            with open(f'/tmp/quality_pre_r21_{mode}.json', 'w') as f:
                json.dump({'passed': result.passed, 'n_checks': len(result.checks),
                           'checks': result.checks}, f, indent=2)
        "
    """
    scope_args: dict = {}
    if scope_mode == "cohort_decision":
        scope_args["parent_sweep_prefixes"] = ["phase_a_"]
    uc = ValidateAnalyticsQualityUseCase(
        analytics_silver_dir=str(_SILVER_DIR),
        analytics_gold_dir=str(_GOLD_DIR),
        scope_spec=ScopeSpec.create(scope_mode=scope_mode, **scope_args),
    )
    result = uc.execute()
    actual = {
        "passed": result.passed,
        "n_checks": len(result.checks),
        "checks": result.checks,
    }

    fixture_path = _FIXTURE_DIR / fixture_name
    assert fixture_path.exists(), (
        f"Fixture missing: {fixture_path}. Regenerate via the snippet in the "
        "module docstring on a checkout of main before Stage R-21 merged."
    )
    expected = json.loads(fixture_path.read_text())

    # Compare the structural counts first to surface gross divergences fast.
    assert actual["n_checks"] == expected["n_checks"], (
        f"n_checks diverged for {scope_mode}: "
        f"got {actual['n_checks']}, expected {expected['n_checks']}"
    )
    assert actual["passed"] == expected["passed"], (
        f"passed flag diverged for {scope_mode}: "
        f"got {actual['passed']}, expected {expected['passed']}"
    )

    # Now byte-byte: each check entry must match in name + passed + detail
    # (detail includes the scope_detail suffix appended by the orchestrator).
    if actual["checks"] != expected["checks"]:
        diffs: list[str] = []
        for i, (a, e) in enumerate(zip(actual["checks"], expected["checks"])):
            if a != e:
                diffs.append(f"  index {i}:\n    expected={e}\n    actual={a}")
        diffs_msg = "\n".join(diffs) if diffs else "(length/order mismatch)"
        pytest.fail(
            f"Bit-identical regression in {scope_mode}:\n{diffs_msg}"
        )
