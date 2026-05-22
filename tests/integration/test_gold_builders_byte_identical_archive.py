"""Byte-identical regression smoke for the Stage R-22 refactor.

Runs `RefreshAnalyticsStoreUseCase.execute()` against the read-only
`data/analytics_archive_pre_phase_b/silver` (499M of real F.1 smoke
data) and compares each emitted gold parquet against a fingerprint
fixture captured from the legacy monolithic implementation BEFORE
the registry migration shipped.

Why fingerprint-based regression guard:
- Full parquet fixtures total ~351M; unsuitable for git.
- A fingerprint per table (n_rows, n_cols, column ordering, dtypes,
  per-column sum/mean/min/max for numeric, nunique for object) is a
  ~5-10KB JSON per table — small enough to commit and sensitive enough
  to catch any non-trivial divergence (a single value flipped sign in
  any cell shifts the column sum/mean).

Fixtures live in `tests/integration/fixtures/gold_builders/byte_identical_archive/`
and were generated via the snippet in this module's docstring on the
checkout of main at d244a8d (post-R-21, pre-R-22).

Skipped (not xfail) when the archive directory is absent — keeps CI
green on contributor checkouts that legitimately lack the 851M archive.
"""
from __future__ import annotations

import json
import math

from pathlib import Path

import pandas as pd
import pytest

from src.use_cases.refresh_analytics_store_use_case import RefreshAnalyticsStoreUseCase

_ARCHIVE_ROOT = Path("data/analytics_archive_pre_phase_b")
_SILVER_DIR = _ARCHIVE_ROOT / "silver"
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "gold_builders" / "byte_identical_archive"

_FLOAT_TOLERANCE = 1e-9


def _archive_available() -> bool:
    return _SILVER_DIR.exists() and any(_SILVER_DIR.iterdir())


def _fingerprint(df: pd.DataFrame) -> dict:
    """Mirror of the snippet that generated the committed fixtures.

    Compute: n_rows, n_cols, column order, dtypes, and per-column
    statistics (sum/mean/min/max for numeric; n_true for bool; nunique
    for object). Used both to generate fixtures (one-time, run on main
    pre-R-22) and to assert against them (this test).
    """
    fp = {
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
    }
    stats: dict[str, dict] = {}
    for c in df.columns:
        s = df[c]
        d: dict = {}
        try:
            n_null = int(s.isna().sum())
        except Exception:
            n_null = -1
        d["n_null"] = n_null
        if pd.api.types.is_numeric_dtype(s):
            sn = pd.to_numeric(s, errors="coerce")
            sn_valid = sn.dropna()
            if not sn_valid.empty:
                d["sum"] = float(sn_valid.sum())
                d["mean"] = float(sn_valid.mean())
                d["min"] = float(sn_valid.min())
                d["max"] = float(sn_valid.max())
        elif pd.api.types.is_bool_dtype(s):
            d["n_true"] = int(s.fillna(False).astype(bool).sum())
        else:
            try:
                d["nunique"] = int(s.nunique(dropna=True))
            except Exception:
                d["nunique"] = -1
        stats[c] = d
    fp["column_stats"] = stats
    return fp


def _close(a: float, b: float, tol: float = _FLOAT_TOLERANCE) -> bool:
    """Float comparison robust to nan-vs-nan."""
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isnan(a) or math.isnan(b):
        return False
    if a == 0.0 or b == 0.0:
        return abs(a - b) < tol
    return abs(a - b) / max(abs(a), abs(b)) < tol


@pytest.mark.skipif(
    not _archive_available(),
    reason="data/analytics_archive_pre_phase_b/silver not present on this checkout",
)
def test_modular_gold_refresh_against_archive_matches_monolith_fingerprints(
    tmp_path: Path,
) -> None:
    """All 25 modular gold parquets must match the pre-R-22 monolith
    output fingerprints exactly.

    Regenerate fixtures via the following snippet on main (pre-R-22):

        # On main before R-22 (commit d244a8d or earlier):
        python -c "
        import json
        from pathlib import Path
        import pandas as pd
        from src.use_cases.refresh_analytics_store_use_case \\
            import RefreshAnalyticsStoreUseCase
        out_dir = Path('/tmp/gold_main_baseline')
        out_dir.mkdir(parents=True, exist_ok=True)
        uc = RefreshAnalyticsStoreUseCase(
            analytics_silver_dir='data/analytics_archive_pre_phase_b/silver',
            analytics_gold_dir=str(out_dir),
        )
        uc.execute()
        # then call _fingerprint() on each parquet and json.dump to
        # tests/integration/fixtures/gold_builders/byte_identical_archive/<table>.json
        "
    """
    gold_dir = tmp_path / "gold"
    use_case = RefreshAnalyticsStoreUseCase(
        analytics_silver_dir=str(_SILVER_DIR),
        analytics_gold_dir=str(gold_dir),
    )
    result = use_case.execute()
    assert len(result.outputs) == 25, "Registry should emit all 25 gold tables"

    failures: list[str] = []
    for parquet_path in sorted(gold_dir.glob("gold_*.parquet")):
        tbl = parquet_path.stem
        fixture_path = _FIXTURE_DIR / f"{tbl}.json"
        assert fixture_path.exists(), (
            f"Fixture missing: {fixture_path}. Regenerate via the snippet "
            f"in the module docstring."
        )
        expected = json.loads(fixture_path.read_text(encoding="utf-8"))
        actual = _fingerprint(pd.read_parquet(parquet_path))

        if actual["n_rows"] != expected["n_rows"]:
            failures.append(
                f"{tbl}: n_rows pre={expected['n_rows']} post={actual['n_rows']}"
            )
            continue
        if actual["columns"] != expected["columns"]:
            extra = set(actual["columns"]) - set(expected["columns"])
            missing = set(expected["columns"]) - set(actual["columns"])
            failures.append(
                f"{tbl}: columns diverge "
                f"(only_in_modular={sorted(extra)} only_in_baseline={sorted(missing)})"
            )
            continue
        if actual["dtypes"] != expected["dtypes"]:
            diffs = {
                c: (expected["dtypes"][c], actual["dtypes"][c])
                for c in expected["columns"]
                if expected["dtypes"][c] != actual["dtypes"][c]
            }
            failures.append(f"{tbl}: dtype mismatch {diffs}")
            continue

        col_failures = []
        for c, exp_stats in expected["column_stats"].items():
            act_stats = actual["column_stats"][c]
            for stat_name in ("n_null", "n_true", "nunique"):
                if stat_name in exp_stats:
                    if exp_stats[stat_name] != act_stats.get(stat_name):
                        col_failures.append(
                            f"{c}.{stat_name} pre={exp_stats[stat_name]} "
                            f"post={act_stats.get(stat_name)}"
                        )
            for stat_name in ("sum", "mean", "min", "max"):
                if stat_name in exp_stats:
                    a = act_stats.get(stat_name)
                    e = exp_stats[stat_name]
                    if a is None:
                        col_failures.append(f"{c}.{stat_name} missing post")
                    elif not _close(float(a), float(e)):
                        col_failures.append(
                            f"{c}.{stat_name} pre={e} post={a} (delta={float(a) - float(e)})"
                        )
        if col_failures:
            failures.append(f"{tbl}: " + "; ".join(col_failures[:5]))

    if failures:
        msg = (
            f"R-22 modular pipeline diverges from pre-R-22 baseline on "
            f"{len(failures)} gold tables:\n  "
            + "\n  ".join(failures)
        )
        pytest.fail(msg)
