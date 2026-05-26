from __future__ import annotations

import pandas as pd
import pytest

from src.domain.services.holm_family_6 import apply_holm_one_sided


def test_holm_adjustment_matches_classic_six_pvalue_case() -> None:
    pvalues = pd.Series([0.001, 0.01, 0.02, 0.04, 0.20, 0.50])
    out = apply_holm_one_sided(pvalues)
    assert out["pvalue_adj_holm"].tolist() == [0.006, 0.05, 0.08, 0.12, 0.40, 0.50]


def test_significance_uses_strict_alpha_threshold() -> None:
    pvalues = pd.Series([0.001, 0.05])
    out = apply_holm_one_sided(pvalues)
    assert bool(out.loc[0, "significant_adj_0_05"]) is True
    assert bool(out.loc[1, "significant_adj_0_05"]) is False


def test_preserves_input_index_and_original_pvalues() -> None:
    pvalues = pd.Series([0.03, 0.01], index=["b", "a"])
    out = apply_holm_one_sided(pvalues)
    assert out.index.tolist() == ["b", "a"]
    assert out.loc["a", "pvalue_one_sided"] == 0.01


def test_handles_nan_without_adjusting_it() -> None:
    out = apply_holm_one_sided(pd.Series([0.01, None, 0.03]))
    assert pd.isna(out.loc[1, "pvalue_adj_holm"])
    assert out.loc[0, "pvalue_adj_holm"] == 0.02


def test_invalid_pvalues_remain_nan_adjusted() -> None:
    out = apply_holm_one_sided(pd.Series([0.01, -0.1, 1.2]))
    assert out.loc[0, "pvalue_adj_holm"] == 0.01
    assert pd.isna(out.loc[1, "pvalue_adj_holm"])
    assert pd.isna(out.loc[2, "pvalue_adj_holm"])


def test_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        apply_holm_one_sided(pd.Series([0.01]), alpha=0.0)
