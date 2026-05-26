from __future__ import annotations

import numpy as np
import pandas as pd


def apply_holm_one_sided(
    pvalues: pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    if not 0 < float(alpha) < 1:
        raise ValueError("alpha must be between 0 and 1")

    pv = pd.to_numeric(pvalues.copy(), errors="coerce")
    out = pd.DataFrame(index=pv.index)
    out["pvalue_one_sided"] = pv.astype(float)
    out["pvalue_adj_holm"] = np.nan

    valid = pv.dropna()
    valid = valid[(valid >= 0.0) & (valid <= 1.0)]
    m = int(len(valid))
    if m:
        ordered = valid.sort_values(kind="mergesort")
        raw_adj = []
        for rank, (_, pvalue) in enumerate(ordered.items(), start=1):
            raw_adj.append((m - rank + 1) * float(pvalue))
        adjusted = np.minimum(1.0, np.maximum.accumulate(raw_adj))
        for idx, adj in zip(ordered.index, adjusted):
            out.at[idx, "pvalue_adj_holm"] = float(adj)

    out["significant_adj_0_05"] = (
        pd.to_numeric(out["pvalue_adj_holm"], errors="coerce") < float(alpha)
    )
    return out

