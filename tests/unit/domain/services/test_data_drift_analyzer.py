from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.services.data_drift_analyzer import DataDriftAnalyzer


def test_analyze_features_returns_detail_and_summary() -> None:
    train_df = pd.DataFrame(
        {
            "f1": [0.0, 1.0, 2.0, 3.0, np.nan],
            "f2": [10.0, 10.0, 10.0, 10.0, 10.0],
        }
    )
    val_df = pd.DataFrame(
        {
            "f1": [0.0, 1.0, 1.0, 2.0, 2.0],
            "f2": [11.0, 11.0, 11.0, 11.0, 11.0],
        }
    )
    test_df = pd.DataFrame(
        {
            "f1": [5.0, 6.0, 7.0, 8.0, 9.0],
            "f2": [9.0, 9.0, 9.0, 9.0, 9.0],
        }
    )

    detail, summary = DataDriftAnalyzer.analyze_features(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=["f1", "f2"],
        psi_bins=5,
    )

    assert not detail.empty
    assert set(["feature", "ks_train_vs_val", "ks_train_vs_test", "psi_train_vs_val", "psi_train_vs_test"]).issubset(
        set(detail.columns)
    )
    assert summary["n_features"] == 2
    assert summary["avg_ks_train_vs_val"] is not None
    assert summary["avg_ks_train_vs_test"] is not None
    assert summary["avg_psi_train_vs_val"] is not None
    assert summary["avg_psi_train_vs_test"] is not None

