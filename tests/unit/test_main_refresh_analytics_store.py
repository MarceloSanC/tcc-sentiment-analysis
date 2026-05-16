from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import src.main_refresh_analytics_store as main_refresh


class _FakeRefreshResult:
    def __init__(self) -> None:
        self.gold_dir = "data/analytics/gold"
        self.outputs = {"ok": "ok"}


class _FakeRefreshUseCase:
    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.kwargs = kwargs
        self.scope_spec = kwargs.get("scope_spec")

    def execute(self) -> _FakeRefreshResult:
        return _FakeRefreshResult()


class _FakeQualityResult:
    def __init__(self) -> None:
        self.passed = True
        self.checks = [{"passed": True}]


class _FakeValidateUseCase:
    last_kwargs = None

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.kwargs = kwargs
        _FakeValidateUseCase.last_kwargs = kwargs

    def execute(self) -> _FakeQualityResult:
        return _FakeQualityResult()


class _FakePlotsUseCase:
    execute_kwargs = None

    def __init__(self, **kwargs) -> None:  # noqa: ANN003
        self.kwargs = kwargs

    def execute(self, **kwargs):  # noqa: ANN003
        _FakePlotsUseCase.execute_kwargs = kwargs

        class _Result:
            output_dir = "data/analytics/reports/prediction_analysis_plots/asset=AAPL"
            outputs = {"manifest": "x"}

        return _Result()


def test_main_refresh_passes_scope_flags_to_plots_use_case(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main_refresh, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_refresh,
        "parse_args",
        lambda: Namespace(
            fail_on_quality=False,
            min_samples_train=1,
            min_samples_val=1,
            min_samples_test=1,
            skip_prediction_plots=False,
            plots_asset="AAPL",
            plots_output_dir=str(tmp_path / "out"),
            plots_scope_csv=str(tmp_path / "scope.csv"),
            plots_scope_sweep_prefixes="0_2_2_, 0_2_3_",
            scope_mode=None,
            scope_sweep_prefixes=None,
            scope_splits=None,
            scope_horizons=None,
            block_a_scope_sweep_prefixes=None,
            block_a_splits=None,
            block_a_horizons=None,
            block_a_max_crossing_bruto_rate=0.001,
            block_a_max_negative_interval_width_count=0,
            block_a_max_crossing_post_guardrail_rate=0.0,
            block_a_require_post_guardrail=False,
        ),
    )
    monkeypatch.setattr(
        main_refresh,
        "load_data_paths",
        lambda: {
            "analytics_silver": str(tmp_path / "silver"),
            "analytics_gold": str(tmp_path / "gold"),
        },
    )
    monkeypatch.setattr(main_refresh, "RefreshAnalyticsStoreUseCase", _FakeRefreshUseCase)
    monkeypatch.setattr(main_refresh, "ValidateAnalyticsQualityUseCase", _FakeValidateUseCase)
    monkeypatch.setattr(main_refresh, "GeneratePredictionAnalysisPlotsUseCase", _FakePlotsUseCase)

    main_refresh.main()

    assert _FakePlotsUseCase.execute_kwargs is not None
    assert _FakePlotsUseCase.execute_kwargs["asset"] == "AAPL"
    assert _FakePlotsUseCase.execute_kwargs["scope_csv_path"] == str(tmp_path / "scope.csv")
    assert _FakePlotsUseCase.execute_kwargs["scope_sweep_prefixes"] == ["0_2_2_", "0_2_3_"]



def test_main_refresh_passes_block_a_flags_to_quality_use_case(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main_refresh, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        main_refresh,
        "parse_args",
        lambda: Namespace(
            fail_on_quality=False,
            min_samples_train=1,
            min_samples_val=1,
            min_samples_test=1,
            skip_prediction_plots=True,
            plots_asset=None,
            plots_output_dir=None,
            plots_scope_csv=None,
            plots_scope_sweep_prefixes=None,
            scope_mode=None,
            scope_sweep_prefixes=None,
            scope_splits=None,
            scope_horizons=None,
            block_a_scope_sweep_prefixes="0_2_3_,0_2_4_",
            block_a_splits="val,test",
            block_a_horizons="1,7,30",
            block_a_max_crossing_bruto_rate=0.0005,
            block_a_max_negative_interval_width_count=0,
            block_a_max_crossing_post_guardrail_rate=0.0,
            block_a_require_post_guardrail=True,
        ),
    )
    monkeypatch.setattr(
        main_refresh,
        "load_data_paths",
        lambda: {
            "analytics_silver": str(tmp_path / "silver"),
            "analytics_gold": str(tmp_path / "gold"),
        },
    )
    monkeypatch.setattr(main_refresh, "RefreshAnalyticsStoreUseCase", _FakeRefreshUseCase)
    monkeypatch.setattr(main_refresh, "ValidateAnalyticsQualityUseCase", _FakeValidateUseCase)

    main_refresh.main()

    assert _FakeValidateUseCase.last_kwargs is not None
    kwargs = _FakeValidateUseCase.last_kwargs
    assert kwargs["block_a_parent_sweep_prefixes"] == ["0_2_3_", "0_2_4_"]
    assert kwargs["block_a_splits"] == ["val", "test"]
    assert kwargs["block_a_horizons"] == [1, 7, 30]
    assert kwargs["block_a_max_crossing_bruto_rate"] == 0.0005
    assert kwargs["block_a_max_negative_interval_width_count"] == 0
    assert kwargs["block_a_max_crossing_post_guardrail_rate"] == 0.0
    assert kwargs["block_a_require_post_guardrail"] is True
