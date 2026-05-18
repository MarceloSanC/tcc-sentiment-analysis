from __future__ import annotations

from typing import Any

import pytest

from src.use_cases.run_tft_optuna_search_use_case import RunTFTOptunaSearchUseCase


class _DummyTrainRunner:
    def run(self, **_: Any) -> tuple[str, dict[str, Any]]:
        return "v1", {}


def _build_kwargs(objective_metric: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "train_runner": _DummyTrainRunner(),
        "base_training_config": {},
        "split_config": {},
        "walk_forward_config": {},
        "replica_seeds": [7],
        "continue_on_error": False,
    }
    if objective_metric is not None:
        kwargs["objective_metric"] = objective_metric
    return kwargs


def test_objective_metric_rejects_mean_test_rmse() -> None:
    with pytest.raises(ValueError, match=r"mean_test_rmse|leakage|M1-Q4"):
        RunTFTOptunaSearchUseCase(**_build_kwargs(objective_metric="mean_test_rmse"))


def test_objective_metric_rejects_joint_val_test_rmse() -> None:
    with pytest.raises(ValueError, match=r"joint_val_test_rmse|leakage|M1-Q4"):
        RunTFTOptunaSearchUseCase(**_build_kwargs(objective_metric="joint_val_test_rmse"))


def test_objective_metric_accepts_robust_score() -> None:
    use_case = RunTFTOptunaSearchUseCase(**_build_kwargs(objective_metric="robust_score"))
    assert use_case.objective_metric == "robust_score"


def test_objective_metric_accepts_mean_val_rmse() -> None:
    use_case = RunTFTOptunaSearchUseCase(**_build_kwargs(objective_metric="mean_val_rmse"))
    assert use_case.objective_metric == "mean_val_rmse"


def test_objective_metric_default_is_robust_score() -> None:
    use_case = RunTFTOptunaSearchUseCase(**_build_kwargs())
    assert use_case.objective_metric == "robust_score"


def test_objective_metric_rejects_arbitrary_string() -> None:
    with pytest.raises(ValueError, match=r"invalido|leakage|M1-Q4"):
        RunTFTOptunaSearchUseCase(**_build_kwargs(objective_metric="foobar"))


def test_objective_from_summary_has_no_test_set_branches() -> None:
    use_case = RunTFTOptunaSearchUseCase(**_build_kwargs(objective_metric="mean_val_rmse"))
    # Force a non-whitelisted value to hit the fallback ValueError, confirming
    # no removed branch silently catches it (defense in depth).
    use_case.objective_metric = "mean_test_rmse"  # type: ignore[assignment]
    with pytest.raises(ValueError, match=r"Unsupported objective_metric"):
        use_case._objective_from_summary(top_run={"mean_test_rmse": 0.1})

    use_case.objective_metric = "joint_val_test_rmse"  # type: ignore[assignment]
    with pytest.raises(ValueError, match=r"Unsupported objective_metric"):
        use_case._objective_from_summary(
            top_run={"mean_val_rmse": 0.1, "mean_test_rmse": 0.2}
        )
