from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SweepExperiment:
    run_label: str
    config: dict[str, Any]
    varied_param: str | None
    varied_value: Any | None


def build_one_at_a_time_experiments(
    *,
    base_config: dict[str, Any],
    param_ranges: dict[str, list[Any]],
) -> list[SweepExperiment]:
    experiments: list[SweepExperiment] = [
        SweepExperiment(
            run_label="baseline",
            config=dict(base_config),
            varied_param=None,
            varied_value=None,
        )
    ]
    for param, values in param_ranges.items():
        default_value = base_config.get(param)
        for value in values:
            if value == default_value:
                continue
            cfg = dict(base_config)
            cfg[param] = value
            if cfg["max_prediction_length"] > cfg["max_encoder_length"]:
                continue
            experiments.append(
                SweepExperiment(
                    run_label=f"{param}={value}",
                    config=cfg,
                    varied_param=param,
                    varied_value=value,
                )
            )
    return experiments
