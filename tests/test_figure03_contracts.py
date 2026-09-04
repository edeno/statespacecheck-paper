"""Durable name/field contracts for the Figure-3 family.

These pin the public repository-internal API so a future rename that silently
changes a field name or the frozen invariant fails here.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from statespacecheck_paper.figure03_protocol import Figure3Config
from statespacecheck_paper.figure03_simulation import (
    Figure3RateTables,
    Figure3SimulationResult,
)
from statespacecheck_paper.figure03_summary import (
    Figure3RealizationSummary,
    Figure3SummaryCondition,
)


def _field_names(cls: Any) -> list[str]:
    return [f.name for f in dataclasses.fields(cls)]


def test_figure3_config_is_frozen_with_exact_fields() -> None:
    assert Figure3Config.__dataclass_params__.frozen
    assert _field_names(Figure3Config) == [
        "phase_boundaries",
        "prediction_step_std",
        "drift_momentum",
        "history_refractory_steps",
        "history_burst_window",
        "history_burst_factor",
        "position_min",
        "position_max",
        "position_bin_size",
        "place_field_std",
        "place_field_centers",
        "place_field_rate_scale",
        "random_seed",
        "place_field_remapping",
        "replay_start_fraction",
        "replay_end_fraction",
        "replay_speed_per_step",
        "replay_place_field_rate_scale",
        "sparse_position",
        "sparse_approach_duration_steps",
        "sparse_control_ordinary_rate_scale",
        "sparse_cell_count",
        "sparse_place_field_spread",
        "sparse_place_field_std",
        "sparse_cell_peak_rate_per_step",
        "sparse_cell_baseline_rate_fraction",
    ]


def test_simulation_result_fields() -> None:
    assert _field_names(Figure3SimulationResult) == [
        "config",
        "position_bins",
        "true_position",
        "spike_counts",
        "diagnostics",
        "phase_labels",
        "phase_boundaries",
        "sparse_place_field_centers",
    ]


def test_rate_tables_fields() -> None:
    assert _field_names(Figure3RateTables) == [
        "baseline_firing_rates",
        "remapped_firing_rates",
        "replay_firing_rates",
        "sparse_population_firing_rates",
        "baseline_sparse_firing_rates",
    ]


def test_summary_condition_and_realization_summary_fields() -> None:
    assert _field_names(Figure3SummaryCondition) == ["label", "step_windows", "model_component"]
    assert _field_names(Figure3RealizationSummary) == [
        "diagnostic_thresholds",
        "median_flag_percentages",
        "median_decoding_accuracy",
        "n_realizations",
    ]
