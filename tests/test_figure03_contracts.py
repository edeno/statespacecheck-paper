"""Durable name/field/signature contracts for the Figure-3 family.

These pin the public repository-internal API so a future rename that silently
changes a field name, argument order, or the frozen invariant fails here.
"""

from __future__ import annotations

import dataclasses
import inspect
from typing import Any

from statespacecheck_paper import simulation
from statespacecheck_paper.figure03_plotting import compose_figure03
from statespacecheck_paper.figure03_protocol import Figure3Config
from statespacecheck_paper.figure03_simulation import (
    Figure3RateTables,
    Figure3SimulationResult,
)
from statespacecheck_paper.figure03_summary import (
    Figure3RealizationSummary,
    Figure3SummaryCondition,
    build_summary_conditions,
    compute_condition_flag_percentages,
    estimate_realization_summary,
    extract_condition_flag_values,
    flag_percentages_from_values,
)


def _param_names(func: Any) -> list[str]:
    return list(inspect.signature(func).parameters)


def _field_names(cls: Any) -> list[str]:
    return [f.name for f in dataclasses.fields(cls)]


def test_figure3_config_is_frozen_with_exact_fields() -> None:
    assert Figure3Config.__dataclass_params__.frozen  # type: ignore[attr-defined]
    assert _field_names(Figure3Config) == [
        "phase_boundaries",
        "prediction_step_std",
        "drift_momentum",
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
        "n_realizations",
    ]


def test_general_simulation_signatures() -> None:
    assert _param_names(simulation.gaussian_transition_matrix) == ["position_bins", "step_std"]
    assert _param_names(simulation.place_field_rates) == [
        "position_bins",
        "place_field_centers",
        "place_field_std",
        "place_field_rate_scale",
    ]
    assert _param_names(simulation.simulate_walk) == [
        "n_time_steps",
        "step_std",
        "initial_position",
        "position_min",
        "position_max",
        "rng",
    ]
    assert _param_names(simulation.simulate_spikes_position_tuned) == [
        "position",
        "place_field_centers",
        "place_field_std",
        "place_field_rate_scale",
        "rng",
    ]
    assert _param_names(simulation.simulate_spikes_history_dependent) == [
        "position",
        "place_field_centers",
        "place_field_std",
        "place_field_rate_scale",
        "rng",
        "refractory_steps",
        "burst_window",
        "burst_factor",
    ]


def test_summary_signatures() -> None:
    assert _param_names(build_summary_conditions) == ["config"]
    assert _param_names(extract_condition_flag_values) == ["diagnostics", "conditions"]
    assert _param_names(flag_percentages_from_values) == ["values", "diagnostic_thresholds"]
    assert _param_names(compute_condition_flag_percentages) == [
        "diagnostics",
        "diagnostic_thresholds",
        "conditions",
    ]
    assert _param_names(estimate_realization_summary) == [
        "config",
        "n_realizations",
        "first_random_seed",
    ]


def test_compose_figure03_signature() -> None:
    assert _param_names(compose_figure03) == [
        "true_position",
        "spike_counts",
        "diagnostics",
        "diagnostic_thresholds",
        "config",
        "place_field_centers",
        "median_flag_percentages",
    ]
