"""Tests for real data analysis module."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from statespacecheck_paper.real_data_analysis import (
    compute_per_cell_diagnostics,
    compute_running_average,
    extract_place_fields,
    find_sustained_low_overlap,
    gaussian_smooth,
    get_multiunit_population_firing_rate,
    get_state_marginalized_posterior,
)
from statespacecheck_paper.real_data_plotting import (
    plot_per_cell_diagnostic_scatter,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def per_cell_setup(rng: np.random.Generator) -> dict[str, Any]:
    """Standard inputs for ``compute_per_cell_diagnostics``."""
    n_time, n_bins, n_cells = 100, 50, 10
    return {
        "n_time": n_time,
        "n_bins": n_bins,
        "n_cells": n_cells,
        "predictive": rng.dirichlet(np.ones(n_bins), size=n_time),
        "place_fields": rng.random((n_cells, n_bins)) * 10 + 0.1,
        "spike_counts": rng.poisson(0.5, (n_time, n_cells)).astype(np.int64),
    }


def _xarray_results(
    posterior_data: np.ndarray,
    name: str,
    state_bins: pd.MultiIndex | np.ndarray | None = None,
) -> xr.Dataset:
    """Build a 2-variable Dataset matching the on-disk results layout."""
    n_time, n_state_bins = posterior_data.shape
    if state_bins is None:
        state_bins = np.arange(n_state_bins)
    return xr.Dataset(
        {
            name: xr.DataArray(
                posterior_data,
                dims=["time", "state_bins"],
                coords={"time": np.arange(n_time), "state_bins": state_bins},
            )
        }
    )


# ---------------------------------------------------------------------------
# gaussian_smooth, get_multiunit_population_firing_rate
# ---------------------------------------------------------------------------


class TestGaussianSmooth:
    def test_output_shape_matches_input(self, rng: np.random.Generator) -> None:
        data = rng.standard_normal(1000)
        result = gaussian_smooth(data, sigma=0.01, sampling_frequency=1000)
        assert result.shape == data.shape

    def test_smoothing_reduces_variance_of_noise(self, rng: np.random.Generator) -> None:
        data = rng.standard_normal(1000)
        result = gaussian_smooth(data, sigma=0.02, sampling_frequency=500)
        assert result.var() < data.var()


def test_get_multiunit_population_firing_rate_collapses_cell_dim(
    rng: np.random.Generator,
) -> None:
    """Collapses (n_time, n_cells) multiunit array to (n_time,)."""
    multiunit = rng.poisson(0.1, size=(1000, 50)).astype(np.float64)
    result = get_multiunit_population_firing_rate(
        multiunit, sampling_frequency=500, smoothing_sigma=0.015
    )
    assert result.shape == (1000,)


# ---------------------------------------------------------------------------
# find_sustained_low_overlap
# ---------------------------------------------------------------------------


class TestFindSustainedLowOverlap:
    def test_returns_list(self, rng: np.random.Generator) -> None:
        result = find_sustained_low_overlap(rng.random(1000), threshold=0.3)
        assert isinstance(result, list)

    def test_finds_clearly_below_threshold_region(self) -> None:
        hpd_overlap = np.ones(1000)
        hpd_overlap[200:300] = 0.1
        result = find_sustained_low_overlap(
            hpd_overlap, threshold=0.5, min_duration=0.05, sampling_frequency=1000
        )
        assert len(result) >= 1
        start, end = result[0]
        assert 150 < start < 250
        assert 250 < end < 350

    def test_returns_empty_when_nothing_below_threshold(self) -> None:
        """Edge case: no low-overlap regions => empty list, not error."""
        hpd_overlap = np.ones(1000) * 0.8
        result = find_sustained_low_overlap(hpd_overlap, threshold=0.5)
        assert result == []


# ---------------------------------------------------------------------------
# extract_place_fields
# ---------------------------------------------------------------------------


def _mock_model(
    place_fields: np.ndarray,
    position_bins: np.ndarray,
    env_name: str = "",
    encoding_group: int = 0,
) -> MagicMock:
    """Mock with the small surface used by ``extract_place_fields``."""
    mock_model = MagicMock()
    mock_model.encoding_model_ = {(env_name, encoding_group): {"place_fields": place_fields}}
    mock_env = MagicMock()
    mock_env.place_bin_centers_ = position_bins.reshape(-1, 1)
    envs: list[MagicMock | None] = [None] * (encoding_group + 1)
    envs[encoding_group] = mock_env
    mock_model.environments = envs
    return mock_model


class TestExtractPlaceFields:
    def test_extracts_default_environment(self, rng: np.random.Generator) -> None:
        n_cells, n_bins = 10, 50
        place_fields = rng.random((n_cells, n_bins)) * 10
        position_bins = np.linspace(0, 100, n_bins)
        model = _mock_model(place_fields, position_bins)
        pf, bins = extract_place_fields(model)
        np.testing.assert_array_equal(pf, place_fields)
        np.testing.assert_array_equal(bins, position_bins)

    def test_extracts_named_environment_and_group(self, rng: np.random.Generator) -> None:
        n_cells, n_bins = 5, 30
        place_fields = rng.random((n_cells, n_bins)) * 10
        position_bins = np.linspace(0, 50, n_bins)
        model = _mock_model(place_fields, position_bins, env_name="env1", encoding_group=1)
        pf, bins = extract_place_fields(model, environment_name="env1", encoding_group=1)
        np.testing.assert_array_equal(pf, place_fields)
        np.testing.assert_array_equal(bins, position_bins)


# ---------------------------------------------------------------------------
# compute_per_cell_diagnostics
# ---------------------------------------------------------------------------


class TestComputePerCellDiagnostics:
    def test_shapes_and_keys(self, per_cell_setup: dict) -> None:
        result = compute_per_cell_diagnostics(
            per_cell_setup["predictive"],
            per_cell_setup["spike_counts"],
            per_cell_setup["place_fields"],
        )
        for key in ("hpd_overlap", "kl_divergence", "spike_prob"):
            assert key in result
            assert result[key].shape == (per_cell_setup["n_time"], per_cell_setup["n_cells"])

    def test_nan_exactly_where_no_spikes(self, per_cell_setup: dict) -> None:
        result = compute_per_cell_diagnostics(
            per_cell_setup["predictive"],
            per_cell_setup["spike_counts"],
            per_cell_setup["place_fields"],
        )
        no_spike = per_cell_setup["spike_counts"] == 0
        for key in ("hpd_overlap", "kl_divergence", "spike_prob"):
            assert np.all(np.isnan(result[key][no_spike]))

    @pytest.mark.parametrize("metric", ["hpd_overlap", "spike_prob"])
    def test_metric_in_unit_range_with_gaussian_place_fields(
        self, rng: np.random.Generator, metric: str
    ) -> None:
        """HPD overlap and spike_prob are bounded in [0, 1] (allowing tiny
        floating-point overshoot above 1 for spike_prob)."""
        n_time, n_bins, n_cells = 100, 50, 10
        predictive = rng.dirichlet(np.ones(n_bins), size=n_time)
        # Gaussian place fields ensure spike_prob_rank stays well-defined.
        place_fields = np.zeros((n_cells, n_bins))
        for j, center in enumerate(np.linspace(5, n_bins - 5, n_cells)):
            place_fields[j] = np.exp(-0.5 * ((np.arange(n_bins) - center) / 5) ** 2)
        place_fields = place_fields * 10 + 0.1
        spike_counts = np.ones((n_time, n_cells), dtype=np.int64)

        result = compute_per_cell_diagnostics(predictive, spike_counts, place_fields)
        valid = result[metric][~np.isnan(result[metric])]
        assert (valid >= 0.0).all()
        assert (valid <= 1.0 + 1e-9).all()

    def test_diagnostics_only_at_spike_times(self, rng: np.random.Generator) -> None:
        n_time, n_bins, n_cells = 20, 10, 3
        predictive = rng.dirichlet(np.ones(n_bins), size=n_time)
        place_fields = rng.random((n_cells, n_bins)) * 10 + 0.1
        spike_counts = np.zeros((n_time, n_cells), dtype=np.int64)
        spike_counts[[0, 5, 10], 0] = 1
        spike_counts[[2, 7], 1] = 1
        spike_counts[15, 2] = 1

        result = compute_per_cell_diagnostics(predictive, spike_counts, place_fields)
        # Diagnostics finite at spike times, NaN elsewhere — single check.
        np.testing.assert_array_equal(np.isnan(result["hpd_overlap"]), spike_counts == 0)

    def test_duplicate_spikes_in_same_bin_are_separate_events(
        self, rng: np.random.Generator
    ) -> None:
        """Two spikes in one (time, cell) bin must yield two event entries
        with the same value as the bin's matrix entry."""
        n_time, n_bins, n_cells = 4, 8, 1
        predictive = rng.dirichlet(np.ones(n_bins), size=n_time)
        place_fields = rng.random((n_cells, n_bins)) + 0.1
        spike_counts = np.zeros((n_time, n_cells), dtype=np.int64)
        spike_counts[1, 0] = 2

        result = compute_per_cell_diagnostics(
            predictive,
            spike_counts,
            place_fields,
            spike_times=[np.array([1.10, 1.20])],
            time=np.arange(n_time, dtype=np.float64),
        )

        np.testing.assert_allclose(result["event_time"], [1.10, 1.20])
        np.testing.assert_array_equal(result["event_time_ind"], [1, 1])
        np.testing.assert_array_equal(result["event_cell_ind"], [0, 0])
        for event_key in ("event_hpd_overlap", "event_kl_divergence", "event_spike_prob"):
            assert result[event_key].shape == (2,)
        np.testing.assert_allclose(
            result["event_kl_divergence"],
            np.repeat(result["kl_divergence"][1, 0], 2),
        )


# ---------------------------------------------------------------------------
# get_state_marginalized_posterior
# ---------------------------------------------------------------------------


class TestGetStateMarginalizedPosterior:
    @pytest.mark.parametrize("posterior_type", ["predictive", "acausal"])
    def test_single_state_passthrough(
        self,
        rng: np.random.Generator,
        posterior_type: Literal["predictive", "acausal"],
    ) -> None:
        """Single-state model: no states to marginalize, output equals input."""
        n_time, n_bins = 100, 50
        posterior_data = rng.dirichlet(np.ones(n_bins), size=n_time)
        results = _xarray_results(posterior_data, f"{posterior_type}_posterior")
        result = get_state_marginalized_posterior(results, posterior_type)
        assert result.shape == (n_time, n_bins)
        np.testing.assert_allclose(result, posterior_data)

    def test_multi_state_sums_over_states(self, rng: np.random.Generator) -> None:
        """Multi-state model: result is the sum across states."""
        n_time, n_bins, n_states = 100, 50, 2
        posterior_per_state = rng.dirichlet(np.ones(n_bins), size=(n_time, n_states))
        posterior_per_state = posterior_per_state / posterior_per_state.sum(
            axis=(1, 2), keepdims=True
        )
        states = ["Continuous", "Fragmented"]
        positions = np.arange(n_bins, dtype=float)
        multi_index = pd.MultiIndex.from_product([states, positions], names=["state", "position"])
        results = _xarray_results(
            posterior_per_state.reshape(n_time, -1),
            "predictive_posterior",
            state_bins=multi_index,
        )
        result = get_state_marginalized_posterior(results, "predictive")
        np.testing.assert_allclose(result, posterior_per_state.sum(axis=1), rtol=1e-5)


# ---------------------------------------------------------------------------
# plot_per_cell_diagnostic_scatter (spike-time alignment behavior)
# ---------------------------------------------------------------------------


def _scatter_offsets(ax: plt.Axes) -> np.ndarray:
    offsets = ax.collections[0].get_offsets()
    mask = np.ma.getmaskarray(offsets)
    return np.asarray(offsets)[~mask.any(axis=1)]


class TestPlotPerCellDiagnosticScatter:
    def test_with_spike_times_aligns_at_actual_spike_times(self) -> None:
        """``spike_times`` shifts scatter dots to the actual spike instants
        instead of the bin starts (which are 100ms apart here)."""
        time = np.linspace(0.0, 0.9, 10)
        diagnostics = {"hpd_overlap": np.full((10, 3), np.nan)}
        diagnostics["hpd_overlap"][1, 0] = 0.8
        diagnostics["hpd_overlap"][3, 1] = 0.6
        diagnostics["hpd_overlap"][5, 2] = 0.4

        fig, ax = plt.subplots()
        plot_per_cell_diagnostic_scatter(
            time,
            diagnostics,
            ax=ax,
            spike_times=[
                np.array([0.15]),
                np.array([0.35]),
                np.array([0.55]),
            ],
        )
        offsets = _scatter_offsets(ax)
        np.testing.assert_allclose(sorted(offsets[:, 0]), [0.15, 0.35, 0.55])
        plt.close(fig)

    def test_event_diagnostics_plot_at_exact_event_times(self) -> None:
        """When ``event_*`` arrays are present, scatter uses their times
        directly with no bin lookup."""
        time = np.linspace(0.0, 0.9, 10)
        diagnostics = {
            "hpd_overlap": np.full((10, 1), np.nan),
            "event_time": np.array([0.151, 0.157]),
            "event_hpd_overlap": np.array([0.8, 0.6]),
        }
        diagnostics["hpd_overlap"][1, 0] = 0.7

        fig, ax = plt.subplots()
        plot_per_cell_diagnostic_scatter(time, diagnostics, ax=ax)
        offsets = _scatter_offsets(ax)
        np.testing.assert_allclose(offsets[:, 0], [0.151, 0.157])
        np.testing.assert_allclose(offsets[:, 1], [0.8, 0.6])
        plt.close(fig)

    def test_without_spike_times_uses_bin_centers(self) -> None:
        """Without per-spike alignment, scatter uses bin-start times."""
        time = np.linspace(0.0, 0.9, 10)
        diagnostics = {"hpd_overlap": np.full((10, 2), np.nan)}
        diagnostics["hpd_overlap"][1, 0] = 0.8
        diagnostics["hpd_overlap"][3, 1] = 0.6

        fig, ax = plt.subplots()
        plot_per_cell_diagnostic_scatter(time, diagnostics, ax=ax, spike_times=None)
        offsets = _scatter_offsets(ax)
        np.testing.assert_allclose(sorted(offsets[:, 0]), [0.1, 0.3])
        plt.close(fig)


# ---------------------------------------------------------------------------
# compute_running_average
# ---------------------------------------------------------------------------


class TestComputeRunningAverage:
    def test_output_shape_collapses_cells(self, rng: np.random.Generator) -> None:
        n_time, n_cells = 100, 10
        metric = rng.random((n_time, n_cells))
        time = np.linspace(0, 1, n_time)
        running_avg, time_out = compute_running_average(metric, time, window_size=0.1)
        assert running_avg.shape == (n_time,)
        np.testing.assert_array_equal(time_out, time)

    def test_partial_nan_input_yields_finite_output(self, rng: np.random.Generator) -> None:
        n_time, n_cells = 100, 10
        metric = rng.random((n_time, n_cells))
        metric[::2] = np.nan
        time = np.linspace(0, 1, n_time)
        running_avg, _ = compute_running_average(metric, time, window_size=0.1)
        assert not np.any(np.isnan(running_avg))

    def test_all_nan_input_yields_all_nan_output(self) -> None:
        """All-NaN must propagate, not be silently filled with zeros."""
        n_time, n_cells = 100, 10
        metric = np.full((n_time, n_cells), np.nan)
        time = np.linspace(0, 1, n_time)
        running_avg, _ = compute_running_average(metric, time, window_size=0.1)
        assert np.all(np.isnan(running_avg))

    def test_larger_window_smooths_more(self, rng: np.random.Generator) -> None:
        n_time, n_cells = 1000, 10
        metric = rng.random((n_time, n_cells))
        time = np.linspace(0, 1, n_time)
        small, _ = compute_running_average(metric, time, window_size=0.01)
        large, _ = compute_running_average(metric, time, window_size=0.1)
        assert large.var() < small.var()

    def test_event_inputs_count_duplicates_at_same_time(self) -> None:
        """Two events at the same time both contribute to the running mean."""
        metric = np.full((3, 1), np.nan)
        time = np.array([0.0, 1.0, 2.0])
        running_avg, _ = compute_running_average(
            metric,
            time,
            window_size=0.1,
            event_times=np.array([1.0, 1.0]),
            event_values=np.array([1.0, 3.0]),
        )
        assert np.isnan(running_avg[0])
        assert running_avg[1] == 2.0
        assert np.isnan(running_avg[2])


class TestPlotPerCellDiagnosticScatterRunningAverage:
    def test_running_average_adds_a_line_to_axis(self, rng: np.random.Generator) -> None:
        time = np.linspace(0.0, 1.0, 100)
        diagnostics = {"hpd_overlap": rng.random((100, 10))}

        fig_off, ax_off = plt.subplots()
        plot_per_cell_diagnostic_scatter(time, diagnostics, ax=ax_off, show_running_average=False)
        n_off = len(ax_off.lines)
        plt.close(fig_off)

        fig_on, ax_on = plt.subplots()
        plot_per_cell_diagnostic_scatter(time, diagnostics, ax=ax_on, show_running_average=True)
        assert len(ax_on.lines) == n_off + 1
        plt.close(fig_on)

    def test_running_average_window_size_changes_curve(self, rng: np.random.Generator) -> None:
        time = np.linspace(0.0, 1.0, 100)
        diagnostics = {"hpd_overlap": rng.random((100, 10))}

        def _line_y(window: float) -> np.ndarray:
            fig, ax = plt.subplots()
            plot_per_cell_diagnostic_scatter(
                time,
                diagnostics,
                ax=ax,
                show_running_average=True,
                running_average_window=window,
            )
            y = np.asarray(ax.lines[0].get_ydata()).copy()
            plt.close(fig)
            return y

        assert not np.allclose(_line_y(0.05), _line_y(0.2))

    def test_spike_prob_running_average_uses_raw_then_transforms(self) -> None:
        """Critical correctness: -log10(mean(p)) != mean(-log10(p)). Running
        average must average raw probabilities first, then take -log10."""
        spike_probs = np.array(
            [
                [0.01, 0.99],  # mean(raw) = 0.5
                [0.1, 0.9],  # mean(raw) = 0.5
                [0.5, 0.5],  # mean(raw) = 0.5 (control)
            ]
        )
        time = np.linspace(0, 0.2, 3)
        diagnostics = {"spike_prob": spike_probs}

        fig, ax = plt.subplots()
        plot_per_cell_diagnostic_scatter(
            time,
            diagnostics,
            ax=ax,
            metric_name="spike_prob",
            show_running_average=True,
            running_average_window=0.01,
        )
        y_actual = np.asarray(ax.lines[0].get_ydata())

        # Correct path: average raw, then -log10.
        expected = -np.log10(np.maximum(np.mean(spike_probs, axis=1), 1e-10))
        np.testing.assert_allclose(y_actual, expected, rtol=1e-3)

        # Wrong path: -log10 first, then average. Different on rows 0 and 1.
        wrong = np.mean(-np.log10(np.maximum(spike_probs, 1e-10)), axis=1)
        assert not np.allclose(y_actual, wrong, rtol=1e-3)
        plt.close(fig)
