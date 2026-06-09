"""Tests for real data analysis module."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from statespacecheck_paper.analysis import PerCellDiagnostics
from statespacecheck_paper.real_data_analysis import (
    compute_flag_confusion,
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


def _diagnostics_from_metric(
    metric_name: str,
    metric: np.ndarray,
    *,
    event_time: np.ndarray | None = None,
    event_values: np.ndarray | None = None,
) -> PerCellDiagnostics:
    """Build a ``PerCellDiagnostics`` from a single (n_time, n_cells) metric.

    Other metric fields are filled with NaN/zeros matching shape; the
    scatter helper under test only consumes the named metric plus the
    optional ``event_*`` arrays.
    """
    n_time, n_cells = metric.shape
    blank_2d = np.full((n_time, n_cells), np.nan)
    n_spikes = 0 if event_time is None else event_time.shape[0]
    blank_evt = np.zeros(n_spikes)

    def _named(name: str, value: np.ndarray) -> np.ndarray:
        return value if name == metric_name else blank_2d

    def _named_evt(name: str) -> np.ndarray:
        if event_values is not None and name == f"event_{metric_name}":
            return event_values
        return blank_evt

    return PerCellDiagnostics(
        event_time_ind=np.zeros(n_spikes, dtype=np.intp),
        event_cell_ind=np.zeros(n_spikes, dtype=np.intp),
        event_hpd_overlap=_named_evt("event_hpd_overlap"),
        event_kl_divergence=_named_evt("event_kl_divergence"),
        event_spike_prob=_named_evt("event_spike_prob"),
        hpd_overlap=_named("hpd_overlap", metric),
        kl_divergence=_named("kl_divergence", metric),
        spike_prob=_named("spike_prob", metric),
        per_spike_likelihood=np.zeros((n_spikes, 1)),
        event_time=event_time,
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
            arr = getattr(result, key)
            assert arr is not None
            assert arr.shape == (per_cell_setup["n_time"], per_cell_setup["n_cells"])

    def test_nan_exactly_where_no_spikes(self, per_cell_setup: dict) -> None:
        result = compute_per_cell_diagnostics(
            per_cell_setup["predictive"],
            per_cell_setup["spike_counts"],
            per_cell_setup["place_fields"],
        )
        no_spike = per_cell_setup["spike_counts"] == 0
        for key in ("hpd_overlap", "kl_divergence", "spike_prob"):
            arr = getattr(result, key)
            assert arr is not None
            assert np.all(np.isnan(arr[no_spike]))

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
        arr = getattr(result, metric)
        assert arr is not None
        valid = arr[~np.isnan(arr)]
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
        assert result.hpd_overlap is not None
        np.testing.assert_array_equal(np.isnan(result.hpd_overlap), spike_counts == 0)

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

        assert result.event_time is not None
        assert result.kl_divergence is not None
        np.testing.assert_allclose(result.event_time, [1.10, 1.20])
        np.testing.assert_array_equal(result.event_time_ind, [1, 1])
        np.testing.assert_array_equal(result.event_cell_ind, [0, 0])
        for event_key in ("event_hpd_overlap", "event_kl_divergence", "event_spike_prob"):
            assert getattr(result, event_key).shape == (2,)
        np.testing.assert_allclose(
            result.event_kl_divergence,
            np.repeat(result.kl_divergence[1, 0], 2),
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

    def test_unstack_failure_raises(self) -> None:
        """A malformed state_bins index that fails to unstack must
        raise — silently treating it as single-state and returning a
        per-state slice would render a wrong figure with no warning."""
        # Build a posterior with a state_bins coordinate that has
        # duplicate (state, position) entries, which xarray cannot
        # unstack into a (state, position) rectangular product.
        n_time = 20
        # Duplicated (state, position) tuples — unstack fails.
        broken_index = pd.MultiIndex.from_tuples(
            [("A", 0), ("A", 0), ("B", 0), ("B", 0)], names=["state", "position"]
        )
        results = _xarray_results(
            np.random.default_rng(0).random((n_time, 4)),
            "predictive_posterior",
            state_bins=broken_index,
        )
        with pytest.raises(ValueError, match="Failed to unstack"):
            get_state_marginalized_posterior(results, "predictive")


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
        hpd = np.full((10, 3), np.nan)
        hpd[1, 0] = 0.8
        hpd[3, 1] = 0.6
        hpd[5, 2] = 0.4
        diagnostics = _diagnostics_from_metric("hpd_overlap", hpd)

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
        hpd = np.full((10, 1), np.nan)
        hpd[1, 0] = 0.7
        diagnostics = _diagnostics_from_metric(
            "hpd_overlap",
            hpd,
            event_time=np.array([0.151, 0.157]),
            event_values=np.array([0.8, 0.6]),
        )

        fig, ax = plt.subplots()
        plot_per_cell_diagnostic_scatter(time, diagnostics, ax=ax)
        offsets = _scatter_offsets(ax)
        np.testing.assert_allclose(offsets[:, 0], [0.151, 0.157])
        np.testing.assert_allclose(offsets[:, 1], [0.8, 0.6])
        plt.close(fig)

    def test_without_spike_times_uses_bin_centers(self) -> None:
        """Without per-spike alignment, scatter uses bin-start times."""
        time = np.linspace(0.0, 0.9, 10)
        hpd = np.full((10, 2), np.nan)
        hpd[1, 0] = 0.8
        hpd[3, 1] = 0.6
        diagnostics = _diagnostics_from_metric("hpd_overlap", hpd)

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
        diagnostics = _diagnostics_from_metric("hpd_overlap", rng.random((100, 10)))

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
        diagnostics = _diagnostics_from_metric("hpd_overlap", rng.random((100, 10)))

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
        """Critical correctness: -log(mean(p)) != mean(-log(p)). Running
        average must average raw probabilities first, then take -log."""
        spike_probs = np.array(
            [
                [0.01, 0.99],  # mean(raw) = 0.5
                [0.1, 0.9],  # mean(raw) = 0.5
                [0.5, 0.5],  # mean(raw) = 0.5 (control)
            ]
        )
        time = np.linspace(0, 0.2, 3)
        diagnostics = _diagnostics_from_metric("spike_prob", spike_probs)

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

        # Correct path: average raw, then -log (natural log).
        expected = -np.log(np.maximum(np.mean(spike_probs, axis=1), 1e-10))
        np.testing.assert_allclose(y_actual, expected, rtol=1e-3)

        # Wrong path: -log first, then average. Different on rows 0 and 1.
        wrong = np.mean(-np.log(np.maximum(spike_probs, 1e-10)), axis=1)
        assert not np.allclose(y_actual, wrong, rtol=1e-3)
        plt.close(fig)


def _diag_from_events(
    *,
    hpd: np.ndarray | None = None,
    kl: np.ndarray | None = None,
    sp: np.ndarray | None = None,
) -> PerCellDiagnostics:
    """Minimal ``PerCellDiagnostics`` carrying only the per-spike event arrays.

    ``compute_flag_confusion`` reads a single ``event_*`` array; the rest of the
    dataclass is required by the constructor but unused here.
    """
    present = [a for a in (hpd, kl, sp) if a is not None]
    n = present[0].shape[0]
    zeros = np.zeros(n)
    return PerCellDiagnostics(
        event_time_ind=np.zeros(n, dtype=np.intp),
        event_cell_ind=np.zeros(n, dtype=np.intp),
        event_hpd_overlap=hpd if hpd is not None else zeros,
        event_kl_divergence=kl if kl is not None else zeros,
        event_spike_prob=sp if sp is not None else zeros,
        hpd_overlap=None,
        kl_divergence=None,
        spike_prob=None,
        per_spike_likelihood=None,
    )


class TestComputeFlagConfusion:
    def test_below_direction_counts_and_rescue_rate(self) -> None:
        a = _diag_from_events(hpd=np.array([0.01, 0.02, 0.10, 0.20, 0.03]))
        b = _diag_from_events(hpd=np.array([0.01, 0.20, 0.02, 0.20, 0.20]))
        conf = compute_flag_confusion(a, b, "hpd_overlap", 0.05, worse_when="below")
        assert (conf.n, conf.both, conf.a_only, conf.b_only, conf.neither) == (5, 1, 2, 1, 1)
        assert conf.both + conf.a_only + conf.b_only + conf.neither == conf.n
        assert conf.rescue_rate == pytest.approx(2 / 3)

    def test_above_direction(self) -> None:
        a = _diag_from_events(kl=np.array([5.0, 6.0, 1.0, 2.0]))
        b = _diag_from_events(kl=np.array([5.0, 1.0, 7.0, 1.0]))
        conf = compute_flag_confusion(a, b, "kl_divergence", 4.0, worse_when="above")
        assert (conf.both, conf.a_only, conf.b_only, conf.neither) == (1, 1, 1, 1)
        assert conf.rescue_rate == pytest.approx(0.5)

    def test_nan_events_are_dropped(self) -> None:
        a = _diag_from_events(hpd=np.array([0.01, np.nan, 0.02]))
        b = _diag_from_events(hpd=np.array([0.20, 0.01, 0.02]))
        conf = compute_flag_confusion(a, b, "hpd_overlap", 0.05, worse_when="below")
        # The NaN spike is dropped; remaining A=[0.01, 0.02], B=[0.20, 0.02].
        assert (conf.n, conf.both, conf.a_only, conf.b_only, conf.neither) == (2, 1, 1, 0, 0)

    def test_rescue_rate_nan_when_a_flags_nothing(self) -> None:
        a = _diag_from_events(hpd=np.array([0.5, 0.6]))  # none below 0.05
        b = _diag_from_events(hpd=np.array([0.01, 0.6]))
        conf = compute_flag_confusion(a, b, "hpd_overlap", 0.05, worse_when="below")
        assert conf.a_only == 0 and conf.both == 0
        assert np.isnan(conf.rescue_rate)

    def test_rejects_bad_direction(self) -> None:
        a = _diag_from_events(hpd=np.array([0.1]))
        bad: Any = "sideways"
        with pytest.raises(ValueError, match="worse_when"):
            compute_flag_confusion(a, a, "hpd_overlap", 0.05, worse_when=bad)

    def test_rejects_length_mismatch(self) -> None:
        a = _diag_from_events(hpd=np.array([0.1, 0.2]))
        b = _diag_from_events(hpd=np.array([0.1]))
        with pytest.raises(ValueError, match="same set of spike events"):
            compute_flag_confusion(a, b, "hpd_overlap", 0.05, worse_when="below")
