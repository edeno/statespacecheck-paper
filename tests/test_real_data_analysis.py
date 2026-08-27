"""Tests for real data analysis module."""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

import statespacecheck_paper.real_data_analysis as real_data_analysis
from statespacecheck_paper.analysis import PerCellDiagnostics
from statespacecheck_paper.real_data_analysis import (
    compute_flag_confusion,
    compute_model_diagnostics,
    compute_per_cell_diagnostics,
    compute_running_average,
    extract_place_fields,
    extract_shared_position_place_fields,
    gaussian_smooth,
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
        event_predictive_pvalue=_named_evt("event_predictive_pvalue"),
        hpd_overlap=_named("hpd_overlap", metric),
        kl_divergence=_named("kl_divergence", metric),
        predictive_pvalue=_named("predictive_pvalue", metric),
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
# gaussian_smooth
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
        for key in ("hpd_overlap", "kl_divergence", "predictive_pvalue"):
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
        for key in ("hpd_overlap", "kl_divergence", "predictive_pvalue"):
            arr = getattr(result, key)
            assert arr is not None
            assert np.all(np.isnan(arr[no_spike]))

    @pytest.mark.parametrize("metric", ["hpd_overlap", "predictive_pvalue"])
    def test_metric_in_unit_range_with_gaussian_place_fields(
        self, rng: np.random.Generator, metric: str
    ) -> None:
        """HPD overlap and predictive_pvalue are bounded in [0, 1] (allowing tiny
        floating-point overshoot above 1 for predictive_pvalue)."""
        n_time, n_bins, n_cells = 100, 50, 10
        predictive = rng.dirichlet(np.ones(n_bins), size=n_time)
        # Gaussian place fields ensure the event spike-prob rank stays well-defined.
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
        for event_key in ("event_hpd_overlap", "event_kl_divergence", "event_predictive_pvalue"):
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


class TestMeanPerSpikeLikelihoodByTime:
    def test_averages_normalized_fields_weighted_by_spike_count(self) -> None:
        from statespacecheck_paper.real_data_analysis import (
            mean_per_spike_likelihood_by_time,
        )

        # cell 0 concentrates at bin 0, cell 1 at bin 2 (unnormalized).
        place_fields = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
        spike_counts = np.array([[1, 0], [1, 1], [0, 0]], dtype=np.int64)

        mean_lik, has_spikes = mean_per_spike_likelihood_by_time(spike_counts, place_fields)

        expected = np.array([[1.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(mean_lik, expected)
        np.testing.assert_array_equal(has_spikes, np.array([True, True, False]))

    def test_uses_normalized_poisson_likelihood_not_raw_rate(self) -> None:
        from scipy.stats import poisson

        from statespacecheck_paper.real_data_analysis import (
            mean_per_spike_likelihood_by_time,
        )

        # One cell, two bins, with rates large enough that the Poisson
        # exp(-rate) factor matters. Raw-rate normalization would give
        # [0.8, 0.2]; the diagnostics normalize Poisson(k=1, mu=rate).
        place_fields = np.array([[2.0, 0.5]])
        spike_counts = np.array([[1]], dtype=np.int64)

        mean_lik, _ = mean_per_spike_likelihood_by_time(spike_counts, place_fields)

        pmf = poisson.pmf(k=1, mu=place_fields[0])
        expected = pmf / pmf.sum()
        np.testing.assert_allclose(mean_lik[0], expected)
        assert not np.allclose(mean_lik[0], place_fields[0] / place_fields[0].sum())

    def test_matches_diagnostics_per_spike_likelihood(self) -> None:
        # Parity across the two entry points that both claim to plot/consume
        # the same per-spike likelihood: the plotting helper (place fields,
        # (n_cells, n_bins)) and the diagnostics (rates, (n_bins, n_cells)).
        from statespacecheck_paper.analysis import (
            compute_per_cell_diagnostics_from_rates,
        )
        from statespacecheck_paper.real_data_analysis import (
            mean_per_spike_likelihood_by_time,
        )

        place_fields = np.array([[2.0, 0.5, 1.0], [0.1, 0.4, 0.3]])  # (n_cells, n_bins)
        spike_counts = np.array([[0, 1]], dtype=np.int64)  # one spike from cell 1 at t=0
        mean_lik, _ = mean_per_spike_likelihood_by_time(spike_counts, place_fields)

        predictive = np.full((1, 3), 1.0 / 3.0)
        diag = compute_per_cell_diagnostics_from_rates(
            predictive,
            place_fields.T,
            np.array([0], dtype=np.intp),
            np.array([1], dtype=np.intp),
            include_dense_matrices=True,
        )
        assert diag.per_spike_likelihood is not None
        np.testing.assert_allclose(mean_lik[0], diag.per_spike_likelihood[0])


# ---------------------------------------------------------------------------
# position-marginal model diagnostics
# ---------------------------------------------------------------------------


def _two_state_model(
    place_fields: np.ndarray,
    position_bins: np.ndarray,
    interior_mask: np.ndarray,
) -> MagicMock:
    model = MagicMock()
    model.observation_models = [
        MagicMock(environment_name="", encoding_group=0),
        MagicMock(environment_name="", encoding_group=0),
    ]
    model.encoding_model_ = {("", 0): {"place_fields": place_fields}}
    environment = MagicMock()
    environment.place_bin_centers_ = position_bins[:, np.newaxis]
    model.environments = [environment]
    model.is_track_interior_state_bins_ = np.tile(interior_mask, 2)
    return model


class TestComputeModelDiagnostics:
    def test_marginalizes_state_and_uses_one_shared_place_field_copy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        position_bins = np.array([0.0, 1.0, 2.0])
        interior_mask = np.array([True, False, True])
        place_fields = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        model = _two_state_model(place_fields, position_bins, interior_mask)

        posterior_per_state = np.array(
            [
                [[0.10, np.nan, 0.20], [0.30, np.nan, 0.40]],
                [[0.25, np.nan, 0.15], [0.35, np.nan, 0.25]],
            ]
        )
        state_bins = pd.MultiIndex.from_product(
            [["Continuous", "Fragmented"], position_bins],
            names=["state", "position"],
        )
        results = _xarray_results(
            posterior_per_state.reshape(2, -1),
            "predictive_posterior",
            state_bins=state_bins,
        )

        captured: dict[str, Any] = {}
        sentinel = MagicMock()

        def _capture(
            predictive_posterior: np.ndarray,
            spike_counts: np.ndarray,
            diagnostic_place_fields: np.ndarray,
            **kwargs: Any,
        ) -> MagicMock:
            captured["predictive"] = predictive_posterior
            captured["place_fields"] = diagnostic_place_fields
            captured["kwargs"] = kwargs
            return sentinel

        monkeypatch.setattr(real_data_analysis, "compute_per_cell_diagnostics", _capture)
        spike_counts = np.zeros((2, 2), dtype=np.int64)
        time = np.array([0.0, 0.002])
        result = compute_model_diagnostics(model, results, spike_counts, time)

        assert result is sentinel
        np.testing.assert_allclose(
            captured["predictive"],
            posterior_per_state[:, :, interior_mask].sum(axis=1),
        )
        np.testing.assert_allclose(captured["place_fields"], place_fields[:, interior_mask])
        assert captured["kwargs"]["time"] is time

    def test_rejects_state_specific_observation_likelihoods(self) -> None:
        position_bins = np.array([0.0, 1.0, 2.0])
        model = MagicMock()
        model.observation_models = [
            MagicMock(environment_name="", encoding_group=0),
            MagicMock(environment_name="", encoding_group=1),
        ]
        model.encoding_model_ = {
            ("", 0): {"place_fields": np.ones((2, 3))},
            ("", 1): {"place_fields": np.full((2, 3), 2.0)},
        }
        environments = []
        for _ in range(2):
            environment = MagicMock()
            environment.place_bin_centers_ = position_bins[:, np.newaxis]
            environments.append(environment)
        model.environments = environments
        model.is_track_interior_state_bins_ = np.ones(6, dtype=bool)

        with pytest.raises(ValueError, match="likelihood differs"):
            extract_shared_position_place_fields(model)

    def test_rejects_state_dependent_interior_mask(self) -> None:
        # Same place fields and grid across states, but the track-interior
        # mask differs per state -> the shared position marginal is ill-defined
        # and must be rejected rather than silently reshaped into a wrong mask.
        position_bins = np.array([0.0, 1.0, 2.0])
        place_fields = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        model = MagicMock()
        model.observation_models = [
            MagicMock(environment_name="", encoding_group=0),
            MagicMock(environment_name="", encoding_group=0),
        ]
        model.encoding_model_ = {("", 0): {"place_fields": place_fields}}
        environment = MagicMock()
        environment.place_bin_centers_ = position_bins[:, np.newaxis]
        model.environments = [environment]
        # State 0 interior [T, F, T]; state 1 interior [T, T, F] -> differ.
        model.is_track_interior_state_bins_ = np.array([True, False, True, True, True, False])

        with pytest.raises(ValueError, match="interior mask differs"):
            extract_shared_position_place_fields(model)


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

    def test_predictive_pvalue_running_average_uses_raw_then_transforms(self) -> None:
        """Critical correctness: -log(mean(p)) != mean(-log(p)). Running
        average must average raw probabilities first, then take -log."""
        predictive_pvalues = np.array(
            [
                [0.01, 0.99],  # mean(raw) = 0.5
                [0.1, 0.9],  # mean(raw) = 0.5
                [0.5, 0.5],  # mean(raw) = 0.5 (control)
            ]
        )
        time = np.linspace(0, 0.2, 3)
        diagnostics = _diagnostics_from_metric("predictive_pvalue", predictive_pvalues)

        fig, ax = plt.subplots()
        plot_per_cell_diagnostic_scatter(
            time,
            diagnostics,
            ax=ax,
            metric_name="predictive_pvalue",
            show_running_average=True,
            running_average_window=0.01,
        )
        y_actual = np.asarray(ax.lines[0].get_ydata())

        # Correct path: average raw, then -log (natural log).
        expected = -np.log(np.maximum(np.mean(predictive_pvalues, axis=1), 1e-10))
        np.testing.assert_allclose(y_actual, expected, rtol=1e-3)

        # Wrong path: -log first, then average. Different on rows 0 and 1.
        wrong = np.mean(-np.log(np.maximum(predictive_pvalues, 1e-10)), axis=1)
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
        event_predictive_pvalue=sp if sp is not None else zeros,
        hpd_overlap=None,
        kl_divergence=None,
        predictive_pvalue=None,
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

    def test_threshold_values_are_inclusive(self) -> None:
        hpd_a = _diag_from_events(hpd=np.array([0.05, 0.10]))
        hpd_b = _diag_from_events(hpd=np.array([0.10, 0.05]))
        hpd_conf = compute_flag_confusion(hpd_a, hpd_b, "hpd_overlap", 0.05, worse_when="below")
        assert (hpd_conf.a_only, hpd_conf.b_only) == (1, 1)

        kl_a = _diag_from_events(kl=np.array([4.0, 3.0]))
        kl_b = _diag_from_events(kl=np.array([3.0, 4.0]))
        kl_conf = compute_flag_confusion(kl_a, kl_b, "kl_divergence", 4.0, worse_when="above")
        assert (kl_conf.a_only, kl_conf.b_only) == (1, 1)

    def test_nan_events_are_dropped(self) -> None:
        a = _diag_from_events(hpd=np.array([0.01, np.nan, 0.02]))
        b = _diag_from_events(hpd=np.array([0.20, 0.01, 0.02]))
        conf = compute_flag_confusion(a, b, "hpd_overlap", 0.05, worse_when="below")
        # The NaN spike is dropped; remaining A=[0.01, 0.02], B=[0.20, 0.02].
        assert (conf.n, conf.both, conf.a_only, conf.b_only, conf.neither) == (2, 1, 1, 0, 0)

    def test_rescue_rate_nan_when_a_flags_nothing(self) -> None:
        a = _diag_from_events(hpd=np.array([0.5, 0.6]))  # none at or below 0.05
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


class TestFigure4ConfigMatchesManuscript:
    """Drift guard: the decoder models the code actually builds must carry the
    exact parameters stated in the manuscript (``main.tex:294``).

    Rather than reconstruct the nested transition structure (which risks
    changing the decode), this fits nothing and asserts the *resolved*
    attributes of the models produced by
    :func:`real_data_analysis.build_decoder_models`. These attributes are set at
    construction time from ``non_local_detector`` class defaults plus the two
    explicit KDE parameters, so the check needs neither real data nor a fit; a
    dependency bump that silently changes any default fails here.
    """

    @staticmethod
    def _build_models() -> tuple[Any, Any]:
        pytest.importorskip("non_local_detector")
        from non_local_detector.environment import Environment

        # A default Environment reproduces create_decoder_environment's place
        # bin size: that function passes only the track graph / edge order /
        # spacing, so place_bin_size falls back to the class default.
        return real_data_analysis.build_decoder_models(Environment())

    def test_continuous_observation_and_transition(self) -> None:
        from non_local_detector.continuous_state_transitions import RandomWalk

        continuous_model, _ = self._build_models()
        config = real_data_analysis.Figure4Config()

        # main.tex:294 -- sorted-spikes KDE positional bandwidth sqrt(12.5).
        assert continuous_model.sorted_spikes_algorithm_params["position_std"] == pytest.approx(
            config.position_std
        )
        assert config.position_std == pytest.approx(float(np.sqrt(12.5)))

        # main.tex:294 -- zero-mean Gaussian random walk, movement_var = 6.0 cm^2.
        random_walk = continuous_model.continuous_transition_types[0][0]
        assert isinstance(random_walk, RandomWalk)
        assert random_walk.movement_var == pytest.approx(config.movement_var)
        assert config.movement_var == pytest.approx(6.0)

    def test_contfrag_discrete_dynamics(self) -> None:
        from non_local_detector.continuous_state_transitions import RandomWalk
        from non_local_detector.discrete_state_transitions import DiscreteStationaryDiagonal

        _, contfrag_model = self._build_models()
        config = real_data_analysis.Figure4Config()

        # main.tex:294 -- ContFrag Continuous-to-Continuous transition reuses the
        # same random walk (movement_var = 6.0).
        random_walk = contfrag_model.continuous_transition_types[0][0]
        assert isinstance(random_walk, RandomWalk)
        assert random_walk.movement_var == pytest.approx(config.movement_var)

        # main.tex:294 -- mode-transition matrix [[0.98, 0.02], [0.02, 0.98]],
        # i.e. a stationary diagonal (0.98, 0.98).
        discrete_transition_type = contfrag_model.discrete_transition_type
        assert isinstance(discrete_transition_type, DiscreteStationaryDiagonal)
        np.testing.assert_array_equal(
            np.asarray(discrete_transition_type.diagonal_values, dtype=float),
            np.asarray(config.contfrag_diagonal_values, dtype=float),
        )

        # main.tex:294 -- Continuous / Fragmented modes initialized at (0.5, 0.5).
        np.testing.assert_array_equal(
            np.asarray(contfrag_model.discrete_initial_conditions, dtype=float),
            np.asarray(config.contfrag_discrete_initial_conditions, dtype=float),
        )

    def test_unprinted_effective_defaults(self) -> None:
        """Concentration / regularization are not printed in the manuscript but
        shape the decode; pin them so a dependency bump fails loudly."""
        continuous_model, contfrag_model = self._build_models()
        config = real_data_analysis.Figure4Config()

        assert contfrag_model.discrete_transition_concentration == pytest.approx(
            config.discrete_transition_concentration
        )
        assert config.discrete_transition_concentration == pytest.approx(1.1)

        for model in (continuous_model, contfrag_model):
            assert model.discrete_transition_regularization == pytest.approx(
                config.discrete_transition_regularization
            )
        assert config.discrete_transition_regularization == pytest.approx(1e-10)

    def test_binning_values_the_code_uses(self) -> None:
        """Position bin size (from the Environment) and time bin size (from the
        sampling frequency) are the values the decode actually uses."""
        continuous_model, contfrag_model = self._build_models()
        config = real_data_analysis.Figure4Config()

        # main.tex:294 -- ~2 cm spatial bins.
        for model in (continuous_model, contfrag_model):
            assert model.environments[0].place_bin_size == pytest.approx(
                config.position_bin_size_cm
            )
        assert config.position_bin_size_cm == pytest.approx(2.0)

        # main.tex:294 -- 2 ms spike bins == 500 Hz sampling frequency.
        for model in (continuous_model, contfrag_model):
            assert model.sampling_frequency == pytest.approx(config.sampling_frequency_hz)
        assert config.sampling_frequency_hz == pytest.approx(500.0)
        assert config.time_bin_size_ms == pytest.approx(2.0)

    def test_config_records_manuscript_dependency_version(self) -> None:
        """The provenance string must match the manuscript-stated version."""
        config = real_data_analysis.Figure4Config()
        assert config.non_local_detector_version == "0.6.10.dev214+g956fdccaf"
