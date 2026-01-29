"""Tests for real data analysis module."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import xarray as xr

from statespacecheck_paper.real_data_analysis import (
    compute_per_cell_diagnostics,
    extract_place_fields,
    find_sustained_low_overlap,
    gaussian_smooth,
    get_multiunit_population_firing_rate,
    get_state_marginalized_posterior,
)


class TestGaussianSmooth:
    """Tests for gaussian_smooth function."""

    def test_output_shape_matches_input(self) -> None:
        """Test that output has same shape as input."""
        data = np.random.randn(1000)
        result = gaussian_smooth(data, sigma=0.01, sampling_frequency=1000)
        assert result.shape == data.shape

    def test_smoothing_reduces_variance(self) -> None:
        """Test that smoothing reduces variance of noisy data."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(1000)
        result = gaussian_smooth(data, sigma=0.02, sampling_frequency=500)
        assert np.var(result) < np.var(data)


class TestGetMultiunitPopulationFiringRate:
    """Tests for get_multiunit_population_firing_rate function."""

    def test_output_shape(self) -> None:
        """Test that output has shape (n_time,)."""
        multiunit = np.random.poisson(0.1, size=(1000, 50))
        result = get_multiunit_population_firing_rate(
            multiunit, sampling_frequency=500, smoothing_sigma=0.015
        )
        assert result.shape == (1000,)


class TestFindSustainedLowOverlap:
    """Tests for find_sustained_low_overlap function."""

    def test_returns_list(self) -> None:
        """Test that function returns a list."""
        hpd_overlap = np.random.rand(1000)
        result = find_sustained_low_overlap(hpd_overlap, threshold=0.3)
        assert isinstance(result, list)

    def test_finds_low_regions(self) -> None:
        """Test that function finds regions below threshold."""
        # Create data with a clear low region
        hpd_overlap = np.ones(1000)
        hpd_overlap[200:300] = 0.1  # Low region
        result = find_sustained_low_overlap(
            hpd_overlap, threshold=0.5, min_duration=0.05, sampling_frequency=1000
        )
        # Should find the low region
        assert len(result) >= 1
        # First region should be around 200-300
        start, end = result[0]
        assert 150 < start < 250
        assert 250 < end < 350


class TestExtractPlaceFields:
    """Tests for extract_place_fields function."""

    def test_extracts_place_fields_from_mock_model(self) -> None:
        """Test extraction from a mock model object."""
        # Arrange: create mock model
        n_cells, n_bins = 10, 50
        mock_place_fields = np.random.rand(n_cells, n_bins) * 10
        mock_position_bins = np.linspace(0, 100, n_bins)

        mock_model = MagicMock()
        mock_model.encoding_model_ = {("", 0): {"place_fields": mock_place_fields}}
        mock_env = MagicMock()
        mock_env.place_bin_centers_ = mock_position_bins.reshape(-1, 1)
        mock_model.environments = [mock_env]

        # Act
        place_fields, position_bins = extract_place_fields(mock_model)

        # Assert
        assert place_fields.shape == (n_cells, n_bins)
        assert position_bins.shape == (n_bins,)
        np.testing.assert_array_equal(place_fields, mock_place_fields)
        np.testing.assert_array_equal(position_bins, mock_position_bins)

    def test_custom_environment_name_and_group(self) -> None:
        """Test extraction with custom environment name and encoding group."""
        # Arrange
        n_cells, n_bins = 5, 30
        mock_place_fields = np.random.rand(n_cells, n_bins) * 10
        mock_position_bins = np.linspace(0, 50, n_bins)

        mock_model = MagicMock()
        mock_model.encoding_model_ = {("env1", 1): {"place_fields": mock_place_fields}}
        mock_env = MagicMock()
        mock_env.place_bin_centers_ = mock_position_bins.reshape(-1, 1)
        mock_model.environments = [None, mock_env]  # Index 1 for encoding_group=1

        # Act
        place_fields, position_bins = extract_place_fields(
            mock_model, environment_name="env1", encoding_group=1
        )

        # Assert
        np.testing.assert_array_equal(place_fields, mock_place_fields)
        np.testing.assert_array_equal(position_bins, mock_position_bins)


class TestComputePerCellDiagnostics:
    """Tests for compute_per_cell_diagnostics function.

    The function computes diagnostics using the actual spike count for each
    spike event, matching the approach in analysis.py for simulated data.
    """

    def test_output_shapes(self) -> None:
        """Test that output dict has arrays with correct shapes."""
        n_time, n_bins, n_cells = 100, 50, 10
        rng = np.random.default_rng(42)
        predictive = rng.dirichlet(np.ones(n_bins), size=n_time)
        place_fields = rng.random((n_cells, n_bins)) * 10 + 0.1
        spike_counts = rng.poisson(0.5, (n_time, n_cells)).astype(np.int64)

        result = compute_per_cell_diagnostics(predictive, spike_counts, place_fields)

        assert "hpd_overlap" in result
        assert "kl_divergence" in result
        assert "spike_prob" in result
        assert result["hpd_overlap"].shape == (n_time, n_cells)
        assert result["kl_divergence"].shape == (n_time, n_cells)
        assert result["spike_prob"].shape == (n_time, n_cells)

    def test_nan_where_no_spikes(self) -> None:
        """Test that metrics are NaN where spike_counts == 0."""
        n_time, n_bins, n_cells = 50, 30, 5
        rng = np.random.default_rng(42)
        predictive = rng.dirichlet(np.ones(n_bins), size=n_time)
        place_fields = rng.random((n_cells, n_bins)) * 10 + 0.1
        spike_counts = rng.poisson(0.3, (n_time, n_cells)).astype(np.int64)

        result = compute_per_cell_diagnostics(predictive, spike_counts, place_fields)

        # Check that NaN where no spikes
        no_spike_mask = spike_counts == 0
        assert np.all(np.isnan(result["hpd_overlap"][no_spike_mask]))
        assert np.all(np.isnan(result["kl_divergence"][no_spike_mask]))
        assert np.all(np.isnan(result["spike_prob"][no_spike_mask]))

    def test_hpd_overlap_values_in_range(self) -> None:
        """Test that HPD overlap values are in [0, 1]."""
        n_time, n_bins, n_cells = 100, 50, 10
        rng = np.random.default_rng(42)
        predictive = rng.dirichlet(np.ones(n_bins), size=n_time)
        place_fields = rng.random((n_cells, n_bins)) * 10 + 0.1
        # Ensure all cells have spikes
        spike_counts = np.ones((n_time, n_cells), dtype=np.int64)

        result = compute_per_cell_diagnostics(predictive, spike_counts, place_fields)

        valid_values = result["hpd_overlap"][~np.isnan(result["hpd_overlap"])]
        assert np.all(valid_values >= 0.0)
        assert np.all(valid_values <= 1.0)

    def test_spike_prob_values_in_range(self) -> None:
        """Test that spike probability values are in [0, 1].

        Note: spike_prob_rank returns cumulative probability mass which
        should be in [0, 1] for a proper distribution. We use a simple
        case where the place fields are well-defined.
        """
        n_time, n_bins, n_cells = 100, 50, 10
        rng = np.random.default_rng(42)

        # Create proper predictive posterior (sums to 1 over bins)
        predictive = rng.dirichlet(np.ones(n_bins), size=n_time)

        # Create place fields with proper structure
        # Each cell has a Gaussian tuning curve
        place_fields = np.zeros((n_cells, n_bins))
        centers = np.linspace(5, n_bins - 5, n_cells)
        for j, center in enumerate(centers):
            place_fields[j, :] = np.exp(-0.5 * ((np.arange(n_bins) - center) / 5) ** 2)
        place_fields = place_fields * 10 + 0.1  # Scale and add baseline

        # All cells spike
        spike_counts = np.ones((n_time, n_cells), dtype=np.int64)

        result = compute_per_cell_diagnostics(predictive, spike_counts, place_fields)

        valid_values = result["spike_prob"][~np.isnan(result["spike_prob"])]
        assert np.all(valid_values >= 0.0)
        # Allow small floating point error beyond 1.0
        assert np.all(valid_values <= 1.0 + 1e-9)

    def test_diagnostics_at_spike_times_only(self) -> None:
        """Test that diagnostics are computed only at spike times."""
        n_time, n_bins, n_cells = 20, 10, 3
        rng = np.random.default_rng(42)

        # Create simple predictive posterior
        predictive = rng.dirichlet(np.ones(n_bins), size=n_time)

        # Create place fields
        place_fields = rng.random((n_cells, n_bins)) * 10 + 0.1

        # Cell 0 spikes at times 0, 5, 10
        # Cell 1 spikes at times 2, 7
        # Cell 2 spikes at time 15
        spike_counts = np.zeros((n_time, n_cells), dtype=np.int64)
        spike_counts[[0, 5, 10], 0] = 1
        spike_counts[[2, 7], 1] = 1
        spike_counts[15, 2] = 1

        result = compute_per_cell_diagnostics(predictive, spike_counts, place_fields)

        # Check that we have valid values only at spike times
        assert not np.isnan(result["hpd_overlap"][0, 0])
        assert not np.isnan(result["hpd_overlap"][5, 0])
        assert not np.isnan(result["hpd_overlap"][10, 0])
        assert not np.isnan(result["hpd_overlap"][2, 1])
        assert not np.isnan(result["hpd_overlap"][7, 1])
        assert not np.isnan(result["hpd_overlap"][15, 2])

        # Check that non-spike times are NaN
        assert np.isnan(result["hpd_overlap"][1, 0])
        assert np.isnan(result["hpd_overlap"][0, 1])
        assert np.isnan(result["hpd_overlap"][0, 2])


class TestGetStateMarginalizedPosterior:
    """Tests for get_state_marginalized_posterior function."""

    def test_single_state_model(self) -> None:
        """Test extraction from single-state model (no state dimension)."""
        n_time, n_bins = 100, 50
        posterior_data = np.random.dirichlet(np.ones(n_bins), size=n_time)

        # Create xarray Dataset mimicking single-state model
        results = xr.Dataset(
            {
                "predictive_posterior": xr.DataArray(
                    posterior_data,
                    dims=["time", "state_bins"],
                    coords={
                        "time": np.arange(n_time),
                        "state_bins": np.arange(n_bins),
                    },
                )
            }
        )

        posterior = get_state_marginalized_posterior(results, "predictive")

        assert posterior.shape == (n_time, n_bins)
        np.testing.assert_allclose(posterior, posterior_data)

    def test_multi_state_model(self) -> None:
        """Test extraction from multi-state model (sums over states)."""
        n_time, n_bins, n_states = 100, 50, 2

        # Create posterior over (time, state, position)
        rng = np.random.default_rng(42)
        posterior_per_state = rng.dirichlet(np.ones(n_bins), size=(n_time, n_states))

        # Normalize to be proper joint distribution
        posterior_per_state = posterior_per_state / posterior_per_state.sum(
            axis=(1, 2), keepdims=True
        )

        # Create MultiIndex for state_bins (mimics non_local_detector format)
        states = ["Continuous", "Fragmented"]
        positions = np.arange(n_bins, dtype=float)

        # Create MultiIndex directly
        import pandas as pd

        multi_index = pd.MultiIndex.from_product([states, positions], names=["state", "position"])

        # Flatten the posterior for xarray
        posterior_flat = posterior_per_state.reshape(n_time, -1)

        # Create Dataset with state_bins as MultiIndex
        results = xr.Dataset(
            {
                "predictive_posterior": xr.DataArray(
                    posterior_flat,
                    dims=["time", "state_bins"],
                    coords={
                        "time": np.arange(n_time),
                        "state_bins": multi_index,
                    },
                )
            }
        )

        posterior = get_state_marginalized_posterior(results, "predictive")

        # Check shape
        assert posterior.shape == (n_time, n_bins)

        # Check that it's the sum over states
        expected = posterior_per_state.sum(axis=1)
        np.testing.assert_allclose(posterior, expected, rtol=1e-5)

    def test_acausal_posterior_type(self) -> None:
        """Test that acausal posterior_type works."""
        n_time, n_bins = 50, 30
        posterior_data = np.random.dirichlet(np.ones(n_bins), size=n_time)

        results = xr.Dataset(
            {
                "acausal_posterior": xr.DataArray(
                    posterior_data,
                    dims=["time", "state_bins"],
                    coords={
                        "time": np.arange(n_time),
                        "state_bins": np.arange(n_bins),
                    },
                )
            }
        )

        posterior = get_state_marginalized_posterior(results, "acausal")

        assert posterior.shape == (n_time, n_bins)
        np.testing.assert_allclose(posterior, posterior_data)
