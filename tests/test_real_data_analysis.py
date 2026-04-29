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

    def test_duplicate_spikes_in_same_bin_are_separate_events(self) -> None:
        """Test that same-bin spikes remain separate event-level diagnostics."""
        n_time, n_bins, n_cells = 4, 8, 1
        rng = np.random.default_rng(42)
        predictive = rng.dirichlet(np.ones(n_bins), size=n_time)
        place_fields = rng.random((n_cells, n_bins)) + 0.1
        time = np.arange(n_time, dtype=np.float64)
        spike_counts = np.zeros((n_time, n_cells), dtype=np.int64)
        spike_counts[1, 0] = 2
        spike_times = [np.array([1.10, 1.20])]

        result = compute_per_cell_diagnostics(
            predictive,
            spike_counts,
            place_fields,
            spike_times=spike_times,
            time=time,
        )

        np.testing.assert_allclose(result["event_time"], [1.10, 1.20])
        np.testing.assert_array_equal(result["event_time_ind"], [1, 1])
        np.testing.assert_array_equal(result["event_cell_ind"], [0, 0])
        assert result["event_hpd_overlap"].shape == (2,)
        assert result["event_kl_divergence"].shape == (2,)
        assert result["event_spike_prob"].shape == (2,)
        np.testing.assert_allclose(
            result["event_kl_divergence"],
            np.repeat(result["kl_divergence"][1, 0], 2),
        )


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


class TestPlotPerCellDiagnosticScatterWithSpikeTimes:
    """Tests for plot_per_cell_diagnostic_scatter with spike_times alignment."""

    def test_spike_times_aligns_with_raster(self) -> None:
        """Test that spike_times parameter aligns scatter points with actual spike times."""
        import matplotlib.pyplot as plt

        from statespacecheck_paper.real_data_plotting import (
            plot_per_cell_diagnostic_scatter,
        )

        # Create time bins: 0.0, 0.1, 0.2, ..., 0.9 (10 bins, 100ms each)
        time = np.linspace(0.0, 0.9, 10)
        n_time, n_cells = 10, 3

        # Create spike times at non-bin-center positions
        # Cell 0: spike at 0.15 (falls in bin 1, which starts at 0.1)
        # Cell 1: spike at 0.35 (falls in bin 3, which starts at 0.3)
        # Cell 2: spike at 0.55 (falls in bin 5, which starts at 0.5)
        spike_times_list = [
            np.array([0.15]),  # Cell 0
            np.array([0.35]),  # Cell 1
            np.array([0.55]),  # Cell 2
        ]

        # Create diagnostics with NaN everywhere except at spike times
        diagnostics = {"hpd_overlap": np.full((n_time, n_cells), np.nan)}
        diagnostics["hpd_overlap"][1, 0] = 0.8  # Bin 1, cell 0 (spike at 0.15)
        diagnostics["hpd_overlap"][3, 1] = 0.6  # Bin 3, cell 1 (spike at 0.35)
        diagnostics["hpd_overlap"][5, 2] = 0.4  # Bin 5, cell 2 (spike at 0.55)

        fig, ax = plt.subplots()

        # Plot with spike_times
        plot_per_cell_diagnostic_scatter(
            time,
            diagnostics,
            ax=ax,
            spike_times=spike_times_list,
        )

        # Get the scatter collection
        scatter = ax.collections[0]
        offsets = scatter.get_offsets()

        # Filter out masked values (NaN positions)
        valid_mask = ~np.ma.getmask(offsets).any(axis=1)
        valid_offsets = offsets[valid_mask]

        # Should have 3 points at x = 0.15, 0.35, 0.55 (actual spike times)
        assert len(valid_offsets) == 3
        expected_x = np.array([0.15, 0.35, 0.55])
        np.testing.assert_allclose(sorted(valid_offsets[:, 0]), sorted(expected_x))

        plt.close(fig)

    def test_event_diagnostics_plot_at_exact_spike_times(self) -> None:
        """Test event diagnostics are plotted directly without bin remapping."""
        import matplotlib.pyplot as plt

        from statespacecheck_paper.real_data_plotting import (
            plot_per_cell_diagnostic_scatter,
        )

        time = np.linspace(0.0, 0.9, 10)
        diagnostics = {
            "hpd_overlap": np.full((10, 1), np.nan),
            "event_time": np.array([0.151, 0.157]),
            "event_hpd_overlap": np.array([0.8, 0.6]),
        }
        diagnostics["hpd_overlap"][1, 0] = 0.7

        fig, ax = plt.subplots()
        plot_per_cell_diagnostic_scatter(time, diagnostics, ax=ax)

        offsets = ax.collections[0].get_offsets()
        valid_mask = ~np.ma.getmask(offsets).any(axis=1)
        valid_offsets = offsets[valid_mask]

        assert len(valid_offsets) == 2
        np.testing.assert_allclose(valid_offsets[:, 0], [0.151, 0.157])
        np.testing.assert_allclose(valid_offsets[:, 1], [0.8, 0.6])

        plt.close(fig)

    def test_without_spike_times_uses_bin_values(self) -> None:
        """Test that without spike_times, scatter uses bin values."""
        import matplotlib.pyplot as plt

        from statespacecheck_paper.real_data_plotting import (
            plot_per_cell_diagnostic_scatter,
        )

        time = np.linspace(0.0, 0.9, 10)
        n_time, n_cells = 10, 2

        # Create simple diagnostics
        diagnostics = {"hpd_overlap": np.full((n_time, n_cells), np.nan)}
        diagnostics["hpd_overlap"][1, 0] = 0.8  # Bin 1
        diagnostics["hpd_overlap"][3, 1] = 0.6  # Bin 3

        fig, ax = plt.subplots()

        # Plot WITHOUT spike_times (original behavior)
        plot_per_cell_diagnostic_scatter(
            time,
            diagnostics,
            ax=ax,
            spike_times=None,
        )

        scatter = ax.collections[0]
        offsets = scatter.get_offsets()
        valid_mask = ~np.ma.getmask(offsets).any(axis=1)
        valid_offsets = offsets[valid_mask]

        # Should have points at x = 0.1, 0.3 (bin values, not spike times)
        assert len(valid_offsets) == 2
        expected_x = np.array([0.1, 0.3])
        np.testing.assert_allclose(sorted(valid_offsets[:, 0]), sorted(expected_x))

        plt.close(fig)


class TestComputeRunningAverage:
    """Tests for compute_running_average function."""

    def test_output_shape_matches_input(self) -> None:
        """Test that output has same shape as input time dimension."""
        from statespacecheck_paper.real_data_analysis import compute_running_average

        n_time, n_cells = 100, 10
        metric = np.random.rand(n_time, n_cells)
        time = np.linspace(0, 1, n_time)

        running_avg, time_out = compute_running_average(metric, time, window_size=0.1)

        assert running_avg.shape == (n_time,)
        assert time_out.shape == (n_time,)
        np.testing.assert_array_equal(time_out, time)

    def test_handles_nan_values(self) -> None:
        """Test that NaN values are handled correctly."""
        from statespacecheck_paper.real_data_analysis import compute_running_average

        n_time, n_cells = 100, 10
        metric = np.random.rand(n_time, n_cells)
        # Add NaN values (simulating sparse spikes)
        metric[::2, :] = np.nan
        time = np.linspace(0, 1, n_time)

        running_avg, _ = compute_running_average(metric, time, window_size=0.1)

        # Should not have NaN in output (interpolated)
        assert not np.any(np.isnan(running_avg))

    def test_all_nan_returns_nan_array(self) -> None:
        """Test that all-NaN input returns NaN array."""
        from statespacecheck_paper.real_data_analysis import compute_running_average

        n_time, n_cells = 100, 10
        metric = np.full((n_time, n_cells), np.nan)
        time = np.linspace(0, 1, n_time)

        running_avg, _ = compute_running_average(metric, time, window_size=0.1)

        assert np.all(np.isnan(running_avg))

    def test_smoothing_effect(self) -> None:
        """Test that larger window produces smoother output."""
        from statespacecheck_paper.real_data_analysis import compute_running_average

        n_time, n_cells = 1000, 10
        # Create noisy data
        rng = np.random.default_rng(42)
        metric = rng.random((n_time, n_cells))
        time = np.linspace(0, 1, n_time)

        # Compute with small and large windows
        small_window, _ = compute_running_average(metric, time, window_size=0.01)
        large_window, _ = compute_running_average(metric, time, window_size=0.1)

        # Larger window should have smaller variance (smoother)
        assert np.var(large_window) < np.var(small_window)

    def test_window_size_affects_output(self) -> None:
        """Test that different window sizes produce different outputs."""
        from statespacecheck_paper.real_data_analysis import compute_running_average

        n_time, n_cells = 100, 10
        rng = np.random.default_rng(42)
        metric = rng.random((n_time, n_cells))
        time = np.linspace(0, 1, n_time)

        result1, _ = compute_running_average(metric, time, window_size=0.05)
        result2, _ = compute_running_average(metric, time, window_size=0.2)

        # Results should be different
        assert not np.allclose(result1, result2)

    def test_event_inputs_count_duplicate_spikes(self) -> None:
        """Test event running average counts duplicate events at same time."""
        from statespacecheck_paper.real_data_analysis import compute_running_average

        metric = np.full((3, 1), np.nan)
        time = np.array([0.0, 1.0, 2.0])
        event_times = np.array([1.0, 1.0])
        event_values = np.array([1.0, 3.0])

        running_avg, _ = compute_running_average(
            metric,
            time,
            window_size=0.1,
            event_times=event_times,
            event_values=event_values,
        )

        assert np.isnan(running_avg[0])
        assert running_avg[1] == 2.0
        assert np.isnan(running_avg[2])


class TestPlotPerCellDiagnosticScatterWithRunningAverage:
    """Tests for plot_per_cell_diagnostic_scatter with running average."""

    def test_running_average_adds_line(self) -> None:
        """Test that show_running_average=True adds a line to the plot."""
        import matplotlib.pyplot as plt

        from statespacecheck_paper.real_data_plotting import (
            plot_per_cell_diagnostic_scatter,
        )

        n_time, n_cells = 100, 10
        time = np.linspace(0.0, 1.0, n_time)
        diagnostics = {"hpd_overlap": np.random.rand(n_time, n_cells)}

        fig, ax = plt.subplots()

        # Plot WITHOUT running average
        plot_per_cell_diagnostic_scatter(
            time,
            diagnostics,
            ax=ax,
            show_running_average=False,
        )
        n_lines_without = len(ax.lines)

        plt.close(fig)

        # Plot WITH running average
        fig, ax = plt.subplots()
        plot_per_cell_diagnostic_scatter(
            time,
            diagnostics,
            ax=ax,
            show_running_average=True,
        )
        n_lines_with = len(ax.lines)

        # Should have one more line
        assert n_lines_with == n_lines_without + 1

        plt.close(fig)

    def test_running_average_custom_window(self) -> None:
        """Test that custom window size is used."""
        import matplotlib.pyplot as plt

        from statespacecheck_paper.real_data_plotting import (
            plot_per_cell_diagnostic_scatter,
        )

        n_time, n_cells = 100, 10
        time = np.linspace(0.0, 1.0, n_time)
        rng = np.random.default_rng(42)
        diagnostics = {"hpd_overlap": rng.random((n_time, n_cells))}

        # Plot with different window sizes
        fig1, ax1 = plt.subplots()
        plot_per_cell_diagnostic_scatter(
            time,
            diagnostics,
            ax=ax1,
            show_running_average=True,
            running_average_window=0.05,
        )
        line1_y = ax1.lines[0].get_ydata()

        fig2, ax2 = plt.subplots()
        plot_per_cell_diagnostic_scatter(
            time,
            diagnostics,
            ax=ax2,
            show_running_average=True,
            running_average_window=0.2,
        )
        line2_y = ax2.lines[0].get_ydata()

        # Different window sizes should produce different lines
        assert not np.allclose(line1_y, line2_y)

        plt.close(fig1)
        plt.close(fig2)

    def test_spike_prob_running_average_uses_raw_values(self) -> None:
        """Test that running average for spike_prob uses raw probabilities.

        The running average should be computed on raw probability values,
        then transformed to -log10 scale. This differs from computing
        the mean of transformed values:
            -log10(mean(p_i)) != mean(-log10(p_i))

        For example, with probabilities [0.01, 0.99]:
            mean(raw) = 0.5, then -log10(0.5) = 0.301
            mean(-log10([0.01, 0.99])) = mean([2.0, 0.004]) = 1.002
        These are very different!
        """
        import matplotlib.pyplot as plt

        from statespacecheck_paper.real_data_plotting import (
            plot_per_cell_diagnostic_scatter,
        )

        # Create spike_prob values with different values across cells
        # This ensures mean(raw) differs from mean(transformed)
        n_time = 3
        spike_probs = np.array(
            [
                [0.01, 0.99],  # mean_raw=0.5, -log10(0.5)=0.301; mean_transformed=1.002
                [0.1, 0.9],  # mean_raw=0.5, -log10(0.5)=0.301; mean_transformed=0.523
                [0.5, 0.5],  # mean_raw=0.5, same either way (control case)
            ]
        )
        time = np.linspace(0, 0.2, n_time)
        diagnostics = {"spike_prob": spike_probs}

        fig, ax = plt.subplots()
        plot_per_cell_diagnostic_scatter(
            time,
            diagnostics,
            ax=ax,
            metric_name="spike_prob",
            show_running_average=True,
            running_average_window=0.01,  # Small window = minimal smoothing
        )

        # Extract running average line
        line = ax.lines[0]
        y_actual = line.get_ydata()

        # Expected: avg(raw) then transform to -log10
        mean_raw = np.mean(spike_probs, axis=1)  # [0.5, 0.5, 0.5]
        expected = -np.log10(np.maximum(mean_raw, 1e-10))  # [0.301, 0.301, 0.301]

        # Should match expected (avg then transform)
        np.testing.assert_allclose(y_actual, expected, rtol=1e-3)

        # Verify this differs from incorrect approach (transform then avg)
        transformed = -np.log10(np.maximum(spike_probs, 1e-10))
        incorrect = np.mean(transformed, axis=1)  # [1.002, 0.523, 0.301]

        # The incorrect approach gives different values for rows 0 and 1
        assert not np.allclose(y_actual, incorrect, rtol=1e-3)

        plt.close(fig)
