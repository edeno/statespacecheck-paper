"""Tests for plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from statespacecheck_paper.analysis import DecodeParams, Thresholds, Transformed
from statespacecheck_paper.plotting import (
    compute_hpd_region,
    plot_combined_diagnostics,
    plot_misfit_examples,
    plot_original,
    plot_transformed,
)


class TestComputeHpdRegion:
    """Tests for compute_hpd_region function."""

    def test_output_shape(self) -> None:
        """Test that output has correct shape."""
        x = np.linspace(-5, 5, 100)
        pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)  # Gaussian
        mask = compute_hpd_region(x, pdf, coverage=0.95)
        assert mask.shape == x.shape
        assert mask.dtype == bool

    def test_coverage_approximately_correct(self) -> None:
        """Test that HPD region achieves approximate coverage."""
        x = np.linspace(-5, 5, 1000)
        pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        mask = compute_hpd_region(x, pdf, coverage=0.95)

        # Compute actual coverage
        dx = x[1] - x[0]
        pdf_normalized = pdf / (np.sum(pdf) * dx)
        actual_coverage = np.sum(pdf_normalized[mask]) * dx

        # Should be approximately 0.95 (within 5% tolerance)
        assert 0.90 <= actual_coverage <= 1.0

    def test_hpd_is_contiguous_for_unimodal(self) -> None:
        """Test that HPD region is contiguous for unimodal distribution."""
        x = np.linspace(-10, 10, 500)
        pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        mask = compute_hpd_region(x, pdf, coverage=0.95)

        # Find indices where mask is True
        true_indices = np.where(mask)[0]
        if len(true_indices) > 1:
            # Check that indices are contiguous
            expected_indices = np.arange(true_indices[0], true_indices[-1] + 1)
            # Allow for small gaps due to discretization
            contiguous_ratio = len(true_indices) / len(expected_indices)
            assert contiguous_ratio > 0.90

    def test_different_coverages(self) -> None:
        """Test with different coverage levels."""
        x = np.linspace(-5, 5, 500)
        pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

        mask_50 = compute_hpd_region(x, pdf, coverage=0.50)
        mask_95 = compute_hpd_region(x, pdf, coverage=0.95)

        # Higher coverage should include more points
        assert np.sum(mask_95) > np.sum(mask_50)

    def test_uniform_distribution(self) -> None:
        """Test with uniform distribution."""
        x = np.linspace(0, 10, 100)
        pdf = np.ones_like(x)
        mask = compute_hpd_region(x, pdf, coverage=0.95)

        # For uniform, approximately all points should be included
        assert np.sum(mask) / len(mask) > 0.90


class TestPlotOriginal:
    """Tests for plot_original function."""

    def test_creates_figure(self) -> None:
        """Test that plot_original creates a Figure object."""
        # Setup synthetic data
        n_time, n_bins = 100, 50
        xs = np.linspace(0, 1, n_bins)
        x_true = np.random.uniform(0, n_bins - 1, n_time)

        metrics = {
            "post": np.random.dirichlet(np.ones(n_bins), size=n_time),
            "HPDO": np.random.uniform(0, 1, n_time),
            "KL": np.random.uniform(0, 5, n_time),
            "spikeProb": np.random.uniform(0, 1, (n_time, 10)),
        }

        th = Thresholds(HPDO=0.8, KL=2.0, spike_prob=0.05)

        # Execute
        fig = plot_original(xs, x_true, metrics, th)

        # Assert
        assert isinstance(fig, plt.Figure)
        # Should have 4 subplots (posterior, HPDO, KL, spike prob)
        assert len(fig.axes) >= 4

        # Cleanup
        plt.close(fig)

    def test_with_phase_boundaries(self) -> None:
        """Test plot_original with phase boundaries."""
        n_time, n_bins = 100, 50
        xs = np.linspace(0, 1, n_bins)
        x_true = np.random.uniform(0, n_bins - 1, n_time)

        metrics = {
            "post": np.random.dirichlet(np.ones(n_bins), size=n_time),
            "HPDO": np.random.uniform(0, 1, n_time),
            "KL": np.random.uniform(0, 5, n_time),
            "spikeProb": np.random.uniform(0, 1, (n_time, 10)),
        }

        th = Thresholds(HPDO=0.8, KL=2.0, spike_prob=0.05)
        phase_boundaries = (10, 20, 30, 40, 50, 60, 70, 80)

        fig = plot_original(xs, x_true, metrics, th, phase_boundaries=phase_boundaries)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_title(self) -> None:
        """Test plot_original with custom title."""
        n_time, n_bins = 50, 30
        xs = np.linspace(0, 1, n_bins)
        x_true = np.random.uniform(0, n_bins - 1, n_time)

        metrics = {
            "post": np.random.dirichlet(np.ones(n_bins), size=n_time),
            "HPDO": np.random.uniform(0, 1, n_time),
            "KL": np.random.uniform(0, 5, n_time),
            "spikeProb": np.random.uniform(0, 1, (n_time, 5)),
        }

        th = Thresholds(HPDO=0.7, KL=1.5, spike_prob=0.1)

        fig = plot_original(xs, x_true, metrics, th, title="Test Figure")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTransformed:
    """Tests for plot_transformed function."""

    def test_creates_figure(self) -> None:
        """Test that plot_transformed creates a Figure object."""
        n_time, n_bins = 100, 50
        xs = np.linspace(0, 1, n_bins)
        x_true = np.random.uniform(0, n_bins - 1, n_time)
        post = np.random.dirichlet(np.ones(n_bins), size=n_time)

        tr = Transformed(
            HPDO=np.random.uniform(0, 5, n_time),
            KL=np.random.uniform(0, 3, n_time),
            spike_prob=np.random.uniform(0, 10, n_time),
            HPDO_th=3.0,
            KL_th=2.0,
            spike_prob_th=5.0,
        )

        fig = plot_transformed(xs, x_true, post, tr)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 4
        plt.close(fig)

    def test_with_remap_window(self) -> None:
        """Test plot_transformed with remap window."""
        n_time, n_bins = 80, 40
        xs = np.linspace(0, 1, n_bins)
        x_true = np.random.uniform(0, n_bins - 1, n_time)
        post = np.random.dirichlet(np.ones(n_bins), size=n_time)

        tr = Transformed(
            HPDO=np.random.uniform(0, 5, n_time),
            KL=np.random.uniform(0, 3, n_time),
            spike_prob=np.random.uniform(0, 10, n_time),
            HPDO_th=3.0,
            KL_th=2.0,
            spike_prob_th=5.0,
        )

        fig = plot_transformed(xs, x_true, post, tr, remap_window=(20, 40))

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotMisfitExamples:
    """Tests for plot_misfit_examples function."""

    def test_creates_figure(self) -> None:
        """Test that plot_misfit_examples runs without errors."""
        # Setup synthetic data - need enough time points for baseline (starts at 1000)
        n_time, n_bins, n_cells = 6000, 50, 10
        xs = np.linspace(0, 1, n_bins)
        x_true = np.random.uniform(0, n_bins - 1, n_time)
        spikes = np.random.poisson(0.5, (n_time, n_cells))

        metrics = {
            "post": np.random.dirichlet(np.ones(n_bins), size=n_time),
            "HPDO": np.random.uniform(0, 1, n_time),
            "KL": np.random.uniform(0, 5, n_time),
            "spikeProb": np.random.uniform(0, 1, (n_time, n_cells)),
        }

        # Create DecodeParams with timeline structure
        # Must have T_remap_start > 2000 for baseline window slice(1000, T_remap_start-1000) to be valid
        # Override remap_from_to to match number of cells (10 cells = indices 0-9)
        params = DecodeParams(
            T_remap_start=3000,
            T_remap_end=3500,
            T_recovery1_end=4000,
            T_flat_end=4500,
            T_recovery2_end=5000,
            T_fast_end=5500,
            T_recovery3_end=5700,
            T_slow_end=5900,
            remap_from_to=(
                (0, 5),
                (1, 6),
                (2, 7),
                (3, 8),
                (4, 9),
                (5, 0),
                (6, 1),
                (7, 2),
                (8, 3),
                (9, 4),
            ),
        )

        pf_centers = np.linspace(0, 1, n_cells)
        pf_width = 0.1
        rate_scale = 10.0

        # Execute
        fig = plot_misfit_examples(
            xs, x_true, spikes, metrics, params, pf_centers, pf_width, rate_scale
        )

        # Assert
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_different_params(self) -> None:
        """Test plot_misfit_examples with different parameters."""
        n_time, n_bins, n_cells = 3500, 30, 5
        xs = np.linspace(0, 1, n_bins)
        x_true = np.random.uniform(0, n_bins - 1, n_time)
        spikes = np.random.poisson(1.0, (n_time, n_cells))

        metrics = {
            "post": np.random.dirichlet(np.ones(n_bins), size=n_time),
            "HPDO": np.random.uniform(0, 1, n_time),
            "KL": np.random.uniform(0, 5, n_time),
            "spikeProb": np.random.uniform(0, 1, (n_time, n_cells)),
        }

        params = DecodeParams(
            T_remap_start=2100,
            T_remap_end=2400,
            T_recovery1_end=2600,
            T_flat_end=2900,
            T_recovery2_end=3100,
            T_fast_end=3300,
            T_recovery3_end=3400,
            T_slow_end=3450,
            sigx_pred=0.05,
            remap_from_to=((0, 2), (1, 3), (2, 4), (3, 0), (4, 1)),  # 5 cells
        )

        pf_centers = np.linspace(0, 1, n_cells)
        pf_width = 0.15
        rate_scale = 5.0

        fig = plot_misfit_examples(
            xs, x_true, spikes, metrics, params, pf_centers, pf_width, rate_scale
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotCombinedDiagnostics:
    """Tests for plot_combined_diagnostics function."""

    def test_creates_figure(self) -> None:
        """Test that plot_combined_diagnostics runs without errors."""
        # Setup synthetic data - need enough time points for baseline (starts at 1000)
        n_time, n_bins, n_cells = 6000, 50, 10
        xs = np.linspace(0, 1, n_bins)
        x_true = np.random.uniform(0, n_bins - 1, n_time)
        spikes = np.random.poisson(0.5, (n_time, n_cells))

        metrics = {
            "post": np.random.dirichlet(np.ones(n_bins), size=n_time),
            "HPDO": np.random.uniform(0, 1, n_time),
            "KL": np.random.uniform(0, 5, n_time),
            "spikeProb": np.random.uniform(0, 1, (n_time, n_cells)),
        }

        th = Thresholds(HPDO=0.8, KL=2.0, spike_prob=0.05)

        params = DecodeParams(
            T_remap_start=3000,
            T_remap_end=3500,
            T_recovery1_end=4000,
            T_flat_end=4500,
            T_recovery2_end=5000,
            T_fast_end=5500,
            T_recovery3_end=5700,
            T_slow_end=5900,
            remap_from_to=(
                (0, 5),
                (1, 6),
                (2, 7),
                (3, 8),
                (4, 9),
                (5, 0),
                (6, 1),
                (7, 2),
                (8, 3),
                (9, 4),
            ),
        )

        pf_centers = np.linspace(0, 1, n_cells)
        pf_width = 0.1
        rate_scale = 10.0

        # Execute
        fig = plot_combined_diagnostics(
            xs, x_true, spikes, metrics, th, params, pf_centers, pf_width, rate_scale
        )

        # Assert
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_small_dataset(self) -> None:
        """Test plot_combined_diagnostics with smaller dataset."""
        n_time, n_bins, n_cells = 3500, 30, 5
        xs = np.linspace(0, 1, n_bins)
        x_true = np.random.uniform(0, n_bins - 1, n_time)
        spikes = np.random.poisson(1.0, (n_time, n_cells))

        metrics = {
            "post": np.random.dirichlet(np.ones(n_bins), size=n_time),
            "HPDO": np.random.uniform(0, 1, n_time),
            "KL": np.random.uniform(0, 5, n_time),
            "spikeProb": np.random.uniform(0, 1, (n_time, n_cells)),
        }

        th = Thresholds(HPDO=0.7, KL=1.5, spike_prob=0.1)

        params = DecodeParams(
            T_remap_start=2100,
            T_remap_end=2400,
            T_recovery1_end=2600,
            T_flat_end=2900,
            T_recovery2_end=3100,
            T_fast_end=3300,
            T_recovery3_end=3400,
            T_slow_end=3450,
            remap_from_to=((0, 2), (1, 3), (2, 4), (3, 0), (4, 1)),  # 5 cells
        )

        pf_centers = np.linspace(0, 1, n_cells)
        pf_width = 0.15
        rate_scale = 5.0

        fig = plot_combined_diagnostics(
            xs, x_true, spikes, metrics, th, params, pf_centers, pf_width, rate_scale
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
