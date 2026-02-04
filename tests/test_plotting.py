"""Tests for plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from statespacecheck_paper.analysis import DecodeParams, Thresholds, Transformed
from statespacecheck_paper.plotting import (
    compute_hpd_region,
    create_distribution_comparison_panel,
    extract_contiguous_regions,
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

    def test_very_high_coverage(self) -> None:
        """Test edge case with very high coverage (line 72)."""
        x = np.linspace(-5, 5, 50)
        pdf = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        # Request coverage close to 1.0 to trigger threshold_idx >= len check
        mask = compute_hpd_region(x, pdf, coverage=0.999)

        # Should still return a valid mask
        assert mask.shape == x.shape
        assert mask.dtype == bool
        # Most points should be included (allow some margin for discrete approximation)
        assert np.sum(mask) / len(mask) > 0.60


class TestPlotOriginal:
    """Tests for plot_original function."""

    def test_creates_figure(self) -> None:
        """Test that plot_original creates a Figure object."""
        # Setup synthetic data - metrics now 2D: (n_time, n_cells)
        rng = np.random.default_rng(42)
        n_time, n_bins, n_cells = 100, 50, 5
        xs = np.linspace(0, 1, n_bins)
        x_true = rng.uniform(0, n_bins - 1, n_time)

        metrics = {
            "posterior": rng.dirichlet(np.ones(n_bins), size=n_time),
            "hpd_overlap": rng.uniform(0, 1, (n_time, n_cells)),
            "kl_divergence": rng.uniform(0, 5, (n_time, n_cells)),
            "spike_prob": rng.uniform(0, 1, (n_time, n_cells)),
        }

        thresholds = Thresholds(
            hpd_overlap=0.8,
            kl_divergence=2.0,
            spike_prob=0.05,
        )

        # Execute
        fig = plot_original(xs, x_true, metrics, thresholds)

        # Assert
        assert isinstance(fig, plt.Figure)
        # Should have 4 subplots (posterior, HPDO, KL, spike_prob)
        assert len(fig.axes) >= 4

        # Cleanup
        plt.close(fig)

    def test_with_phase_boundaries(self) -> None:
        """Test plot_original with phase boundaries."""
        rng = np.random.default_rng(42)
        n_time, n_bins, n_cells = 100, 50, 5
        xs = np.linspace(0, 1, n_bins)
        x_true = rng.uniform(0, n_bins - 1, n_time)

        metrics = {
            "posterior": rng.dirichlet(np.ones(n_bins), size=n_time),
            "hpd_overlap": rng.uniform(0, 1, (n_time, n_cells)),
            "kl_divergence": rng.uniform(0, 5, (n_time, n_cells)),
            "spike_prob": rng.uniform(0, 1, (n_time, n_cells)),
        }

        thresholds = Thresholds(
            hpd_overlap=0.8,
            kl_divergence=2.0,
            spike_prob=0.05,
        )
        phase_boundaries = (10, 20, 30, 40, 50, 60, 70, 80)

        fig = plot_original(xs, x_true, metrics, thresholds, phase_boundaries=phase_boundaries)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_custom_title(self) -> None:
        """Test plot_original with custom title."""
        rng = np.random.default_rng(42)
        n_time, n_bins, n_cells = 50, 30, 5
        xs = np.linspace(0, 1, n_bins)
        x_true = rng.uniform(0, n_bins - 1, n_time)

        metrics = {
            "posterior": rng.dirichlet(np.ones(n_bins), size=n_time),
            "hpd_overlap": rng.uniform(0, 1, (n_time, n_cells)),
            "kl_divergence": rng.uniform(0, 5, (n_time, n_cells)),
            "spike_prob": rng.uniform(0, 1, (n_time, n_cells)),
        }

        thresholds = Thresholds(
            hpd_overlap=0.7,
            kl_divergence=1.5,
            spike_prob=0.05,
        )

        fig = plot_original(xs, x_true, metrics, thresholds, title="Test Figure")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTransformed:
    """Tests for plot_transformed function."""

    def test_creates_figure(self) -> None:
        """Test that plot_transformed creates a Figure object."""
        rng = np.random.default_rng(42)
        n_time, n_bins, n_cells = 100, 50, 5
        xs = np.linspace(0, 1, n_bins)
        x_true = rng.uniform(0, n_bins - 1, n_time)
        post = rng.dirichlet(np.ones(n_bins), size=n_time)

        # Transformed metrics are now 2D: (n_time, n_cells)
        transformed = Transformed(
            hpd_overlap=rng.uniform(0, 5, (n_time, n_cells)),
            kl_divergence=rng.uniform(0, 3, (n_time, n_cells)),
            spike_prob=rng.uniform(0, 10, (n_time, n_cells)),
            hpd_overlap_threshold=3.0,
            kl_divergence_threshold=2.0,
            spike_prob_threshold=5.0,
        )

        fig = plot_transformed(xs, x_true, post, transformed)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 4
        plt.close(fig)

    def test_with_remap_window(self) -> None:
        """Test plot_transformed with remap window."""
        rng = np.random.default_rng(42)
        n_time, n_bins, n_cells = 80, 40, 5
        xs = np.linspace(0, 1, n_bins)
        x_true = rng.uniform(0, n_bins - 1, n_time)
        post = rng.dirichlet(np.ones(n_bins), size=n_time)

        # Transformed metrics are now 2D: (n_time, n_cells)
        transformed = Transformed(
            hpd_overlap=rng.uniform(0, 5, (n_time, n_cells)),
            kl_divergence=rng.uniform(0, 3, (n_time, n_cells)),
            spike_prob=rng.uniform(0, 10, (n_time, n_cells)),
            hpd_overlap_threshold=3.0,
            kl_divergence_threshold=2.0,
            spike_prob_threshold=5.0,
        )

        fig = plot_transformed(xs, x_true, post, transformed, remap_window=(20, 40))

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_phase_boundaries(self) -> None:
        """Test plot_transformed with phase boundaries (lines 360-362)."""
        rng = np.random.default_rng(42)
        n_time, n_bins, n_cells = 100, 50, 5
        xs = np.linspace(0, 1, n_bins)
        x_true = rng.uniform(0, n_bins - 1, n_time)
        post = rng.dirichlet(np.ones(n_bins), size=n_time)

        # Transformed metrics are now 2D: (n_time, n_cells)
        transformed = Transformed(
            hpd_overlap=rng.uniform(0, 5, (n_time, n_cells)),
            kl_divergence=rng.uniform(0, 3, (n_time, n_cells)),
            spike_prob=rng.uniform(0, 10, (n_time, n_cells)),
            hpd_overlap_threshold=3.0,
            kl_divergence_threshold=2.0,
            spike_prob_threshold=5.0,
        )

        # Provide phase_boundaries to test lines 360-362
        fig = plot_transformed(xs, x_true, post, transformed, phase_boundaries=(30, 70))

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotMisfitExamples:
    """Tests for plot_misfit_examples function."""

    def test_creates_figure(self) -> None:
        """Test that plot_misfit_examples runs without errors."""
        # Setup synthetic data - need enough time points for baseline (starts at 1000)
        # Metrics are now 2D: (n_time, n_cells)
        rng = np.random.default_rng(42)
        n_time, n_bins, n_cells = 6000, 50, 10
        xs = np.linspace(0, 1, n_bins)
        x_true = rng.uniform(0, n_bins - 1, n_time)
        spikes = rng.poisson(0.5, (n_time, n_cells))

        metrics = {
            "posterior": rng.dirichlet(np.ones(n_bins), size=n_time),
            "hpd_overlap": rng.uniform(0, 1, (n_time, n_cells)),
            "kl_divergence": rng.uniform(0, 5, (n_time, n_cells)),
            "spike_prob": rng.uniform(0, 1, (n_time, n_cells)),
        }

        # Create DecodeParams with timeline structure
        # Must have T_remap_start > 2000 for baseline window
        # slice(1000, T_remap_start-1000) to be valid
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

        placefield_centers = np.linspace(0, 1, n_cells)
        placefield_width = 0.1
        rate_scale = 10.0

        # Execute
        fig = plot_misfit_examples(
            xs, x_true, spikes, metrics, params, placefield_centers, placefield_width, rate_scale
        )

        # Assert
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_different_params(self) -> None:
        """Test plot_misfit_examples with different parameters."""
        rng = np.random.default_rng(42)
        n_time, n_bins, n_cells = 3500, 30, 5
        xs = np.linspace(0, 1, n_bins)
        x_true = rng.uniform(0, n_bins - 1, n_time)
        spikes = rng.poisson(1.0, (n_time, n_cells))

        # Metrics are now 2D: (n_time, n_cells)
        metrics = {
            "posterior": rng.dirichlet(np.ones(n_bins), size=n_time),
            "hpd_overlap": rng.uniform(0, 1, (n_time, n_cells)),
            "kl_divergence": rng.uniform(0, 5, (n_time, n_cells)),
            "spike_prob": rng.uniform(0, 1, (n_time, n_cells)),
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

        placefield_centers = np.linspace(0, 1, n_cells)
        placefield_width = 0.15
        rate_scale = 5.0

        fig = plot_misfit_examples(
            xs, x_true, spikes, metrics, params, placefield_centers, placefield_width, rate_scale
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotCombinedDiagnostics:
    """Tests for plot_combined_diagnostics function."""

    def test_creates_figure(self) -> None:
        """Test that plot_combined_diagnostics runs without errors."""
        # Setup synthetic data - need enough time points for baseline (starts at 1000)
        # Metrics are now 2D: (n_time, n_cells)
        rng = np.random.default_rng(42)
        n_time, n_bins, n_cells = 6000, 50, 10
        xs = np.linspace(0, 1, n_bins)
        x_true = rng.uniform(0, n_bins - 1, n_time)
        spikes = rng.poisson(0.5, (n_time, n_cells))

        metrics = {
            "predictive": rng.dirichlet(np.ones(n_bins), size=n_time),
            "likelihood": rng.dirichlet(np.ones(n_bins), size=n_time),
            "posterior": rng.dirichlet(np.ones(n_bins), size=n_time),
            "hpd_overlap": rng.uniform(0, 1, (n_time, n_cells)),
            "kl_divergence": rng.uniform(0, 5, (n_time, n_cells)),
            "spike_prob": rng.uniform(0, 1, (n_time, n_cells)),
        }

        thresholds = Thresholds(
            hpd_overlap=0.8,
            kl_divergence=2.0,
            spike_prob=0.05,
        )

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

        placefield_centers = np.linspace(0, 1, n_cells)
        placefield_width = 0.1
        rate_scale = 10.0

        # Execute
        fig = plot_combined_diagnostics(
            xs,
            x_true,
            spikes,
            metrics,
            thresholds,
            params,
            placefield_centers,
            placefield_width,
            rate_scale,
        )

        # Assert
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_small_dataset(self) -> None:
        """Test plot_combined_diagnostics with smaller dataset."""
        rng = np.random.default_rng(42)
        n_time, n_bins, n_cells = 3500, 30, 5
        xs = np.linspace(0, 1, n_bins)
        x_true = rng.uniform(0, n_bins - 1, n_time)
        spikes = rng.poisson(1.0, (n_time, n_cells))

        # Metrics are now 2D: (n_time, n_cells)
        metrics = {
            "predictive": rng.dirichlet(np.ones(n_bins), size=n_time),
            "likelihood": rng.dirichlet(np.ones(n_bins), size=n_time),
            "posterior": rng.dirichlet(np.ones(n_bins), size=n_time),
            "hpd_overlap": rng.uniform(0, 1, (n_time, n_cells)),
            "kl_divergence": rng.uniform(0, 5, (n_time, n_cells)),
            "spike_prob": rng.uniform(0, 1, (n_time, n_cells)),
        }

        thresholds = Thresholds(
            hpd_overlap=0.7,
            kl_divergence=1.5,
            spike_prob=0.05,
        )

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

        placefield_centers = np.linspace(0, 1, n_cells)
        placefield_width = 0.15
        rate_scale = 5.0

        fig = plot_combined_diagnostics(
            xs,
            x_true,
            spikes,
            metrics,
            thresholds,
            params,
            placefield_centers,
            placefield_width,
            rate_scale,
        )

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestExtractContiguousRegions:
    """Tests for extract_contiguous_regions function."""

    def test_single_region(self) -> None:
        """Test extraction of single contiguous region."""
        x = np.linspace(0, 10, 100)
        mask = (x > 2) & (x < 8)
        regions = extract_contiguous_regions(mask, x)

        assert len(regions) == 1
        start, end = regions[0]
        assert 2.0 < start < 2.5
        assert 7.5 < end < 8.0

    def test_multiple_regions(self) -> None:
        """Test extraction of multiple regions."""
        x = np.linspace(0, 10, 100)
        mask = ((x > 1) & (x < 3)) | ((x > 6) & (x < 9))
        regions = extract_contiguous_regions(mask, x)

        assert len(regions) == 2
        # First region ~(1, 3)
        assert 1.0 < regions[0][0] < 1.5
        assert 2.5 < regions[0][1] < 3.0
        # Second region ~(6, 9)
        assert 6.0 < regions[1][0] < 6.5
        assert 8.5 < regions[1][1] < 9.0

    def test_empty_mask(self) -> None:
        """Test with no True values."""
        x = np.linspace(0, 10, 100)
        mask = np.zeros(100, dtype=bool)
        regions = extract_contiguous_regions(mask, x)

        assert regions == []

    def test_all_true(self) -> None:
        """Test with all True values."""
        x = np.linspace(0, 10, 100)
        mask = np.ones(100, dtype=bool)
        regions = extract_contiguous_regions(mask, x)

        assert len(regions) == 1
        assert regions[0][0] == x[0]
        assert regions[0][1] == x[-1]

    def test_edges_true(self) -> None:
        """Test with True values at array edges."""
        x = np.linspace(0, 10, 100)
        mask = (x < 2) | (x > 8)
        regions = extract_contiguous_regions(mask, x)

        assert len(regions) == 2
        # First region starts at 0
        assert regions[0][0] == x[0]
        # Second region ends at 10
        assert regions[1][1] == x[-1]


class TestCreateDistributionComparisonPanel:
    """Tests for create_distribution_comparison_panel function."""

    def test_creates_panel(self) -> None:
        """Test that panel is created without errors."""
        fig, ax = plt.subplots()
        x = np.linspace(-20, 20, 1000)

        create_distribution_comparison_panel(
            ax,
            x,
            predictive_params=(0, 1.5),
            likelihood_params=(5, 1.5),
            color_predictive="blue",
            color_likelihood="orange",
        )

        # Should have at least 2 lines (predictive and likelihood)
        assert len(ax.lines) >= 2
        plt.close(fig)

    def test_with_title(self) -> None:
        """Test panel with title."""
        fig, ax = plt.subplots()
        x = np.linspace(-20, 20, 1000)

        create_distribution_comparison_panel(
            ax,
            x,
            predictive_params=(0, 1.5),
            likelihood_params=(5, 1.5),
            color_predictive="blue",
            color_likelihood="orange",
            title="Test Title",
        )

        assert ax.get_title() == "Test Title"
        plt.close(fig)

    def test_with_labels(self) -> None:
        """Test panel with labels shown."""
        fig, ax = plt.subplots()
        x = np.linspace(-20, 20, 1000)

        create_distribution_comparison_panel(
            ax,
            x,
            predictive_params=(0, 1.5),
            likelihood_params=(5, 1.5),
            color_predictive="blue",
            color_likelihood="orange",
            show_labels=True,
        )

        # Should have text labels
        texts = ax.texts
        assert len(texts) >= 2  # At least "Predictive" and "Likelihood"
        plt.close(fig)

    def test_hpd_bars_added(self) -> None:
        """Test that HPD bars are added as patches."""
        fig, ax = plt.subplots()
        x = np.linspace(-20, 20, 1000)

        create_distribution_comparison_panel(
            ax,
            x,
            predictive_params=(0, 1.5),
            likelihood_params=(5, 1.5),
            color_predictive="blue",
            color_likelihood="orange",
        )

        # Should have patches (Rectangle for HPD bars)
        assert len(ax.patches) >= 2  # At least 2 HPD bars
        plt.close(fig)

    def test_different_scenarios(self) -> None:
        """Test with different distribution parameters."""
        fig, ax = plt.subplots()
        x = np.linspace(-20, 20, 1000)

        # Overlapping distributions (consistent)
        create_distribution_comparison_panel(
            ax,
            x,
            predictive_params=(0, 5.0),
            likelihood_params=(2, 3.0),
            color_predictive="blue",
            color_likelihood="orange",
        )

        # Should complete without error
        plt.close(fig)

        # Non-overlapping distributions (inconsistent)
        fig, ax = plt.subplots()
        create_distribution_comparison_panel(
            ax,
            x,
            predictive_params=(-10, 1.0),
            likelihood_params=(10, 1.0),
            color_predictive="blue",
            color_likelihood="orange",
        )

        plt.close(fig)
