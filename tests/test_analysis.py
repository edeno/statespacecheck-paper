"""Tests for analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from statespacecheck_paper.analysis import (
    DecodeParams,
    Thresholds,
    Transformed,
    apply_remap_for_likelihoods,
    compute_thresholds,
    decode_and_diagnostics,
    likelihood_grid_for_counts,
    transform_metrics,
)


class TestDecodeParams:
    """Tests for DecodeParams dataclass."""

    def test_default_values(self) -> None:
        """Test that default values are set correctly."""
        # Arrange & Act
        params = DecodeParams()

        # Assert
        assert params.T_remap_start == 6_000
        assert params.T_remap_end == 10_000
        assert params.T_recovery1_end == 14_000
        assert params.T_flat_end == 16_000
        assert params.T_recovery2_end == 20_000
        assert params.T_fast_end == 24_000
        assert params.T_recovery3_end == 28_000
        assert params.T_slow_end == 32_000
        assert params.sigx_pred == 0.5
        assert params.sigx_pred_fast_phase == 0.1
        assert params.sigx_pred_slow_phase == 20.0
        assert params.sigx_true_fast == 10.0
        assert params.sigx_true_slow == 0.0
        assert params.xs_min == 0
        assert params.xs_max == 100
        assert params.xs_step == 1
        assert params.pf_width == 10.0
        assert params.rate_scale == 20.0
        assert params.base_seed == 1

    def test_post_init_initializes_pf_centers(self) -> None:
        """Test that __post_init__ initializes pf_centers if not provided."""
        # Arrange & Act
        params = DecodeParams()

        # Assert
        assert params.pf_centers is not None
        expected_centers = np.arange(0, 101, 10, dtype=float)
        np.testing.assert_array_equal(params.pf_centers, expected_centers)

    def test_post_init_respects_provided_pf_centers(self) -> None:
        """Test that __post_init__ doesn't override provided pf_centers."""
        # Arrange
        custom_centers = np.array([0.0, 25.0, 50.0, 75.0, 100.0])

        # Act
        params = DecodeParams(pf_centers=custom_centers)

        # Assert
        np.testing.assert_array_equal(params.pf_centers, custom_centers)

    def test_remap_window_property(self) -> None:
        """Test that remap_window property returns correct tuple."""
        # Arrange
        params = DecodeParams(T_remap_start=1000, T_remap_end=2000)

        # Act
        remap_window = params.remap_window

        # Assert
        assert remap_window == (1000, 2000)


class TestLikelihoodGridForCounts:
    """Tests for likelihood_grid_for_counts function."""

    def test_output_shape(self) -> None:
        """Test that output has correct shape (n_bins, n_cells)."""
        # Arrange
        xs = np.linspace(0, 100, 21)  # 21 bins
        pf_centers = np.array([25.0, 50.0, 75.0])  # 3 cells
        pf_width = 5.0
        rate_scale = 0.1
        counts = np.array([2, 1, 3])  # 3 cells

        # Act
        likelihood = likelihood_grid_for_counts(xs, pf_centers, pf_width, rate_scale, counts)

        # Assert
        assert likelihood.shape == (21, 3)

    def test_normalized_per_cell(self) -> None:
        """Test that values are normalized per cell (over bins)."""
        # Arrange
        xs = np.linspace(0, 100, 21)
        pf_centers = np.array([50.0])
        pf_width = 5.0
        rate_scale = 0.1
        counts = np.array([2])

        # Act
        likelihood = likelihood_grid_for_counts(xs, pf_centers, pf_width, rate_scale, counts)

        # Assert - sum over bins (axis 0) should be 1 for each cell
        np.testing.assert_allclose(np.sum(likelihood, axis=0), 1.0, rtol=1e-5)

    def test_handles_zero_counts(self) -> None:
        """Test that function handles zero spike counts correctly."""
        # Arrange
        xs = np.linspace(0, 100, 21)
        pf_centers = np.array([25.0, 75.0])
        pf_width = 5.0
        rate_scale = 0.1
        counts = np.array([0, 0])  # Zero spikes

        # Act
        likelihood = likelihood_grid_for_counts(xs, pf_centers, pf_width, rate_scale, counts)

        # Assert - should still be normalized and not contain NaN or Inf
        assert not np.any(np.isnan(likelihood))
        assert not np.any(np.isinf(likelihood))
        np.testing.assert_allclose(np.sum(likelihood, axis=0), 1.0, rtol=1e-5)


class TestApplyRemapForLikelihoods:
    """Tests for apply_remap_for_likelihoods function."""

    def test_inactive_returns_unchanged(self) -> None:
        """Test that active=False returns unchanged likelihood."""
        # Arrange
        rng = np.random.default_rng(42)
        likelihood = rng.random((20, 5))
        remap_from_to = (0, 1)
        active = False

        # Act
        result = apply_remap_for_likelihoods(likelihood, remap_from_to, active)

        # Assert
        np.testing.assert_array_equal(result, likelihood)
        # Verify it's not a copy (same object)
        assert result is likelihood

    def test_single_remapping(self) -> None:
        """Test that single remapping (src, dst) works correctly."""
        # Arrange
        likelihood = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2 bins, 3 cells)
        remap_from_to = (0, 2)  # Cell 0 should become like cell 2
        active = True

        # Act
        result = apply_remap_for_likelihoods(likelihood, remap_from_to, active)

        # Assert
        expected = np.array([[3.0, 2.0, 3.0], [6.0, 5.0, 6.0]])
        np.testing.assert_array_equal(result, expected)
        # Verify original is unchanged
        np.testing.assert_array_equal(likelihood, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_multiple_remappings(self) -> None:
        """Test that multiple remappings work correctly."""
        # Arrange
        likelihood = np.array([[1.0, 2.0, 3.0, 4.0]])  # (1 bin, 4 cells)
        remap_from_to = ((0, 1), (2, 3))  # Cell 0→1, Cell 2→3
        active = True

        # Act
        result = apply_remap_for_likelihoods(likelihood, remap_from_to, active)

        # Assert
        expected = np.array([[2.0, 2.0, 4.0, 4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_returns_copy_when_active(self) -> None:
        """Test that active=True returns a copy, not modifying original."""
        # Arrange
        likelihood = np.array([[1.0, 2.0, 3.0]])
        remap_from_to = (0, 1)
        active = True

        # Act
        result = apply_remap_for_likelihoods(likelihood, remap_from_to, active)

        # Assert
        assert result is not likelihood
        # Original should be unchanged
        np.testing.assert_array_equal(likelihood, [[1.0, 2.0, 3.0]])


class TestDecodeAndDiagnostics:
    """Tests for decode_and_diagnostics function (integration test)."""

    def test_output_keys(self) -> None:
        """Test that output dictionary has expected keys."""
        # Arrange
        rng = np.random.default_rng(42)
        n_time, n_cells = 10, 3
        n_bins = 21
        spikes = rng.poisson(1.0, size=(n_time, n_cells))
        xs = np.linspace(0, 100, n_bins)
        transition_matrix = np.eye(n_bins) * 0.9 + 0.1 / n_bins  # Diagonal dominant
        pf_centers = np.array([25.0, 50.0, 75.0])
        pf_width = 5.0
        rate_scale = 0.1
        remap_window = (5, 7)
        remap_from_to = (0, 1)

        # Act
        result = decode_and_diagnostics(
            spikes,
            xs,
            transition_matrix,
            pf_centers,
            pf_width,
            rate_scale,
            remap_window,
            remap_from_to,
        )

        # Assert
        assert "posterior" in result
        assert "hpd_overlap" in result
        assert "kl_divergence" in result
        assert "conditional_pvalue" in result

    def test_output_shapes(self) -> None:
        """Test that output arrays have correct shapes."""
        # Arrange
        rng = np.random.default_rng(42)
        n_time, n_cells = 10, 3
        n_bins = 21
        spikes = rng.poisson(1.0, size=(n_time, n_cells))
        xs = np.linspace(0, 100, n_bins)
        transition_matrix = np.eye(n_bins) * 0.9 + 0.1 / n_bins
        pf_centers = np.array([25.0, 50.0, 75.0])
        pf_width = 5.0
        rate_scale = 0.1
        remap_window = (5, 7)
        remap_from_to = (0, 1)

        # Act
        result = decode_and_diagnostics(
            spikes,
            xs,
            transition_matrix,
            pf_centers,
            pf_width,
            rate_scale,
            remap_window,
            remap_from_to,
        )

        # Assert
        assert result["posterior"].shape == (n_time, n_bins)
        assert result["hpd_overlap"].shape == (n_time,)
        assert result["kl_divergence"].shape == (n_time,)
        assert result["conditional_pvalue"].shape == (n_time,)

    def test_nan_handling(self) -> None:
        """Test that NaN values are correctly placed in outputs."""
        # Arrange
        n_bins = 11
        # spikes: t=0 has spikes, t=1 has spikes, t=2 has spikes, t=3 has NO spikes, t=4 has spikes
        spikes = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [2, 2]])
        xs = np.linspace(0, 100, n_bins)
        transition_matrix = np.eye(n_bins) * 0.9 + 0.1 / n_bins
        pf_centers = np.array([25.0, 75.0])
        pf_width = 5.0
        rate_scale = 0.1
        remap_window = (10, 10)  # Outside range
        remap_from_to = (0, 1)

        # Act
        result = decode_and_diagnostics(
            spikes,
            xs,
            transition_matrix,
            pf_centers,
            pf_width,
            rate_scale,
            remap_window,
            remap_from_to,
        )

        # Assert
        # t=0 should have NaN for hpd_overlap, kl_divergence (no prior available)
        assert np.isnan(result["hpd_overlap"][0])
        assert np.isnan(result["kl_divergence"][0])
        # t=3 has no spikes, so hpd_overlap and kl_divergence should be NaN
        assert np.isnan(result["hpd_overlap"][3])
        assert np.isnan(result["kl_divergence"][3])
        # t=1, t=2, t=4 have spikes, so hpd_overlap and kl_divergence should be computed
        assert not np.isnan(result["hpd_overlap"][1])
        assert not np.isnan(result["hpd_overlap"][2])
        assert not np.isnan(result["hpd_overlap"][4])
        assert not np.isnan(result["kl_divergence"][1])
        assert not np.isnan(result["kl_divergence"][2])
        assert not np.isnan(result["kl_divergence"][4])
        # conditional_pvalue:
        # - t=0: NaN because spike_prob is not computed at t=0 (loop starts at t=1)
        # - t=3: NaN because no spikes
        # - t=1, t=2, t=4: computed because spikes occurred and spike_prob is available
        assert np.isnan(result["conditional_pvalue"][0])  # spike_prob not computed at t=0
        assert np.isnan(result["conditional_pvalue"][3])  # No spikes
        assert not np.isnan(result["conditional_pvalue"][1])
        assert not np.isnan(result["conditional_pvalue"][2])
        assert not np.isnan(result["conditional_pvalue"][4])

    def test_with_narrow_transition_matrix(self) -> None:
        """Test that narrow transition matrix is used in specified window."""
        # Arrange
        rng = np.random.default_rng(42)
        n_time, n_cells = 10, 2
        n_bins = 11
        spikes = rng.poisson(1.0, size=(n_time, n_cells))
        xs = np.linspace(0, 100, n_bins)
        transition_matrix = np.eye(n_bins) * 0.5 + 0.5 / n_bins
        transition_matrix_narrow = np.eye(n_bins) * 0.9 + 0.1 / n_bins
        pf_centers = np.array([25.0, 75.0])
        pf_width = 5.0
        rate_scale = 0.1
        remap_window = (20, 20)  # Outside range
        remap_from_to = (0, 1)
        narrow_window = (3, 6)

        # Act - should not raise error
        result = decode_and_diagnostics(
            spikes,
            xs,
            transition_matrix,
            pf_centers,
            pf_width,
            rate_scale,
            remap_window,
            remap_from_to,
            transition_matrix_narrow=transition_matrix_narrow,
            narrow_window=narrow_window,
        )

        # Assert - output structure should be correct
        assert result["posterior"].shape == (n_time, n_bins)

    def test_with_inflated_transition_matrix(self) -> None:
        """Test that inflated transition matrix is used in specified window."""
        # Arrange
        rng = np.random.default_rng(42)
        n_time, n_cells = 10, 2
        n_bins = 11
        spikes = rng.poisson(1.0, size=(n_time, n_cells))
        xs = np.linspace(0, 100, n_bins)
        transition_matrix = np.eye(n_bins) * 0.5 + 0.5 / n_bins
        transition_matrix_inflated = np.eye(n_bins) * 0.2 + 0.8 / n_bins
        pf_centers = np.array([25.0, 75.0])
        pf_width = 5.0
        rate_scale = 0.1
        remap_window = (20, 20)  # Outside range
        remap_from_to = (0, 1)
        inflate_window = (4, 7)

        # Act - should not raise error
        result = decode_and_diagnostics(
            spikes,
            xs,
            transition_matrix,
            pf_centers,
            pf_width,
            rate_scale,
            remap_window,
            remap_from_to,
            transition_matrix_inflated=transition_matrix_inflated,
            inflate_window=inflate_window,
        )

        # Assert - output structure should be correct
        assert result["posterior"].shape == (n_time, n_bins)


class TestThresholds:
    """Tests for Thresholds dataclass."""

    def test_instantiation(self) -> None:
        """Test that Thresholds can be instantiated correctly."""
        # Arrange & Act
        thresholds = Thresholds(
            hpd_overlap=0.5,
            kl_divergence=2.0,
            conditional_pvalue=-3.0,  # Sum of logs (negative)
        )

        # Assert
        assert thresholds.hpd_overlap == 0.5
        assert thresholds.kl_divergence == 2.0
        assert thresholds.conditional_pvalue == -3.0


class TestComputeThresholds:
    """Tests for compute_thresholds function."""

    def test_threshold_computation(self) -> None:
        """Test that thresholds are computed correctly from baseline."""
        # Arrange
        rng = np.random.default_rng(42)
        n_time = 100
        metrics = {
            "hpd_overlap": rng.uniform(0.5, 1.0, n_time),
            "kl_divergence": rng.uniform(0.0, 2.0, n_time),
            # conditional_pvalue is now sum of logs (negative values)
            "conditional_pvalue": rng.uniform(-5.0, -0.1, n_time),
        }
        baseline_end = 50

        # Act
        thresholds = compute_thresholds(metrics, baseline_end=baseline_end)

        # Assert
        assert isinstance(thresholds, Thresholds)
        # HPD overlap threshold should be 1st percentile of baseline
        expected_hpd_overlap = np.nanquantile(metrics["hpd_overlap"][:baseline_end], 0.01)
        assert thresholds.hpd_overlap == pytest.approx(expected_hpd_overlap)
        # KL divergence threshold should be 99th percentile of baseline
        expected_kl_divergence = np.nanquantile(metrics["kl_divergence"][:baseline_end], 0.99)
        assert thresholds.kl_divergence == pytest.approx(expected_kl_divergence)
        # conditional_pvalue threshold should be 1st percentile (more negative = worse fit)
        expected_conditional_pvalue = np.nanquantile(
            metrics["conditional_pvalue"][:baseline_end], 0.01
        )
        assert thresholds.conditional_pvalue == pytest.approx(expected_conditional_pvalue)

    def test_handles_nan_values(self) -> None:
        """Test that compute_thresholds handles NaN values correctly."""
        # Arrange
        n_time = 20
        hpdo = np.full(n_time, 0.8)
        hpdo[:5] = np.nan  # First 5 are NaN
        metrics = {
            "hpd_overlap": hpdo,
            "kl_divergence": np.full(n_time, 1.0),
            "conditional_pvalue": np.full(n_time, -1.0),  # Sum of logs (negative)
        }

        # Act
        thresholds = compute_thresholds(metrics, baseline_end=10)

        # Assert - should compute from non-NaN values
        assert not np.isnan(thresholds.hpd_overlap)
        assert not np.isnan(thresholds.kl_divergence)
        assert not np.isnan(thresholds.conditional_pvalue)


class TestTransformed:
    """Tests for Transformed dataclass."""

    def test_instantiation(self) -> None:
        """Test that Transformed can be instantiated correctly."""
        # Arrange
        hpd_overlap = np.array([1.0, 2.0, 3.0])
        kl_divergence = np.array([0.5, 1.0, 1.5])
        conditional_pvalue = np.array([0.1, 0.5, 0.9])

        # Act
        transformed = Transformed(
            hpd_overlap=hpd_overlap,
            kl_divergence=kl_divergence,
            conditional_pvalue=conditional_pvalue,
            hpd_overlap_threshold=1.5,
            kl_divergence_threshold=1.0,
            conditional_pvalue_threshold=3.0,
        )

        # Assert
        np.testing.assert_array_equal(transformed.hpd_overlap, hpd_overlap)
        np.testing.assert_array_equal(transformed.kl_divergence, kl_divergence)
        np.testing.assert_array_equal(transformed.conditional_pvalue, conditional_pvalue)
        assert transformed.hpd_overlap_threshold == 1.5
        assert transformed.kl_divergence_threshold == 1.0
        assert transformed.conditional_pvalue_threshold == 3.0


class TestTransformMetrics:
    """Tests for transform_metrics function."""

    def test_transformations_applied(self) -> None:
        """Test that transformations are applied correctly."""
        # Arrange
        metrics = {
            "hpd_overlap": np.array([0.5, 0.8, 0.9]),
            "kl_divergence": np.array([1.0, 4.0, 9.0]),
            # conditional_pvalue is now sum of logs (negative values)
            "conditional_pvalue": np.array([-1.0, -0.5, -0.1]),
        }
        thresholds = Thresholds(
            hpd_overlap=0.6,
            kl_divergence=5.0,
            conditional_pvalue=-2.0,  # More negative = worse fit
        )
        eps1 = 1e-2

        # Act
        transformed = transform_metrics(metrics, thresholds, eps1=eps1)

        # Assert
        # HPDO: -log(HPDO + eps1)
        expected_hpdo = -np.log(metrics["hpd_overlap"] + eps1)
        np.testing.assert_allclose(transformed.hpd_overlap, expected_hpdo)
        # KL: sqrt(KL)
        expected_kl = np.sqrt(metrics["kl_divergence"])
        np.testing.assert_allclose(transformed.kl_divergence, expected_kl)
        # conditional_pvalue: negated (already sum of logs)
        expected_conditional_pvalue = -metrics["conditional_pvalue"]
        np.testing.assert_allclose(transformed.conditional_pvalue, expected_conditional_pvalue)
        # Thresholds transformed
        expected_hpd_overlap_threshold = -np.log(thresholds.hpd_overlap + eps1)
        expected_kl_divergence_threshold = np.sqrt(thresholds.kl_divergence)
        expected_conditional_pvalue_threshold = -thresholds.conditional_pvalue
        assert transformed.hpd_overlap_threshold == pytest.approx(expected_hpd_overlap_threshold)
        assert transformed.kl_divergence_threshold == pytest.approx(
            expected_kl_divergence_threshold
        )
        assert transformed.conditional_pvalue_threshold == pytest.approx(
            expected_conditional_pvalue_threshold
        )

    def test_handles_nan_values(self) -> None:
        """Test that transform_metrics preserves NaN values."""
        # Arrange
        metrics = {
            "hpd_overlap": np.array([0.5, np.nan, 0.9]),
            "kl_divergence": np.array([1.0, 4.0, np.nan]),
            "conditional_pvalue": np.array([-1.0, np.nan, -0.5]),
        }
        thresholds = Thresholds(
            hpd_overlap=0.6,
            kl_divergence=5.0,
            conditional_pvalue=-2.0,
        )

        # Act
        transformed = transform_metrics(metrics, thresholds)

        # Assert - NaN values should be preserved
        assert np.isnan(transformed.hpd_overlap[1])
        assert np.isnan(transformed.kl_divergence[2])
        assert np.isnan(transformed.conditional_pvalue[1])

    def test_default_eps_values(self) -> None:
        """Test that default eps values are used correctly."""
        # Arrange
        metrics = {
            "hpd_overlap": np.array([0.5]),
            "kl_divergence": np.array([1.0]),
            "conditional_pvalue": np.array([-1.0]),  # Sum of logs
        }
        thresholds = Thresholds(
            hpd_overlap=0.6,
            kl_divergence=5.0,
            conditional_pvalue=-2.0,
        )

        # Act - use defaults
        transformed = transform_metrics(metrics, thresholds)

        # Assert - should not raise errors and produce finite values
        assert np.isfinite(transformed.hpd_overlap[0])
        assert np.isfinite(transformed.kl_divergence[0])
        assert np.isfinite(transformed.conditional_pvalue[0])
