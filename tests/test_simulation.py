"""Tests for simulation utilities."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from statespacecheck_paper.simulation import (
    gaussian_transition_matrix,
    normalize,
    placefield_rates,
    reflect_into_interval,
    safe_log,
    simulate_spikes_flat_rate,
    simulate_spikes_position_tuned,
    simulate_walk,
    spike_prob_rank,
)


class TestNormalize:
    """Tests for normalize function."""

    def test_normalize_1d_array(self) -> None:
        """Test normalization of 1D array."""
        p = np.array([1.0, 2.0, 3.0])
        result = normalize(p)
        assert_allclose(result, [1 / 6, 2 / 6, 3 / 6])
        assert_allclose(np.sum(result), 1.0)

    def test_normalize_2d_array_axis0(self) -> None:
        """Test normalization of 2D array along axis 0."""
        p = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = normalize(p, axis=0)
        assert_allclose(result, [[1 / 4, 2 / 6], [3 / 4, 4 / 6]])
        assert_allclose(np.sum(result, axis=0), [1.0, 1.0])

    def test_normalize_2d_array_axis1(self) -> None:
        """Test normalization of 2D array along axis 1."""
        p = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = normalize(p, axis=1)
        assert_allclose(result, [[1 / 3, 2 / 3], [3 / 7, 4 / 7]])
        assert_allclose(np.sum(result, axis=1), [1.0, 1.0])

    def test_normalize_zeros(self) -> None:
        """Test normalization of array with zeros."""
        p = np.array([0.0, 0.0, 0.0])
        result = normalize(p)
        # Should not divide by zero (uses eps)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_normalize_with_custom_eps(self) -> None:
        """Test normalization with custom epsilon."""
        p = np.array([0.0, 0.0])
        result = normalize(p, eps=1e-6)
        assert not np.any(np.isnan(result))


class TestReflectIntoInterval:
    """Tests for reflect_into_interval function."""

    def test_values_inside_bounds_unchanged(self) -> None:
        """Test that values inside bounds are unchanged."""
        x = np.array([0.5, 1.0, 1.5])
        result = reflect_into_interval(x, 0.0, 2.0)
        assert_allclose(result, x)

    def test_values_above_upper_bound_reflected(self) -> None:
        """Test that values above upper bound are reflected."""
        x = np.array([2.5])  # 0.5 above upper bound of 2.0
        result = reflect_into_interval(x, 0.0, 2.0)
        assert_allclose(result, [1.5])  # Reflected to 1.5

    def test_values_below_lower_bound_reflected(self) -> None:
        """Test that values below lower bound are reflected."""
        x = np.array([-0.5])  # 0.5 below lower bound of 0.0
        result = reflect_into_interval(x, 0.0, 2.0)
        assert_allclose(result, [0.5])  # Reflected to 0.5

    def test_multiple_reflections(self) -> None:
        """Test that multiple reflections work correctly."""
        x = np.array([4.5])  # More than length above upper bound
        result = reflect_into_interval(x, 0.0, 2.0)
        # 4.5 → reflects back and forth
        assert 0.0 <= result[0] <= 2.0

    def test_array_of_values(self) -> None:
        """Test with array of values."""
        x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
        result = reflect_into_interval(x, 0.0, 2.0)
        # All should be in [0, 2]
        assert np.all(result >= 0.0)
        assert np.all(result <= 2.0)


class TestGaussianTransitionMatrix:
    """Tests for gaussian_transition_matrix function."""

    def test_transition_matrix_shape(self) -> None:
        """Test that transition matrix has correct shape."""
        xs = np.array([0.0, 1.0, 2.0])
        matrix = gaussian_transition_matrix(xs, sig=1.0)
        assert matrix.shape == (3, 3)

    def test_columns_sum_to_one(self) -> None:
        """Test that each column sums to 1 (probability distribution)."""
        xs = np.linspace(0, 10, 11)
        matrix = gaussian_transition_matrix(xs, sig=1.0)
        col_sums = np.sum(matrix, axis=0)
        assert_allclose(col_sums, np.ones(11), rtol=1e-10)

    def test_diagonal_dominant_for_small_sigma(self) -> None:
        """Test that diagonal is dominant for small sigma."""
        xs = np.array([0.0, 1.0, 2.0])
        matrix = gaussian_transition_matrix(xs, sig=0.1)
        # Diagonal should be largest in each column
        for i in range(3):
            assert matrix[i, i] == np.max(matrix[:, i])

    def test_larger_sigma_spreads_probability(self) -> None:
        """Test that larger sigma spreads probability more."""
        xs = np.array([0.0, 1.0, 2.0])
        matrix_small = gaussian_transition_matrix(xs, sig=0.1)
        matrix_large = gaussian_transition_matrix(xs, sig=2.0)
        # Small sigma should have more concentrated diagonal
        assert matrix_small[0, 0] > matrix_large[0, 0]


class TestSafeLog:
    """Tests for safe_log function."""

    def test_safe_log_positive_values(self) -> None:
        """Test safe_log on positive values."""
        x = np.array([1.0, 2.0, np.e])
        result = safe_log(x)
        assert_allclose(result, np.log(x))

    def test_safe_log_with_zeros(self) -> None:
        """Test safe_log with zeros (should not produce -inf)."""
        x = np.array([0.0, 1.0, 2.0])
        result = safe_log(x)
        assert not np.any(np.isinf(result))
        assert not np.any(np.isnan(result))

    def test_safe_log_custom_eps(self) -> None:
        """Test safe_log with custom epsilon."""
        x = np.array([0.0])
        result = safe_log(x, eps=1e-6)
        assert_allclose(result, np.log(1e-6))


class TestPlacefieldRates:
    """Tests for placefield_rates function."""

    def test_placefield_rates_shape(self) -> None:
        """Test that output has correct shape."""
        xs = np.linspace(0, 10, 11)
        centers = np.array([2.0, 5.0, 8.0])
        rates = placefield_rates(xs, centers, width=1.0, scale=1.0)
        assert rates.shape == (11, 3)

    def test_placefield_rates_peak_at_center(self) -> None:
        """Test that rate is highest at place field center."""
        xs = np.linspace(0, 10, 101)
        centers = np.array([5.0])
        rates = placefield_rates(xs, centers, width=1.0, scale=1.0)
        # Find where xs is closest to center
        center_idx = np.argmin(np.abs(xs - 5.0))
        # Rate should be highest at center
        assert rates[center_idx, 0] == np.max(rates[:, 0])

    def test_placefield_rates_scale_effect(self) -> None:
        """Test that scale parameter multiplies rates."""
        xs = np.linspace(0, 10, 11)
        centers = np.array([5.0])
        rates1 = placefield_rates(xs, centers, width=1.0, scale=1.0)
        rates2 = placefield_rates(xs, centers, width=1.0, scale=2.0)
        assert_allclose(rates2, 2.0 * rates1)


class TestSpikeProbRank:
    """Tests for spike_prob_rank function."""

    def test_spike_prob_rank_shape(self) -> None:
        """Test that output has correct shape."""
        prior = np.array([0.5, 0.3, 0.2])
        lambda_ratio = np.array([[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]])
        # Normalize lambda_ratio rows
        lambda_ratio = lambda_ratio / lambda_ratio.sum(axis=0, keepdims=True)
        result = spike_prob_rank(prior, lambda_ratio)
        assert result.shape == (2,)

    def test_spike_prob_rank_values_in_range(self) -> None:
        """Test that output values are in [0, 1]."""
        prior = np.array([0.5, 0.3, 0.2])
        lambda_ratio = np.array([[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]])
        lambda_ratio = lambda_ratio / lambda_ratio.sum(axis=0, keepdims=True)
        result = spike_prob_rank(prior, lambda_ratio)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_spike_prob_rank_uniform(self) -> None:
        """Test with uniform contributions."""
        n_bins = 10
        n_cells = 5
        prior = np.ones(n_bins) / n_bins
        lambda_ratio = np.ones((n_bins, n_cells)) / n_bins
        result = spike_prob_rank(prior, lambda_ratio)
        # All cells should have same rank when contributions are equal
        assert_allclose(result, result[0] * np.ones(n_cells), atol=1e-10)


class TestSimulateWalk:
    """Tests for simulate_walk function."""

    def test_simulate_walk_shape(self) -> None:
        """Test that output has correct shape."""
        rng = np.random.default_rng(42)
        result = simulate_walk(100, sig=1.0, x0=50.0, xs_min=0.0, xs_max=100.0, rng=rng)
        assert result.shape == (100,)

    def test_simulate_walk_respects_boundaries(self) -> None:
        """Test that walk stays within boundaries."""
        rng = np.random.default_rng(42)
        result = simulate_walk(1000, sig=5.0, x0=50.0, xs_min=0.0, xs_max=100.0, rng=rng)
        assert np.all(result >= 0.0)
        assert np.all(result <= 100.0)

    def test_simulate_walk_reproducible(self) -> None:
        """Test that walk is reproducible with same seed."""
        rng1 = np.random.default_rng(42)
        result1 = simulate_walk(100, sig=1.0, x0=50.0, xs_min=0.0, xs_max=100.0, rng=rng1)
        rng2 = np.random.default_rng(42)
        result2 = simulate_walk(100, sig=1.0, x0=50.0, xs_min=0.0, xs_max=100.0, rng=rng2)
        assert_array_equal(result1, result2)

    def test_simulate_walk_initial_position(self) -> None:
        """Test that walk starts at initial position."""
        rng = np.random.default_rng(42)
        result = simulate_walk(100, sig=0.0, x0=50.0, xs_min=0.0, xs_max=100.0, rng=rng)
        # With sig=0, all values should be x0
        assert_allclose(result, 50.0 * np.ones(100))

    def test_simulate_walk_larger_sigma_more_variance(self) -> None:
        """Test that larger sigma produces more variance."""
        rng1 = np.random.default_rng(42)
        result1 = simulate_walk(1000, sig=0.5, x0=50.0, xs_min=0.0, xs_max=100.0, rng=rng1)
        rng2 = np.random.default_rng(43)
        result2 = simulate_walk(1000, sig=5.0, x0=50.0, xs_min=0.0, xs_max=100.0, rng=rng2)
        # Larger sigma should have more variance
        assert np.std(result2) > np.std(result1)


class TestSimulateSpikesPositionTuned:
    """Tests for simulate_spikes_position_tuned function."""

    def test_simulate_spikes_position_tuned_shape(self) -> None:
        """Test that output has correct shape."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 100, 100)
        pf_centers = np.array([25.0, 50.0, 75.0])
        result = simulate_spikes_position_tuned(
            x, pf_centers, pf_width=5.0, rate_scale=0.1, rng=rng
        )
        assert result.shape == (100, 3)

    def test_simulate_spikes_position_tuned_nonnegative_integers(self) -> None:
        """Test that spikes are non-negative integers."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 100, 100)
        pf_centers = np.array([25.0, 50.0, 75.0])
        result = simulate_spikes_position_tuned(
            x, pf_centers, pf_width=5.0, rate_scale=0.1, rng=rng
        )
        assert np.all(result >= 0)
        assert result.dtype == np.int_ or np.issubdtype(result.dtype, np.integer)

    def test_simulate_spikes_position_tuned_higher_near_center(self) -> None:
        """Test that spike rate is higher near place field center."""
        rng = np.random.default_rng(42)
        # Create long trajectory to get good statistics
        x = np.linspace(0, 100, 10000)
        pf_centers = np.array([50.0])
        result = simulate_spikes_position_tuned(
            x, pf_centers, pf_width=5.0, rate_scale=1.0, rng=rng
        )
        # Split into near center and far from center
        near_center = np.abs(x - 50.0) < 10.0
        far_from_center = np.abs(x - 50.0) > 40.0
        mean_near = result[near_center, 0].mean()
        mean_far = result[far_from_center, 0].mean()
        assert mean_near > mean_far

    def test_simulate_spikes_position_tuned_reproducible(self) -> None:
        """Test that spikes are reproducible with same seed."""
        x = np.linspace(0, 100, 100)
        pf_centers = np.array([50.0])
        rng1 = np.random.default_rng(42)
        result1 = simulate_spikes_position_tuned(
            x, pf_centers, pf_width=5.0, rate_scale=0.1, rng=rng1
        )
        rng2 = np.random.default_rng(42)
        result2 = simulate_spikes_position_tuned(
            x, pf_centers, pf_width=5.0, rate_scale=0.1, rng=rng2
        )
        assert_array_equal(result1, result2)


class TestSimulateSpikesFlatRate:
    """Tests for simulate_spikes_flat_rate function."""

    def test_simulate_spikes_flat_rate_shape(self) -> None:
        """Test that output has correct shape."""
        rng = np.random.default_rng(42)
        result = simulate_spikes_flat_rate(100, 5, rate=0.1, rng=rng)
        assert result.shape == (100, 5)

    def test_simulate_spikes_flat_rate_nonnegative_integers(self) -> None:
        """Test that spikes are non-negative integers."""
        rng = np.random.default_rng(42)
        result = simulate_spikes_flat_rate(100, 5, rate=0.1, rng=rng)
        assert np.all(result >= 0)
        assert result.dtype == np.int_ or np.issubdtype(result.dtype, np.integer)

    def test_simulate_spikes_flat_rate_mean_close_to_rate(self) -> None:
        """Test that mean spike count is close to specified rate."""
        rng = np.random.default_rng(42)
        rate = 2.0
        result = simulate_spikes_flat_rate(10000, 10, rate=rate, rng=rng)
        mean_rate = result.mean()
        # Should be close to specified rate (within 10% for large sample)
        assert_allclose(mean_rate, rate, rtol=0.1)

    def test_simulate_spikes_flat_rate_reproducible(self) -> None:
        """Test that spikes are reproducible with same seed."""
        rng1 = np.random.default_rng(42)
        result1 = simulate_spikes_flat_rate(100, 5, rate=0.1, rng=rng1)
        rng2 = np.random.default_rng(42)
        result2 = simulate_spikes_flat_rate(100, 5, rate=0.1, rng=rng2)
        assert_array_equal(result1, result2)
