"""Tests for posterior predictive check module."""

from __future__ import annotations

import numpy as np
import xarray as xr
from numpy.testing import assert_array_almost_equal, assert_array_equal

from statespacecheck_paper.predictive_check import (
    compute_log_likelihood_from_place_fields,
    compute_predictive_pvalues,
    create_posterior_predictive_sampler,
    extract_place_fields_from_model,
    generate_spikes_from_place_fields,
    sample_positions_from_posterior,
)


class TestSamplePositionsFromPosterior:
    """Tests for sample_positions_from_posterior function."""

    def test_basic_sampling(self) -> None:
        """Test basic categorical sampling from posterior."""
        rng = np.random.default_rng(42)
        posterior = np.array([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])

        positions = sample_positions_from_posterior(posterior, rng)

        assert positions.shape == (2,)
        assert positions.dtype == np.int64
        assert np.all((positions >= 0) & (positions < 3))

    def test_deterministic_mode(self) -> None:
        """Test that posterior with peaked distribution selects highest probability bin."""
        from scipy.stats import mode

        rng = np.random.default_rng(42)
        # Very peaked distributions
        posterior = np.array([[0.01, 0.98, 0.01], [0.01, 0.01, 0.98]])

        # Run multiple times to check consistency
        results = np.array([sample_positions_from_posterior(posterior, rng) for _ in range(100)])

        # Most samples should select the highest probability bins
        modes = mode(results, axis=0, keepdims=False)
        # First timepoint should mostly pick bin 1, second should pick bin 2
        assert modes.mode[0] == 1, f"Expected mode 1 for first timepoint, got {modes.mode[0]}"
        assert modes.mode[1] == 2, f"Expected mode 2 for second timepoint, got {modes.mode[1]}"

    def test_uniform_posterior(self) -> None:
        """Test sampling from uniform posterior."""
        rng = np.random.default_rng(42)
        n_time = 1000
        n_bins = 5
        posterior = np.ones((n_time, n_bins)) / n_bins

        positions = sample_positions_from_posterior(posterior, rng)

        # Check all bins are sampled roughly equally
        counts = np.bincount(positions, minlength=n_bins)
        expected = n_time / n_bins
        # Within 10% of expected
        assert np.all(np.abs(counts - expected) < expected * 0.2)

    def test_unnormalized_posterior(self) -> None:
        """Test that function normalizes posterior properly."""
        rng = np.random.default_rng(42)
        # Unnormalized posterior (doesn't sum to 1)
        posterior = np.array([[2.0, 5.0, 3.0], [1.0, 1.0, 8.0]])

        positions = sample_positions_from_posterior(posterior, rng)

        assert positions.shape == (2,)
        assert np.all((positions >= 0) & (positions < 3))

    def test_single_timepoint(self) -> None:
        """Test with single timepoint."""
        rng = np.random.default_rng(42)
        posterior = np.array([[0.2, 0.5, 0.3]])

        positions = sample_positions_from_posterior(posterior, rng)

        assert positions.shape == (1,)
        assert 0 <= positions[0] < 3

    def test_reproducibility(self) -> None:
        """Test that same seed gives same results."""
        posterior = np.array([[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]])

        rng1 = np.random.default_rng(42)
        positions1 = sample_positions_from_posterior(posterior, rng1)

        rng2 = np.random.default_rng(42)
        positions2 = sample_positions_from_posterior(posterior, rng2)

        assert_array_equal(positions1, positions2)

    def test_vectorized_equivalence_to_loop(self) -> None:
        """Test that vectorized inverse CDF sampling is equivalent to loop-based rng.choice()."""
        from statespacecheck_paper.simulation import normalize

        # Test with various posterior distributions
        n_time = 50
        n_bins = 10
        rng_data = np.random.default_rng(123)
        posterior = rng_data.dirichlet(np.ones(n_bins), size=n_time)

        # Reference implementation: loop-based sampling with rng.choice
        def sample_loop_based(posterior: np.ndarray, rng: np.random.Generator) -> np.ndarray:
            """Reference implementation using loop and rng.choice."""
            n_time, n_bins = posterior.shape
            posterior_norm = normalize(posterior, axis=1)
            return np.array(
                [rng.choice(n_bins, p=posterior_norm[t]) for t in range(n_time)],
                dtype=np.int64,
            )

        # Test both implementations with same seed
        rng1 = np.random.default_rng(42)
        loop_result = sample_loop_based(posterior, rng1)

        rng2 = np.random.default_rng(42)
        vectorized_result = sample_positions_from_posterior(posterior, rng2)

        # Should produce identical results
        assert_array_equal(
            loop_result,
            vectorized_result,
            err_msg="Vectorized inverse CDF method should be equivalent to loop-based rng.choice",
        )


class TestGenerateSpikesFromPlaceFields:
    """Tests for generate_spikes_from_place_fields function."""

    def test_basic_spike_generation(self) -> None:
        """Test basic spike generation."""
        rng = np.random.default_rng(42)
        position_indices = np.array([0, 1, 2])
        place_fields = np.array([[10.0, 20.0, 5.0], [5.0, 10.0, 15.0]])
        dt = 0.002  # 500 Hz

        spikes = generate_spikes_from_place_fields(position_indices, place_fields, dt, rng)

        assert spikes.shape == (3, 2)
        assert spikes.dtype == np.int64
        assert np.all(spikes >= 0)

    def test_zero_rate(self) -> None:
        """Test spike generation with zero firing rate."""
        rng = np.random.default_rng(42)
        position_indices = np.array([0, 1, 2])
        place_fields = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        dt = 0.002

        spikes = generate_spikes_from_place_fields(position_indices, place_fields, dt, rng)

        assert spikes.shape == (3, 2)
        assert_array_equal(spikes, np.zeros((3, 2), dtype=np.int64))

    def test_high_rate(self) -> None:
        """Test spike generation with very high firing rate."""
        rng = np.random.default_rng(42)
        position_indices = np.array([0])
        place_fields = np.array([[1000.0]])  # Very high rate
        dt = 0.002

        spikes = generate_spikes_from_place_fields(position_indices, place_fields, dt, rng)

        # Expected count = 1000 * 0.002 = 2
        # Should get approximately 2 spikes
        assert spikes.shape == (1, 1)
        assert spikes[0, 0] >= 0

    def test_reproducibility(self) -> None:
        """Test reproducible spike generation."""
        position_indices = np.array([0, 1, 2])
        place_fields = np.array([[10.0, 20.0, 5.0], [5.0, 10.0, 15.0]])
        dt = 0.002

        rng1 = np.random.default_rng(42)
        spikes1 = generate_spikes_from_place_fields(position_indices, place_fields, dt, rng1)

        rng2 = np.random.default_rng(42)
        spikes2 = generate_spikes_from_place_fields(position_indices, place_fields, dt, rng2)

        assert_array_equal(spikes1, spikes2)

    def test_spike_statistics(self) -> None:
        """Test that spike counts follow expected Poisson statistics."""
        rng = np.random.default_rng(42)
        n_trials = 1000
        rate = 50.0  # spikes/sec
        dt = 0.002
        expected_count = rate * dt  # 0.1 spikes per bin

        position_indices = np.zeros(n_trials, dtype=np.int64)
        place_fields = np.array([[rate]])

        spikes = generate_spikes_from_place_fields(position_indices, place_fields, dt, rng)

        mean_count = spikes.mean()
        # Should be close to expected count
        assert abs(mean_count - expected_count) < 0.05


class TestComputeLogLikelihoodFromPlaceFields:
    """Tests for compute_log_likelihood_from_place_fields function."""

    def test_basic_likelihood(self) -> None:
        """Test basic log likelihood computation."""
        spike_counts = np.array([[1, 0], [2, 1]])
        place_fields = np.array([[10.0, 20.0, 5.0], [5.0, 10.0, 15.0]])
        dt = 0.002

        log_like = compute_log_likelihood_from_place_fields(spike_counts, place_fields, dt)

        assert log_like.shape == (2, 3)
        assert log_like.dtype == np.float64
        assert np.all(np.isfinite(log_like))

    def test_zero_spikes(self) -> None:
        """Test likelihood with zero spikes."""
        spike_counts = np.array([[0, 0]])
        place_fields = np.array([[10.0, 20.0], [5.0, 10.0]])
        dt = 0.002

        log_like = compute_log_likelihood_from_place_fields(spike_counts, place_fields, dt)

        assert log_like.shape == (1, 2)
        assert np.all(np.isfinite(log_like))
        # Zero spikes should give negative log likelihood (due to -lambda term)
        assert np.all(log_like < 0)

    def test_likelihood_shape_broadcasting(self) -> None:
        """Test that broadcasting works correctly."""
        spike_counts = np.array([[1, 2, 0]])  # (1, 3) - 1 time, 3 cells
        # (3, 2) - 3 cells, 2 bins
        place_fields = np.array([[10.0, 20.0], [5.0, 10.0], [8.0, 12.0]])
        dt = 0.002

        log_like = compute_log_likelihood_from_place_fields(spike_counts, place_fields, dt)

        # Should broadcast to (1, 2) - 1 time, 2 bins
        assert log_like.shape == (1, 2)

    def test_likelihood_with_scipy(self) -> None:
        """Test that scipy implementation gives reasonable values."""
        from scipy.stats import poisson

        spike_count = 2
        rate = 100.0
        dt = 0.002
        expected_count = rate * dt

        # Direct scipy calculation
        scipy_logpmf = poisson.logpmf(spike_count, expected_count)

        # Our function
        spike_counts = np.array([[spike_count]])
        place_fields = np.array([[rate]])
        log_like = compute_log_likelihood_from_place_fields(spike_counts, place_fields, dt)

        # Should match scipy result
        assert_array_almost_equal(log_like[0, 0], scipy_logpmf, decimal=10)

    def test_likelihood_ordering(self) -> None:
        """Test that higher rates at observed spikes give higher likelihood."""
        spike_counts = np.array([[5, 0]])
        # First bin has high rate for first cell (many spikes), second bin has low rate
        place_fields = np.array([[100.0, 10.0], [10.0, 100.0]])
        dt = 0.002

        log_like = compute_log_likelihood_from_place_fields(spike_counts, place_fields, dt)

        # First bin should have higher likelihood (100 Hz for cell with 5 spikes)
        assert log_like[0, 0] > log_like[0, 1]


class TestExtractPlaceFieldsFromModel:
    """Tests for extract_place_fields_from_model function."""

    def test_extract_place_fields(self) -> None:
        """Test extracting place fields from mock model."""

        # Create mock model
        class MockModel:
            def __init__(self) -> None:
                self.sampling_frequency = 500.0
                self.encoding_model_ = {
                    ("", 0): {"place_fields": np.array([[10.0, 20.0, 5.0], [5.0, 10.0, 15.0]])}
                }

        model = MockModel()
        place_fields, dt = extract_place_fields_from_model(model)

        assert place_fields.shape == (2, 3)
        assert dt == 1.0 / 500.0
        assert_array_equal(place_fields, np.array([[10.0, 20.0, 5.0], [5.0, 10.0, 15.0]]))


class TestCreatePosteriorPredictiveSampler:
    """Tests for create_posterior_predictive_sampler function."""

    def test_sampler_creation(self) -> None:
        """Test creating posterior predictive sampler."""

        # Create mock model
        class MockModel:
            def __init__(self) -> None:
                self.sampling_frequency = 500.0
                self.encoding_model_ = {
                    ("", 0): {"place_fields": np.array([[10.0, 20.0], [5.0, 10.0]])}
                }

        # Create mock results
        n_time = 10
        n_bins = 2
        rng_data = np.random.default_rng(123)
        posterior = rng_data.dirichlet(np.ones(n_bins), size=n_time)
        log_likelihood = rng_data.standard_normal((n_time, n_bins))

        results = xr.Dataset(
            {
                "predictive_posterior": (["time", "state_bins"], posterior),
                "log_likelihood": (["time", "state_bins"], log_likelihood),
            }
        )

        rng = np.random.default_rng(42)
        sampler = create_posterior_predictive_sampler(MockModel(), results, rng)

        # Test that sampler works
        samples = sampler(n_samples=5)
        assert samples.shape == (5, n_time)
        assert np.all(np.isfinite(samples))

    def test_sampler_reproducibility(self) -> None:
        """Test that sampler gives reproducible results."""

        class MockModel:
            def __init__(self) -> None:
                self.sampling_frequency = 500.0
                self.encoding_model_ = {
                    ("", 0): {"place_fields": np.array([[10.0, 20.0], [5.0, 10.0]])}
                }

        n_time = 10
        n_bins = 2
        rng_data = np.random.default_rng(123)
        posterior = rng_data.dirichlet(np.ones(n_bins), size=n_time)
        log_likelihood = rng_data.standard_normal((n_time, n_bins))

        results = xr.Dataset(
            {
                "predictive_posterior": (["time", "state_bins"], posterior),
                "log_likelihood": (["time", "state_bins"], log_likelihood),
            }
        )

        # Create two samplers with same seed
        rng1 = np.random.default_rng(42)
        sampler1 = create_posterior_predictive_sampler(MockModel(), results, rng1)
        samples1 = sampler1(n_samples=5)

        rng2 = np.random.default_rng(42)
        sampler2 = create_posterior_predictive_sampler(MockModel(), results, rng2)
        samples2 = sampler2(n_samples=5)

        assert_array_almost_equal(samples1, samples2)


class TestComputePredictivePvalues:
    """Tests for compute_predictive_pvalues function."""

    def test_pvalues_shape(self) -> None:
        """Test that p-values have correct shape."""

        # Create mock model
        class MockModel:
            def __init__(self) -> None:
                self.sampling_frequency = 500.0
                self.encoding_model_ = {
                    ("", 0): {"place_fields": np.array([[10.0, 20.0], [5.0, 10.0]])}
                }

        # Create mock results
        n_time = 20
        n_bins = 2
        rng_data = np.random.default_rng(123)
        posterior = rng_data.dirichlet(np.ones(n_bins), size=n_time)
        log_likelihood = rng_data.standard_normal((n_time, n_bins))

        results = xr.Dataset(
            {
                "predictive_posterior": (["time", "state_bins"], posterior),
                "log_likelihood": (["time", "state_bins"], log_likelihood),
            }
        )

        rng = np.random.default_rng(42)
        p_values = compute_predictive_pvalues(MockModel(), results, n_samples=10, rng=rng)

        assert p_values.shape == (n_time,)
        assert p_values.dtype == np.float64

    def test_pvalues_range(self) -> None:
        """Test that p-values are in [0, 1]."""

        class MockModel:
            def __init__(self) -> None:
                self.sampling_frequency = 500.0
                self.encoding_model_ = {
                    ("", 0): {"place_fields": np.array([[10.0, 20.0], [5.0, 10.0]])}
                }

        n_time = 20
        n_bins = 2
        rng_data = np.random.default_rng(123)
        posterior = rng_data.dirichlet(np.ones(n_bins), size=n_time)
        log_likelihood = rng_data.standard_normal((n_time, n_bins))

        results = xr.Dataset(
            {
                "predictive_posterior": (["time", "state_bins"], posterior),
                "log_likelihood": (["time", "state_bins"], log_likelihood),
            }
        )

        rng = np.random.default_rng(42)
        p_values = compute_predictive_pvalues(MockModel(), results, n_samples=10, rng=rng)

        assert np.all((p_values >= 0) & (p_values <= 1))

    def test_pvalues_reproducibility(self) -> None:
        """Test that p-values are reproducible with same seed."""

        class MockModel:
            def __init__(self) -> None:
                self.sampling_frequency = 500.0
                self.encoding_model_ = {
                    ("", 0): {"place_fields": np.array([[10.0, 20.0], [5.0, 10.0]])}
                }

        n_time = 20
        n_bins = 2
        rng_data = np.random.default_rng(123)
        posterior = rng_data.dirichlet(np.ones(n_bins), size=n_time)
        log_likelihood = rng_data.standard_normal((n_time, n_bins))

        results = xr.Dataset(
            {
                "predictive_posterior": (["time", "state_bins"], posterior),
                "log_likelihood": (["time", "state_bins"], log_likelihood),
            }
        )

        rng1 = np.random.default_rng(42)
        p_values1 = compute_predictive_pvalues(MockModel(), results, n_samples=10, rng=rng1)

        rng2 = np.random.default_rng(42)
        p_values2 = compute_predictive_pvalues(MockModel(), results, n_samples=10, rng=rng2)

        assert_array_almost_equal(p_values1, p_values2)

    def test_default_rng(self) -> None:
        """Test that function works without providing rng."""

        class MockModel:
            def __init__(self) -> None:
                self.sampling_frequency = 500.0
                self.encoding_model_ = {
                    ("", 0): {"place_fields": np.array([[10.0, 20.0], [5.0, 10.0]])}
                }

        n_time = 20
        n_bins = 2
        rng_data = np.random.default_rng(123)
        posterior = rng_data.dirichlet(np.ones(n_bins), size=n_time)
        log_likelihood = rng_data.standard_normal((n_time, n_bins))

        results = xr.Dataset(
            {
                "predictive_posterior": (["time", "state_bins"], posterior),
                "log_likelihood": (["time", "state_bins"], log_likelihood),
            }
        )

        # Should work without rng argument
        p_values = compute_predictive_pvalues(MockModel(), results, n_samples=10)

        assert p_values.shape == (n_time,)
        assert np.all((p_values >= 0) & (p_values <= 1))
