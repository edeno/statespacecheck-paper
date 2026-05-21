"""Tests for simulation utilities."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from statespacecheck_paper.simulation import (
    gaussian_transition_matrix,
    normalize,
    placefield_rates,
    reflect_into_interval,
    safe_log,
    simulate_spikes_flat_rate,
    simulate_spikes_history_dependent,
    simulate_spikes_position_tuned,
    simulate_walk,
    spike_prob_rank,
)

# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_normalize_1d_array(self) -> None:
        result = normalize(np.array([1.0, 2.0, 3.0]))
        assert_allclose(result, [1 / 6, 2 / 6, 3 / 6])
        assert_allclose(result.sum(), 1.0)

    @pytest.mark.parametrize("axis", [0, 1])
    def test_normalize_2d_along_axis(self, axis: int) -> None:
        p = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = normalize(p, axis=axis)
        assert_allclose(result.sum(axis=axis), [1.0, 1.0])

    def test_normalize_zeros_uses_eps_to_avoid_nan_and_warns(self) -> None:
        """Zero-sum input must produce finite output (uses eps internally) and
        must emit a RuntimeWarning so the situation isn't silent."""
        with pytest.warns(RuntimeWarning, match="normalize: input sum"):
            result = normalize(np.zeros(5))
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# reflect_into_interval
# ---------------------------------------------------------------------------


class TestReflectIntoInterval:
    @pytest.mark.parametrize(
        ("x", "lower", "upper", "expected"),
        [
            (np.array([0.5, 1.0, 1.5]), 0.0, 2.0, np.array([0.5, 1.0, 1.5])),
            (np.array([2.5]), 0.0, 2.0, np.array([1.5])),
            (np.array([-0.5]), 0.0, 2.0, np.array([0.5])),
            # Reflection at the exact boundary (edge case)
            (np.array([2.0]), 0.0, 2.0, np.array([2.0])),
            (np.array([0.0]), 0.0, 2.0, np.array([0.0])),
        ],
    )
    def test_known_reflections(
        self,
        x: np.ndarray,
        lower: float,
        upper: float,
        expected: np.ndarray,
    ) -> None:
        assert_allclose(reflect_into_interval(x, lower, upper), expected)

    def test_multiple_reflections_stay_in_bounds(self) -> None:
        """Values many bounds away should still land inside the interval."""
        result = reflect_into_interval(np.array([4.5, -7.3, 100.0]), 0.0, 2.0)
        assert ((result >= 0.0) & (result <= 2.0)).all()


# ---------------------------------------------------------------------------
# gaussian_transition_matrix
# ---------------------------------------------------------------------------


class TestGaussianTransitionMatrix:
    def test_shape(self) -> None:
        matrix = gaussian_transition_matrix(np.array([0.0, 1.0, 2.0]), sig=1.0)
        assert matrix.shape == (3, 3)

    def test_columns_sum_to_one(self) -> None:
        matrix = gaussian_transition_matrix(np.linspace(0, 10, 11), sig=1.0)
        assert_allclose(matrix.sum(axis=0), np.ones(11), rtol=1e-10)

    def test_diagonal_dominant_for_small_sigma(self) -> None:
        matrix = gaussian_transition_matrix(np.array([0.0, 1.0, 2.0]), sig=0.1)
        for i in range(3):
            assert matrix[i, i] == matrix[:, i].max()

    def test_larger_sigma_spreads_probability(self) -> None:
        xs = np.array([0.0, 1.0, 2.0])
        narrow = gaussian_transition_matrix(xs, sig=0.1)
        wide = gaussian_transition_matrix(xs, sig=2.0)
        assert narrow[0, 0] > wide[0, 0]

    def test_single_bin_is_identity(self) -> None:
        """Edge case: single-bin grid yields a 1x1 matrix that sums to 1."""
        matrix = gaussian_transition_matrix(np.array([0.0]), sig=1.0)
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# safe_log
# ---------------------------------------------------------------------------


class TestSafeLog:
    def test_matches_log_for_positive_values(self) -> None:
        x = np.array([1.0, 2.0, np.e])
        assert_allclose(safe_log(x), np.log(x))

    def test_zero_yields_finite_log_eps(self) -> None:
        result = safe_log(np.array([0.0, 1.0, 2.0]))
        assert np.isfinite(result).all()

    @pytest.mark.parametrize("eps", [1e-6, 1e-12])
    def test_zero_uses_eps(self, eps: float) -> None:
        assert_allclose(safe_log(np.array([0.0]), eps=eps), np.log(eps))


# ---------------------------------------------------------------------------
# placefield_rates
# ---------------------------------------------------------------------------


class TestPlacefieldRates:
    def test_shape(self) -> None:
        rates = placefield_rates(
            np.linspace(0, 10, 11), np.array([2.0, 5.0, 8.0]), width=1.0, scale=1.0
        )
        assert rates.shape == (11, 3)

    def test_peak_is_at_place_field_center(self) -> None:
        xs = np.linspace(0, 10, 101)
        rates = placefield_rates(xs, np.array([5.0]), width=1.0, scale=1.0)
        center_idx = int(np.argmin(np.abs(xs - 5.0)))
        assert rates[center_idx, 0] == rates[:, 0].max()

    def test_scale_multiplies_rates(self) -> None:
        xs = np.linspace(0, 10, 11)
        centers = np.array([5.0])
        baseline = placefield_rates(xs, centers, width=1.0, scale=1.0)
        doubled = placefield_rates(xs, centers, width=1.0, scale=2.0)
        assert_allclose(doubled, 2.0 * baseline)

    def test_empty_centers_yields_empty_columns(self) -> None:
        """Edge case: no cells -> shape (n_bins, 0) without error."""
        rates = placefield_rates(np.linspace(0, 10, 11), np.array([]), width=1.0, scale=1.0)
        assert rates.shape == (11, 0)


# ---------------------------------------------------------------------------
# spike_prob_rank
# ---------------------------------------------------------------------------


def _normalize_lambda(lam: np.ndarray) -> np.ndarray:
    return lam / lam.sum(axis=0, keepdims=True)


class TestSpikeProbRank:
    def test_single_timestep_shape(self) -> None:
        prior = np.array([0.5, 0.3, 0.2])
        lam = _normalize_lambda(np.array([[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]]))
        assert spike_prob_rank(prior, lam).shape == (2,)

    def test_values_in_unit_range(self) -> None:
        prior = np.array([0.5, 0.3, 0.2])
        lam = _normalize_lambda(np.array([[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]]))
        result = spike_prob_rank(prior, lam)
        assert ((result >= 0.0) & (result <= 1.0)).all()

    def test_uniform_contribution_yields_equal_ranks(self) -> None:
        n_bins, n_cells = 10, 5
        prior = np.ones(n_bins) / n_bins
        lam = np.ones((n_bins, n_cells)) / n_bins
        result = spike_prob_rank(prior, lam)
        # All cells equal => all ranks equal.
        assert_allclose(result, result[0])

    def test_batched_shape(self, rng: np.random.Generator) -> None:
        n_time, n_bins, n_cells = 10, 5, 3
        prior = rng.dirichlet(np.ones(n_bins), size=n_time)
        lam = np.ones((n_bins, n_cells)) / n_bins
        assert spike_prob_rank(prior, lam).shape == (n_time, n_cells)

    def test_batched_matches_per_timestep_loop(self, rng: np.random.Generator) -> None:
        """Batched and looped paths must produce identical results."""
        n_time, n_bins, n_cells = 10, 5, 3
        prior = rng.dirichlet(np.ones(n_bins), size=n_time)
        lam = _normalize_lambda(rng.random((n_bins, n_cells)))

        batched = spike_prob_rank(prior, lam)
        looped = np.stack([spike_prob_rank(prior[t], lam) for t in range(n_time)])
        assert_allclose(batched, looped)


# ---------------------------------------------------------------------------
# simulate_walk
# ---------------------------------------------------------------------------


class TestSimulateWalk:
    def test_shape(self, rng: np.random.Generator) -> None:
        result = simulate_walk(100, sig=1.0, x0=50.0, xs_min=0.0, xs_max=100.0, rng=rng)
        assert result.shape == (100,)

    def test_respects_boundaries_under_large_steps(self, rng: np.random.Generator) -> None:
        result = simulate_walk(1000, sig=5.0, x0=50.0, xs_min=0.0, xs_max=100.0, rng=rng)
        assert ((result >= 0.0) & (result <= 100.0)).all()

    def test_reproducible_with_same_seed(self) -> None:
        a = simulate_walk(
            n_time=100,
            sig=1.0,
            x0=50.0,
            xs_min=0.0,
            xs_max=100.0,
            rng=np.random.default_rng(42),
        )
        b = simulate_walk(
            n_time=100,
            sig=1.0,
            x0=50.0,
            xs_min=0.0,
            xs_max=100.0,
            rng=np.random.default_rng(42),
        )
        assert_array_equal(a, b)

    def test_zero_step_size_stays_at_initial(self, rng: np.random.Generator) -> None:
        result = simulate_walk(100, sig=0.0, x0=50.0, xs_min=0.0, xs_max=100.0, rng=rng)
        assert_allclose(result, 50.0)

    def test_larger_sigma_increases_variance(self) -> None:
        narrow = simulate_walk(
            1000,
            sig=0.5,
            x0=50.0,
            xs_min=0.0,
            xs_max=100.0,
            rng=np.random.default_rng(42),
        )
        wide = simulate_walk(
            1000,
            sig=5.0,
            x0=50.0,
            xs_min=0.0,
            xs_max=100.0,
            rng=np.random.default_rng(43),
        )
        assert wide.std() > narrow.std()


# ---------------------------------------------------------------------------
# simulate_spikes_position_tuned
# ---------------------------------------------------------------------------


class TestSimulateSpikesPositionTuned:
    def test_shape_and_dtype(self, rng: np.random.Generator) -> None:
        result = simulate_spikes_position_tuned(
            np.linspace(0, 100, 100),
            np.array([25.0, 50.0, 75.0]),
            pf_width=5.0,
            rate_scale=0.1,
            rng=rng,
        )
        assert result.shape == (100, 3)
        assert (result >= 0).all()
        assert np.issubdtype(result.dtype, np.integer)

    def test_higher_rate_near_place_field_center(self) -> None:
        """Long trajectory: spikes near pf center > spikes far from it."""
        rng = np.random.default_rng(42)
        x = np.linspace(0, 100, 10000)
        result = simulate_spikes_position_tuned(
            x, np.array([50.0]), pf_width=5.0, rate_scale=1.0, rng=rng
        )
        near = result[np.abs(x - 50.0) < 10.0, 0].mean()
        far = result[np.abs(x - 50.0) > 40.0, 0].mean()
        assert near > far

    def test_reproducible_with_same_seed(self) -> None:
        x = np.linspace(0, 100, 100)
        pf_centers = np.array([50.0])
        a = simulate_spikes_position_tuned(
            x,
            pf_centers,
            pf_width=5.0,
            rate_scale=0.1,
            rng=np.random.default_rng(42),
        )
        b = simulate_spikes_position_tuned(
            x,
            pf_centers,
            pf_width=5.0,
            rate_scale=0.1,
            rng=np.random.default_rng(42),
        )
        assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# simulate_spikes_flat_rate
# ---------------------------------------------------------------------------


class TestSimulateSpikesFlatRate:
    def test_shape_and_dtype(self, rng: np.random.Generator) -> None:
        result = simulate_spikes_flat_rate(100, 5, rate=0.1, rng=rng)
        assert result.shape == (100, 5)
        assert (result >= 0).all()
        assert np.issubdtype(result.dtype, np.integer)

    def test_mean_close_to_specified_rate(self, rng: np.random.Generator) -> None:
        rate = 2.0
        result = simulate_spikes_flat_rate(10000, 10, rate=rate, rng=rng)
        assert_allclose(result.mean(), rate, rtol=0.1)

    def test_reproducible_with_same_seed(self) -> None:
        a = simulate_spikes_flat_rate(100, 5, rate=0.1, rng=np.random.default_rng(42))
        b = simulate_spikes_flat_rate(100, 5, rate=0.1, rng=np.random.default_rng(42))
        assert_array_equal(a, b)

    def test_zero_rate_yields_no_spikes(self, rng: np.random.Generator) -> None:
        """Edge case: rate=0 must produce only zeros."""
        result = simulate_spikes_flat_rate(100, 5, rate=0.0, rng=rng)
        assert (result == 0).all()


# ---------------------------------------------------------------------------
# simulate_spikes_history_dependent
# ---------------------------------------------------------------------------


class TestSimulateSpikesHistoryDependent:
    def test_shape_and_dtype(self, rng: np.random.Generator) -> None:
        x = np.full(500, 50.0)
        pf = np.array([20.0, 50.0, 80.0])
        spikes = simulate_spikes_history_dependent(x, pf, 10.0, 5.0, rng)
        assert spikes.shape == (500, 3)
        assert (spikes >= 0).all()
        assert np.issubdtype(spikes.dtype, np.integer)

    def test_hard_refractory_suppresses_adjacent_spikes(self, rng: np.random.Generator) -> None:
        """A 1-step refractory means no cell fires on two consecutive steps.

        Driven hard (high rate, animal parked on a place-field center) so
        that without the refractory adjacent spikes would be common.
        """
        x = np.full(2000, 50.0)  # parked on cell 1's PF center
        pf = np.array([20.0, 50.0, 80.0])
        spikes = simulate_spikes_history_dependent(
            x, pf, pf_width=10.0, rate_scale=50.0, rng=rng, refractory_steps=1
        )
        fired = spikes > 0
        adjacent = fired[1:] & fired[:-1]
        assert not adjacent.any(), (
            f"{int(adjacent.sum())} adjacent-step spike pairs survived a 1-step refractory"
        )

    def test_longer_refractory_enforces_minimum_gap(self, rng: np.random.Generator) -> None:
        """With refractory_steps=3, every inter-spike interval is >= 3."""
        x = np.full(3000, 50.0)
        pf = np.array([50.0])
        spikes = simulate_spikes_history_dependent(
            x, pf, pf_width=10.0, rate_scale=50.0, rng=rng, refractory_steps=3
        )
        spike_steps = np.flatnonzero(spikes[:, 0] > 0)
        if spike_steps.size > 1:
            assert np.diff(spike_steps).min() >= 3

    def test_burst_window_elevates_post_spike_firing(self, rng: np.random.Generator) -> None:
        """Firing probability in the burst window exceeds firing well
        outside it (post-refractory, pre-burst-decay)."""
        x = np.full(20000, 50.0)
        pf = np.array([50.0])
        spikes = simulate_spikes_history_dependent(
            x,
            pf,
            pf_width=10.0,
            rate_scale=5.0,
            rng=rng,
            refractory_steps=1,
            burst_window=(2, 10),
            burst_factor=5.0,
        )[:, 0]
        spike_steps = np.flatnonzero(spikes > 0)
        # For each spike, was there a spike 2-10 steps later (burst window)
        # vs. 30-100 steps later (well outside)?
        fired = spikes > 0
        in_burst = np.zeros(spikes.size, dtype=bool)
        far = np.zeros(spikes.size, dtype=bool)
        for s in spike_steps:
            in_burst[s + 2 : s + 11] = True
            far[s + 30 : s + 101] = True
        burst_rate = fired[in_burst].mean()
        far_rate = fired[far].mean()
        assert burst_rate > far_rate, (
            f"burst-window firing {burst_rate:.4f} should exceed "
            f"far-from-spike firing {far_rate:.4f}"
        )

    def test_reproducible_with_same_seed(self) -> None:
        x = np.full(300, 50.0)
        pf = np.array([20.0, 50.0, 80.0])
        a = simulate_spikes_history_dependent(x, pf, 10.0, 5.0, np.random.default_rng(7))
        b = simulate_spikes_history_dependent(x, pf, 10.0, 5.0, np.random.default_rng(7))
        assert_array_equal(a, b)

    def test_empty_timeline_returns_empty(self, rng: np.random.Generator) -> None:
        spikes = simulate_spikes_history_dependent(np.empty(0), np.array([50.0]), 10.0, 5.0, rng)
        assert spikes.shape == (0, 1)

    def test_zero_rate_scale_yields_no_spikes(self, rng: np.random.Generator) -> None:
        x = np.full(200, 50.0)
        spikes = simulate_spikes_history_dependent(x, np.array([50.0]), 10.0, 0.0, rng)
        assert (spikes == 0).all()

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"refractory_steps": 0}, "refractory_steps must be >= 1"),
            ({"burst_window": (5, 2)}, "burst_window must satisfy"),
            ({"burst_window": (-1, 3)}, "burst_window must satisfy"),
            ({"burst_factor": 0.0}, "burst_factor must be positive"),
        ],
    )
    def test_invalid_parameters_raise(
        self, rng: np.random.Generator, kwargs: dict, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            simulate_spikes_history_dependent(
                np.full(50, 50.0), np.array([50.0]), 10.0, 5.0, rng, **kwargs
            )
