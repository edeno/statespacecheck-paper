"""Simulation utilities for state space models.

This module provides utilities for simulating random walks, spike trains, and
computing likelihood functions for position decoding in neuroscience applications.

Key Components
--------------
- **Normalization**: Safe normalization of probability distributions
- **Boundary conditions**: Reflecting boundary conditions for random walks
- **Transition matrices**: Gaussian transition matrices for state space models
- **Place fields**: Gaussian place field models for spatial tuning
- **Spike generation**: Poisson spike generation for position-tuned and flat-rate neurons
- **Likelihood computation**: Functions for computing spike probability rankings

Examples
--------
Simulate a random walk with reflecting boundaries:

>>> import numpy as np
>>> rng = np.random.default_rng(42)
>>> walk = simulate_walk(n_time=100, sig=1.0, x0=50.0,
...                      xs_min=0.0, xs_max=100.0, rng=rng)
>>> walk.shape
(100,)
>>> (walk >= 0.0).all() and (walk <= 100.0).all()
True

Generate position-tuned spikes:

>>> x = np.linspace(0, 100, 100)
>>> pf_centers = np.array([25.0, 50.0, 75.0])
>>> spikes = simulate_spikes_position_tuned(x, pf_centers, pf_width=5.0,
...                                         rate_scale=0.1, rng=rng)
>>> spikes.shape
(100, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


def normalize(
    p: NDArray[np.floating], axis: int | None = None, eps: float = 1e-12
) -> NDArray[np.floating]:
    """Normalize array to sum to 1 along specified axis with numerical safety.

    Parameters
    ----------
    p : np.ndarray
        Array to normalize.
    axis : int or None, optional
        Axis along which to normalize. If None, normalizes entire array.
    eps : float, default 1e-12
        Small epsilon to prevent division by zero.

    Returns
    -------
    normalized : np.ndarray
        Normalized array with same shape as input, where sum along axis equals 1.

    Examples
    --------
    Normalize a 1D probability distribution:

    >>> p = np.array([1.0, 2.0, 3.0])
    >>> result = normalize(p)
    >>> np.allclose(result, [1/6, 2/6, 3/6])
    True
    >>> np.allclose(result.sum(), 1.0)
    True

    Normalize columns of a 2D array:

    >>> p = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> result = normalize(p, axis=0)
    >>> np.allclose(result.sum(axis=0), [1.0, 1.0])
    True
    """
    s: NDArray[np.floating] = np.sum(p, axis=axis, keepdims=True)
    s = np.maximum(s, eps)
    result: NDArray[np.floating] = p / s
    return result


def reflect_into_interval(x: NDArray[np.floating], lo: float, hi: float) -> NDArray[np.floating]:
    """Reflect values into interval [lo, hi] using triangle wave method.

    This implements reflecting boundary conditions for random walks by treating
    the walk as a triangle wave that bounces off the boundaries.

    Parameters
    ----------
    x : np.ndarray
        Array of values to reflect.
    lo : float
        Lower bound of interval.
    hi : float
        Upper bound of interval.

    Returns
    -------
    reflected : np.ndarray
        Array of same shape as x with all values in [lo, hi].

    Examples
    --------
    Reflect values outside bounds:

    >>> x = np.array([-1.0, 0.5, 2.5])
    >>> result = reflect_into_interval(x, 0.0, 2.0)
    >>> (result >= 0.0).all() and (result <= 2.0).all()
    True

    Values inside bounds are unchanged:

    >>> x = np.array([0.5, 1.0, 1.5])
    >>> result = reflect_into_interval(x, 0.0, 2.0)
    >>> np.allclose(result, x)
    True
    """
    length = hi - lo
    y: NDArray[np.floating] = np.mod(x - lo, 2 * length)
    y = np.where(y <= length, y, 2 * length - y)
    result: NDArray[np.floating] = y + lo
    return result


def gaussian_transition_matrix(xs: NDArray[np.floating], sig: float) -> NDArray[np.floating]:
    """Compute one-step Gaussian transition matrix for random walk.

    Computes transition probabilities for Gaussian random walk on discrete grid.
    Each column represents a probability distribution over next states given
    current state.

    Parameters
    ----------
    xs : np.ndarray, shape (n_bins,)
        Grid of position values.
    sig : float
        Standard deviation of Gaussian transition kernel.

    Returns
    -------
    transition_matrix : np.ndarray, shape (n_bins, n_bins)
        Transition matrix where element [i, j] is probability of transitioning
        to state i given current state j. Each column sums to 1.

    Examples
    --------
    Create transition matrix for 3-state system:

    >>> xs = np.array([0.0, 1.0, 2.0])
    >>> matrix = gaussian_transition_matrix(xs, sig=1.0)
    >>> matrix.shape
    (3, 3)
    >>> np.allclose(matrix.sum(axis=0), 1.0)  # Columns sum to 1
    True
    """
    diff = xs[:, None] - xs[None, :]
    matrix = norm.pdf(diff, loc=0.0, scale=sig)
    return normalize(matrix, axis=0)  # columns sum to 1


def safe_log(x: NDArray[np.floating], eps: float = 1e-12) -> NDArray[np.floating]:
    """Compute log(x) with numerical safety to avoid log(0).

    Parameters
    ----------
    x : np.ndarray
        Array of values.
    eps : float, default 1e-12
        Small epsilon added to prevent log(0).

    Returns
    -------
    log_x : np.ndarray
        Array of log values with same shape as input.

    Examples
    --------
    Safe log handles zeros:

    >>> x = np.array([0.0, 1.0, 2.0])
    >>> result = safe_log(x)
    >>> np.isfinite(result).all()  # No -inf values
    True
    """
    result: NDArray[np.floating] = np.log(np.maximum(x, eps))
    return result


def placefield_rates(
    xs: NDArray[np.floating], centers: NDArray[np.floating], width: float, scale: float
) -> NDArray[np.floating]:
    """Compute Gaussian place field firing rates.

    Computes firing rate for each neuron at each position using Gaussian place
    field model. Each neuron has a place field centered at one location with
    specified width.

    Parameters
    ----------
    xs : np.ndarray, shape (n_bins,)
        Position bin centers.
    centers : np.ndarray, shape (n_cells,)
        Place field center for each neuron.
    width : float
        Standard deviation of Gaussian place field.
    scale : float
        Peak firing rate (scales the Gaussian).

    Returns
    -------
    rates : np.ndarray, shape (n_bins, n_cells)
        Firing rate for each position bin and neuron.

    Examples
    --------
    Compute place field rates for 3 neurons:

    >>> xs = np.linspace(0, 10, 11)
    >>> centers = np.array([2.0, 5.0, 8.0])
    >>> rates = placefield_rates(xs, centers, width=1.0, scale=1.0)
    >>> rates.shape
    (11, 3)
    >>> rates.max(axis=0)  # Peak at each center
    array([0.398..., 0.398..., 0.398...])
    """
    result: NDArray[np.floating] = norm.pdf(xs[:, None], loc=centers[None, :], scale=width) * scale
    return result


def spike_prob_rank(
    prior: NDArray[np.floating],
    lambda_ratio: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute cumulative probability mass of cells with low expected contribution.

    This function computes the probability ranking for each neuron based on its
    expected contribution to the likelihood. For each cell, it computes the
    cumulative probability mass of all cells with equal or lower contribution.

    Parameters
    ----------
    prior : np.ndarray, shape (n_bins,)
        Prior probability distribution over position bins.
    lambda_ratio : np.ndarray, shape (n_bins, n_cells)
        Normalized likelihood contributions where each row sums to 1.

    Returns
    -------
    spike_probs : np.ndarray, shape (n_cells,)
        For each cell, the cumulative probability mass of cells with
        contribution <= that cell's contribution. Values in [0, 1].

    Examples
    --------
    Compute spike probability ranks:

    >>> prior = np.array([0.5, 0.3, 0.2])
    >>> lambda_ratio = np.array([[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]])
    >>> lambda_ratio = lambda_ratio / lambda_ratio.sum(axis=0, keepdims=True)
    >>> ranks = spike_prob_rank(prior, lambda_ratio)
    >>> ranks.shape
    (2,)
    >>> (ranks >= 0.0).all() and (ranks <= 1.0).all()
    True

    Notes
    -----
    This matches the MATLAB implementation:
    sum(lambda_expect(lambda_expect <= lambda_expect(j)))
    where lambda_expect are probabilities summing to 1.
    """
    contrib: NDArray[np.floating] = prior @ lambda_ratio  # (n_cells,) - probabilities summing to ~1
    # Sum probability mass of cells with contribution <= each cell's contribution
    mask = contrib[:, None] <= contrib[None, :]  # (n_cells, n_cells)
    # Sum probabilities, not counts
    spike_probs: NDArray[np.floating] = (contrib[None, :] * mask).sum(axis=1)
    return spike_probs


def simulate_walk(
    n_time: int,
    sig: float,
    x0: float,
    xs_min: float,
    xs_max: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Simulate random walk with reflecting boundary conditions.

    Simulates a Gaussian random walk on continuous space with reflecting
    boundaries. The walk starts at x0 and takes steps drawn from a Gaussian
    distribution with standard deviation sig.

    Parameters
    ----------
    n_time : int
        Number of time steps to simulate.
    sig : float
        Standard deviation of step size distribution.
    x0 : float
        Initial position.
    xs_min : float
        Lower boundary (reflecting).
    xs_max : float
        Upper boundary (reflecting).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    trajectory : np.ndarray, shape (n_time,)
        Simulated trajectory with all values in [xs_min, xs_max].

    Examples
    --------
    Simulate a 100-step random walk:

    >>> rng = np.random.default_rng(42)
    >>> walk = simulate_walk(100, sig=1.0, x0=50.0, xs_min=0.0, xs_max=100.0, rng=rng)
    >>> walk.shape
    (100,)
    >>> (walk >= 0.0).all() and (walk <= 100.0).all()
    True

    With zero step size, trajectory is constant:

    >>> rng = np.random.default_rng(42)
    >>> walk = simulate_walk(10, sig=0.0, x0=50.0, xs_min=0.0, xs_max=100.0, rng=rng)
    >>> np.allclose(walk, 50.0)
    True
    """
    steps = rng.normal(loc=0.0, scale=sig, size=n_time)
    x = x0 + np.cumsum(steps)
    return reflect_into_interval(x, xs_min, xs_max)


def simulate_spikes_position_tuned(
    x: NDArray[np.floating],
    pf_centers: NDArray[np.floating],
    pf_width: float,
    rate_scale: float,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Simulate Poisson spikes for position-tuned neurons.

    Generates spike counts from Poisson distribution with position-dependent
    firing rates. Each neuron has a Gaussian place field determining its
    firing rate at each position.

    Parameters
    ----------
    x : np.ndarray, shape (n_time,)
        Position at each time step.
    pf_centers : np.ndarray, shape (n_cells,)
        Place field center for each neuron.
    pf_width : float
        Standard deviation of Gaussian place field.
    rate_scale : float
        Peak firing rate (scales the Gaussian).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    spikes : np.ndarray, shape (n_time, n_cells)
        Spike counts for each time step and neuron (non-negative integers).

    Examples
    --------
    Simulate spikes for 3 neurons:

    >>> rng = np.random.default_rng(42)
    >>> x = np.linspace(0, 100, 100)
    >>> pf_centers = np.array([25.0, 50.0, 75.0])
    >>> spikes = simulate_spikes_position_tuned(x, pf_centers, pf_width=5.0,
    ...                                         rate_scale=0.1, rng=rng)
    >>> spikes.shape
    (100, 3)
    >>> (spikes >= 0).all()
    True
    """
    lam = norm.pdf(x[:, None], loc=pf_centers[None, :], scale=pf_width) * rate_scale
    spikes: NDArray[np.int_] = rng.poisson(lam)
    return spikes


def simulate_spikes_flat_rate(
    n_time: int, n_cells: int, rate: float, rng: np.random.Generator
) -> NDArray[np.int_]:
    """Simulate Poisson spikes with flat (non-position-tuned) firing rate.

    Generates spike counts from Poisson distribution with constant firing rate
    across all positions. Useful for simulating background activity or
    non-spatial neurons.

    Parameters
    ----------
    n_time : int
        Number of time steps.
    n_cells : int
        Number of neurons.
    rate : float
        Constant firing rate (spikes per time bin).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    spikes : np.ndarray, shape (n_time, n_cells)
        Spike counts for each time step and neuron (non-negative integers).

    Examples
    --------
    Simulate flat-rate spikes:

    >>> rng = np.random.default_rng(42)
    >>> spikes = simulate_spikes_flat_rate(100, 5, rate=0.1, rng=rng)
    >>> spikes.shape
    (100, 5)
    >>> (spikes >= 0).all()
    True
    """
    return rng.poisson(rate, size=(n_time, n_cells))
