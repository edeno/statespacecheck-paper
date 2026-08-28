"""Simulation utilities for state space models.

This module provides utilities for simulating random walks, spike trains, and
computing likelihood functions for position decoding in neuroscience applications.

Key Components
--------------
- **Normalization**: Safe normalization of probability distributions
- **Boundary conditions**: Reflecting boundary conditions for random walks
- **Transition matrices**: Gaussian transition matrices for state space models
- **Place fields**: Gaussian place field models for spatial tuning
- **Spike generation**: Poisson spike generation for position-tuned neurons
- **Place-field rates**: Gaussian place-field firing-rate tables

Examples
--------
Simulate a random walk with reflecting boundaries:

>>> import numpy as np
>>> rng = np.random.default_rng(42)
>>> walk = simulate_walk(n_time_steps=100, step_std=1.0, initial_position=50.0,
...                      position_min=0.0, position_max=100.0, rng=rng)
>>> walk.shape
(100,)
>>> bool((walk >= 0.0).all() and (walk <= 100.0).all())
True

Generate position-tuned spikes:

>>> x = np.linspace(0, 100, 100)
>>> place_field_centers = np.array([25.0, 50.0, 75.0])
>>> spikes = simulate_spikes_position_tuned(x, place_field_centers, place_field_std=5.0,
...                                         place_field_rate_scale=0.1, rng=rng)
>>> spikes.shape
(100, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


def softmax_with_shift(ll: NDArray[np.floating]) -> NDArray[np.floating]:
    """Numerically stable softmax via the log-sum-exp shift.

    Subtracts ``ll.max()`` before exponentiation so the largest entry
    is exactly 1 and the rest are bounded in (0, 1]. The result sums
    to 1. An observation whose log-likelihood is ``-inf`` everywhere is
    undefined and raises instead of being replaced by a uniform distribution.

    Parameters
    ----------
    ll : np.ndarray
        Log-likelihood values to normalize into a probability distribution.

    Returns
    -------
    np.ndarray
        Normalized probabilities, same shape as ``ll``, summing to 1.

    Raises
    ------
    ValueError
        If ``ll`` is empty, contains NaN or ``+inf``, is ``-inf`` everywhere
        (zero likelihood under every state), or yields an invalid weight sum.
    """
    if ll.size == 0:
        raise ValueError("ll must contain at least one value")
    if np.any(np.isnan(ll)) or np.any(np.isposinf(ll)):
        raise ValueError("ll must not contain NaN or +inf values")
    lmax = float(np.max(ll))
    if np.isneginf(lmax):
        raise ValueError(
            "Cannot normalize log-likelihood: every value is -inf, so the "
            "observation has zero likelihood under every state."
        )
    weighted = np.exp(ll - lmax)
    total = float(weighted.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Cannot normalize log-likelihood: exponential weights have invalid sum")
    normalized: NDArray[np.floating] = weighted / total
    return normalized


def normalize(p: NDArray[np.floating], axis: int | None = None) -> NDArray[np.floating]:
    """Normalize a finite, nonnegative array along the specified axis.

    Parameters
    ----------
    p : np.ndarray
        Array to normalize.
    axis : int or None, optional
        Axis along which to normalize. If None, normalizes entire array.

    Returns
    -------
    normalized : np.ndarray
        Normalized array with same shape as input, where sum along axis equals 1.

    Raises
    ------
    ValueError
        If ``p`` is empty, contains a non-finite or negative value, or has
        zero total mass along any normalized slice.

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
    p = np.asarray(p)
    if p.size == 0:
        raise ValueError("p must contain at least one value")
    if not np.all(np.isfinite(p)) or np.any(p < 0.0):
        raise ValueError("p must contain only finite nonnegative values")
    # Finite inputs can still overflow during the reduction (for example,
    # ``[1e308, 1e308]``). Suppress NumPy's generic warning so we can raise a
    # precise contract error instead of returning a zero-mass result from
    # division by infinity.
    with np.errstate(over="ignore", invalid="ignore"):
        s: NDArray[np.floating] = np.sum(p, axis=axis, keepdims=True)
    if np.any(~np.isfinite(s)):
        bad = np.flatnonzero(~np.isfinite(np.asarray(s).reshape(-1)))
        raise ValueError(
            "Cannot normalize non-finite total mass; invalid flattened slice indices: "
            f"{bad[:10].tolist()}"
        )
    if np.any(s <= 0.0):
        bad = np.flatnonzero(np.asarray(s).reshape(-1) <= 0.0)
        raise ValueError(
            "Cannot normalize zero total mass; invalid flattened slice indices: "
            f"{bad[:10].tolist()}"
        )
    result: NDArray[np.floating] = p / s
    return result


def reflect_into_interval(
    x: NDArray[np.floating], lower_bound: float, upper_bound: float
) -> NDArray[np.floating]:
    """Reflect values into interval [lower_bound, upper_bound] using triangle wave method.

    This implements reflecting boundary conditions for random walks by treating
    the walk as a triangle wave that bounces off the boundaries.

    Parameters
    ----------
    x : np.ndarray
        Array of values to reflect.
    lower_bound : float
        Lower bound of interval.
    upper_bound : float
        Upper bound of interval.

    Returns
    -------
    reflected : np.ndarray
        Array of same shape as x with all values in [lower_bound, upper_bound].

    Examples
    --------
    Reflect values outside bounds:

    >>> x = np.array([-1.0, 0.5, 2.5])
    >>> result = reflect_into_interval(x, 0.0, 2.0)
    >>> bool((result >= 0.0).all() and (result <= 2.0).all())
    True

    Values inside bounds are unchanged:

    >>> x = np.array([0.5, 1.0, 1.5])
    >>> result = reflect_into_interval(x, 0.0, 2.0)
    >>> np.allclose(result, x)
    True
    """
    interval_length = upper_bound - lower_bound
    y: NDArray[np.floating] = np.mod(x - lower_bound, 2 * interval_length)
    y = np.where(y <= interval_length, y, 2 * interval_length - y)
    result: NDArray[np.floating] = y + lower_bound
    return result


def gaussian_transition_matrix(
    position_bins: NDArray[np.floating], step_std: float
) -> NDArray[np.floating]:
    """Compute one-step Gaussian transition matrix for random walk.

    Computes transition probabilities for Gaussian random walk on discrete grid.
    Each column represents a probability distribution over next states given
    current state.

    Parameters
    ----------
    position_bins : np.ndarray, shape (n_bins,)
        Grid of position values.
    step_std : float
        Standard deviation of Gaussian transition kernel.

    Returns
    -------
    transition_matrix : np.ndarray, shape (n_bins, n_bins)
        Transition matrix where element [i, j] is probability of transitioning
        to state i given current state j. Each column sums to 1.

    Examples
    --------
    Create transition matrix for 3-state system:

    >>> position_bins = np.array([0.0, 1.0, 2.0])
    >>> matrix = gaussian_transition_matrix(position_bins, step_std=1.0)
    >>> matrix.shape
    (3, 3)
    >>> np.allclose(matrix.sum(axis=0), 1.0)  # Columns sum to 1
    True
    """
    position_bins = np.asarray(position_bins)
    if position_bins.ndim != 1 or position_bins.size == 0:
        raise ValueError("position_bins must be a non-empty 1-D array")
    if not np.all(np.isfinite(position_bins)):
        raise ValueError("position_bins must contain only finite values")
    if not np.isfinite(step_std) or step_std <= 0.0:
        raise ValueError(f"step_std must be positive and finite; got {step_std}")
    diff = position_bins[:, None] - position_bins[None, :]
    matrix = norm.pdf(diff, loc=0.0, scale=step_std)
    col_sums = matrix.sum(axis=0, keepdims=True)
    if not np.all(np.isfinite(col_sums)) or np.any(col_sums <= 0.0):
        raise ValueError("Gaussian transition kernel has a non-positive or non-finite column sum")
    result: NDArray[np.floating] = matrix / col_sums
    return result


def place_field_rates(
    position_bins: NDArray[np.floating],
    place_field_centers: NDArray[np.floating],
    place_field_std: float,
    place_field_rate_scale: float,
) -> NDArray[np.floating]:
    """Compute Gaussian place field firing rates.

    Computes firing rate for each neuron at each position using Gaussian place
    field model. Each neuron has a place field centered at one location with
    specified place_field_std.

    Parameters
    ----------
    position_bins : np.ndarray, shape (n_bins,)
        Position bin centers.
    place_field_centers : np.ndarray, shape (n_cells,)
        Place field center for each neuron.
    place_field_std : float
        Standard deviation of Gaussian place field.
    place_field_rate_scale : float
        Scale factor multiplying the normalized Gaussian place-field density.

    Returns
    -------
    rates : np.ndarray, shape (n_bins, n_cells)
        Firing rate for each position bin and neuron.

    Examples
    --------
    Compute place field rates for 3 neurons:

    >>> position_bins = np.linspace(0, 10, 11)
    >>> place_field_centers = np.array([2.0, 5.0, 8.0])
    >>> rates = place_field_rates(
    ...     position_bins, place_field_centers, place_field_std=1.0, place_field_rate_scale=1.0
    ... )
    >>> rates.shape
    (11, 3)
    >>> rates.max(axis=0).round(3)  # Peak at each center
    array([0.399, 0.399, 0.399])
    """
    result: NDArray[np.floating] = (
        norm.pdf(position_bins[:, None], loc=place_field_centers[None, :], scale=place_field_std)
        * place_field_rate_scale
    )
    return result


def peak_rate_to_place_field_scale(peak_rate_per_step: float, place_field_std: float) -> float:
    """Convert a desired peak firing rate to the ``place_field_rate_scale`` argument.

    ``place_field_rates`` multiplies a normalized Gaussian density (peaking at
    ``1 / (place_field_std * sqrt(2 * pi))`` at the field center) by
    ``place_field_rate_scale``. This is its inverse: the scale that makes the
    field peak at ``peak_rate_per_step``.

    Parameters
    ----------
    peak_rate_per_step : float
        Target firing rate at the place-field center (spikes per time step).
    place_field_std : float
        Standard deviation of the Gaussian place field.

    Returns
    -------
    place_field_rate_scale : float
        Scale factor to pass to ``place_field_rates`` for the given peak rate.
    """
    return float(peak_rate_per_step * np.sqrt(2.0 * np.pi) * place_field_std)


def simulate_walk(
    n_time_steps: int,
    step_std: float,
    initial_position: float,
    position_min: float,
    position_max: float,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Simulate random walk with reflecting boundary conditions.

    Simulates a Gaussian random walk on continuous space with reflecting
    boundaries. The walk starts at initial_position and takes steps drawn from a Gaussian
    distribution with standard deviation step_std.

    Parameters
    ----------
    n_time_steps : int
        Number of time steps to simulate.
    step_std : float
        Standard deviation of step size distribution.
    initial_position : float
        Initial position.
    position_min : float
        Lower boundary (reflecting).
    position_max : float
        Upper boundary (reflecting).
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    trajectory : np.ndarray, shape (n_time_steps,)
        Simulated trajectory with all values in [position_min, position_max].

    Examples
    --------
    Simulate a 100-step random walk:

    >>> rng = np.random.default_rng(42)
    >>> walk = simulate_walk(
    ...     100, step_std=1.0, initial_position=50.0, position_min=0.0, position_max=100.0, rng=rng
    ... )
    >>> walk.shape
    (100,)
    >>> bool((walk >= 0.0).all() and (walk <= 100.0).all())
    True

    With zero step size, trajectory is constant:

    >>> rng = np.random.default_rng(42)
    >>> walk = simulate_walk(
    ...     10, step_std=0.0, initial_position=50.0, position_min=0.0, position_max=100.0, rng=rng
    ... )
    >>> np.allclose(walk, 50.0)
    True
    """
    steps = rng.normal(loc=0.0, scale=step_std, size=n_time_steps)
    x = initial_position + np.cumsum(steps)
    return reflect_into_interval(x, position_min, position_max)


def simulate_spikes_position_tuned(
    position: NDArray[np.floating],
    place_field_centers: NDArray[np.floating],
    place_field_std: float,
    place_field_rate_scale: float,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Simulate Poisson spikes for position-tuned neurons.

    Generates spike counts from Poisson distribution with position-dependent
    firing rates. Each neuron has a Gaussian place field determining its
    firing rate at each position.

    Parameters
    ----------
    position : np.ndarray, shape (n_time,)
        Position at each time step.
    place_field_centers : np.ndarray, shape (n_cells,)
        Place field center for each neuron.
    place_field_std : float
        Standard deviation of Gaussian place field.
    place_field_rate_scale : float
        Scale factor multiplying the normalized Gaussian place-field density.
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
    >>> position = np.linspace(0, 100, 100)
    >>> place_field_centers = np.array([25.0, 50.0, 75.0])
    >>> spikes = simulate_spikes_position_tuned(position, place_field_centers, place_field_std=5.0,
    ...                                         place_field_rate_scale=0.1, rng=rng)
    >>> spikes.shape
    (100, 3)
    >>> bool((spikes >= 0).all())
    True
    """
    lam = (
        norm.pdf(position[:, None], loc=place_field_centers[None, :], scale=place_field_std)
        * place_field_rate_scale
    )
    spikes: NDArray[np.int_] = rng.poisson(lam)
    return spikes


def simulate_spikes_history_dependent(
    position: NDArray[np.floating],
    place_field_centers: NDArray[np.floating],
    place_field_std: float,
    place_field_rate_scale: float,
    rng: np.random.Generator,
    *,
    refractory_steps: int = 1,
    burst_window: tuple[int, int] = (2, 10),
    burst_factor: float = 3.0,
) -> NDArray[np.int_]:
    """Position-tuned spikes with hippocampal-style refractory + bursting.

    Generates per-step Poisson spikes whose rate is modulated by each cell's
    own recent history:

    - Hard refractory: a cell that just fired cannot fire for the next
      ``refractory_steps`` steps (rate set to 0).
    - Burst window: ``burst_window[0]`` to ``burst_window[1]`` steps after
      a spike, the rate is multiplied by ``burst_factor``.
    - Outside both windows, the rate is the standard Gaussian place-field
      rate, same as :func:`simulate_spikes_position_tuned`.

    At 1 ms / step (the default temporal interpretation of the figure-3
    simulation), the defaults ``refractory_steps=1`` and ``burst_window=(2,
    10)`` correspond to a 1 ms hard refractory period followed by a
    burst-prone window 2-10 ms post-spike, matching the rough phenomenology
    of CA1 pyramidal cells.

    The Poisson assumption is violated by this generator: the
    spike-spike correlations introduced by the burst window create a
    joint distribution that is not memoryless. Per-spike spatial
    likelihoods (which evaluate ``Poisson(k=1 | rate(position))``) are
    unchanged for any individual spike — the misfit is in the
    *temporal* joint distribution, not the per-step marginal.

    Parameters
    ----------
    position : np.ndarray, shape (n_time,)
        Position at each time step.
    place_field_centers : np.ndarray, shape (n_cells,)
        Place field centers for each cell.
    place_field_std : float
        Standard deviation of the Gaussian place field.
    place_field_rate_scale : float
        Scale factor multiplying the normalized Gaussian place-field density.
    rng : np.random.Generator
        Random number generator for reproducibility.
    refractory_steps : int, optional
        Number of steps after a spike during which the cell cannot fire.
        Must be >= 1. Defaults to 1 (the immediately-following step is
        suppressed).
    burst_window : tuple[int, int], optional
        ``(start, end)`` step offsets after a spike during which the
        rate is boosted. End is inclusive; must satisfy
        ``0 <= start <= end``. Defaults to ``(2, 10)``.
    burst_factor : float, optional
        Multiplier on the base rate during the burst window. Must be
        positive. Defaults to 3.0.

    Returns
    -------
    spikes : np.ndarray, shape (n_time, n_cells)
        Spike counts (non-negative integers).

    Raises
    ------
    ValueError
        If ``refractory_steps < 1``, ``burst_window`` does not satisfy
        ``0 <= start <= end``, or ``burst_factor <= 0``.

    Notes
    -----
    History dependence breaks vectorization over time, so this routine
    loops per timestep (still vectorized over cells per step). For
    figure-3-scale runs (~40k timesteps, 11 cells) this is fast enough
    to be unnoticeable.

    When ``burst_window`` overlaps the refractory region
    (``burst_start < refractory_steps``), the refractory zero is applied
    first, so the effective burst window is
    ``[max(burst_start, refractory_steps), burst_end]``.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> position = np.linspace(0, 100, 200)
    >>> place_field_centers = np.array([20.0, 50.0, 80.0])
    >>> spikes = simulate_spikes_history_dependent(
    ...     position, place_field_centers, place_field_std=10.0, place_field_rate_scale=5.0, rng=rng
    ... )
    >>> spikes.shape
    (200, 3)
    >>> bool((spikes >= 0).all())
    True
    """
    # Validate up front: silently-accepted nonsense (refractory_steps=0,
    # a reversed or negative burst_window) would disable the refractory or
    # burst mechanism with no error, producing a generative model subtly
    # different from the one requested.
    if refractory_steps < 1:
        raise ValueError(f"refractory_steps must be >= 1, got {refractory_steps}")
    burst_start, burst_end = burst_window
    if not (0 <= burst_start <= burst_end):
        raise ValueError(f"burst_window must satisfy 0 <= start <= end, got {burst_window}")
    if burst_factor <= 0:
        raise ValueError(f"burst_factor must be positive, got {burst_factor}")

    n_time = position.shape[0]
    n_cells = place_field_centers.shape[0]
    # (n_time, n_cells) Gaussian place-field rate at each step's position.
    base_rates = place_field_rates(
        position, place_field_centers, place_field_std, place_field_rate_scale
    )

    spikes = np.zeros((n_time, n_cells), dtype=np.int_)
    # ``steps_since_spike[c]`` = number of steps since cell ``c`` last fired.
    # Initialize to ``burst_end + 1`` so every cell starts outside both
    # the refractory and burst regimes.
    steps_since_spike = np.full(n_cells, burst_end + 1, dtype=np.int64)

    for t in range(n_time):
        rate = base_rates[t].copy()  # (n_cells,)
        in_refractory = steps_since_spike < refractory_steps
        in_burst = (steps_since_spike >= burst_start) & (steps_since_spike <= burst_end)
        rate[in_refractory] = 0.0
        rate[in_burst] *= burst_factor

        step_spikes = rng.poisson(rate)
        spikes[t] = step_spikes
        # Cells that fired reset to 0; everyone else increments.
        fired = step_spikes > 0
        steps_since_spike = np.where(fired, 0, steps_since_spike + 1)

    return spikes
