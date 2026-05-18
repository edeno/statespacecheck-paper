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
    >>> (result >= 0.0).all() and (result <= 2.0).all()
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
    # Normalize columns in-place to avoid copy
    col_sums = matrix.sum(axis=0, keepdims=True)
    col_sums = np.maximum(col_sums, 1e-12)
    result: NDArray[np.floating] = matrix / col_sums
    return result


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
    cell_fraction_per_bin: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute cumulative probability mass of cells with low expected contribution.

    This function computes the probability ranking for each neuron based on its
    expected contribution to the likelihood. For each cell, it computes the
    cumulative probability mass of all cells with equal or lower contribution.

    Supports both single-timestep and batched (multi-timestep) inputs.

    Parameters
    ----------
    prior : np.ndarray, shape (n_bins,) or (n_time, n_bins)
        Prior probability distribution over position bins. Can be a single
        distribution or batched over time.
    cell_fraction_per_bin : np.ndarray, shape (n_bins, n_cells)
        Normalized firing rate fractions where each row sums to 1.

    Returns
    -------
    spike_probs : np.ndarray, shape (n_cells,) or (n_time, n_cells)
        For each cell, the cumulative probability mass of cells with
        contribution <= that cell's contribution. Values in [0, 1].
        Output shape matches input: 1D for single timestep, 2D for batched.

    Examples
    --------
    Compute spike probability ranks for a single timestep:

    >>> prior = np.array([0.5, 0.3, 0.2])
    >>> cell_fraction_per_bin = np.array([[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]])
    >>> cell_fraction_per_bin = cell_fraction_per_bin / cell_fraction_per_bin.sum(
    ...     axis=0, keepdims=True
    ... )
    >>> ranks = spike_prob_rank(prior, cell_fraction_per_bin)
    >>> ranks.shape
    (2,)
    >>> (ranks >= 0.0).all() and (ranks <= 1.0).all()
    True

    Compute spike probability ranks for multiple timesteps (batched):

    >>> prior_batched = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
    >>> ranks_batched = spike_prob_rank(prior_batched, cell_fraction_per_bin)
    >>> ranks_batched.shape
    (2, 2)

    Notes
    -----
    This matches the MATLAB implementation:
    sum(lambda_expect(lambda_expect <= lambda_expect(j)))
    where lambda_expect are probabilities summing to 1.
    """
    # prior @ cell_fraction_per_bin gives expected contribution per cell
    # Shape: (n_cells,) for 1D prior, (n_time, n_cells) for 2D prior
    contrib: NDArray[np.floating] = prior @ cell_fraction_per_bin

    # Comparison tolerance for the rank mask. The matmul above sums
    # ``n_bins`` floating-point products in BLAS-defined order, which
    # differs across platforms (Accelerate vs OpenBLAS) and produces
    # FP-noise-different contributions even for inputs that should be
    # exactly equal. A strict ``<=`` comparison turns those tiny
    # differences into rank flips, breaking platform reproducibility.
    # The tolerance scales with the contribution magnitude × the
    # accumulation depth so it absorbs reduction-order noise without
    # affecting genuine differences in real-world non-uniform inputs.
    eps = np.finfo(contrib.dtype).eps
    n_terms = max(1, prior.shape[-1])
    rank_atol = float(eps * n_terms * 16) * float(np.max(contrib))

    if contrib.ndim == 1:
        # Single timestep: (n_cells,)
        # mask[i, j] = True means contrib[i] <= contrib[j] (within tolerance).
        mask = contrib[:, None] <= contrib + rank_atol  # (n_cells, n_cells)
        # For each cell j, sum contrib[i] over all cells i where contrib[i] <= contrib[j].
        # axis=0 accumulates across the i dimension (rows) for each j (column).
        spike_probs: NDArray[np.floating] = (contrib[:, None] * mask).sum(axis=0)
    else:
        # Batched: (n_time, n_cells)
        # mask[t, i, j] = True means contrib[t, i] <= contrib[t, j].
        mask = contrib[:, :, None] <= contrib[:, None, :] + rank_atol
        # axis=1 accumulates across the i dimension for each (t, j).
        spike_probs = (contrib[:, :, None] * mask).sum(axis=1)  # (n_time, n_cells)

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


def simulate_spikes_history_dependent(
    x: NDArray[np.floating],
    pf_centers: NDArray[np.floating],
    pf_width: float,
    rate_scale: float,
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
    likelihoods (which evaluate ``Poisson(k=1 | rate(x))``) are
    unchanged for any individual spike — the misfit is in the
    *temporal* joint distribution, not the per-step marginal.

    Parameters
    ----------
    x : np.ndarray, shape (n_time,)
        Position at each time step.
    pf_centers : np.ndarray, shape (n_cells,)
        Place field centers for each cell.
    pf_width : float
        Standard deviation of the Gaussian place field.
    rate_scale : float
        Peak firing rate (scales the Gaussian).
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
    >>> x = np.linspace(0, 100, 200)
    >>> pf_centers = np.array([20.0, 50.0, 80.0])
    >>> spikes = simulate_spikes_history_dependent(
    ...     x, pf_centers, pf_width=10.0, rate_scale=5.0, rng=rng
    ... )
    >>> spikes.shape
    (200, 3)
    >>> (spikes >= 0).all()
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

    n_time = x.shape[0]
    n_cells = pf_centers.shape[0]
    # (n_time, n_cells) Gaussian place-field rate at each step's position.
    base_rates = placefield_rates(x, pf_centers, pf_width, rate_scale)

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


def wiggly_flat_rates(
    xs: NDArray[np.floating],
    n_cells: int,
    *,
    base_rate: float = 0.05,
    wiggle_amp: float = 0.01,
    n_wiggles: float = 5.0,
) -> NDArray[np.floating]:
    """Per-cell rate functions that are mostly flat with small spatial wiggles.

    Each cell ``c`` gets a rate ``base_rate + wiggle_amp * sin(2π * n_wiggles
    * (x - x_min) / (x_max - x_min) + φ_c)`` with cell-specific phase
    ``φ_c = 2π * c / n_cells``. The rate is nearly position-independent —
    a spike from such a cell carries little spatial information.

    Used to construct the "wiggly-flat likelihood" misfit phase: the
    decoder uses this rate table to compute likelihoods, so per-spike
    likelihoods are wiggly-flat rather than Gaussian. HPD regions of
    such likelihoods are unstable (small perturbations move the HPD
    threshold across many bins), and the rank-based p-value becomes
    ambiguous because no cell has a clearly higher expected contribution.

    Parameters
    ----------
    xs : np.ndarray, shape (n_bins,)
        Position grid.
    n_cells : int
        Number of cells whose rates to construct.
    base_rate : float, optional
        Constant rate floor (in same units as ``rate_scale``). Must be
        positive. Defaults to 0.05.
    wiggle_amp : float, optional
        Amplitude of the sinusoidal modulation. Must be non-negative and
        strictly smaller than ``base_rate`` so rates stay positive
        everywhere. Defaults to 0.01.
    n_wiggles : float, optional
        Number of full sine cycles across the position grid. Defaults
        to 5.

    Returns
    -------
    rates : np.ndarray, shape (n_bins, n_cells)
        Wiggly-flat rate table. Strictly positive everywhere.

    Raises
    ------
    ValueError
        If ``n_cells < 1``, ``xs`` has fewer than 2 points or a
        non-positive range, ``base_rate <= 0``, ``wiggle_amp < 0``, or
        ``wiggle_amp >= base_rate``.

    Examples
    --------
    >>> import numpy as np
    >>> xs = np.linspace(0, 100, 101)
    >>> rates = wiggly_flat_rates(xs, n_cells=11)
    >>> rates.shape
    (101, 11)
    >>> (rates > 0).all()
    True
    """
    # Validate up front: a sign-flipped or out-of-range argument here would
    # otherwise produce negative rates, which silently become NaN once they
    # reach ``poisson.pmf`` downstream in ``decode_and_diagnostics``.
    if n_cells < 1:
        raise ValueError(f"n_cells must be >= 1, got {n_cells}")
    if xs.size < 2:
        raise ValueError("xs must have at least 2 points to define a position range")
    x_range = float(xs[-1] - xs[0])
    if x_range <= 0:
        raise ValueError("xs must be increasing with positive range")
    if base_rate <= 0:
        raise ValueError(f"base_rate must be positive, got {base_rate}")
    if wiggle_amp < 0:
        raise ValueError(f"wiggle_amp must be non-negative, got {wiggle_amp}")
    if wiggle_amp >= base_rate:
        raise ValueError("wiggle_amp must be < base_rate to keep rates positive")

    normalized_x = (xs - xs[0]) / x_range  # (n_bins,) in [0, 1]
    phases = 2 * np.pi * np.arange(n_cells, dtype=float) / n_cells  # (n_cells,)
    angle = 2 * np.pi * n_wiggles * normalized_x[:, None] + phases[None, :]  # (n_bins, n_cells)
    rates: NDArray[np.floating] = base_rate + wiggle_amp * np.sin(angle)
    return rates
