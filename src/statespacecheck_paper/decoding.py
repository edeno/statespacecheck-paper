"""General Bayesian decoder with per-spike goodness-of-fit diagnostics.

This module holds the figure-agnostic decoder — the Bayesian filter
(``decode_with_diagnostics``) and its optional per-window override mechanism
(``DecoderOverrideWindow`` / ``DecoderOverrideSchedule``). It takes scientific
primitives (spike counts, a position grid, a transition matrix, and place-field
parameters or an explicit baseline firing-rate table), computes the per-spike
diagnostics via :mod:`statespacecheck_paper.diagnostics`, and returns a
``DecodingDiagnostics``. It depends only on ``diagnostics`` and the general
``simulation`` primitives — no figure-specific module.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import logsumexp
from scipy.stats import poisson

from statespacecheck_paper.diagnostics import (
    DecodingDiagnostics,
    SpikeEventDiagnostics,
    compute_spike_event_diagnostics_from_rates,
)
from statespacecheck_paper.simulation import (
    normalize,
    place_field_rates,
    softmax_with_shift,
)

# Sentinel for observations with no finite joint prior-likelihood mass.
_NEG_INF = float("-inf")


def _condition_on(
    probs: NDArray[np.floating],
    ll: NDArray[np.floating],
) -> tuple[NDArray[np.floating], float]:
    """Bayesian update: multiply prior by emission likelihood, normalize.

    Adapted from ``non_local_detector.core._condition_on`` (which itself
    is adapted from ``dynamax``). The update is evaluated as
    ``log(probs) + ll`` and normalized with log-sum-exp, so a small but
    representable overlap between prior and likelihood is retained rather
    than being replaced by an arbitrary probability cutoff.

    If every joint log-weight is ``-inf``, the observation has zero
    probability everywhere under the prior and the Bayesian posterior is
    undefined. In that degenerate case, this function explicitly returns a
    uniform posterior and ``log_norm = -inf``. Callers should treat
    ``log_norm == -inf`` as the signal to flag the step.

    Parameters
    ----------
    probs : np.ndarray, shape (n_bins,)
        Linear-space prior, must sum to 1.
    ll : np.ndarray, shape (n_bins,)
        Log-likelihood of the observation at each bin (unnormalized).

    Returns
    -------
    new_probs : np.ndarray, shape (n_bins,)
        Posterior; sums to 1. Equals ``1/n_bins`` everywhere when the
        zero-probability fallback runs.
    log_norm : float
        Log marginal likelihood for this step (``log p(obs | past)``).
        ``-inf`` when the zero-probability fallback runs.
    """
    n_bins = probs.size
    if (
        probs.ndim != 1
        or ll.shape != probs.shape
        or not np.all(np.isfinite(probs))
        or np.any(probs < 0.0)
        or not np.isclose(float(probs.sum()), 1.0)
        or np.any(np.isnan(ll))
        or np.any(np.isposinf(ll))
    ):
        raise ValueError(
            "probs must be a finite nonnegative 1D distribution and ll must "
            "have the same shape with no NaN or +inf values"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_joint = np.log(probs) + ll
    log_norm = float(logsumexp(log_joint))
    if np.isneginf(log_norm):
        return np.full(n_bins, 1.0 / n_bins), _NEG_INF
    if not np.isfinite(log_norm):
        raise ValueError("Prior and log-likelihood produced a nonfinite joint log-normalizer")
    new_probs = np.exp(log_joint - log_norm)
    return new_probs, log_norm


@dataclass(frozen=True)
class DecoderOverrideWindow:
    """One decoder-side misfit, active over the half-open interval ``[start, end)``.

    A misfit window substitutes any of three baseline quantities while the
    decoder runs inside it. Each field is optional; ``None`` means "use the
    baseline".

    Parameters
    ----------
    start, end : int
        Half-open time-step bounds ``[start, end)``. ``start < end`` is
        required.
    transition_matrix : np.ndarray, shape (n_bins, n_bins), optional
        Replaces the baseline transition matrix in the predict step.
    firing_rate_table : np.ndarray, shape (n_bins, n_cells), optional
        Replaces the baseline Gaussian place-field rate table used to
        form the posterior-update likelihood, the per-spike diagnostics,
        and the displayed per-spike likelihood. Used by the remap misfit
        (remapped place fields).

    Raises
    ------
    ValueError
        If ``start >= end`` or if ``firing_rate_table`` contains negative or
        non-finite entries.

    Notes
    -----
    Supplied ``transition_matrix`` and ``firing_rate_table`` are copied at
    construction and marked write-protected via ``setflags(write=False)``,
    extending the dataclass's ``frozen=True`` invariant to the array
    contents.

    Shape parity with the decoder's grid is checked by
    :meth:`validate_against`, which the decoder calls once per
    schedule entry — too late to check at construction because the
    schedule may be built before ``position_bins`` is pinned down.

    Examples
    --------
    Remap-style misfit — the decoder and its diagnostics use an alternate
    rate table inside the window:

    >>> import numpy as np
    >>> remapped = np.full((5, 3), 0.1)
    >>> w = DecoderOverrideWindow(10, 20, firing_rate_table=remapped)
    >>> w.start, w.end
    (10, 20)
    """

    start: int
    end: int
    transition_matrix: NDArray[np.floating] | None = None
    firing_rate_table: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        """Validate the window bounds and any supplied rate tables.

        Makes a write-protected copy of any supplied table so the
        ``frozen=True`` invariant extends to the array contents, not
        just the dataclass field bindings.
        """
        if self.start >= self.end:
            raise ValueError(
                f"DecoderOverrideWindow requires start < end, got ({self.start}, {self.end})"
            )

        # A negative or non-finite rate table would become NaN once it
        # reaches ``poisson.pmf`` and propagate silently through the
        # posterior — reject it at construction.
        if self.firing_rate_table is not None and not (
            np.all(np.isfinite(self.firing_rate_table)) and np.all(self.firing_rate_table >= 0.0)
        ):
            raise ValueError(
                "DecoderOverrideWindow.firing_rate_table must be finite and non-negative everywhere"
            )

        # Write-protect any supplied tables. A frozen dataclass only
        # prevents rebinding ``self.firing_rate_table``; the underlying
        # ndarray is still mutable. Take a defensive copy and mark it
        # read-only so callers can't bypass the validation above by
        # mutating in place after construction.
        for name in ("transition_matrix", "firing_rate_table"):
            table = getattr(self, name)
            if table is None:
                continue
            copy = table.copy()
            copy.setflags(write=False)
            object.__setattr__(self, name, copy)

    def validate_against(self, *, n_bins: int, n_cells: int) -> None:
        """Validate that supplied rate tables match the decoder's grid.

        Shape parity with the decoder's position grid and cell count
        can't be checked at construction time because the schedule may
        be built before the decoder's ``position_bins`` is pinned down. Call this
        once per schedule entry inside the decoder.

        Parameters
        ----------
        n_bins : int
            Number of position bins in the decoder's grid (``position_bins.size``).
        n_cells : int
            Number of cells in the spike train (``spike_counts.shape[1]``).

        Raises
        ------
        ValueError
            If ``firing_rate_table`` shape doesn't equal ``(n_bins, n_cells)``,
            or ``transition_matrix`` shape doesn't equal ``(n_bins, n_bins)``.
        """
        if self.firing_rate_table is not None and self.firing_rate_table.shape != (n_bins, n_cells):
            raise ValueError(
                f"DecoderOverrideWindow.firing_rate_table shape "
                f"{self.firing_rate_table.shape} does not "
                f"match decoder grid ({n_bins}, {n_cells})."
            )
        if self.transition_matrix is not None and self.transition_matrix.shape != (
            n_bins,
            n_bins,
        ):
            raise ValueError(
                f"DecoderOverrideWindow.transition_matrix shape "
                f"{self.transition_matrix.shape} does not match decoder grid "
                f"({n_bins}, {n_bins})."
            )


@dataclass(frozen=True)
class DecoderOverrideSchedule:
    """An ordered set of non-overlapping :class:`DecoderOverrideWindow` entries.

    Time steps not covered by any window decode with the baseline
    transition matrix and Gaussian place-field rates. The empty schedule
    (the default) is a clean decode with no misfits — used for real-data
    decoding.

    Parameters
    ----------
    windows : tuple[DecoderOverrideWindow, ...]
        The misfit windows. Must not overlap; order is not significant.

    Raises
    ------
    ValueError
        If any two windows overlap.

    Examples
    --------
    >>> DecoderOverrideSchedule().window_at(5) is None
    True
    >>> w1, w2 = DecoderOverrideWindow(10, 20), DecoderOverrideWindow(30, 40)
    >>> sched = DecoderOverrideSchedule((w1, w2))
    >>> sched.window_at(15).start
    10
    >>> sched.window_at(25) is None
    True
    """

    windows: tuple[DecoderOverrideWindow, ...] = ()

    def __post_init__(self) -> None:
        """Reject overlapping windows."""
        ordered = sorted(self.windows, key=lambda w: w.start)
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if later.start < earlier.end:
                raise ValueError(
                    "DecoderOverrideSchedule windows must not overlap; "
                    f"[{earlier.start}, {earlier.end}) overlaps "
                    f"[{later.start}, {later.end})"
                )

    def window_at(self, t: int) -> DecoderOverrideWindow | None:
        """Return the window containing time step ``t``, or ``None``.

        Windows are non-overlapping (enforced at construction), so at most
        one can match.
        """
        for window in self.windows:
            if window.start <= t < window.end:
                return window
        return None


def _resolve_baseline_firing_rates(
    baseline_firing_rates: NDArray[np.floating] | None,
    position_bins: NDArray[np.floating],
    place_field_centers: NDArray[np.floating],
    place_field_std: float,
    place_field_rate_scale: float,
    n_bins: int,
    n_cells: int,
) -> NDArray[np.floating]:
    """Build or validate the baseline ``(n_bins, n_cells)`` Poisson rate table.

    When ``baseline_firing_rates`` is supplied it is validated against the decoder grid and
    rejected if it is the wrong shape or holds any negative / non-finite rate
    (mirroring ``DecoderOverrideWindow.firing_rate_table`` validation), so a bad table fails
    loudly here rather than surfacing as an opaque Poisson error or a silent NaN
    deep in the filter loop. When omitted, the table is built from the Gaussian
    place-field parameters.
    """
    if baseline_firing_rates is not None:
        rates = np.asarray(baseline_firing_rates, dtype=float)
        if rates.shape != (n_bins, n_cells):
            raise ValueError(
                f"baseline_firing_rates shape {rates.shape} does not match the decoder grid "
                f"(n_bins={n_bins}, n_cells={n_cells})."
            )
        # Reject invalid rate tables up front (as ``DecoderOverrideWindow.firing_rate_table``
        # does), rather than letting a negative/nonfinite rate surface as an
        # opaque Poisson error or a silent NaN deep in the filter loop.
        if not (np.all(np.isfinite(rates)) and np.all(rates >= 0.0)):
            raise ValueError("baseline_firing_rates must contain only finite, non-negative rates.")
        return rates
    return place_field_rates(
        position_bins, place_field_centers, place_field_std, place_field_rate_scale
    )


def _expand_spike_events(
    spike_counts: NDArray[np.int_],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Expand a ``(n_time, n_cells)`` spike-count matrix into per-event indices.

    Excludes ``t=0`` (which has no valid prior) and expands multiplicities: a
    bin with count ``k`` contributes ``k`` repeated events. The returned time
    indices carry the ``+1`` offset that undoes the ``spike_counts[1:]`` slice.
    """
    spike_time_ind, spike_cell_ind = np.nonzero(spike_counts[1:])
    spike_counts_at_events = spike_counts[1:][spike_time_ind, spike_cell_ind].astype(np.intp)
    spike_time_ind = np.repeat(spike_time_ind, spike_counts_at_events)
    spike_cell_ind = np.repeat(spike_cell_ind, spike_counts_at_events)
    spike_time_ind = spike_time_ind + 1  # Adjust for offset from [1:]
    return spike_time_ind, spike_cell_ind


def _select_decoder_components_for_step(
    window: DecoderOverrideWindow | None,
    base_transition: NDArray[np.floating],
    baseline_firing_rates: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Select the transition matrix and rate table for one filter step.

    Pure per-step *selector*: returns the baseline transition matrix and rate
    table unless the active misfit ``window`` overrides either. It performs
    neither the predictive matmul nor the predictive-posterior store — those
    are the recursion itself and stay in ``decode_with_diagnostics``.
    """
    transition_t = base_transition
    if window is not None and window.transition_matrix is not None:
        transition_t = window.transition_matrix
    rates_t = baseline_firing_rates
    if window is not None and window.firing_rate_table is not None:
        rates_t = window.firing_rate_table
    return transition_t, rates_t


def _apply_window_rate_overrides(
    diagnostics: SpikeEventDiagnostics,
    predictive_posterior: NDArray[np.floating],
    windows: tuple[DecoderOverrideWindow, ...],
    spike_time_ind: NDArray[np.intp],
    spike_cell_ind: NDArray[np.intp],
    coverage: float = 0.95,
) -> SpikeEventDiagnostics:
    """Overwrite per-event / dense diagnostics inside each rate-override window.

    The baseline ``diagnostics`` were computed against the decoder's default
    rate table. For every misfit window that swaps ``firing_rate_table``, the events
    falling inside it are recomputed against that window's table so the
    posterior update, per-event diagnostics, and displayed likelihood stay on
    one internally consistent decoder model. The base diagnostics' seven arrays
    are copied first, so this mutates copies and leaves the passed-in dataclass
    untouched; the returned dataclass is assembled from the mutated copies.
    """
    assert diagnostics.hpd_overlap is not None
    assert diagnostics.kl_divergence is not None
    assert diagnostics.predictive_pvalue is not None
    assert diagnostics.per_spike_likelihood is not None

    hpd_overlap = diagnostics.hpd_overlap.copy()
    kl_divergence = diagnostics.kl_divergence.copy()
    predictive_pvalue = diagnostics.predictive_pvalue.copy()
    event_hpd_overlap = diagnostics.event_hpd_overlap.copy()
    event_kl_divergence = diagnostics.event_kl_divergence.copy()
    event_predictive_pvalue = diagnostics.event_predictive_pvalue.copy()
    decoder_per_spike_lik = diagnostics.per_spike_likelihood.copy()

    for window in windows:
        if window.firing_rate_table is None:
            continue
        in_window = (spike_time_ind >= window.start) & (spike_time_ind < window.end)
        if not np.any(in_window):
            continue

        window_diagnostics = compute_spike_event_diagnostics_from_rates(
            predictive_posterior,
            window.firing_rate_table,
            spike_time_ind[in_window],
            spike_cell_ind[in_window],
            coverage=coverage,
        )
        assert window_diagnostics.per_spike_likelihood is not None

        event_hpd_overlap[in_window] = window_diagnostics.event_hpd_overlap
        event_kl_divergence[in_window] = window_diagnostics.event_kl_divergence
        event_predictive_pvalue[in_window] = window_diagnostics.event_predictive_pvalue
        decoder_per_spike_lik[in_window] = window_diagnostics.per_spike_likelihood

        window_times = spike_time_ind[in_window]
        window_cells = spike_cell_ind[in_window]
        hpd_overlap[window_times, window_cells] = window_diagnostics.event_hpd_overlap
        kl_divergence[window_times, window_cells] = window_diagnostics.event_kl_divergence
        predictive_pvalue[window_times, window_cells] = window_diagnostics.event_predictive_pvalue

    return SpikeEventDiagnostics(
        event_time_ind=diagnostics.event_time_ind,
        event_cell_ind=diagnostics.event_cell_ind,
        event_hpd_overlap=event_hpd_overlap,
        event_kl_divergence=event_kl_divergence,
        event_predictive_pvalue=event_predictive_pvalue,
        hpd_overlap=hpd_overlap,
        kl_divergence=kl_divergence,
        predictive_pvalue=predictive_pvalue,
        per_spike_likelihood=decoder_per_spike_lik,
    )


def decode_with_diagnostics(
    spike_counts: NDArray[np.int_],
    position_bins: NDArray[np.floating],
    transition_matrix: NDArray[np.floating],
    place_field_centers: NDArray[np.floating],
    place_field_std: float,
    place_field_rate_scale: float,
    override_schedule: DecoderOverrideSchedule | None = None,
    baseline_firing_rates: NDArray[np.floating] | None = None,
) -> DecodingDiagnostics:
    """Run the Bayesian filter with per-time, per-cell diagnostics.

    This function implements a Bayesian decoder for position from neural spike_counts,
    computing diagnostic metrics at each timestep to assess model goodness-of-fit.

    **Algorithm**:
    1. Initialize with flat prior at t=0
    2. For each timestep t:
       a. Predict: prior = transition_matrix @ post[t-1]
       b. Likelihood: compute P(spike_counts[t] | position) for all cells
       c. Diagnostic metrics: compare prior vs. combined likelihood
       d. Update: post[t] = normalize(prior * combined_likelihood)

    **Diagnostic metrics**:
    - HPD overlap: overlap between prior and combined likelihood HPD regions
    - KL divergence: divergence from prior to combined likelihood
    - Predictive p-value: cumulative probability mass for low-contribution cells

    Parameters
    ----------
    spike_counts : np.ndarray, shape (n_time, n_cells)
        Observed spike counts at each timestep for each cell.
    position_bins : np.ndarray, shape (n_bins,)
        Position grid (spatial bins).
    transition_matrix : np.ndarray, shape (n_bins, n_bins)
        State transition matrix for baseline dynamics.
    place_field_centers : np.ndarray, shape (n_cells,)
        Place field center positions for each cell.
    place_field_std : float
        Width (standard deviation) of Gaussian place fields.
    place_field_rate_scale : float
        Scaling factor for firing rates.
    override_schedule : DecoderOverrideSchedule, optional
        Decoder-side rate or transition regimes, such as remapping and the
        sparse-population control.
        Each :class:`DecoderOverrideWindow` swaps the transition matrix and/or
        the per-cell rate table for its interval. Defaults to an empty
        schedule: a clean decode with no
        misfits (the real-data decoding case).
    baseline_firing_rates : np.ndarray, shape (n_bins, n_cells), optional
        Baseline per-cell Poisson rate table. Supply this when cells do not
        share one place-field width and scale, as in Figure 3's sparse
        population.
        If omitted, rates are built from ``place_field_centers``, ``place_field_std``, and
        ``place_field_rate_scale``.

    Returns
    -------
    results : DecodingDiagnostics
        Frozen dataclass with the following fields (see
        :class:`DecodingDiagnostics` for the full schema):

        Dense ``(n_time, n_bins)`` distributions
            ``posterior`` (filtered posterior), ``predictive`` (one-step
            ahead, flat at t=0), ``likelihood`` (normalized combined
            likelihood from all cells, flat at t=0), and
            ``spike_likelihood`` (combined likelihood from only spiking
            cells; NaN where no spike_counts).

        Dense ``(n_time, n_cells)`` per-cell diagnostic matrices
            ``hpd_overlap``, ``kl_divergence``, ``predictive_pvalue``. NaN at
            t=0 and at any (t, cell) without a spike.

        Per-spike-event arrays of shape ``(n_spikes,)``
            ``event_time_ind`` (time bin), ``event_cell_ind`` (cell
            index), and ``event_hpd_overlap`` / ``event_kl_divergence``
            / ``event_predictive_pvalue`` (the dense matrices scattered to one
            value per event). Spike-count > 1 in a bin produces that
            many repeated events. The legacy ``spike_time_ind`` /
            ``spike_cell_ind`` aliases were removed; use the
            ``event_*_ind`` fields instead.

        ``per_spike_likelihood`` of shape ``(n_spikes, n_bins)``
            Normalized likelihood for each individual spike event,
            computed against the decoder's actual rates (remapped
            inside any misfit window with ``firing_rate_table`` set).

    Notes
    -----
    Invalid misfit configurations (overlapping windows, ``start >= end``,
    negative/non-finite rate tables) are rejected when the
    :class:`DecoderOverrideSchedule` / :class:`DecoderOverrideWindow` is *constructed*, not
    here.

    The per-cell likelihood combination and the posterior update both
    run in log-space via the :func:`_condition_on` pattern adapted from
    ``dynamax`` / ``non_local_detector.core``. The posterior update
    itself cannot underflow on the inner step (it uses an explicit
    log-sum-exp shift). The stored ``combined_likelihood`` and
    ``spike_likelihood`` arrays are renormalized after the same shift,
    so individual bins still underflow to zero in linear space but the
    row as a whole remains a proper probability distribution.

    When the observation has zero probability at every state with nonzero
    prior mass, :func:`_condition_on` falls back to a uniform posterior and
    signals via ``log_norm = -inf``. This function counts such steps and
    emits a single summary ``RuntimeWarning`` at the end so the situation is
    visible. The reset affects the following predictive distribution;
    consumers should inspect or mask downstream events if this occurs.

    Examples
    --------
    >>> import numpy as np
    >>> from statespacecheck_paper.simulation import gaussian_transition_matrix
    >>> # Set up small problem
    >>> n_time, n_cells, n_bins = 10, 3, 21
    >>> spike_counts = np.random.poisson(1.0, size=(n_time, n_cells))
    >>> position_bins = np.linspace(0, 100, n_bins)
    >>> transition_matrix = gaussian_transition_matrix(position_bins, sig=0.5)
    >>> place_field_centers = np.array([25.0, 50.0, 75.0])
    >>> place_field_std = 5.0
    >>> place_field_rate_scale = 0.1
    >>> # Clean decode, no misfits
    >>> results = decode_with_diagnostics(
    ...     spike_counts,
    ...     position_bins,
    ...     transition_matrix,
    ...     place_field_centers,
    ...     place_field_std,
    ...     place_field_rate_scale,
    ... )
    >>> results.posterior.shape
    (10, 21)
    >>> results.hpd_overlap.shape  # Now per-cell
    (10, 3)
    >>> bool(np.all(np.isnan(results.hpd_overlap[0])))  # t=0 has no prior
    True
    """
    n_time = spike_counts.shape[0]
    n_bins = position_bins.size

    if override_schedule is None:
        override_schedule = DecoderOverrideSchedule()

    # Shape-validate every schedule entry against the decoder's grid.
    # DecoderOverrideWindow's __post_init__ can't check this because the schedule
    # may be built before position_bins / spike_counts are pinned down.
    n_cells = spike_counts.shape[1]
    for schedule_entry in override_schedule.windows:
        schedule_entry.validate_against(n_bins=n_bins, n_cells=n_cells)

    # Preallocate outputs
    posterior: NDArray[np.floating] = np.zeros((n_time, n_bins))
    predictive_posterior: NDArray[np.floating] = np.zeros((n_time, n_bins))  # p(x_t | y_{1:t-1})
    combined_likelihood_all: NDArray[np.floating] = np.zeros((n_time, n_bins))  # p(y_t | x_t)
    # Spike-only likelihood: product over only cells that fired (for display).
    # NaN at times with no spike_counts.
    spike_likelihood_all: NDArray[np.floating] = np.full((n_time, n_bins), np.nan)

    # t=0: flat prior. Diagnostic values at t=0 are NaN (no posterior update
    # has happened yet); downstream code masks those entries.
    posterior[0] = normalize(np.ones(n_bins))
    predictive_posterior[0] = posterior[0]  # At t=0, predictive = prior
    combined_likelihood_all[0] = normalize(np.ones(n_bins))  # Flat at t=0

    # Baseline per-cell Poisson rate table. Used at every timestep not
    # covered by a misfit window whose ``firing_rate_table`` is set. Callers
    # can inject ``baseline_firing_rates`` directly when the decoder's cell set does
    # not reduce to one shared Gaussian width/scale — e.g. the figure-3
    # simulation appends a narrow sparse-population of cells with a small
    # baseline rate that increases during a correctly modeled, low-activity
    # window.
    rates = _resolve_baseline_firing_rates(
        baseline_firing_rates,
        position_bins,
        place_field_centers,
        place_field_std,
        place_field_rate_scale,
        n_bins,
        n_cells,
    )

    # Track timesteps where _condition_on fell back to uniform because the
    # observation had zero probability on the prior's support. The fallback
    # keeps the filter alive; report a summary at the end so callers can
    # inspect or mask downstream events if desired.
    n_zero_support_steps = 0
    first_zero_support_t = -1

    for t in range(1, n_time):
        window = override_schedule.window_at(t)

        # Select this step's transition matrix and per-cell rate table —
        # the baseline pair unless the active misfit window overrides either.
        # The predictive matmul and predictive-posterior store below are the
        # recursion itself and stay in the loop.
        current_transition, rates_t = _select_decoder_components_for_step(
            window, transition_matrix, rates
        )

        # ``current_transition`` is column-stochastic: column j is the
        # distribution over next states given current state j (see
        # ``gaussian_transition_matrix``). The predictive marginal is therefore
        # ``T @ post``, not ``post @ T`` — the two differ near the track
        # boundaries where column normalization breaks the kernel's symmetry.
        prior = normalize(current_transition @ posterior[t - 1])  # (n_bins,)
        predictive_posterior[t] = prior  # stored for p-value computation

        # Per-cell log-likelihoods. Log-space avoids underflow when
        # ``n_cells * log(peak)`` crosses the float64 floor (~700) —
        # likely on real-data sessions with many sparsely-firing cells.
        log_lik_per_cell = poisson.logpmf(spike_counts[t][None, :], rates_t)  # (n_bins, n_cells)

        # Combined log-likelihood across cells (sum in log space =
        # product in linear space).
        log_lik_combined = log_lik_per_cell.sum(axis=1)  # (n_bins,)

        # Normalize the stored combined likelihood independently with a
        # max-shifted softmax for display.
        combined_likelihood_all[t] = softmax_with_shift(log_lik_combined)

        # Spike-only likelihood: product over only cells that fired.
        spiking_mask = spike_counts[t] > 0
        if np.any(spiking_mask):
            spike_likelihood_all[t] = softmax_with_shift(
                log_lik_per_cell[:, spiking_mask].sum(axis=1)
            )

        # Posterior update via the _condition_on pattern (dynamax /
        # non_local_detector). ``log_norm = -inf`` flags steps where the
        # observation has zero probability on the prior's support; the helper
        # explicitly returns a uniform posterior in that case and we
        # surface the count post-loop rather than letting the situation
        # silently propagate.
        posterior[t], log_norm = _condition_on(prior, log_lik_combined)
        if log_norm == _NEG_INF:
            if n_zero_support_steps == 0:
                first_zero_support_t = t
            n_zero_support_steps += 1

    if n_zero_support_steps > 0:
        warnings.warn(
            f"decode_with_diagnostics: observation had zero probability on "
            f"the prior support at "
            f"{n_zero_support_steps} timestep(s); first at t={first_zero_support_t}. "
            f"Posterior was reset to uniform at those steps; subsequent "
            f"predictive distributions and diagnostics should be interpreted "
            f"accordingly.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Find all spike events (excluding t=0 which has no valid prior). Count
    # matrices are expanded so a bin with count k contributes k spike events.
    spike_time_ind, spike_cell_ind = _expand_spike_events(spike_counts)

    # Compute the baseline diagnostics first. Events inside a window with
    # ``firing_rate_table`` are overwritten below using that same rate table,
    # keeping the posterior update, per-event diagnostics, and displayed
    # likelihood on one internally consistent decoder model.
    diagnostics = compute_spike_event_diagnostics_from_rates(
        predictive_posterior,
        rates,
        spike_time_ind,
        spike_cell_ind,
        coverage=0.95,
    )

    overridden = _apply_window_rate_overrides(
        diagnostics,
        predictive_posterior,
        override_schedule.windows,
        spike_time_ind,
        spike_cell_ind,
    )
    assert overridden.hpd_overlap is not None  # dense matrices requested above
    assert overridden.kl_divergence is not None
    assert overridden.predictive_pvalue is not None
    assert overridden.per_spike_likelihood is not None

    return DecodingDiagnostics(
        posterior=posterior,
        predictive=predictive_posterior,
        likelihood=combined_likelihood_all,
        spike_likelihood=spike_likelihood_all,
        hpd_overlap=overridden.hpd_overlap,
        kl_divergence=overridden.kl_divergence,
        predictive_pvalue=overridden.predictive_pvalue,
        per_spike_likelihood=overridden.per_spike_likelihood,
        event_time_ind=overridden.event_time_ind,
        event_cell_ind=overridden.event_cell_ind,
        event_hpd_overlap=overridden.event_hpd_overlap,
        event_kl_divergence=overridden.event_kl_divergence,
        event_predictive_pvalue=overridden.event_predictive_pvalue,
    )
