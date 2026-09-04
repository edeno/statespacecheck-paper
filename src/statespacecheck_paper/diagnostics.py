"""Shared goodness-of-fit diagnostics for neural decoding.

This module holds the general, figure-agnostic diagnostic layer: the per-spike
diagnostic containers, the core per-spike-event computation (HPD overlap, KL
divergence, and the rank-based predictive p-value), the single-event likelihood
and predictive-mark calculations they build on, and the baseline-threshold
estimation used to flag misfit.

It depends only on ``numpy``/``scipy`` and the external ``statespacecheck``
package — it imports no sibling ``statespacecheck_paper`` module, so it is the
leaf of the paper's dependency graph.

**Key Components**:
- **SpikeEventDiagnostics**: per-spike-event diagnostic arrays (dense matrices optional)
- **DecodingDiagnostics**: full decoder return with dense distributions + diagnostics
- **compute_spike_event_diagnostics_from_rates**: the shared per-spike-event computation
- **compute_normalized_event_likelihood**: normalized single-event intensity likelihood
- **compute_predictive_mark_probabilities**: predictive mark distribution for an event
- **DiagnosticThresholds** / **compute_baseline_diagnostic_thresholds**: baseline flags
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import numpy as np
import statespacecheck as ssc
from numpy.typing import NDArray
from scipy.special import logsumexp

# Spike-batch size for ``compute_spike_event_diagnostics_from_rates``.
# Caps the (n_spikes, n_bins) scratch arrays at ``_PER_SPIKE_BATCH``
# rows so full-session real-data builds (~870 K spikes) don't allocate
# multi-GB working buffers. 50 K × 512 bins × float64 ≈ 200 MB per
# scratch array, ~600 MB peak with three live (pred / rates / lik).
_PER_SPIKE_BATCH = 50_000

# Per-event metric field names — shared by ``SpikeEventDiagnostics`` and
# ``DecodingDiagnostics`` shape-validation loops.
_PER_EVENT_METRIC_NAMES = ("event_hpd_overlap", "event_kl_divergence", "event_predictive_pvalue")

# Baseline-threshold definitions used by ``compute_baseline_diagnostic_thresholds``.
# Named (rather than inlined at the ``nanquantile`` calls) because the manuscript
# reports the percentile levels themselves, and the figure summaries record these
# constants alongside the threshold values they produce — so a change to the rule
# cannot silently move a published number.
BASELINE_HPD_OVERLAP_QUANTILE = 0.01
BASELINE_KL_DIVERGENCE_QUANTILE = 0.99
FIXED_PREDICTIVE_PVALUE_CUTOFF = 0.05


def _validate_diagnostic_range(
    arr: NDArray[np.floating],
    name: str,
    *,
    lo: float,
    hi: float | None,
    allow_nan: bool,
    allow_positive_infinity: bool = False,
    atol: float = 1e-9,
) -> None:
    """Validate missingness, infinities, and the scientific metric range.

    Dense matrices may opt into NaN because they encode "no spike at this
    (t, cell)" structurally. Per-event arrays may not: every row represents an
    observed spike and must carry a value. KL divergence may opt into positive
    infinity, which is a meaningful result for disjoint support; negative
    infinity is never valid. ``atol`` absorbs harmless floating-point range
    overshoot at the producer boundary.
    """
    if not allow_nan and np.any(np.isnan(arr)):
        raise ValueError(f"{name}: NaN found in a required per-event value")
    if np.any(np.isneginf(arr)):
        raise ValueError(f"{name}: -inf is not a valid diagnostic value")
    if not allow_positive_infinity and np.any(np.isposinf(arr)):
        raise ValueError(f"{name}: +inf is not permitted for this diagnostic")
    present = ~np.isnan(arr)
    if not np.any(present):
        return
    valid = arr[present]
    if np.any(valid < lo - atol):
        raise ValueError(f"{name}: values below {lo} found (min={float(valid.min())})")
    if hi is not None and np.any(valid > hi + atol):
        raise ValueError(f"{name}: values above {hi} found (max={float(valid.max())})")


@dataclass(frozen=True)
class SpikeEventDiagnostics:
    """Return of :func:`compute_spike_event_diagnostics_from_rates`.

    Per-spike-event arrays are always present; the four dense
    ``(n_time, n_cells)`` / ``(n_spikes, n_bins)`` arrays are
    optional, populated only when ``include_dense_matrices=True``.
    Frozen + write-protected so a downstream consumer cannot
    accidentally mutate a metric mid-pipeline.

    Parameters
    ----------
    event_time_ind, event_cell_ind : np.ndarray, shape (n_spikes,)
        Time-bin index and cell index for each spike event.
    event_hpd_overlap, event_kl_divergence, event_predictive_pvalue : np.ndarray, shape (n_spikes,)
        Per-event diagnostic values.
    hpd_overlap, kl_divergence, predictive_pvalue : np.ndarray, shape (n_time, n_cells), optional
        Dense scattered matrices; ``NaN`` where no spike occurred. ``None`` when
        the producer was called with ``include_dense_matrices=False``.
    per_spike_likelihood : np.ndarray, shape (n_spikes, n_bins), optional
        Per-spike normalized likelihood. ``None`` when ``include_dense_matrices=False``.
    event_time : np.ndarray, shape (n_spikes,), optional
        Wall-clock spike time for each event. Populated by the real-data path;
        ``None`` for simulated paths that carry only bin indices.

    Raises
    ------
    ValueError
        If the per-event arrays don't share leading dim ``n_spikes``, or
        the dense matrices (when present) don't share leading dim ``n_time``.
    """

    event_time_ind: NDArray[np.intp]
    event_cell_ind: NDArray[np.intp]
    event_hpd_overlap: NDArray[np.floating]
    event_kl_divergence: NDArray[np.floating]
    event_predictive_pvalue: NDArray[np.floating]
    hpd_overlap: NDArray[np.floating] | None
    kl_divergence: NDArray[np.floating] | None
    predictive_pvalue: NDArray[np.floating] | None
    per_spike_likelihood: NDArray[np.floating] | None
    # Real-data path supplies wall-clock spike times alongside the
    # bin indices; simulated paths leave this ``None``.
    event_time: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        n_spikes = self.event_time_ind.shape[0]
        for name in (
            "event_cell_ind",
            "event_hpd_overlap",
            "event_kl_divergence",
            "event_predictive_pvalue",
        ):
            arr = getattr(self, name)
            if arr.shape != (n_spikes,):
                raise ValueError(f"SpikeEventDiagnostics.{name} shape {arr.shape} != ({n_spikes},)")
        if self.event_time is not None and self.event_time.shape != (n_spikes,):
            raise ValueError(
                f"SpikeEventDiagnostics.event_time shape {self.event_time.shape} != ({n_spikes},)"
            )
        if np.any(self.event_time_ind < 0) or np.any(self.event_cell_ind < 0):
            raise ValueError("SpikeEventDiagnostics event indices must be non-negative")
        if self.event_time is not None and not np.all(np.isfinite(self.event_time)):
            raise ValueError("SpikeEventDiagnostics.event_time must contain only finite values")
        # Dense matrices are an all-or-nothing group.
        dense_names = ("hpd_overlap", "kl_divergence", "predictive_pvalue", "per_spike_likelihood")
        dense_provided = [getattr(self, n) is not None for n in dense_names]
        if any(dense_provided) and not all(dense_provided):
            missing = [n for n, p in zip(dense_names, dense_provided, strict=True) if not p]
            raise ValueError(
                f"SpikeEventDiagnostics: dense matrices must be all-or-nothing; missing {missing}"
            )
        if self.hpd_overlap is not None:
            assert self.kl_divergence is not None  # narrowed by all-or-nothing
            assert self.predictive_pvalue is not None
            assert self.per_spike_likelihood is not None
            n_time, n_cells = self.hpd_overlap.shape
            if self.kl_divergence.shape != (n_time, n_cells):
                raise ValueError(
                    f"kl_divergence shape {self.kl_divergence.shape} != ({n_time}, {n_cells})"
                )
            if self.predictive_pvalue.shape != (n_time, n_cells):
                raise ValueError(
                    "predictive_pvalue shape "
                    f"{self.predictive_pvalue.shape} != ({n_time}, {n_cells})"
                )
            if self.per_spike_likelihood.shape[0] != n_spikes:
                raise ValueError(
                    f"per_spike_likelihood leading dim {self.per_spike_likelihood.shape[0]} "
                    f"!= n_spikes={n_spikes}"
                )
        _validate_diagnostic_range(
            self.event_hpd_overlap,
            "SpikeEventDiagnostics.event_hpd_overlap",
            lo=0.0,
            hi=1.0,
            allow_nan=False,
        )
        _validate_diagnostic_range(
            self.event_predictive_pvalue,
            "SpikeEventDiagnostics.event_predictive_pvalue",
            lo=0.0,
            hi=1.0,
            allow_nan=False,
        )
        _validate_diagnostic_range(
            self.event_kl_divergence,
            "SpikeEventDiagnostics.event_kl_divergence",
            lo=0.0,
            hi=None,
            allow_nan=False,
            allow_positive_infinity=True,
        )
        if self.hpd_overlap is not None:
            assert self.kl_divergence is not None
            assert self.predictive_pvalue is not None
            _validate_diagnostic_range(
                self.hpd_overlap,
                "SpikeEventDiagnostics.hpd_overlap",
                lo=0.0,
                hi=1.0,
                allow_nan=True,
            )
            _validate_diagnostic_range(
                self.predictive_pvalue,
                "SpikeEventDiagnostics.predictive_pvalue",
                lo=0.0,
                hi=1.0,
                allow_nan=True,
            )
            _validate_diagnostic_range(
                self.kl_divergence,
                "SpikeEventDiagnostics.kl_divergence",
                lo=0.0,
                hi=None,
                allow_nan=True,
                allow_positive_infinity=True,
            )
        # Write-protect everything that's not None.
        for name in (
            "event_time_ind",
            "event_cell_ind",
            *_PER_EVENT_METRIC_NAMES,
            *dense_names,
            "event_time",
        ):
            arr = getattr(self, name)
            if arr is not None:
                arr.setflags(write=False)


@dataclass(frozen=True)
class DecodingDiagnostics:
    """Return of :func:`decode_with_diagnostics`.

    Frozen so downstream code cannot rebind fields; arrays are
    write-protected so it cannot mutate them in place either.

    Parameters
    ----------
    posterior, predictive, likelihood, spike_likelihood : np.ndarray, shape (n_time, n_bins)
        Dense distributions over position.
    hpd_overlap, kl_divergence, predictive_pvalue : np.ndarray, shape (n_time, n_cells)
        Dense per-cell diagnostic matrices; ``NaN`` where no spike.
    event_time_ind, event_cell_ind : np.ndarray, shape (n_spikes,)
        Time-bin / cell index for each spike event.
    event_hpd_overlap, event_kl_divergence, event_predictive_pvalue : np.ndarray, shape (n_spikes,)
        Per-event diagnostic values.
    per_spike_likelihood : np.ndarray, shape (n_spikes, n_bins)
        Per-spike normalized likelihood as seen by the decoder
        (uses ``firing_rate_table`` inside override windows where set).

    Raises
    ------
    ValueError
        If shape invariants are violated — all dense ``(n_time, ...)``
        arrays must share leading dim, all per-event ``(n_spikes,)``
        arrays must share leading dim, dense ``(n_time, n_bins)``
        arrays must share trailing dim with each other, and dense
        ``(n_time, n_cells)`` arrays must share trailing dim with each
        other.
    """

    posterior: NDArray[np.floating]
    predictive: NDArray[np.floating]
    likelihood: NDArray[np.floating]
    spike_likelihood: NDArray[np.floating]
    hpd_overlap: NDArray[np.floating]
    kl_divergence: NDArray[np.floating]
    predictive_pvalue: NDArray[np.floating]
    event_time_ind: NDArray[np.intp]
    event_cell_ind: NDArray[np.intp]
    event_hpd_overlap: NDArray[np.floating]
    event_kl_divergence: NDArray[np.floating]
    event_predictive_pvalue: NDArray[np.floating]
    per_spike_likelihood: NDArray[np.floating]

    def __post_init__(self) -> None:
        # 2-D guard before unpacking — a 1-D ``posterior`` would
        # otherwise raise the less-informative ``IndexError`` on the
        # next line instead of the ``ValueError`` the docstring promises.
        if self.posterior.ndim != 2:
            raise ValueError(
                f"DecodingDiagnostics.posterior must be 2-D (n_time, n_bins); "
                f"got shape {self.posterior.shape}"
            )
        if self.hpd_overlap.ndim != 2:
            raise ValueError(
                "DecodingDiagnostics.hpd_overlap must be 2-D (n_time, n_cells); "
                f"got shape {self.hpd_overlap.shape}"
            )
        n_time, n_bins = self.posterior.shape
        for name in ("predictive", "likelihood", "spike_likelihood"):
            arr = getattr(self, name)
            if arr.shape != (n_time, n_bins):
                raise ValueError(
                    f"DecodingDiagnostics.{name} shape {arr.shape} != ({n_time}, {n_bins})"
                )
        n_cells = self.hpd_overlap.shape[1]
        for name in ("kl_divergence", "predictive_pvalue"):
            arr = getattr(self, name)
            if arr.shape != (n_time, n_cells):
                raise ValueError(
                    f"DecodingDiagnostics.{name} shape {arr.shape} != ({n_time}, {n_cells})"
                )
        n_spikes = self.event_time_ind.shape[0]
        if self.event_cell_ind.shape != (n_spikes,):
            raise ValueError(
                f"DecodingDiagnostics.event_cell_ind shape "
                f"{self.event_cell_ind.shape} != ({n_spikes},)"
            )
        for name in _PER_EVENT_METRIC_NAMES:
            arr = getattr(self, name)
            if arr.shape != (n_spikes,):
                raise ValueError(f"DecodingDiagnostics.{name} shape {arr.shape} != ({n_spikes},)")
        if self.per_spike_likelihood.shape != (n_spikes, n_bins):
            raise ValueError(
                f"DecodingDiagnostics.per_spike_likelihood shape "
                f"{self.per_spike_likelihood.shape} != ({n_spikes}, {n_bins})"
            )
        # Value-range invariants on the per-cell metrics + their per-event
        # counterparts. NaN is legitimate at (t, cell) without a spike, so
        # the range check ignores NaN. A buggy decoder otherwise ships
        # out-of-range values that only surface much later (e.g., as a NaN
        # ``DiagnosticThresholds`` or a misleading hexbin).
        _validate_diagnostic_range(
            self.hpd_overlap,
            "DecodingDiagnostics.hpd_overlap",
            lo=0.0,
            hi=1.0,
            allow_nan=True,
        )
        _validate_diagnostic_range(
            self.predictive_pvalue,
            "DecodingDiagnostics.predictive_pvalue",
            lo=0.0,
            hi=1.0,
            allow_nan=True,
        )
        _validate_diagnostic_range(
            self.kl_divergence,
            "DecodingDiagnostics.kl_divergence",
            lo=0.0,
            hi=None,
            allow_nan=True,
            allow_positive_infinity=True,
        )
        _validate_diagnostic_range(
            self.event_hpd_overlap,
            "DecodingDiagnostics.event_hpd_overlap",
            lo=0.0,
            hi=1.0,
            allow_nan=False,
        )
        _validate_diagnostic_range(
            self.event_predictive_pvalue,
            "DecodingDiagnostics.event_predictive_pvalue",
            lo=0.0,
            hi=1.0,
            allow_nan=False,
        )
        _validate_diagnostic_range(
            self.event_kl_divergence,
            "DecodingDiagnostics.event_kl_divergence",
            lo=0.0,
            hi=None,
            allow_nan=False,
            allow_positive_infinity=True,
        )
        # Write-protect every backing buffer.
        for name in (
            "posterior",
            "predictive",
            "likelihood",
            "spike_likelihood",
            "hpd_overlap",
            "kl_divergence",
            "predictive_pvalue",
            "event_time_ind",
            "event_cell_ind",
            *_PER_EVENT_METRIC_NAMES,
            "per_spike_likelihood",
        ):
            getattr(self, name).setflags(write=False)


def compute_normalized_event_likelihood(
    event_intensities: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute the normalized single-event likelihood over position.

    For position-dependent intensities (or expected counts per bin), the
    likelihood contribution of one selected event is proportional to its mark
    intensity. This function normalizes that intensity to sum to 1 over the
    last axis. It is the per-event observation likelihood the diagnostics
    compare against the predictive distribution, and the quantity shown in the
    likelihood row of the simulation and real-data figures.

    The Poisson exposure term is deliberately absent. In the full binned count
    likelihood, ``exp(-sum_c lambda_c(x))`` is shared by the whole bin, while
    each observed event contributes one factor ``lambda_mark(x)``. Attaching
    ``Poisson(1; lambda) = lambda * exp(-lambda)`` to every event would duplicate
    the exposure term when a bin contains multiple spikes and would not match
    the event-conditioned predictive-mark diagnostic.

    Normalization is done in log space (``log`` + ``logsumexp``) so rows with
    tiny but nonzero intensities keep their correct shape: e.g. intensities
    ``[1e-20, 2e-20, 4e-20]`` normalize to ``[1/7, 2/7, 4/7]`` rather than
    collapsing to uniform. A row with zero intensity at every position has no
    defined event likelihood and raises rather than being assigned a shape.

    Parameters
    ----------
    event_intensities : np.ndarray, shape (..., n_bins)
        Position-dependent event intensities or expected counts per bin for one
        or more events/marks. Normalization is over the last axis.

    Returns
    -------
    likelihood : np.ndarray, shape (..., n_bins)
        Normalized single-event likelihood over position; each row sums to 1.
    """
    event_intensities = np.asarray(event_intensities)
    if event_intensities.ndim < 1 or event_intensities.shape[-1] == 0:
        raise ValueError("event_intensities must have a non-empty position axis")
    if not np.all(np.isfinite(event_intensities)) or np.any(event_intensities < 0.0):
        raise ValueError("event_intensities must contain only finite nonnegative values")
    with np.errstate(divide="ignore"):
        log_intensity = np.log(event_intensities)
    log_norm = logsumexp(log_intensity, axis=-1, keepdims=True)
    degenerate = np.isneginf(log_norm)
    if np.any(degenerate):
        bad = np.flatnonzero(degenerate.reshape(-1))
        raise ValueError(
            "Cannot compute an event likelihood for intensity rows that are "
            "zero at every position; invalid flattened row indices: "
            f"{bad[:10].tolist()}"
        )
    likelihood: NDArray[np.floating] = np.exp(log_intensity - log_norm)
    return likelihood


def compute_predictive_mark_probabilities(
    predictive_distribution: NDArray[np.floating],
    mark_intensities: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute the predictive mark distribution for a randomly selected event.

    Raw mark intensities are first averaged over the predictive state
    distribution and the resulting expected intensities are then normalized
    across marks. For discrete marks ``c``, this evaluates

    ``q[c] = sum_x p[x] * intensity[x, c] / sum_d sum_x p[x] * intensity[x, d]``.

    Normalizing at each state before averaging would instead integrate the
    state-conditional mark distribution against the unconditioned state
    distribution. That omits the event-rate weighting of the latent state and
    is only equivalent when total event intensity is constant across states.

    Parameters
    ----------
    predictive_distribution : np.ndarray, shape (n_bins,) or (n_time, n_bins)
        Predictive probability distribution over state bins.
    mark_intensities : np.ndarray, shape (n_bins, n_marks)
        Nonnegative event intensities (or expected event counts per bin) for
        every state and discrete mark.

    Returns
    -------
    mark_probabilities : np.ndarray, shape (n_marks,) or (n_time, n_marks)
        Predictive mark probabilities for a randomly selected event.

    Raises
    ------
    ValueError
        If an input violates its shape/value contract, or if any row has zero
        predictive event intensity (for which the conditional mark distribution
        is undefined).
    """
    if predictive_distribution.ndim not in (1, 2):
        raise ValueError("predictive_distribution must have shape (n_bins,) or (n_time, n_bins)")
    if mark_intensities.ndim != 2 or mark_intensities.shape[0] != predictive_distribution.shape[-1]:
        raise ValueError("mark_intensities must have shape (n_bins, n_marks)")
    if mark_intensities.shape[1] == 0:
        raise ValueError("mark_intensities must contain at least one mark")
    if not np.all(np.isfinite(predictive_distribution)) or np.any(predictive_distribution < 0.0):
        raise ValueError("predictive_distribution must contain finite nonnegative values")
    if not np.all(np.isfinite(mark_intensities)) or np.any(mark_intensities < 0.0):
        raise ValueError("mark_intensities must contain finite nonnegative values")

    # Finite inputs can overflow in the matrix product or the subsequent
    # across-mark reduction. Convert those arithmetic failures into explicit
    # contract errors rather than returning zeros from division by infinity.
    with np.errstate(over="ignore", invalid="ignore"):
        expected_intensities: NDArray[np.floating] = predictive_distribution @ mark_intensities
    if not np.all(np.isfinite(expected_intensities)):
        raise ValueError("Predictive expected mark intensities are non-finite after integration")
    if expected_intensities.ndim == 1:
        with np.errstate(over="ignore", invalid="ignore"):
            total_intensity = float(expected_intensities.sum())
        if not np.isfinite(total_intensity):
            raise ValueError("Predictive total event intensity is non-finite")
        if total_intensity <= 0.0:
            raise ValueError(
                "Predictive mark distribution is undefined because total event intensity is zero"
            )
        mark_probabilities: NDArray[np.floating] = expected_intensities / total_intensity
        return mark_probabilities

    with np.errstate(over="ignore", invalid="ignore"):
        total_intensity = expected_intensities.sum(axis=1, keepdims=True)
    nonfinite_total = ~np.isfinite(total_intensity[:, 0])
    if nonfinite_total.any():
        bad = np.flatnonzero(nonfinite_total)
        raise ValueError(
            f"Predictive total event intensity is non-finite for row indices: {bad[:10].tolist()}"
        )
    zero_total = total_intensity[:, 0] == 0.0
    if zero_total.any():
        bad = np.flatnonzero(zero_total)
        raise ValueError(
            "Predictive mark distribution is undefined for rows with zero total "
            f"event intensity; row indices: {bad[:10].tolist()}"
        )
    mark_probabilities = expected_intensities / total_intensity
    return mark_probabilities


def _compute_spike_event_predictive_pvalue_rank(
    pred_chunk: NDArray[np.floating],
    rates: NDArray[np.floating],
    cell_ind: NDArray[np.intp],
) -> NDArray[np.floating]:
    """Per-event spike-probability rank for one batch of spike events.

    For each event the rank is the cumulative predictive mass of cells whose
    expected contribution is ``<=`` the firing cell's contribution:

        rank[k] = sum_j contrib[k, j] where contrib[k, j] <= contrib[k, cell_ind[k]]

    ``contrib`` is the predictive cell probability conditional on an event,
    obtained by integrating the raw cell intensities in ``rates`` over the
    predictive state distribution and then normalizing across cells (via
    :func:`compute_predictive_mark_probabilities` on the **full**
    ``(n_bins, n_cells)`` table, so every cell competes for the rank, not just
    the firing one). The ``rank_atol`` slack on the ``<=`` comparison absorbs
    BLAS reduction-order floating-point noise so equal contributions yield equal
    ranks across platforms.

    This is the memory-lean, per-event specialization used inside the chunked
    real-data loop: it materializes only ``(n_events, n_cells)`` working arrays.
    It is deliberately distinct from the general all-cells rank that lived in
    ``simulation.spike_prob_rank`` (removed as it had no production caller),
    which built a batched ``(n_time, n_cells, n_cells)`` mask; the two are not
    merged because their memory profiles and indexing differ.

    Parameters
    ----------
    pred_chunk : np.ndarray, shape (n_events, n_bins)
        Predictive state distribution gathered at each event's time bin.
    rates : np.ndarray, shape (n_bins, n_cells)
        Expected spike rate at each position for each cell.
    cell_ind : np.ndarray, shape (n_events,)
        Firing cell index for each event.

    Returns
    -------
    np.ndarray, shape (n_events,)
        Per-event rank in ``[0, 1]``.
    """
    n_bins = rates.shape[0]
    contrib_chunk = compute_predictive_mark_probabilities(pred_chunk, rates)
    chunk_size = pred_chunk.shape[0]
    target_contrib = contrib_chunk[np.arange(chunk_size), cell_ind]  # (n_events,)
    rank_atol = (
        float(np.finfo(contrib_chunk.dtype).eps * n_bins * 16) * float(np.max(contrib_chunk))
        if contrib_chunk.size
        else 0.0
    )
    rank_mask = contrib_chunk <= target_contrib[:, None] + rank_atol
    result: NDArray[np.floating] = (contrib_chunk * rank_mask).sum(axis=1)
    # Summation may overshoot one by a few ulps; pin only that representational
    # error at the mathematical boundary so display code can enforce [0, 1].
    np.minimum(result, 1.0, out=result)
    return result


def compute_spike_event_diagnostics_from_rates(
    predictive_posterior: NDArray[np.floating],
    rates: NDArray[np.floating],
    spike_time_ind: NDArray[np.intp],
    spike_cell_ind: NDArray[np.intp],
    coverage: float = 0.95,
    include_dense_matrices: bool = True,
) -> SpikeEventDiagnostics:
    """Compute per-cell diagnostic metrics at spike times.

    This is the core computation shared by both simulated and real data analysis.
    It computes HPD overlap, KL divergence, and predictive p-value ranking for
    each spike event using the firing cell's event-intensity factor.

    Parameters
    ----------
    predictive_posterior : np.ndarray, shape (n_time, n_bins)
        Predictive posterior distribution over position at each time.
    rates : np.ndarray, shape (n_bins, n_cells)
        Expected spike rate (spikes/bin) at each position for each cell.
    spike_time_ind : np.ndarray, shape (n_spikes,)
        Time indices where spikes occurred.
    spike_cell_ind : np.ndarray, shape (n_spikes,)
        Cell indices for each spike event.
    coverage : float, default 0.95
        Coverage probability for HPD region computation.
    include_dense_matrices : bool, default True
        If True (default), also populate the (n_time, n_cells) ``hpd_overlap``,
        ``kl_divergence``, ``predictive_pvalue`` matrices and the (n_spikes, n_bins)
        ``per_spike_likelihood`` on the returned dataclass. If False, those four
        attributes are left ``None`` and the matching allocations / scatters are
        skipped — useful for callers that only need the per-spike event arrays
        (the cache builder is the canonical example), since for real
        recordings the dense matrices can be hundreds of MB.

    Returns
    -------
    diagnostics : SpikeEventDiagnostics
        Frozen dataclass (see :class:`SpikeEventDiagnostics`) whose per-event
        arrays are always populated:

        - ``event_time_ind`` / ``event_cell_ind``: shape (n_spikes,)
        - ``event_hpd_overlap``: shape (n_spikes,), per-spike HPD overlap
        - ``event_kl_divergence``: shape (n_spikes,), per-spike KL divergence
        - ``event_predictive_pvalue``: shape (n_spikes,), per-spike predictive p-value

        If ``include_dense_matrices`` (the default), the optional dense
        attributes are also populated (otherwise each is ``None``):

        - ``hpd_overlap``: shape (n_time, n_cells), NaN where no spike
        - ``kl_divergence``: shape (n_time, n_cells), NaN where no spike
        - ``predictive_pvalue``: shape (n_time, n_cells), NaN where no spike
        - ``per_spike_likelihood``: shape (n_spikes, n_bins), normalized
          likelihood distribution for each individual spike event

    Notes
    -----
    Each event likelihood is the firing cell's intensity normalized over
    position. If multiple spikes occur in the same time/cell bin, callers
    should pass repeated entries in ``spike_time_ind`` and ``spike_cell_ind``
    so every observed spike contributes one event to the returned arrays. Such
    events have identical local diagnostics because the binned model holds the
    predictive distribution and intensity table constant within the bin.

    The predictive cell distribution used by ``event_predictive_pvalue`` is
    event-weighted: raw cell intensities are averaged over the predictive
    state distribution and the resulting expected intensities are normalized
    across cells. Normalizing across cells at each state before averaging would
    omit the state-dependent total event intensity.
    """
    n_time, n_bins = predictive_posterior.shape
    n_cells = rates.shape[1]
    n_spikes = len(spike_time_ind)

    event_hpd_overlap: NDArray[np.floating] = np.empty(n_spikes)
    event_kl_divergence: NDArray[np.floating] = np.empty(n_spikes)
    event_predictive_pvalue: NDArray[np.floating] = np.empty(n_spikes)

    # Dense (n_time, n_cells) matrices are only allocated when requested;
    # for real recordings with millions of time bins they can dwarf the
    # rest of the working set, so the cache builder opts out.
    hpd_overlap: NDArray[np.floating] | None = None
    kl_divergence: NDArray[np.floating] | None = None
    predictive_pvalue: NDArray[np.floating] | None = None
    per_spike_likelihood: NDArray[np.floating] | None = None
    if include_dense_matrices:
        hpd_overlap = np.full((n_time, n_cells), np.nan)
        kl_divergence = np.full((n_time, n_cells), np.nan)
        predictive_pvalue = np.full((n_time, n_cells), np.nan)
        per_spike_likelihood = np.empty((n_spikes, n_bins))

    if n_spikes > 0:
        # Per-event likelihood / HPD / KL / predictive p-value all need
        # ``(S, n_bins)`` or ``(S, n_cells)`` working arrays. For
        # full-session real-data builds (~870 K spikes × 256 bins
        # × 8 B ≈ 1.8 GB *per array*) materializing them in one shot
        # blows the working set even when ``include_dense_matrices=False``
        # skips the (n_time, n_cells) outputs. Process in chunks to
        # bound peak memory to ``_PER_SPIKE_BATCH × n_bins × 8 B`` per
        # scratch array. ``predictive_pvalue`` is the per-event rank
        # ``sum_i contrib[i] where contrib[i] <= contrib[j]``;
        # computing it per event (not vectorized over unique times)
        # bounds the rank computation's working set to
        # ``B × n_cells``.
        batch = max(1, _PER_SPIKE_BATCH)
        for start in range(0, n_spikes, batch):
            stop = min(start + batch, n_spikes)
            sti = spike_time_ind[start:stop]
            sci = spike_cell_ind[start:stop]

            # (chunk, n_bins) predictive + event-likelihood arrays for this batch only.
            pred_chunk = predictive_posterior[sti]
            rates_chunk = rates[:, sci].T
            lik_chunk = compute_normalized_event_likelihood(rates_chunk)

            event_hpd_overlap[start:stop] = ssc.hpd_overlap(
                pred_chunk, lik_chunk, coverage=coverage
            )
            event_kl_divergence[start:stop] = ssc.kl_divergence(pred_chunk, lik_chunk)

            # Per-event spike-prob rank over the full cell set, with a
            # reduction-order tolerance for cross-platform reproducibility.
            event_predictive_pvalue[start:stop] = _compute_spike_event_predictive_pvalue_rank(
                pred_chunk, rates, sci
            )

            if per_spike_likelihood is not None:
                per_spike_likelihood[start:stop] = lik_chunk

        if hpd_overlap is not None:
            hpd_overlap[spike_time_ind, spike_cell_ind] = event_hpd_overlap
        if kl_divergence is not None:
            kl_divergence[spike_time_ind, spike_cell_ind] = event_kl_divergence
        if predictive_pvalue is not None:
            predictive_pvalue[spike_time_ind, spike_cell_ind] = event_predictive_pvalue

    return SpikeEventDiagnostics(
        event_time_ind=spike_time_ind,
        event_cell_ind=spike_cell_ind,
        event_hpd_overlap=event_hpd_overlap,
        event_kl_divergence=event_kl_divergence,
        event_predictive_pvalue=event_predictive_pvalue,
        hpd_overlap=hpd_overlap,
        kl_divergence=kl_divergence,
        predictive_pvalue=predictive_pvalue,
        per_spike_likelihood=per_spike_likelihood,
    )


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Threshold values for diagnostic metrics.

    Computed from the baseline period across all cells (flattened).
    Frozen so a downstream consumer cannot rebind a field mid-pipeline.

    Parameters
    ----------
    hpd_overlap : float
        HPD overlap threshold; must lie in ``[0, 1]`` (the underlying
        diagnostic is a probability overlap). Lower values indicate
        worse fit.
    kl_divergence : float
        KL divergence threshold; must be non-negative finite. Higher
        values indicate worse fit.
    predictive_pvalue : float
        Spike-probability threshold; must lie in ``[0, 1]``. Defaulted
        to 0.05 by :func:`compute_baseline_diagnostic_thresholds`. Lower
        values indicate misfit.

    Raises
    ------
    ValueError
        If any field falls outside its documented range, or is NaN.
        The construction-time check prevents a NaN threshold (e.g.
        from an all-NaN baseline) silently making every downstream
        ``metric < threshold`` comparison evaluate ``False``.

    Examples
    --------
    >>> thresholds = DiagnosticThresholds(
    ...     hpd_overlap=0.5,
    ...     kl_divergence=2.0,
    ...     predictive_pvalue=0.05,
    ... )
    >>> thresholds.hpd_overlap
    0.5
    """

    hpd_overlap: float
    kl_divergence: float
    predictive_pvalue: float

    def __post_init__(self) -> None:
        if not (0.0 <= self.hpd_overlap <= 1.0):
            raise ValueError(
                f"DiagnosticThresholds.hpd_overlap must lie in [0, 1]; got {self.hpd_overlap}"
            )
        if not (np.isfinite(self.kl_divergence) and self.kl_divergence >= 0.0):
            raise ValueError(
                f"DiagnosticThresholds.kl_divergence must be finite and non-negative; "
                f"got {self.kl_divergence}"
            )
        if not (0.0 <= self.predictive_pvalue <= 1.0):
            raise ValueError(
                f"DiagnosticThresholds.predictive_pvalue must lie in [0, 1]; "
                f"got {self.predictive_pvalue}"
            )


def compute_baseline_diagnostic_thresholds(
    diagnostics: DecodingDiagnostics | Mapping[str, NDArray[np.floating] | NDArray[np.intp]],
    *,
    baseline_end_index: int,
) -> DiagnosticThresholds:
    """Compute threshold values from baseline period.

    Thresholds are computed across all cells (flattened (n_time, n_cells)
    → 1D) so a single threshold scalar can compare against any cell's
    diagnostic time series:

    - HPD overlap threshold: 1st percentile (low values indicate misfit)
    - KL divergence threshold: 99th percentile (high values indicate misfit)
    - predictive_pvalue threshold: fixed at 0.05 (a conventional rank-statistic cutoff)

    Parameters
    ----------
    diagnostics : DecodingDiagnostics or Mapping[str, NDArray]
        Either a :class:`DecodingDiagnostics` (the typical caller, produced by
        :func:`decode_with_diagnostics`) or a plain dict — the dict
        form is retained so synthetic test fixtures don't need to
        construct a full ``DecodingDiagnostics``. Only ``hpd_overlap`` and
        ``kl_divergence`` are read; the ``predictive_pvalue`` threshold is a
        fixed constant (0.05) and is not derived from the input.
    baseline_end_index : int, keyword-only
        Index marking end of baseline period (exclusive). Required —
        silently slicing the whole recording would contaminate
        "baseline" thresholds with misfit data and is rarely what
        the caller intends.

    Returns
    -------
    thresholds : DiagnosticThresholds
        Threshold values for each diagnostic metric.

    Raises
    ------
    ValueError
        If the baseline slice of ``hpd_overlap`` or ``kl_divergence``
        contains no finite values (thresholds would be NaN and
        downstream comparisons would silently evaluate False), or if the
        slice contains infinity (a finite empirical threshold cannot be
        estimated).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> diagnostics = {
    ...     'hpd_overlap': rng.uniform(0.5, 1.0, (100, 5)),
    ...     'kl_divergence': rng.uniform(0.0, 2.0, (100, 5)),
    ...     'predictive_pvalue': rng.uniform(0.0, 1.0, (100, 5)),
    ... }
    >>> thresholds = compute_baseline_diagnostic_thresholds(diagnostics, baseline_end_index=50)
    >>> thresholds.predictive_pvalue  # Fixed at 0.05
    0.05
    """

    def _get(name: str) -> NDArray[np.floating]:
        arr = (
            getattr(diagnostics, name)
            if isinstance(diagnostics, DecodingDiagnostics)
            else diagnostics[name]
        )
        return cast("NDArray[np.floating]", arr)

    # Flatten (n_time, n_cells) to 1D for quantile computation. ``np.nanquantile``
    # returns ``np.floating``; cast to plain ``float`` to match the
    # ``DiagnosticThresholds`` dataclass signature.
    hpd_baseline = _get("hpd_overlap")[:baseline_end_index].ravel()
    if np.any(np.isinf(hpd_baseline)):
        raise ValueError(
            "compute_baseline_diagnostic_thresholds: hpd_overlap baseline contains infinity"
        )
    if not np.any(np.isfinite(hpd_baseline)):
        raise ValueError(
            "compute_baseline_diagnostic_thresholds: hpd_overlap baseline slice "
            f"(:{baseline_end_index}) contains no finite values; threshold "
            "would be NaN."
        )
    hpd_overlap_threshold = float(np.nanquantile(hpd_baseline, BASELINE_HPD_OVERLAP_QUANTILE))

    kl_baseline = _get("kl_divergence")[:baseline_end_index].ravel()
    if np.any(np.isinf(kl_baseline)):
        raise ValueError(
            "compute_baseline_diagnostic_thresholds: kl_divergence baseline contains infinity; "
            "a finite empirical threshold cannot be estimated"
        )
    if not np.any(np.isfinite(kl_baseline)):
        raise ValueError(
            "compute_baseline_diagnostic_thresholds: kl_divergence baseline slice "
            f"(:{baseline_end_index}) contains no finite values; threshold "
            "would be NaN."
        )
    kl_divergence_threshold = float(np.nanquantile(kl_baseline, BASELINE_KL_DIVERGENCE_QUANTILE))

    # Fixed rank-statistic cutoff; not derived from the data.
    predictive_pvalue_threshold = FIXED_PREDICTIVE_PVALUE_CUTOFF

    return DiagnosticThresholds(
        hpd_overlap=hpd_overlap_threshold,
        kl_divergence=kl_divergence_threshold,
        predictive_pvalue=predictive_pvalue_threshold,
    )
