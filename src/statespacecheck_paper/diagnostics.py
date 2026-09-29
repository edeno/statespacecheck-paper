"""Shared goodness-of-fit diagnostics for neural decoding.

This module holds the general, figure-agnostic diagnostic layer: the per-spike
diagnostic containers, the adapter from the paper's decoder outputs to the
external ``statespacecheck`` package's per-event diagnostics (HPD overlap, KL
divergence, and the rank-based predictive p-value), and the paper's baseline-threshold
rule used to flag misfit. The diagnostics themselves, including the single-event
likelihood and predictive-mark calculations, live in ``statespacecheck`` so that
other projects can use them directly.

It depends only on ``numpy`` and the external ``statespacecheck`` package — it
imports no sibling ``statespacecheck_paper`` module, so it is a leaf of the
paper's dependency graph.

**Key Components**:
- **SpikeEventDiagnostics**: per-spike-event diagnostic arrays (dense matrices optional)
- **DecodingDiagnostics**: full decoder return with dense distributions + diagnostics
- **expand_spike_events**: one event per spike from a spike-count matrix
- **compute_spike_event_diagnostics_from_rates**: per-spike diagnostics via ``statespacecheck``
- **DiagnosticThresholds** / **compute_baseline_diagnostic_thresholds**: baseline flags
- **METRIC_FLAG_DIRECTIONS** / **flag_mask**: the inclusive flag rule
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import statespacecheck as ssc
from numpy.typing import NDArray

# Baseline-threshold definitions used by ``compute_baseline_diagnostic_thresholds``.
# Named (rather than inlined at the quantile calls) because the manuscript
# reports the percentile levels themselves, and the figure summaries record these
# constants alongside the threshold values they produce — so a change to the rule
# cannot silently move a published number.
BASELINE_HPD_OVERLAP_QUANTILE = 0.01
BASELINE_KL_DIVERGENCE_QUANTILE = 0.99
FIXED_PREDICTIVE_PVALUE_CUTOFF = 0.05

# Probability mass of the highest-density regions compared by the HPD-overlap
# diagnostic, throughout the paper.
HPD_COVERAGE = 0.95

FlagDirection = Literal["below", "above"]

# Side of the threshold on which each raw per-event diagnostic indicates worse
# fit, in the paper's metric order: low HPD overlap and low predictive p-values
# are misfit, and so is high KL divergence.
METRIC_FLAG_DIRECTIONS: dict[str, FlagDirection] = {
    "hpd_overlap": "below",
    "predictive_pvalue": "below",
    "kl_divergence": "above",
}

# Per-event metric field names — shared by ``SpikeEventDiagnostics`` and
# ``DecodingDiagnostics`` shape-validation loops.
_PER_EVENT_METRIC_NAMES = tuple(f"event_{metric}" for metric in METRIC_FLAG_DIRECTIONS)

# Name a figure summary records for each direction's inclusive comparison.
INCLUSIVE_FLAG_COMPARISONS: dict[FlagDirection, str] = {
    "below": "less_than_or_equal",
    "above": "greater_than_or_equal",
}


def flag_mask(
    values: NDArray[np.floating],
    threshold: float,
    direction: FlagDirection,
) -> NDArray[np.bool_]:
    """Flag diagnostic values on the worse-fit side of ``threshold``, inclusively.

    Parameters
    ----------
    values : np.ndarray, shape (n_events,)
        Per-event diagnostic values.
    threshold : float
        Flag threshold. Values equal to it are flagged.
    direction : {"below", "above"}
        ``"below"`` flags ``values <= threshold``; ``"above"`` flags
        ``values >= threshold``. See :data:`METRIC_FLAG_DIRECTIONS`.

    Returns
    -------
    np.ndarray of bool, shape (n_events,)
        True where the value is flagged. NaN is never flagged.

    Raises
    ------
    ValueError
        If ``direction`` is not ``"below"`` or ``"above"``.

    Examples
    --------
    >>> flag_mask(np.array([0.01, 0.05, 0.5, np.nan]), 0.05, "below").tolist()
    [True, True, False, False]
    """
    array = np.asarray(values, dtype=np.float64)
    if direction == "below":
        return array <= threshold
    if direction == "above":
        return array >= threshold
    raise ValueError(f"direction must be 'below' or 'above'; got {direction!r}")


def _validate_diagnostic_range(
    arr: NDArray[np.floating],
    name: str,
    *,
    lo: float,
    hi: float | None,
    allow_nan: bool,
    allow_positive_infinity: bool = False,
) -> None:
    """Validate missingness, infinities, and the scientific metric range.

    Dense matrices may opt into NaN because they encode "no spike at this
    (t, cell)" structurally. Per-event arrays may not: every row represents an
    observed spike and must carry a value. KL divergence may opt into positive
    infinity, which is a meaningful result for disjoint support; negative
    infinity is never valid. A tolerance of ``1e-9`` absorbs harmless
    floating-point range overshoot at the producer boundary.
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
    atol = 1e-9
    if np.any(valid < lo - atol):
        raise ValueError(f"{name}: values below {lo} found (min={float(valid.min())})")
    if hi is not None and np.any(valid > hi + atol):
        raise ValueError(f"{name}: values above {hi} found (max={float(valid.max())})")


def _validate_metric_ranges(
    container: SpikeEventDiagnostics | DecodingDiagnostics,
    owner: str,
    prefix: str,
    *,
    allow_nan: bool,
) -> None:
    """Range-check a container's three ``{prefix}{metric}`` diagnostic arrays.

    HPD overlap and the predictive p-value must lie in ``[0, 1]``; KL
    divergence must be non-negative and may be ``+inf``. ``prefix`` is
    ``"event_"`` for the per-event arrays and ``""`` for the dense matrices;
    ``owner`` names the container in error messages.
    """
    for metric, hi, allow_positive_infinity in (
        ("hpd_overlap", 1.0, False),
        ("predictive_pvalue", 1.0, False),
        ("kl_divergence", None, True),
    ):
        field = prefix + metric
        _validate_diagnostic_range(
            getattr(container, field),
            f"{owner}.{field}",
            lo=0.0,
            hi=hi,
            allow_nan=allow_nan,
            allow_positive_infinity=allow_positive_infinity,
        )


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
    event_likelihood : np.ndarray, shape (n_spikes, n_bins), optional
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
    event_likelihood: NDArray[np.floating] | None
    # Real-data path supplies wall-clock spike times alongside the
    # bin indices; simulated paths leave this ``None``.
    event_time: NDArray[np.floating] | None = None

    def __post_init__(self) -> None:
        n_spikes = self.event_time_ind.shape[0]
        for name in ("event_cell_ind", *_PER_EVENT_METRIC_NAMES):
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
        dense_names = ("hpd_overlap", "kl_divergence", "predictive_pvalue", "event_likelihood")
        dense_provided = [getattr(self, n) is not None for n in dense_names]
        if any(dense_provided) and not all(dense_provided):
            missing = [n for n, p in zip(dense_names, dense_provided, strict=True) if not p]
            raise ValueError(
                f"SpikeEventDiagnostics: dense matrices must be all-or-nothing; missing {missing}"
            )
        if self.hpd_overlap is not None:
            assert self.kl_divergence is not None  # narrowed by all-or-nothing
            assert self.predictive_pvalue is not None
            assert self.event_likelihood is not None
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
            if self.event_likelihood.shape[0] != n_spikes:
                raise ValueError(
                    f"event_likelihood leading dim {self.event_likelihood.shape[0]} "
                    f"!= n_spikes={n_spikes}"
                )
        _validate_metric_ranges(self, "SpikeEventDiagnostics", "event_", allow_nan=False)
        if self.hpd_overlap is not None:
            _validate_metric_ranges(self, "SpikeEventDiagnostics", "", allow_nan=True)
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
    posterior, predictive, combined_likelihood : np.ndarray, shape (n_time, n_bins)
        Dense distributions over position: the filtered posterior, the
        predictive distribution, and the normalized likelihood of all cells'
        spikes combined.
    hpd_overlap, kl_divergence, predictive_pvalue : np.ndarray, shape (n_time, n_cells)
        Dense per-cell diagnostic matrices; ``NaN`` where no spike.
    event_time_ind, event_cell_ind : np.ndarray, shape (n_spikes,)
        Time-bin / cell index for each spike event.
    event_hpd_overlap, event_kl_divergence, event_predictive_pvalue : np.ndarray, shape (n_spikes,)
        Per-event diagnostic values.
    event_likelihood : np.ndarray, shape (n_spikes, n_bins)
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
    combined_likelihood: NDArray[np.floating]
    hpd_overlap: NDArray[np.floating]
    kl_divergence: NDArray[np.floating]
    predictive_pvalue: NDArray[np.floating]
    event_time_ind: NDArray[np.intp]
    event_cell_ind: NDArray[np.intp]
    event_hpd_overlap: NDArray[np.floating]
    event_kl_divergence: NDArray[np.floating]
    event_predictive_pvalue: NDArray[np.floating]
    event_likelihood: NDArray[np.floating]

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
        for name in ("predictive", "combined_likelihood"):
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
        if self.event_likelihood.shape != (n_spikes, n_bins):
            raise ValueError(
                f"DecodingDiagnostics.event_likelihood shape "
                f"{self.event_likelihood.shape} != ({n_spikes}, {n_bins})"
            )
        # Value-range invariants on the per-cell metrics + their per-event
        # counterparts. NaN is legitimate at (t, cell) without a spike, so
        # the range check ignores NaN. A buggy decoder otherwise ships
        # out-of-range values that only surface much later (e.g., as a NaN
        # ``DiagnosticThresholds`` or a misleading hexbin).
        _validate_metric_ranges(self, "DecodingDiagnostics", "", allow_nan=True)
        _validate_metric_ranges(self, "DecodingDiagnostics", "event_", allow_nan=False)
        # Write-protect every backing buffer.
        for name in (
            "posterior",
            "predictive",
            "combined_likelihood",
            "hpd_overlap",
            "kl_divergence",
            "predictive_pvalue",
            "event_time_ind",
            "event_cell_ind",
            *_PER_EVENT_METRIC_NAMES,
            "event_likelihood",
        ):
            getattr(self, name).setflags(write=False)


def expand_spike_events(
    spike_counts: NDArray[np.integer],
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Expand a spike-count matrix into one event per spike.

    Every bin, including ``t=0``, contributes events; a bin with count ``k``
    contributes ``k`` repeated events, in row-major ``(time, cell)`` order.

    Parameters
    ----------
    spike_counts : np.ndarray, shape (n_time, n_cells)
        Non-negative spike count per time bin and cell.

    Returns
    -------
    event_time_ind, event_cell_ind : np.ndarray, shape (n_spikes,)
        Time-bin and cell index of each spike event.

    Examples
    --------
    >>> time_ind, cell_ind = expand_spike_events(np.array([[0, 2], [1, 0]]))
    >>> time_ind.tolist(), cell_ind.tolist()
    ([0, 0, 1], [1, 1, 0])
    """
    event_time_ind, event_cell_ind = np.nonzero(spike_counts)
    counts = spike_counts[event_time_ind, event_cell_ind].astype(np.intp)
    event_time_ind = np.repeat(event_time_ind, counts).astype(np.intp)
    event_cell_ind = np.repeat(event_cell_ind, counts).astype(np.intp)
    return event_time_ind, event_cell_ind


def compute_spike_event_diagnostics_from_rates(
    predictive: NDArray[np.floating],
    rates: NDArray[np.floating],
    event_time_ind: NDArray[np.intp],
    event_cell_ind: NDArray[np.intp],
    coverage: float = HPD_COVERAGE,
    include_dense_matrices: bool = True,
) -> SpikeEventDiagnostics:
    """Compute per-cell diagnostic metrics at spike times.

    Paper-side entry point shared by the simulated and real-data analyses. It
    delegates to :func:`statespacecheck.event_diagnostics`, which computes HPD
    overlap, KL divergence, and the rank-based predictive p-value for each spike
    event, and optionally scatters the results into dense per-(time, cell)
    matrices.

    Parameters
    ----------
    predictive : np.ndarray, shape (n_time, n_bins)
        Predictive distribution over position at each time.
    rates : np.ndarray, shape (n_bins, n_cells)
        Expected spike rate (spikes/bin) at each position for each cell.
    event_time_ind : np.ndarray, shape (n_spikes,)
        Time indices where spikes occurred.
    event_cell_ind : np.ndarray, shape (n_spikes,)
        Cell indices for each spike event.
    coverage : float, default ``HPD_COVERAGE``
        Coverage probability for HPD region computation.
    include_dense_matrices : bool, default True
        If True (default), also populate the (n_time, n_cells) ``hpd_overlap``,
        ``kl_divergence``, ``predictive_pvalue`` matrices and the (n_spikes, n_bins)
        ``event_likelihood`` on the returned dataclass. If False, those four
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
        - ``event_likelihood``: shape (n_spikes, n_bins), normalized
          likelihood distribution for each individual spike event

    Notes
    -----
    If multiple spikes occur in the same time/cell bin, pass repeated entries in
    ``event_time_ind`` and ``event_cell_ind`` so every observed spike contributes
    one event. See :func:`statespacecheck.event_diagnostics` for how each
    diagnostic is computed.
    """
    events = ssc.event_diagnostics(
        predictive,
        rates,
        event_time_ind,
        event_cell_ind,
        coverage=coverage,
        return_likelihood=include_dense_matrices,
    )

    def _dense(values: NDArray[np.floating]) -> NDArray[np.floating] | None:
        """Scatter per-event values into a NaN-filled (n_time, n_cells) matrix."""
        if not include_dense_matrices:
            return None
        matrix = np.full((predictive.shape[0], rates.shape[1]), np.nan)
        matrix[event_time_ind, event_cell_ind] = values
        return matrix

    return SpikeEventDiagnostics(
        event_time_ind=event_time_ind,
        event_cell_ind=event_cell_ind,
        event_hpd_overlap=events.hpd_overlap,
        event_kl_divergence=events.kl_divergence,
        event_predictive_pvalue=events.predictive_pvalue,
        hpd_overlap=_dense(events.hpd_overlap),
        kl_divergence=_dense(events.kl_divergence),
        predictive_pvalue=_dense(events.predictive_pvalue),
        event_likelihood=events.likelihood,
    )


@dataclass(frozen=True)
class DiagnosticThresholds:
    """Threshold values for diagnostic metrics.

    Figure 3 computes them with :func:`compute_baseline_diagnostic_thresholds`
    (baseline quantiles for HPD overlap and KL divergence, a fixed
    predictive p-value cutoff). Frozen so a downstream consumer cannot rebind
    a field mid-pipeline.

    Parameters
    ----------
    hpd_overlap : float
        HPD overlap threshold; must lie in ``[0, 1]`` (the underlying
        diagnostic is the overlap coefficient of the two HPD regions: the
        volume of their intersection divided by the volume of the smaller
        region). Lower values indicate worse fit.
    kl_divergence : float
        KL divergence threshold; must be non-negative finite. Higher
        values indicate worse fit.
    predictive_pvalue : float
        Predictive p-value threshold; must lie in ``[0, 1]``. Defaulted
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
    *,
    hpd_overlap: NDArray[np.floating],
    kl_divergence: NDArray[np.floating],
) -> DiagnosticThresholds:
    """Compute flag thresholds from baseline diagnostic values.

    The caller selects the baseline values (Figure 3 pools every spike event
    of the opening baseline window across its realizations); each array is
    flattened so a single threshold scalar applies to any cell or event:

    - HPD overlap threshold: 1st percentile (low values indicate misfit)
    - KL divergence threshold: 99th percentile (high values indicate misfit)
    - predictive_pvalue threshold: fixed at 0.05 (a conventional rank-statistic
      cutoff), not derived from the data

    Parameters
    ----------
    hpd_overlap : np.ndarray, any shape, keyword-only
        Baseline HPD-overlap values. NaN values are ignored.
    kl_divergence : np.ndarray, any shape, keyword-only
        Baseline KL-divergence values. NaN values are ignored.

    Returns
    -------
    thresholds : DiagnosticThresholds
        Threshold values for each diagnostic metric.

    Raises
    ------
    ValueError
        If either baseline has no finite values (thresholds would be NaN and
        downstream comparisons would silently evaluate False) or holds values
        :func:`statespacecheck.baseline_threshold` rejects, or if the resulting
        threshold is out of range (e.g. an infinite KL threshold).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> thresholds = compute_baseline_diagnostic_thresholds(
    ...     hpd_overlap=rng.uniform(0.5, 1.0, 500),
    ...     kl_divergence=rng.uniform(0.0, 2.0, 500),
    ... )
    >>> thresholds.predictive_pvalue  # Fixed at 0.05
    0.05
    """

    def _threshold(name: str, values: NDArray[np.floating], quantile: float) -> float:
        try:
            return ssc.baseline_threshold(values, quantile)
        except ValueError as err:
            raise ValueError(
                f"compute_baseline_diagnostic_thresholds: {name} baseline: {err}"
            ) from err

    hpd_overlap_threshold = _threshold("hpd_overlap", hpd_overlap, BASELINE_HPD_OVERLAP_QUANTILE)
    kl_divergence_threshold = _threshold(
        "kl_divergence", kl_divergence, BASELINE_KL_DIVERGENCE_QUANTILE
    )

    # Fixed rank-statistic cutoff; not derived from the data.
    predictive_pvalue_threshold = FIXED_PREDICTIVE_PVALUE_CUTOFF

    return DiagnosticThresholds(
        hpd_overlap=hpd_overlap_threshold,
        kl_divergence=kl_divergence_threshold,
        predictive_pvalue=predictive_pvalue_threshold,
    )
