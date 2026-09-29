"""Real-data goodness-of-fit diagnostic computations.

Per-spike-event diagnostics for real neural recordings: the spike-event
expansion from exact spike times, the HPD/KL/predictive-p-value computation
delegating to :mod:`statespacecheck_paper.diagnostics`, the mean per-spike
likelihood, the driver that diagnoses a (cached) decode's results, and the
two-decoder flag-agreement tabulation.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import statespacecheck as ssc
from numpy.typing import NDArray

from statespacecheck_paper.diagnostics import (
    HPD_COVERAGE,
    FlagDirection,
    SpikeEventDiagnostics,
    compute_spike_event_diagnostics_from_rates,
    expand_spike_events,
    flag_mask,
)
from statespacecheck_paper.figure04_place_fields import get_state_marginalized_posterior


def _get_spike_events_from_spike_times(
    spike_times: list[NDArray[np.float64]],
    time: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """Map exact spike timestamps to predictive-distribution time indices.

    Uses the same bin assignment as the decoder's spike binning
    (``non_local_detector.likelihoods.common.get_spikecount_per_time_bin``):
    spikes within ``[time[0], time[-1]]`` are kept and assigned with
    ``np.digitize(spike_times, time[1:-1])``, so a spike at or after
    ``time[-2]`` -- including one exactly at the final timestamp -- falls in
    the penultimate row, exactly where the decoder counted it. Per-event
    diagnostics therefore compare each spike with the prediction the decoder
    actually updated with that spike.
    """
    time = np.asarray(time, dtype=np.float64)
    spike_time_inds = []
    spike_cell_inds = []
    event_times = []

    for cell_ind, cell_spike_times in enumerate(spike_times):
        cell_spike_times = np.asarray(cell_spike_times, dtype=np.float64)
        in_bounds = (cell_spike_times >= time[0]) & (cell_spike_times <= time[-1])
        cell_event_times = cell_spike_times[in_bounds]
        cell_time_inds = np.digitize(cell_event_times, time[1:-1])

        spike_time_inds.append(cell_time_inds.astype(np.intp))
        spike_cell_inds.append(np.full(len(cell_event_times), cell_ind, dtype=np.intp))
        event_times.append(cell_event_times)

    if not event_times:
        return (
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.intp),
            np.empty(0, dtype=np.float64),
        )

    spike_time_ind = np.concatenate(spike_time_inds)
    spike_cell_ind = np.concatenate(spike_cell_inds)
    event_time = np.concatenate(event_times)
    sort_ind = np.argsort(event_time)

    return spike_time_ind[sort_ind], spike_cell_ind[sort_ind], event_time[sort_ind]


def compute_spike_event_diagnostics(
    predictive: NDArray[np.float64],
    spike_counts: NDArray[np.int64],
    place_fields: NDArray[np.float64],
    coverage: float = HPD_COVERAGE,
    spike_times: list[NDArray[np.float64]] | None = None,
    time: NDArray[np.float64] | None = None,
    include_dense_matrices: bool = True,
) -> SpikeEventDiagnostics:
    """Compute per-cell diagnostic metrics for model checking.

    Computes HPD overlap, KL divergence, and the rank-based predictive p-value for each
    spike event. Matrix outputs are retained for backward-compatible plotting,
    and event arrays preserve one row per spike with exact timestamps when
    ``spike_times`` and ``time`` are supplied.

    Parameters
    ----------
    predictive : np.ndarray, shape (n_time, n_bins)
        State-marginalized predictive distribution over position.
    spike_counts : np.ndarray, shape (n_time, n_cells)
        Spike count for each cell at each time point.
    place_fields : np.ndarray, shape (n_cells, n_bins)
        Expected spike count at each position bin for each cell (spikes/bin).
        This is the format returned by non_local_detector.
    coverage : float, default ``HPD_COVERAGE``
        Coverage probability for HPD region computation.
    spike_times : list of np.ndarray, optional
        Exact spike timestamps for each cell. If supplied, diagnostics are
        computed per spike event and plotted at exact spike times.
    time : np.ndarray, optional
        Decoder time grid used to map spike timestamps to predictive-distribution
        rows. Required when ``spike_times`` is supplied.
    include_dense_matrices : bool, default True
        Forwarded to ``compute_spike_event_diagnostics_from_rates``. Set False
        when only the per-spike event arrays are needed (avoids the
        ``(n_time, n_cells)`` allocations, which can be hundreds of MB
        for full-session real-data builds).

    Returns
    -------
    diagnostics : SpikeEventDiagnostics
        Always populated:

        - ``event_time_ind`` (n_spikes,) and ``event_cell_ind`` (n_spikes,):
          decoder-bin and cell indices per spike event.
        - ``event_hpd_overlap``, ``event_kl_divergence``, ``event_predictive_pvalue``:
          shape (n_spikes,), one value per spike event.

        Optionally populated:

        - ``event_time``: shape (n_spikes,), exact wall-clock spike time.
          Populated when either ``spike_times`` (preferred) or ``time``
          alone is supplied; ``None`` when both are ``None``.

        When ``include_dense_matrices`` (the default), additionally:

        - ``hpd_overlap``, ``kl_divergence``, ``predictive_pvalue``: shape
          (n_time, n_cells), NaN where the cell has no spike at that
          timestep.
        - ``event_likelihood``: shape (n_spikes, n_bins), normalized
          per-event intensity likelihood.

        When ``include_dense_matrices=False`` those four dense fields
        are ``None`` together (the all-or-nothing invariant in
        ``SpikeEventDiagnostics.__post_init__``).

    Notes
    -----
    The normalized event-intensity likelihood is computed once for each
    observed spike.
    Multiple spikes from the same cell in the same decoder bin contribute
    multiple event rows rather than being collapsed into one binned count.

    This function delegates to ``compute_spike_event_diagnostics_from_rates`` in
    ``diagnostics.py`` to ensure identical computation for simulated and real data.

    Examples
    --------
    >>> import numpy as np
    >>> n_time, n_bins, n_cells = 100, 50, 10
    >>> predictive = np.random.dirichlet(np.ones(n_bins), size=n_time)
    >>> place_fields = np.random.rand(n_cells, n_bins) * 10
    >>> spike_counts = np.random.poisson(0.5, (n_time, n_cells))
    >>> diagnostics = compute_spike_event_diagnostics(
    ...     predictive, spike_counts, place_fields
    ... )
    >>> diagnostics.hpd_overlap.shape
    (100, 10)
    """
    # Ensure all inputs are NumPy arrays (handles JAX arrays from decoder)
    predictive = np.asarray(predictive)
    spike_counts = np.asarray(spike_counts)

    event_times: NDArray[np.float64] | None
    if spike_times is not None:
        if time is None:
            raise ValueError("time must be provided when spike_times is provided")
        spike_time_ind, spike_cell_ind, event_times = _get_spike_events_from_spike_times(
            spike_times, time
        )
    else:
        spike_time_ind, spike_cell_ind = expand_spike_events(spike_counts)
        event_times = None if time is None else np.asarray(time, dtype=np.float64)[spike_time_ind]

    result = compute_spike_event_diagnostics_from_rates(
        predictive,
        place_fields.T,  # (n_bins, n_cells)
        spike_time_ind.astype(np.intp),
        spike_cell_ind.astype(np.intp),
        coverage=coverage,
        include_dense_matrices=include_dense_matrices,
    )

    return dataclasses.replace(result, event_time=event_times)


def mean_event_likelihood_by_time(
    spike_counts: NDArray[np.int64],
    place_fields: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Mean normalized per-spike likelihood in each time bin.

    Each cell's place field is turned into the normalized single-event
    likelihood over position via
    :func:`statespacecheck.event_likelihood`
    --- the exact quantity the diagnostics compare against the predictive
    distribution. In every time bin the normalized likelihoods of the spiking
    cells are averaged, weighted by spike count, so a bin with several spikes
    contributes the mean of their normalized likelihoods. This matches the
    simulation figure's likelihood row, and it depends only on the observation
    model, so it is identical across decoders that share place fields.

    Parameters
    ----------
    spike_counts : np.ndarray, shape (n_time, n_cells)
        Spike count per cell per time bin. Columns must be in the same cell
        order as the rows of ``place_fields``.
    place_fields : np.ndarray, shape (n_cells, n_bins)
        Per-cell place fields (expected spikes per bin) over the position grid.

    Returns
    -------
    mean_likelihood : np.ndarray, shape (n_time, n_bins)
        Mean normalized per-spike likelihood over position in each time bin.
        Rows for time bins with no spikes are all zero.
    has_spikes : np.ndarray, shape (n_time,)
        True in time bins containing at least one spike.
    """
    pf = np.asarray(place_fields, dtype=np.float64)
    pf_norm = ssc.event_likelihood(pf)

    counts = np.asarray(spike_counts, dtype=np.float64)
    n_per_bin = counts.sum(axis=1)
    has_spikes = n_per_bin > 0

    weighted = counts @ pf_norm
    mean_likelihood = np.zeros_like(weighted)
    mean_likelihood[has_spikes] = weighted[has_spikes] / n_per_bin[has_spikes, np.newaxis]
    return mean_likelihood, has_spikes


def compute_results_diagnostics(
    results: Any,
    place_fields: NDArray[np.float64],
    spike_counts: NDArray[np.int64],
    time: NDArray[np.float64],
    spike_times: list[NDArray[np.float64]] | None = None,
    *,
    coverage: float = HPD_COVERAGE,
) -> SpikeEventDiagnostics:
    """Compute per-spike diagnostics from decode outputs and shared place fields.

    Takes the decoder's ``predict`` output and the one shared position-dependent
    observation likelihood (already restricted to track-interior bins), so the
    diagnostics can be (re)computed from a cached decode without the fitted
    model object. The predictive distribution is marginalized over any discrete
    dynamics mode before comparison, so models with different numbers of
    modes are diagnosed over the same position grid.

    Parameters
    ----------
    results : xr.Dataset
        Decoding results from ``model.predict()``.
    place_fields : np.ndarray, shape (n_cells, n_bins)
        Shared interior place fields (expected spikes per bin), as returned by
        :func:`~figure04_place_fields.extract_shared_position_place_fields`.
    spike_counts : np.ndarray, shape (n_time, n_cells)
        Spike count matrix.
    time : np.ndarray, shape (n_time,)
        Decoder time grid.
    spike_times : list of np.ndarray, optional
        Exact spike timestamps for each cell. If supplied, diagnostics are
        computed as one event per spike instead of one event per nonzero bin.
    coverage : float, default ``HPD_COVERAGE``
        HPD-region coverage used by the HPD-overlap diagnostic.

    Returns
    -------
    diagnostics : SpikeEventDiagnostics
        See :func:`compute_spike_event_diagnostics` for the schema. The dense
        ``(n_time, n_cells)`` matrices, hundreds of MB for a full recording and
        read by no consumer, are left ``None``.
    """
    predictive = get_state_marginalized_posterior(results, "predictive")
    place_fields = np.asarray(place_fields, dtype=np.float64)
    if predictive.shape[1] != place_fields.shape[1]:
        raise ValueError(
            f"Position-marginal predictive distribution has "
            f"{predictive.shape[1]} bins but the shared observation "
            f"likelihood has {place_fields.shape[1]}."
        )
    return compute_spike_event_diagnostics(
        predictive,
        spike_counts,
        place_fields,
        coverage=coverage,
        spike_times=spike_times,
        time=time,
        include_dense_matrices=False,
    )


@dataclasses.dataclass(frozen=True)
class FlagConfusion:
    """Per-spike flag agreement between two decoders for one diagnostic metric.

    A spike is "flagged" when its per-spike diagnostic crosses ``threshold`` in
    the direction of worse fit. The four counts partition every aligned spike
    event by whether model A and/or model B flags it. Model A is the reference
    decoder and model B the comparison decoder; in Figure 4, A is the Continuous
    and B the Continuous-Fragmented model. ``rescued`` counts spikes the
    reference flags but the comparison does not (Figure 4's "flagged by Cont.
    only" quadrant), and ``newly_flagged`` spikes the comparison flags but the
    reference does not.

    Attributes
    ----------
    metric : str
        Diagnostic name (e.g. ``"hpd_overlap"``).
    threshold : float
        Flag threshold applied to the raw per-spike diagnostic.
    n : int
        Number of aligned spike events.
    both, rescued, newly_flagged, neither : int
        Counts of spikes flagged by both decoders, by model A (the reference)
        only, by model B (the comparison) only, and by neither. They sum to
        ``n``.
    """

    metric: str
    threshold: float
    n: int
    both: int
    rescued: int
    newly_flagged: int
    neither: int

    @property
    def rescue_rate(self) -> float:
        """Fraction of model-A-flagged spikes that model B does not flag (``rescued``).

        Returns ``nan`` when model A flags no spikes.
        """
        a_flagged = self.rescued + self.both
        return self.rescued / a_flagged if a_flagged else float("nan")


def compute_flag_confusion(
    diagnostics_a: SpikeEventDiagnostics,
    diagnostics_b: SpikeEventDiagnostics,
    metric: str,
    threshold: float,
    *,
    worse_when: FlagDirection,
) -> FlagConfusion:
    """Tabulate per-spike flag agreement between two decoders for one metric.

    Parameters
    ----------
    diagnostics_a, diagnostics_b : SpikeEventDiagnostics
        Per-spike diagnostics for the reference (A) and comparison (B)
        decoders, carrying the same spike events in the same order (e.g.
        Continuous vs Continuous--Fragmented).
    metric : str
        Diagnostic base name; the per-spike array ``event_{metric}`` is used.
    threshold : float
        Flag threshold applied to the raw per-spike diagnostic.
    worse_when : {"below", "above"}
        Whether values at or below, or at or above, ``threshold`` indicate
        worse fit (i.e. a flag). HPD overlap and the predictive p-value use
        ``"below"``; the KL divergence uses ``"above"``.

    Returns
    -------
    FlagConfusion
        The 2x2 flag agreement: ``rescued`` is flagged by A only,
        ``newly_flagged`` by B only.

    Raises
    ------
    ValueError
        If the two diagnostics carry different numbers of spike events, or if
        ``worse_when`` is not ``"below"`` or ``"above"`` (from :func:`flag_mask`).
    """
    event_key = f"event_{metric}"
    a = np.asarray(getattr(diagnostics_a, event_key), dtype=np.float64)
    b = np.asarray(getattr(diagnostics_b, event_key), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"diagnostics_a[{event_key!r}] and diagnostics_b[{event_key!r}] must carry "
            f"the same set of spike events in the same order; got {a.shape} vs {b.shape}."
        )
    if not np.array_equal(diagnostics_a.event_time_ind, diagnostics_b.event_time_ind) or not (
        np.array_equal(diagnostics_a.event_cell_ind, diagnostics_b.event_cell_ind)
    ):
        raise ValueError(
            "diagnostics_a and diagnostics_b must carry identical event_time_ind and "
            "event_cell_ind arrays in the same order"
        )
    # Belt-and-suspenders: SpikeEventDiagnostics.__post_init__ already rejects
    # NaN/-inf in the event arrays, so this cannot fire for a constructed
    # dataclass. Kept as a cheap guard in case a metric ever names an
    # unvalidated event array.
    if (
        np.any(np.isnan(a))
        or np.any(np.isnan(b))
        or np.any(np.isneginf(a))
        or np.any(np.isneginf(b))
    ):
        raise ValueError(f"{event_key} contains an undefined per-event value")

    flag_a = flag_mask(a, threshold, worse_when)
    flag_b = flag_mask(b, threshold, worse_when)

    return FlagConfusion(
        metric=metric,
        threshold=float(threshold),
        n=int(a.size),
        both=int(np.sum(flag_a & flag_b)),
        rescued=int(np.sum(flag_a & ~flag_b)),
        newly_flagged=int(np.sum(~flag_a & flag_b)),
        neither=int(np.sum(~flag_a & ~flag_b)),
    )
