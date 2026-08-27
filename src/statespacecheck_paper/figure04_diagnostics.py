"""Real-data goodness-of-fit diagnostic computations.

Per-spike-event diagnostics for real neural recordings: temporal smoothing and
running averages, the spike-event expansion helpers, the HPD/KL/predictive-p-value
computation delegating to :mod:`statespacecheck_paper.diagnostics`, the mean
per-spike likelihood, the end-to-end per-model diagnostic driver, and the
two-decoder flag-agreement tabulation.

Examples
--------
>>> import numpy as np
>>> from statespacecheck_paper.figure04_diagnostics import gaussian_smooth
>>> data = np.random.randn(1000)
>>> smoothed = gaussian_smooth(data, sigma=0.02, sampling_frequency=500)
>>> smoothed.shape
(1000,)
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

from statespacecheck_paper.diagnostics import (
    SpikeEventDiagnostics,
    compute_normalized_spike_likelihood,
    compute_spike_event_diagnostics_from_rates,
)
from statespacecheck_paper.figure04_place_fields import (
    extract_shared_position_place_fields,
    get_state_marginalized_posterior,
)


def gaussian_smooth(
    data: NDArray[np.float64],
    sigma: float,
    sampling_frequency: float,
    axis: int = 0,
    truncate: int = 8,
) -> NDArray[np.float64]:
    """Apply 1D Gaussian convolution to data.

    The standard deviation of the gaussian is in the units of the sampling
    frequency. The function is a wrapper around scipy's `gaussian_filter1d`.
    The support is truncated at 8 by default, instead of 4 in `gaussian_filter1d`.

    Parameters
    ----------
    data : np.ndarray
        Input data to smooth.
    sigma : float
        Standard deviation of the Gaussian kernel in seconds.
    sampling_frequency : float
        Number of samples per second.
    axis : int, default=0
        Axis along which to apply the filter.
    truncate : int, default=8
        Truncate the filter at this many standard deviations.

    Returns
    -------
    smoothed_data : np.ndarray
        Gaussian-smoothed data with same shape as input.

    Examples
    --------
    >>> data = np.random.randn(1000)
    >>> smoothed = gaussian_smooth(data, sigma=0.01, sampling_frequency=1000)
    >>> smoothed.shape
    (1000,)
    """
    result: NDArray[np.float64] = gaussian_filter1d(
        data,
        sigma * sampling_frequency,
        truncate=truncate,
        axis=axis,
        mode="constant",
    )
    return result


def compute_running_average(
    metric: NDArray[np.floating],
    time: NDArray[np.float64],
    window_size: float = 0.050,
    event_times: NDArray[np.floating] | None = None,
    event_values: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute running average of per-cell diagnostic metric over time.

    Implements the event-weighted average formula from the manuscript:

        D(t) = sum(metric_k * I(t_k in window)) / sum(I(t_k in window))

    where the sum is over all spike events (time, cell) pairs and I(*) is the
    indicator function selecting events within the sliding window centered at t.
    Each spike event contributes equally regardless of how many cells fire in
    a given time bin.

    Parameters
    ----------
    metric : np.ndarray, shape (n_time, n_cells)
        Per-cell diagnostic metric values. NaN where cell has no spike.
    time : np.ndarray, shape (n_time,)
        Time values for each bin.
    window_size : float, default 0.050
        Size of the sliding window in seconds.
    event_times : np.ndarray, shape (n_events,), optional
        Exact event times. If provided with ``event_values``, the running
        average is computed directly over events instead of bin/cell matrix
        entries.
    event_values : np.ndarray, shape (n_events,), optional
        Diagnostic values for each event.

    Returns
    -------
    running_avg : np.ndarray, shape (n_time,)
        Running average of the metric over time. NaN where no events fall
        within the window.
    time_out : np.ndarray, shape (n_time,)
        Time values (same as input, for convenience).

    Notes
    -----
    The implementation computes sum(values) and count(events) per time bin,
    then applies a boxcar filter to both before dividing. This is equivalent
    to the event-weighted sliding window average but runs in O(n) time.
    Edge effects are handled by ``uniform_filter1d(mode="constant")``, which
    zero-pads outside the array. Bins with no events in the window produce NaN.

    Examples
    --------
    >>> import numpy as np
    >>> n_time, n_cells = 100, 10
    >>> metric = np.random.rand(n_time, n_cells)
    >>> metric[::2, :] = np.nan  # Sparse spikes
    >>> time = np.linspace(0, 1, n_time)
    >>> running_avg, time_out = compute_running_average(metric, time, window_size=0.1)
    >>> running_avg.shape
    (100,)
    """
    n_time_pts = len(time)

    if event_times is not None and event_values is not None:
        event_times = np.asarray(event_times, dtype=np.float64)
        event_values = np.asarray(event_values, dtype=np.float64)

        valid_events = ~np.isnan(event_values)
        event_times = event_times[valid_events]
        event_values = event_values[valid_events]

        if len(event_times) == 0:
            return np.full(n_time_pts, np.nan), time.copy()

        sort_ind = np.argsort(event_times)
        sorted_times = event_times[sort_ind]
        sorted_values = event_values[sort_ind]
        cumsum = np.concatenate(([0.0], np.cumsum(sorted_values)))

        half_window = window_size / 2.0
        starts = np.searchsorted(sorted_times, time - half_window, side="left")
        stops = np.searchsorted(sorted_times, time + half_window, side="right")
        counts = stops - starts
        sums = cumsum[stops] - cumsum[starts]

        running_avg = np.full(n_time_pts, np.nan)
        has_events = counts > 0
        running_avg[has_events] = sums[has_events] / counts[has_events]

        return running_avg, time.copy()

    event_values = np.where(np.isnan(metric), 0.0, metric)
    event_counts = np.where(np.isnan(metric), 0.0, 1.0)
    bin_sum = event_values.sum(axis=1)  # (n_time,)
    bin_count = event_counts.sum(axis=1)  # (n_time,)

    dt = float(np.median(np.diff(time))) if len(time) > 1 else 1.0
    window_bins = max(1, int(np.round(window_size / dt)))

    # Zero-pad mode rather than nearest-value so edge bins have fewer
    # events instead of inflated counts from extended-value extension.
    windowed_sum = uniform_filter1d(bin_sum, size=window_bins, mode="constant")
    windowed_count = uniform_filter1d(bin_count, size=window_bins, mode="constant")

    running_avg = np.full(n_time_pts, np.nan)
    has_events = windowed_count > 0
    running_avg[has_events] = windowed_sum[has_events] / windowed_count[has_events]

    return running_avg, time.copy()


def _get_spike_events_from_counts(
    spike_counts: NDArray[np.int64],
    time: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64] | None]:
    """Expand binned spike counts into one event per spike."""
    spike_time_ind, spike_cell_ind = np.nonzero(spike_counts)
    counts = spike_counts[spike_time_ind, spike_cell_ind].astype(np.intp)

    spike_time_ind = np.repeat(spike_time_ind, counts).astype(np.intp)
    spike_cell_ind = np.repeat(spike_cell_ind, counts).astype(np.intp)
    event_times = None if time is None else np.asarray(time, dtype=np.float64)[spike_time_ind]

    return spike_time_ind, spike_cell_ind, event_times


def _get_spike_events_from_spike_times(
    spike_times: list[NDArray[np.float64]],
    time: NDArray[np.float64],
) -> tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float64]]:
    """Map exact spike timestamps to predictive-posterior time indices."""
    time = np.asarray(time, dtype=np.float64)
    spike_time_inds = []
    spike_cell_inds = []
    event_times = []

    for cell_ind, cell_spike_times in enumerate(spike_times):
        cell_spike_times = np.asarray(cell_spike_times, dtype=np.float64)
        in_bounds = (cell_spike_times >= time[0]) & (cell_spike_times <= time[-1])
        cell_event_times = cell_spike_times[in_bounds]
        cell_time_inds = np.searchsorted(time, cell_event_times, side="right") - 1
        cell_time_inds = np.clip(cell_time_inds, 0, len(time) - 1)

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
    predictive_posterior: NDArray[np.float64],
    spike_counts: NDArray[np.int64],
    place_fields: NDArray[np.float64],
    coverage: float = 0.95,
    spike_times: list[NDArray[np.float64]] | None = None,
    time: NDArray[np.float64] | None = None,
    include_dense_matrices: bool = True,
) -> SpikeEventDiagnostics:
    """Compute per-cell diagnostic metrics for model checking.

    Computes HPD overlap, KL divergence, and predictive p-value ranking for each
    spike event. Matrix outputs are retained for backward-compatible plotting,
    and event arrays preserve one row per spike with exact timestamps when
    ``spike_times`` and ``time`` are supplied.

    Parameters
    ----------
    predictive_posterior : np.ndarray, shape (n_time, n_bins)
        State-marginalized predictive posterior distribution over position.
    spike_counts : np.ndarray, shape (n_time, n_cells)
        Spike count for each cell at each time point.
    place_fields : np.ndarray, shape (n_cells, n_bins)
        Expected spike count at each position bin for each cell (spikes/bin).
        This is the format returned by non_local_detector.
    coverage : float, default 0.95
        Coverage probability for HPD region computation.
    spike_times : list of np.ndarray, optional
        Exact spike timestamps for each cell. If supplied, diagnostics are
        computed per spike event and plotted at exact spike times.
    time : np.ndarray, optional
        Decoder time grid used to map spike timestamps to predictive posterior
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
        - ``per_spike_likelihood``: shape (n_spikes, n_bins), normalized
          per-event Poisson likelihood.

        When ``include_dense_matrices=False`` those four dense fields
        are ``None`` together (the all-or-nothing invariant in
        ``SpikeEventDiagnostics.__post_init__``).

    Notes
    -----
    The likelihood P(k=1 | position) is computed once for each observed spike.
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
    predictive_posterior = np.asarray(predictive_posterior)
    spike_counts = np.asarray(spike_counts)

    event_times: NDArray[np.float64] | None
    if spike_times is not None:
        if time is None:
            raise ValueError("time must be provided when spike_times is provided")
        spike_time_ind, spike_cell_ind, event_times = _get_spike_events_from_spike_times(
            spike_times, time
        )
    else:
        spike_time_ind, spike_cell_ind, event_times = _get_spike_events_from_counts(
            spike_counts,
            time,
        )

    result = compute_spike_event_diagnostics_from_rates(
        predictive_posterior,
        place_fields.T,  # (n_bins, n_cells)
        spike_time_ind.astype(np.intp),
        spike_cell_ind.astype(np.intp),
        coverage=coverage,
        include_dense_matrices=include_dense_matrices,
    )

    return dataclasses.replace(result, event_time=event_times)


def mean_per_spike_likelihood_by_time(
    spike_counts: NDArray[np.int64],
    place_fields: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Mean normalized per-spike likelihood in each time bin.

    Each cell's place field is turned into the normalized one-spike Poisson
    likelihood over position via
    :func:`statespacecheck_paper.diagnostics.compute_normalized_spike_likelihood`
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
    pf_norm = compute_normalized_spike_likelihood(pf)

    counts = np.asarray(spike_counts, dtype=np.float64)
    n_per_bin = counts.sum(axis=1)
    has_spikes = n_per_bin > 0

    weighted = counts @ pf_norm
    mean_likelihood = np.zeros_like(weighted)
    mean_likelihood[has_spikes] = weighted[has_spikes] / n_per_bin[has_spikes, np.newaxis]
    return mean_likelihood, has_spikes


def compute_model_diagnostics(
    model: Any,
    results: Any,
    spike_counts: NDArray[np.int64],
    time: NDArray[np.float64],
    spike_times: list[NDArray[np.float64]] | None = None,
) -> SpikeEventDiagnostics:
    """Compute per-cell diagnostics for a fitted decoder model.

    The decoder itself may operate over a joint discrete-state-by-position
    space. For diagnostics, the predictive posterior is marginalized over the
    discrete state and compared with one shared position-dependent observation
    likelihood. This makes the metric domain identical for models with
    different numbers of discrete states.

    Parameters
    ----------
    model : decoder model
        Fitted SortedSpikesDecoder or ContFragSortedSpikesClassifier.
    results : xr.Dataset
        Decoding results from model.predict().
    spike_counts : np.ndarray, shape (n_time, n_cells)
        Spike count matrix.
    time : np.ndarray, shape (n_time,)
        Time values.
    spike_times : list of np.ndarray, optional
        Exact spike timestamps for each cell. If supplied, diagnostics are
        computed as one event per spike instead of one event per nonzero bin.

    Returns
    -------
    diagnostics : SpikeEventDiagnostics
        See :func:`compute_spike_event_diagnostics` for the schema. The
        ``event_time`` field carries either the original ``spike_times``
        (when supplied) or the decoder-grid time at each event's index.

    Examples
    --------
    >>> # Requires fitted model and decoding results
    >>> # diagnostics = compute_model_diagnostics(model, results, spike_counts, time)
    >>> # diagnostics.hpd_overlap.shape  # (n_time, n_cells)
    """
    place_fields, _ = extract_shared_position_place_fields(model)
    predictive_posterior = get_state_marginalized_posterior(results, "predictive")

    if predictive_posterior.shape[1] != place_fields.shape[1]:
        raise ValueError(
            f"Position-marginal predictive posterior has "
            f"{predictive_posterior.shape[1]} bins but the shared observation "
            f"likelihood has {place_fields.shape[1]}."
        )

    # Compute diagnostics using actual spike counts
    # place_fields are already in spikes per time bin from non_local_detector
    diagnostics = compute_spike_event_diagnostics(
        predictive_posterior,
        spike_counts,
        place_fields,
        spike_times=spike_times,
        time=time,
    )

    return diagnostics


@dataclasses.dataclass(frozen=True)
class FlagConfusion:
    """Per-spike flag agreement between two decoders for one diagnostic metric.

    A spike is "flagged" when its per-spike diagnostic crosses ``threshold`` in
    the direction of worse fit. The four counts partition every spike event
    (finite in both models) by whether model A and/or model B flags it.
    ``a_only`` is the rescue quadrant: spikes flagged by model A but not model B.

    Attributes
    ----------
    metric : str
        Diagnostic name (e.g. ``"hpd_overlap"``).
    threshold : float
        Flag threshold applied to the raw per-spike diagnostic.
    n : int
        Number of spike events finite in both models.
    both, a_only, b_only, neither : int
        Counts of spikes flagged by both decoders, by model A only, by model B
        only, and by neither. They sum to ``n``.
    """

    metric: str
    threshold: float
    n: int
    both: int
    a_only: int
    b_only: int
    neither: int

    @property
    def rescue_rate(self) -> float:
        """Fraction of model-A-flagged spikes that model B does not flag.

        Returns ``nan`` when model A flags no spikes.
        """
        a_flagged = self.a_only + self.both
        return self.a_only / a_flagged if a_flagged else float("nan")


def compute_flag_confusion(
    diagnostics_a: SpikeEventDiagnostics,
    diagnostics_b: SpikeEventDiagnostics,
    metric: str,
    threshold: float,
    *,
    worse_when: Literal["below", "above"],
) -> FlagConfusion:
    """Tabulate per-spike flag agreement between two decoders for one metric.

    Parameters
    ----------
    diagnostics_a, diagnostics_b : SpikeEventDiagnostics
        Per-spike diagnostics for the two decoders, carrying the same spike
        events in the same order (e.g. Continuous vs Continuous--Fragmented).
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
        The 2x2 flag agreement (``a`` = model A, ``b`` = model B).

    Raises
    ------
    ValueError
        If the two diagnostics carry different numbers of spike events, or if
        ``worse_when`` is not ``"below"`` or ``"above"``.
    """
    if worse_when not in ("below", "above"):
        raise ValueError(f"worse_when must be 'below' or 'above', got {worse_when!r}")

    event_key = f"event_{metric}"
    a = np.asarray(getattr(diagnostics_a, event_key), dtype=np.float64)
    b = np.asarray(getattr(diagnostics_b, event_key), dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(
            f"diagnostics_a[{event_key!r}] and diagnostics_b[{event_key!r}] must carry "
            f"the same set of spike events in the same order; got {a.shape} vs {b.shape}."
        )

    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if worse_when == "below":
        flag_a, flag_b = a <= threshold, b <= threshold
    else:
        flag_a, flag_b = a >= threshold, b >= threshold

    return FlagConfusion(
        metric=metric,
        threshold=float(threshold),
        n=int(a.size),
        both=int(np.sum(flag_a & flag_b)),
        a_only=int(np.sum(flag_a & ~flag_b)),
        b_only=int(np.sum(~flag_a & flag_b)),
        neither=int(np.sum(~flag_a & ~flag_b)),
    )
