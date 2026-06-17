"""Reusable figure-3 simulation driver.

The figure-3 demo simulates a hippocampal-style decoder under a
sequence of misfit conditions (remap, history-dependent firing,
drift, wide-dynamics noise). The simulation pipeline drives both
``scripts/generate_figure03.py`` and
``statespacecheck_paper.interactive.cache.build_simulated_cache``;
both call ``run_figure03_simulation`` so the figure and the
interactive viewer's simulated cache stay byte-identical.

The figure-generation script extends this with diagnostic threshold
computation + plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from statespacecheck_paper.analysis import (
    DecodeParams,
    Diagnostics,
    MisfitSchedule,
    MisfitWindow,
    PhaseBoundary,
    Thresholds,
    compute_thresholds,
    decode_and_diagnostics,
    extract_phase_flag_values,
    flag_fractions_from_values,
    get_remapped_pf_centers,
    summary_phase_windows,
)
from statespacecheck_paper.simulation import (
    gaussian_transition_matrix,
    placefield_rates,
    reflect_into_interval,
    simulate_spikes_history_dependent,
    simulate_spikes_position_tuned,
    simulate_walk,
)

# Canonical ordered phase labels — the public contract of
# ``SimulationResult["phase_labels"]``. ``run_figure03_simulation`` emits
# these by position (one per ``_add_phase`` call); tests and downstream
# code import this tuple rather than re-typing the strings.
PHASE_LABELS: tuple[str, ...] = (
    "Clean Baseline",
    "Remap Misfit",
    "Clean Recovery",
    "History-Dependent Firing",
    "Clean Recovery",
    "Drift Misfit",
    "Clean Recovery",
    "Wide Dynamics Noise",
)


@dataclass(frozen=True)
class SimulationResult:
    """Result of :func:`run_figure03_simulation`.

    Promoted from ``TypedDict`` to frozen dataclass so the load-bearing
    length invariants — one ``phase_labels`` entry per phase, boundaries
    delimit those phases, ``spikes`` and ``x_true`` share the timeline,
    and the final boundary equals the timeline length — are checked at
    construction. Without this, adding or removing a phase silently
    changes downstream lengths and the figure-3 pipeline would run with
    miscounted indices.
    """

    params: DecodeParams
    xs: NDArray[np.floating]
    x_true: NDArray[np.floating]
    spikes: NDArray[np.int_]
    metrics: Diagnostics
    # Sequence fields are declared as tuple so ``frozen=True``'s
    # immutability extends to the contents — list would leave
    # ``sim.phase_labels.append(...)`` and ``sim.phase_boundaries[-1] = 9999``
    # as silent invariant-breakers. Callers passing a list at construction
    # are coerced in __post_init__.
    phase_labels: tuple[str, ...]
    phase_boundaries: tuple[int, ...]

    def __post_init__(self) -> None:
        """Enforce length and timeline-consistency invariants.

        Also coerces the two sequence fields to tuple (in case the
        caller supplied a list) and validates each metrics array shares
        the spike timeline.
        """
        # Coerce list -> tuple so frozen=True's immutability extends to
        # the contents. ``object.__setattr__`` because frozen blocks the
        # normal binding.
        if not isinstance(self.phase_labels, tuple):
            object.__setattr__(self, "phase_labels", tuple(self.phase_labels))
        if not isinstance(self.phase_boundaries, tuple):
            object.__setattr__(self, "phase_boundaries", tuple(self.phase_boundaries))

        if self.phase_labels != PHASE_LABELS:
            raise ValueError(
                f"phase_labels must equal PHASE_LABELS in order; "
                f"got {list(self.phase_labels)!r} vs canonical {list(PHASE_LABELS)!r}"
            )
        if len(self.phase_boundaries) != len(self.phase_labels):
            raise ValueError(
                f"phase_boundaries length ({len(self.phase_boundaries)}) "
                f"must equal phase_labels length ({len(self.phase_labels)})."
            )
        n_time = self.x_true.shape[0]
        if self.spikes.shape[0] != n_time:
            raise ValueError(
                f"spikes timeline ({self.spikes.shape[0]}) must equal x_true timeline ({n_time})."
            )
        if self.phase_boundaries[-1] != n_time:
            raise ValueError(
                f"final phase boundary ({self.phase_boundaries[-1]}) must "
                f"equal x_true timeline ({n_time})."
            )
        # ``Diagnostics.__post_init__`` enforces shape agreement across
        # its own fields; cross-check that ``Diagnostics``'s leading dim
        # matches the ``x_true`` timeline supplied here.
        if self.metrics.posterior.shape[0] != n_time:
            raise ValueError(
                f"metrics.posterior leading dim {self.metrics.posterior.shape[0]} "
                f"does not match x_true timeline ({n_time})."
            )


def run_figure03_simulation(
    params: DecodeParams | None = None,
    *,
    seed: int | None = None,
) -> SimulationResult:
    """Run the figure-3 phased simulation and decode it.

    Phases (in order, with their misfit class):

    1. Clean Baseline
    2. **Remap Misfit** (observation: wrong place-field identities for
       a subset of cells)
    3. Clean Recovery
    4. **History-Dependent Firing Misfit** (observation: spikes
       generated with hard refractory + bursting; decoder still
       assumes Poisson. Per-spike spatial likelihood is unchanged,
       so the per-spike diagnostics largely miss this — deliberate
       demonstration of the spatial-only nature of the metrics.)
    5. Clean Recovery
    6. **Drift Misfit** (transition: trajectory has persistent velocity
       at AR(1) coefficient ``params.drift_momentum``; decoder assumes
       memoryless walk)
    7. Clean Recovery
    8. **Wide Dynamics Noise** (transition: decoder uses inflated
       transition matrix ``sigx_wide_dynamics ~ 40× baseline``;
       engineered to inflate KL while HPD overlap and the rank-based
       p-value stay near baseline — the KL false-positive case)

    Parameters
    ----------
    params : DecodeParams, optional
        Simulation configuration. If ``None``, uses default
        ``DecodeParams()``.
    seed : int, optional
        Override ``params.base_seed`` so callers can vary stochastic
        draws without mutating the params dataclass.

    Returns
    -------
    SimulationResult
        Dataclass with attributes ``params``, ``xs``, ``x_true``,
        ``spikes``, ``metrics``, ``phase_labels``, ``phase_boundaries``.
        Access via attribute (``sim.metrics``), not subscript.
    """
    if params is None:
        params = DecodeParams()
    base_seed = params.base_seed if seed is None else seed
    rng = np.random.default_rng(base_seed)

    if params.pf_centers is None:
        raise ValueError("params.pf_centers must be initialized")
    pf_centers = params.pf_centers

    xs = np.arange(params.xs_min, params.xs_max + params.xs_step, params.xs_step, dtype=float)
    transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred)
    transition_matrix_inflated = gaussian_transition_matrix(xs, params.sigx_wide_dynamics)

    phases: list[tuple[NDArray[np.floating], NDArray[np.int_]]] = []
    phase_labels: list[str] = []
    x_last: float = 0.0

    def _walk(n: int, sig: float) -> NDArray[np.floating]:
        return simulate_walk(n, sig, x_last, params.xs_min, params.xs_max, rng)

    def _spikes_position_tuned(x: NDArray[np.floating]) -> NDArray[np.int_]:
        return simulate_spikes_position_tuned(
            x, pf_centers, params.pf_width, params.rate_scale, rng
        )

    def _add_phase(x: NDArray[np.floating], sp: NDArray[np.int_]) -> None:
        """Append one phase; its label is ``PHASE_LABELS`` at this position."""
        nonlocal x_last
        phase_labels.append(PHASE_LABELS[len(phases)])
        phases.append((x, sp))
        x_last = float(x[-1])

    bnd = params.phase_boundaries

    # 1. Clean baseline
    n = bnd[PhaseBoundary.REMAP_START]
    x = _walk(n, params.sigx_pred)
    _add_phase(x, _spikes_position_tuned(x))

    # 2. Remap misfit — the spike *generation* is normal position-tuned;
    #    the decoder is the one that uses remapped PF centers during this
    #    window (via ``MisfitWindow`` below).
    n = bnd[PhaseBoundary.REMAP_END] - bnd[PhaseBoundary.REMAP_START]
    x = _walk(n, params.sigx_pred)
    _add_phase(x, _spikes_position_tuned(x))

    # 3. Clean recovery 1
    n = bnd[PhaseBoundary.RECOVERY1_END] - bnd[PhaseBoundary.REMAP_END]
    x = _walk(n, params.sigx_pred)
    _add_phase(x, _spikes_position_tuned(x))

    # 4. History-Dependent Firing Misfit
    #    Cells generate spikes via ``simulate_spikes_history_dependent``:
    #    hard 1-step (1 ms) refractory + 2-10 step (2-10 ms) burst window
    #    with 3× rate boost. Decoder still treats every spike as an
    #    independent Poisson draw at the cell's standard rate; the misfit
    #    lives in the *temporal* correlations and is largely invisible to
    #    per-spike spatial diagnostics.
    n = bnd[PhaseBoundary.HIST_DEP_END] - bnd[PhaseBoundary.RECOVERY1_END]
    x = _walk(n, params.sigx_pred)
    sp = simulate_spikes_history_dependent(x, pf_centers, params.pf_width, params.rate_scale, rng)
    _add_phase(x, sp)

    # 5. Clean recovery 2
    n = bnd[PhaseBoundary.RECOVERY2_END] - bnd[PhaseBoundary.HIST_DEP_END]
    x = _walk(n, params.sigx_pred)
    _add_phase(x, _spikes_position_tuned(x))

    # 6. Drift Misfit — persistent-velocity walk; decoder assumes memoryless.
    n = bnd[PhaseBoundary.DRIFT_END] - bnd[PhaseBoundary.RECOVERY2_END]
    momentum = params.drift_momentum
    x_mom = np.zeros(n)
    x_mom[0] = x_last
    velocity = 0.0
    for t in range(1, n):
        velocity = momentum * velocity + rng.normal(0, params.sigx_pred)
        x_mom[t] = x_mom[t - 1] + velocity
    x = reflect_into_interval(x_mom, float(params.xs_min), float(params.xs_max))
    _add_phase(x, _spikes_position_tuned(x))

    # 7. Clean recovery 3
    n = bnd[PhaseBoundary.RECOVERY3_END] - bnd[PhaseBoundary.DRIFT_END]
    x = _walk(n, params.sigx_pred)
    _add_phase(x, _spikes_position_tuned(x))

    # 8. Wide Dynamics Noise — decoder applies an inflated transition matrix
    #    (~40× baseline). Predictive becomes wide; per-spike likelihoods stay
    #    narrow at the firing cell's PF -> KL inflates strongly while HPD
    #    overlap and the rank-based p-value stay near baseline (KL
    #    false-positive case).
    n = bnd[PhaseBoundary.WIDE_DYNAMICS_END] - bnd[PhaseBoundary.RECOVERY3_END]
    x = _walk(n, params.sigx_pred)
    _add_phase(x, _spikes_position_tuned(x))

    x_true = np.concatenate([p_x for p_x, _ in phases], axis=0)
    spikes = np.vstack([p_s for _, p_s in phases])

    # The two decoder-side misfits as a single schedule:
    # - Remap: the decoder's posterior update uses remapped place-field
    #   centers (``decoder_rates``); diagnostics always reference the
    #   original Gaussian fields, so the mismatch surfaces.
    # - Wide dynamics noise: an inflated transition matrix only.
    remapped_rates = placefield_rates(
        xs,
        get_remapped_pf_centers(pf_centers, params.remap_from_to, active=True),
        params.pf_width,
        params.rate_scale,
    )
    misfit_schedule = MisfitSchedule(
        (
            MisfitWindow(
                bnd[PhaseBoundary.REMAP_START],
                bnd[PhaseBoundary.REMAP_END],
                decoder_rates=remapped_rates,
            ),
            MisfitWindow(
                bnd[PhaseBoundary.RECOVERY3_END],
                bnd[PhaseBoundary.WIDE_DYNAMICS_END],
                transition_matrix=transition_matrix_inflated,
            ),
        )
    )

    metrics = decode_and_diagnostics(
        spikes=spikes,
        xs=xs,
        transition_matrix=transition_matrix,
        pf_centers=pf_centers,
        pf_width=params.pf_width,
        rate_scale=params.rate_scale,
        misfit_schedule=misfit_schedule,
    )

    boundaries = np.cumsum([len(p_x) for p_x, _ in phases]).tolist()

    return SimulationResult(
        params=params,
        xs=xs,
        x_true=x_true,
        spikes=spikes,
        metrics=metrics,
        phase_labels=tuple(phase_labels),
        phase_boundaries=tuple(boundaries),
    )


@dataclass(frozen=True)
class StableSummary:
    """Stabilized Figure-3 thresholds and per-phase flag fractions.

    Aggregates ``n_realizations`` independent realizations of the figure-3
    simulation so the Figure-3b heatmap and its flag thresholds no longer
    depend on a single noisy run (a single run's KL 99th-percentile
    threshold varies ~17% across seeds).

    - ``thresholds`` are computed from the per-spike baseline diagnostics
      pooled across all realizations — a far more stable estimate of the
      baseline interval than one run's quantile.
    - ``frac_median`` is the median, across realizations, of the percent of
      spike events flagged in each phase column by each metric (each
      realization scored against the shared pooled-baseline ``thresholds``).
      The median is used in place of the mean because the remapping column
      is strongly trajectory-dependent and skewed across realizations.

    Parameters
    ----------
    thresholds : Thresholds
        Pooled-baseline flag thresholds.
    frac_median : np.ndarray, shape (3, n_columns)
        Median percent flagged. Rows follow
        :data:`statespacecheck_paper.analysis.SUMMARY_FLAG_METRICS`;
        columns follow
        :func:`statespacecheck_paper.analysis.summary_phase_windows`.
    n_realizations : int
        Number of realizations aggregated.

    Raises
    ------
    ValueError
        If ``frac_median`` is not 2-D, or ``n_realizations`` is not positive.
    """

    thresholds: Thresholds
    frac_median: NDArray[np.floating]
    n_realizations: int

    def __post_init__(self) -> None:
        if self.n_realizations < 1:
            raise ValueError(f"n_realizations must be >= 1; got {self.n_realizations}")
        if self.frac_median.ndim != 2:
            raise ValueError(
                f"StableSummary.frac_median must be 2-D (n_metrics, n_columns); "
                f"got shape {self.frac_median.shape}"
            )
        self.frac_median.setflags(write=False)


def estimate_stable_summary(
    params: DecodeParams,
    *,
    n_realizations: int = 100,
    base_seed: int | None = None,
) -> StableSummary:
    """Pool many realizations into stable Figure-3 thresholds and fractions.

    Runs ``n_realizations`` independent realizations of the figure-3
    simulation (seeds ``base_seed, base_seed + 1, ...``), pools their
    per-spike *baseline-window* diagnostics to compute the flag
    thresholds, then scores every realization's per-phase flag fractions
    against those shared thresholds and returns the across-realization
    median. A single pass holds only the finite per-spike values (not the
    dense ``Diagnostics``) per realization, so memory stays bounded even at
    large ``n_realizations``.

    Parameters
    ----------
    params : DecodeParams
        Simulation configuration. ``params.pf_centers`` must be set
        (the dataclass initializes it by default).
    n_realizations : int, default 100
        Number of independent realizations to aggregate. Must be >= 1.
    base_seed : int, optional
        First seed; subsequent realizations use consecutive seeds. If
        ``None``, uses ``params.base_seed`` so the canonical displayed run
        (seed ``params.base_seed``) is one of the aggregated realizations.

    Returns
    -------
    StableSummary
        Pooled thresholds and mean/SD per-phase flag fractions.

    Raises
    ------
    ValueError
        If ``n_realizations < 1``.
    """
    if n_realizations < 1:
        raise ValueError(f"n_realizations must be >= 1; got {n_realizations}")

    base = params.base_seed if base_seed is None else base_seed
    baseline_end = params.phase_boundaries[PhaseBoundary.REMAP_START]
    windows = summary_phase_windows(params)

    # ``compute_thresholds`` reads only hpd_overlap and kl_divergence (the
    # spike_prob threshold is the fixed 0.05 cutoff), but pool all three so
    # the dict is a faithful baseline sample if that ever changes.
    baseline_keys = ("hpd_overlap", "kl_divergence", "spike_prob")
    baseline_values: dict[str, list[NDArray[np.floating]]] = {key: [] for key in baseline_keys}
    per_realization_values: list[list[list[NDArray[np.floating]]]] = []

    for offset in range(n_realizations):
        sim = run_figure03_simulation(params, seed=base + offset)
        metrics = sim.metrics
        for key in baseline_keys:
            baseline_slice = np.asarray(getattr(metrics, key))[:baseline_end].ravel()
            baseline_values[key].append(baseline_slice[np.isfinite(baseline_slice)])
        per_realization_values.append(extract_phase_flag_values(metrics, windows))

    pooled_baseline = {key: np.concatenate(vals) for key, vals in baseline_values.items()}
    thresholds = compute_thresholds(
        pooled_baseline, baseline_end=pooled_baseline["hpd_overlap"].shape[0]
    )

    # (n_realizations, n_metrics, n_columns) flag-fraction stack.
    frac = np.stack(
        [flag_fractions_from_values(values, thresholds) for values in per_realization_values],
        axis=0,
    )
    return StableSummary(
        thresholds=thresholds,
        frac_median=np.median(frac, axis=0),
        n_realizations=n_realizations,
    )
