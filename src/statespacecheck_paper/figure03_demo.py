"""Reusable figure-3 simulation driver.

The figure-3 demo simulates a hippocampal-style decoder under a
sequence of misfit conditions (remap, history-dependent firing, drift) and
two specificity controls (a replay event embedded in clean-recovery 2 and a
final sparse-population epoch). The simulation pipeline drives both
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
    replay_window,
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
    "Sparse Population",
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
    # Fixed sparse-population field centers; let the raster sort all cells by
    # location without deriving a field center from the realized trajectory.
    sparse_cell_centers: tuple[float, ...] = ()

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
    2. **Remap Misfit** (observation: all place-field identities undergo
       one fixed, spatially incoherent permutation)
    3. Clean Recovery
    4. **History-Dependent Firing Misfit** (observation: spikes
       generated with hard refractory + bursting; decoder still
       assumes Poisson. Per-spike spatial likelihood is unchanged,
       so the per-spike diagnostics largely miss this — deliberate
       demonstration of the spatial-only nature of the metrics.)
    5. Clean Recovery (contains the **Replay control**: an out-and-back
       trajectory sweep while the animal is immobile. The decoder tracks the
       sweep, so the decoded position departs from the true fixed position
       yet stays consistent with each spike's likelihood — a benign
       decoded-vs-true divergence that none of the metrics should flag.)
    6. **Drift Misfit** (transition: trajectory has persistent velocity
       at AR(1) coefficient ``params.drift_momentum``; decoder assumes
       memoryless walk)
    7. Clean Recovery
    8. **Sparse Population** (control: the ordinary ensemble is quiet while a
       small population of pre-existing, sharply tuned cells clustered at one
       location fires sparsely — each an independent Poisson process, the
       decoder's rates matching exactly. The prediction spreads between the
       isolated spikes, and each spike's narrow likelihood remains contained
       within it. KL responds to the concentration difference while HPD overlap
       and the rank-based p-value remain consistent.)

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
    #    the decoder is the one that uses randomly scrambled PF centers
    #    during this window (via ``MisfitWindow`` below).
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

    # 5. Clean recovery 2 — with a replay event. The animal is immobile
    #    (true position held fixed) while a coherent trajectory sweeps the
    #    track out-and-back at ``replay_speed`` a.u./step (slow enough for
    #    the narrow transition to track). The decoder follows the sweep, so
    #    the *decoded* position departs from the fixed true position while
    #    every metric stays at baseline — a decoded-vs-true divergence is not
    #    a model misspecification. Spikes during the sweep fire at the
    #    elevated ``replay_rate_scale`` to densely sample the trajectory.
    n = bnd[PhaseBoundary.RECOVERY2_END] - bnd[PhaseBoundary.HIST_DEP_END]
    # Local (within-phase) replay bounds derived from the shared global
    # ``replay_window`` helper, so the sweep and the figure-3b Replay column
    # cover exactly the same steps.
    r0_global, r1_global = replay_window(params)
    r0 = r0_global - bnd[PhaseBoundary.HIST_DEP_END]
    r1 = r1_global - bnd[PhaseBoundary.HIST_DEP_END]
    x_pre = _walk(r0, params.sigx_pred)
    x_still = float(x_pre[-1]) if r0 > 0 else x_last
    replay_len = r1 - r0
    # Out-and-back sweep that returns to ``x_still`` so the animal's real
    # position resumes continuously after the replay (no boundary jump to
    # flag). The forward ramp reflects off the track ends; the reversed ramp
    # retraces it back to the start.
    half = (replay_len + 1) // 2
    ramp = reflect_into_interval(
        x_still + params.replay_speed * np.arange(half),
        float(params.xs_min),
        float(params.xs_max),
    )
    x_sweep = np.concatenate([ramp, ramp[::-1]])[:replay_len]
    x_post = simulate_walk(n - r1, params.sigx_pred, x_still, params.xs_min, params.xs_max, rng)
    x_rec2 = np.concatenate([x_pre, np.full(replay_len, x_still), x_post])
    sp_rec2 = np.vstack(
        [
            _spikes_position_tuned(x_pre),
            simulate_spikes_position_tuned(
                x_sweep, pf_centers, params.pf_width, params.replay_rate_scale, rng
            ),
            _spikes_position_tuned(x_post),
        ]
    )
    _add_phase(x_rec2, sp_rec2)

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

    # 7. Clean recovery 3. During the final part of this otherwise matched
    #    phase, the animal approaches the fixed sparse-population location so
    #    the sparse-population control begins without a position jump.
    n = bnd[PhaseBoundary.RECOVERY3_END] - bnd[PhaseBoundary.DRIFT_END]
    approach_steps = min(params.sparse_approach_steps, n)
    walk_steps = n - approach_steps
    if walk_steps > 0:
        x_walk = _walk(walk_steps, params.sigx_pred)
        approach_start = float(x_walk[-1])
    else:
        x_walk = np.empty(0, dtype=float)
        approach_start = x_last
    if approach_steps > 0:
        # Drop the first point so the approach continues from, rather than
        # duplicates, the preceding sample.
        x_approach = np.linspace(
            approach_start,
            params.sparse_position,
            approach_steps + 1,
        )[1:]
    else:
        x_approach = np.empty(0, dtype=float)
    x = np.concatenate([x_walk, x_approach])
    _add_phase(x, _spikes_position_tuned(x))

    # 8. Sparse Population — the animal remains immobile at the location while
    #    the ordinary ensemble becomes quiet. The baseline transition is still
    #    used, so the prediction spreads naturally between the isolated spikes.
    #    Each sparse cell's narrow likelihood falls inside that prediction: HPD
    #    overlap and predictive p remain good, while KL responds to their
    #    concentration difference.
    n = bnd[PhaseBoundary.SPARSE_POP_END] - bnd[PhaseBoundary.RECOVERY3_END]
    x = np.full(n, params.sparse_position, dtype=float)
    sparse_normal_spikes = simulate_spikes_position_tuned(
        x,
        pf_centers,
        params.pf_width,
        params.rate_scale * params.sparse_ensemble_rate_scale,
        rng,
    )
    _add_phase(x, sparse_normal_spikes)

    x_true = np.concatenate([p_x for p_x, _ in phases], axis=0)
    spikes = np.vstack([p_s for _, p_s in phases])  # (n_time, n_normal_cells)

    # Sparse population: a small set of pre-existing, sharply tuned cells
    # clustered around a fixed location. Each is an independent Poisson process
    # with a small baseline gain that rises to full rate during the sparse
    # window. Per-cell rates are sized so the population's *aggregate* rate
    # stays sparse (~5 Hz); a higher aggregate would shorten the gaps between
    # spikes and let the prediction re-concentrate. Use a phase-specific RNG
    # stream so upstream changes do not silently alter the illustrative spike
    # train while retaining seed-to-seed variability in the pooled summary.
    w0 = bnd[PhaseBoundary.RECOVERY3_END]
    w1 = bnd[PhaseBoundary.SPARSE_POP_END]
    if params.n_sparse_cells == 1:
        center_offsets = np.zeros(1, dtype=float)
    else:
        center_offsets = np.linspace(
            -params.sparse_field_spread, params.sparse_field_spread, params.n_sparse_cells
        )
    sparse_centers = params.sparse_position + center_offsets
    sparse_cell_scale = (
        params.sparse_cell_peak_rate * np.sqrt(2.0 * np.pi) * params.sparse_cell_width
    )
    sparse_rng = np.random.default_rng(np.random.SeedSequence([base_seed, 12]))
    sparse_cell_spikes = np.zeros((x_true.shape[0], sparse_centers.size), dtype=spikes.dtype)
    sparse_cell_spikes[:w0] = simulate_spikes_position_tuned(
        x_true[:w0],
        sparse_centers,
        params.sparse_cell_width,
        sparse_cell_scale * params.sparse_cell_baseline_gain,
        sparse_rng,
    )
    sparse_cell_spikes[w0:w1] = simulate_spikes_position_tuned(
        x_true[w0:w1],
        sparse_centers,
        params.sparse_cell_width,
        sparse_cell_scale,
        sparse_rng,
    )
    # (n_time, n_normal_cells + n_sparse_cells)
    spikes = np.hstack([spikes, sparse_cell_spikes])

    # Per-cell decoder rate tables. The decoder knows the sparse population's
    # small baseline gain and the low-activity regime; the final phase
    # therefore tests metric behavior under a consistent model rather than
    # creating an impossible observation.
    normal_rates = placefield_rates(xs, pf_centers, params.pf_width, params.rate_scale)
    sparse_cell_rates = placefield_rates(
        xs,
        sparse_centers,
        params.sparse_cell_width,
        sparse_cell_scale,
    )
    baseline_sparse_rates = params.sparse_cell_baseline_gain * sparse_cell_rates
    base_rates = np.hstack([normal_rates, baseline_sparse_rates])
    sparse_rates = np.hstack([params.sparse_ensemble_rate_scale * normal_rates, sparse_cell_rates])

    # Three decoder-rate windows share one schedule:
    # - Remap: the posterior update uses randomly scrambled place-field
    #   centers (``decoder_rates``); its diagnostics use that same likelihood,
    #   so the misfit surfaces from the scramble's spatial incoherence, not
    #   from any reference to the true fields.
    # - Replay: the ensemble fires at the elevated ``replay_rate_scale`` to
    #   densely sample the swept trajectory; the decoder is given that same
    #   elevated rate so the replay is a correctly-specified observation model
    #   (the decoded state simply tracks the replayed trajectory rather than
    #   the animal's position — not a fit failure).
    # - Sparse population: the decoder uses the correctly scaled quiet
    #   ensemble and active sparse-population rates.
    remapped_rates = np.hstack(
        [
            placefield_rates(
                xs,
                get_remapped_pf_centers(pf_centers, params.remap_from_to, active=True),
                params.pf_width,
                params.rate_scale,
            ),
            baseline_sparse_rates,
        ]
    )
    replay_rates = np.hstack(
        [
            placefield_rates(xs, pf_centers, params.pf_width, params.replay_rate_scale),
            baseline_sparse_rates,
        ]
    )
    replay_r0, replay_r1 = replay_window(params)
    misfit_schedule = MisfitSchedule(
        (
            MisfitWindow(
                bnd[PhaseBoundary.REMAP_START],
                bnd[PhaseBoundary.REMAP_END],
                decoder_rates=remapped_rates,
            ),
            MisfitWindow(
                replay_r0,
                replay_r1,
                decoder_rates=replay_rates,
            ),
            MisfitWindow(
                w0,
                w1,
                decoder_rates=sparse_rates,
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
        base_rates=base_rates,
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
        sparse_cell_centers=tuple(float(c) for c in sparse_centers),
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
        Pooled thresholds and median per-phase flag fractions.

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
    # the dict is a faithful baseline sample if that ever changes. Pool the
    # per-*event* baseline values (one per spike event), matching the
    # event-based phase fractions from ``extract_phase_flag_values``.
    baseline_keys = ("hpd_overlap", "kl_divergence", "spike_prob")
    baseline_values: dict[str, list[NDArray[np.floating]]] = {key: [] for key in baseline_keys}
    per_realization_values: list[list[list[NDArray[np.floating]]]] = []

    for offset in range(n_realizations):
        sim = run_figure03_simulation(params, seed=base + offset)
        metrics = sim.metrics
        base_mask = np.asarray(metrics.event_time_ind) < baseline_end
        for key in baseline_keys:
            ev = np.asarray(getattr(metrics, "event_" + key), dtype=float)[base_mask]
            baseline_values[key].append(ev[np.isfinite(ev)])
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
