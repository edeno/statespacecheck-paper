"""Reusable figure-3 simulation driver.

The figure-3 demo simulates a hippocampal-style decoder under a
sequence of misfit conditions (remap, history-dependent firing, drift) and
two specificity controls (a replay event embedded in clean-recovery 2 and a
final sparse-population epoch). The simulation pipeline drives both
``statespacecheck_paper.figure03_generation`` and
``statespacecheck_paper.interactive.cache.build_simulated_cache``;
both call ``run_figure03_simulation`` so the figure and the
interactive viewer's simulated cache stay byte-identical.

The figure-generation recipe extends this with diagnostic threshold
computation + plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from statespacecheck_paper.decoding import (
    DecoderOverrideSchedule,
    DecoderOverrideWindow,
    decode_with_diagnostics,
)
from statespacecheck_paper.diagnostics import DecodingDiagnostics
from statespacecheck_paper.figure03_protocol import (
    PHASE_LABELS,
    Figure3Config,
    PhaseBoundary,
    compute_replay_step_window,
)
from statespacecheck_paper.simulation import (
    gaussian_transition_matrix,
    peak_rate_to_place_field_scale,
    place_field_rates,
    reflect_into_interval,
    simulate_spikes_history_dependent,
    simulate_spikes_position_tuned,
    simulate_walk,
)


def remap_place_field_centers(
    place_field_centers: NDArray[np.floating],
    place_field_remapping: tuple[tuple[int, int], ...] | tuple[int, int],
    active: bool,
) -> NDArray[np.floating]:
    """Get place field centers with optional remapping.

    This function creates remapped place field centers for computing likelihoods
    during model misfit periods. When active, the source cell's place field center
    is replaced with the target cell's center, so the likelihood is computed using
    the wrong place field for that cell's spike_counts.

    Each ``(src, dst)`` pair makes cell ``src`` use cell ``dst``'s place-field
    center; original centers are snapshotted before any writes so a pair of
    swaps (``(a, b)``, ``(b, a)``) works correctly.

    Parameters
    ----------
    place_field_centers : np.ndarray, shape (n_cells,)
        Original place field centers for each cell.
    place_field_remapping : tuple of tuples or tuple of ints
        Remapping specification. Can be:
        - Single remapping: (src, dst) - cell src uses cell dst's place field center
        - Multiple remappings: ((src1, dst1), (src2, dst2), ...) - apply all remappings
    active : bool
        If False, returns place_field_centers unchanged. If True, applies remapping.

    Returns
    -------
    place_field_centers : np.ndarray, shape (n_cells,)
        Place field centers, potentially modified if active=True.
        Returns original array if active=False, copy if active=True.

    Examples
    --------
    >>> import numpy as np
    >>> place_field_centers = np.array([0.0, 10.0, 20.0, 30.0])
    >>> # Single remapping: cell 2 uses cell 0's place field
    >>> result = remap_place_field_centers(place_field_centers, (2, 0), active=True)
    >>> result
    array([ 0., 10.,  0., 30.])

    >>> # Inactive (no remapping)
    >>> result = remap_place_field_centers(place_field_centers, (2, 0), active=False)
    >>> np.array_equal(result, place_field_centers)
    True
    """
    if not active:
        return place_field_centers
    place_field_centers = place_field_centers.copy()

    # Normalize to iterable of pairs (handles both single and multiple remappings)
    if len(place_field_remapping) == 2 and isinstance(place_field_remapping[0], int):
        # Single remapping: (src, dst)
        src, dst = place_field_remapping
        place_field_centers[src] = place_field_centers[dst]
    else:
        # Multiple remappings: ((src1, dst1), (src2, dst2), ...)
        # Note: We need the ORIGINAL centers for all targets, so copy first
        original_centers = place_field_centers.copy()
        for src, dst in place_field_remapping:
            place_field_centers[src] = original_centers[dst]

    return place_field_centers


def _single_out_and_back_sweep(
    start: float,
    n_steps: int,
    lower: float,
    upper: float,
    max_speed: float,
) -> NDArray[np.floating]:
    """Construct one speed-capped sweep toward the farther endpoint and back."""
    if n_steps < 3:
        raise ValueError(f"An out-and-back sweep requires at least 3 steps; got {n_steps}.")
    if not lower <= start <= upper:
        raise ValueError(f"Sweep start {start} lies outside [{lower}, {upper}].")
    if not np.isfinite(max_speed) or max_speed <= 0.0:
        raise ValueError(f"Sweep max_speed must be positive; got {max_speed}.")

    farther_endpoint = upper if upper - start >= start - lower else lower
    direction = 1.0 if farther_endpoint > start else -1.0
    distance_to_endpoint = abs(farther_endpoint - start)
    # The shorter leg in an even-length sweep has one fewer transition.
    # Limit the excursion by that leg so neither direction exceeds max_speed.
    transitions_per_leg = (n_steps - 1) // 2
    excursion = min(distance_to_endpoint, max_speed * transitions_per_leg)
    turning_point = start + direction * excursion
    # Both legs include the turning point; drop its duplicate when joining.
    # Splitting this way preserves exactly ``n_steps`` samples and includes
    # the start, turning point, and final return for odd or even lengths.
    n_out = n_steps // 2 + 1
    n_back = n_steps - n_out + 1
    outbound = np.linspace(start, turning_point, n_out)
    inbound = np.linspace(turning_point, start, n_back)
    return np.concatenate([outbound, inbound[1:]])


@dataclass(frozen=True)
class Figure3SimulationResult:
    """Result of :func:`run_figure03_simulation`.

    Promoted from ``TypedDict`` to frozen dataclass so the load-bearing
    length invariants — one ``phase_labels`` entry per phase, boundaries
    delimit those phases, ``spike_counts`` and ``true_position`` share the timeline,
    and the final boundary equals the timeline length — are checked at
    construction. Without this, adding or removing a phase silently
    changes downstream lengths and the figure-3 pipeline would run with
    miscounted indices.
    """

    config: Figure3Config
    position_bins: NDArray[np.floating]
    true_position: NDArray[np.floating]
    spike_counts: NDArray[np.int_]
    diagnostics: DecodingDiagnostics
    # Sequence fields are declared as tuple so ``frozen=True``'s
    # immutability extends to the contents — list would leave
    # ``sim.phase_labels.append(...)`` and ``sim.phase_boundaries[-1] = 9999``
    # as silent invariant-breakers. Callers passing a list at construction
    # are coerced in __post_init__.
    phase_labels: tuple[str, ...]
    phase_boundaries: tuple[int, ...]
    # Fixed sparse-population field centers; let the raster sort all cells by
    # location without deriving a field center from the realized trajectory.
    sparse_place_field_centers: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        """Enforce length and timeline-consistency invariants.

        Also coerces the two sequence fields to tuple (in case the
        caller supplied a list) and validates each diagnostics array shares
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
        n_time = self.true_position.shape[0]
        if self.spike_counts.shape[0] != n_time:
            raise ValueError(
                f"spike_counts timeline ({self.spike_counts.shape[0]}) must equal "
                f"true_position timeline ({n_time})."
            )
        if self.phase_boundaries[-1] != n_time:
            raise ValueError(
                f"final phase boundary ({self.phase_boundaries[-1]}) must "
                f"equal true_position timeline ({n_time})."
            )
        # ``DecodingDiagnostics.__post_init__`` enforces shape agreement across
        # its own fields; cross-check that ``DecodingDiagnostics``'s leading dim
        # matches the ``true_position`` timeline supplied here.
        if self.diagnostics.posterior.shape[0] != n_time:
            raise ValueError(
                f"diagnostics.posterior leading dim {self.diagnostics.posterior.shape[0]} "
                f"does not match true_position timeline ({n_time})."
            )


@dataclass(frozen=True)
class Figure3RateTables:
    """Per-cell decoder rate tables for the figure-3 decoding windows.

    Each table stacks the eleven ordinary place cells with the sparse
    population (``n_normal_cells + sparse_cell_count`` columns), so it plugs
    directly into ``decode_with_diagnostics`` / ``DecoderOverrideWindow`` without
    further reshaping.

    Attributes
    ----------
    baseline_firing_rates : np.ndarray, shape (n_bins, n_cells)
        Default decoder rates: ordinary place fields plus the sparse
        population at its small baseline gain.
    remapped_firing_rates : np.ndarray, shape (n_bins, n_cells)
        Remap-window rates: the scrambled ordinary place fields plus the
        baseline sparse population.
    replay_firing_rates : np.ndarray, shape (n_bins, n_cells)
        Replay-window rates: ordinary place fields at the elevated
        ``replay_place_field_rate_scale`` plus the baseline sparse population.
    sparse_population_firing_rates : np.ndarray, shape (n_bins, n_cells)
        Sparse-window rates: the quiet ordinary ensemble
        (``sparse_control_ordinary_rate_scale``) plus the fully active sparse
        population.
    baseline_sparse_firing_rates : np.ndarray, shape (n_bins, sparse_cell_count)
        Sparse-population columns at the baseline gain; the shared block
        appended to ``baseline_firing_rates``/``remapped_firing_rates``/``replay_firing_rates``.
    """

    baseline_firing_rates: NDArray[np.floating]
    remapped_firing_rates: NDArray[np.floating]
    replay_firing_rates: NDArray[np.floating]
    sparse_population_firing_rates: NDArray[np.floating]
    baseline_sparse_firing_rates: NDArray[np.floating]


def _record_phase(
    phases: list[tuple[NDArray[np.floating], NDArray[np.int_]]],
    phase_labels: list[str],
    label: str,
    x: NDArray[np.floating],
    sp: NDArray[np.int_],
) -> float:
    """Append one phase with an explicit label; return its end position.

    The returned value is the next phase's starting position. ``x_last``
    trajectory continuity between phases is intentional and threaded
    explicitly by the caller, never inferred from phase order or a generic
    accumulator.
    """
    phase_labels.append(label)
    phases.append((x, sp))
    return float(x[-1])


def simulate_history_dependent_phase(
    n: int,
    x_last: float,
    config: Figure3Config,
    place_field_centers: NDArray[np.floating],
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating], NDArray[np.int_]]:
    """History-dependent firing misfit: a normal walk with bursting spike_counts.

    Cells fire via ``simulate_spikes_history_dependent`` (hard refractory
    plus a burst window); the decoder still treats every spike as an
    independent Poisson draw, so the misfit lives in the temporal
    correlations and is largely invisible to the per-spike spatial
    diagnostics.

    Draw order: the trajectory walk, then the history-dependent spike_counts.
    """
    x = simulate_walk(
        n, config.prediction_step_std, x_last, config.position_min, config.position_max, rng
    )
    sp = simulate_spikes_history_dependent(
        x, place_field_centers, config.place_field_std, config.place_field_rate_scale, rng
    )
    return x, sp


def simulate_replay_phase(
    n: int,
    r0: int,
    r1: int,
    x_last: float,
    config: Figure3Config,
    place_field_centers: NDArray[np.floating],
    rng: np.random.Generator,
) -> tuple[NDArray[np.floating], NDArray[np.int_]]:
    """Clean-recovery-2 window containing the replay control.

    The animal is immobile (physical position held fixed at ``x_still``)
    while a coherent represented trajectory sweeps the track once out and
    back over the sub-window ``[r0, r1)``. Spikes during the sweep fire at
    the elevated ``replay_place_field_rate_scale``; before and after they are ordinary
    position-tuned spike_counts.

    RNG-order contract: the shared ``rng`` draws BOTH walks (``x_pre``
    then ``x_post``) before ANY spike_counts, matching the original ``vstack``
    order. Reordering to walk -> spike -> walk -> spike would move the
    ``x_post`` walk ahead of the ``x_pre`` spike_counts and shift every
    downstream draw, changing Figure 3 and the interactive simulated
    cache. Do not reorder.
    """
    x_pre = simulate_walk(
        r0, config.prediction_step_std, x_last, config.position_min, config.position_max, rng
    )
    x_still = float(x_pre[-1]) if r0 > 0 else x_last
    replay_len = r1 - r0
    # One outbound leg toward the farther track end and one return to
    # ``x_still`` (draws nothing from ``rng``).
    x_sweep = _single_out_and_back_sweep(
        x_still,
        replay_len,
        float(config.position_min),
        float(config.position_max),
        config.replay_speed_per_step,
    )
    # Second walk drawn before any spike_counts (shared-rng draw-order contract).
    x_post = simulate_walk(
        n - r1, config.prediction_step_std, x_still, config.position_min, config.position_max, rng
    )
    x_rec2 = np.concatenate([x_pre, np.full(replay_len, x_still), x_post])
    sp_rec2 = np.vstack(
        [
            simulate_spikes_position_tuned(
                x_pre,
                place_field_centers,
                config.place_field_std,
                config.place_field_rate_scale,
                rng,
            ),
            simulate_spikes_position_tuned(
                x_sweep,
                place_field_centers,
                config.place_field_std,
                config.replay_place_field_rate_scale,
                rng,
            ),
            simulate_spikes_position_tuned(
                x_post,
                place_field_centers,
                config.place_field_std,
                config.place_field_rate_scale,
                rng,
            ),
        ]
    )
    return x_rec2, sp_rec2


def simulate_drift_phase(
    n: int,
    x_last: float,
    config: Figure3Config,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Drift misfit: an AR(1) persistent-velocity trajectory.

    ``x[t] = x[t-1] + v[t]`` with
    ``v[t] = drift_momentum * v[t-1] + N(0, prediction_step_std)``; the decoder
    assumes a memoryless walk. Returns the trajectory only; the caller
    draws the position-tuned spike_counts from it, matching the original draw
    order (drift steps, then spike_counts).
    """
    momentum = config.drift_momentum
    x_mom = np.zeros(n)
    x_mom[0] = x_last
    velocity = 0.0
    for t in range(1, n):
        velocity = momentum * velocity + rng.normal(0, config.prediction_step_std)
        x_mom[t] = x_mom[t - 1] + velocity
    return reflect_into_interval(x_mom, float(config.position_min), float(config.position_max))


def simulate_sparse_approach_phase(
    n: int,
    x_last: float,
    config: Figure3Config,
    rng: np.random.Generator,
) -> NDArray[np.floating]:
    """Clean-recovery-3 that ends by approaching ``config.sparse_position``.

    A normal walk for ``n - sparse_approach_duration_steps`` steps, then a smooth
    ramp to ``config.sparse_position`` so the sparse-population control
    begins without a position jump. Returns the trajectory only; the
    caller draws the position-tuned spike_counts (original draw order: walk,
    then spike_counts).
    """
    approach_steps = min(config.sparse_approach_duration_steps, n)
    walk_steps = n - approach_steps
    if walk_steps > 0:
        x_walk = simulate_walk(
            walk_steps,
            config.prediction_step_std,
            x_last,
            config.position_min,
            config.position_max,
            rng,
        )
        approach_start = float(x_walk[-1])
    else:
        x_walk = np.empty(0, dtype=float)
        approach_start = x_last
    if approach_steps > 0:
        # Drop the first point so the approach continues from, rather than
        # duplicates, the preceding sample.
        x_approach = np.linspace(
            approach_start,
            config.sparse_position,
            approach_steps + 1,
        )[1:]
    else:
        x_approach = np.empty(0, dtype=float)
    return np.concatenate([x_walk, x_approach])


def build_sparse_population(
    true_position: NDArray[np.floating],
    config: Figure3Config,
    random_seed: int,
    w0: int,
    w1: int,
) -> tuple[NDArray[np.int_], NDArray[np.floating]]:
    """Build the sparse-population spike columns and their field centers.

    A small set of pre-existing, sharply tuned cells clustered around
    ``config.sparse_position``, each an independent Poisson process. A
    small baseline gain applies before ``w0``; the full rate applies
    within ``[w0, w1)``; after ``w1`` the columns stay zero. Per-cell
    rates are sized so the population's *aggregate* rate stays sparse.

    Uses an INDEPENDENT ``SeedSequence([random_seed, 12])`` stream so this
    illustrative spike train is stable under upstream changes and its
    extraction does not perturb the main ``rng`` draw order.

    Returns
    -------
    sparse_cell_spikes : np.ndarray, shape (n_time, sparse_cell_count)
        Spike counts for the sparse population.
    sparse_centers : np.ndarray, shape (sparse_cell_count,)
        Fixed field centers of the sparse population.
    """
    if config.sparse_cell_count == 1:
        center_offsets = np.zeros(1, dtype=float)
    else:
        center_offsets = np.linspace(
            -config.sparse_place_field_spread,
            config.sparse_place_field_spread,
            config.sparse_cell_count,
        )
    sparse_centers = config.sparse_position + center_offsets
    sparse_cell_scale = peak_rate_to_place_field_scale(
        config.sparse_cell_peak_rate_per_step, config.sparse_place_field_std
    )
    sparse_rng = np.random.default_rng(np.random.SeedSequence([random_seed, 12]))
    # Draw the baseline window first (matching the original stream order),
    # then the elevated window; leave post-``w1`` samples at zero.
    baseline_block = simulate_spikes_position_tuned(
        true_position[:w0],
        sparse_centers,
        config.sparse_place_field_std,
        sparse_cell_scale * config.sparse_cell_baseline_rate_fraction,
        sparse_rng,
    )
    sparse_cell_spikes = np.zeros(
        (true_position.shape[0], sparse_centers.size), dtype=baseline_block.dtype
    )
    sparse_cell_spikes[:w0] = baseline_block
    sparse_cell_spikes[w0:w1] = simulate_spikes_position_tuned(
        true_position[w0:w1],
        sparse_centers,
        config.sparse_place_field_std,
        sparse_cell_scale,
        sparse_rng,
    )
    return sparse_cell_spikes, sparse_centers


def build_figure03_rate_tables(
    position_bins: NDArray[np.floating],
    place_field_centers: NDArray[np.floating],
    sparse_centers: NDArray[np.floating],
    config: Figure3Config,
) -> Figure3RateTables:
    """Assemble the four figure-3 decoder rate tables.

    The decoder knows the sparse population's small baseline gain and the
    low-activity regime, so each misfit window tests metric behavior under
    a consistent (correctly specified) model:

    - Remap: the posterior update uses randomly scrambled place-field
      centers; its diagnostics use that same likelihood, so the misfit
      surfaces from the scramble's spatial incoherence.
    - Replay: the ensemble fires at the elevated ``replay_place_field_rate_scale`` and
      the decoder is given that same elevated rate, so the replay is a
      correctly-specified observation model.
    - Sparse population: the decoder uses the correctly scaled quiet
      ensemble and active sparse-population rates.
    """
    normal_rates = place_field_rates(
        position_bins, place_field_centers, config.place_field_std, config.place_field_rate_scale
    )
    sparse_cell_scale = peak_rate_to_place_field_scale(
        config.sparse_cell_peak_rate_per_step, config.sparse_place_field_std
    )
    sparse_cell_rates = place_field_rates(
        position_bins,
        sparse_centers,
        config.sparse_place_field_std,
        sparse_cell_scale,
    )
    baseline_sparse_firing_rates = config.sparse_cell_baseline_rate_fraction * sparse_cell_rates
    baseline_firing_rates = np.hstack([normal_rates, baseline_sparse_firing_rates])
    sparse_population_firing_rates = np.hstack(
        [config.sparse_control_ordinary_rate_scale * normal_rates, sparse_cell_rates]
    )
    remapped_firing_rates = np.hstack(
        [
            place_field_rates(
                position_bins,
                remap_place_field_centers(
                    place_field_centers, config.place_field_remapping, active=True
                ),
                config.place_field_std,
                config.place_field_rate_scale,
            ),
            baseline_sparse_firing_rates,
        ]
    )
    replay_firing_rates = np.hstack(
        [
            place_field_rates(
                position_bins,
                place_field_centers,
                config.place_field_std,
                config.replay_place_field_rate_scale,
            ),
            baseline_sparse_firing_rates,
        ]
    )
    return Figure3RateTables(
        baseline_firing_rates=baseline_firing_rates,
        remapped_firing_rates=remapped_firing_rates,
        replay_firing_rates=replay_firing_rates,
        sparse_population_firing_rates=sparse_population_firing_rates,
        baseline_sparse_firing_rates=baseline_sparse_firing_rates,
    )


def run_figure03_simulation(
    config: Figure3Config | None = None,
    *,
    seed: int | None = None,
) -> Figure3SimulationResult:
    """Run the figure-3 phased simulation and decode it.

    Phases (in order, with their misfit class):

    1. Clean Baseline
    2. **Remap Misfit** (observation: all place-field identities undergo
       one fixed, spatially incoherent permutation)
    3. Clean Recovery
    4. **History-Dependent Firing Misfit** (observation: spike_counts
       generated with hard refractory + bursting; decoder still
       assumes Poisson. Per-spike spatial likelihood is unchanged,
       so the per-spike diagnostics largely miss this — deliberate
       demonstration of the spatial-only nature of the diagnostics.)
    5. Clean Recovery (contains the **Replay control**: an out-and-back
       trajectory sweep while the animal is immobile. The decoder tracks the
       sweep, so the decoded position departs from the true fixed position
       yet stays consistent with each spike's likelihood — a benign
       decoded-vs-true divergence that none of the diagnostics should flag.)
    6. **Drift Misfit** (transition: trajectory has persistent velocity
       at AR(1) coefficient ``config.drift_momentum``; decoder assumes
       memoryless walk)
    7. Clean Recovery
    8. **Sparse Population** (control: the ordinary ensemble is quiet while a
       small population of pre-existing, sharply tuned cells clustered at one
       location fires sparsely — each an independent Poisson process, the
       decoder's rates matching exactly. The prediction spreads between the
       isolated spike_counts, and each spike's narrow likelihood remains contained
       within it. KL responds to the concentration difference while HPD overlap
       and the rank-based p-value remain consistent.)

    Parameters
    ----------
    config : Figure3Config, optional
        Simulation configuration. If ``None``, uses default
        ``Figure3Config()``.
    seed : int, optional
        Override ``config.random_seed`` so callers can vary stochastic
        draws without mutating the config dataclass.

    Returns
    -------
    Figure3SimulationResult
        Dataclass with attributes ``config``, ``position_bins``, ``true_position``,
        ``spike_counts``, ``diagnostics``, ``phase_labels``, ``phase_boundaries``,
        and ``sparse_place_field_centers`` (fixed centers for the appended
        sparse-population cells). Access via attribute (``sim.diagnostics``),
        not subscript.
    """
    if config is None:
        config = Figure3Config()
    random_seed = config.random_seed if seed is None else seed
    rng = np.random.default_rng(random_seed)

    if config.place_field_centers is None:
        raise ValueError("config.place_field_centers must be initialized")
    place_field_centers = config.place_field_centers

    position_bins = np.arange(
        config.position_min,
        config.position_max + config.position_bin_size,
        config.position_bin_size,
        dtype=float,
    )
    transition_matrix = gaussian_transition_matrix(position_bins, config.prediction_step_std)

    phases: list[tuple[NDArray[np.floating], NDArray[np.int_]]] = []
    phase_labels: list[str] = []
    x_last: float = 0.0
    bnd = config.phase_boundaries

    def _walk(n: int, x0: float) -> NDArray[np.floating]:
        return simulate_walk(
            n, config.prediction_step_std, x0, config.position_min, config.position_max, rng
        )

    def _spikes(x: NDArray[np.floating]) -> NDArray[np.int_]:
        return simulate_spikes_position_tuned(
            x, place_field_centers, config.place_field_std, config.place_field_rate_scale, rng
        )

    # The eight phases are a plainly-written ordered sequence; ``x_last``
    # carries the trajectory's end position forward as the next phase's
    # start (continuity is intended and threaded explicitly).

    # 1. Clean baseline
    n = bnd[PhaseBoundary.REMAP_START]
    x = _walk(n, x_last)
    x_last = _record_phase(phases, phase_labels, "Clean Baseline", x, _spikes(x))

    # 2. Remap misfit — spike *generation* is normal position-tuned; only the
    #    decoder uses randomly scrambled PF centers during this window (via
    #    ``DecoderOverrideWindow`` below).
    n = bnd[PhaseBoundary.REMAP_END] - bnd[PhaseBoundary.REMAP_START]
    x = _walk(n, x_last)
    x_last = _record_phase(phases, phase_labels, "Remap Misfit", x, _spikes(x))

    # 3. Clean recovery 1
    n = bnd[PhaseBoundary.RECOVERY1_END] - bnd[PhaseBoundary.REMAP_END]
    x = _walk(n, x_last)
    x_last = _record_phase(phases, phase_labels, "Clean Recovery", x, _spikes(x))

    # 4. History-dependent firing misfit
    n = bnd[PhaseBoundary.HIST_DEP_END] - bnd[PhaseBoundary.RECOVERY1_END]
    x, sp = simulate_history_dependent_phase(n, x_last, config, place_field_centers, rng)
    x_last = _record_phase(phases, phase_labels, "History-Dependent Firing", x, sp)

    # 5. Clean recovery 2 — with the embedded replay control. Local
    #    (within-phase) replay bounds derived from the shared global
    #    ``compute_replay_step_window`` helper, so the sweep and the figure-3b Replay column
    #    cover exactly the same steps.
    n = bnd[PhaseBoundary.RECOVERY2_END] - bnd[PhaseBoundary.HIST_DEP_END]
    r0_global, r1_global = compute_replay_step_window(config)
    r0 = r0_global - bnd[PhaseBoundary.HIST_DEP_END]
    r1 = r1_global - bnd[PhaseBoundary.HIST_DEP_END]
    x, sp = simulate_replay_phase(n, r0, r1, x_last, config, place_field_centers, rng)
    x_last = _record_phase(phases, phase_labels, "Clean Recovery", x, sp)

    # 6. Drift misfit — persistent-velocity walk; decoder assumes memoryless.
    n = bnd[PhaseBoundary.DRIFT_END] - bnd[PhaseBoundary.RECOVERY2_END]
    x = simulate_drift_phase(n, x_last, config, rng)
    x_last = _record_phase(phases, phase_labels, "Drift Misfit", x, _spikes(x))

    # 7. Clean recovery 3 — ends by approaching the sparse-population location.
    n = bnd[PhaseBoundary.RECOVERY3_END] - bnd[PhaseBoundary.DRIFT_END]
    x = simulate_sparse_approach_phase(n, x_last, config, rng)
    x_last = _record_phase(phases, phase_labels, "Clean Recovery", x, _spikes(x))

    # 8. Sparse population — the animal remains immobile at the location while
    #    the ordinary ensemble becomes quiet. The baseline transition is still
    #    used, so the prediction spreads naturally between the isolated sparse
    #    spike_counts (built below with an independent RNG stream).
    n = bnd[PhaseBoundary.SPARSE_POP_END] - bnd[PhaseBoundary.RECOVERY3_END]
    x = np.full(n, config.sparse_position, dtype=float)
    sparse_normal_spikes = simulate_spikes_position_tuned(
        x,
        place_field_centers,
        config.place_field_std,
        config.place_field_rate_scale * config.sparse_control_ordinary_rate_scale,
        rng,
    )
    _record_phase(phases, phase_labels, "Sparse Population", x, sparse_normal_spikes)

    true_position = np.concatenate([p_x for p_x, _ in phases], axis=0)
    spike_counts = np.vstack([p_s for _, p_s in phases])  # (n_time, n_normal_cells)

    w0 = bnd[PhaseBoundary.RECOVERY3_END]
    w1 = bnd[PhaseBoundary.SPARSE_POP_END]
    sparse_cell_spikes, sparse_centers = build_sparse_population(
        true_position, config, random_seed, w0, w1
    )
    # (n_time, n_normal_cells + sparse_cell_count)
    spike_counts = np.hstack([spike_counts, sparse_cell_spikes])

    rate_tables = build_figure03_rate_tables(
        position_bins, place_field_centers, sparse_centers, config
    )

    replay_r0, replay_r1 = compute_replay_step_window(config)
    override_schedule = DecoderOverrideSchedule(
        (
            DecoderOverrideWindow(
                bnd[PhaseBoundary.REMAP_START],
                bnd[PhaseBoundary.REMAP_END],
                firing_rate_table=rate_tables.remapped_firing_rates,
            ),
            DecoderOverrideWindow(
                replay_r0,
                replay_r1,
                firing_rate_table=rate_tables.replay_firing_rates,
            ),
            DecoderOverrideWindow(
                w0,
                w1,
                firing_rate_table=rate_tables.sparse_population_firing_rates,
            ),
        )
    )

    diagnostics = decode_with_diagnostics(
        spike_counts=spike_counts,
        position_bins=position_bins,
        transition_matrix=transition_matrix,
        place_field_centers=place_field_centers,
        place_field_std=config.place_field_std,
        place_field_rate_scale=config.place_field_rate_scale,
        override_schedule=override_schedule,
        baseline_firing_rates=rate_tables.baseline_firing_rates,
    )

    boundaries = np.cumsum([len(p_x) for p_x, _ in phases]).tolist()

    return Figure3SimulationResult(
        config=config,
        position_bins=position_bins,
        true_position=true_position,
        spike_counts=spike_counts,
        diagnostics=diagnostics,
        phase_labels=tuple(phase_labels),
        phase_boundaries=tuple(boundaries),
        sparse_place_field_centers=tuple(float(c) for c in sparse_centers),
    )
