"""Figure-3 experimental protocol: phase ladder and immutable configuration.

The figure-3 simulation walks a hippocampal-style decoder through four misfit
conditions (an incoherent place-field remap, history-dependent firing, drift,
and a coherent reflected map) and two controls (a replay event embedded in
clean-recovery 2, and a final matched low-information regime), separated by
clean-recovery windows. Outside the perturbed windows the latent trajectory is
drawn from the decoder's own discrete transition matrix and initial law, so
those windows are an exactly matched reference for the decoder. This module
holds the immutable experimental configuration (:class:`Figure3Config`), the
phase-transition index enum (:class:`PhaseBoundary`), the canonical ordered
phase labels (:data:`PHASE_LABELS`), and the replay-window step-bound helper
(:func:`compute_replay_step_window`). It imports no sibling paper module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

TrajectoryModel = Literal["discrete_matched", "continuous_reflected"]


class PhaseBoundary(IntEnum):
    """Indices into :attr:`Figure3Config.phase_boundaries`.

    Each member is the position of one figure-3 phase transition in
    the 10-tuple. Use as ``config.phase_boundaries[PhaseBoundary.REMAP_END]``
    rather than indexing by literal integer, so a phase-ladder
    reshuffle stays compile-time-checkable.
    """

    REMAP_START = 0  # end of the opening matched-null baseline
    REMAP_END = 1  # end of remap misfit
    RECOVERY1_END = 2  # end of clean recovery 1
    HIST_DEP_END = 3  # end of history-dependent firing misfit
    RECOVERY2_END = 4  # end of clean recovery 2 (contains the replay control)
    DRIFT_END = 5  # end of drift misfit
    RECOVERY3_END = 6  # end of clean recovery 3
    REFLECT_END = 7  # end of the coherent reflected-map misfit
    RECOVERY4_END = 8  # end of clean recovery 4
    SPARSE_POP_END = 9  # end of the matched low-information (sparse) regime


# Default phase ladder in 1-ms steps. Used as the default of
# ``Figure3Config.phase_boundaries`` and re-exported here so tests and
# scripts that want to override a subset don't have to re-list the
# unchanged entries.
_DEFAULT_PHASE_BOUNDARIES: tuple[int, ...] = (
    6_000,
    10_000,
    14_000,
    18_000,
    22_000,
    26_000,
    30_000,
    34_000,
    38_000,
    42_000,
)

# History-dependence rate-matching gain. The refractory/burst modulation raises
# a cell's marginal rate above its Poisson place-field mean (the burst window
# outweighs the single suppressed step). Scaling the generator's place-field
# rate by this gain brings the history-phase marginal rate back to the Poisson
# baseline on average over matched-null calibration trajectories, so the
# history condition perturbs the temporal structure of spiking rather than
# its overall rate. The value was estimated by
# ``figure03_calibration.estimate_history_rate_matching_gain`` on independent
# calibration seeds and is pinned here (the calibration test re-derives it);
# the residual per-phase rate difference is reported in the figure summary.
DEFAULT_HISTORY_RATE_MATCHING_GAIN = 0.512


@dataclass(frozen=True)
class Figure3Config:
    """Parameters for the figure-3 decoding simulation.

    The simulation walks through four misfit conditions plus two controls —
    a replay event embedded in the second clean-recovery window and a final
    matched low-information regime — separated by clean-recovery windows in
    which the trajectory follows the decoder's own transition law. Time steps are
    1 ms by convention — the simulation math itself is dt-agnostic, but the
    default parameters
    (`place_field_rate_scale=5.0`, `history_refractory_steps`,
    `history_burst_window`) are tuned for that mapping
    and yield hippocampally-realistic spike rates and timescales.

    **Timeline Structure** (default; all indices in 1-ms steps):

    - 0–6k: Matched-null baseline (trajectory drawn from the decoder's own
      initial law and transition matrix)
    - 6k–10k: Remap misfit (4 s)
    - 10k–14k: Clean recovery
    - 14k–18k: History-dependent firing misfit (4 s)
    - 18k–22k: Clean recovery, with the replay control occupying the
      ``[replay_start_fraction, replay_end_fraction)`` sub-window
    - 22k–26k: Drift misfit (4 s)
    - 26k–30k: Clean recovery
    - 30k–34k: Reflected-map misfit (4 s; a coherent wrong map)
    - 34k–38k: Clean recovery
    - 38k–42k: Matched low-information regime (4 s; quiet ordinary
      ensemble, sparse narrow cells, trajectory still matched)

    Parameters
    ----------
    phase_boundaries : tuple of int, default ``_DEFAULT_PHASE_BOUNDARIES``
        Strictly increasing end-of-phase indices, one per member of
        :class:`PhaseBoundary`. Read via the enum
        (``config.phase_boundaries[PhaseBoundary.REMAP_END]``) rather
        than by literal integer. Override a subset by spelling out the
        whole tuple — partial overrides aren't supported because the
        invariant the dataclass enforces ("strictly increasing ladder")
        only makes sense over the full ladder.
    trajectory_model : {"discrete_matched", "continuous_reflected"}
        How the latent position evolves in the unperturbed phases.
        ``"discrete_matched"`` (default) samples the position on the decoder's
        grid from the decoder's own column-stochastic transition matrix
        (:func:`~statespacecheck_paper.simulation.gaussian_transition_matrix`,
        a truncated-and-renormalized Gaussian kernel) starting from the
        decoder's uniform initial law, so the well-specified phases are an
        *exactly* matched null. ``"continuous_reflected"`` retains the earlier
        continuous Gaussian walk with reflecting boundaries, whose reflected
        continuous increments only approximate the decoder's grid transition;
        it is kept as an explicitly approximate benchmark for sensitivity
        checks.
    hpd_coverage : float, default 0.95
        Coverage of the HPD regions compared by the HPD-overlap diagnostic.
    prediction_step_std : float, default 0.5
        Decoder's baseline dynamics standard deviation.
    drift_momentum : float, default 0.88
        AR(1) coefficient on the animal's velocity during the drift
        misfit phase. The true trajectory is
        ``x[t] = x[t-1] + v[t]`` with
        ``v[t] = drift_momentum * v[t-1] + N(0, prediction_step_std)``. The
        decoder assumes ``x[t] = x[t-1] + N(0, prediction_step_std)`` (no
        persistent velocity).
    history_refractory_steps : int, default 1
        Post-spike suppression length, in steps, during the history-dependence
        misfit: a cell's Poisson mean is zero for this many steps after a
        spike-containing step. At 1 ms/step this is the 1 ms suppressed step
        reported in the Methods (a binned count process, not a hard
        refractory point process; several spikes can share one step).
    history_burst_window : tuple of int, default (2, 10)
        Inclusive ``(start, end)`` step offsets after a spike over which the
        cell's rate is multiplied by ``history_burst_factor`` — the 2–10 ms
        burst window at 1 ms/step.
    history_burst_factor : float, default 3.0
        Rate multiplier applied inside ``history_burst_window``.
    history_rate_matching_gain : float, default ``DEFAULT_HISTORY_RATE_MATCHING_GAIN``
        Multiplies the generator's place-field rate during the history
        phase so that the modulated process's marginal rate approximately
        matches the unmodulated Poisson baseline. Set to ``1.0`` for the
        unmatched process. The decoder always uses the unmodulated baseline
        rates, so the misfit is the temporal dependence itself.
    position_min, position_max, position_bin_size : int
        Position grid bounds and step.
    place_field_std : float, default 10.0
        Gaussian place-field std (in position units).
    place_field_centers : NDArray[np.floating] | None
        Place-field center positions; defaults to ``np.arange(0, 101, 10)``.
    place_field_rate_scale : float, default 5.0
        Scale factor multiplying the normalized Gaussian place-field density.
        With the default field width and a 1-ms step, a value of 5.0 gives a
        peak rate of approximately 200 Hz.
    random_seed : int, default 1
        Random seed for reproducibility.
    place_field_remapping : tuple of (int, int) pairs, default see source
        Specification of which cells get remapped during the remap
        window. By default, all eleven cells participate in one fixed
        permutation that moves every field by at least three center spacings.
        The reflected-map misfit needs no specification: the decoder mirrors
        every center about the track midpoint
        (``position_min + position_max - mu_c``), a coherent wrong map.
    replay_start_fraction, replay_end_fraction : float, default 0.25, 0.75
        Fractional bounds of the replay sub-window within clean-recovery 2.
        The coherent sweep fires over ``[replay_start_fraction,
        replay_end_fraction)``.
    replay_speed_per_step : float, default 0.5
        Maximum per-step displacement of the replay trajectory as it sweeps
        toward the farther track end and returns.
    replay_place_field_rate_scale : float, default 20.0
        Elevated place-field rate scale applied during the replay sweep.
    sparse_control_ordinary_rate_scale : float, default 0.0
        Multiplicative rate applied to the eleven ordinary place cells during
        the low-information regime, in both generation and decoding. Zero
        represents a silent ordinary ensemble.
    sparse_place_field_centers : tuple of float, default (0, 10, ..., 100)
        Field centers of the narrow sparse-population cells, spread along the
        track (one per ordinary field center) so the matched trajectory keeps
        visiting them.
    sparse_place_field_std : float, default 2.0
        Standard deviation, in position units, of each narrow sparse-population
        field.
    sparse_cell_peak_rate_per_step : float, default 0.005
        Reference per-cell peak firing rate in spikes per 1-ms step (5 Hz),
        multiplied per cell by ``sparse_cell_rate_multipliers``. Sized so the
        population's aggregate rate stays low (a few spikes per second) and
        its spikes temporally isolated, letting the prediction spread between
        them.
    sparse_cell_rate_multipliers : tuple of float, default see source
        Heterogeneous per-cell gains (0.5-2) on the reference peak rate, one
        per sparse cell, so the active marks are not equally probable.
    sparse_cell_baseline_rate_fraction : float, default 0.01
        Fraction of the active per-cell rate used before the final regime.

    Examples
    --------
    >>> config = Figure3Config()
    >>> config.phase_boundaries[PhaseBoundary.REMAP_START]
    6000
    >>> config.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
    42000
    >>> config.place_field_centers
    array([  0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 100.])
    """

    # Phase ladder. One boundary per :class:`PhaseBoundary` member,
    # strictly increasing; validated in __post_init__.
    phase_boundaries: tuple[int, ...] = _DEFAULT_PHASE_BOUNDARIES

    # Latent trajectory law in the unperturbed phases and HPD coverage.
    trajectory_model: TrajectoryModel = "discrete_matched"
    hpd_coverage: float = 0.95

    # Decoder & dynamics parameters
    prediction_step_std: float = 0.5  # baseline dynamics std
    drift_momentum: float = 0.88  # AR(1) coefficient for drift-misfit trajectory

    # History-dependence misfit: post-spike suppression + burst window.
    # At the 1-ms step these are the 1 ms suppressed step, 2-10 ms burst
    # window, and threefold rate increase reported in the Methods. They live
    # on the config (rather than as ``simulate_spikes_history_dependent``
    # defaults) so the published summary records the values the figure
    # actually used. ``history_rate_matching_gain`` rescales the generator so
    # the phase's marginal rate matches the Poisson baseline on average.
    history_refractory_steps: int = 1
    history_burst_window: tuple[int, int] = (2, 10)
    history_burst_factor: float = 3.0
    history_rate_matching_gain: float = DEFAULT_HISTORY_RATE_MATCHING_GAIN

    # Position grid
    position_min: int = 0
    position_max: int = 100
    position_bin_size: int = 1

    # Place fields
    place_field_std: float = 10.0
    place_field_centers: NDArray[np.floating] | None = None  # set in __post_init__
    place_field_rate_scale: float = 5.0

    random_seed: int = 1
    # Global-remapping model: a fixed random permutation (derangement) of
    # the eleven place-field centers, so each cell ``src`` adopts cell
    # ``dst``'s center. Chosen using the first NumPy default_rng seed (737)
    # whose permutation displaces every field by >=3 positions, so no cell
    # keeps a field near its original location. The scramble is spatially
    # *incoherent*: because
    # the decoder's diagnostics judge each spike against this same remapped
    # likelihood (not the true fields), the misfit surfaces only through the
    # genuine conflict between the smooth-motion prior and the scattered
    # per-spike likelihoods — a coherent remap (e.g. a pure reflection)
    # would be self-consistent and correctly go undetected.
    place_field_remapping: tuple[tuple[int, int], ...] | tuple[int, int] = (
        (0, 7),
        (1, 10),
        (2, 9),
        (3, 0),
        (4, 8),
        (5, 2),
        (6, 3),
        (7, 1),
        (8, 4),
        (9, 6),
        (10, 5),
    )

    # Replay event embedded in the second clean-recovery window. The animal
    # is immobile while a coherent trajectory sweeps the track; the decoder
    # tracks the sweep, so the decoded position departs from the true
    # (fixed) position without any diagnostic flagging it -- replay is not a
    # misspecification. The sweep occupies the fractional sub-window
    # ``[replay_start_fraction, replay_end_fraction)`` of clean-recovery 2 and fires
    # at an elevated ``replay_place_field_rate_scale``. The trajectory makes one sweep
    # toward the farther track end, capped at ``replay_speed_per_step`` per step, and
    # returns to its starting position.
    replay_start_fraction: float = 0.25
    replay_end_fraction: float = 0.75
    replay_speed_per_step: float = 0.5
    replay_place_field_rate_scale: float = 20.0

    # Matched low-information regime. The trajectory keeps following the
    # decoder's transition matrix while the ordinary ensemble becomes quiet.
    # A small population of narrow, sharply tuned cells spread along the
    # track fires sparsely, each cell an independent Poisson process increasing
    # from a small baseline gain to its full (heterogeneous) rate. With little
    # intervening population information, the predictive spreads between the
    # isolated spikes; each spike supplies a narrow likelihood inside that
    # broad prediction. Both the transition and the observation model are the
    # decoder's own, so any flag here is a false positive against a correct
    # model.
    sparse_control_ordinary_rate_scale: float = 0.0
    sparse_place_field_centers: tuple[float, ...] = tuple(float(c) for c in range(0, 101, 10))
    sparse_place_field_std: float = 2.0
    sparse_cell_peak_rate_per_step: float = 0.005  # spikes/ms = 5 Hz reference peak
    sparse_cell_rate_multipliers: tuple[float, ...] = (
        0.5,
        1.0,
        2.0,
        0.75,
        1.5,
        0.5,
        1.0,
        2.0,
        0.75,
        1.5,
        1.0,
    )
    sparse_cell_baseline_rate_fraction: float = 0.01

    def __post_init__(self) -> None:
        """Validate the timeline and initialize ``place_field_centers`` if not provided.

        ``phase_boundaries`` must have exactly one entry per
        :class:`PhaseBoundary` member and be strictly increasing —
        ``run_figure03_simulation`` builds each phase as
        ``T_next - T_prev`` and a non-monotonic timeline would yield a
        negative phase length, which ``np.arange``/``np.zeros``
        silently turn into an empty phase, shifting every later misfit
        window. Catch that here at construction rather than as a
        misaligned figure downstream.
        """
        bnds = tuple(self.phase_boundaries)
        if len(bnds) != len(PhaseBoundary):
            raise ValueError(
                f"Figure3Config.phase_boundaries must have "
                f"{len(PhaseBoundary)} entries "
                f"(one per PhaseBoundary member); got {len(bnds)}."
            )
        if any(later <= earlier for earlier, later in zip(bnds, bnds[1:], strict=False)):
            raise ValueError(
                f"Figure3Config.phase_boundaries must be strictly increasing; got {list(bnds)}."
            )
        # Coerce to tuple so the field is hashable and immutable. ``frozen=True``
        # blocks normal assignment, so use ``object.__setattr__`` for the
        # normalization the dataclass performs on its own fields.
        object.__setattr__(self, "phase_boundaries", bnds)

        if self.place_field_centers is None:
            centers = np.arange(self.position_min, self.position_max + 1, 10, dtype=float)
        else:
            # Copy the caller's array so we don't write-protect their
            # reference; they keep a writable original.
            centers = np.asarray(self.place_field_centers).copy()
        # Write-protect against in-place mutation. ``Figure3Config`` is frozen,
        # so the field cannot be rebound (``config.place_field_centers = other``
        # raises), and marking the array read-only also blocks
        # ``config.place_field_centers[i] = x`` — the more dangerous case,
        # because it would silently corrupt every downstream decoder call.
        centers.setflags(write=False)
        object.__setattr__(self, "place_field_centers", centers)

        if self.trajectory_model not in ("discrete_matched", "continuous_reflected"):
            raise ValueError(
                "trajectory_model must be 'discrete_matched' or 'continuous_reflected'; "
                f"got {self.trajectory_model!r}."
            )
        if not (0.0 < self.hpd_coverage < 1.0):
            raise ValueError(f"hpd_coverage must lie in (0, 1); got {self.hpd_coverage}.")
        if not (
            np.isfinite(self.history_rate_matching_gain) and self.history_rate_matching_gain > 0.0
        ):
            raise ValueError(
                "history_rate_matching_gain must be positive; "
                f"got {self.history_rate_matching_gain}."
            )
        if not (0.0 <= self.sparse_control_ordinary_rate_scale <= 1.0):
            raise ValueError(
                "sparse_control_ordinary_rate_scale must lie in [0, 1]; "
                f"got {self.sparse_control_ordinary_rate_scale}."
            )
        sparse_centers = np.asarray(self.sparse_place_field_centers, dtype=float)
        if sparse_centers.ndim != 1 or sparse_centers.size < 1:
            raise ValueError("sparse_place_field_centers must contain at least one center.")
        if not np.all(np.isfinite(sparse_centers)) or np.any(
            (sparse_centers < self.position_min) | (sparse_centers > self.position_max)
        ):
            raise ValueError(
                "sparse_place_field_centers must lie in "
                f"[{self.position_min}, {self.position_max}]; "
                f"got {self.sparse_place_field_centers}."
            )
        multipliers = np.asarray(self.sparse_cell_rate_multipliers, dtype=float)
        if multipliers.shape != sparse_centers.shape:
            raise ValueError(
                "sparse_cell_rate_multipliers must have one entry per sparse cell; "
                f"got {multipliers.size} for {sparse_centers.size} centers."
            )
        if not np.all(np.isfinite(multipliers)) or np.any(multipliers <= 0.0):
            raise ValueError(
                "sparse_cell_rate_multipliers must be positive; "
                f"got {self.sparse_cell_rate_multipliers}."
            )
        object.__setattr__(
            self, "sparse_place_field_centers", tuple(float(c) for c in sparse_centers)
        )
        object.__setattr__(
            self, "sparse_cell_rate_multipliers", tuple(float(m) for m in multipliers)
        )
        if not np.isfinite(self.sparse_place_field_std) or self.sparse_place_field_std <= 0.0:
            raise ValueError(
                f"sparse_place_field_std must be positive; got {self.sparse_place_field_std}."
            )
        if (
            not np.isfinite(self.sparse_cell_peak_rate_per_step)
            or self.sparse_cell_peak_rate_per_step <= 0.0
        ):
            raise ValueError(
                f"sparse_cell_peak_rate_per_step must be positive; "
                f"got {self.sparse_cell_peak_rate_per_step}."
            )
        if not (0.0 <= self.sparse_cell_baseline_rate_fraction <= 1.0):
            raise ValueError(
                "sparse_cell_baseline_rate_fraction must lie in [0, 1]; "
                f"got {self.sparse_cell_baseline_rate_fraction}."
            )
        # Replay sub-window fractions must be ordered inside [0, 1]; an equal
        # or reversed pair silently empties/reverses the Replay window and
        # overlaps the well-specified baseline pool it is carved out of.
        if not (0.0 <= self.replay_start_fraction < self.replay_end_fraction <= 1.0):
            raise ValueError(
                "replay_start_fraction/replay_end_fraction must satisfy "
                "0 <= start < end <= 1; got "
                f"start={self.replay_start_fraction}, end={self.replay_end_fraction}."
            )
        if not (np.isfinite(self.replay_speed_per_step) and self.replay_speed_per_step > 0.0):
            raise ValueError(
                f"replay_speed_per_step must be positive; got {self.replay_speed_per_step}."
            )
        if not (
            np.isfinite(self.replay_place_field_rate_scale)
            and self.replay_place_field_rate_scale > 0.0
        ):
            raise ValueError(
                f"replay_place_field_rate_scale must be positive; "
                f"got {self.replay_place_field_rate_scale}."
            )


# Canonical ordered phase labels — the public contract of
# ``Figure3SimulationResult.phase_labels``. ``run_figure03_simulation`` passes each
# label explicitly at its ``_record_phase`` call site, in this order;
# ``Figure3SimulationResult.__post_init__`` checks the emitted sequence equals this
# tuple. Tests and downstream code import this tuple rather than re-typing
# the strings.
PHASE_LABELS: tuple[str, ...] = (
    "Matched Null Baseline",
    "Remap Misfit",
    "Clean Recovery",
    "History-Dependent Firing",
    "Clean Recovery",
    "Drift Misfit",
    "Clean Recovery",
    "Reflected Map Misfit",
    "Clean Recovery",
    "Sparse Population",
)


def compute_replay_step_window(config: Figure3Config) -> tuple[int, int]:
    """Global ``[start, end)`` step bounds of the replay sub-window.

    The replay event lives inside clean-recovery 2, spanning the fractional
    sub-window ``[replay_start_fraction, replay_end_fraction)`` of that phase.
    Deterministic from the phase ladder and the replay fractions so
    ``run_figure03_simulation`` and the figure-3b summary columns stay in
    sync.

    Parameters
    ----------
    config : Figure3Config
        Provides the phase-boundary ladder and replay fractions.

    Returns
    -------
    tuple of (int, int)
        Half-open ``[start, end)`` global step indices of the replay sweep.
    """
    bnd = config.phase_boundaries
    start = bnd[PhaseBoundary.HIST_DEP_END]
    end = bnd[PhaseBoundary.RECOVERY2_END]
    n = end - start
    r0 = start + int(round(n * config.replay_start_fraction))
    r1 = start + int(round(n * config.replay_end_fraction))
    # Ordered floating-point fractions can still collapse to the same integer
    # step after rounding. A genuine out-and-back trajectory needs at least a
    # start, a turn, and a return sample.
    if not (start <= r0 < r1 <= end) or r1 - r0 < 3:
        raise ValueError(
            "Replay fractions must resolve to a window of at least 3 steps "
            f"inside clean-recovery 2; got [{r0}, {r1}) within [{start}, {end})."
        )
    return r0, r1
