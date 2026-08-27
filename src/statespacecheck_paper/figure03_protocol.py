"""Figure-3 experimental protocol: phase ladder and immutable configuration.

The figure-3 simulation walks a hippocampal-style decoder through three misfit
conditions (remap, history-dependent firing, drift) and two specificity controls
(a replay event embedded in clean-recovery 2, and a final sparse-population
epoch), separated by clean-recovery windows. This module holds the immutable
experimental configuration (:class:`Figure3Config`), the phase-transition index
enum (:class:`PhaseBoundary`), the canonical ordered phase labels
(:data:`PHASE_LABELS`), and the replay-window step-bound helper
(:func:`compute_replay_step_window`). It imports no sibling paper module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray


class PhaseBoundary(IntEnum):
    """Indices into :attr:`Figure3Config.phase_boundaries`.

    Each member is the position of one figure-3 phase transition in
    the 8-tuple. Use as ``config.phase_boundaries[PhaseBoundary.REMAP_END]``
    rather than indexing by literal integer, so a phase-ladder
    reshuffle stays compile-time-checkable.
    """

    REMAP_START = 0  # end of clean baseline
    REMAP_END = 1  # end of remap misfit
    RECOVERY1_END = 2  # end of clean recovery 1
    HIST_DEP_END = 3  # end of history-dependent firing misfit
    RECOVERY2_END = 4  # end of clean recovery 2
    DRIFT_END = 5  # end of drift misfit
    RECOVERY3_END = 6  # end of clean recovery 3
    SPARSE_POP_END = 7  # end of sparse-population control


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
    32_000,
)


@dataclass(frozen=True)
class Figure3Config:
    """Parameters for the figure-3 decoding simulation.

    The simulation walks through three misfit conditions and one sparse-
    activity control, separated by clean-recovery windows. Time steps are
    1 ms by convention — the simulation math itself is dt-agnostic, but the
    default parameters
    (`place_field_rate_scale=5.0`, refractory and burst windows in
    ``simulate_spikes_history_dependent``) are tuned for that mapping
    and yield hippocampally-realistic spike rates and timescales.

    **Timeline Structure** (default; all indices in 1-ms steps):

    - 0–6k: Clean baseline
    - 6k–10k: Remap misfit (4 s)
    - 10k–14k: Clean recovery
    - 14k–18k: History-dependent firing misfit (4 s)
    - 18k–22k: Clean recovery
    - 22k–26k: Drift misfit (4 s)
    - 26k–30k: Clean recovery
    - 30k–32k: Sparse-population control (2 s)

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
    prediction_step_std : float, default 0.5
        Decoder's baseline dynamics standard deviation.
    drift_momentum : float, default 0.8
        AR(1) coefficient on the animal's velocity during the drift
        misfit phase. The true trajectory is
        ``x[t] = x[t-1] + v[t]`` with
        ``v[t] = drift_momentum * v[t-1] + N(0, prediction_step_std)``. The
        decoder assumes ``x[t] = x[t-1] + N(0, prediction_step_std)`` (no
        persistent velocity).
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
    sparse_position : float, default 30.0
        Fixed location where the sparse population is active in the final
        control phase.
    sparse_approach_duration_steps : int, default 1000
        Number of steps at the end of clean recovery 3 used for a gradual
        approach to ``sparse_position``.
    sparse_control_ordinary_rate_scale : float, default 0.0
        Multiplicative rate applied to the eleven ordinary place cells during
        the sparse-population control. Zero represents a silent ordinary
        ensemble.
    sparse_cell_count : int, default 5
        Number of narrow cells forming the sparse population clustered at
        ``sparse_position``.
    sparse_place_field_spread : float, default 1.5
        Half-range (position units) over which the ``sparse_cell_count`` field
        centers are spread symmetrically about ``sparse_position``. Zero
        stacks all centers at ``sparse_position``.
    sparse_place_field_std : float, default 2.0
        Standard deviation, in position units, of each narrow sparse-population
        field.
    sparse_cell_peak_rate_per_step : float, default 0.001
        Per-cell peak firing rate in spikes per 1-ms step (1 Hz). Sized so the
        population's *aggregate* rate stays ~5 Hz: with more cells firing, a
        higher aggregate rate would shorten the gaps between spikes and let the
        prediction re-concentrate, suppressing the KL response that the sparse,
        immobile regime is meant to illustrate.
    sparse_cell_baseline_rate_fraction : float, default 0.01
        Fraction of the active per-cell rate used before the final control.

    Examples
    --------
    >>> config = Figure3Config()
    >>> config.phase_boundaries[PhaseBoundary.REMAP_START]
    6000
    >>> config.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
    32000
    >>> config.place_field_centers
    array([  0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 100.])
    """

    # Phase ladder. One boundary per :class:`PhaseBoundary` member,
    # strictly increasing; validated in __post_init__.
    phase_boundaries: tuple[int, ...] = _DEFAULT_PHASE_BOUNDARIES

    # Decoder & dynamics parameters
    prediction_step_std: float = 0.5  # baseline dynamics std
    drift_momentum: float = 0.8  # AR(1) coefficient for drift-misfit trajectory

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

    # Sparse-population control. The animal approaches a fixed location at the
    # end of clean recovery 3, then remains there (immobile) while the ordinary
    # ensemble becomes quiet. A small population of narrow, sharply tuned cells
    # clustered at that location fires sparsely, each cell an independent
    # Poisson process increasing from a small baseline gain to its full rate.
    # With little intervening population information, the predictive spreads
    # between the isolated spikes; each spike supplies a narrow likelihood
    # contained in that broad prediction. This is a correctly modeled,
    # low-activity observation regime, not a transition-model perturbation.
    sparse_position: float = 30.0
    sparse_approach_duration_steps: int = 1_000
    sparse_control_ordinary_rate_scale: float = 0.0
    sparse_cell_count: int = 5
    sparse_place_field_spread: float = 1.5
    sparse_place_field_std: float = 2.0
    sparse_cell_peak_rate_per_step: float = 0.001  # spikes/ms = 1 Hz/cell (~5 Hz aggregate)
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

        if not (self.position_min <= self.sparse_position <= self.position_max):
            raise ValueError(
                f"sparse_position must lie in [{self.position_min}, {self.position_max}]; "
                f"got {self.sparse_position}."
            )
        if self.sparse_approach_duration_steps < 0:
            raise ValueError(
                f"sparse_approach_duration_steps must be non-negative; "
                f"got {self.sparse_approach_duration_steps}."
            )
        if not (0.0 <= self.sparse_control_ordinary_rate_scale <= 1.0):
            raise ValueError(
                "sparse_control_ordinary_rate_scale must lie in [0, 1]; "
                f"got {self.sparse_control_ordinary_rate_scale}."
            )
        if self.sparse_cell_count < 1:
            raise ValueError(f"sparse_cell_count must be >= 1; got {self.sparse_cell_count}.")
        if not np.isfinite(self.sparse_place_field_spread) or self.sparse_place_field_spread < 0.0:
            raise ValueError(
                f"sparse_place_field_spread must be finite and non-negative; "
                f"got {self.sparse_place_field_spread}."
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
    "Clean Baseline",
    "Remap Misfit",
    "Clean Recovery",
    "History-Dependent Firing",
    "Clean Recovery",
    "Drift Misfit",
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
