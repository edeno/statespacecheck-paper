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
    MisfitSchedule,
    MisfitWindow,
    decode_and_diagnostics,
    get_remapped_pf_centers,
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
    metrics: dict[str, NDArray[np.floating] | NDArray[np.intp]]
    phase_labels: list[str]
    phase_boundaries: list[int]

    def __post_init__(self) -> None:
        """Enforce length and timeline-consistency invariants."""
        if list(self.phase_labels) != list(PHASE_LABELS):
            raise ValueError(
                f"phase_labels must equal PHASE_LABELS in order; "
                f"got {self.phase_labels!r} vs canonical {list(PHASE_LABELS)!r}"
            )
        if len(self.phase_boundaries) != len(self.phase_labels):
            raise ValueError(
                f"phase_boundaries length ({len(self.phase_boundaries)}) "
                f"must equal phase_labels length ({len(self.phase_labels)})."
            )
        if self.spikes.shape[0] != self.x_true.shape[0]:
            raise ValueError(
                f"spikes timeline ({self.spikes.shape[0]}) must equal "
                f"x_true timeline ({self.x_true.shape[0]})."
            )
        if self.phase_boundaries[-1] != self.x_true.shape[0]:
            raise ValueError(
                f"final phase boundary ({self.phase_boundaries[-1]}) must "
                f"equal x_true timeline ({self.x_true.shape[0]})."
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
        TypedDict with ``params``, ``xs``, ``x_true``, ``spikes``,
        ``metrics``, ``phase_labels``, ``phase_boundaries``.
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

    # 1. Clean baseline
    n = params.T_remap_start
    x = _walk(n, params.sigx_pred)
    _add_phase(x, _spikes_position_tuned(x))

    # 2. Remap misfit — handled by ``remap_window`` inside decode_and_diagnostics.
    #    The spike *generation* is normal position-tuned; the decoder is the one
    #    that uses remapped PF centers during this window.
    n = params.T_remap_end - params.T_remap_start
    x = _walk(n, params.sigx_pred)
    _add_phase(x, _spikes_position_tuned(x))

    # 3. Clean recovery 1
    n = params.T_recovery1_end - params.T_remap_end
    x = _walk(n, params.sigx_pred)
    _add_phase(x, _spikes_position_tuned(x))

    # 4. History-Dependent Firing Misfit
    #    Cells generate spikes via ``simulate_spikes_history_dependent``:
    #    hard 1-step (1 ms) refractory + 2-10 step (2-10 ms) burst window
    #    with 3× rate boost. Decoder still treats every spike as an
    #    independent Poisson draw at the cell's standard rate; the misfit
    #    lives in the *temporal* correlations and is largely invisible to
    #    per-spike spatial diagnostics.
    n = params.T_hist_dep_end - params.T_recovery1_end
    x = _walk(n, params.sigx_pred)
    sp = simulate_spikes_history_dependent(x, pf_centers, params.pf_width, params.rate_scale, rng)
    _add_phase(x, sp)

    # 5. Clean recovery 2
    n = params.T_recovery2_end - params.T_hist_dep_end
    x = _walk(n, params.sigx_pred)
    _add_phase(x, _spikes_position_tuned(x))

    # 6. Drift Misfit — persistent-velocity walk; decoder assumes memoryless.
    n = params.T_drift_end - params.T_recovery2_end
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
    n = params.T_recovery3_end - params.T_drift_end
    x = _walk(n, params.sigx_pred)
    _add_phase(x, _spikes_position_tuned(x))

    # 8. Wide Dynamics Noise — decoder applies an inflated transition matrix
    #    (~40× baseline). Predictive becomes wide; per-spike likelihoods stay
    #    narrow at the firing cell's PF -> KL inflates strongly while HPD
    #    overlap and the rank-based p-value stay near baseline (KL
    #    false-positive case).
    n = params.T_wide_dynamics_end - params.T_recovery3_end
    x = _walk(n, params.sigx_pred)
    _add_phase(x, _spikes_position_tuned(x))

    x_true = np.concatenate([p_x for p_x, _ in phases], axis=0)
    spikes = np.vstack([p_s for _, p_s in phases])

    # The two decoder-side misfits as a single schedule:
    # - Remap: the decoder's posterior update uses remapped place-field
    #   centers (``decoder_rates``); the diagnostics still reference the
    #   original Gaussian fields (``diagnostic_rates`` left None) so they
    #   correctly flag the mismatch.
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
                params.T_remap_start,
                params.T_remap_end,
                decoder_rates=remapped_rates,
            ),
            MisfitWindow(
                params.T_recovery3_end,
                params.T_wide_dynamics_end,
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
        phase_labels=phase_labels,
        phase_boundaries=boundaries,
    )
