"""Reusable figure-3 simulation driver.

The figure-3 demo simulates a hippocampal-style decoder under a
sequence of misfit conditions (clean baseline, remapping, flat
firing, fast movement, momentum). The simulation pipeline is the
same one that drives ``scripts/generate_figure03.py`` and
``statespacecheck_paper.interactive.cache.build_simulated_cache``;
both call ``run_figure03_simulation`` so the figure and the
interactive viewer's simulated cache stay byte-identical.

The figure-generation script extends this with diagnostic threshold
computation + plotting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from statespacecheck_paper.analysis import DecodeParams, decode_and_diagnostics
from statespacecheck_paper.simulation import (
    gaussian_transition_matrix,
    reflect_into_interval,
    simulate_spikes_flat_rate,
    simulate_spikes_position_tuned,
    simulate_walk,
)


def run_figure03_simulation(
    params: DecodeParams | None = None,
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run the figure-3 phased simulation and decode it.

    Phases (in order):

    1. Clean baseline
    2. Remapping misfit
    3. Clean recovery
    4. Flat-firing misfit
    5. Clean recovery
    6. Fast-movement misfit
    7. Clean recovery
    8. Drift misfit
    9. Clean recovery
    10. Broad-decoder phase (decoder uses ~40x inflated transition matrix;
        engineered to inflate KL while HPD overlap and the rank-based
        p-value stay near baseline)
    11. Clean recovery
    12. Tight-decoder phase (decoder uses ~50x tighter transition matrix
        with stationary animal; engineered to inflate KL in the opposite
        shape-mismatch direction while HPD overlap and p-value stay near
        baseline)

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
    dict
        Dictionary with keys:

        - ``params``: the ``DecodeParams`` used.
        - ``xs``: position grid, shape ``(n_bins,)``.
        - ``x_true``: true linear position, shape ``(n_time,)``.
        - ``spikes``: spike-count matrix, shape ``(n_time, n_cells)``.
        - ``metrics``: dict from :func:`decode_and_diagnostics`
          with ``predictive``, ``posterior``, ``likelihood``, plus
          per-cell diagnostics and per-spike event arrays.
        - ``phase_labels``: list of per-phase descriptors.
        - ``phase_boundaries``: cumulative phase end indices, useful
          for marking misfit windows on the time axis.
    """
    if params is None:
        params = DecodeParams()
    base_seed = params.base_seed if seed is None else seed
    rng = np.random.default_rng(base_seed)

    if params.pf_centers is None:
        raise ValueError("params.pf_centers must be initialized")
    # Bind to a local so closures below see a non-Optional type.
    pf_centers = params.pf_centers

    xs = np.arange(params.xs_min, params.xs_max + params.xs_step, params.xs_step, dtype=float)
    transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred)
    transition_matrix_narrow = gaussian_transition_matrix(xs, params.sigx_pred_fast_phase)
    transition_matrix_inflated = gaussian_transition_matrix(xs, params.sigx_pred_slow_phase)
    transition_matrix_tight = gaussian_transition_matrix(xs, params.sigx_pred_tight)

    phases: list[tuple[NDArray[np.floating], NDArray[np.int_]]] = []
    phase_labels: list[str] = []
    x_last: float = 0.0

    def _walk(n: int, sig: float) -> NDArray[np.floating]:
        return simulate_walk(n, sig, x_last, params.xs_min, params.xs_max, rng)

    def _spikes_position_tuned(x: NDArray[np.floating]) -> NDArray[np.int_]:
        return simulate_spikes_position_tuned(
            x, pf_centers, params.pf_width, params.rate_scale, rng
        )

    def _add_phase(label: str, x: NDArray[np.floating], sp: NDArray[np.int_]) -> None:
        nonlocal x_last
        phases.append((x, sp))
        phase_labels.append(label)
        x_last = float(x[-1])

    # 1. Clean baseline
    n = params.T_remap_start
    x = _walk(n, params.sigx_pred)
    _add_phase("Clean Baseline", x, _spikes_position_tuned(x))

    # 2. Remapping misfit
    n = params.T_remap_end - params.T_remap_start
    x = _walk(n, params.sigx_pred)
    _add_phase("Remapping Misfit", x, _spikes_position_tuned(x))

    # 3. Recovery 1
    n = params.T_recovery1_end - params.T_remap_end
    x = _walk(n, params.sigx_pred)
    _add_phase("Clean Recovery", x, _spikes_position_tuned(x))

    # 4. Flat firing misfit (cells lose spatial tuning)
    n = params.T_flat_end - params.T_recovery1_end
    x = _walk(n, params.sigx_pred)
    sp = simulate_spikes_flat_rate(n, len(pf_centers), rate=7e-3, rng=rng)
    _add_phase("Flat Firing Misfit", x, sp)

    # 5. Recovery 2
    n = params.T_recovery2_end - params.T_flat_end
    x = _walk(n, params.sigx_pred)
    _add_phase("Clean Recovery", x, _spikes_position_tuned(x))

    # 6. Fast movement misfit (decoder uses narrow transition; animal moves fast)
    n = params.T_fast_end - params.T_recovery2_end
    x = _walk(n, params.sigx_true_fast)
    _add_phase("Fast Movement Misfit", x, _spikes_position_tuned(x))

    # 7. Recovery 3
    n = params.T_recovery3_end - params.T_fast_end
    x = _walk(n, params.sigx_pred)
    _add_phase("Clean Recovery", x, _spikes_position_tuned(x))

    # 8. Drift misfit (persistent velocity vs memoryless random walk)
    n = params.T_slow_end - params.T_recovery3_end
    momentum = 0.95
    x_mom = np.zeros(n)
    x_mom[0] = x_last
    velocity = 0.0
    for t in range(1, n):
        velocity = momentum * velocity + rng.normal(0, params.sigx_pred)
        x_mom[t] = x_mom[t - 1] + velocity
    x = reflect_into_interval(x_mom, float(params.xs_min), float(params.xs_max))
    _add_phase("Drift Misfit", x, _spikes_position_tuned(x))

    # 9. Clean Recovery 4 (resets walk after drift)
    n = params.T_recovery4_end - params.T_slow_end
    x = _walk(n, params.sigx_pred)
    _add_phase("Clean Recovery", x, _spikes_position_tuned(x))

    # 10. Broad-Decoder Phase
    #     True walk and spikes are normal; the decoder applies an inflated
    #     transition matrix (sigx_pred_slow_phase ~ 40x baseline). Predictive
    #     becomes very wide while per-spike likelihoods stay narrow at the
    #     correct location -> KL inflates strongly while HPD overlap and the
    #     rank-based p-value stay near baseline (KL false-positive).
    n = params.T_broad_decoder_end - params.T_recovery4_end
    x = _walk(n, params.sigx_pred)
    _add_phase("Broad-Decoder Phase", x, _spikes_position_tuned(x))

    # 11. Clean Recovery 5
    n = params.T_recovery5_end - params.T_broad_decoder_end
    x = _walk(n, params.sigx_pred)
    _add_phase("Clean Recovery", x, _spikes_position_tuned(x))

    # 12. Tight-Decoder Phase
    #     Animal stationary (sigx_true_slow=0.0); decoder applies a very
    #     tight transition matrix (sigx_pred_tight ~ 50x tighter than
    #     baseline). Predictive collapses to nearly a point estimate while
    #     per-spike likelihoods are normal-width at the same location -> KL
    #     inflates in the opposite shape-mismatch direction; HPD overlap and
    #     p-value stay near baseline.
    n = params.T_tight_decoder_end - params.T_recovery5_end
    x = _walk(n, params.sigx_true_slow)  # stationary
    _add_phase("Tight-Decoder Phase", x, _spikes_position_tuned(x))

    x_true = np.concatenate([p_x for p_x, _ in phases], axis=0)
    spikes = np.vstack([p_s for _, p_s in phases])

    metrics = decode_and_diagnostics(
        spikes=spikes,
        xs=xs,
        transition_matrix=transition_matrix,
        pf_centers=pf_centers,
        pf_width=params.pf_width,
        rate_scale=params.rate_scale,
        remap_window=params.remap_window,
        remap_from_to=params.remap_from_to,
        transition_matrix_narrow=transition_matrix_narrow,
        narrow_window=(params.T_recovery2_end, params.T_fast_end),
        transition_matrix_inflated=transition_matrix_inflated,
        inflate_window=(params.T_recovery4_end, params.T_broad_decoder_end),
        transition_matrix_tight=transition_matrix_tight,
        tight_window=(params.T_recovery5_end, params.T_tight_decoder_end),
    )

    boundaries = np.cumsum([len(p_x) for p_x, _ in phases]).tolist()

    return {
        "params": params,
        "xs": xs,
        "x_true": x_true,
        "spikes": spikes,
        "metrics": metrics,
        "phase_labels": phase_labels,
        "phase_boundaries": boundaries,
    }
