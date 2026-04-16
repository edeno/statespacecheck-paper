"""Demonstration simulation for state space model diagnostics.

This script simulates a Bayesian decoder with periods of good and poor model fit,
then computes diagnostic metrics using the statespacecheck package.
"""

from __future__ import annotations

import numpy as np

from statespacecheck_paper.analysis import (
    DecodeParams,
    compute_thresholds,
    decode_and_diagnostics,
)
from statespacecheck_paper.plotting import (
    plot_combined_diagnostics,
)
from statespacecheck_paper.simulation import (
    gaussian_transition_matrix,
    reflect_into_interval,
    simulate_spikes_flat_rate,
    simulate_spikes_position_tuned,
    simulate_walk,
)
from statespacecheck_paper.style import save_figure

# -----------------------------
# Main orchestration
# -----------------------------


def run_demo(params: DecodeParams) -> None:
    """Run the full diagnostic demonstration with multiple simulation phases.

    Generates Figure 3 showing a Bayesian decoder with periods of good and poor
    model fit across 8 phases: baseline, remapping misfit, flat firing misfit,
    fast movement misfit, and momentum misfit, with recovery periods between.

    Parameters
    ----------
    params : DecodeParams
        Decoding parameters containing timeline structure, place field settings,
        and simulation configuration. Must have pf_centers initialized.

    Returns
    -------
    None
        Saves figure to figures/main/figure03.{pdf,png}.

    Notes
    -----
    The simulation includes the following phases:
    1. Clean baseline (0 - T_remap_start): Model fits well
    2. Remapping misfit (T_remap_start - T_remap_end): Place fields remap
    3. Recovery 1 (T_remap_end - T_recovery1_end): Return to good fit
    4. Flat firing misfit (T_recovery1_end - T_flat_end): Cells lose tuning
    5. Recovery 2 (T_flat_end - T_recovery2_end): Return to good fit
    6. Fast movement misfit (T_recovery2_end - T_fast_end): Animal moves faster
       than model expects
    7. Recovery 3 (T_fast_end - T_recovery3_end): Return to good fit
    8. Momentum misfit (T_recovery3_end - T_slow_end): Animal has persistent
       velocity but model assumes memoryless random walk

    Diagnostic thresholds are computed from the clean baseline period.
    """
    rng = np.random.default_rng(params.base_seed)

    # Ensure pf_centers is initialized
    assert params.pf_centers is not None, "pf_centers must be initialized"

    # Grid & transition matrices
    xs = np.arange(params.xs_min, params.xs_max + params.xs_step, params.xs_step, dtype=float)
    transition_matrix = gaussian_transition_matrix(xs, params.sigx_pred)
    transition_matrix_narrow = gaussian_transition_matrix(xs, params.sigx_pred_fast_phase)

    # Generate all phases with recovery periods
    phases = []
    phase_labels = []

    # Phase 1: Clean baseline (0 - T_remap_start)
    x_last = 0.0
    n_time = params.T_remap_start
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Clean Baseline")
    x_last = x_true_phase[-1]

    # Phase 2: Remapping misfit (T_remap_start - T_remap_end)
    n_time = params.T_remap_end - params.T_remap_start
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Remapping Misfit")
    x_last = x_true_phase[-1]

    # Phase 3: Recovery 1 (T_remap_end - T_recovery1_end)
    n_time = params.T_recovery1_end - params.T_remap_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Clean Recovery")
    x_last = x_true_phase[-1]

    # Phase 4: Flat firing misfit (T_recovery1_end - T_flat_end)
    n_time = params.T_flat_end - params.T_recovery1_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_flat_rate(n_time, len(params.pf_centers), rate=7e-3, rng=rng)
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Flat Firing Misfit")
    x_last = x_true_phase[-1]

    # Phase 5: Recovery 2 (T_flat_end - T_recovery2_end)
    n_time = params.T_recovery2_end - params.T_flat_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Clean Recovery")
    x_last = x_true_phase[-1]

    # Phase 6: Fast movement misfit (T_recovery2_end - T_fast_end)
    # Transition model misfit: decoder uses narrow transition matrix (sigx=0.1),
    # animal moves fast (sigx=10.0)
    n_time = params.T_fast_end - params.T_recovery2_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_true_fast, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Fast Movement Misfit")
    x_last = x_true_phase[-1]

    # Phase 7: Recovery 3 (T_fast_end - T_recovery3_end)
    n_time = params.T_recovery3_end - params.T_fast_end
    x_true_phase = simulate_walk(
        n_time, params.sigx_pred, x_last, params.xs_min, params.xs_max, rng
    )
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Clean Recovery")
    x_last = x_true_phase[-1]

    # Phase 8: Momentum misfit (T_recovery3_end - T_slow_end)
    # Transition model misfit: decoder assumes memoryless random walk,
    # but the animal has persistent momentum (correlated steps).
    # The predictive distribution is systematically behind the true position.
    n_time = params.T_slow_end - params.T_recovery3_end
    momentum = 0.95  # High autocorrelation of velocity
    x_momentum = np.zeros(n_time)
    x_momentum[0] = x_last
    velocity = 0.0
    for t in range(1, n_time):
        velocity = momentum * velocity + rng.normal(0, params.sigx_pred)
        x_momentum[t] = x_momentum[t - 1] + velocity
    x_true_phase = reflect_into_interval(x_momentum, float(params.xs_min), float(params.xs_max))
    spikes_phase = simulate_spikes_position_tuned(
        x_true_phase, params.pf_centers, params.pf_width, params.rate_scale, rng
    )
    phases.append((x_true_phase, spikes_phase))
    phase_labels.append("Drift Misfit")

    # Concatenate all phases
    x_true = np.concatenate([x for x, _ in phases], axis=0)
    spikes = np.vstack([s for _, s in phases])

    # Decode (vectorized within time)
    metrics = decode_and_diagnostics(
        spikes=spikes,
        xs=xs,
        transition_matrix=transition_matrix,
        pf_centers=params.pf_centers,
        pf_width=params.pf_width,
        rate_scale=params.rate_scale,
        remap_window=params.remap_window,
        remap_from_to=params.remap_from_to,
        transition_matrix_narrow=transition_matrix_narrow,
        narrow_window=(params.T_recovery2_end, params.T_fast_end),
    )

    # Thresholds from clean baseline window (first 6k timesteps, before remapping starts)
    thresholds = compute_thresholds(metrics, baseline_end=params.T_remap_start)

    # Plot combined diagnostics figure
    plot_combined_diagnostics(
        xs,
        x_true,
        spikes.astype(np.float64),
        metrics,
        thresholds,
        params,
        params.pf_centers,
    )

    # Save figure (uses plt.gcf() to get current figure)
    save_figure("figures/main/figure03", close=False)
    print("\nFigure 3 saved to figures/main/figure03.{pdf,png}")


if __name__ == "__main__":
    # Default params mirror the MATLAB script. To run quickly while prototyping,
    # reduce T1/T2/T3 here.
    params = DecodeParams()
    # e.g., for a fast smoke test:
    # params = DecodeParams(T1=3_000, T2=4_000, T3=5_000)
    run_demo(params)
