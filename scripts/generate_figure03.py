"""Demonstration simulation for state space model diagnostics.

This script simulates a Bayesian decoder with periods of good and poor model fit,
then computes diagnostic metrics using the statespacecheck package.

The simulation + decode pipeline lives in
``statespacecheck_paper.figure03_demo.run_figure03_simulation``; this script
adds threshold computation + figure rendering on top so the same simulation
arrays drive both the static figure here and the figure-3 simulation cache
consumed by the interactive viewer.
"""

from __future__ import annotations

import numpy as np

# Imports preserved as the public surface of this script (tests/test_figures.py
# asserts these are accessible at module level — see TestFigure03Integration).
from statespacecheck_paper.analysis import (  # noqa: F401
    DecodeParams,
    compute_thresholds,
    decode_and_diagnostics,
)
from statespacecheck_paper.figure03_demo import run_figure03_simulation
from statespacecheck_paper.plotting import (
    plot_combined_diagnostics,
)
from statespacecheck_paper.simulation import (  # noqa: F401
    gaussian_transition_matrix,
    reflect_into_interval,
    simulate_spikes_flat_rate,
    simulate_spikes_position_tuned,
    simulate_walk,
)
from statespacecheck_paper.style import save_figure


def run_demo(params: DecodeParams) -> None:
    """Run the full diagnostic demonstration with multiple simulation phases.

    Generates Figure 3 showing a Bayesian decoder with periods of good and poor
    model fit across 12 phases: baseline, four misfits + recoveries, drift, and
    two engineered metric-stress phases (broad-decoder and tight-decoder).

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
    8. Drift misfit (T_recovery3_end - T_slow_end): Animal has persistent
       velocity but model assumes memoryless random walk
    9. Recovery 4 (T_slow_end - T_recovery4_end): Return to good fit
    10. Broad-decoder phase (T_recovery4_end - T_broad_decoder_end): Decoder
        applies an inflated transition matrix; engineered to inflate KL while
        HPD overlap and p-value stay near baseline (KL false-positive).
    11. Recovery 5 (T_broad_decoder_end - T_recovery5_end): Return to good fit
    12. Tight-decoder phase (T_recovery5_end - T_tight_decoder_end): Decoder
        applies a very tight transition matrix with stationary animal;
        engineered to inflate KL in the opposite shape-mismatch direction
        while HPD overlap and p-value stay near baseline.

    Beyond the time-series block, figure 3 includes a recovery-transient panel
    (KL recovers more slowly than HPD overlap and p-value after each misfit),
    a per-spike scatter (KL vs HPD overlap, colored by phase, showing the
    broad-decoder cluster sitting at high KL AND high HPDO), and a summary
    heatmap of % events exceeding baseline threshold per phase per metric.

    Diagnostic thresholds are computed from the clean baseline period.
    """
    sim = run_figure03_simulation(params)
    xs = sim["xs"]
    x_true = sim["x_true"]
    spikes = sim["spikes"]
    metrics = sim["metrics"]

    # ``run_figure03_simulation`` already validates this, but rebind to
    # a non-Optional local so ``plot_combined_diagnostics``'s typed
    # signature is satisfied without a cast.
    pf_centers = params.pf_centers
    assert pf_centers is not None, "pf_centers must be initialized"

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
        pf_centers,
        phase_boundaries=sim["phase_boundaries"],
        phase_labels=sim["phase_labels"],
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
