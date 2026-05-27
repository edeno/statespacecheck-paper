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

# ``decode_and_diagnostics`` and ``simulate_walk`` are re-exported as part of
# this script's public surface — tests/test_figures.py asserts they are
# accessible at module level.
from statespacecheck_paper.analysis import (
    DecodeParams,
    PhaseBoundary,
    compute_thresholds,
    decode_and_diagnostics,  # noqa: F401 — re-exported public surface
)
from statespacecheck_paper.figure03_demo import run_figure03_simulation
from statespacecheck_paper.plotting import plot_combined_diagnostics
from statespacecheck_paper.simulation import (
    simulate_walk,  # noqa: F401 — re-exported public surface
)
from statespacecheck_paper.style import save_figure


def run_demo(params: DecodeParams) -> None:
    """Run the full diagnostic demonstration with multiple simulation phases.

    Generates Figure 3: a Bayesian decoder stepped through five model-misfit
    conditions separated by clean-recovery windows, chosen to span the
    metric-disagreement space (which misfits each of HPD overlap, KL
    divergence, and the rank-based predictive p-value detects vs. misses).

    Parameters
    ----------
    params : DecodeParams
        Decoding parameters containing timeline structure, place field settings,
        and simulation configuration. Must have pf_centers initialized.

    Returns
    -------
    None
        Saves figure to manuscript/figures/main/figure03.{pdf,png}.

    Notes
    -----
    The simulation includes 8 phases (4 misfits, each preceded by a
    clean-recovery or baseline window):

    1. Clean baseline (0 - T_remap_start): Model fits well.
    2. Remap misfit (T_remap_start - T_remap_end): A subset of cells use
       swapped place-field identities — an observation-model misfit.
    3. Clean recovery (T_remap_end - T_recovery1_end).
    4. History-dependent firing misfit (T_recovery1_end - T_hist_dep_end):
       Spikes are generated with a hard refractory period plus bursting;
       the decoder still assumes Poisson. The misfit lives in spike-train
       temporal correlations, so per-spike spatial diagnostics largely
       miss it.
    5. Clean recovery (T_hist_dep_end - T_recovery2_end).
    6. Drift misfit (T_recovery2_end - T_drift_end): The trajectory has
       persistent velocity (AR(1), drift_momentum); the decoder assumes a
       memoryless random walk.
    7. Clean recovery (T_drift_end - T_recovery3_end).
    8. Wide dynamics noise (T_recovery3_end - T_wide_dynamics_end): The
       decoder applies an inflated transition matrix; engineered to
       inflate KL while HPD overlap and the rank-based p-value stay near
       baseline (the KL false-positive case).

    Figure 3 has two panels: a time-series block (predictive, likelihood,
    raster, and the three diagnostics over time) and a summary heatmap of
    the fraction of spike events exceeding the baseline threshold per
    phase per metric.

    Diagnostic thresholds are computed from the clean baseline period.
    """
    sim = run_figure03_simulation(params)
    x_true = sim.x_true
    spikes = sim.spikes
    metrics = sim.metrics

    # ``run_figure03_simulation`` already validates this, but rebind to
    # a non-Optional local so ``plot_combined_diagnostics``'s typed
    # signature is satisfied without a cast.
    pf_centers = params.pf_centers
    assert pf_centers is not None, "pf_centers must be initialized"

    # Thresholds from the clean-baseline window (everything before the remap
    # misfit starts).
    thresholds = compute_thresholds(
        metrics, baseline_end=params.phase_boundaries[PhaseBoundary.REMAP_START]
    )

    # Plot combined diagnostics figure
    plot_combined_diagnostics(
        x_true,
        spikes.astype(np.float64),
        metrics,
        thresholds,
        params,
        pf_centers,
    )

    # Save figure (uses plt.gcf() to get current figure)
    save_figure("manuscript/figures/main/figure03", close=False)
    print("\nFigure 3 saved to manuscript/figures/main/figure03.{pdf,png}")


if __name__ == "__main__":
    # Full-size run (~32k 1-ms steps). To prototype quickly, shrink the
    # timeline by passing a smaller ``phase_boundaries`` tuple, e.g.:
    #   params = DecodeParams(
    #       phase_boundaries=(600, 900, 1100, 1400, 1600, 1900, 2100, 2400),
    #   )
    params = DecodeParams()
    run_demo(params)
