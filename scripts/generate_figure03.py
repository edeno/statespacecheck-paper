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
    decode_and_diagnostics,  # noqa: F401 — re-exported public surface
)
from statespacecheck_paper.figure03_demo import (
    estimate_stable_summary,
    run_figure03_simulation,
)
from statespacecheck_paper.plotting import plot_combined_diagnostics
from statespacecheck_paper.simulation import (
    simulate_walk,  # noqa: F401 — re-exported public surface
)
from statespacecheck_paper.style import save_figure

# Number of independent realizations pooled to stabilize the panel-(b)
# summary. A single run's flag thresholds and per-phase fractions are
# noisy (the KL 99th-percentile threshold varies ~17% across seeds, and
# the remap flag fraction swings with the trajectory); pooling many
# realizations gives a stable threshold and a median per-phase summary.
# The seed-1 realization shown in panel (a) is one of these.
N_REALIZATIONS = 100


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

    Figure 3 has two panels: panel (a) is a time-series block (predictive,
    likelihood, raster, and the three diagnostics over time) for a single
    realization (seed ``params.base_seed``); panel (b) is a summary heatmap
    of the percent of spike events flagged as poor fit per phase per
    metric, reported as the median across ``N_REALIZATIONS`` independent
    realizations.

    Both the flag thresholds and the panel-(b) fractions are stabilized by
    pooling ``N_REALIZATIONS`` realizations via
    :func:`statespacecheck_paper.figure03_demo.estimate_stable_summary`
    (the thresholds from the pooled clean-baseline windows), rather than
    relying on a single noisy run.
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

    # Pool many realizations for a stable threshold (from the pooled
    # clean-baseline windows) and a stable median [IQR] panel-(b) summary.
    summary = estimate_stable_summary(params, n_realizations=N_REALIZATIONS)

    # Plot combined diagnostics figure. Panel (a) shows the single seed-1
    # realization; panel (b) shows the pooled median fractions scored
    # against the pooled-baseline thresholds.
    plot_combined_diagnostics(
        x_true,
        spikes.astype(np.float64),
        metrics,
        summary.thresholds,
        params,
        pf_centers,
        summary_median=summary.frac_median,
    )

    # Save figure (uses plt.gcf() to get current figure)
    save_figure("manuscript/figures/main/figure03", close=False)
    print(
        f"\nFigure 3 saved to manuscript/figures/main/figure03.{{pdf,png}} "
        f"(panel b pooled over {N_REALIZATIONS} realizations)"
    )


if __name__ == "__main__":
    # Full-size run (~32k 1-ms steps). To prototype quickly, shrink the
    # timeline by passing a smaller ``phase_boundaries`` tuple, e.g.:
    #   params = DecodeParams(
    #       phase_boundaries=(600, 900, 1100, 1400, 1600, 1900, 2100, 2400),
    #   )
    # drift_momentum=0.88 (vs the 0.8 default) makes the drift misfit a bit
    # larger/faster so it is more visible in the figure.
    params = DecodeParams(drift_momentum=0.88)
    run_demo(params)
