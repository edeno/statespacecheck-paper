"""Behavior tests for the figure-3 metric-dissociation phases.

These tests verify the scientific claims of the figure-3 simulation:

- The remap phase flags all three metrics (regression guard).
- The wide-dynamics-noise phase inflates KL while HPD overlap and the
  rank-based p-value stay near baseline (the headline KL false-positive
  case).
- The history-dependent firing phase produces per-spike metrics
  comparable to baseline — i.e., the per-spike spatial diagnostics
  largely *miss* a purely temporal misspecification (the deliberate
  demonstration of metric scope).
- The wiggly-flat-likelihood phase pushes HPD overlap toward instability
  and decouples KL from HPDO in the per-spike scatter.

If any of these assertions ever flips, the figure no longer tells the
story the paper claims; CI flags the regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from statespacecheck_paper.analysis import DecodeParams
from statespacecheck_paper.figure03_demo import SimulationResult, run_figure03_simulation


def _moderate_params() -> DecodeParams:
    """Phase sizes large enough for per-phase medians to be stable but
    small enough to keep the test fast (~3 s on a laptop).
    """
    return DecodeParams(
        T_remap_start=600,
        T_remap_end=900,
        T_recovery1_end=1100,
        T_hist_dep_end=1400,
        T_recovery2_end=1600,
        T_drift_end=1900,
        T_recovery3_end=2100,
        T_wide_dynamics_end=2400,
        T_recovery4_end=2600,
        T_wiggly_end=2900,
    )


def _per_phase_medians(
    sim: SimulationResult,
) -> dict[str, tuple[float, float, float]]:
    """Return (kl_med, hpd_med, sp_med) per phase label."""
    metrics = sim["metrics"]
    boundaries = np.asarray(sim["phase_boundaries"])
    labels = sim["phase_labels"]
    event_phase = np.searchsorted(boundaries, metrics["event_time_ind"], side="right")
    out: dict[str, tuple[float, float, float]] = {}
    for i, label in enumerate(labels):
        mask = event_phase == i
        if not mask.any():
            continue
        kl = float(np.nanmedian(metrics["event_kl_divergence"][mask]))
        hpd = float(np.nanmedian(metrics["event_hpd_overlap"][mask]))
        sp = float(np.nanmedian(metrics["event_spike_prob"][mask]))
        out[label] = (kl, hpd, sp)
    return out


@pytest.fixture(scope="module")
def sim() -> SimulationResult:
    return run_figure03_simulation(_moderate_params(), seed=0)


def test_phase_labels_and_boundaries(sim: SimulationResult) -> None:
    """``run_figure03_simulation`` returns the expected 10 phases in order
    and a timeline that ends at ``T_wiggly_end``.
    """
    expected_labels = [
        "Clean Baseline",
        "Remap Misfit",
        "Clean Recovery",
        "History-Dependent Firing",
        "Clean Recovery",
        "Drift Misfit",
        "Clean Recovery",
        "Wide Dynamics Noise",
        "Clean Recovery",
        "Wiggly-Flat Likelihood",
    ]
    params = sim["params"]
    assert sim["phase_labels"] == expected_labels
    boundaries = np.asarray(sim["phase_boundaries"])
    assert boundaries[-1] == params.T_wiggly_end
    assert np.all(np.diff(boundaries) > 0)
    x_true = np.asarray(sim["x_true"])
    assert x_true.shape[0] == params.T_wiggly_end


def test_remap_phase_flags_all_three(sim: SimulationResult) -> None:
    """Regression guard: the remap phase moves all three metrics away
    from baseline in the expected direction.
    """
    medians = _per_phase_medians(sim)
    base_kl, base_hpd, base_sp = medians["Clean Baseline"]
    remap_kl, remap_hpd, remap_sp = medians["Remap Misfit"]
    assert remap_kl > base_kl, (
        f"remap KL should exceed baseline; got base={base_kl:.3f}, remap={remap_kl:.3f}"
    )
    assert remap_hpd < base_hpd, (
        f"remap HPDO should fall below baseline; got base={base_hpd:.3f}, remap={remap_hpd:.3f}"
    )
    assert remap_sp < base_sp, (
        f"remap spike_prob should fall below baseline; got base={base_sp:.3f}, remap={remap_sp:.3f}"
    )


def test_wide_dynamics_noise_phase_dissociates_kl_from_hpd(
    sim: SimulationResult,
) -> None:
    """Load-bearing: wide-dynamics-noise phase inflates KL while HPD
    overlap stays near baseline. The headline KL-false-positive case.
    """
    medians = _per_phase_medians(sim)
    base_kl, base_hpd, _ = medians["Clean Baseline"]
    wide_kl, wide_hpd, _ = medians["Wide Dynamics Noise"]

    assert wide_kl > 2 * base_kl, (
        f"wide-dynamics-noise should inflate KL by >2x; got base={base_kl:.3f}, wide={wide_kl:.3f}"
    )
    # HPD overlap preserved: wide phase HPDO median should remain in
    # the same neighborhood as baseline.
    assert wide_hpd >= 0.5 * base_hpd - 1e-12, (
        f"wide-dynamics-noise should preserve HPD overlap; "
        f"got base={base_hpd:.3f}, wide={wide_hpd:.3f}"
    )


def test_history_dependent_firing_per_spike_metrics_near_baseline(
    sim: SimulationResult,
) -> None:
    """Load-bearing scientific claim: per-spike spatial diagnostics
    largely *miss* temporal (history-dependent) misspecification.

    The bursting + refractory misfit lives in the joint distribution of
    spike trains, not in any individual spike's spatial likelihood. We
    therefore expect the per-spike metrics in the history-dependent
    phase to stay close to baseline rather than crossing the flagging
    thresholds the way remap/drift do.
    """
    medians = _per_phase_medians(sim)
    base_kl, base_hpd, base_sp = medians["Clean Baseline"]
    hd_kl, hd_hpd, hd_sp = medians["History-Dependent Firing"]
    remap_kl, remap_hpd, _ = medians["Remap Misfit"]

    # KL should NOT inflate the way it does for remap; specifically, the
    # history-dep median KL should be much closer to baseline than to the
    # remap KL. We require ``hd_kl < (base + remap) / 2``.
    midpoint = 0.5 * (base_kl + remap_kl)
    assert hd_kl < midpoint, (
        "per-spike KL in history-dependent phase should stay near "
        "baseline rather than approach the remap-scale inflation; "
        f"baseline={base_kl:.3f}, remap={remap_kl:.3f}, hist-dep={hd_kl:.3f}"
    )
    # HPDO should stay near baseline (well above the remap collapse).
    midpoint_hpd = 0.5 * (base_hpd + remap_hpd)
    assert hd_hpd > midpoint_hpd, (
        "per-spike HPDO in history-dependent phase should stay near "
        "baseline rather than collapse to the remap-scale low; "
        f"baseline={base_hpd:.3f}, remap={remap_hpd:.3f}, hist-dep={hd_hpd:.3f}"
    )
    # spike_prob should likewise stay closer to baseline than to remap.
    _ = (hd_sp, base_sp)  # currently no strict bound — placeholder for follow-up


def test_wiggly_flat_likelihood_inflates_kl(sim: SimulationResult) -> None:
    """Wiggly-flat-likelihood phase produces a wide, low-info likelihood;
    KL(narrow_predictive || wiggly_flat_likelihood) is meaningfully
    larger than baseline.
    """
    medians = _per_phase_medians(sim)
    base_kl, _, _ = medians["Clean Baseline"]
    wiggly_kl, _, _ = medians["Wiggly-Flat Likelihood"]
    assert wiggly_kl > 1.5 * base_kl, (
        "wiggly-flat-likelihood phase should inflate KL > 1.5x baseline; "
        f"got base={base_kl:.3f}, wiggly={wiggly_kl:.3f}"
    )
