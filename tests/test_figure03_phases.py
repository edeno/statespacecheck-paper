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

If any of these assertions ever flips, the figure no longer tells the
story the paper claims; CI flags the regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from statespacecheck_paper.analysis import DecodeParams
from statespacecheck_paper.figure03_demo import (
    PHASE_LABELS,
    SimulationResult,
    run_figure03_simulation,
)


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
    """``run_figure03_simulation`` emits every canonical phase in order
    and a timeline that ends at ``T_wide_dynamics_end``.
    """
    params = sim["params"]
    # The simulation must emit exactly the canonical phase set, in order.
    assert sim["phase_labels"] == list(PHASE_LABELS)
    # Sanity-check the canonical set itself: 8 phases, the 4 expected
    # misfits each appearing once.
    assert len(PHASE_LABELS) == 8
    for misfit in (
        "Remap Misfit",
        "History-Dependent Firing",
        "Drift Misfit",
        "Wide Dynamics Noise",
    ):
        assert PHASE_LABELS.count(misfit) == 1
    boundaries = np.asarray(sim["phase_boundaries"])
    assert boundaries[-1] == params.T_wide_dynamics_end
    assert np.all(np.diff(boundaries) > 0)
    x_true = np.asarray(sim["x_true"])
    assert x_true.shape[0] == params.T_wide_dynamics_end


def test_remap_phase_flags_all_three(sim: SimulationResult) -> None:
    """Regression guard: the remap phase is a strong, unambiguous misfit —
    all three metrics move far from baseline, not merely in the right
    direction. Magnitude bounds (not bare inequalities) so a remap that
    barely perturbed the metrics would fail.
    """
    medians = _per_phase_medians(sim)
    base_kl, _, _ = medians["Clean Baseline"]
    remap_kl, remap_hpd, remap_sp = medians["Remap Misfit"]
    # KL inflates by at least 5x (observed ~30x at the test scale).
    assert remap_kl > 5 * base_kl, (
        f"remap KL should be >5x baseline; got base={base_kl:.3f}, remap={remap_kl:.3f}"
    )
    # HPD overlap collapses toward zero (observed ~0.0).
    assert remap_hpd < 0.5, f"remap HPDO should collapse below 0.5; got {remap_hpd:.3f}"
    # Rank-based p-value collapses toward zero (observed ~0.0).
    assert remap_sp < 0.2, f"remap spike_prob should collapse below 0.2; got {remap_sp:.3f}"


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
    # HPD overlap preserved: wide-phase HPDO must stay within 10% of
    # baseline. The dissociation claim is that HPDO barely moves while KL
    # inflates — a 50%-drop tolerance would not distinguish "preserved"
    # from "moderately degraded".
    assert wide_hpd >= 0.9 * base_hpd, (
        f"wide-dynamics-noise should preserve HPD overlap (>=0.9x baseline); "
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

    # All three per-spike metrics stay near baseline — the temporal
    # misfit barely registers. Bounds are absolute (vs. baseline), not
    # relative to the remap collapse: the claim is "near baseline", and
    # a midpoint-vs-remap bound would be trivially satisfied because the
    # remap inflation is ~30x.
    assert hd_kl < 3 * base_kl, (
        "per-spike KL in the history-dependent phase should stay within "
        f"3x baseline; got baseline={base_kl:.3f}, hist-dep={hd_kl:.3f}"
    )
    assert hd_hpd > 0.9 * base_hpd, (
        "per-spike HPDO in the history-dependent phase should stay within "
        f"10% of baseline; got baseline={base_hpd:.3f}, hist-dep={hd_hpd:.3f}"
    )
    # spike_prob stays in a band around baseline (neither collapsing like
    # remap nor spuriously inflating).
    assert 0.5 * base_sp < hd_sp < 1.5 * base_sp, (
        "per-spike spike_prob in the history-dependent phase should stay "
        f"within +/-50% of baseline; got baseline={base_sp:.3f}, "
        f"hist-dep={hd_sp:.3f}"
    )
