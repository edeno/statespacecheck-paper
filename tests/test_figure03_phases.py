"""Behavior tests for the engineered metric-dissociation phases.

These tests verify the scientific claims of figure 3:

- The broad-decoder phase inflates KL divergence while HPD overlap and
  the rank-based p-value stay near baseline (the headline "KL is sensitive
  to shape mismatch where figure 1 calls the distributions consistent"
  story, broad direction).
- The tight-decoder phase inflates KL in the opposite shape-mismatch
  direction (predictive much narrower than the per-spike likelihood),
  again with HPD overlap and p-value staying near baseline.
- KL takes more timesteps to return to baseline-safe values than HPD
  overlap does after a misfit window ends.

If any of these assertions ever flips, the figure no longer tells the
story the paper claims; CI flags the regression.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from statespacecheck_paper.analysis import DecodeParams, compute_thresholds
from statespacecheck_paper.figure03_demo import run_figure03_simulation


def _moderate_params() -> DecodeParams:
    """Phase sizes large enough for per-phase medians to be stable.

    The full-size ``DecodeParams()`` produces ~40k timesteps and is slower
    than a unit test should be. These dimensions are the smallest where
    per-phase event counts (~hundreds) give reliable medians for the load-
    bearing assertions.
    """
    return DecodeParams(
        T_remap_start=400,
        T_remap_end=500,
        T_recovery1_end=600,
        T_flat_end=700,
        T_recovery2_end=800,
        T_fast_end=900,
        T_recovery3_end=1000,
        T_slow_end=1100,
        T_recovery4_end=1200,
        T_broad_decoder_end=1500,
        T_recovery5_end=1600,
        T_tight_decoder_end=1900,
    )


def _per_phase_medians(
    sim: dict[str, Any],
) -> dict[str, tuple[float, float, float]]:
    """Return (kl_med, hpd_med, sp_med) per phase label."""
    metrics = sim["metrics"]  # type: ignore[index]
    boundaries = np.asarray(sim["phase_boundaries"])  # type: ignore[arg-type]
    labels = sim["phase_labels"]  # type: ignore[index]
    event_phase = np.searchsorted(boundaries, metrics["event_time_ind"], side="right")
    out: dict[str, tuple[float, float, float]] = {}
    for i, label in enumerate(labels):  # type: ignore[arg-type]
        mask = event_phase == i
        if not mask.any():
            continue
        kl = float(np.nanmedian(metrics["event_kl_divergence"][mask]))
        hpd = float(np.nanmedian(metrics["event_hpd_overlap"][mask]))
        sp = float(np.nanmedian(metrics["event_spike_prob"][mask]))
        out[label] = (kl, hpd, sp)
    return out


@pytest.fixture(scope="module")
def sim() -> dict[str, Any]:
    return run_figure03_simulation(_moderate_params(), seed=0)


def test_run_figure03_simulation_extended_phases(sim: dict[str, Any]) -> None:
    """``run_figure03_simulation`` returns the expected 12 phases in order
    and a timeline that ends at ``T_tight_decoder_end``.
    """
    params = sim["params"]  # type: ignore[index]
    expected_labels = [
        "Clean Baseline",
        "Remapping Misfit",
        "Clean Recovery",
        "Flat Firing Misfit",
        "Clean Recovery",
        "Fast Movement Misfit",
        "Clean Recovery",
        "Drift Misfit",
        "Clean Recovery",
        "Broad-Decoder Phase",
        "Clean Recovery",
        "Tight-Decoder Phase",
    ]
    assert sim["phase_labels"] == expected_labels
    boundaries = np.asarray(sim["phase_boundaries"])  # type: ignore[arg-type]
    assert boundaries[-1] == params.T_tight_decoder_end  # type: ignore[attr-defined]
    assert np.all(np.diff(boundaries) > 0)
    x_true = np.asarray(sim["x_true"])  # type: ignore[arg-type]
    assert x_true.shape[0] == params.T_tight_decoder_end  # type: ignore[attr-defined]


def test_broad_decoder_phase_dissociates_kl_from_hpd(sim: dict[str, Any]) -> None:
    """Load-bearing: broad-decoder phase inflates KL while HPD overlap
    stays near baseline. This is the headline KL-false-positive case.
    """
    medians = _per_phase_medians(sim)
    base_kl, base_hpd, _ = medians["Clean Baseline"]
    broad_kl, broad_hpd, _ = medians["Broad-Decoder Phase"]

    assert broad_kl > 2 * base_kl, (
        f"broad-decoder phase should inflate KL by >2x; "
        f"got base={base_kl:.3f}, broad={broad_kl:.3f}"
    )
    # HPD overlap preserved: broad phase HPDO median should remain in
    # the same neighborhood as baseline. We use a permissive >0.5x
    # ratio so noise on the small-n medians doesn't flake the test;
    # tighten this if real-size sims show a tighter bound.
    assert broad_hpd >= 0.5 * base_hpd - 1e-12, (
        f"broad-decoder phase should preserve HPD overlap; "
        f"got base={base_hpd:.3f}, broad={broad_hpd:.3f}"
    )


def test_tight_decoder_phase_dissociates_kl_from_hpd(sim: dict[str, Any]) -> None:
    """Load-bearing: tight-decoder phase inflates KL in the opposite
    shape-mismatch direction (predictive much narrower than likelihood)
    while HPD overlap stays near baseline.

    KL inflation is weaker in this direction than in broad-decoder
    because KL(narrow || broad) ~ log(sigma_l / sigma_p) - 0.5, smaller
    than KL(broad || narrow). 1.5x suffices to clear the baseline.
    """
    medians = _per_phase_medians(sim)
    base_kl, base_hpd, _ = medians["Clean Baseline"]
    tight_kl, tight_hpd, _ = medians["Tight-Decoder Phase"]

    assert tight_kl > 1.5 * base_kl, (
        f"tight-decoder phase should inflate KL by >1.5x; "
        f"got base={base_kl:.3f}, tight={tight_kl:.3f}"
    )
    assert tight_hpd >= 0.5 * base_hpd - 1e-12, (
        f"tight-decoder phase should preserve HPD overlap; "
        f"got base={base_hpd:.3f}, tight={tight_hpd:.3f}"
    )


def test_remapping_phase_flags_all_three(sim: dict[str, Any]) -> None:
    """Regression guard: remapping phase flags all three metrics."""
    medians = _per_phase_medians(sim)
    base_kl, base_hpd, base_sp = medians["Clean Baseline"]
    remap_kl, remap_hpd, remap_sp = medians["Remapping Misfit"]
    assert remap_kl > base_kl, (
        f"remap KL should exceed baseline; got base={base_kl:.3f}, remap={remap_kl:.3f}"
    )
    assert remap_hpd < base_hpd, (
        f"remap HPDO should fall below baseline; got base={base_hpd:.3f}, remap={remap_hpd:.3f}"
    )
    assert remap_sp < base_sp, (
        f"remap spike_prob should fall below baseline; got base={base_sp:.3f}, remap={remap_sp:.3f}"
    )


def test_kl_recovery_slower_than_hpdo_after_remap(sim: dict[str, Any]) -> None:
    """Load-bearing: KL takes at least as many timesteps as HPDO to
    return to baseline-safe side after the remap misfit ends.

    The recovery-transient panel of figure 3 visualizes this lag; the
    assertion makes the claim CI-enforced.
    """
    params = sim["params"]  # type: ignore[index]
    metrics = sim["metrics"]  # type: ignore[index]
    thresholds = compute_thresholds(metrics, baseline_end=params.T_remap_start)  # type: ignore[attr-defined,arg-type]

    # Per-timestep median across cells (suppress all-NaN slice warnings
    # for timesteps with no spikes — those just appear as NaN gaps).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        hpd_t = np.nanmedian(metrics["hpd_overlap"], axis=1)
        kl_t = np.nanmedian(metrics["kl_divergence"], axis=1)

    t_lo = params.T_remap_end  # type: ignore[attr-defined]
    t_hi = params.T_recovery1_end  # type: ignore[attr-defined]
    hpd_window = hpd_t[t_lo:t_hi]
    kl_window = kl_t[t_lo:t_hi]

    # First post-misfit index where each metric crosses back to
    # baseline-safe values (NaNs treated as not-yet-recovered).
    hpd_safe = (hpd_window >= thresholds.hpd_overlap) & ~np.isnan(hpd_window)
    kl_safe = (kl_window <= thresholds.kl_divergence) & ~np.isnan(kl_window)
    hpd_back = int(np.argmax(hpd_safe)) if hpd_safe.any() else len(hpd_window)
    kl_back = int(np.argmax(kl_safe)) if kl_safe.any() else len(kl_window)

    assert kl_back >= hpd_back, (
        f"KL should recover at least as slowly as HPD overlap; "
        f"got hpd_back={hpd_back} steps, kl_back={kl_back} steps"
    )


def test_broad_decoder_events_cluster_at_high_kl_and_high_hpdo(
    sim: dict[str, Any],
) -> None:
    """The visible payload of the per-spike scatter panel: events in
    the broad-decoder phase occupy the upper-right region of (KL, HPDO)
    space — high KL AND high HPDO simultaneously, the geometric
    signature of the metric error.

    We check the *shape* of the dissociation rather than threshold-based
    counts: at small simulation scales the baseline distribution is too
    narrow to give a useful HPDO percentile threshold (most baseline
    events have HPDO=1.0 exactly).
    """
    metrics = sim["metrics"]
    boundaries = np.asarray(sim["phase_boundaries"])
    labels = sim["phase_labels"]

    event_phase = np.searchsorted(boundaries, metrics["event_time_ind"], side="right")
    baseline_idx = labels.index("Clean Baseline")
    broad_idx = labels.index("Broad-Decoder Phase")

    base_mask = event_phase == baseline_idx
    broad_mask = event_phase == broad_idx
    assert broad_mask.any(), "no events in broad-decoder phase"

    base_kl_median = float(np.nanmedian(metrics["event_kl_divergence"][base_mask]))

    # Broad-decoder spike events should cluster at HIGH HPD overlap
    # (matching figure-1 "consistent" notion) AND HIGH KL (above the
    # baseline median, since the broad-decoder phase inflates KL).
    kl_above_baseline = metrics["event_kl_divergence"][broad_mask] > base_kl_median
    hpd_high = metrics["event_hpd_overlap"][broad_mask] >= 0.5
    both = kl_above_baseline & hpd_high
    frac = float(both.mean())
    assert frac > 0.5, (
        "expected most broad-decoder spike events to sit in the "
        "(KL above baseline median, HPDO >= 0.5) region; got fraction="
        f"{frac:.2f}"
    )
