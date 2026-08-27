"""Tests for the Figure-3b summary (per-condition flag percentages)."""

from __future__ import annotations

import numpy as np
import pytest

from statespacecheck_paper.diagnostics import DiagnosticThresholds
from statespacecheck_paper.figure03_protocol import Figure3Config, PhaseBoundary
from statespacecheck_paper.figure03_summary import (
    _flag_percentage,
    build_summary_conditions,
    compute_condition_flag_percentages,
    extract_condition_flag_values,
    flag_percentages_from_values,
)


class TestSummaryFlagPercentages:
    """The summary-heatmap helpers are the single source of truth shared by
    the single-run renderer and the multi-realization averaging path; these
    tests pin the column layout and the flag-fraction arithmetic."""

    @staticmethod
    def _params() -> Figure3Config:
        # Tiny strictly-increasing ladder so conditions map to known slices.
        return Figure3Config(phase_boundaries=(6, 10, 14, 18, 26, 30, 34, 36))

    def test_summary_phase_windows_structure(self) -> None:
        cols = build_summary_conditions(self._params())
        assert [c.label for c in cols] == [
            "Well-\nspecified",
            "Remap",
            "History-\ndep.",
            "Replay",
            "Drift",
            "Sparse\npopulation",
        ]
        assert [c.model_component for c in cols] == [
            "—",
            "Observation",
            "Observation",
            "—",
            "Transition",
            "—",
        ]
        # Well-specified concatenates the clean-recovery conditions, with the
        # replay sub-window (20, 24) carved out of clean-recovery 2 (18, 26).
        assert cols[0].step_windows == ((10, 14), (18, 20), (24, 26), (30, 34))
        assert cols[1].step_windows == ((6, 10),)  # Remap
        assert cols[3].step_windows == ((20, 24),)  # Replay
        assert cols[5].step_windows == ((34, 36),)  # Sparse population

    def test_replay_window_rejects_fractions_that_round_to_empty(self) -> None:
        params = Figure3Config(replay_start_fraction=0.25001, replay_end_fraction=0.25002)
        with pytest.raises(ValueError, match="at least 3 steps"):
            build_summary_conditions(params)

    @pytest.mark.parametrize(
        ("direction", "expected"),
        [("below", 100.0 * 2 / 3), ("above", 100.0 * 2 / 3)],
    )
    def test_flag_fraction_directions(self, direction: str, expected: float) -> None:
        vals = np.array([0.0, 0.5, 1.0])
        # below: {0.0, 0.5} <= 0.5 ; above: {0.5, 1.0} >= 0.5 — both 2/3.
        assert _flag_percentage(vals, 0.5, direction) == pytest.approx(expected)

    def test_flag_fraction_empty_is_zero(self) -> None:
        assert _flag_percentage(np.array([]), 0.5, "below") == 0.0

    def test_flag_fraction_bad_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            _flag_percentage(np.array([1.0]), 0.5, "sideways")

    def test_compute_phase_flag_fractions_isolates_remap(self) -> None:
        """A KL spike confined to the remap window must flag 100% in the
        remap column and 0% elsewhere; HPD/spike-prob rows that never cross
        their thresholds must be 0% everywhere.

        Row order follows ``SUMMARY_FLAG_METRICS``: HPD (0), spike-prob (1),
        KL (2). Column order: well-specified (0), remap (1), history (2),
        replay (3), drift (4), sparse population (5)."""
        params = self._params()
        n_time = params.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
        # One spike event per time step; KL high only inside remap [6, 10).
        event_time = np.arange(n_time, dtype=np.intp)
        event_kl = np.zeros(n_time)
        event_kl[6:10] = 10.0
        metrics: dict[str, np.ndarray] = {
            "event_time_ind": event_time,
            "event_hpd_overlap": np.ones(n_time),  # never below 0.5
            "event_kl_divergence": event_kl,
            "event_predictive_pvalue": np.ones(n_time),  # never below 0.05
        }
        thresholds = DiagnosticThresholds(
            hpd_overlap=0.5, kl_divergence=5.0, predictive_pvalue=0.05
        )
        conditions = build_summary_conditions(params)
        frac = compute_condition_flag_percentages(metrics, thresholds, conditions)

        assert frac.shape == (3, 6)
        # KL row (index 2): only the remap column (index 1) flags.
        assert frac[2, 1] == pytest.approx(100.0)
        assert np.allclose(np.delete(frac[2], 1), 0.0)
        # HPD (0) and spike-prob (1) rows never cross their thresholds.
        assert np.allclose(frac[0], 0.0)
        assert np.allclose(frac[1], 0.0)

    def test_extract_drops_nan_and_matches_compute(self) -> None:
        """``extract_condition_flag_values`` strips NaNs, and the two-step
        extract→flag path agrees with the one-shot wrapper."""
        params = self._params()
        n_time = params.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
        rng = np.random.default_rng(0)
        n_events = 2 * n_time
        event_time = rng.integers(0, n_time, n_events).astype(np.intp)
        kl = rng.uniform(0.0, 10.0, n_events)
        kl[::3] = np.nan  # defensive: NaN per-event values must be stripped
        metrics: dict[str, np.ndarray] = {
            "event_time_ind": event_time,
            "event_hpd_overlap": rng.uniform(0.0, 1.0, n_events),
            "event_kl_divergence": kl,
            "event_predictive_pvalue": rng.uniform(0.0, 1.0, n_events),
        }
        thresholds = DiagnosticThresholds(hpd_overlap=0.3, kl_divergence=5.0, predictive_pvalue=0.2)
        conditions = build_summary_conditions(params)

        values = extract_condition_flag_values(metrics, conditions)
        # No NaNs survive extraction.
        for per_metric in values:
            for arr in per_metric:
                assert np.all(np.isfinite(arr))

        np.testing.assert_allclose(
            flag_percentages_from_values(values, thresholds),
            compute_condition_flag_percentages(metrics, thresholds, conditions),
        )
