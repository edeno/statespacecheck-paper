"""Tests for the Figure-3b summary (per-condition flag percentages)."""

from __future__ import annotations

import numpy as np
import pytest

from statespacecheck_paper.diagnostics import DiagnosticThresholds
from statespacecheck_paper.figure03_protocol import Figure3Config, PhaseBoundary
from statespacecheck_paper.figure03_summary import (
    CONDITION_IDS,
    SUMMARY_ACCURACY_METRICS,
    _flag_percentage,
    build_summary_conditions,
    compute_condition_decoding_accuracy,
    compute_condition_flag_percentages,
    extract_condition_flag_values,
)


class TestSummaryFlagPercentages:
    """The summary-heatmap helpers are the single source of truth shared by
    the single-run renderer and the multi-realization averaging path; these
    tests pin the column layout and the flag-fraction arithmetic."""

    @staticmethod
    def _params() -> Figure3Config:
        # Tiny strictly-increasing ladder so conditions map to known slices.
        return Figure3Config(phase_boundaries=(6, 10, 14, 18, 26, 30, 34, 38, 42, 46))

    def test_summary_phase_windows_structure(self) -> None:
        cols = build_summary_conditions(self._params())
        assert [c.condition_id for c in cols] == list(CONDITION_IDS)
        assert [c.label for c in cols] == [
            "Matched\nnull",
            "Recovery",
            "Remap",
            "History-\ndep.",
            "Replay",
            "Drift",
            "Reflected\nmap",
            "Sparse\npopulation",
        ]
        assert [c.model_component for c in cols] == [
            "—",
            "—",
            "Observation",
            "Observation",
            "—",
            "Transition",
            "Observation",
            "—",
        ]
        # The matched null is the opening baseline; recovery concatenates the
        # clean-recovery windows, with the replay sub-window (20, 24) carved
        # out of clean-recovery 2 (18, 26).
        assert cols[0].step_windows == ((0, 6),)
        assert cols[1].step_windows == ((10, 14), (18, 20), (24, 26), (30, 34), (38, 42))
        assert cols[2].step_windows == ((6, 10),)  # Remap
        assert cols[4].step_windows == ((20, 24),)  # Replay
        assert cols[6].step_windows == ((34, 38),)  # Reflected map
        assert cols[7].step_windows == ((42, 46),)  # Sparse population

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

    def test_flag_fraction_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="no spike events"):
            _flag_percentage(np.array([]), 0.5, "below")
        assert np.isnan(_flag_percentage(np.array([]), 0.5, "below", empty_value=np.nan))

    def test_flag_fraction_bad_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            _flag_percentage(np.array([1.0]), 0.5, "sideways")

    def test_compute_phase_flag_fractions_isolates_remap(self) -> None:
        """A KL spike confined to the remap window must flag 100% in the
        remap column and 0% elsewhere; HPD/spike-prob rows that never cross
        their thresholds must be 0% everywhere.

        Row order follows ``SUMMARY_FLAG_METRICS``: HPD (0), spike-prob (1),
        KL (2). Column order follows ``CONDITION_IDS`` (remap is column 2)."""
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

        assert frac.shape == (3, 8)
        # KL row (index 2): only the remap column flags.
        remap = CONDITION_IDS.index("remap")
        assert frac[2, remap] == pytest.approx(100.0)
        assert np.allclose(np.delete(frac[2], remap), 0.0)
        # HPD (0) and spike-prob (1) rows never cross their thresholds.
        assert np.allclose(frac[0], 0.0)
        assert np.allclose(frac[1], 0.0)

    def test_extract_rejects_nan_event_value(self) -> None:
        """Every event must remain represented in condition summaries."""
        params = self._params()
        n_time = params.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
        rng = np.random.default_rng(0)
        n_events = 2 * n_time
        event_time = rng.integers(0, n_time, n_events).astype(np.intp)
        kl = rng.uniform(0.0, 10.0, n_events)
        kl[::3] = np.nan
        metrics: dict[str, np.ndarray] = {
            "event_time_ind": event_time,
            "event_hpd_overlap": rng.uniform(0.0, 1.0, n_events),
            "event_kl_divergence": kl,
            "event_predictive_pvalue": rng.uniform(0.0, 1.0, n_events),
        }
        conditions = build_summary_conditions(params)

        with pytest.raises(ValueError, match="undefined value"):
            extract_condition_flag_values(metrics, conditions)


class TestConditionDecodingAccuracy:
    """Per-phase decoding accuracy of the filtered posterior against the
    stored true position: median absolute error of the posterior mean (row
    0, position units). Columns follow ``build_summary_conditions``."""

    @staticmethod
    def _params() -> Figure3Config:
        return Figure3Config(phase_boundaries=(6, 10, 14, 18, 26, 30, 34, 38, 42, 46))

    @staticmethod
    def _delta_posterior(bin_index: np.ndarray, n_bins: int) -> np.ndarray:
        posterior = np.zeros((bin_index.shape[0], n_bins))
        posterior[np.arange(bin_index.shape[0]), bin_index] = 1.0
        return posterior

    def test_metric_order(self) -> None:
        assert SUMMARY_ACCURACY_METRICS == (
            "median_absolute_error",
            "filtered_hpd_coverage_percent",
            "median_filtered_hpd_size",
            "median_predictive_hpd_size",
        )

    def test_perfect_decoder_has_zero_error_full_coverage_and_unit_regions(self) -> None:
        params = self._params()
        n_time = params.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
        position_bins = np.arange(10, dtype=float)
        true_bin = np.arange(n_time) % 10
        posterior = self._delta_posterior(true_bin, position_bins.size)
        conditions = build_summary_conditions(params)

        accuracy = compute_condition_decoding_accuracy(
            posterior,
            posterior,
            position_bins,
            position_bins[true_bin],
            conditions,
            hpd_coverage=0.95,
        )

        assert accuracy.shape == (4, 8)
        assert np.allclose(accuracy[0], 0.0)
        assert np.allclose(accuracy[1], 100.0)  # every true bin inside its point-mass region
        assert np.allclose(accuracy[2], 1.0) and np.allclose(accuracy[3], 1.0)

    def test_coverage_and_region_size_follow_the_hpd_region(self) -> None:
        """A two-bin posterior that never contains the truth has 0% coverage
        and size 2; a broad predictive reports its own (larger) size."""
        params = self._params()
        n_time = params.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
        position_bins = np.arange(10, dtype=float)
        posterior = np.zeros((n_time, 10))
        posterior[:, 3] = 0.5
        posterior[:, 4] = 0.5
        predictive = np.full((n_time, 10), 0.1)
        true_position = np.full(n_time, 8.0)
        conditions = build_summary_conditions(params)
        accuracy = compute_condition_decoding_accuracy(
            posterior, predictive, position_bins, true_position, conditions, hpd_coverage=0.95
        )
        assert np.allclose(accuracy[1], 0.0)
        assert np.allclose(accuracy[2], 2.0)
        assert np.allclose(accuracy[3], 10.0)
        # Off-grid truth maps to its nearest cell for the coverage check.
        accuracy = compute_condition_decoding_accuracy(
            posterior,
            predictive,
            position_bins,
            np.full(n_time, 3.4),
            conditions,
            hpd_coverage=0.95,
        )
        assert np.allclose(accuracy[1], 100.0)

    def test_shift_confined_to_remap_window(self) -> None:
        """A posterior displaced by two bins only inside remap [6, 10) must
        give a two-bin error in the remap column and zero error elsewhere."""
        params = self._params()
        n_time = params.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
        position_bins = np.arange(0.0, 20.0, 2.0)  # bin width 2 a.u.
        true_bin = np.arange(n_time) % 5
        decoded_bin = true_bin.copy()
        decoded_bin[6:10] += 2
        posterior = self._delta_posterior(decoded_bin, position_bins.size)
        conditions = build_summary_conditions(params)

        accuracy = compute_condition_decoding_accuracy(
            posterior,
            posterior,
            position_bins,
            position_bins[true_bin],
            conditions,
            hpd_coverage=0.95,
        )

        remap = CONDITION_IDS.index("remap")
        assert accuracy[0, remap] == pytest.approx(4.0)  # two bins of 2 a.u.
        assert np.allclose(np.delete(accuracy[0], remap), 0.0)

    def test_error_uses_continuous_true_position(self) -> None:
        """The error is measured against the continuous position, not its bin."""
        params = self._params()
        n_time = params.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
        position_bins = np.arange(10, dtype=float)
        posterior = self._delta_posterior(np.full(n_time, 4), position_bins.size)
        true_position = np.full(n_time, 4.3)
        conditions = build_summary_conditions(params)

        accuracy = compute_condition_decoding_accuracy(
            posterior, posterior, position_bins, true_position, conditions, hpd_coverage=0.95
        )

        assert np.allclose(accuracy[0], 0.3)

    def test_shape_mismatch_raises(self) -> None:
        params = self._params()
        n_time = params.phase_boundaries[PhaseBoundary.SPARSE_POP_END]
        position_bins = np.arange(10, dtype=float)
        posterior = self._delta_posterior(np.zeros(n_time, dtype=int), position_bins.size)
        conditions = build_summary_conditions(params)
        with pytest.raises(ValueError, match="true_position"):
            compute_condition_decoding_accuracy(
                posterior,
                posterior,
                position_bins,
                np.zeros(n_time - 1),
                conditions,
                hpd_coverage=0.95,
            )
        with pytest.raises(ValueError, match="position_bins"):
            compute_condition_decoding_accuracy(
                posterior,
                posterior,
                position_bins[:-1],
                np.zeros(n_time),
                conditions,
                hpd_coverage=0.95,
            )
        with pytest.raises(ValueError, match="predictive"):
            compute_condition_decoding_accuracy(
                posterior,
                posterior[:-1],
                position_bins,
                np.zeros(n_time),
                conditions,
                hpd_coverage=0.95,
            )
