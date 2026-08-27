"""Tests for analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from statespacecheck_paper.analysis import (
    DecodeParams,
    PhaseBoundary,
    _flag_fraction,
    compute_phase_flag_fractions,
    extract_phase_flag_values,
    flag_fractions_from_values,
    get_remapped_pf_centers,
    summary_phase_windows,
)
from statespacecheck_paper.diagnostics import DecodingDiagnostics, DiagnosticThresholds
from statespacecheck_paper.figure03_demo import PHASE_LABELS, SimulationResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _zero_diagnostics(
    *, n_time: int, n_bins: int, n_cells: int = 1, n_spikes: int = 0
) -> DecodingDiagnostics:
    """Construct a well-shaped all-zero/empty ``DecodingDiagnostics`` for tests
    that only need a valid placeholder, not real diagnostic content."""
    posterior = np.full((n_time, n_bins), 1.0 / n_bins)
    return DecodingDiagnostics(
        posterior=posterior,
        predictive=posterior.copy(),
        likelihood=posterior.copy(),
        spike_likelihood=posterior.copy(),
        hpd_overlap=np.zeros((n_time, n_cells)),
        kl_divergence=np.zeros((n_time, n_cells)),
        predictive_pvalue=np.zeros((n_time, n_cells)),
        event_time_ind=np.zeros(n_spikes, dtype=np.intp),
        event_cell_ind=np.zeros(n_spikes, dtype=np.intp),
        event_hpd_overlap=np.zeros(n_spikes),
        event_kl_divergence=np.zeros(n_spikes),
        event_predictive_pvalue=np.zeros(n_spikes),
        per_spike_likelihood=np.zeros((n_spikes, n_bins)),
    )


# ---------------------------------------------------------------------------
# DecodeParams
# ---------------------------------------------------------------------------


class TestDecodeParams:
    def test_post_init_initializes_pf_centers_to_grid(self) -> None:
        params = DecodeParams()
        np.testing.assert_array_equal(params.pf_centers, np.arange(0, 101, 10, dtype=float))

    def test_post_init_respects_provided_pf_centers(self) -> None:
        custom = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        params = DecodeParams(pf_centers=custom)
        np.testing.assert_array_equal(params.pf_centers, custom)

    def test_phase_boundaries_indexed_by_enum(self) -> None:
        params = DecodeParams(
            phase_boundaries=(1000, 2000, 14_000, 18_000, 22_000, 26_000, 30_000, 32_000)
        )
        assert params.phase_boundaries[PhaseBoundary.REMAP_START] == 1000
        assert params.phase_boundaries[PhaseBoundary.REMAP_END] == 2000


# ---------------------------------------------------------------------------
# get_remapped_pf_centers
# ---------------------------------------------------------------------------


class TestGetRemappedPfCenters:
    def test_inactive_returns_input_unchanged_without_copy(self) -> None:
        """active=False is the hot path — must avoid the copy."""
        pf_centers = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        result = get_remapped_pf_centers(pf_centers, (0, 1), active=False)
        np.testing.assert_array_equal(result, pf_centers)
        assert result is pf_centers

    def test_active_returns_copy_and_does_not_mutate_input(self) -> None:
        pf_centers = np.array([0.0, 10.0, 20.0])
        result = get_remapped_pf_centers(pf_centers, (0, 1), active=True)
        assert result is not pf_centers
        np.testing.assert_array_equal(pf_centers, [0.0, 10.0, 20.0])

    def test_single_remapping_assigns_dst_center_to_src(self) -> None:
        pf_centers = np.array([0.0, 10.0, 20.0, 30.0])
        result = get_remapped_pf_centers(pf_centers, (2, 0), active=True)
        np.testing.assert_array_equal(result, [0.0, 10.0, 0.0, 30.0])

    def test_multiple_remappings_use_original_dst_values(self) -> None:
        """Bidirectional swap (0->1, 1->0) must use *original* values, not
        sequentially overwritten ones — otherwise both end up with the same
        center."""
        pf_centers = np.array([0.0, 10.0, 20.0, 30.0])
        result = get_remapped_pf_centers(pf_centers, ((0, 1), (1, 0)), active=True)
        np.testing.assert_array_equal(result, [10.0, 0.0, 20.0, 30.0])

    def test_default_global_remap_is_complete_and_well_separated(self) -> None:
        """Every default cell is remapped exactly once and moves at least
        three center spacings, as required by the Figure 3 positive control.
        """
        params = DecodeParams()
        assert params.pf_centers is not None
        mapping = np.asarray(params.remap_from_to, dtype=int)
        n_cells = params.pf_centers.size

        np.testing.assert_array_equal(np.sort(mapping[:, 0]), np.arange(n_cells))
        np.testing.assert_array_equal(np.sort(mapping[:, 1]), np.arange(n_cells))
        assert np.all(np.abs(mapping[:, 0] - mapping[:, 1]) >= 3)

        result = get_remapped_pf_centers(
            params.pf_centers,
            params.remap_from_to,
            active=True,
        )
        np.testing.assert_array_equal(result, params.pf_centers[mapping[:, 1]])


# ---------------------------------------------------------------------------
# Figure-3 summary heatmap helpers (phase windows + flag fractions)
# ---------------------------------------------------------------------------


class TestSummaryFlagFractions:
    """The summary-heatmap helpers are the single source of truth shared by
    the single-run renderer and the multi-realization averaging path; these
    tests pin the column layout and the flag-fraction arithmetic."""

    @staticmethod
    def _params() -> DecodeParams:
        # Tiny strictly-increasing ladder so windows map to known slices.
        return DecodeParams(phase_boundaries=(6, 10, 14, 18, 26, 30, 34, 36))

    def test_summary_phase_windows_structure(self) -> None:
        cols = summary_phase_windows(self._params())
        assert [c.label for c in cols] == [
            "Well-\nspecified",
            "Remap",
            "History-\ndep.",
            "Replay",
            "Drift",
            "Sparse\npopulation",
        ]
        assert [c.component for c in cols] == [
            "—",
            "Observation",
            "Observation",
            "—",
            "Transition",
            "—",
        ]
        # Well-specified concatenates the clean-recovery windows, with the
        # replay sub-window (20, 24) carved out of clean-recovery 2 (18, 26).
        assert cols[0].slices == ((10, 14), (18, 20), (24, 26), (30, 34))
        assert cols[1].slices == ((6, 10),)  # Remap
        assert cols[3].slices == ((20, 24),)  # Replay
        assert cols[5].slices == ((34, 36),)  # Sparse population

    def test_replay_window_rejects_fractions_that_round_to_empty(self) -> None:
        params = DecodeParams(replay_frac_start=0.25001, replay_frac_end=0.25002)
        with pytest.raises(ValueError, match="at least 3 steps"):
            summary_phase_windows(params)

    @pytest.mark.parametrize(
        ("direction", "expected"),
        [("below", 100.0 * 2 / 3), ("above", 100.0 * 2 / 3)],
    )
    def test_flag_fraction_directions(self, direction: str, expected: float) -> None:
        vals = np.array([0.0, 0.5, 1.0])
        # below: {0.0, 0.5} <= 0.5 ; above: {0.5, 1.0} >= 0.5 — both 2/3.
        assert _flag_fraction(vals, 0.5, direction) == pytest.approx(expected)

    def test_flag_fraction_empty_is_zero(self) -> None:
        assert _flag_fraction(np.array([]), 0.5, "below") == 0.0

    def test_flag_fraction_bad_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            _flag_fraction(np.array([1.0]), 0.5, "sideways")

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
        windows = summary_phase_windows(params)
        frac = compute_phase_flag_fractions(metrics, thresholds, windows)

        assert frac.shape == (3, 6)
        # KL row (index 2): only the remap column (index 1) flags.
        assert frac[2, 1] == pytest.approx(100.0)
        assert np.allclose(np.delete(frac[2], 1), 0.0)
        # HPD (0) and spike-prob (1) rows never cross their thresholds.
        assert np.allclose(frac[0], 0.0)
        assert np.allclose(frac[1], 0.0)

    def test_extract_drops_nan_and_matches_compute(self) -> None:
        """``extract_phase_flag_values`` strips NaNs, and the two-step
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
        windows = summary_phase_windows(params)

        values = extract_phase_flag_values(metrics, windows)
        # No NaNs survive extraction.
        for per_metric in values:
            for arr in per_metric:
                assert np.all(np.isfinite(arr))

        np.testing.assert_allclose(
            flag_fractions_from_values(values, thresholds),
            compute_phase_flag_fractions(metrics, thresholds, windows),
        )


# ---------------------------------------------------------------------------
# DecodeParams.phase_boundaries tuple
# ---------------------------------------------------------------------------


class TestDecodeParamsPhaseBoundaries:
    """The phase ladder collapsed from 8 ``T_*`` fields to one
    ``phase_boundaries`` tuple. Lock the invariants in.
    """

    def test_default_boundaries_match_documented_defaults(self) -> None:
        params = DecodeParams()
        assert params.phase_boundaries == (
            6_000,
            10_000,
            14_000,
            18_000,
            22_000,
            26_000,
            30_000,
            32_000,
        )

    def test_phase_boundaries_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="must have 8 entries"):
            DecodeParams(phase_boundaries=(1, 2, 3))

    def test_phase_boundaries_non_monotonic_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            DecodeParams(
                phase_boundaries=(100, 200, 200, 400, 500, 600, 700, 800),
            )

    def test_phase_boundaries_equal_consecutive_raises(self) -> None:
        """Equal consecutive entries (zero-width phase) must reject too."""
        with pytest.raises(ValueError, match="strictly increasing"):
            DecodeParams(
                phase_boundaries=(100, 200, 300, 300, 500, 600, 700, 800),
            )

    @pytest.mark.parametrize(
        ("member", "index"),
        [
            (PhaseBoundary.REMAP_START, 0),
            (PhaseBoundary.REMAP_END, 1),
            (PhaseBoundary.RECOVERY1_END, 2),
            (PhaseBoundary.HIST_DEP_END, 3),
            (PhaseBoundary.RECOVERY2_END, 4),
            (PhaseBoundary.DRIFT_END, 5),
            (PhaseBoundary.RECOVERY3_END, 6),
            (PhaseBoundary.SPARSE_POP_END, 7),
        ],
    )
    def test_phase_boundary_enum_indexes_into_tuple(
        self, member: PhaseBoundary, index: int
    ) -> None:
        boundaries = (100, 200, 300, 400, 500, 600, 700, 800)
        params = DecodeParams(phase_boundaries=boundaries)
        assert params.phase_boundaries[member] == boundaries[index]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"sparse_position": -1.0}, "sparse_position"),
            ({"sparse_approach_steps": -1}, "sparse_approach_steps"),
            ({"sparse_ensemble_rate_scale": 1.1}, "sparse_ensemble_rate_scale"),
            ({"n_sparse_cells": 0}, "n_sparse_cells"),
            ({"sparse_field_spread": -1.0}, "sparse_field_spread"),
            ({"sparse_field_spread": np.nan}, "sparse_field_spread"),
            ({"sparse_cell_width": 0.0}, "sparse_cell_width"),
            ({"sparse_cell_width": np.nan}, "sparse_cell_width"),
            ({"sparse_cell_peak_rate": 0.0}, "sparse_cell_peak_rate"),
            ({"sparse_cell_baseline_gain": -0.1}, "sparse_cell_baseline_gain"),
        ],
    )
    def test_sparse_population_parameters_reject_invalid_values(
        self,
        kwargs: dict[str, float | int],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            DecodeParams(**kwargs)


class TestSimulationResultDataclass:
    """The ``TypedDict`` → frozen-dataclass conversion brought length
    validation. Cover the success contract and the failure modes."""

    def test_valid_construction_succeeds(self) -> None:
        """Happy path: a well-formed SimulationResult constructs cleanly,
        coerces list inputs to tuple, and exposes attribute access on
        every field."""

        n_bins = 5
        n_time = 10
        sim = SimulationResult(
            params=DecodeParams(),
            xs=np.linspace(0.0, 100.0, n_bins),
            x_true=np.zeros(n_time),
            spikes=np.zeros((n_time, 1), dtype=np.int_),
            metrics=_zero_diagnostics(n_time=n_time, n_bins=n_bins),
            phase_labels=PHASE_LABELS,
            phase_boundaries=(1, 2, 3, 4, 5, 6, 7, n_time),
        )
        # Sequence fields coerced to tuple by __post_init__.
        assert isinstance(sim.phase_labels, tuple)
        assert isinstance(sim.phase_boundaries, tuple)
        # Attribute access works (the migration test).
        assert sim.xs.shape == (n_bins,)
        assert sim.x_true.shape == (n_time,)

    def test_phase_labels_wrong_order_raises(self) -> None:
        n_bins = 5
        n_time = 10
        bogus_labels = tuple(reversed(PHASE_LABELS))
        with pytest.raises(ValueError, match="phase_labels must equal PHASE_LABELS"):
            SimulationResult(
                params=DecodeParams(),
                xs=np.linspace(0.0, 100.0, n_bins),
                x_true=np.zeros(n_time),
                spikes=np.zeros((n_time, 1), dtype=np.int_),
                metrics=_zero_diagnostics(n_time=n_time, n_bins=n_bins),
                phase_labels=bogus_labels,
                phase_boundaries=(1, 2, 3, 4, 5, 6, 7, n_time),
            )

    def test_phase_boundary_length_mismatch_raises(self) -> None:
        n_bins = 5
        n_time = 10
        with pytest.raises(ValueError, match="phase_boundaries length"):
            SimulationResult(
                params=DecodeParams(),
                xs=np.linspace(0.0, 100.0, n_bins),
                x_true=np.zeros(n_time),
                spikes=np.zeros((n_time, 1), dtype=np.int_),
                metrics=_zero_diagnostics(n_time=n_time, n_bins=n_bins),
                phase_labels=PHASE_LABELS,
                phase_boundaries=(1, 2, 3),  # wrong length
            )

    def test_spikes_and_x_true_timeline_mismatch_raises(self) -> None:
        n_bins = 5
        n_time = 10
        with pytest.raises(ValueError, match="spikes timeline"):
            SimulationResult(
                params=DecodeParams(),
                xs=np.linspace(0.0, 100.0, n_bins),
                x_true=np.zeros(n_time),
                spikes=np.zeros((n_time + 1, 1), dtype=np.int_),  # off by one
                metrics=_zero_diagnostics(n_time=n_time, n_bins=n_bins),
                phase_labels=PHASE_LABELS,
                phase_boundaries=(1, 2, 3, 4, 5, 6, 7, n_time),
            )

    def test_final_boundary_must_equal_timeline_length(self) -> None:
        n_bins = 5
        n_time = 10
        with pytest.raises(ValueError, match="final phase boundary"):
            SimulationResult(
                params=DecodeParams(),
                xs=np.linspace(0.0, 100.0, n_bins),
                x_true=np.zeros(n_time),
                spikes=np.zeros((n_time, 1), dtype=np.int_),
                metrics=_zero_diagnostics(n_time=n_time, n_bins=n_bins),
                phase_labels=PHASE_LABELS,
                phase_boundaries=(1, 2, 3, 4, 5, 6, 7, n_time + 1),
            )

    def test_metrics_timeline_mismatch_against_x_true_raises(self) -> None:
        """``SimulationResult.__post_init__`` rejects a ``DecodingDiagnostics``
        whose ``posterior`` timeline doesn't match ``x_true``. The
        ``DecodingDiagnostics`` dataclass itself enforces internal-shape
        consistency; this test pins the cross-check against the outer
        timeline that only ``SimulationResult`` can see."""
        n_bins = 5
        n_time = 10
        # A perfectly-shaped DecodingDiagnostics with the wrong leading dim.
        bad_metrics = _zero_diagnostics(n_time=n_time + 1, n_bins=n_bins)
        with pytest.raises(ValueError, match=r"metrics.posterior leading dim"):
            SimulationResult(
                params=DecodeParams(),
                xs=np.linspace(0.0, 100.0, n_bins),
                x_true=np.zeros(n_time),
                spikes=np.zeros((n_time, 1), dtype=np.int_),
                metrics=bad_metrics,
                phase_labels=PHASE_LABELS,
                phase_boundaries=(1, 2, 3, 4, 5, 6, 7, n_time),
            )
