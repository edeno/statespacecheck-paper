"""Tests for the Figure-3 simulation (remap, result dataclass)."""

from __future__ import annotations

import numpy as np
import pytest

from statespacecheck_paper.diagnostics import DecodingDiagnostics
from statespacecheck_paper.figure03_protocol import PHASE_LABELS, Figure3Config
from statespacecheck_paper.figure03_simulation import (
    Figure3SimulationResult,
    remap_place_field_centers,
)


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


class TestRemapPlaceFieldCenters:
    def test_inactive_returns_input_unchanged_without_copy(self) -> None:
        """active=False is the hot path — must avoid the copy."""
        place_field_centers = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        result = remap_place_field_centers(place_field_centers, (0, 1), active=False)
        np.testing.assert_array_equal(result, place_field_centers)
        assert result is place_field_centers

    def test_active_returns_copy_and_does_not_mutate_input(self) -> None:
        place_field_centers = np.array([0.0, 10.0, 20.0])
        result = remap_place_field_centers(place_field_centers, (0, 1), active=True)
        assert result is not place_field_centers
        np.testing.assert_array_equal(place_field_centers, [0.0, 10.0, 20.0])

    def test_single_remapping_assigns_dst_center_to_src(self) -> None:
        place_field_centers = np.array([0.0, 10.0, 20.0, 30.0])
        result = remap_place_field_centers(place_field_centers, (2, 0), active=True)
        np.testing.assert_array_equal(result, [0.0, 10.0, 0.0, 30.0])

    def test_multiple_remappings_use_original_dst_values(self) -> None:
        """Bidirectional swap (0->1, 1->0) must use *original* values, not
        sequentially overwritten ones — otherwise both end up with the same
        center."""
        place_field_centers = np.array([0.0, 10.0, 20.0, 30.0])
        result = remap_place_field_centers(place_field_centers, ((0, 1), (1, 0)), active=True)
        np.testing.assert_array_equal(result, [10.0, 0.0, 20.0, 30.0])

    def test_default_global_remap_is_complete_and_well_separated(self) -> None:
        """Every default cell is remapped exactly once and moves at least
        three center spacings, as required by the Figure 3 positive control.
        """
        params = Figure3Config()
        assert params.place_field_centers is not None
        mapping = np.asarray(params.place_field_remapping, dtype=int)
        n_cells = params.place_field_centers.size

        np.testing.assert_array_equal(np.sort(mapping[:, 0]), np.arange(n_cells))
        np.testing.assert_array_equal(np.sort(mapping[:, 1]), np.arange(n_cells))
        assert np.all(np.abs(mapping[:, 0] - mapping[:, 1]) >= 3)

        result = remap_place_field_centers(
            params.place_field_centers,
            params.place_field_remapping,
            active=True,
        )
        np.testing.assert_array_equal(result, params.place_field_centers[mapping[:, 1]])


class TestFigure3SimulationResultDataclass:
    """The ``TypedDict`` → frozen-dataclass conversion brought length
    validation. Cover the success contract and the failure modes."""

    def test_valid_construction_succeeds(self) -> None:
        """Happy path: a well-formed Figure3SimulationResult constructs cleanly,
        coerces list inputs to tuple, and exposes attribute access on
        every field."""

        n_bins = 5
        n_time = 10
        sim = Figure3SimulationResult(
            config=Figure3Config(),
            position_bins=np.linspace(0.0, 100.0, n_bins),
            true_position=np.zeros(n_time),
            spike_counts=np.zeros((n_time, 1), dtype=np.int_),
            diagnostics=_zero_diagnostics(n_time=n_time, n_bins=n_bins),
            phase_labels=PHASE_LABELS,
            phase_boundaries=(1, 2, 3, 4, 5, 6, 7, n_time),
        )
        # Sequence fields coerced to tuple by __post_init__.
        assert isinstance(sim.phase_labels, tuple)
        assert isinstance(sim.phase_boundaries, tuple)
        # Attribute access works (the migration test).
        assert sim.position_bins.shape == (n_bins,)
        assert sim.true_position.shape == (n_time,)

    def test_phase_labels_wrong_order_raises(self) -> None:
        n_bins = 5
        n_time = 10
        bogus_labels = tuple(reversed(PHASE_LABELS))
        with pytest.raises(ValueError, match="phase_labels must equal PHASE_LABELS"):
            Figure3SimulationResult(
                config=Figure3Config(),
                position_bins=np.linspace(0.0, 100.0, n_bins),
                true_position=np.zeros(n_time),
                spike_counts=np.zeros((n_time, 1), dtype=np.int_),
                diagnostics=_zero_diagnostics(n_time=n_time, n_bins=n_bins),
                phase_labels=bogus_labels,
                phase_boundaries=(1, 2, 3, 4, 5, 6, 7, n_time),
            )

    def test_phase_boundary_length_mismatch_raises(self) -> None:
        n_bins = 5
        n_time = 10
        with pytest.raises(ValueError, match="phase_boundaries length"):
            Figure3SimulationResult(
                config=Figure3Config(),
                position_bins=np.linspace(0.0, 100.0, n_bins),
                true_position=np.zeros(n_time),
                spike_counts=np.zeros((n_time, 1), dtype=np.int_),
                diagnostics=_zero_diagnostics(n_time=n_time, n_bins=n_bins),
                phase_labels=PHASE_LABELS,
                phase_boundaries=(1, 2, 3),  # wrong length
            )

    def test_spikes_and_x_true_timeline_mismatch_raises(self) -> None:
        n_bins = 5
        n_time = 10
        with pytest.raises(ValueError, match="spike_counts timeline"):
            Figure3SimulationResult(
                config=Figure3Config(),
                position_bins=np.linspace(0.0, 100.0, n_bins),
                true_position=np.zeros(n_time),
                spike_counts=np.zeros((n_time + 1, 1), dtype=np.int_),  # off by one
                diagnostics=_zero_diagnostics(n_time=n_time, n_bins=n_bins),
                phase_labels=PHASE_LABELS,
                phase_boundaries=(1, 2, 3, 4, 5, 6, 7, n_time),
            )

    def test_final_boundary_must_equal_timeline_length(self) -> None:
        n_bins = 5
        n_time = 10
        with pytest.raises(ValueError, match="final phase boundary"):
            Figure3SimulationResult(
                config=Figure3Config(),
                position_bins=np.linspace(0.0, 100.0, n_bins),
                true_position=np.zeros(n_time),
                spike_counts=np.zeros((n_time, 1), dtype=np.int_),
                diagnostics=_zero_diagnostics(n_time=n_time, n_bins=n_bins),
                phase_labels=PHASE_LABELS,
                phase_boundaries=(1, 2, 3, 4, 5, 6, 7, n_time + 1),
            )

    def test_metrics_timeline_mismatch_against_x_true_raises(self) -> None:
        """``Figure3SimulationResult.__post_init__`` rejects a ``DecodingDiagnostics``
        whose ``posterior`` timeline doesn't match ``x_true``. The
        ``DecodingDiagnostics`` dataclass itself enforces internal-shape
        consistency; this test pins the cross-check against the outer
        timeline that only ``Figure3SimulationResult`` can see."""
        n_bins = 5
        n_time = 10
        # A perfectly-shaped DecodingDiagnostics with the wrong leading dim.
        bad_metrics = _zero_diagnostics(n_time=n_time + 1, n_bins=n_bins)
        with pytest.raises(ValueError, match=r"diagnostics.posterior leading dim"):
            Figure3SimulationResult(
                config=Figure3Config(),
                position_bins=np.linspace(0.0, 100.0, n_bins),
                true_position=np.zeros(n_time),
                spike_counts=np.zeros((n_time, 1), dtype=np.int_),
                diagnostics=bad_metrics,
                phase_labels=PHASE_LABELS,
                phase_boundaries=(1, 2, 3, 4, 5, 6, 7, n_time),
            )
