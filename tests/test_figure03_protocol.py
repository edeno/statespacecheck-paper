"""Tests for the Figure-3 protocol (config + phase ladder)."""

from __future__ import annotations

import numpy as np
import pytest

from statespacecheck_paper.figure03_protocol import Figure3Config, PhaseBoundary


class TestFigure3Config:
    def test_post_init_initializes_pf_centers_to_grid(self) -> None:
        params = Figure3Config()
        np.testing.assert_array_equal(
            params.place_field_centers, np.arange(0, 101, 10, dtype=float)
        )

    def test_post_init_respects_provided_pf_centers(self) -> None:
        custom = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        params = Figure3Config(place_field_centers=custom)
        np.testing.assert_array_equal(params.place_field_centers, custom)


class TestFigure3ConfigPhaseBoundaries:
    """The phase ladder collapsed from 8 ``T_*`` fields to one
    ``phase_boundaries`` tuple. Lock the invariants in.
    """

    def test_phase_boundaries_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="must have 8 entries"):
            Figure3Config(phase_boundaries=(1, 2, 3))

    def test_phase_boundaries_non_monotonic_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            Figure3Config(
                phase_boundaries=(100, 200, 200, 400, 500, 600, 700, 800),
            )

    def test_phase_boundaries_equal_consecutive_raises(self) -> None:
        """Equal consecutive entries (zero-width phase) must reject too."""
        with pytest.raises(ValueError, match="strictly increasing"):
            Figure3Config(
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
        params = Figure3Config(phase_boundaries=boundaries)
        assert params.phase_boundaries[member] == boundaries[index]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"sparse_position": -1.0}, "sparse_position"),
            ({"sparse_approach_duration_steps": -1}, "sparse_approach_duration_steps"),
            ({"sparse_control_ordinary_rate_scale": 1.1}, "sparse_control_ordinary_rate_scale"),
            ({"sparse_cell_count": 0}, "sparse_cell_count"),
            ({"sparse_place_field_spread": -1.0}, "sparse_place_field_spread"),
            ({"sparse_place_field_spread": np.nan}, "sparse_place_field_spread"),
            ({"sparse_place_field_std": 0.0}, "sparse_place_field_std"),
            ({"sparse_place_field_std": np.nan}, "sparse_place_field_std"),
            ({"sparse_cell_peak_rate_per_step": 0.0}, "sparse_cell_peak_rate_per_step"),
            ({"sparse_cell_baseline_rate_fraction": -0.1}, "sparse_cell_baseline_rate_fraction"),
        ],
    )
    def test_sparse_population_parameters_reject_invalid_values(
        self,
        kwargs: dict[str, float | int],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            Figure3Config(**kwargs)
