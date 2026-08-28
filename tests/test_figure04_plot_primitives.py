"""Tests for shared low-level Figure-4 plotting helpers."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # noqa: E402

from statespacecheck_paper.figure04_plot_primitives import (  # noqa: E402
    compute_half_pixel_extent,
)


class TestHalfpixelExtent:
    def test_reproduces_inline_formula_on_normal_grid(self) -> None:
        """On a >=2-element grid the helper must reproduce the original inline
        ``(t0 - dt, t1 + dt, p0 - dp, p1 + dp)`` 4-tuple exactly."""
        time_coords = np.array([0.0, 1.0, 2.0, 3.0])
        pos_coords = np.array([10.0, 12.0, 14.0])

        # Original inline geometry (with the max(len-1, 1) guard).
        t0, t1 = float(time_coords[0]), float(time_coords[-1])
        p0, p1 = float(pos_coords[0]), float(pos_coords[-1])
        dt = (t1 - t0) / max(len(time_coords) - 1, 1) / 2
        dp = (p1 - p0) / max(len(pos_coords) - 1, 1) / 2
        expected = (t0 - dt, t1 + dt, p0 - dp, p1 + dp)

        assert compute_half_pixel_extent(time_coords, pos_coords) == expected
        assert expected == (-0.5, 3.5, 9.0, 15.0)

    def test_raises_on_single_time_coordinate(self) -> None:
        with pytest.raises(ValueError, match=">=2 coordinates"):
            compute_half_pixel_extent(np.array([1.0]), np.array([10.0, 12.0]))

    def test_raises_on_single_position_coordinate(self) -> None:
        with pytest.raises(ValueError, match=">=2 coordinates"):
            compute_half_pixel_extent(np.array([0.0, 1.0]), np.array([10.0]))
