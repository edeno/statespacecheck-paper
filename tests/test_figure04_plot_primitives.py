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
    def test_pads_each_axis_by_half_a_pixel(self) -> None:
        """On a >=2-element grid the helper pads each axis outward by half a
        pixel: ``(t0 - dt, t1 + dt, p0 - dp, p1 + dp)``."""
        time_coords = np.array([0.0, 1.0, 2.0, 3.0])  # dt/2 = 0.5
        pos_coords = np.array([10.0, 12.0, 14.0])  # dp/2 = 1.0

        assert compute_half_pixel_extent(time_coords, pos_coords) == (-0.5, 3.5, 9.0, 15.0)

    def test_raises_on_single_time_coordinate(self) -> None:
        with pytest.raises(ValueError, match=">=2 coordinates"):
            compute_half_pixel_extent(np.array([1.0]), np.array([10.0, 12.0]))

    def test_raises_on_single_position_coordinate(self) -> None:
        with pytest.raises(ValueError, match=">=2 coordinates"):
            compute_half_pixel_extent(np.array([0.0, 1.0]), np.array([10.0]))
