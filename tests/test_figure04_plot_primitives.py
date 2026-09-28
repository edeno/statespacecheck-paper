"""Tests for shared low-level Figure-4 plotting helpers."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from statespacecheck_paper.figure04_plot_primitives import (  # noqa: E402
    _state_bins_with_any_value,
    compute_half_pixel_extent,
    plot_distribution_heatmap,
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


def _two_state_distribution(values: np.ndarray, positions: np.ndarray) -> xr.DataArray:
    state_bins = pd.MultiIndex.from_product(
        [["Continuous", "Fragmented"], positions], names=["state", "position"]
    )
    return xr.DataArray(
        values,
        dims=("time", "state_bins"),
        coords={"time": np.arange(values.shape[0], dtype=float), "state_bins": state_bins},
    )


class TestPlotDistributionHeatmap:
    @pytest.mark.parametrize("time_chunk", [1, 3, 16_384])
    def test_kept_bins_match_session_wide_dropna(self, time_chunk: int) -> None:
        """The chunked scan keeps exactly the bins ``dropna(how="all")`` keeps."""
        rng = np.random.default_rng(0)
        values = rng.random((10, 8))
        values[:, [0, 4]] = np.nan  # off-track at every time
        values[2:6, 1] = np.nan  # NaN only inside part of the session
        values[:, 2] = np.nan
        values[7, 2] = 0.5  # a single non-NaN value keeps the bin
        da = _two_state_distribution(values, np.arange(4.0))

        kept = _state_bins_with_any_value(da, time_chunk=time_chunk)

        expected = da.dropna("state_bins", how="all").indexes["state_bins"]
        assert da.indexes["state_bins"][kept].equals(expected)

    def test_bin_nan_only_inside_window_is_kept_as_nan_row(self) -> None:
        """A bin that is NaN inside the plotted window but valid elsewhere in the
        session keeps its position row (drawn as NaN), as with a session-wide
        dropna before slicing; off-track bins are dropped."""
        rng = np.random.default_rng(1)
        values = rng.random((12, 10))
        values[:, [0, 5]] = np.nan  # position 0 off-track in both states
        values[3:8, [2, 7]] = np.nan  # position 2 NaN only within the window
        da = _two_state_distribution(values, np.arange(5.0))
        fig, ax = plt.subplots()

        plot_distribution_heatmap(
            ax, da, da.time.values, np.zeros(12), slice(3, 8), show_position=False
        )

        mesh = ax.collections[0]
        plotted = np.ma.getdata(mesh.get_array()).reshape(4, 5)  # (position, time)
        assert np.isnan(plotted[1]).all()  # position 2 kept, NaN in the window
        assert np.isfinite(np.delete(plotted, 1, axis=0)).all()
        plt.close(fig)
