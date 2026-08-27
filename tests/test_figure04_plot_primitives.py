"""Tests for shared low-level Figure-4 plotting helpers."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # noqa: E402

import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402

from statespacecheck_paper.figure04_plot_primitives import (  # noqa: E402
    _decoder_likelihood_to_columns,
    _halfpixel_extent,
    _neglog,
)


class TestNeglog:
    def test_matches_maximum_floor_elementwise(self) -> None:
        """``_neglog`` equals ``-log(max(x, eps))`` on every element, including
        values at or below the floor where the ``eps`` clamp dominates."""
        x = np.array([1.0, 0.5, 0.1, 1e-10, 1e-12, 0.0])
        expected = -np.log(np.maximum(x, 1e-10))
        np.testing.assert_array_equal(_neglog(x), expected)

    def test_respects_custom_eps(self) -> None:
        x = np.array([1.0, 1e-4, 0.0])
        expected = -np.log(np.maximum(x, 1e-3))
        np.testing.assert_array_equal(_neglog(x, eps=1e-3), expected)

    def test_scalar_threshold_transform(self) -> None:
        """The scalar overload (used on flag thresholds) floors identically."""
        assert _neglog(0.2) == -np.log(max(0.2, 1e-10))
        assert _neglog(0.0) == -np.log(1e-10)


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

        assert _halfpixel_extent(time_coords, pos_coords) == expected
        assert expected == (-0.5, 3.5, 9.0, 15.0)

    def test_raises_on_single_time_coordinate(self) -> None:
        with pytest.raises(ValueError, match=">=2 coordinates"):
            _halfpixel_extent(np.array([1.0]), np.array([10.0, 12.0]))

    def test_raises_on_single_position_coordinate(self) -> None:
        with pytest.raises(ValueError, match=">=2 coordinates"):
            _halfpixel_extent(np.array([0.0, 1.0]), np.array([10.0]))


def _decoder_results(
    n_time: int = 5,
    states: tuple[str, ...] = ("s0", "s1"),
    positions: tuple[float, ...] = (0.0, 2.0, 4.0),
) -> xr.Dataset:
    """Build a synthetic decoder result with a ``(state, position)`` MultiIndex."""
    pos = np.array(positions)
    state_bins = pd.MultiIndex.from_product([list(states), pos], names=["state", "position"])
    rng = np.random.default_rng(0)
    log_lik = rng.normal(size=(n_time, len(state_bins)))
    time = 10.0 + 0.002 * np.arange(n_time)
    da = xr.DataArray(
        log_lik,
        dims=("time", "state_bins"),
        coords={"time": time, "state_bins": state_bins},
    )
    return xr.Dataset({"log_likelihood": da})


class TestDecoderLikelihoodToColumns:
    def test_marginalizes_state_and_returns_coords(self) -> None:
        """Reproduces ``exp -> unstack -> sum(state) -> isel`` columns plus the
        time/position coordinate arrays used to build the imshow extent."""
        results = _decoder_results()
        lik_np, time_coords, pos_coords = _decoder_likelihood_to_columns(results, slice(None))

        n_time, n_states, n_pos = 5, 2, 3
        raw = results["log_likelihood"].values  # (n_time, n_states * n_pos)
        expected = np.exp(raw).reshape(n_time, n_states, n_pos).sum(axis=1)

        np.testing.assert_allclose(lik_np, expected)
        np.testing.assert_array_equal(pos_coords, np.array([0.0, 2.0, 4.0]))
        np.testing.assert_allclose(time_coords, 10.0 + 0.002 * np.arange(n_time))

    def test_honors_time_slice(self) -> None:
        results = _decoder_results()
        lik_full, time_full, _ = _decoder_likelihood_to_columns(results, slice(None))
        lik_win, time_win, _ = _decoder_likelihood_to_columns(results, slice(1, 3))

        assert lik_win.shape == (2, 3)
        np.testing.assert_allclose(lik_win, lik_full[1:3])
        np.testing.assert_allclose(time_win, time_full[1:3])

    def test_single_state_index_raises(self) -> None:
        """Data without a (state, position) MultiIndex has no position axis."""
        n_time = 4
        da = xr.DataArray(
            np.random.default_rng(0).normal(size=(n_time, 3)),
            dims=("time", "state_bins"),
            coords={"time": np.arange(n_time, dtype=float), "state_bins": [0, 1, 2]},
        )
        results = xr.Dataset({"log_likelihood": da})
        with pytest.raises(NotImplementedError, match="MultiIndex"):
            _decoder_likelihood_to_columns(results, slice(None))
