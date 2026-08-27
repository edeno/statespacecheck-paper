"""Tests for plotting helpers in ``real_data_plotting``.

Currently focused on ``plot_per_spike_metric_hexbin_row`` — the Figure 4(c)
whole-session comparison panel — which had no direct coverage before.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics  # noqa: E402
from statespacecheck_paper.real_data_plotting import (  # noqa: E402
    _decoder_likelihood_to_columns,
    _halfpixel_extent,
    _neglog,
    plot_per_spike_metric_hexbin_row,
)


def _per_spike_diagnostics(
    hpd: np.ndarray, kl: np.ndarray, sp: np.ndarray
) -> SpikeEventDiagnostics:
    """Build a ``SpikeEventDiagnostics`` from per-spike metric arrays only.

    The hexbin helper consumes the three ``event_*`` arrays; the rest
    of the dataclass is required by the constructor but unused here.
    """
    n_spikes = hpd.shape[0]
    return SpikeEventDiagnostics(
        event_time_ind=np.zeros(n_spikes, dtype=np.intp),
        event_cell_ind=np.zeros(n_spikes, dtype=np.intp),
        event_hpd_overlap=hpd,
        event_kl_divergence=kl,
        event_predictive_pvalue=sp,
        hpd_overlap=None,
        kl_divergence=None,
        predictive_pvalue=None,
        per_spike_likelihood=None,
    )


@pytest.fixture
def paired_diagnostics() -> tuple[SpikeEventDiagnostics, SpikeEventDiagnostics]:
    """Two ``SpikeEventDiagnostics`` with 50 matched spike events each.

    Synthesized in-test so the helper is exercised without requiring real-data
    fixtures. Same n_spikes so the same-length contract in
    ``plot_per_spike_metric_hexbin_row`` is satisfied; correlated noise
    gives a hexbin with mass on the identity line plus a spread.
    """
    rng = np.random.default_rng(0)
    n_spikes = 50

    hpd_a = rng.uniform(0.0, 1.0, n_spikes)
    hpd_b = np.clip(hpd_a + rng.normal(0.0, 0.05, n_spikes), 0.0, 1.0)

    kl_a = rng.gamma(2.0, 0.5, n_spikes)
    kl_b = kl_a + rng.normal(0.0, 0.1, n_spikes)

    sp_a = rng.uniform(0.01, 1.0, n_spikes)
    sp_b = np.clip(sp_a + rng.normal(0.0, 0.02, n_spikes), 1e-3, 1.0)

    return _per_spike_diagnostics(hpd_a, kl_a, sp_a), _per_spike_diagnostics(hpd_b, kl_b, sp_b)


class TestPlotPerSpikeMetricHexbinRow:
    def test_renders_three_panels_with_hexbin_and_identity_line(
        self,
        paired_diagnostics: tuple[SpikeEventDiagnostics, SpikeEventDiagnostics],
    ) -> None:
        """Every panel must carry a hexbin (PolyCollection) and an identity
        Line2D — the two load-bearing visual elements of the comparison.
        """
        diag_a, diag_b = paired_diagnostics
        fig, axes = plt.subplots(1, 3)
        plot_per_spike_metric_hexbin_row(diag_a, diag_b, axes)

        for ax in axes:
            polys = [c for c in ax.collections if isinstance(c, PolyCollection)]
            assert polys, f"axis {ax.get_title()!r} has no PolyCollection (hexbin)"
            lines = [child for child in ax.get_children() if isinstance(child, Line2D)]
            # At least one Line2D (the identity reference) should be present.
            assert lines, f"axis {ax.get_title()!r} has no Line2D (identity reference)"
            # Aspect ratio locked to equal so the identity line is at 45 degrees.
            # Matplotlib normalises ``set_aspect("equal")`` to the float 1.0.
            assert ax.get_aspect() == 1.0

        plt.close(fig)

    def test_kl_panel_count_annotation_drops_nans(
        self,
        paired_diagnostics: tuple[SpikeEventDiagnostics, SpikeEventDiagnostics],
    ) -> None:
        """The KL panel's ``n=...`` annotation reports finite-in-both events,
        not the raw input length.

        Introduces 5 NaNs into each metric's arrays so the count should drop by
        exactly 5 in the one panel that carries the shared annotation.
        """
        diag_a, diag_b = paired_diagnostics
        n_total = diag_a.event_hpd_overlap.size
        n_nans = 5

        # Same-position NaNs across the two arrays so the mask is unambiguous.
        nan_positions = [0, 7, 13, 24, 41]

        def with_nans(d: SpikeEventDiagnostics) -> SpikeEventDiagnostics:
            hpd = d.event_hpd_overlap.copy()
            kl = d.event_kl_divergence.copy()
            sp = d.event_predictive_pvalue.copy()
            for arr in (hpd, kl, sp):
                arr[nan_positions] = np.nan
            return _per_spike_diagnostics(hpd, kl, sp)

        fig, axes = plt.subplots(1, 3)
        plot_per_spike_metric_hexbin_row(with_nans(diag_a), with_nans(diag_b), axes)

        import re

        expected_count = n_total - n_nans
        # Match any ``n=<digits>`` (with or without thousands separator
        # and surrounding whitespace) so a cosmetic format change
        # ("n = 45" or "n=45_000") doesn't break the test. The
        # behavioural contract is the integer in the annotation.
        pattern = re.compile(r"n\s*=\s*([\d,_]+)")
        panel_counts = {}
        for ax in axes:
            panel_counts[ax.get_title()] = [
                int(m.group(1).replace(",", "").replace("_", ""))
                for t in ax.texts
                for m in [pattern.match(t.get_text())]
                if m is not None
            ]
        assert panel_counts["HPD overlap"] == []
        assert panel_counts["KL divergence"] == [expected_count]
        assert panel_counts[r"$-\log(p)$"] == []

        plt.close(fig)

    def test_validates_same_length(
        self,
        paired_diagnostics: tuple[SpikeEventDiagnostics, SpikeEventDiagnostics],
    ) -> None:
        """Mismatched-shape inputs must raise — the helper would otherwise
        produce a plausible-looking hexbin on misaligned arrays.
        """
        diag_a, diag_b = paired_diagnostics
        diag_b_short = _per_spike_diagnostics(
            diag_b.event_hpd_overlap[:25],
            diag_b.event_kl_divergence[:25],
            diag_b.event_predictive_pvalue[:25],
        )
        fig, axes = plt.subplots(1, 3)
        with pytest.raises(ValueError, match="same set of spike events"):
            plot_per_spike_metric_hexbin_row(diag_a, diag_b_short, axes)
        plt.close(fig)

    def test_rejects_wrong_axes_count(
        self,
        paired_diagnostics: tuple[SpikeEventDiagnostics, SpikeEventDiagnostics],
    ) -> None:
        """The helper expects exactly three axes (one per metric)."""
        diag_a, diag_b = paired_diagnostics
        fig, axes = plt.subplots(1, 2)
        with pytest.raises(ValueError, match="axes must have length 3"):
            plot_per_spike_metric_hexbin_row(diag_a, diag_b, axes)
        plt.close(fig)

    def test_thresholds_draw_dotted_lines_and_rescue_patch(
        self,
        paired_diagnostics: tuple[SpikeEventDiagnostics, SpikeEventDiagnostics],
    ) -> None:
        """Passing ``thresholds`` adds two dotted threshold lines (one per
        axis) and one shaded rescue-quadrant rectangle to every panel.
        """
        from matplotlib.patches import Rectangle

        diag_a, diag_b = paired_diagnostics
        thresholds = {"hpd_overlap": 0.05, "kl_divergence": 4.52, "predictive_pvalue": 0.05}

        fig, axes = plt.subplots(1, 3)
        plot_per_spike_metric_hexbin_row(diag_a, diag_b, axes, thresholds=thresholds)

        for ax in axes:
            dotted = [ln for ln in ax.lines if ln.get_linestyle() in (":", "dotted")]
            assert len(dotted) >= 2, (
                f"{ax.get_title()!r}: expected 2 dotted threshold lines, got {len(dotted)}"
            )
            rects = [p for p in ax.patches if isinstance(p, Rectangle)]
            assert rects, f"{ax.get_title()!r}: no shaded rescue-quadrant patch"
        plt.close(fig)

    def test_no_thresholds_leaves_panels_unshaded(
        self,
        paired_diagnostics: tuple[SpikeEventDiagnostics, SpikeEventDiagnostics],
    ) -> None:
        """Without ``thresholds`` (the default), no dotted lines or rescue
        patches are drawn — only the dashed identity line per panel.
        """
        from matplotlib.patches import Rectangle

        diag_a, diag_b = paired_diagnostics
        fig, axes = plt.subplots(1, 3)
        plot_per_spike_metric_hexbin_row(diag_a, diag_b, axes)

        for ax in axes:
            assert not [ln for ln in ax.lines if ln.get_linestyle() in (":", "dotted")]
            assert not [p for p in ax.patches if isinstance(p, Rectangle)]
        plt.close(fig)


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
