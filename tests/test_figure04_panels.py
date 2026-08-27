"""Tests for Figure-4 raster and diagnostic panels.

Covers ``plot_per_spike_metric_hexbin_row`` — the Figure 4(c) whole-session
comparison panel — and the per-cell diagnostic scatter's spike-time alignment
and running-average behavior.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
import xarray as xr  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from statespacecheck_paper.diagnostics import SpikeEventDiagnostics  # noqa: E402
from statespacecheck_paper.figure04_panels import (  # noqa: E402
    ModelDiagnosticPanelData,
    _draw_decoder_likelihood_image,
    _draw_predictive_heatmap_row,
    _draw_track_graph_edges,
    plot_exploratory_model_comparison,
    plot_per_spike_metric_hexbin_row,
    plot_single_model_diagnostics,
    plot_spike_event_diagnostic_scatter,
)

# ---------------------------------------------------------------------------
# plot_per_spike_metric_hexbin_row
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# plot_spike_event_diagnostic_scatter (spike-time alignment behavior)
# ---------------------------------------------------------------------------


def _diagnostics_from_metric(
    metric_name: str,
    metric: np.ndarray,
    *,
    event_time: np.ndarray | None = None,
    event_values: np.ndarray | None = None,
) -> SpikeEventDiagnostics:
    """Build a ``SpikeEventDiagnostics`` from a single (n_time, n_cells) metric.

    Other metric fields are filled with NaN/zeros matching shape; the
    scatter helper under test only consumes the named metric plus the
    optional ``event_*`` arrays.
    """
    n_time, n_cells = metric.shape
    blank_2d = np.full((n_time, n_cells), np.nan)
    n_spikes = 0 if event_time is None else event_time.shape[0]
    blank_evt = np.zeros(n_spikes)

    def _named(name: str, value: np.ndarray) -> np.ndarray:
        return value if name == metric_name else blank_2d

    def _named_evt(name: str) -> np.ndarray:
        if event_values is not None and name == f"event_{metric_name}":
            return event_values
        return blank_evt

    return SpikeEventDiagnostics(
        event_time_ind=np.zeros(n_spikes, dtype=np.intp),
        event_cell_ind=np.zeros(n_spikes, dtype=np.intp),
        event_hpd_overlap=_named_evt("event_hpd_overlap"),
        event_kl_divergence=_named_evt("event_kl_divergence"),
        event_predictive_pvalue=_named_evt("event_predictive_pvalue"),
        hpd_overlap=_named("hpd_overlap", metric),
        kl_divergence=_named("kl_divergence", metric),
        predictive_pvalue=_named("predictive_pvalue", metric),
        per_spike_likelihood=np.zeros((n_spikes, 1)),
        event_time=event_time,
    )


def _scatter_offsets(ax: plt.Axes) -> np.ndarray:
    offsets = ax.collections[0].get_offsets()
    mask = np.ma.getmaskarray(offsets)
    return np.asarray(offsets)[~mask.any(axis=1)]


class TestPlotSpikeEventDiagnosticScatter:
    def test_with_spike_times_aligns_at_actual_spike_times(self) -> None:
        """``spike_times`` shifts scatter dots to the actual spike instants
        instead of the bin starts (which are 100ms apart here)."""
        time = np.linspace(0.0, 0.9, 10)
        hpd = np.full((10, 3), np.nan)
        hpd[1, 0] = 0.8
        hpd[3, 1] = 0.6
        hpd[5, 2] = 0.4
        diagnostics = _diagnostics_from_metric("hpd_overlap", hpd)

        fig, ax = plt.subplots()
        plot_spike_event_diagnostic_scatter(
            time,
            diagnostics,
            ax=ax,
            spike_times=[
                np.array([0.15]),
                np.array([0.35]),
                np.array([0.55]),
            ],
        )
        offsets = _scatter_offsets(ax)
        np.testing.assert_allclose(sorted(offsets[:, 0]), [0.15, 0.35, 0.55])
        plt.close(fig)

    def test_event_diagnostics_plot_at_exact_event_times(self) -> None:
        """When ``event_*`` arrays are present, scatter uses their times
        directly with no bin lookup."""
        time = np.linspace(0.0, 0.9, 10)
        hpd = np.full((10, 1), np.nan)
        hpd[1, 0] = 0.7
        diagnostics = _diagnostics_from_metric(
            "hpd_overlap",
            hpd,
            event_time=np.array([0.151, 0.157]),
            event_values=np.array([0.8, 0.6]),
        )

        fig, ax = plt.subplots()
        plot_spike_event_diagnostic_scatter(time, diagnostics, ax=ax)
        offsets = _scatter_offsets(ax)
        np.testing.assert_allclose(offsets[:, 0], [0.151, 0.157])
        np.testing.assert_allclose(offsets[:, 1], [0.8, 0.6])
        plt.close(fig)

    def test_without_spike_times_uses_bin_centers(self) -> None:
        """Without per-spike alignment, scatter uses bin-start times."""
        time = np.linspace(0.0, 0.9, 10)
        hpd = np.full((10, 2), np.nan)
        hpd[1, 0] = 0.8
        hpd[3, 1] = 0.6
        diagnostics = _diagnostics_from_metric("hpd_overlap", hpd)

        fig, ax = plt.subplots()
        plot_spike_event_diagnostic_scatter(time, diagnostics, ax=ax, spike_times=None)
        offsets = _scatter_offsets(ax)
        np.testing.assert_allclose(sorted(offsets[:, 0]), [0.1, 0.3])
        plt.close(fig)


class TestPlotSpikeEventDiagnosticScatterRunningAverage:
    def test_running_average_adds_a_line_to_axis(self, rng: np.random.Generator) -> None:
        time = np.linspace(0.0, 1.0, 100)
        diagnostics = _diagnostics_from_metric("hpd_overlap", rng.random((100, 10)))

        fig_off, ax_off = plt.subplots()
        plot_spike_event_diagnostic_scatter(
            time, diagnostics, ax=ax_off, show_running_average=False
        )
        n_off = len(ax_off.lines)
        plt.close(fig_off)

        fig_on, ax_on = plt.subplots()
        plot_spike_event_diagnostic_scatter(time, diagnostics, ax=ax_on, show_running_average=True)
        assert len(ax_on.lines) == n_off + 1
        plt.close(fig_on)

    def test_running_average_window_size_changes_curve(self, rng: np.random.Generator) -> None:
        time = np.linspace(0.0, 1.0, 100)
        diagnostics = _diagnostics_from_metric("hpd_overlap", rng.random((100, 10)))

        def _line_y(window: float) -> np.ndarray:
            fig, ax = plt.subplots()
            plot_spike_event_diagnostic_scatter(
                time,
                diagnostics,
                ax=ax,
                show_running_average=True,
                running_average_window=window,
            )
            y = np.asarray(ax.lines[0].get_ydata()).copy()
            plt.close(fig)
            return y

        assert not np.allclose(_line_y(0.05), _line_y(0.2))

    def test_predictive_pvalue_running_average_uses_raw_then_transforms(self) -> None:
        """Critical correctness: -log(mean(p)) != mean(-log(p)). Running
        average must average raw probabilities first, then take -log."""
        predictive_pvalues = np.array(
            [
                [0.01, 0.99],  # mean(raw) = 0.5
                [0.1, 0.9],  # mean(raw) = 0.5
                [0.5, 0.5],  # mean(raw) = 0.5 (control)
            ]
        )
        time = np.linspace(0, 0.2, 3)
        diagnostics = _diagnostics_from_metric("predictive_pvalue", predictive_pvalues)

        fig, ax = plt.subplots()
        plot_spike_event_diagnostic_scatter(
            time,
            diagnostics,
            ax=ax,
            metric_name="predictive_pvalue",
            show_running_average=True,
            running_average_window=0.01,
        )
        y_actual = np.asarray(ax.lines[0].get_ydata())

        # Correct path: average raw, then -log (natural log).
        expected = -np.log(np.maximum(np.mean(predictive_pvalues, axis=1), 1e-10))
        np.testing.assert_allclose(y_actual, expected, rtol=1e-3)

        # Wrong path: -log first, then average. Different on rows 0 and 1.
        wrong = np.mean(-np.log(np.maximum(predictive_pvalues, 1e-10)), axis=1)
        assert not np.allclose(y_actual, wrong, rtol=1e-3)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Composite panels + extracted row renderers
# ---------------------------------------------------------------------------

_N_TIME, _N_CELLS, _N_POS = 20, 6, 10
_STATES = ("Continuous", "Fragmented")


def _multistate_results(seed: int) -> xr.Dataset:
    """Synthetic decoder results with a (state, position) MultiIndex."""
    rng = np.random.default_rng(seed)
    pos = np.linspace(0.0, 100.0, _N_POS)
    state_bins = pd.MultiIndex.from_product([list(_STATES), pos], names=["state", "position"])
    time = np.arange(_N_TIME, dtype=float)
    n_sb = len(state_bins)

    def _da(a: np.ndarray) -> xr.DataArray:
        return xr.DataArray(
            a, dims=("time", "state_bins"), coords={"time": time, "state_bins": state_bins}
        )

    return xr.Dataset(
        {
            "predictive_posterior": _da(rng.dirichlet(np.ones(n_sb), size=_N_TIME)),
            "log_likelihood": _da(rng.normal(size=(_N_TIME, n_sb))),
        }
    )


def _dense_diagnostics(seed: int) -> SpikeEventDiagnostics:
    """SpikeEventDiagnostics with dense (n_time, n_cells) metric matrices."""
    rng = np.random.default_rng(seed)
    n_spk = 15
    return SpikeEventDiagnostics(
        event_time_ind=rng.integers(0, _N_TIME, n_spk).astype(np.intp),
        event_cell_ind=rng.integers(0, _N_CELLS, n_spk).astype(np.intp),
        event_hpd_overlap=rng.uniform(0, 1, n_spk),
        event_kl_divergence=rng.gamma(2.0, 0.5, n_spk),
        event_predictive_pvalue=rng.uniform(0.01, 1, n_spk),
        hpd_overlap=rng.uniform(0, 1, (_N_TIME, _N_CELLS)),
        kl_divergence=rng.gamma(2.0, 0.5, (_N_TIME, _N_CELLS)),
        predictive_pvalue=rng.uniform(0.01, 1, (_N_TIME, _N_CELLS)),
        per_spike_likelihood=rng.uniform(0, 1, (n_spk, _N_POS)),
        event_time=rng.uniform(0, _N_TIME, n_spk),
    )


def _linear_track_graph() -> nx.Graph:
    g = nx.Graph()
    for i in range(6):
        g.add_node(i, pos=(float(i), 0.0))
    for i in range(5):
        g.add_edge(i, i + 1, distance=1.0)
    return g


def _panel_inputs() -> dict:
    rng = np.random.default_rng(7)
    return {
        "time": np.arange(_N_TIME, dtype=float),
        "position": np.linspace(10.0, 90.0, _N_TIME),
        "spike_times": [
            np.sort(rng.uniform(0, _N_TIME, rng.integers(3, 8))) for _ in range(_N_CELLS)
        ],
        "spike_counts": rng.poisson(0.4, (_N_TIME, _N_CELLS)).astype(np.int64),
        "place_field_peaks": np.linspace(5.0, 95.0, _N_CELLS),
    }


class TestPlotExploratoryModelComparison:
    def test_renders_six_by_two_grid_with_all_rows(self) -> None:
        c = _panel_inputs()
        fig, axes = plot_exploratory_model_comparison(
            c["time"],
            c["position"],
            _multistate_results(1),
            _multistate_results(2),
            _dense_diagnostics(3),
            _dense_diagnostics(4),
            spike_times=c["spike_times"],
            spike_counts=c["spike_counts"],
            place_field_peaks=c["place_field_peaks"],
            thresholds={"hpd_overlap": 0.05, "predictive_pvalue": 0.05},
            track_graph=_linear_track_graph(),
            edge_order=[(i, i + 1) for i in range(5)],
            show_running_average=True,
        )
        assert axes.shape == (6, 2)
        # Predictive heatmap row draws a pcolormesh (collections).
        assert axes[0, 0].collections
        # Raster row draws eventplot line collections.
        assert axes[2, 0].collections
        plt.close(fig)


class TestPlotSingleModelDiagnostics:
    def test_renders_six_rows_with_place_field_likelihood(self) -> None:
        c = _panel_inputs()
        rng = np.random.default_rng(9)
        panel_data = ModelDiagnosticPanelData(
            time=c["time"],
            position=c["position"],
            results=_multistate_results(5),
            diagnostics=_dense_diagnostics(6),
            spike_times=c["spike_times"],
            spike_counts=c["spike_counts"],
            place_field_peaks=c["place_field_peaks"],
            place_fields=rng.random((_N_CELLS, _N_POS)) * 10 + 0.1,
            position_bins=np.linspace(0.0, 100.0, _N_POS),
            track_graph=_linear_track_graph(),
            edge_order=[(i, i + 1) for i in range(5)],
        )
        fig, axes = plot_single_model_diagnostics(
            panel_data,
            thresholds={"hpd_overlap": 0.05},
        )
        assert axes.shape == (6,)
        assert axes[0].images or axes[0].collections
        plt.close(fig)

    def test_panel_data_rejects_misaligned_position(self) -> None:
        with pytest.raises(ValueError, match="position must match"):
            ModelDiagnosticPanelData(
                time=np.arange(_N_TIME, dtype=float),
                position=np.zeros(_N_TIME - 1),
                results=_multistate_results(5),
                diagnostics=_dense_diagnostics(6),
            )


class TestExtractedRowRenderers:
    def test_predictive_row_sets_title_and_ylabel(self) -> None:
        fig, ax = plt.subplots()
        _draw_predictive_heatmap_row(
            ax,
            _multistate_results(1),
            np.arange(_N_TIME, dtype=float),
            np.linspace(0.0, 100.0, _N_TIME),
            slice(None),
            title="ModelX",
            ylabel="Predictive",
        )
        assert ax.get_title() == "ModelX"
        assert ax.get_ylabel() == "Predictive"
        plt.close(fig)

    def test_decoder_likelihood_image_is_noop_without_log_likelihood(self) -> None:
        results = _multistate_results(1).drop_vars("log_likelihood")
        fig, ax = plt.subplots()
        before = len(ax.images) + len(ax.collections)
        _draw_decoder_likelihood_image(ax, results, slice(None), None)
        assert len(ax.images) + len(ax.collections) == before
        plt.close(fig)

    def test_decoder_likelihood_image_draws_when_present(self) -> None:
        fig, ax = plt.subplots()
        _draw_decoder_likelihood_image(ax, _multistate_results(1), slice(None), None)
        assert ax.images or ax.collections
        plt.close(fig)

    def test_track_graph_edges_draw_lines_on_each_axis(self) -> None:
        fig, (ax0, ax1) = plt.subplots(1, 2)
        _draw_track_graph_edges(
            [ax0, ax1],
            _linear_track_graph(),
            [(i, i + 1) for i in range(5)],
            0.0,
            np.arange(_N_TIME, dtype=float),
            slice(None),
        )
        assert ax0.lines and ax1.lines
        plt.close(fig)


# ---------------------------------------------------------------------------
# ModelDiagnosticPanelData validation
# ---------------------------------------------------------------------------


def _diag_all_dense_none() -> SpikeEventDiagnostics:
    """Diagnostics with the dense matrices absent (event arrays only)."""
    n = 5
    return SpikeEventDiagnostics(
        event_time_ind=np.zeros(n, dtype=np.intp),
        event_cell_ind=np.zeros(n, dtype=np.intp),
        event_hpd_overlap=np.zeros(n),
        event_kl_divergence=np.zeros(n),
        event_predictive_pvalue=np.zeros(n),
        hpd_overlap=None,
        kl_divergence=None,
        predictive_pvalue=None,
        per_spike_likelihood=None,
    )


def _diag_wrong_time_rows() -> SpikeEventDiagnostics:
    """Diagnostics whose dense matrices have one too many time rows."""
    n = 5
    bad = (_N_TIME + 1, _N_CELLS)
    return SpikeEventDiagnostics(
        event_time_ind=np.zeros(n, dtype=np.intp),
        event_cell_ind=np.zeros(n, dtype=np.intp),
        event_hpd_overlap=np.zeros(n),
        event_kl_divergence=np.zeros(n),
        event_predictive_pvalue=np.zeros(n),
        hpd_overlap=np.zeros(bad),
        kl_divergence=np.zeros(bad),
        predictive_pvalue=np.zeros(bad),
        per_spike_likelihood=np.zeros((n, _N_POS)),
    )


def _valid_panel_kwargs() -> dict:
    c = _panel_inputs()
    rng = np.random.default_rng(0)
    return {
        "time": c["time"],
        "position": c["position"],
        "results": _multistate_results(1),
        "diagnostics": _dense_diagnostics(2),
        "spike_times": c["spike_times"],
        "spike_counts": c["spike_counts"],
        "place_field_peaks": c["place_field_peaks"],
        "place_fields": rng.random((_N_CELLS, _N_POS)) * 10 + 0.1,
        "position_bins": np.linspace(0.0, 100.0, _N_POS),
        "track_graph": _linear_track_graph(),
        "edge_order": [(i, i + 1) for i in range(5)],
    }


class TestModelDiagnosticPanelDataValidation:
    def test_valid_kwargs_construct(self) -> None:
        # The baseline the rejection cases mutate must itself be valid.
        ModelDiagnosticPanelData(**_valid_panel_kwargs())

    @pytest.mark.parametrize(
        "override, match",
        [
            ({"time": np.zeros((_N_TIME, 2))}, "time must be 1-D"),
            ({"diagnostics": _diag_all_dense_none()}, "must include the dense"),
            ({"diagnostics": _diag_wrong_time_rows()}, "one row per time sample"),
            (
                {"spike_counts": np.zeros((_N_TIME, _N_CELLS + 1), dtype=np.int64)},
                "spike_counts must have shape",
            ),
            ({"spike_times": [np.array([0.1])] * (_N_CELLS - 1)}, "spike_times must contain"),
            ({"place_field_peaks": np.zeros(_N_CELLS + 1)}, "place_field_peaks must have shape"),
            ({"position_bins": None}, "must be provided together"),
            ({"place_fields": np.zeros((_N_CELLS, _N_POS + 1))}, "place_fields must have shape"),
            (
                {"results": _multistate_results(1).drop_vars("predictive_posterior")},
                "predictive_posterior",
            ),
            (
                {"results": _multistate_results(1).isel(time=slice(0, _N_TIME - 1))},
                "results 'time' dimension",
            ),
            ({"track_graph": None}, "edge_order requires track_graph"),
        ],
    )
    def test_rejects_invalid(self, override: dict, match: str) -> None:
        kwargs = _valid_panel_kwargs()
        kwargs.update(override)
        with pytest.raises(ValueError, match=match):
            ModelDiagnosticPanelData(**kwargs)


def test_exploratory_comparison_alias_is_pinned() -> None:
    # Retained so pre-rename notebooks (notebooks/archive/fig4.ipynb) keep working.
    from statespacecheck_paper import figure04_panels

    assert (
        figure04_panels.plot_model_comparison_with_posterior
        is figure04_panels.plot_exploratory_model_comparison
    )
