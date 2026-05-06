"""Panel widgets for the Figure 4 interactive viewer.

This module owns the per-row plot widgets the viewer composes:

- ``PosteriorPanel`` and ``LikelihoodPanel`` — heatmaps stacked along
  the left-hand time axis.
- ``RasterPanel`` — sorted spike raster.
- ``MetricPanel`` — per-spike scatter for one of HPD overlap,
  KL divergence, or ``-log10(spike_prob)``.
- ``SlicePanel`` — the right-hand stacked slice column with the
  population-likelihood plot and a pool of per-cell-likelihood rows.

Each panel is self-contained (it does not import from ``viewer``)
and is driven by viewer-side updates: ``update_window`` for the
time-axis panels, ``update_for_index`` + ``set_per_cell_slices``
for the slice panel.

Module-internal constants (color palettes, stylesheets, the
``_pin_slice_axes`` helper, and the ``_PerCellRow`` / ``CellSlice``
dataclasses) live here too.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PySide6 import QtCore, QtWidgets

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

# Per-state colors (Continuous, Fragmented for ContFrag).
_STATE_POSTERIOR_RGB: tuple[tuple[int, int, int], ...] = (
    (31, 119, 180),  # blue
    (44, 160, 44),  # green
)
_STATE_LIKELIHOOD_RGB: tuple[tuple[int, int, int], ...] = (
    (255, 127, 14),  # orange
    (214, 39, 40),  # red
)
_LIKELIHOOD_PEN_RGB = (200, 90, 0)
_TRUE_POSITION_PEN = pg.mkPen((50, 50, 50), width=1, style=QtCore.Qt.PenStyle.DashLine)

# Palette for the per-cell place-field overlay. Picked to be distinct
# from the joint posterior (blue), joint likelihood (orange), pinned
# curve (gold), and true-position line (gray). Cells are colored by
# ``cell_id % len(palette)``, so distinct cells in the same bin land
# on different hues even when their IDs are adjacent.
_PER_CELL_PALETTE: tuple[tuple[int, int, int], ...] = (
    (44, 160, 44),  # green
    (214, 39, 40),  # red
    (148, 103, 189),  # purple
    (227, 119, 194),  # pink
    (23, 190, 207),  # cyan
    (140, 86, 75),  # brown
    (188, 189, 34),  # olive
    (227, 49, 165),  # magenta
)

METRIC_COLORS: dict[str, tuple[int, int, int]] = {
    "event_hpd_overlap": (44, 160, 44),
    "event_kl_divergence": (214, 39, 40),
    "event_spike_prob": (148, 103, 189),
}
METRIC_TITLES: dict[str, str] = {
    "event_hpd_overlap": "HPD overlap",
    "event_kl_divergence": "KL divergence",
    "event_spike_prob": "-log10(p)",
}

# Slice-panel y-range hard limits. All curves in the slice column are
# peak-normalized to 1, so this fits a priori; ``_pin_slice_axes``
# clamps the viewbox to this range.
_SLICE_Y_MIN = 0.0
_SLICE_Y_MAX = 1.05

# Cap on simultaneously visible per-cell rows; bins with more cells get
# a "(+K more)" indicator below.
MAX_PER_CELL_PLOTS = 6

_SLICE_LEGEND_STYLE = (
    "QLabel { background-color: #ffffff; color: #202020; "
    "padding: 4px 6px; border: 1px solid #cccccc; border-radius: 3px; "
    "font-size: 11pt; }"
)
_SLICE_READOUT_STYLE = (
    "QLabel { background-color: #ffffff; color: #202020; "
    "padding: 4px 6px; border: 1px solid #cccccc; border-radius: 3px; "
    "font-family: 'Menlo', 'Consolas', monospace; font-size: 11pt; }"
)
_SLICE_ANNOTATION_STYLE = (
    "QLabel { background-color: #fff7d6; color: #6a4f00; "
    "padding: 4px 6px; border: 1px solid #d4b85a; border-radius: 3px; "
    "font-family: 'Menlo', 'Consolas', monospace; font-size: 11pt; }"
)
_SLICE_CELL_HEADER_STYLE = (
    "QLabel { background-color: #f4f4f4; color: #202020; "
    "padding: 2px 6px; border: 1px solid #d8d8d8; border-radius: 3px; "
    "font-family: 'Menlo', 'Consolas', monospace; font-size: 10pt; }"
)
_SLICE_CELL_HEADER_PINNED_STYLE = (
    "QLabel { background-color: #fff2a8; color: #4d3700; "
    "padding: 2px 6px; border: 2px solid #d4b85a; border-radius: 3px; "
    "font-family: 'Menlo', 'Consolas', monospace; font-size: 10pt; "
    "font-weight: bold; }"
)


# ---------------------------------------------------------------------------
# Time-axis panels (left-hand column)
# ---------------------------------------------------------------------------


class _BaseHeatmapPanel(pg.PlotWidget):
    """Common posterior / likelihood heatmap behavior.

    Subclasses provide ``update_with_window`` which decides how to
    reduce a ``(n_visible, n_state_bins)`` window to a 2-D image; the
    rest of the panel (the ``ImageItem``, the time / position labels,
    the center-time and pin markers, the percentile-pinned color
    levels) is shared.
    """

    def __init__(self, *, title: str, position_bins: NDArray[np.float64]) -> None:
        super().__init__()
        self.setBackground("w")
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.setLabel("left", "Position (cm)")
        self.setLabel("bottom", "Time relative to center (s)")
        # Disable pyqtgraph's auto-SI-prefix on the time axis so the
        # tick labels do not flip between e.g. ``0.500`` and ``500
        # (×10⁻³)`` as the user scrolls through small / large values.
        self.getAxis("bottom").enableAutoSIPrefix(False)
        self.getPlotItem().setTitle(title)

        # ``axisOrder='col-major'`` interprets ``image[i, j]`` as
        # ``(x_index=i, y_index=j)``. Window arrays come in as
        # ``(n_time, n_state_bins)``, which we want displayed with
        # time on x and position on y — col-major lines up the array's
        # first axis with the time axis without a transpose.
        self._image = pg.ImageItem(axisOrder="col-major")
        self._image.setLookupTable(pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256))
        self.addItem(self._image)

        self._position_bins = np.asarray(position_bins, dtype=np.float64)
        self._y0 = float(self._position_bins[0])
        self._y1 = float(self._position_bins[-1])

        # Color levels are pinned from the first window's percentiles
        # (see ``update_window``) and then frozen so subsequent
        # ``setImage`` calls don't trigger a full nanmin/nanmax scan.
        self._levels: tuple[float, float] | None = None

        self._center_line = pg.InfiniteLine(
            angle=90,
            pos=0.0,
            pen=pg.mkPen((100, 100, 100), width=1, style=QtCore.Qt.PenStyle.DashLine),
            movable=False,
        )
        self.addItem(self._center_line)

        self._pin_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen((255, 255, 0), width=2),
            movable=False,
        )
        self._pin_line.setVisible(False)
        self.addItem(self._pin_line)

    def update_pinned_event(self, relative_time: float | None) -> None:
        if relative_time is None:
            self._pin_line.setVisible(False)
            return
        self._pin_line.setPos(relative_time)
        self._pin_line.setVisible(True)

    def update_window(
        self,
        time_start: float,
        time_end: float,
        array: NDArray[np.float32],
    ) -> None:
        if array.size == 0:
            return
        if self._levels is None:
            # ``(p1, p99)`` instead of ``(min, max)`` — robust to
            # outliers in the first window so the colormap doesn't
            # saturate or wash out later frames.
            lo = float(np.percentile(array, 1.0))
            hi = float(np.percentile(array, 99.0))
            if hi <= lo:
                hi = lo + 1.0
            self._levels = (lo, hi)
        self._image.setImage(
            array,
            autoLevels=False,
            levels=self._levels,
            autoDownsample=False,
        )
        self._image.setRect(
            QtCore.QRectF(
                time_start,
                self._y0,
                time_end - time_start,
                self._y1 - self._y0,
            )
        )
        self.setYRange(self._y0, self._y1, padding=0)


class PosteriorPanel(_BaseHeatmapPanel):
    """Predictive distribution heatmap (state-summed for multi-state models)."""

    def __init__(self, *, position_bins: NDArray[np.float64], n_states: int) -> None:
        super().__init__(title="Predictive distribution", position_bins=position_bins)
        self._n_states = max(1, n_states)
        self._n_pos = position_bins.shape[0]

    def update_with_window(
        self,
        time_start: float,
        time_end: float,
        post: NDArray[np.float32],
    ) -> None:
        arr = post
        if self._n_states > 1:
            n_visible = arr.shape[0]
            arr = arr.reshape(n_visible, self._n_states, self._n_pos).sum(axis=1)
        self.update_window(time_start, time_end, arr.astype(np.float32, copy=False))


class LikelihoodPanel(_BaseHeatmapPanel):
    """Likelihood heatmap (already exp'd by the worker thread)."""

    def __init__(self, *, position_bins: NDArray[np.float64], n_states: int) -> None:
        super().__init__(title="Likelihood", position_bins=position_bins)
        self._n_states = max(1, n_states)
        self._n_pos = position_bins.shape[0]

    def update_with_window(
        self,
        time_start: float,
        time_end: float,
        likelihood: NDArray[np.float32],
    ) -> None:
        lik = likelihood
        if self._n_states > 1:
            n_visible = lik.shape[0]
            lik = lik.reshape(n_visible, self._n_states, self._n_pos).sum(axis=1)
        self.update_window(time_start, time_end, lik)


class RasterPanel(pg.PlotWidget):
    """Spike raster sorted by place-field peak position."""

    def __init__(
        self,
        *,
        n_cells: int,
        place_field_peaks: NDArray[np.float64],
    ) -> None:
        super().__init__()
        self.setBackground("w")
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.setLabel("left", "Cell (PF rank)")
        self.setLabel("bottom", "Time relative to center (s)")
        self.getAxis("bottom").enableAutoSIPrefix(False)
        self.getPlotItem().setTitle("Raster")

        self._n_cells = int(n_cells)
        order = np.argsort(np.nan_to_num(place_field_peaks, nan=np.inf))
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        self._cell_rank = rank

        self._scatter = pg.ScatterPlotItem(
            pen=None,
            brush=pg.mkBrush(0, 0, 0, 255),
            size=2,
            pxMode=True,
            useCache=True,
        )
        self.addItem(self._scatter)
        self.setYRange(-0.5, n_cells - 0.5, padding=0)

        self._window_event_indices: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._on_click: Callable[[int], None] | None = None
        self._scatter.sigClicked.connect(self._handle_click)

        self._center_line = pg.InfiniteLine(
            angle=90,
            pos=0.0,
            pen=pg.mkPen((100, 100, 100), width=1, style=QtCore.Qt.PenStyle.DashLine),
            movable=False,
        )
        self.addItem(self._center_line)

        self._pin_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen((50, 50, 50), width=2),
            movable=False,
        )
        self._pin_line.setVisible(False)
        self.addItem(self._pin_line)
        self._pin_dot = pg.ScatterPlotItem(
            pen=pg.mkPen("k", width=1),
            brush=pg.mkBrush(255, 255, 0, 255),
            size=10,
            pxMode=True,
        )
        self._pin_dot.setVisible(False)
        self.addItem(self._pin_dot)

    def set_click_handler(self, handler: Callable[[int], None]) -> None:
        self._on_click = handler

    def update_window(
        self,
        time_start: float,
        time_end: float,
        events_time: NDArray[np.float64],
        events_cell_id: NDArray[np.int32],
        time_offset: float,
        global_event_indices: NDArray[np.int64] | None = None,
    ) -> None:
        if global_event_indices is None:
            self._window_event_indices = np.empty(0, dtype=np.int64)
        else:
            self._window_event_indices = np.asarray(global_event_indices, dtype=np.int64)

        if events_time.size == 0:
            self._scatter.setData(x=[], y=[], data=[])
            return
        x = events_time - time_offset
        y = self._cell_rank[events_cell_id]
        self._scatter.setData(x=x, y=y, data=np.arange(x.shape[0], dtype=np.int64))

    def update_pinned_event(
        self,
        *,
        relative_time: float | None,
        cell_id: int | None,
    ) -> None:
        if relative_time is None or cell_id is None:
            self._pin_line.setVisible(False)
            self._pin_dot.setVisible(False)
            return
        rank = float(self._cell_rank[int(cell_id)])
        self._pin_line.setPos(relative_time)
        self._pin_line.setVisible(True)
        self._pin_dot.setData(x=[relative_time], y=[rank])
        self._pin_dot.setVisible(True)

    def _handle_click(self, _scatter: pg.ScatterPlotItem, points: list[Any]) -> None:
        if not points or self._on_click is None:
            return
        spot = points[0]
        try:
            local_idx = int(spot.data())
        except (TypeError, ValueError):
            return
        if not 0 <= local_idx < self._window_event_indices.shape[0]:
            return
        self._on_click(int(self._window_event_indices[local_idx]))


class MetricPanel(pg.PlotWidget):
    """Per-spike scatter for one diagnostic metric."""

    def __init__(self, *, metric: str, threshold: float | None = None) -> None:
        super().__init__()
        self.setBackground("w")
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.setLabel("bottom", "Time relative to center (s)")
        self.getAxis("bottom").enableAutoSIPrefix(False)
        self.setLabel("left", METRIC_TITLES.get(metric, metric))

        self._metric = metric
        rgb = METRIC_COLORS.get(metric, (50, 50, 50))
        self._scatter = pg.ScatterPlotItem(
            pen=pg.mkPen(rgb, width=0),
            brush=pg.mkBrush(*rgb, 200),
            size=4,
            pxMode=True,
            useCache=True,
        )
        self.addItem(self._scatter)

        # Threshold horizontal line for the two metrics that have one
        # in the existing Figure 4 (HPD overlap = 0.05, spike_prob =
        # 0.05 ⇒ -log10(0.05) ≈ 1.30 on this axis).
        self._threshold_line: pg.InfiniteLine | None = None
        if threshold is not None:
            disp = -np.log10(threshold) if metric == "event_spike_prob" else threshold
            self._threshold_line = pg.InfiniteLine(
                pos=float(disp),
                angle=0,
                pen=pg.mkPen((100, 100, 100), width=1, style=QtCore.Qt.PenStyle.DashLine),
                movable=False,
            )
            self.addItem(self._threshold_line)

        self._center_line = pg.InfiniteLine(
            angle=90,
            pos=0.0,
            pen=pg.mkPen((100, 100, 100), width=1, style=QtCore.Qt.PenStyle.DashLine),
            movable=False,
        )
        self.addItem(self._center_line)

        pin_rgb = (rgb[0], rgb[1], rgb[2])
        self._pin_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen(pin_rgb, width=2),
            movable=False,
        )
        self._pin_line.setVisible(False)
        self.addItem(self._pin_line)
        self._pin_dot = pg.ScatterPlotItem(
            pen=pg.mkPen("k", width=1),
            brush=pg.mkBrush(*pin_rgb, 255),
            size=10,
            pxMode=True,
        )
        self._pin_dot.setVisible(False)
        self.addItem(self._pin_dot)

        self._window_event_indices: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._on_click: Callable[[int], None] | None = None
        self._scatter.sigClicked.connect(self._handle_click)

    @property
    def metric(self) -> str:
        return self._metric

    def set_click_handler(self, handler: Callable[[int], None]) -> None:
        self._on_click = handler

    def update_window(
        self,
        time_start: float,
        time_end: float,
        events_time: NDArray[np.float64],
        events_metric: NDArray[np.float32],
        time_offset: float,
        global_event_indices: NDArray[np.int64],
    ) -> None:
        self._window_event_indices = np.asarray(global_event_indices, dtype=np.int64)
        if events_time.size == 0:
            self._scatter.setData(x=[], y=[], data=[])
            return
        x = events_time - time_offset
        y = self._display_values(events_metric)
        self._scatter.setData(x=x, y=y, data=np.arange(x.shape[0], dtype=np.int64))

    def update_pinned_event(
        self,
        *,
        relative_time: float | None,
        metric_value: float | None,
    ) -> None:
        if relative_time is None or metric_value is None:
            self._pin_line.setVisible(False)
            self._pin_dot.setVisible(False)
            return
        self._pin_line.setPos(relative_time)
        self._pin_line.setVisible(True)
        disp = -np.log10(metric_value) if self._metric == "event_spike_prob" else metric_value
        self._pin_dot.setData(x=[relative_time], y=[float(disp)])
        self._pin_dot.setVisible(True)

    def _display_values(self, raw: NDArray[np.float32]) -> NDArray[np.float32]:
        if self._metric == "event_spike_prob":
            safe = np.maximum(raw, 1e-12, dtype=np.float32)
            return np.asarray(-np.log10(safe), dtype=np.float32)
        return np.asarray(raw, dtype=np.float32)

    def _handle_click(self, _scatter: pg.ScatterPlotItem, points: list[Any]) -> None:
        if not points or self._on_click is None:
            return
        spot = points[0]
        try:
            local_idx = int(spot.data())
        except (TypeError, ValueError):
            return
        if not 0 <= local_idx < self._window_event_indices.shape[0]:
            return
        self._on_click(int(self._window_event_indices[local_idx]))


# ---------------------------------------------------------------------------
# Slice column (right-hand)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSlice:
    """One per-cell row payload pushed by the viewer per tick."""

    cell_id: int
    place_field_norm: NDArray[np.float32]
    hpd: float
    kl: float
    spike_prob: float
    n_spikes: int
    is_pinned: bool


@dataclass
class _PerCellRow:
    """One per-cell plot row inside the slice panel."""

    container: QtWidgets.QWidget
    header: QtWidgets.QLabel
    plot: pg.PlotWidget
    cell_curve: pg.PlotDataItem
    predictive_curve: pg.PlotDataItem
    true_position_line: pg.InfiniteLine


def _pin_slice_axes(plot: pg.PlotWidget, position_bins: NDArray[np.float64]) -> None:
    """Hard-pin a slice subplot's x and y axes.

    All curves in the slice column are peak-normalized to 1, so the
    y-axis range is known a priori. ``setLimits`` clamps the
    viewbox to the same window, so subsequent ``addItem``/``setData``
    calls cannot nudge the visible range during scrolling.
    """
    vb = plot.getViewBox()
    vb.disableAutoRange()
    vb.setXRange(float(position_bins[0]), float(position_bins[-1]), padding=0)
    vb.setYRange(_SLICE_Y_MIN, _SLICE_Y_MAX, padding=0)
    vb.setLimits(
        xMin=float(position_bins[0]),
        xMax=float(position_bins[-1]),
        yMin=_SLICE_Y_MIN,
        yMax=_SLICE_Y_MAX,
    )


def _make_slice_subplot(
    *,
    title: str | None,
    position_bins: NDArray[np.float64],
    height: int,
) -> pg.PlotWidget:
    plot = pg.PlotWidget()
    plot.setBackground("w")
    plot.setMenuEnabled(False)
    plot.setMouseEnabled(x=False, y=False)
    plot.setLabel("bottom", "Position (cm)")
    plot.setLabel("left", "Density")
    plot.getAxis("bottom").enableAutoSIPrefix(False)
    plot.getAxis("left").enableAutoSIPrefix(False)
    if title is not None:
        plot.getPlotItem().setTitle(title)
    plot.setMinimumHeight(height)
    _pin_slice_axes(plot, position_bins)
    return plot


class SlicePanel(QtWidgets.QWidget):
    """Stacked slice plots: population likelihood + per-cell likelihoods.

    Layout (top to bottom):

    1. Legend label.
    2. Population-likelihood plot (orange line + thin blue
       predictive overlay + dashed true-position).
    3. Pool of per-cell-likelihood rows; pre-allocated and hidden
       when not in the current bin so the column's vertical layout
       is fixed across ticks.
    4. Truncation label (``+K more cells in this bin``).
    5. Live readout (``t = …``, ``predictive(x_true) = …``).
    6. Pinned-event annotation.

    Every plot has the predictive distribution overlaid (peak-
    normalized to 1) so each row is a direct shape comparison
    against the predictive prior — which is what the HPD/KL
    diagnostics actually quantify.

    Hot path: ``update_for_index`` (per UI tick) + the viewer's
    ``set_per_cell_slices`` call. Both are sub-millisecond at typical
    bin sizes.
    """

    def __init__(
        self,
        *,
        position_bins: NDArray[np.float64],
        n_states: int,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._position_bins = np.asarray(position_bins, dtype=np.float64)
        self._n_pos = int(self._position_bins.shape[0])
        self._n_states = max(1, int(n_states))
        self._zero_curve = np.zeros(self._n_pos, dtype=np.float32)

        self._per_cell_visible = True

        # Pre-rendered predictive curve, peak-normalized; same array is
        # passed by reference into every plot's ``predictive_curve``
        # via ``setData``, so updating it once per tick refreshes all
        # plots.
        self._predictive_norm = self._zero_curve.copy()

        self._likelihood_plot = _make_slice_subplot(
            title="Population likelihood",
            position_bins=self._position_bins,
            height=140,
        )
        self._lik_predictive_curve = pg.PlotDataItem(
            self._position_bins,
            self._zero_curve,
            pen=pg.mkPen(*_STATE_POSTERIOR_RGB[0], 230, width=2),
        )
        self._likelihood_plot.addItem(self._lik_predictive_curve)

        self._lik_top_curves: list[pg.PlotDataItem] = []
        for s in range(self._n_states):
            lik_rgb = _STATE_LIKELIHOOD_RGB[s % len(_STATE_LIKELIHOOD_RGB)]
            top = pg.PlotDataItem(
                self._position_bins,
                self._zero_curve,
                pen=pg.mkPen(*lik_rgb, 240, width=3),
            )
            self._likelihood_plot.addItem(top)
            self._lik_top_curves.append(top)

        self._lik_true_position_line = pg.InfiniteLine(
            angle=90, movable=False, pen=_TRUE_POSITION_PEN
        )
        self._likelihood_plot.addItem(self._lik_true_position_line)
        # Re-pin axes after every ``addItem`` so pyqtgraph's
        # auto-range hooks cannot nudge the viewbox.
        _pin_slice_axes(self._likelihood_plot, self._position_bins)

        self._per_cell_rows: list[_PerCellRow] = []
        self._n_active_per_cell_rows = 0

        self._truncation_label = QtWidgets.QLabel("")
        self._truncation_label.setStyleSheet(
            "QLabel { color: #777; font-style: italic; padding: 1px 6px; }"
        )
        self._truncation_label.setVisible(False)

        self._legend_label = QtWidgets.QLabel(self._build_legend_html())
        self._legend_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self._legend_label.setAutoFillBackground(True)
        self._legend_label.setStyleSheet(_SLICE_LEGEND_STYLE)
        self._legend_label.setWordWrap(True)

        self._readout_label = QtWidgets.QLabel("")
        self._readout_label.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._readout_label.setAutoFillBackground(True)
        self._readout_label.setStyleSheet(_SLICE_READOUT_STYLE)

        self._annotation = QtWidgets.QLabel("")
        self._annotation.setTextFormat(QtCore.Qt.TextFormat.PlainText)
        self._annotation.setAutoFillBackground(True)
        self._annotation.setStyleSheet(_SLICE_ANNOTATION_STYLE)
        self._annotation.setVisible(False)

        self._per_cell_container = QtWidgets.QWidget()
        self._per_cell_layout = QtWidgets.QVBoxLayout(self._per_cell_container)
        self._per_cell_layout.setContentsMargins(0, 0, 0, 0)
        self._per_cell_layout.setSpacing(2)
        # Pre-allocate the full row pool up front so the slice column's
        # vertical layout is fixed from frame 0.
        for _ in range(MAX_PER_CELL_PLOTS):
            self._ensure_row(len(self._per_cell_rows))

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(4)
        outer.addWidget(self._legend_label)
        outer.addWidget(self._likelihood_plot, stretch=2)
        outer.addWidget(self._per_cell_container, stretch=4)
        outer.addWidget(self._truncation_label)
        outer.addWidget(self._readout_label)
        outer.addWidget(self._annotation)

        self._buffer_slice: slice | None = None
        self._buffer_post: NDArray[np.float32] | None = None
        self._buffer_lik: NDArray[np.float32] | None = None
        self._row_provider: (
            Callable[[int], tuple[NDArray[np.float32], NDArray[np.float32]]] | None
        ) = None

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    def _build_legend_html(self) -> str:
        post_rgb = _STATE_POSTERIOR_RGB[0]
        lik_rgb = _STATE_LIKELIHOOD_RGB[0]
        parts = [
            f"<span style='color:rgb({lik_rgb[0]},{lik_rgb[1]},{lik_rgb[2]});"
            "font-size:14pt'>━</span> Likelihood",
            f"<span style='color:rgb({post_rgb[0]},{post_rgb[1]},{post_rgb[2]});"
            "font-size:14pt'>━</span> Predictive",
            "<span style='color:rgb(50,50,50);font-size:14pt'>┄</span> True position",
        ]
        return " &nbsp;&nbsp; ".join(parts)

    # ------------------------------------------------------------------
    # Buffer + per-tick updates
    # ------------------------------------------------------------------

    def set_window_buffer(
        self,
        sl: slice,
        post: NDArray[np.float32],
        lik: NDArray[np.float32],
    ) -> None:
        self._buffer_slice = sl
        self._buffer_post = post
        self._buffer_lik = lik

    def set_row_provider(
        self,
        provider: Callable[[int], tuple[NDArray[np.float32], NDArray[np.float32]]] | None,
    ) -> None:
        """Install a single-row reader for use when ``t_idx`` is outside the buffer.

        ``provider(t_idx)`` should return a ``(post_row, lik_row)``
        pair already NaN-cleaned and (for ``lik_row``) exponentiated
        from the cache's ``log_likelihood``, matching what the worker
        thread would have produced for the buffered case.
        """
        self._row_provider = provider

    def update_for_index(self, t_idx: int, true_position: float) -> None:
        sl = self._buffer_slice
        post = self._buffer_post
        lik = self._buffer_lik
        if sl is not None and post is not None and lik is not None and sl.start <= t_idx < sl.stop:
            local_idx = t_idx - sl.start
            post_row = post[local_idx]
            lik_row = lik[local_idx]
        elif self._row_provider is not None:
            # Buffer doesn't cover ``t_idx`` (the user has scrubbed
            # past the loaded window). Fall back to a single-row read
            # so the slice keeps animating until the next async load
            # commits.
            post_row, lik_row = self._row_provider(t_idx)
        else:
            return
        if self._n_states > 1:
            post_row_collapsed = post_row.reshape(self._n_states, self._n_pos).sum(axis=0)
            lik_rs = lik_row.reshape(self._n_states, self._n_pos)
        else:
            post_row_collapsed = post_row
            lik_rs = lik_row[None, :]

        peak = float(post_row_collapsed.max())
        if peak > 0:
            self._predictive_norm = (post_row_collapsed / peak).astype(np.float32, copy=False)
        else:
            self._predictive_norm = post_row_collapsed.astype(np.float32, copy=False)

        self._lik_predictive_curve.setData(self._position_bins, self._predictive_norm)
        for s in range(self._n_states):
            row = lik_rs[s]
            row_peak = float(row.max())
            row_norm = (row / row_peak).astype(np.float32, copy=False) if row_peak > 0 else row
            self._lik_top_curves[s].setData(self._position_bins, row_norm)
        self._lik_true_position_line.setPos(true_position)

        for i in range(self._n_active_per_cell_rows):
            row = self._per_cell_rows[i]
            row.predictive_curve.setData(self._position_bins, self._predictive_norm)
            row.true_position_line.setPos(true_position)

    def set_live_readout(self, text: str | None) -> None:
        self._readout_label.setText(text or "")

    # ------------------------------------------------------------------
    # Per-cell rows
    # ------------------------------------------------------------------

    def set_per_cell_visible(self, visible: bool) -> None:
        self._per_cell_visible = bool(visible)
        for i, row in enumerate(self._per_cell_rows):
            row.container.setVisible(self._per_cell_visible and i < self._n_active_per_cell_rows)

    def _ensure_row(self, index: int) -> _PerCellRow:
        while len(self._per_cell_rows) <= index:
            self._per_cell_rows.append(self._build_row())
            self._per_cell_layout.addWidget(self._per_cell_rows[-1].container)
        return self._per_cell_rows[index]

    def _build_row(self) -> _PerCellRow:
        plot = _make_slice_subplot(title=None, position_bins=self._position_bins, height=70)
        cell_curve = pg.PlotDataItem(
            self._position_bins, self._zero_curve, pen=pg.mkPen((44, 160, 44), width=3)
        )
        predictive_curve = pg.PlotDataItem(
            self._position_bins,
            self._predictive_norm,
            pen=pg.mkPen(*_STATE_POSTERIOR_RGB[0], 200, width=2),
        )
        plot.addItem(cell_curve)
        plot.addItem(predictive_curve)
        true_position_line = pg.InfiniteLine(angle=90, movable=False, pen=_TRUE_POSITION_PEN)
        plot.addItem(true_position_line)
        # Re-pin the axes after every ``addItem`` so the y-range
        # cannot get nudged by pyqtgraph's auto-range hooks.
        _pin_slice_axes(plot, self._position_bins)

        header = QtWidgets.QLabel("")
        header.setAutoFillBackground(True)
        header.setStyleSheet(_SLICE_CELL_HEADER_STYLE)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(plot, stretch=1)
        # Hidden rows must keep their footprint so the slice column
        # doesn't reflow as the bin's cell membership changes.
        size_policy = container.sizePolicy()
        size_policy.setRetainSizeWhenHidden(True)
        container.setSizePolicy(size_policy)
        container.setVisible(False)

        return _PerCellRow(
            container=container,
            header=header,
            plot=plot,
            cell_curve=cell_curve,
            predictive_curve=predictive_curve,
            true_position_line=true_position_line,
        )

    def set_per_cell_slices(self, slices: list[CellSlice], total_in_bin: int | None = None) -> None:
        n = len(slices)
        for i, cs in enumerate(slices):
            row = self._ensure_row(i)
            row.cell_curve.setData(self._position_bins, cs.place_field_norm)
            rgb = _PER_CELL_PALETTE[cs.cell_id % len(_PER_CELL_PALETTE)]
            row.cell_curve.setPen(pg.mkPen(*rgb, 230, width=3))
            n_spikes_str = f"  ({cs.n_spikes} spikes)" if cs.n_spikes > 1 else ""
            pin_str = "  ★" if cs.is_pinned else ""
            row.header.setText(
                f"Cell {cs.cell_id:>3d}{pin_str}   "
                f"HPD={cs.hpd:.3f}  KL={cs.kl:.3f}  p={cs.spike_prob:.3g}{n_spikes_str}"
            )
            row.header.setStyleSheet(
                _SLICE_CELL_HEADER_PINNED_STYLE if cs.is_pinned else _SLICE_CELL_HEADER_STYLE
            )
            row.predictive_curve.setData(self._position_bins, self._predictive_norm)
            row.container.setVisible(self._per_cell_visible)
        for i in range(n, self._n_active_per_cell_rows):
            self._per_cell_rows[i].container.setVisible(False)
        self._n_active_per_cell_rows = n

        total = total_in_bin if total_in_bin is not None else n
        extra = max(0, total - n)
        if extra > 0:
            self._truncation_label.setText(f"(+{extra} more cells in this bin)")
            self._truncation_label.setVisible(True)
        else:
            self._truncation_label.setVisible(False)

    def update_pinned_event(
        self,
        *,
        place_field_row: NDArray[np.float32] | None,
        annotation: str | None,
    ) -> None:
        """Show / hide the pinned-event annotation label.

        The pinned-cell highlight on the per-cell row is driven by
        the ``is_pinned`` flag on each ``CellSlice``; this method
        only manages the annotation label below the rows.
        """
        del place_field_row  # honored via the per-cell row highlight
        if annotation is None:
            self._annotation.setVisible(False)
            self._annotation.setText("")
            return
        self._annotation.setText(f"Pinned: {annotation}")
        self._annotation.setVisible(True)

    def is_pin_displayed(self) -> bool:
        return bool(self._annotation.text())
