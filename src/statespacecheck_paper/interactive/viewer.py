"""Figure 4 viewer skeleton (pyqtgraph + PySide6).

Phase 3 of the plan: build the foundation needed to run the performance
benchmark gate. Three panels:

- ``PosteriorPanel``  — predictive posterior heatmap.
- ``LikelihoodPanel`` — log-likelihood heatmap.
- ``RasterPanel``     — spike raster (sorted by place-field peak).

A center-time slider drives the visible window; a ``QThreadPool``
worker loads chunks from the Zarr cache so the UI thread never blocks
on disk. No metric panels, no 1D slice panel, no click handling yet —
those land in subsequent phases.

Run with ::

    python -m statespacecheck_paper.interactive.viewer \\
        --cache-dir data/cache --model continuous

The module imports lazily — running anything under ``__main__`` requires
the ``[interactive]`` extras (``PySide6``, ``pyqtgraph``).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PySide6 import QtCore, QtGui, QtWidgets

from .data_source import Figure4DataSource, ModelName

# Plan defaults.
DEFAULT_WINDOW_SECONDS = 2.0
MIN_WINDOW_SECONDS = 0.1
MAX_WINDOW_SECONDS = 60.0
SLIDER_RESOLUTION = 100_000  # subdivides the full session into this many ticks

# Window-width slider works in log-space so a single slider position can
# resolve both 0.1 s and 60 s endpoints with reasonable granularity.
WINDOW_SLIDER_RESOLUTION = 1000

# Reset shortcut targets (matches the Figure 4a context window from
# scripts/generate_figure04.py via ``index 190000`` at the decoder
# sampling rate of 500 Hz, ~20 s wide).
RESET_WINDOW_SECONDS = 20.0

# Auto-scroll defaults.
AUTOSCROLL_TICK_HZ = 30.0
AUTOSCROLL_RATE_REALTIME = 1.0  # advance 1 second of session per second of wall time
AUTOSCROLL_SPEED_OPTIONS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

# Maximum rows in the per-cell live-readout block before truncation.
MAX_PER_CELL_READOUT_ROWS = 6


def _nearest_index(sorted_arr: NDArray[np.float64], value: float) -> int:
    """Return the index of the entry in ``sorted_arr`` closest to ``value``.

    ``sorted_arr`` must be monotonically increasing. Used by the
    per-tick live-readout path (avoids a ``np.argmin`` allocation).
    """
    n = sorted_arr.size
    if n == 0:
        raise ValueError("Empty array")
    i = int(np.searchsorted(sorted_arr, value))
    if i <= 0:
        return 0
    if i >= n:
        return n - 1
    if abs(float(sorted_arr[i - 1]) - value) <= abs(float(sorted_arr[i]) - value):
        return i - 1
    return i


@dataclass(frozen=True)
class ViewState:
    """Immutable snapshot of the viewer's current request.

    Each load worker receives one of these and reports back with its
    ``request_id``; stale results (older request_id than the current
    state) are dropped on arrival so a fast scrub never displays the
    output of an in-flight request that has been superseded.
    """

    request_id: int
    t_center: float
    t_width: float


class _LoadSignals(QtCore.QObject):
    """Bridge object owning the result signal for the load worker.

    ``QRunnable`` cannot inherit from ``QObject``; the standard
    workaround is to attach a ``QObject`` member that owns the signals.
    """

    finished = QtCore.Signal(int, slice, object, object)  # request_id, slice, post, lik


class _WindowLoadWorker(QtCore.QRunnable):
    """Pull one window's posterior + log-likelihood from the cache.

    Runs on a ``QThreadPool`` worker thread; emits the result on the
    main thread via the bridge ``QObject``'s signal.
    """

    def __init__(
        self,
        data_source: Figure4DataSource,
        state: ViewState,
        signals: _LoadSignals,
    ) -> None:
        super().__init__()
        self._ds = data_source
        self._state = state
        self._signals = signals
        self.setAutoDelete(True)

    @QtCore.Slot()
    def run(self) -> None:  # noqa: D401 - QRunnable contract
        sl = self._ds.window_indices(self._state.t_center, self._state.t_width)
        post = self._ds.load_posterior(sl)
        loglik = self._ds.load_likelihood(sl)

        # Replace NaN at non-interior bins with zero so the heatmap's
        # autoLevels pass does not see all-NaN rows. Done here on the
        # worker thread so the main thread never sees NaNs.
        if post.size:
            np.nan_to_num(post, copy=False, nan=0.0)
        if loglik.size:
            np.nan_to_num(loglik, copy=False, nan=-np.inf)

        # Convert log-likelihood -> normalized linear likelihood here on
        # the worker thread so the main thread does not pay np.exp on
        # every committed update. Subtract the per-row max first to
        # avoid float32 overflow; this only affects the absolute scale,
        # which the panel pins after the first autoLevels pass.
        if loglik.size:
            row_max = loglik.max(axis=1, keepdims=True)
            row_max = np.where(np.isfinite(row_max), row_max, 0.0)
            lik = np.exp(loglik - row_max, dtype=np.float32)
            np.nan_to_num(lik, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            lik = loglik
        self._signals.finished.emit(self._state.request_id, sl, post, lik)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


class _BaseHeatmapPanel(pg.PlotWidget):
    """Common posterior / likelihood heatmap behavior.

    Subclasses provide ``_array_for_window`` which decides whether to
    show the raw posterior or its log; the rest of the panel is
    identical: an ``ImageItem`` with one row per time sample, the time
    axis (relative to window start in seconds), and the position axis.
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
        # ``(x_index=i, y_index=j)``. Our window arrays come in as
        # ``(n_time, n_state_bins)``, which we want displayed with
        # time on x and position on y — col-major lines up the array's
        # first axis with the time axis without a transpose.
        self._image = pg.ImageItem(axisOrder="col-major")
        self._image.setLookupTable(pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256))
        self.addItem(self._image)

        # Position axis y-extent: full interior grid.
        self._position_bins = np.asarray(position_bins, dtype=np.float64)
        # ImageItem rect is set per update; remember the position extent.
        self._y0 = float(self._position_bins[0])
        self._y1 = float(self._position_bins[-1])

        # Auto-level on the first update only; subsequent updates reuse
        # the pinned levels. ``ImageItem.setImage(autoLevels=True)``
        # otherwise runs ``nanmin``/``nanmax`` over the full window on
        # every call, which dominates the main-thread cost when the
        # window is large.
        self._levels: tuple[float, float] | None = None

        # Center-time vertical marker. The time-axis is shifted so
        # x=0 is the current center time (the slice panel's
        # ``t_center``), and this line is always at x=0 so the user
        # can see which column on the heatmap corresponds to the
        # slice panel's curves.
        self._center_line = pg.InfiniteLine(
            angle=90,
            pos=0.0,
            pen=pg.mkPen((100, 100, 100), width=1, style=QtCore.Qt.PenStyle.DashLine),
            movable=False,
        )
        self.addItem(self._center_line)

        # Pinned-event vertical marker on the time axis.
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
        """Update the heatmap for a fresh window.

        Parameters
        ----------
        time_start, time_end : float
            Time bounds of the window in *relative* seconds (start = 0
            at the window's left edge for visual stability under
            scrolling).
        array : np.ndarray, shape (n_time_visible, n_state_bins)
            Already-loaded window data (float32).
        """
        if array.size == 0:
            return
        # The worker has already replaced NaNs from non-interior bins;
        # the array is finite at this point.
        if self._levels is None:
            # First update: let pyqtgraph compute levels and pin them.
            self._image.setImage(array, autoLevels=True, autoDownsample=False)
            levels = self._image.getLevels()
            if levels is not None and len(levels) == 2:
                self._levels = (float(levels[0]), float(levels[1]))
        else:
            self._image.setImage(
                array,
                autoLevels=False,
                levels=self._levels,
                autoDownsample=False,
            )
        # Use the actual data extent for the rect so each pixel sits at
        # its true relative time. The axis range itself is set by the
        # viewer in one place (with the *target* window half-width) so
        # the tick labels do not jitter as ``time_start`` /
        # ``time_end`` shift by sub-millisecond amounts each load.
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
    """Predictive posterior heatmap (state-summed for multi-state models)."""

    def __init__(self, *, position_bins: NDArray[np.float64], n_states: int) -> None:
        # Title: ``predictive`` is the SSM-standard term for
        # ``p(x_t | y_{1:t-1})``; the cache variable is named
        # ``predictive_posterior`` per ``non_local_detector``.
        super().__init__(title="Predictive distribution", position_bins=position_bins)
        self._n_states = max(1, n_states)
        self._n_pos = position_bins.shape[0]

    def update_with_window(
        self,
        time_start: float,
        time_end: float,
        post: NDArray[np.float32],
    ) -> None:
        """Reduce to ``(n_visible, n_pos)`` and forward to the heatmap.

        For ContFrag the cache stores ``(n_visible, n_states * n_pos)``
        with state varying slowest; sum across states to get the
        marginal posterior over position.
        """
        arr = post
        if self._n_states > 1:
            n_pos = self._n_pos
            n_visible = arr.shape[0]
            arr = arr.reshape(n_visible, self._n_states, n_pos).sum(axis=1)
        self.update_window(time_start, time_end, arr.astype(np.float32, copy=False))


class LikelihoodPanel(_BaseHeatmapPanel):
    """Log-likelihood heatmap.

    The cache stores raw log-likelihoods; we exponentiate per-window for
    visual range control. For multi-state models we sum the
    likelihood (not the log) across states, the same convention used by
    the existing ``plot_single_model_diagnostics`` flow at
    ``src/statespacecheck_paper/real_data_plotting.py``.
    """

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
        """Update the heatmap with a pre-exponentiated linear likelihood.

        ``likelihood`` is already on a linear scale (the worker calls
        ``np.exp(loglik - row_max)`` before emitting the result). For
        multi-state classifiers we sum across states to get a marginal
        likelihood over position.
        """
        lik = likelihood
        if self._n_states > 1:
            n_pos = self._n_pos
            n_visible = lik.shape[0]
            lik = lik.reshape(n_visible, self._n_states, n_pos).sum(axis=1)
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
        # Sort cells by their place-field peak (ascending). The rank
        # array maps cell_id -> visual y-position.
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

        # Click + pin support.
        self._window_event_indices: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._on_click: Callable[[int], None] | None = None
        self._scatter.sigClicked.connect(self._handle_click)

        # Center-time vertical marker (matches the heatmap panels).
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
        """Register a callback that takes the global event-row index."""
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
        """Plot spikes whose absolute time is in ``[t_start_abs, t_end_abs]``.

        Parameters
        ----------
        time_start, time_end : float
            Window bounds in relative seconds.
        events_time : np.ndarray, shape (n_window_events,)
            Spike times, absolute (will be shifted by ``time_offset``).
        events_cell_id : np.ndarray, shape (n_window_events,)
            Cell index for each spike.
        time_offset : float
            Subtract from ``events_time`` to convert absolute → relative.
        global_event_indices : np.ndarray, shape (n_window_events,), optional
            Row positions in the full ``Figure4DataSource.events`` frame
            for click-to-recenter. When omitted, clicks are ignored.
        """
        if global_event_indices is None:
            self._window_event_indices = np.empty(0, dtype=np.int64)
        else:
            self._window_event_indices = np.asarray(global_event_indices, dtype=np.int64)

        # The X axis is driven by the master ``PosteriorPanel`` via
        # ``setXLink`` (configured in ``Figure4Viewer._build_panels``),
        # so we only update the scatter data here and leave the range
        # to the viewer's single ``setXRange`` call.
        if events_time.size == 0:
            self._scatter.setData(x=[], y=[], data=[])
            return
        x = events_time - time_offset
        y = self._cell_rank[events_cell_id]
        self._scatter.setData(
            x=x,
            y=y,
            data=np.arange(x.shape[0], dtype=np.int64),
        )

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
        local_idx_obj = spot.data()
        try:
            local_idx = int(local_idx_obj)
        except (TypeError, ValueError):
            return
        if not 0 <= local_idx < self._window_event_indices.shape[0]:
            return
        self._on_click(int(self._window_event_indices[local_idx]))


# ---------------------------------------------------------------------------
# Per-spike metric panel (HPD overlap, KL divergence, -log10 spike prob)
# ---------------------------------------------------------------------------


METRIC_COLORS: dict[str, tuple[int, int, int]] = {
    "event_hpd_overlap": (44, 160, 44),  # green
    "event_kl_divergence": (214, 39, 40),  # red
    "event_spike_prob": (148, 103, 189),  # purple
}
METRIC_TITLES: dict[str, str] = {
    "event_hpd_overlap": "HPD overlap",
    "event_kl_divergence": "KL divergence",
    "event_spike_prob": "-log10(p)",
}


class MetricPanel(pg.PlotWidget):
    """Window-local per-spike scatter for one diagnostic metric.

    Each spike is one dot at ``(time_relative, metric_value)``. Clicks
    on a dot recenter the viewer on the spike's time and pin a marker
    across all panels via the supplied ``on_click`` callback.

    The ``event_spike_prob`` metric is shown as ``-log10(p)`` so
    "worse fit" goes up like the other two diagnostic axes.
    """

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

        # Threshold horizontal line for the two metrics that have one in
        # the existing Figure 4 (HPD overlap = 0.05, spike prob = 0.05
        # which becomes -log10(0.05) ≈ 1.30 on this axis).
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

        # Center-time vertical marker (matches the heatmap panels).
        self._center_line = pg.InfiniteLine(
            angle=90,
            pos=0.0,
            pen=pg.mkPen((100, 100, 100), width=1, style=QtCore.Qt.PenStyle.DashLine),
            movable=False,
        )
        self.addItem(self._center_line)

        # Pinned-event marker: vertical line at the clicked spike's
        # time, plus a single highlighted dot. Both share the metric's
        # color but use a thicker stroke.
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

        # Filled by ``update_window`` so click handlers can map a Qt
        # ``SpotItem`` back to a row index in the events table.
        self._window_event_indices: NDArray[np.int64] = np.empty(0, dtype=np.int64)
        self._on_click: Callable[[int], None] | None = None
        self._scatter.sigClicked.connect(self._handle_click)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    @property
    def metric(self) -> str:
        return self._metric

    def set_click_handler(self, handler: Callable[[int], None]) -> None:
        """Register a callback that takes the global event-row index."""
        self._on_click = handler

    # ------------------------------------------------------------------
    # Per-window updates
    # ------------------------------------------------------------------

    def update_window(
        self,
        time_start: float,
        time_end: float,
        events_time: NDArray[np.float64],
        events_metric: NDArray[np.float32],
        time_offset: float,
        global_event_indices: NDArray[np.int64],
    ) -> None:
        """Plot the in-window events for this metric.

        ``global_event_indices`` is the array of row positions in the
        full ``Figure4DataSource.events`` frame; the click handler uses
        it to map a clicked SpotItem back to the canonical event row.
        """
        self._window_event_indices = np.asarray(global_event_indices, dtype=np.int64)
        # X axis is driven by the master ``PosteriorPanel`` via
        # ``setXLink``; only the scatter data is updated here.
        if events_time.size == 0:
            self._scatter.setData(x=[], y=[], data=[])
            return
        x = events_time - time_offset
        y = self._display_values(events_metric)
        # ``data`` is the per-spot payload pyqtgraph returns on click.
        # We attach the local index so we can map back to the global
        # event-row index without a hash lookup.
        self._scatter.setData(
            x=x,
            y=y,
            data=np.arange(x.shape[0], dtype=np.int64),
        )

    def update_pinned_event(
        self,
        *,
        relative_time: float | None,
        metric_value: float | None,
    ) -> None:
        """Show or hide the pinned-event vertical marker and dot."""
        if relative_time is None or metric_value is None:
            self._pin_line.setVisible(False)
            self._pin_dot.setVisible(False)
            return
        self._pin_line.setPos(relative_time)
        self._pin_line.setVisible(True)
        disp = -np.log10(metric_value) if self._metric == "event_spike_prob" else metric_value
        self._pin_dot.setData(x=[relative_time], y=[float(disp)])
        self._pin_dot.setVisible(True)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _display_values(self, raw: NDArray[np.float32]) -> NDArray[np.float32]:
        if self._metric == "event_spike_prob":
            # Avoid log(0); cache stores values >= 0. Tiny floor is fine.
            safe = np.maximum(raw, 1e-12, dtype=np.float32)
            return np.asarray(-np.log10(safe), dtype=np.float32)
        return np.asarray(raw, dtype=np.float32)

    def _handle_click(self, _scatter: pg.ScatterPlotItem, points: list[Any]) -> None:
        if not points or self._on_click is None:
            return
        spot = points[0]
        local_idx_obj = spot.data()
        try:
            local_idx = int(local_idx_obj)
        except (TypeError, ValueError):
            return
        if not 0 <= local_idx < self._window_event_indices.shape[0]:
            return
        global_idx = int(self._window_event_indices[local_idx])
        self._on_click(global_idx)


# ---------------------------------------------------------------------------
# Slice panel (1D posterior + likelihood at the current center time)
# ---------------------------------------------------------------------------


# Posterior is solid; likelihood is filled-area; they share the position axis.
_LIKELIHOOD_PEN_RGB = (200, 90, 0)
_TRUE_POSITION_PEN = pg.mkPen((50, 50, 50), width=1, style=QtCore.Qt.PenStyle.DashLine)
# Per-state colors (Continuous, Fragmented for ContFrag).
_STATE_POSTERIOR_RGB: tuple[tuple[int, int, int], ...] = (
    (31, 119, 180),  # blue
    (44, 160, 44),  # green
)
_STATE_LIKELIHOOD_RGB: tuple[tuple[int, int, int], ...] = (
    (255, 127, 14),  # orange
    (214, 39, 40),  # red
)
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


@dataclass(frozen=True)
class CellSlice:
    """One row of per-cell information, pushed by the viewer per tick."""

    cell_id: int
    place_field_norm: NDArray[np.float32]
    hpd: float
    kl: float
    spike_prob: float
    n_spikes: int
    is_pinned: bool


# Cap on simultaneously visible per-cell rows; bins with more cells get
# a "(+K more)" indicator below.
MAX_PER_CELL_PLOTS = 6


# Stylesheets shared by the slice-panel labels.
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


@dataclass
class _PerCellRow:
    """One per-cell plot row inside the slice panel.

    Each row owns a header ``QLabel`` (cell + metrics + pinned badge),
    a ``pg.PlotWidget`` containing a colored cell-likelihood curve, an
    overlaid blue predictive curve, and a dashed true-position line.
    The container ``QWidget`` is what gets shown/hidden by the
    visibility toggle and the pool-resize logic.
    """

    container: QtWidgets.QWidget
    header: QtWidgets.QLabel
    plot: pg.PlotWidget
    cell_curve: pg.PlotDataItem
    predictive_curve: pg.PlotDataItem
    true_position_line: pg.InfiniteLine


_SLICE_Y_MIN = 0.0
_SLICE_Y_MAX = 1.05


def _pin_slice_axes(plot: pg.PlotWidget, position_bins: NDArray[np.float64]) -> None:
    """Hard-pin the slice subplot's x and y axes.

    All curves in the slice column are peak-normalized to 1, so the
    y-axis range is known a priori. Auto-range is fully disabled on
    both axes and ``setLimits`` clamps the user's pan/zoom to the
    same window so subsequent ``addItem``/``setData`` calls cannot
    nudge the view.
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
    """Configure a small position-axis plot used in the slice column."""
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

    1. Legend label (color swatches + names).
    2. **Population-likelihood plot**: orange-fill joint likelihood
       overlaid by a thin blue *predictive* line. Always visible.
    3. **Per-cell-likelihood plots**: a pool of up to
       ``MAX_PER_CELL_PLOTS`` rows. When the current bin contains
       spikes from multiple cells (deduped by cell), one row per
       cell shows the cell's normalized place-field overlaid by
       the predictive line. The header label carries that cell's
       ``HPD`` / ``KL`` / ``p`` values; pinned cells get a yellow
       header. Hidden when the bin is empty.
    4. Truncation label (shown when the bin has more than
       ``MAX_PER_CELL_PLOTS`` cells).
    5. Live-readout label (``t = …``, ``predictive(x_true) = …``).
    6. Pinned-event annotation (visible only when a spike is pinned).

    Every plot has the predictive distribution overlaid (peak-
    normalized to 1) so each row is a direct shape comparison
    against the predictive prior, which is what the HPD/KL
    diagnostics measure.

    The hot path is ``update_for_index(t_idx, true_pos)`` and the
    viewer's follow-up ``set_per_cell_slices(slices)`` — both are
    sub-millisecond at typical bin sizes.
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
        # Re-pin axes after every ``addItem`` finishes; pyqtgraph's
        # auto-range can re-engage when items are added even after we
        # explicitly disable it during the subplot's construction.
        _pin_slice_axes(self._likelihood_plot, self._position_bins)

        # Per-cell row pool. Rows are constructed lazily in
        # ``set_per_cell_slices``; the count never shrinks (rows hide
        # rather than getting destroyed) so Qt doesn't churn graphics
        # items as the user scrolls between bins.
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

        # Container for the per-cell rows; we add row widgets to this
        # layout so they sit between the population plot and the
        # truncation label.
        self._per_cell_container = QtWidgets.QWidget()
        self._per_cell_layout = QtWidgets.QVBoxLayout(self._per_cell_container)
        self._per_cell_layout.setContentsMargins(0, 0, 0, 0)
        self._per_cell_layout.setSpacing(2)
        # Pre-allocate the full row pool up front so the slice column's
        # vertical layout is fixed from frame 0 (combined with each row's
        # ``retainSizeWhenHidden`` policy below). Without this the
        # container would grow each time a new ``cell_id`` first appears
        # at runtime, and the population plot above would visibly
        # resize as the user scrolled.
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

        # Window-buffer state pushed by the viewer on every commit.
        self._buffer_slice: slice | None = None
        self._buffer_post: NDArray[np.float32] | None = None
        self._buffer_lik: NDArray[np.float32] | None = None

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

    def update_for_index(self, t_idx: int, true_position: float) -> None:
        """Update the population plot + the predictive overlay every plot uses."""
        sl = self._buffer_slice
        post = self._buffer_post
        lik = self._buffer_lik
        if sl is None or post is None or lik is None:
            return
        if not (sl.start <= t_idx < sl.stop):
            return
        local_idx = t_idx - sl.start
        post_row = post[local_idx]
        lik_row = lik[local_idx]
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

        # Population likelihood + predictive overlay.
        self._lik_predictive_curve.setData(self._position_bins, self._predictive_norm)
        for s in range(self._n_states):
            row = lik_rs[s]
            row_peak = float(row.max())
            row_norm = (row / row_peak).astype(np.float32, copy=False) if row_peak > 0 else row
            self._lik_top_curves[s].setData(self._position_bins, row_norm)
        self._lik_true_position_line.setPos(true_position)

        # Per-cell rows: refresh predictive overlay + true-position
        # line on each row that's currently active. The cell-likelihood
        # curve is set by ``set_per_cell_slices`` which only fires when
        # the bin's cell membership changes; this hot path just pokes
        # the shared predictive into each row.
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
        # Hidden rows must keep their footprint, otherwise the slice
        # column's QVBoxLayout collapses the gap and the population-
        # likelihood plot stretches to fill it -- which is what made
        # the plots appear to "resize" during auto-scroll.
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
        """Refresh the per-cell rows for the cells in the current bin.

        Parameters
        ----------
        slices : list[CellSlice]
            One entry per cell to display (already capped to
            ``MAX_PER_CELL_PLOTS`` and deduped by ``cell_id``).
        total_in_bin : int, optional
            Total cells in the bin before truncation; if it exceeds
            ``len(slices)``, a "(+K more)" label appears below the
            rows. Defaults to ``len(slices)``.
        """
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


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class Figure4Viewer(QtWidgets.QMainWindow):
    """Top-level window owning the panels and view state."""

    def __init__(
        self,
        data_source: Figure4DataSource,
        *,
        parent: QtWidgets.QWidget | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        super().__init__(parent)
        self._ds = data_source
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

        self.setWindowTitle(f"Figure 4 viewer — {data_source.model}")
        self.resize(1200, 900)

        self._t_min = float(data_source.time[0])
        self._t_max = float(data_source.time[-1])
        self._window_seconds = DEFAULT_WINDOW_SECONDS
        self._t_center = 0.5 * (self._t_min + self._t_max)
        self._next_request_id = 0
        self._latest_committed_request_id = -1
        # At most one load worker is in flight at a time. While in
        # flight, ``_pending_dispatch`` records that a newer request
        # arrived and should fire when the current one completes.
        self._inflight_request_id: int | None = None
        self._pending_dispatch = False
        # Pinned-event state. ``None`` when no event is pinned; when
        # set it is a global row index into ``data_source.events`` so
        # the panels can fetch the spike's metrics on demand.
        self._pinned_event_row: int | None = None
        # Auto-scroll (play/pause) state.
        self._autoscroll_rate = AUTOSCROLL_RATE_REALTIME
        self._autoscroll_timer: QtCore.QTimer | None = None

        self._wire_load_worker()
        self._build_central_widget()
        self._wire_keyboard_shortcuts()

        # Trigger initial load.
        self._dispatch_load()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_central_widget(self) -> None:
        """Build (or rebuild) panels + controls and install as central widget.

        Used both at construction and after a model swap. The bottom
        controls hold the persistent UI state (slider position, alpha,
        play state, model choice); the new ``QWidget`` reparents the
        existing widgets and Qt deletes the old central tree.
        """
        self._build_panels()
        self._build_controls()
        self._wire_click_handlers()

    def _build_panels(self) -> None:
        ds = self._ds
        # Heatmap and slice panels use the full per-state position grid
        # (256 bins, including non-interior edges that the worker
        # NaN-cleans to zero). ``position_bins`` (248 interior) is only
        # used for things that operate on the interior subset, like
        # place-field peaks.
        self.posterior_panel = PosteriorPanel(
            position_bins=ds.position_grid_full,
            n_states=ds.n_states,
        )
        self.likelihood_panel = LikelihoodPanel(
            position_bins=ds.position_grid_full,
            n_states=ds.n_states,
        )
        self.raster_panel = RasterPanel(
            n_cells=ds.n_cells,
            place_field_peaks=ds.place_field_peaks,
        )
        # Three diagnostic-metric panels; the thresholds match Figure 4
        # defaults at scripts/generate_figure04.py.
        self.metric_panels: dict[str, MetricPanel] = {
            "event_hpd_overlap": MetricPanel(metric="event_hpd_overlap", threshold=0.05),
            "event_kl_divergence": MetricPanel(metric="event_kl_divergence", threshold=None),
            "event_spike_prob": MetricPanel(metric="event_spike_prob", threshold=0.05),
        }
        self.slice_panel = SlicePanel(
            position_bins=ds.position_grid_full,
            n_states=ds.n_states,
        )
        # Link x-axes so any zoom/range change propagates across the
        # time-axis stack (heatmaps + raster + metric panels).
        x_linked: list[pg.PlotWidget] = [self.likelihood_panel, self.raster_panel]
        x_linked.extend(self.metric_panels.values())
        for panel in x_linked:
            panel.setXLink(self.posterior_panel)

    def _build_controls(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # Horizontal split: time-axis panels (left, ~70%) | slice column
        # (right, ~30%). The right column wraps the slice panel in a
        # vertical layout with a trailing spacer so the slice does not
        # stretch the full window height — the curves are easier to
        # read at the same vertical extent as the posterior heatmap.
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        time_axis = QtWidgets.QWidget()
        time_axis_layout = QtWidgets.QVBoxLayout(time_axis)
        time_axis_layout.setContentsMargins(0, 0, 0, 0)
        time_axis_layout.setSpacing(2)
        time_axis_layout.addWidget(self.posterior_panel, stretch=2)
        time_axis_layout.addWidget(self.likelihood_panel, stretch=2)
        time_axis_layout.addWidget(self.raster_panel, stretch=1)
        for metric in (
            "event_hpd_overlap",
            "event_kl_divergence",
            "event_spike_prob",
        ):
            time_axis_layout.addWidget(self.metric_panels[metric], stretch=1)

        slice_column = QtWidgets.QWidget()
        slice_column_layout = QtWidgets.QVBoxLayout(slice_column)
        slice_column_layout.setContentsMargins(0, 0, 0, 0)
        slice_column_layout.setSpacing(0)
        # Match the posterior panel's stretch (2 of 8 units in the
        # time-axis stack) so the slice's vertical extent lines up
        # with the posterior heatmap above.
        slice_column_layout.addWidget(self.slice_panel, stretch=2)
        slice_column_layout.addStretch(stretch=6)

        split.addWidget(time_axis)
        split.addWidget(slice_column)
        split.setStretchFactor(0, 7)
        split.setStretchFactor(1, 3)
        outer.addWidget(split, stretch=1)

        controls = QtWidgets.QWidget(central)
        controls_layout = QtWidgets.QHBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)

        controls_layout.addWidget(QtWidgets.QLabel("Center time:"))
        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setRange(0, SLIDER_RESOLUTION)
        self._slider.setValue(self._slider_value_for(self._t_center))
        self._slider.valueChanged.connect(self._on_slider_changed)
        controls_layout.addWidget(self._slider, stretch=1)

        self._time_label = QtWidgets.QLabel(self._format_time_label())
        self._time_label.setMinimumWidth(220)
        controls_layout.addWidget(self._time_label)

        controls_layout.addSpacing(12)
        controls_layout.addWidget(QtWidgets.QLabel("Window:"))
        self._window_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._window_slider.setRange(0, WINDOW_SLIDER_RESOLUTION)
        self._window_slider.setValue(self._window_slider_value_for(self._window_seconds))
        self._window_slider.setMaximumWidth(140)
        self._window_slider.valueChanged.connect(self._on_window_slider_changed)
        controls_layout.addWidget(self._window_slider)
        self._window_label = QtWidgets.QLabel(self._format_window_label())
        self._window_label.setMinimumWidth(70)
        controls_layout.addWidget(self._window_label)

        controls_layout.addSpacing(12)
        self._per_cell_checkbox = QtWidgets.QCheckBox("Per-cell rows")
        self._per_cell_checkbox.setToolTip(
            "Show a per-cell likelihood plot for each cell that fired "
            "in the current bin (with the predictive distribution "
            "overlaid)."
        )
        self._per_cell_checkbox.setChecked(True)
        self._per_cell_checkbox.toggled.connect(self._on_per_cell_toggled)
        controls_layout.addWidget(self._per_cell_checkbox)

        controls_layout.addSpacing(12)
        self._play_button = QtWidgets.QToolButton()
        self._play_button.setText("▶")
        self._play_button.setToolTip("Play / pause auto-scroll (Space)")
        self._play_button.setCheckable(True)
        self._play_button.toggled.connect(self._on_play_toggled)
        controls_layout.addWidget(self._play_button)

        self._speed_combo = QtWidgets.QComboBox()
        self._speed_combo.setToolTip(
            "Auto-scroll speed (×realtime). Shortcuts: , faster, . slower."
        )
        for speed in AUTOSCROLL_SPEED_OPTIONS:
            self._speed_combo.addItem(self._format_speed(speed), userData=speed)
        self._speed_combo.setCurrentIndex(self._speed_combo_default_index())
        self._speed_combo.currentIndexChanged.connect(self._on_speed_combo_changed)
        controls_layout.addWidget(self._speed_combo)

        controls_layout.addSpacing(12)
        controls_layout.addWidget(QtWidgets.QLabel("Model:"))
        self._model_combo = QtWidgets.QComboBox()
        self._model_combo.addItems(["continuous", "contfrag"])
        self._model_combo.setCurrentText(self._ds.model)
        self._model_combo.currentTextChanged.connect(self._on_model_changed)
        # Disabled when the cache directory wasn't provided (e.g. tests
        # that construct with a single model in tmp_path).
        self._model_combo.setEnabled(self._cache_dir is not None)
        controls_layout.addWidget(self._model_combo)

        outer.addWidget(controls)

    def _wire_click_handlers(self) -> None:
        """Connect raster + metric ``sigClicked`` to the pin-and-recenter path."""
        self.raster_panel.set_click_handler(self._handle_event_click)
        for panel in self.metric_panels.values():
            panel.set_click_handler(self._handle_event_click)

    def _wire_keyboard_shortcuts(self) -> None:
        """Bind the keyboard shortcuts spec'd in the plan.

        - ``←`` / ``→``         : step center by one decoder time bin.
        - ``Shift+←`` / ``Shift+→``: step by one window-width.
        - ``Space``              : play / pause auto-scroll.
        - ``M``                  : toggle model (Continuous ↔ ContFrag).
        - ``[`` / ``]``          : shrink / grow window width.
        - ``R``                  : reset to a 20 s context window centered
                                    near the Figure 4a default.
        """

        def add(seq: str, slot: Callable[[], None]) -> None:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(seq), self)
            shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(slot)

        add("Right", lambda: self._step_center_by_indices(1))
        add("Left", lambda: self._step_center_by_indices(-1))
        add("Shift+Right", lambda: self._step_center_by_seconds(self._window_seconds))
        add("Shift+Left", lambda: self._step_center_by_seconds(-self._window_seconds))
        add("Space", self._toggle_play)
        add("M", self._toggle_model)
        add("[", lambda: self._scale_window(0.5))
        add("]", lambda: self._scale_window(2.0))
        add("R", self._reset_view)
        # ``,`` and ``.`` (the same keys as ``<`` / ``>`` without
        # Shift) step the auto-scroll speed up / down through the
        # preset list.
        add(",", lambda: self._step_speed(-1))
        add(".", lambda: self._step_speed(+1))

    def _wire_load_worker(self) -> None:
        self._thread_pool = QtCore.QThreadPool.globalInstance()
        self._load_signals = _LoadSignals()
        self._load_signals.finished.connect(self._on_window_loaded)

        # Coalesce slider changes into a single load every ~16 ms.
        self._load_timer = QtCore.QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.setInterval(16)
        self._load_timer.timeout.connect(self._dispatch_load)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    @QtCore.Slot(int)
    def _on_slider_changed(self, value: int) -> None:
        self._t_center = self._time_for_slider(value)
        self._time_label.setText(self._format_time_label())
        # Manual scrolling unpins any previously pinned event.
        if self._pinned_event_row is not None:
            self._set_pinned_event(None)
        # Animate the slice panel immediately — this is the per-tick
        # path the user perceives during scrubbing. It is a single
        # array index into the in-RAM ring buffer, so it is sub-ms.
        self._update_slice_panel_at_center()
        # Re-arm the debounce timer for the heavier window load.
        self._load_timer.start()

    @QtCore.Slot(bool)
    def _on_per_cell_toggled(self, checked: bool) -> None:
        self.slice_panel.set_per_cell_visible(checked)

    @QtCore.Slot(int)
    def _on_window_slider_changed(self, value: int) -> None:
        self._set_window_seconds(self._window_seconds_for(value))

    @QtCore.Slot(bool)
    def _on_play_toggled(self, on: bool) -> None:
        if on:
            self._start_autoscroll()
            self._play_button.setText("⏸")
        else:
            self._stop_autoscroll()
            self._play_button.setText("▶")

    @QtCore.Slot(str)
    def _on_model_changed(self, model: str) -> None:
        if model == self._ds.model:
            return
        if model not in ("continuous", "contfrag"):
            return
        self._switch_model(cast(ModelName, model))

    @QtCore.Slot(int)
    def _on_speed_combo_changed(self, index: int) -> None:
        speed = self._speed_combo.itemData(index)
        if speed is None:
            return
        self._autoscroll_rate = float(speed)

    def _update_slice_panel_at_center(self) -> None:
        ds = self._ds
        t_idx = ds.index_at_time(self._t_center)
        true_pos = float(ds.linear_position[t_idx])
        self.slice_panel.update_for_index(t_idx, true_pos)
        self.slice_panel.set_live_readout(self._format_live_readout(t_idx, true_pos))
        slices, total = self._per_cell_slices_at(t_idx)
        self.slice_panel.set_per_cell_slices(slices, total_in_bin=total)

    def _per_cell_slices_at(self, t_idx: int) -> tuple[list[CellSlice], int]:
        """Build the per-cell row payload for the current bin.

        Deduplicates by ``cell_id`` (a cell that fires twice in the
        same bin gets one row with an ``n_spikes=2`` suffix in the
        header). Caps at ``MAX_PER_CELL_PLOTS`` and returns the
        truncated list plus the unique total so the panel can show a
        ``(+K more)`` indicator.

        Each row's ``place_field_norm`` is the cell's first-state
        place-field tuning embedded into the full per-state position
        grid (non-interior bins stay at zero), normalized to its own
        peak so it sits on a [0, 1] axis alongside the predictive
        overlay.
        """
        ds = self._ds
        i0, i1 = ds.event_indices_at(t_idx)
        if i1 <= i0:
            return [], 0
        cell_ids_in_bin = ds.event_cell_ids[i0:i1]
        seen: dict[int, int] = {}
        for offset, cell_id in enumerate(cell_ids_in_bin):
            seen.setdefault(int(cell_id), i0 + offset)
        unique_cells = list(seen.keys())
        total_unique = len(unique_cells)
        kept = unique_cells[:MAX_PER_CELL_PLOTS]

        n_interior = ds.n_interior
        slices: list[CellSlice] = []
        pinned_cell_id = self._pinned_cell_id()
        for cell_id in kept:
            first_event = seen[cell_id]
            n_spikes = int(np.sum(cell_ids_in_bin == cell_id))
            pf = ds.place_fields[cell_id, :n_interior]
            peak = float(pf.max())
            pf_norm_interior = (pf / peak).astype(np.float32, copy=False) if peak > 0 else pf
            curve = np.zeros(ds.n_position_full, dtype=np.float32)
            curve[ds.interior_mask] = pf_norm_interior
            slices.append(
                CellSlice(
                    cell_id=cell_id,
                    place_field_norm=curve,
                    hpd=float(ds.event_hpd_overlap[first_event]),
                    kl=float(ds.event_kl_divergence[first_event]),
                    spike_prob=float(ds.event_spike_prob[first_event]),
                    n_spikes=n_spikes,
                    is_pinned=(pinned_cell_id is not None and cell_id == pinned_cell_id),
                )
            )
        return slices, total_unique

    def _pinned_cell_id(self) -> int | None:
        row = self._pinned_event_row
        if row is None or not 0 <= row < len(self._ds.events):
            return None
        return int(self._ds.event_cell_ids[row])

    def _format_live_readout(self, t_idx: int, true_pos: float) -> str:
        """Time + predictive(x_true). Per-cell metrics live in the row headers."""
        ds = self._ds
        lines = [f"t = {float(ds.time[t_idx]) - float(ds.time[0]):.3f} s"]
        sl = self.slice_panel._buffer_slice  # noqa: SLF001
        post_buf = self.slice_panel._buffer_post  # noqa: SLF001
        if sl is not None and post_buf is not None and sl.start <= t_idx < sl.stop:
            row = post_buf[t_idx - sl.start]
            if ds.n_states > 1:
                row = row.reshape(ds.n_states, ds.n_position_full).sum(axis=0)
            pos_bin = _nearest_index(ds.position_grid_full, true_pos)
            lines.append(f"predictive(x_true) = {float(row[pos_bin]):.4f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Click / pin
    # ------------------------------------------------------------------

    def _handle_event_click(self, global_event_row: int) -> None:
        """Recenter on the clicked spike's time and pin a marker."""
        ds = self._ds
        if not 0 <= global_event_row < len(ds.events):
            return
        event = ds.events.iloc[global_event_row]
        self._set_pinned_event(global_event_row)
        # Recenter the window on the spike's time. set_center_time
        # animates the slice panel and arms the load-debounce timer.
        self.set_center_time(float(event["time"]))

    def _set_pinned_event(self, global_event_row: int | None) -> None:
        self._pinned_event_row = global_event_row
        self._refresh_pin_markers()

    def _refresh_pin_markers(self) -> None:
        """Sync every panel's pin marker with the current pinned event."""
        ds = self._ds
        row = self._pinned_event_row
        if row is None or not 0 <= row < len(ds.events):
            for panel in self.metric_panels.values():
                panel.update_pinned_event(relative_time=None, metric_value=None)
            self.raster_panel.update_pinned_event(relative_time=None, cell_id=None)
            self.posterior_panel.update_pinned_event(None)
            self.likelihood_panel.update_pinned_event(None)
            self.slice_panel.update_pinned_event(
                place_field_row=None,
                annotation=None,
            )
            return

        event = ds.events.iloc[row]
        sl = self.slice_panel._buffer_slice  # noqa: SLF001
        # The pin's relative time is meaningful only when the event
        # falls within the currently rendered window. The time axis
        # is centered at ``t_center``, so the relative time is the
        # event time minus the current center.
        if sl is None or not (sl.start <= ds.index_at_time(float(event["time"])) < sl.stop):
            relative_time: float | None = None
        else:
            relative_time = float(event["time"]) - float(self._t_center)

        for metric, panel in self.metric_panels.items():
            panel.update_pinned_event(
                relative_time=relative_time,
                metric_value=float(event[metric]),
            )
        self.raster_panel.update_pinned_event(
            relative_time=relative_time,
            cell_id=int(event["cell_id"]),
        )
        self.posterior_panel.update_pinned_event(relative_time)
        self.likelihood_panel.update_pinned_event(relative_time)

        # Slice-panel: the pinned-row highlight is driven via the
        # ``is_pinned`` flag on the next ``set_per_cell_slices`` call;
        # here we only update the annotation label below the rows.
        cell_id = int(event["cell_id"])
        annotation = (
            f"t={float(event['time']):.3f}  cell={cell_id}\n"
            f"HPD={float(event['event_hpd_overlap']):.3f}  "
            f"KL={float(event['event_kl_divergence']):.3f}  "
            f"p={float(event['event_spike_prob']):.3f}"
        )
        self.slice_panel.update_pinned_event(
            place_field_row=None,
            annotation=annotation,
        )
        # Refresh the per-cell rows so the pinned cell's header lights
        # up immediately (without waiting for the next slider tick).
        self._update_slice_panel_at_center()

    @QtCore.Slot(int, slice, object, object)
    def _on_window_loaded(
        self,
        request_id: int,
        sl: slice,
        post: NDArray[np.float32],
        lik: NDArray[np.float32],
    ) -> None:
        # The in-flight worker just finished; clear the slot and fire
        # any deferred dispatch (this is how a paused-then-moved center
        # gets its second load).
        if request_id == self._inflight_request_id:
            self._inflight_request_id = None
            if self._pending_dispatch:
                self._pending_dispatch = False
                self._dispatch_load()

        # Drop stale results from earlier requests.
        if request_id < self._latest_committed_request_id:
            return
        self._latest_committed_request_id = request_id

        time = self._ds.time
        if sl.stop <= sl.start:
            return
        # Time axis is centered at the current ``t_center``: x=0 is
        # the center, the visible window extends symmetrically to
        # negative and positive values. The center-time vertical
        # line on each panel sits at x=0 so the user can see which
        # column corresponds to the slice panel.
        t_offset = float(self._t_center)
        rel_start = float(time[sl.start]) - t_offset
        rel_end = float(time[sl.stop - 1]) - t_offset

        # Pin the visible X range to the *target* half-width (always
        # exactly ``-w/2`` to ``+w/2``) so the tick labels stay
        # rock-stable as the slider moves between samples. The data
        # is positioned with the actual sample extent above; the axis
        # range here is independent of those sub-millisecond shifts.
        # The other time-axis panels follow via ``setXLink``.
        target_half_w = self._window_seconds / 2.0
        self.posterior_panel.setXRange(-target_half_w, target_half_w, padding=0)

        # Slice panel buffer: hand the freshly loaded full-resolution
        # arrays so per-tick ``update_for_index`` is a NumPy index.
        self.slice_panel.set_window_buffer(sl, post, lik)
        # Animate now — the slice should reflect the current center
        # immediately after a load, even if the slider has not moved.
        self._update_slice_panel_at_center()

        self.posterior_panel.update_with_window(rel_start, rel_end, post)
        self.likelihood_panel.update_with_window(rel_start, rel_end, lik)

        events = self._ds.events_in_window(sl)
        if events.empty:
            empty_t = np.empty(0, dtype=np.float64)
            empty_c = np.empty(0, dtype=np.int32)
            empty_m = np.empty(0, dtype=np.float32)
            empty_idx = np.empty(0, dtype=np.int64)
            self.raster_panel.update_window(
                rel_start,
                rel_end,
                empty_t,
                empty_c,
                t_offset,
                empty_idx,
            )
            for panel in self.metric_panels.values():
                panel.update_window(
                    rel_start,
                    rel_end,
                    empty_t,
                    empty_m,
                    t_offset,
                    empty_idx,
                )
        else:
            event_times_arr = events["time"].to_numpy()
            cell_ids = events["cell_id"].to_numpy()
            global_indices = events.index.to_numpy().astype(np.int64, copy=False)
            self.raster_panel.update_window(
                rel_start,
                rel_end,
                event_times_arr,
                cell_ids,
                t_offset,
                global_indices,
            )
            for metric, panel in self.metric_panels.items():
                panel.update_window(
                    rel_start,
                    rel_end,
                    event_times_arr,
                    events[metric].to_numpy(),
                    t_offset,
                    global_indices,
                )

        # Refresh pin markers (the pinned event may now be visible or
        # not depending on the freshly committed window).
        self._refresh_pin_markers()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _dispatch_load(self) -> None:
        # If a worker is already running, defer; the latest center
        # will be picked up by ``_on_window_loaded`` when the current
        # request completes. This keeps at most one load in flight,
        # which avoids queue saturation when the slider is scrubbed
        # faster than disk reads can complete.
        if self._inflight_request_id is not None:
            self._pending_dispatch = True
            return
        self._next_request_id += 1
        self._inflight_request_id = self._next_request_id
        state = ViewState(
            request_id=self._next_request_id,
            t_center=self._t_center,
            t_width=self._window_seconds,
        )
        worker = _WindowLoadWorker(self._ds, state, self._load_signals)
        self._thread_pool.start(worker)

    def _slider_value_for(self, t_center: float) -> int:
        if self._t_max <= self._t_min:
            return 0
        frac = (t_center - self._t_min) / (self._t_max - self._t_min)
        return int(round(np.clip(frac, 0.0, 1.0) * SLIDER_RESOLUTION))

    def _time_for_slider(self, value: int) -> float:
        frac = value / SLIDER_RESOLUTION
        return float(self._t_min + frac * (self._t_max - self._t_min))

    def _format_time_label(self) -> str:
        rel = self._t_center - self._t_min
        return f"t={self._t_center:.3f}  ({rel:.2f} s into session, w={self._window_seconds:.2f} s)"

    # Window-width slider (log-spaced) ---------------------------------

    def _window_slider_value_for(self, window_seconds: float) -> int:
        w = float(np.clip(window_seconds, MIN_WINDOW_SECONDS, MAX_WINDOW_SECONDS))
        log_min = np.log10(MIN_WINDOW_SECONDS)
        log_max = np.log10(MAX_WINDOW_SECONDS)
        frac = (np.log10(w) - log_min) / (log_max - log_min)
        return int(round(frac * WINDOW_SLIDER_RESOLUTION))

    def _window_seconds_for(self, slider_value: int) -> float:
        log_min = np.log10(MIN_WINDOW_SECONDS)
        log_max = np.log10(MAX_WINDOW_SECONDS)
        frac = slider_value / WINDOW_SLIDER_RESOLUTION
        return float(10 ** (log_min + frac * (log_max - log_min)))

    def _format_window_label(self) -> str:
        return f"{self._window_seconds:.2f} s"

    def _set_window_seconds(self, w: float) -> None:
        new_w = float(np.clip(w, MIN_WINDOW_SECONDS, MAX_WINDOW_SECONDS))
        if abs(new_w - self._window_seconds) < 1e-9:
            return
        self._window_seconds = new_w
        # Sync the slider visually (without triggering another event).
        self._window_slider.blockSignals(True)
        self._window_slider.setValue(self._window_slider_value_for(new_w))
        self._window_slider.blockSignals(False)
        self._window_label.setText(self._format_window_label())
        self._time_label.setText(self._format_time_label())
        # Window width changes always require a fresh load.
        self._load_timer.start()

    # Auto-scroll (play/pause) -----------------------------------------

    def _start_autoscroll(self) -> None:
        if self._autoscroll_timer is not None:
            return
        timer = QtCore.QTimer(self)
        interval_ms = max(1, int(round(1000.0 / AUTOSCROLL_TICK_HZ)))
        timer.setInterval(interval_ms)
        timer.timeout.connect(self._autoscroll_step)
        self._autoscroll_timer = timer
        timer.start()

    def _stop_autoscroll(self) -> None:
        if self._autoscroll_timer is None:
            return
        self._autoscroll_timer.stop()
        self._autoscroll_timer.deleteLater()
        self._autoscroll_timer = None

    @QtCore.Slot()
    def _autoscroll_step(self) -> None:
        dt = self._autoscroll_rate / AUTOSCROLL_TICK_HZ
        new_t = self._t_center + dt
        if new_t >= self._t_max:
            new_t = self._t_max
            # End of session — pause and reset the play button.
            if self._play_button.isChecked():
                self._play_button.setChecked(False)
        self.set_center_time(float(new_t))

    @QtCore.Slot()
    def _toggle_play(self) -> None:
        self._play_button.toggle()

    # Speed -----------------------------------------------------------

    @staticmethod
    def _format_speed(speed: float) -> str:
        if speed >= 1.0:
            return f"{speed:.0f}×" if speed.is_integer() else f"{speed:.2g}×"
        return f"{speed:.2g}×"

    def _speed_combo_default_index(self) -> int:
        # Pick the option closest to ``AUTOSCROLL_RATE_REALTIME``.
        diffs = [abs(s - AUTOSCROLL_RATE_REALTIME) for s in AUTOSCROLL_SPEED_OPTIONS]
        return int(np.argmin(diffs))

    def _step_speed(self, delta: int) -> None:
        idx = self._speed_combo.currentIndex() + delta
        idx = int(np.clip(idx, 0, len(AUTOSCROLL_SPEED_OPTIONS) - 1))
        if idx == self._speed_combo.currentIndex():
            return
        self._speed_combo.setCurrentIndex(idx)

    # Keyboard helpers --------------------------------------------------

    def _step_center_by_indices(self, n: int) -> None:
        if n == 0 or self._ds.n_time == 0:
            return
        idx = self._ds.index_at_time(self._t_center) + n
        idx = int(np.clip(idx, 0, self._ds.n_time - 1))
        self.set_center_time(float(self._ds.time[idx]))

    def _step_center_by_seconds(self, dt: float) -> None:
        self.set_center_time(float(self._t_center + dt))

    def _scale_window(self, factor: float) -> None:
        self._set_window_seconds(self._window_seconds * factor)

    @QtCore.Slot()
    def _reset_view(self) -> None:
        # Match the Figure 4a context window: 20 s wide, centered near the
        # session midpoint (the Figure 4 default uses index 190000 of a
        # 709321-point session ~ 27% in, but for synthetic / shorter
        # sessions we pick the geometric default of mid-session).
        mid_idx = max(0, min(self._ds.n_time - 1, self._ds.n_time // 4))
        target_t = float(self._ds.time[mid_idx])
        self._set_window_seconds(RESET_WINDOW_SECONDS)
        self.set_center_time(target_t)

    @QtCore.Slot()
    def _toggle_model(self) -> None:
        if self._cache_dir is None:
            return
        new_model: ModelName = "contfrag" if self._ds.model == "continuous" else "continuous"
        self._switch_model(new_model)

    def _switch_model(self, model: ModelName) -> None:
        if self._cache_dir is None:
            return
        if model == self._ds.model:
            return
        try:
            new_ds = Figure4DataSource(self._cache_dir, model)
        except FileNotFoundError:
            # The requested cache doesn't exist; revert the combo
            # box and bail.
            self._model_combo.blockSignals(True)
            self._model_combo.setCurrentText(self._ds.model)
            self._model_combo.blockSignals(False)
            return

        # Drain the in-flight worker before swapping so the worker
        # never touches the freed Zarr handle.
        self._load_timer.stop()
        deadline = QtCore.QElapsedTimer()
        deadline.start()
        while self._inflight_request_id is not None and deadline.elapsed() < 5000:
            self._thread_pool.waitForDone(50)
            QtWidgets.QApplication.processEvents()

        # Capture persistent UI state before the rebuild.
        was_playing = bool(self._play_button.isChecked())
        if was_playing:
            self._play_button.setChecked(False)  # ensures _stop_autoscroll runs
        window_seconds = float(self._window_seconds)
        speed_index = int(self._speed_combo.currentIndex())
        per_cell_on = bool(self._per_cell_checkbox.isChecked())

        old_ds = self._ds
        self._ds = new_ds
        old_ds.close()
        self._set_pinned_event(None)
        self._latest_committed_request_id = -1
        self._inflight_request_id = None
        self._pending_dispatch = False

        # Update session-level state for the new model.
        self._t_min = float(new_ds.time[0])
        self._t_max = float(new_ds.time[-1])
        self._t_center = float(np.clip(self._t_center, self._t_min, self._t_max))
        self.setWindowTitle(f"Figure 4 viewer — {new_ds.model}")

        # Rebuild the central widget against the new data source. The
        # heatmap, slice and metric panels all bake ``n_states``, so a
        # fresh construction is the safest path.
        self._build_central_widget()

        # Restore captured state onto the new widgets.
        self._set_window_seconds(window_seconds)
        if 0 <= speed_index < self._speed_combo.count():
            self._speed_combo.setCurrentIndex(speed_index)
        self._per_cell_checkbox.setChecked(per_cell_on)
        self._model_combo.blockSignals(True)
        self._model_combo.setCurrentText(new_ds.model)
        self._model_combo.blockSignals(False)
        if was_playing:
            self._play_button.setChecked(True)

        self.force_reload_now()

    # Test hooks ------------------------------------------------------

    def force_reload_now(self) -> None:
        """Bypass the debounce timer; used in benchmarks / tests."""
        self._load_timer.stop()
        self._dispatch_load()

    # Resource cleanup ------------------------------------------------

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802 (Qt API)
        """Wait for any in-flight load worker before tearing down.

        Without this, ``QThreadPool`` may run a worker after the
        ``Figure4DataSource``'s Zarr store has been closed, which
        accesses freed memory and aborts (bus error).
        """
        # Stop any pending dispatches.
        self._load_timer.stop()
        self._pending_dispatch = False
        # Block briefly until the in-flight worker finishes.
        deadline = QtCore.QElapsedTimer()
        deadline.start()
        while self._inflight_request_id is not None and deadline.elapsed() < 5000:
            self._thread_pool.waitForDone(50)
            QtWidgets.QApplication.processEvents()
        # Drop the signals object so any straggler emit hits a noop.
        try:
            self._load_signals.finished.disconnect()
        except (RuntimeError, TypeError):
            pass
        super().closeEvent(event)

    def set_center_time(self, t_center: float) -> None:
        """Programmatic scroll for benchmark scripts and tests."""
        self._t_center = float(t_center)
        self._slider.blockSignals(True)
        self._slider.setValue(self._slider_value_for(self._t_center))
        self._slider.blockSignals(False)
        self._time_label.setText(self._format_time_label())
        # Match the slider path: animate the slice immediately, debounce
        # the heavy heatmap load.
        self._update_slice_panel_at_center()
        self._load_timer.start()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def configure_qt_application(app: QtWidgets.QApplication) -> None:
    """Set viewer-wide Qt defaults (font, pyqtgraph options).

    Setting an explicit application font sidesteps Qt's
    ``Populating font family aliases`` lookup, which can take ~250 ms
    at first show because it expands the missing ``"Sans Serif"``
    alias against the system font set.
    """
    pg.setConfigOptions(antialias=False, useOpenGL=False)
    font = QtGui.QFont("Helvetica", 9)
    if not font.exactMatch():
        font = QtGui.QFont("Arial", 9)
    if not font.exactMatch():
        font = app.font()
    app.setFont(font)


def launch(cache_dir: Path | str, model: ModelName) -> int:
    """Open the viewer for ``model`` from ``cache_dir`` and run the event loop."""
    existing = QtWidgets.QApplication.instance()
    app: QtWidgets.QApplication = (
        existing
        if isinstance(existing, QtWidgets.QApplication)
        else QtWidgets.QApplication(sys.argv)
    )
    configure_qt_application(app)
    ds = Figure4DataSource(cache_dir, model)
    viewer = Figure4Viewer(ds, cache_dir=cache_dir)
    viewer.show()
    try:
        return int(app.exec())
    finally:
        ds.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="statespacecheck_paper.interactive.viewer")
    parser.add_argument("--cache-dir", required=True, help="Cache directory.")
    parser.add_argument(
        "--model",
        choices=("continuous", "contfrag"),
        default="continuous",
        help="Which model's cache to open.",
    )
    args = parser.parse_args(argv)
    return launch(args.cache_dir, args.model)


if __name__ == "__main__":
    raise SystemExit(main())
