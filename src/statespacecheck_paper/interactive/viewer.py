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
from dataclasses import dataclass
from pathlib import Path

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
        self.setLabel("bottom", "Time (s)")
        self.getPlotItem().setTitle(title)

        self._image = pg.ImageItem(axisOrder="row-major")
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
        self._image.setRect(
            QtCore.QRectF(
                time_start,
                self._y0,
                time_end - time_start,
                self._y1 - self._y0,
            )
        )
        self.setXRange(time_start, time_end, padding=0)
        self.setYRange(self._y0, self._y1, padding=0)


class PosteriorPanel(_BaseHeatmapPanel):
    """Predictive posterior heatmap (state-summed for multi-state models)."""

    def __init__(self, *, position_bins: NDArray[np.float64], n_states: int) -> None:
        super().__init__(title="Predictive posterior", position_bins=position_bins)
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
        self.setLabel("bottom", "Time (s)")
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
        )
        self.addItem(self._scatter)
        self.setYRange(-0.5, n_cells - 0.5, padding=0)

    def update_window(
        self,
        time_start: float,
        time_end: float,
        events_time: NDArray[np.float64],
        events_cell_id: NDArray[np.int32],
        time_offset: float,
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
        """
        if events_time.size == 0:
            self._scatter.setData(x=[], y=[])
            self.setXRange(time_start, time_end, padding=0)
            return
        x = events_time - time_offset
        y = self._cell_rank[events_cell_id]
        self._scatter.setData(x=x, y=y)
        self.setXRange(time_start, time_end, padding=0)


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


class SlicePanel(pg.PlotWidget):
    """1D animated posterior + likelihood curves at the current center time.

    The hot path is ``update_for_index(t_idx)``: it indexes one
    pre-loaded window's float32 row and calls ``setData`` on each
    curve. This must stay sub-millisecond so the slice animates
    smoothly while the user scrolls.

    For multi-state models (ContFrag) the panel switches to
    stacked-by-state rendering: one posterior curve and one
    likelihood-fill per state, color-coded. A "sum over state" toggle
    is intentionally deferred to a later phase; until then both states
    are always visible.
    """

    def __init__(
        self,
        *,
        position_bins: NDArray[np.float64],
        n_states: int,
    ) -> None:
        super().__init__()
        self.setBackground("w")
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.setLabel("bottom", "Position (cm)")
        self.setLabel("left", "Density")
        self.getPlotItem().setTitle("Slice at center time")

        self._position_bins = np.asarray(position_bins, dtype=np.float64)
        self._n_pos = int(self._position_bins.shape[0])
        self._n_states = max(1, int(n_states))
        self._zero_curve = np.zeros(self._n_pos, dtype=np.float32)

        # Likelihood alpha — set by an external slider (0..255).
        self._likelihood_alpha = 140

        self._posterior_curves: list[pg.PlotDataItem] = []
        self._likelihood_curves: list[pg.PlotDataItem] = []
        self._likelihood_baseline_curves: list[pg.PlotDataItem] = []
        self._likelihood_fills: list[pg.FillBetweenItem] = []
        for s in range(self._n_states):
            post_rgb = _STATE_POSTERIOR_RGB[s % len(_STATE_POSTERIOR_RGB)]
            lik_rgb = _STATE_LIKELIHOOD_RGB[s % len(_STATE_LIKELIHOOD_RGB)]
            post_curve = pg.PlotDataItem(
                self._position_bins,
                self._zero_curve,
                pen=pg.mkPen(post_rgb, width=2),
            )
            self.addItem(post_curve)
            self._posterior_curves.append(post_curve)

            # Likelihood is shown as a filled area between a zero baseline
            # and the curve. We keep both curves (top + baseline) so the
            # ``FillBetweenItem`` can update via the curves.
            lik_top = pg.PlotDataItem(
                self._position_bins,
                self._zero_curve,
                pen=pg.mkPen(_LIKELIHOOD_PEN_RGB, width=1),
            )
            lik_base = pg.PlotDataItem(self._position_bins, self._zero_curve, pen=None)
            self.addItem(lik_top)
            self.addItem(lik_base)
            self._likelihood_curves.append(lik_top)
            self._likelihood_baseline_curves.append(lik_base)
            fill = pg.FillBetweenItem(
                lik_top,
                lik_base,
                brush=pg.mkBrush(*lik_rgb, self._likelihood_alpha),
            )
            self.addItem(fill)
            self._likelihood_fills.append(fill)

        # True animal position marker.
        self._true_position_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=_TRUE_POSITION_PEN,
        )
        self.addItem(self._true_position_line)

        # Window-buffer state. The viewer pushes a freshly loaded
        # window via ``set_window_buffer``; subsequent
        # ``update_for_index`` calls index into it.
        self._buffer_slice: slice | None = None
        self._buffer_post: NDArray[np.float32] | None = None
        self._buffer_lik: NDArray[np.float32] | None = None

    # ------------------------------------------------------------------
    # Data plumbing
    # ------------------------------------------------------------------

    def set_window_buffer(
        self,
        sl: slice,
        post: NDArray[np.float32],
        lik: NDArray[np.float32],
    ) -> None:
        """Cache the most recent window for fast per-tick row access.

        ``post`` and ``lik`` are the full ``(n_visible, n_state_bins)``
        arrays exactly as ``Figure4DataSource.load_*`` returned them
        (NaN-cleaned and exp'd by the worker thread). We store them as
        a ring of one window — the simplest safe option, since each
        new ``set_window_buffer`` call lines up with a fresh load that
        already brought the data into RAM.
        """
        self._buffer_slice = sl
        self._buffer_post = post
        self._buffer_lik = lik

    def update_for_index(self, t_idx: int, true_position: float) -> None:
        """Update the curves for a single decoder time index.

        ``t_idx`` is an absolute decoder index. If it falls outside the
        currently buffered window, this method silently does nothing
        and waits for the next buffer; the viewer triggers a fresh load
        whenever the center moves outside the visible window.
        """
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
        if self._n_states == 1:
            self._posterior_curves[0].setData(self._position_bins, post_row)
            self._likelihood_curves[0].setData(self._position_bins, lik_row)
        else:
            # Reshape (n_state_bins,) -> (n_states, n_pos) for stacked rendering.
            post_rs = post_row.reshape(self._n_states, self._n_pos)
            lik_rs = lik_row.reshape(self._n_states, self._n_pos)
            for s in range(self._n_states):
                self._posterior_curves[s].setData(self._position_bins, post_rs[s])
                self._likelihood_curves[s].setData(self._position_bins, lik_rs[s])
        self._true_position_line.setPos(true_position)

    def set_likelihood_alpha(self, alpha: int) -> None:
        """Adjust the likelihood-fill opacity (0..255)."""
        alpha = int(np.clip(alpha, 0, 255))
        if alpha == self._likelihood_alpha:
            return
        self._likelihood_alpha = alpha
        for s, fill in enumerate(self._likelihood_fills):
            lik_rgb = _STATE_LIKELIHOOD_RGB[s % len(_STATE_LIKELIHOOD_RGB)]
            fill.setBrush(pg.mkBrush(*lik_rgb, alpha))


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class Figure4Viewer(QtWidgets.QMainWindow):
    """Top-level window owning the panels and view state."""

    def __init__(
        self, data_source: Figure4DataSource, *, parent: QtWidgets.QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self._ds = data_source

        self.setWindowTitle(f"Figure 4 viewer — {data_source.model}")
        self.resize(1100, 800)

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

        self._build_panels()
        self._build_controls()
        self._wire_load_worker()

        # Trigger initial load.
        self._dispatch_load()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

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
        self.slice_panel = SlicePanel(
            position_bins=ds.position_grid_full,
            n_states=ds.n_states,
        )
        # Link x-axes so any zoom/range change propagates.
        self.likelihood_panel.setXLink(self.posterior_panel)
        self.raster_panel.setXLink(self.posterior_panel)

    def _build_controls(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # Horizontal split: time-axis panels (left, ~70%) | slice panel (right, ~30%).
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        time_axis = QtWidgets.QWidget()
        time_axis_layout = QtWidgets.QVBoxLayout(time_axis)
        time_axis_layout.setContentsMargins(0, 0, 0, 0)
        time_axis_layout.setSpacing(2)
        time_axis_layout.addWidget(self.posterior_panel, stretch=2)
        time_axis_layout.addWidget(self.likelihood_panel, stretch=2)
        time_axis_layout.addWidget(self.raster_panel, stretch=1)
        split.addWidget(time_axis)
        split.addWidget(self.slice_panel)
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
        controls_layout.addWidget(QtWidgets.QLabel("Likelihood α:"))
        self._alpha_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._alpha_slider.setRange(0, 255)
        self._alpha_slider.setValue(140)
        self._alpha_slider.setMaximumWidth(140)
        self._alpha_slider.valueChanged.connect(self._on_alpha_changed)
        controls_layout.addWidget(self._alpha_slider)

        outer.addWidget(controls)

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
        # Animate the slice panel immediately — this is the per-tick
        # path the user perceives during scrubbing. It is a single
        # array index into the in-RAM ring buffer, so it is sub-ms.
        self._update_slice_panel_at_center()
        # Re-arm the debounce timer for the heavier window load.
        self._load_timer.start()

    @QtCore.Slot(int)
    def _on_alpha_changed(self, value: int) -> None:
        self.slice_panel.set_likelihood_alpha(value)

    def _update_slice_panel_at_center(self) -> None:
        ds = self._ds
        t_idx = ds.index_at_time(self._t_center)
        true_pos = float(ds.linear_position[t_idx])
        self.slice_panel.update_for_index(t_idx, true_pos)

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
        t_offset = float(time[sl.start])
        t_end = float(time[sl.stop - 1])
        rel_start = 0.0
        rel_end = t_end - t_offset

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
            self.raster_panel.update_window(
                rel_start,
                rel_end,
                np.empty(0, dtype=np.float64),
                np.empty(0, dtype=np.int32),
                t_offset,
            )
        else:
            self.raster_panel.update_window(
                rel_start,
                rel_end,
                events["time"].to_numpy(),
                events["cell_id"].to_numpy(),
                t_offset,
            )

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

    # Test hooks ------------------------------------------------------

    def force_reload_now(self) -> None:
        """Bypass the debounce timer; used in benchmarks / tests."""
        self._load_timer.stop()
        self._dispatch_load()

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
    viewer = Figure4Viewer(ds)
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
