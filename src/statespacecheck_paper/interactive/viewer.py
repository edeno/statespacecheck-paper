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
        self.posterior_panel = PosteriorPanel(
            position_bins=ds.position_bins,
            n_states=ds.n_states,
        )
        self.likelihood_panel = LikelihoodPanel(
            position_bins=ds.position_bins,
            n_states=ds.n_states,
        )
        self.raster_panel = RasterPanel(
            n_cells=ds.n_cells,
            place_field_peaks=ds.place_field_peaks,
        )
        # Link x-axes so any zoom/range change propagates.
        self.likelihood_panel.setXLink(self.posterior_panel)
        self.raster_panel.setXLink(self.posterior_panel)

    def _build_controls(self) -> None:
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        layout.addWidget(self.posterior_panel, stretch=2)
        layout.addWidget(self.likelihood_panel, stretch=2)
        layout.addWidget(self.raster_panel, stretch=1)

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

        layout.addWidget(controls)

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
        # Re-arm the debounce timer; the actual load happens once the
        # user pauses for ~16 ms (i.e. one frame at 60 Hz).
        self._load_timer.start()

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
