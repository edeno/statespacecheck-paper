"""Top-level decoder-viewer window and worker plumbing.

``DecoderViewer`` orchestrates the panels (provided by ``panels.py``)
and the cache reads (provided by ``data_source.py``). It owns:

- view state (center time, window width, model, pinned event),
- the autoscroll / keyboard / model-swap UI behaviors,
- the ``_WindowLoadWorker`` + ``_LoadSignals`` thread-pool harness that
  reads a window's posterior + log-likelihood off the disk cache and
  hands the result to the panels on the main thread.

For the Qt-application entry point and the
``python -m statespacecheck_paper.interactive.viewer`` CLI, see
``app.py``. Panel widget classes (``PosteriorPanel``,
``LikelihoodPanel``, ``RasterPanel``, ``MetricPanel``, ``SlicePanel``)
plus the ``CellSlice`` payload dataclass live in ``panels.py``.

Names imported from those modules (and a few constants) are
re-exported at the bottom of this file so ``from
statespacecheck_paper.interactive.viewer import X`` keeps working for
the existing test suite.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray
from PySide6 import QtCore, QtGui, QtWidgets

from .data_source import DecoderDataSource, ModelName
from .panels import (
    _OVERLAY_LABELS,
    MAX_PER_CELL_PLOTS,
    OVERLAY_CHOICES,
    CellSlice,
    LikelihoodPanel,
    MetricPanel,
    OverlayChoice,
    PosteriorPanel,
    RasterPanel,
    SlicePanel,
)

# Plan defaults.
DEFAULT_WINDOW_SECONDS = 2.0
MIN_WINDOW_SECONDS = 0.1
MAX_WINDOW_SECONDS = 60.0
SLIDER_RESOLUTION = 100_000  # subdivides the full session into this many ticks

# Window-width slider works in log-space so a single slider position
# can resolve both 0.1 s and 60 s endpoints with reasonable granularity.
WINDOW_SLIDER_RESOLUTION = 1000

# Reset shortcut targets (matches the Figure 4a context window from
# scripts/generate_figure04.py via ``index 190000`` at the decoder
# sampling rate of 500 Hz, ~20 s wide).
RESET_WINDOW_SECONDS = 20.0

# Auto-scroll defaults.
AUTOSCROLL_TICK_HZ = 30.0
AUTOSCROLL_RATE_REALTIME = 1.0  # 1 second of session per second of wall time
AUTOSCROLL_SPEED_OPTIONS: tuple[float, ...] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
# Startup default. Real-time playback is so fast that the slice panel
# barely registers; 0.05× is slow enough to actually watch the
# decoder track the animal between bins.
AUTOSCROLL_RATE_DEFAULT = 0.05


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
    # Skip the acausal load on the worker when the slice panel is not
    # using the smoothed overlay — for the default predictive overlay
    # we save one full Zarr read per window dispatch (≈ a third of the
    # load cost on this cache).
    load_acausal: bool


class _LoadSignals(QtCore.QObject):
    """Bridge object owning the result signal for the load worker.

    ``QRunnable`` cannot inherit from ``QObject``; the standard
    workaround is to attach a ``QObject`` member that owns the
    signals.
    """

    # request_id, slice, post, lik, acausal (None if cache lacks it)
    finished = QtCore.Signal(int, slice, object, object, object)


class _WindowLoadWorker(QtCore.QRunnable):
    """Pull one window's posterior + log-likelihood + acausal from the cache.

    Runs on a ``QThreadPool`` worker thread; emits the result on the
    main thread via the bridge ``QObject``'s signal.
    """

    def __init__(
        self,
        data_source: DecoderDataSource,
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
        acausal = self._ds.load_acausal(sl) if self._state.load_acausal else None

        # NaN-clean non-interior bins so the heatmap's level
        # computation never sees all-NaN rows.
        if post.size:
            np.nan_to_num(post, copy=False, nan=0.0)
        if loglik.size:
            np.nan_to_num(loglik, copy=False, nan=-np.inf)
        if acausal is not None and acausal.size:
            np.nan_to_num(acausal, copy=False, nan=0.0)

        # Convert log-likelihood -> normalized linear likelihood here
        # on the worker thread so the main thread does not pay
        # ``np.exp`` per committed update. Subtract the per-row max
        # first to avoid float32 overflow.
        if loglik.size:
            row_max = loglik.max(axis=1, keepdims=True)
            row_max = np.where(np.isfinite(row_max), row_max, 0.0)
            lik = np.exp(loglik - row_max, dtype=np.float32)
            np.nan_to_num(lik, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            lik = loglik
        self._signals.finished.emit(self._state.request_id, sl, post, lik, acausal)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------


class DecoderViewer(QtWidgets.QMainWindow):
    """Top-level window owning the panels and view state."""

    def __init__(
        self,
        data_source: DecoderDataSource,
        *,
        parent: QtWidgets.QWidget | None = None,
        cache_dir: Path | str | None = None,
    ) -> None:
        super().__init__(parent)
        self._ds = data_source
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

        self.setWindowTitle(f"Decoder viewer — {data_source.display_name}")
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
        self._autoscroll_rate = AUTOSCROLL_RATE_DEFAULT
        self._autoscroll_timer: QtCore.QTimer | None = None
        # Cached ``(left_rel, right_rel)`` of the active-bin highlight
        # band so per-tick refreshes can early-return when the bounds
        # haven't moved (see ``_refresh_active_bin_band``).
        self._active_bin_band_bounds: tuple[float, float] | None = None

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
        self.slice_panel.set_row_provider(self._slice_row_at)
        # Link x-axes so any zoom/range change propagates across the
        # time-axis stack (heatmaps + raster + metric panels).
        x_linked: list[pg.PlotWidget] = [self.likelihood_panel, self.raster_panel]
        x_linked.extend(self.metric_panels.values())
        for panel in x_linked:
            panel.setXLink(self.posterior_panel)

        # Wheel-over-time-axis-panel scrolls the window width. Install
        # the event filter both on the panel itself and its viewport
        # because pyqtgraph's ``PlotWidget`` (a ``GraphicsView``) routes
        # wheel events through the viewport widget.
        self._wheel_filter_targets: tuple[pg.PlotWidget, ...] = (
            self.posterior_panel,
            self.likelihood_panel,
            self.raster_panel,
            *self.metric_panels.values(),
        )
        for panel in self._wheel_filter_targets:
            panel.installEventFilter(self)
            viewport = panel.viewport()
            if viewport is not None:
                viewport.installEventFilter(self)

    def _slice_row_at(
        self, t_idx: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32] | None]:
        """Single-row read used by ``SlicePanel`` when the buffer is stale.

        Mirrors the per-window normalization the worker thread does in
        ``_WindowLoadWorker.run`` so the slice panel sees the same kind
        of arrays whether the row came from the buffered window or
        from this fallback path. Returns ``(post, lik, acausal)``;
        ``acausal`` is ``None`` for older caches without
        ``acausal_posterior``.
        """
        ds = self._ds
        post_row = ds.slice_at_index(t_idx, which="posterior").copy()
        loglik_row = ds.slice_at_index(t_idx, which="likelihood").copy()
        np.nan_to_num(post_row, copy=False, nan=0.0)
        np.nan_to_num(loglik_row, copy=False, nan=-np.inf)
        if loglik_row.size:
            row_max = float(loglik_row.max()) if np.isfinite(loglik_row.max()) else 0.0
            lik_row = np.exp(loglik_row - row_max, dtype=np.float32)
            np.nan_to_num(lik_row, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            lik_row = loglik_row.astype(np.float32, copy=False)
        if ds.has_acausal:
            acausal_row = ds.slice_at_index(t_idx, which="acausal").copy()
            np.nan_to_num(acausal_row, copy=False, nan=0.0)
        else:
            acausal_row = None
        return post_row, lik_row, acausal_row

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
        controls_layout.addWidget(QtWidgets.QLabel("Overlay:"))
        self._overlay_combo = QtWidgets.QComboBox()
        self._overlay_combo.setToolTip(
            "Distribution shown as the blue overlay on the top slice "
            "plot. Per-cell rows always use the predictive prior."
        )
        for choice in OVERLAY_CHOICES:
            self._overlay_combo.addItem(_OVERLAY_LABELS[choice], userData=choice)
        # Smoothed requires ``acausal_posterior`` in the cache; older
        # caches don't have it, so disable the option there.
        if not self._ds.has_acausal:
            smoothed_idx = OVERLAY_CHOICES.index("smoothed")
            combo_model = cast(QtGui.QStandardItemModel, self._overlay_combo.model())
            model_item = combo_model.item(smoothed_idx)
            if model_item is not None:
                model_item.setEnabled(False)
                model_item.setToolTip(
                    "Cache built before the smoothed-overlay feature; "
                    "rebuild via 'python -m statespacecheck_paper.interactive.cache build'."
                )
        self._overlay_combo.setCurrentIndex(OVERLAY_CHOICES.index(self.slice_panel.overlay_choice))
        self._overlay_combo.currentIndexChanged.connect(self._on_overlay_combo_changed)
        controls_layout.addWidget(self._overlay_combo)

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

        # Model swap UI only makes sense for real-data caches with both
        # ``continuous`` and ``contfrag`` available. The figure-3
        # simulation has a single decoder baked into the data and no
        # alternative to swap to, so the label + combo are hidden
        # entirely (not just disabled — there's no model concept here).
        self._model_label: QtWidgets.QLabel | None = None
        self._model_combo: QtWidgets.QComboBox | None = None
        if self._ds.dataset_kind == "model":
            controls_layout.addSpacing(12)
            self._model_label = QtWidgets.QLabel("Model:")
            controls_layout.addWidget(self._model_label)
            self._model_combo = QtWidgets.QComboBox()
            self._model_combo.addItems(["continuous", "contfrag"])
            self._model_combo.setCurrentText(self._ds.model or "")
            self._model_combo.currentTextChanged.connect(self._on_model_changed)
            # Disabled when the cache directory wasn't provided (e.g.
            # tests that construct with a single model in tmp_path).
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
        - ``[`` / ``]``          : shrink / grow window width (or
                                    scroll the mouse wheel over any
                                    time-axis panel).
        - ``R``                  : reset to a 20 s context window centered
                                    near the Figure 4a default.
        - ``Esc``                : unpin the currently pinned spike
                                    (clicking the pinned spike again
                                    also unpins).
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
        add("Escape", self._unpin_event)

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
    def _on_overlay_combo_changed(self, index: int) -> None:
        choice = self._overlay_combo.itemData(index)
        if choice is None:
            return
        prev_choice = self.slice_panel.overlay_choice
        self.slice_panel.set_overlay_choice(cast(OverlayChoice, choice))
        # Refresh the slice immediately so the new overlay appears
        # without waiting for a slider tick.
        self._update_slice_panel_at_center()
        # The worker only loads ``acausal_posterior`` when the current
        # overlay is ``smoothed`` (saving one Zarr read per window on
        # the default predictive path). Switching into ``smoothed``
        # therefore needs a fresh window load to populate the buffer
        # — without it the slice would silently fall back to predictive.
        if (
            self._ds.has_acausal
            and choice == "smoothed"
            and prev_choice != "smoothed"
            and self.slice_panel._buffer_acausal is None  # noqa: SLF001
        ):
            self.force_reload_now()

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
        self._refresh_active_bin_band(t_idx)

    def _refresh_active_bin_band(self, t_idx: int) -> None:
        """Push the active-bin highlight band to every time-axis panel.

        Bin ``t_idx`` covers the half-open interval
        ``[time[t_idx], time[t_idx+1])`` (LEFT-EDGE convention). The
        band is positioned in window-relative seconds (``x = 0`` at
        the current center). When the active bin is the last one in
        the session there's no ``time[t_idx + 1]`` so we extrapolate
        by one ``dt`` from the previous interval.

        Called every UI tick. Skip the per-panel ``setRegion`` updates
        when the rendered bounds haven't moved (``t_idx`` and
        ``t_center`` unchanged) — autoscroll between bin transitions
        otherwise pushes ~5 redundant ``setRegion`` calls per tick.
        """
        ds = self._ds
        t_left = float(ds.time[t_idx])
        if t_idx + 1 < ds.n_time:
            t_right = float(ds.time[t_idx + 1])
        elif t_idx > 0:
            t_right = t_left + float(ds.time[t_idx] - ds.time[t_idx - 1])
        else:
            t_right = t_left
        left_rel = t_left - self._t_center
        right_rel = t_right - self._t_center
        if self._active_bin_band_bounds == (left_rel, right_rel):
            return
        self._active_bin_band_bounds = (left_rel, right_rel)
        for panel in self._wheel_filter_targets:
            panel.update_active_bin_band(left_rel, right_rel)

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
        """Pin a clicked spike and recenter on it; clicking the pinned spike unpins."""
        ds = self._ds
        if not 0 <= global_event_row < len(ds.events):
            return
        # Toggle: clicking the already-pinned spike clears the pin.
        # (Manual scrolling also unpins — see ``set_center_time``;
        # ``Escape`` is a no-recenter shortcut that does the same.)
        if global_event_row == self._pinned_event_row:
            self._set_pinned_event(None)
            return
        event = ds.events.iloc[global_event_row]
        self._set_pinned_event(global_event_row)
        # Recenter the window on the spike's time. set_center_time
        # animates the slice panel and arms the load-debounce timer.
        self.set_center_time(float(event["time"]))

    def _unpin_event(self) -> None:
        """Clear any pinned spike without changing the window center."""
        if self._pinned_event_row is not None:
            self._set_pinned_event(None)

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

    @QtCore.Slot(int, slice, object, object, object)
    def _on_window_loaded(
        self,
        request_id: int,
        sl: slice,
        post: NDArray[np.float32],
        lik: NDArray[np.float32],
        acausal: NDArray[np.float32] | None,
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
        # LEFT-EDGE bin convention (matches ``event_time_idx`` and
        # ``np.digitize`` spike binning): bin ``i`` covers the real-time
        # interval ``[time[i], time[i+1])``. The visible window is
        # therefore the half-open interval ``[time[sl.start], time[sl.stop])``.
        # ``rel_start`` / ``rel_end`` are those edges shifted to be
        # relative to the current center marker (``x = 0`` = ``t_center``).
        # When the window includes the very last sample, extrapolate the
        # right edge by one ``dt`` since there's no ``time[sl.stop]``.
        t_offset = float(self._t_center)
        rel_start = float(time[sl.start]) - t_offset
        if sl.stop < self._ds.n_time:
            rel_end = float(time[sl.stop]) - t_offset
        elif sl.stop - sl.start >= 2:
            dt = float(time[sl.stop - 1] - time[sl.stop - 2])
            rel_end = float(time[sl.stop - 1]) + dt - t_offset
        else:
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
        self.slice_panel.set_window_buffer(sl, post, lik, acausal=acausal)
        # Animate now — the slice should reflect the current center
        # immediately after a load, even if the slider has not moved.
        self._update_slice_panel_at_center()

        self.posterior_panel.update_with_window(rel_start, rel_end, post)
        self.likelihood_panel.update_with_window(rel_start, rel_end, lik)

        # Overlay the animal's true position trajectory on both heatmaps.
        # Times are relative to ``t_center`` so the curve aligns with the
        # heatmap's center marker at x=0.
        rel_time_window = np.asarray(time[sl], dtype=np.float64) - t_offset
        linear_pos_window = np.asarray(self._ds.linear_position[sl], dtype=np.float64)
        self.posterior_panel.update_position_trajectory(rel_time_window, linear_pos_window)
        self.likelihood_panel.update_position_trajectory(rel_time_window, linear_pos_window)

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
            load_acausal=(self._ds.has_acausal and self.slice_panel.overlay_choice == "smoothed"),
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
        # Pick the option closest to ``AUTOSCROLL_RATE_DEFAULT``.
        diffs = [abs(s - AUTOSCROLL_RATE_DEFAULT) for s in AUTOSCROLL_SPEED_OPTIONS]
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

    def eventFilter(  # noqa: N802 — overrides ``QObject.eventFilter`` (Qt API name)
        self, watched: QtCore.QObject, event: QtCore.QEvent
    ) -> bool:
        """Intercept wheel events on time-axis panels to scale the window width.

        Mouse-wheel up = zoom in (smaller window, more detail). Mouse-wheel
        down = zoom out (larger window). The event filter runs *before*
        ``pg.PlotWidget``'s own ``wheelEvent`` so we can replace its
        viewbox-zoom behavior cleanly without disabling other plot
        interactions. Touchpads emit a stream of small ``angleDelta``
        events; the exponential factor keeps the response smooth instead
        of stepping.
        """
        if event.type() == QtCore.QEvent.Type.Wheel and isinstance(event, QtGui.QWheelEvent):
            delta = event.angleDelta().y()
            if delta != 0:
                # 120 units = one notch on a standard mouse wheel; touchpad
                # scrolls report smaller deltas and accumulate naturally.
                factor = 1.15 ** (-delta / 120.0)
                self._scale_window(factor)
                event.accept()
                return True
        return bool(super().eventFilter(watched, event))

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
        if self._ds.dataset_kind != "model" or self._cache_dir is None:
            # The simulation dataset has no alternative model; the
            # ``M`` keyboard shortcut and any stray combo signal
            # both no-op.
            return
        new_model: ModelName = "contfrag" if self._ds.model == "continuous" else "continuous"
        self._switch_model(new_model)

    def _switch_model(self, model: ModelName) -> None:
        if self._ds.dataset_kind != "model" or self._cache_dir is None:
            return
        if model == self._ds.model:
            return
        try:
            new_ds = DecoderDataSource(self._cache_dir, model)
        except FileNotFoundError:
            # The requested cache doesn't exist; revert the combo
            # box and bail.
            if self._model_combo is not None:
                self._model_combo.blockSignals(True)
                self._model_combo.setCurrentText(self._ds.model or "")
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
        overlay_choice = self.slice_panel.overlay_choice

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
        self.setWindowTitle(f"Decoder viewer — {new_ds.display_name}")

        # Rebuild the central widget against the new data source. The
        # heatmap, slice and metric panels all bake ``n_states``, so a
        # fresh construction is the safest path.
        self._build_central_widget()

        # Restore captured state onto the new widgets.
        self._set_window_seconds(window_seconds)
        if 0 <= speed_index < self._speed_combo.count():
            self._speed_combo.setCurrentIndex(speed_index)
        self._per_cell_checkbox.setChecked(per_cell_on)
        # Restore overlay choice; falls back to predictive if the
        # new model's cache lacks acausal_posterior.
        if overlay_choice == "smoothed" and not self._ds.has_acausal:
            overlay_choice = "predictive"
        self._overlay_combo.setCurrentIndex(OVERLAY_CHOICES.index(overlay_choice))
        self.slice_panel.set_overlay_choice(overlay_choice)
        if self._model_combo is not None:
            self._model_combo.blockSignals(True)
            self._model_combo.setCurrentText(new_ds.model or "")
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
        ``DecoderDataSource``'s Zarr store has been closed, which
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
# Backward-compat re-exports
# ---------------------------------------------------------------------------
#
# Pre-split, all of the panel classes and the launch / configure_qt_application
# helpers lived in this module. Tests and downstream callers still import
# them via ``from statespacecheck_paper.interactive.viewer import X``; keep
# those imports working without forcing an update.

from .app import configure_qt_application, launch, main  # noqa: E402, F401
from .panels import (  # noqa: E402, F401
    _LIKELIHOOD_PEN_RGB,
    _PER_CELL_PALETTE,
    _SLICE_Y_MAX,
    _SLICE_Y_MIN,
    _STATE_LIKELIHOOD_RGB,
    _STATE_POSTERIOR_RGB,
    _TRUE_POSITION_PEN,
    METRIC_COLORS,
    METRIC_TITLES,
    _make_slice_subplot,
    _PerCellRow,
    _pin_slice_axes,
)

if __name__ == "__main__":
    raise SystemExit(main())
