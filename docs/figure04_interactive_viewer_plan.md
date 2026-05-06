# Figure 4 Interactive Viewer — Implementation Plan

## Goal

A desktop viewer for inspecting Figure 4 data (per-cell diagnostics for
state space decoder models on real hippocampal recording data) with:

- Scrollable, adjustable-width time window over the full session (not a
  preset window).
- Click on a spike (raster) or metric point (HPD overlap / KL divergence
  / spike probability) → recenter the window on that event and pin a
  marker.
- A 1D slice panel (right side) showing the predictive posterior `p(x_t |
  y_{1:t-1})` overlaid with the likelihood at the current center time;
  the slice **animates** as the user scrolls so they can watch the
  distribution evolve.
- One model at a time (Continuous OR Cont-Frag), togglable.

## Why pyqtgraph

Regular use + low-latency scrolling + animated 1D slice rules out
HoloViews/Bokeh (tap-stream → server roundtrip is too slow for
scroll-driven animation) and matplotlib (heatmap redraw is the
bottleneck). pyqtgraph's GL-backed `ImageItem` plus direct
`PlotCurveItem.setData` calls give the right latency profile.

## Verified data inventory

Confirmed by inspecting `data/intermediates/` on `main`:

- `cont_results.nc` (2.2 GB): `predictive_posterior (709321, 256)
  float32`, `log_likelihood` same shape, 1 state ("Continuous").
- `cont_frag_results.nc` (4.4 GB): `predictive_posterior (709321, 512)
  float32`, `log_likelihood` same shape, 2 states ("Continuous",
  "Fragmented") × 256 position bins.
- `state_bins` is stored as a plain integer dim after the NetCDF
  round-trip, but `state` and `position` survive as non-dim coords on
  `state_bins`. `ds.set_index(state_bins=["state", "position"])` cleanly
  restores the MultiIndex; unstack yields `('time', 'state', 'position')`
  with shape `(709321, 2, 256)` for ContFrag and `(709321, 1, 256)` for
  Continuous.
- `fig4_diagnostics.pkl` (joblib, 5.2 GB): dense matrices
  - `hpd_overlap`, `kl_divergence`, `spike_prob`: each `(709321, 203)
    float64`.
  - `spike_time_ind`, `spike_cell_ind`: each `(869047,) float64` —
    869,047 spikes, 203 cells.
  - `per_spike_likelihood (869047, 248) float64` — **not** cached. The
    selected spike's likelihood is recomputed from the cell's place
    field on click.
- Spikes per cell still live as a list of `np.float64` arrays in
  `data/j1620210710_02_r1_HPC_spike_times.pkl` (joblib).
- Animal position is in `data/j1620210710_02_r1_position_info.pkl`
  (`linear_position` column).

The cache builder **reads** these existing files and writes a
viewer-friendly format. It does not re-run `model.predict(...)`.

## Architecture

### Module layout (under `src/statespacecheck_paper/interactive/`)

- `cache.py` — one-shot builder that converts the existing
  `intermediates/` NetCDF + pickle files into a Zarr store + Parquet
  event table + small `.npy` sidecars. Idempotent. CLI entry point.
- `data_source.py` — `DecoderDataSource`: opens the cache lazily, exposes
  `window_indices`, `load_posterior(slice)`, `load_likelihood(slice)`,
  `events_in_window(slice)`. Holds in-RAM the small things (`time`,
  `linear_position`, `place_field_peaks`, the full event Parquet, the
  per-cell spike-time list) and lazy handles to the big things
  (posterior, log_likelihood Zarr arrays).
- `panels.py` — one class per row: `PosteriorPanel`, `LikelihoodPanel`,
  `RasterPanel`, `MetricPanel` (parameterized by metric name),
  `SlicePanel`. Each owns its `pg.PlotItem` and an `update_window` and
  `update_center` method.
- `viewer.py` — `DecoderViewer(QtWidgets.QMainWindow)`: builds the
  layout, owns `view_state` (center_t, window_w, model, pinned_event),
  wires panels, runs the load worker.
- `controls.py` — center-time slider (range = full session), window-width
  slider, model selector, top-overlay choice (predictive / filtered /
  smoothed) for the slice panel, per-cell-rows toggle, play/pause +
  speed combo. (The original plan called for an alpha slider on the
  likelihood overlay; that was dropped during implementation in favour
  of always-on line curves once we moved the likelihood panel to its
  own area.)
- `app.py` — `python -m statespacecheck_paper.interactive` entry point.

### Notebook driver

`notebooks/figure04_viewer.ipynb` does:

```python
from statespacecheck_paper.interactive import launch
launch(animal_date_epoch="j1620210710_02_r1")
```

### Dependencies

Add to `pyproject.toml` as an optional extra so headless CI does not
pull Qt:

```toml
[project.optional-dependencies]
interactive = [
  "pyqtgraph>=0.13",
  "PySide6>=6.6",
  "zarr>=2.16,<3",
  "pyarrow>=14",
]
```

PySide6 chosen over PyQt6 for the LGPL license.

## Cache layout

`data/cache/figure04_<model>.zarr/` written via `xr.Dataset.to_zarr()`:

```
predictive_posterior   (709321, n_state_bins)     float32  chunks=(8192, n_state_bins)
log_likelihood         (709321, n_state_bins)     float32  chunks=(8192, n_state_bins)
acausal_state_probabilities  (709321, n_states)   float32  chunks=(8192, n_states)
state_bins, state, position, states  coords (and the restored MultiIndex)
```

`n_state_bins` is 256 for Continuous and 512 for ContFrag.

**Chunk size 8192 along time (~16 s at 500 Hz):**

- A 2-s detail window loads one chunk.
- A 20-s window loads ≤ 3 chunks.
- 8192 × 512 × 4 B ≈ 16 MB per chunk — fast read, small enough not to
  thrash.
- No chunking along position/state (a window read needs the full
  position axis anyway).

**Time pyramid (NOT IMPLEMENTED).** The original plan called for
strided ``predictive_posterior_pyramid_*`` / ``log_likelihood_pyramid_*``
arrays plus a viewer-side level-of-detail switch, but the as-built
viewer hard-caps the window to ``MAX_WINDOW_SECONDS = 60 s``
(``viewer.py``), at which point the full-resolution chunked Zarr is
fast enough on its own. Pyramids were dropped during implementation;
the cache builder no longer writes them.

**Event Parquet** (`data/cache/figure04_<model>_events.parquet`):

Columns sorted by `time`:

```
time                  f64     spike time (s)
cell_id               i32
event_hpd_overlap     f32
event_kl_divergence   f32
event_spike_prob      f32
```

Built by walking the dense `(709321, 203)` diagnostic matrices using
`spike_time_ind`/`spike_cell_ind`. Result: ~870K rows × 5 cols × ~28
bytes ≈ 25 MB; loads to RAM in full at startup.

**Sidecars** (`data/cache/figure04_meta.npz`):

- `time` `(709321,) float64`
- `linear_position` `(709321,) float64`
- `place_field_peaks` `(203,) float64`
- `n_cells` int

**Spike times per cell** (`data/cache/figure04_spike_times.npz`): list of
`(n_spikes_cell,) float64` arrays, one per cell, for the raster.

`data/cache/` is gitignored. Cache builder verifies counts (709,321
timepoints; 869,047 spikes; 203 cells) and refuses to overwrite without
`--force`.

## Data flow

```
disk cache (zarr + parquet + npz)
  → DecoderDataSource (lazy zarr handles + in-RAM small arrays)
    → on view-state change:
        UI tick (60 Hz)              → SlicePanel.setData (in-RAM ring buffer)
        window-load tick (debounced) → QThreadPool worker reads zarr chunk
                                         → ImageItem.setImage on main thread
        click on raster/metric       → set pinned_event, recenter
```

Only the SlicePanel does work per UI tick (one array index from a
ring-buffered float32 slice). Everything else is a view-range update or
a debounced async chunk read.

## Per-panel detail

### PosteriorPanel / LikelihoodPanel

- `pg.ImageItem` with `axisOrder='row-major'`. Re-`setImage` on every
  window load with a fresh `(n_t_visible, n_position)` array.
- For ContFrag, the heatmap displays the state-summed `(T, 256)`
  projection (matching current
  `plot_single_model_diagnostics` at
  `src/statespacecheck_paper/real_data_plotting.py:2189-2197`).
- True position overlaid as a `PlotCurveItem` updated from the in-RAM
  `linear_position` slice — cheap.
- Full-resolution chunk read for every window. The plan originally
  split this into a detail mode + a pyramid-backed overview mode, but
  the implementation caps the window to 60 s and serves all reads off
  the chunked Zarr — pyramids were dropped (see "Time pyramid (NOT
  IMPLEMENTED)" above).

### RasterPanel

- Window-local `ScatterPlotItem`. On window load, query
  `events_in_window(sl)` → arrays of `(time, cell_rank)` where
  `cell_rank = argsort(place_field_peaks)[cell_id]`. Color by cell_id
  (or by metric, optionally).
- Benchmark first; if window-local rebuild is the bottleneck, fall
  back to a preloaded all-session scatter (~870K points is fine for
  pyqtgraph but ImageItem texture path is preferred everywhere else).

### MetricPanel × 3 (HPD, KL, spike-prob)

- Window-local `ScatterPlotItem` fed from `events_in_window(sl)`.
- `pxMode=True`, `useCache=True`.
- `sigClicked` → `viewer.handle_event_click(event_row)`.
- Threshold lines from current Figure 4 defaults
  (`hpd_overlap=0.05`, `spike_prob=0.05`).

### SlicePanel (right side, ~30 % width)

- Top plot: a posterior overlay line (the choice — predictive,
  filtered, or smoothed — is exposed via a combo in the controls bar,
  defaulting to predictive) plus the population-likelihood line. Both
  are line curves; the original plan called for a filled likelihood
  area with an alpha slider, but during implementation we moved to
  line curves and dropped the slider since the likelihood now also
  has its own dedicated heatmap panel.
- `InfiniteLine` at `linear_position[t_idx]` for the animal's true
  position.
- On every UI tick, `posterior_curve.setData(position_bins,
  posterior_buffer[t_idx_in_buffer])`. **No xarray hit on the hot
  path** — keep a ring buffer of the last loaded window's float32
  posterior; if the center moves outside the buffered window, a fast
  targeted read fills it.
- **ContFrag-specific:** when the loaded model has a multi-state
  `state_bins` MultiIndex, switch to **stacked-by-state** mode: one
  curve per state, color-coded, with a small toggle for "sum over
  state." Never collapse 512 bins onto a single 1D position axis
  without making the state structure explicit. This preserves the
  semantics of `scripts/sanity_check_figure04a.py` and
  `scripts/sanity_check_figure04b.py`.
- **Pinned-spike overlay:** when an event is pinned, a third curve
  shows `placefield_rates[cell_id]` (the spike's own contribution to
  the likelihood at that cell). Recomputed on click — no precomputed
  `per_spike_likelihood`.
- Corner annotation box for the pinned event:
  `t = ...`, `cell_id = ...`, `event_hpd_overlap = ...`,
  `event_kl_divergence = ...`, `event_spike_prob = ...`.

## Window updates (debounced + double-buffered)

Two timers on the main thread:

- **UI timer** (60 Hz): on slider/scroll change, update `view_state`
  and immediately call `slice_panel.update_center(t_center)`. Cheap.
- **Window-load timer** (single-shot, ~16 ms): coalesces window-range
  changes; fires `data_source.load_posterior(sl)` +
  `load_likelihood(sl)` on a `QThreadPool` worker. Result delivered to
  panels via a `QtCore.Signal` carrying the array. If a newer request
  arrives mid-load, the in-flight result is dropped (compare request
  id).

Slice animation stays silky during scroll even when chunk reads stutter.

## Click semantics

- Click on a raster spike → recenter window on that spike's `t`, pin a
  vertical marker on all 6 rows + the slice panel until the next
  click; slice panel adds the cell's place-field curve and the
  annotation box.
- Click on a metric scatter → same; the marker color matches the
  metric's color in `style.py::COLORS`.
- Manual scroll (slider drag, arrow keys, mouse wheel on time axis) →
  unpins, slice follows center time.

## Keyboard / UX

- `←` / `→` — step center by one timestep.
- `Shift+←` / `Shift+→` — step by one window-width.
- `Space` — play/pause auto-scroll.
- `M` — toggle model (Continuous ↔ Cont-Frag); rebuilds data source.
- `[` / `]` — shrink / grow window.
- `R` — reset to Figure 4a context window
  (`FIGURE_4A_CONTEXT_CENTER`, `FIGURE_4A_CONTEXT_HALF_WIDTH` from
  `scripts/generate_figure04.py`).

## Performance milestone (gate before any UX polish)

`scripts/benchmark_figure04_viewer.py`:

1. Load Continuous cache; open Zarr.
2. Build `DecoderViewer` with posterior + likelihood + raster panels
   only. No metrics, no slice, no clicks.
3. Programmatically drag center time across 60 s of session at slider
   rate 60 Hz, with window width 20 s.
4. Log per-frame: window-load latency, image upload time, total frame
   time, dropped requests.
5. Pass criteria:
   - **median frame time ≤ 16 ms** (60 Hz) for the UI path (slice +
     view-range update).
   - **window-load p95 ≤ 50 ms** for 20 s windows.
   - No memory growth over 60 s of scrubbing.

If this fails, fallbacks (in order):

1. Reduce chunk size along time to 4096.
2. Drop log_likelihood from the heatmap (it is the heavier of the two
   for ContFrag).
3. Pre-decimate to 200 Hz for the heatmap rendering only; slice panel
   stays at full rate.

No metric panels, click handlers, model toggle, or keyboard shortcuts
get added until this passes.

## Implementation order

1. **`interactive/cache.py`** — Zarr writer that opens the existing
   `cont_results.nc` / `cont_frag_results.nc`, restores the
   `state_bins` MultiIndex, writes to `data/cache/figure04_<model>.zarr`.
   (Originally specified pyramid arrays here; dropped during
   implementation — see "Time pyramid (NOT IMPLEMENTED)" above.) Walk
   `fig4_diagnostics.pkl` → events Parquet. Write meta and per-cell
   spike-times sidecars. Verify shapes (709,321; 256/512; 869,047;
   203). CLI: `python -m statespacecheck_paper.interactive.cache build`.
2. **`interactive/data_source.py`** — lazy load, `window_indices`,
   `load_posterior`, `load_likelihood`, `events_in_window`. Pure-Python
   smoke test asserting shapes + that a 2-s window read takes < 30 ms.
3. **`interactive/viewer.py` skeleton** — window, center-time slider,
   three panels (posterior, likelihood, raster), QThreadPool window-load
   worker. (Original plan included a level-of-detail switch tied to the
   pyramid arrays; the as-built viewer caps the window to 60 s and
   reads full-resolution Zarr only — see "Time pyramid (NOT
   IMPLEMENTED)" above.)
4. **Run the benchmark.** Iterate until it passes.
5. **`SlicePanel`** + ring buffer + ContFrag stacked-by-state mode +
   top-overlay choice (predictive / filtered / smoothed).
6. **Metric panels** with window-local event scatters; `sigClicked`
   wiring.
7. **Pinning**: vertical markers across all panels, corner annotation
   box, place-field curve on the slice panel.
8. **Window-width slider, model toggle (rebuild data source),
   play/pause `QTimer`, keybindings**.
9. **`tests/test_interactive_smoke.py`** under `QT_QPA_PLATFORM=offscreen`
   — assert no-crash and correct array slicing. No GUI assertions.
10. **`notebooks/figure04_viewer.ipynb`** driver.

## Open questions / things to revisit

- `per_spike_likelihood` second dim is 248, not 256. Probably valid
  position bins after edge-NaN drop, but worth confirming when we
  recompute spike-likelihood for the pinned-spike overlay so the curves
  line up.
- Whether to rebuild the full event-Parquet in cache.py or stream it in
  chunks (5.2 GB pickle). Streaming is safer for memory; benchmark
  decides.
- ContFrag heatmap default: state-summed (matches current Figure 4) vs.
  state-separated (matches the slice panel). Default to state-summed
  for consistency, but expose a toggle.
