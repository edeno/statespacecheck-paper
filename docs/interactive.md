# Desktop decoder viewer

Run commands from the repository root. Install the viewer dependencies first:

```bash
uv sync --frozen --extra interactive
```

For development, `make sync-dev` installs both the development and viewer extras.
For the recording input and decode-cache requirements, see [reproduction](reproduce.md).

A pyqtgraph desktop app (`statespacecheck_paper.interactive`) renders
the decoder's per-time outputs alongside the diagnostics so you can
scrub through a session, click on a spike to inspect its bin, and
swap between the predictive / filtered / smoothed posterior on the
slice column. The viewer reads from a chunked on-disk cache (Zarr +
Parquet + `.npz` sidecars); it never realises the full posterior in
memory.

Two dataset kinds are supported:

- **Real-data decoder caches** (`continuous` / `continuous_fragmented` models from
  fitted `non_local_detector` decoders).
- **Figure-3 simulation cache** — the simulated demonstration with
  baseline / remap / history-dependent-firing / drift phases plus the
  replay and sparse-population specificity controls (clean-recovery
  windows between).

## Build a cache

For Figure 4, first run `make download-data`. The builder loads the same two
canonical caches the static figure uses, the decode bundle
`{epoch}_fig4_cache.joblib` and the diagnostics bundle
`{epoch}_fig4_diagnostics.joblib` under `<data-dir>/intermediates/`. It refits
the models when the decode cache is absent or stale, and recomputes the
diagnostics when the diagnostics cache is absent or stale; either way it
rewrites the canonical cache it rebuilt. `--force-recompute` refits and
recomputes both regardless. Viewer caches require additional disk space and are
ignored by Git.

```bash
# Real data (figure 4): derives figure04_continuous.zarr +
# figure04_continuous_fragmented.zarr and shared sidecars from the canonical Figure 4
# decode and diagnostics caches. --animal-date-epoch defaults to
# STATESPACECHECK_ANIMAL_DATE_EPOCH, else the published epoch.
uv run --frozen python -m statespacecheck_paper.interactive.cache build \
    --data-dir data \
    --cache-dir data/cache \
    --model both

# Figure-3 simulation: runs the demo simulation + decoder and writes
# simulation.zarr + sidecars, recording Figure 3's flag thresholds (from
# figure03_summary.json) for the metric panels' threshold lines.
uv run --frozen python -m statespacecheck_paper.interactive.cache build-simulated \
    --cache-dir data/cache/simulation
```

## Open the viewer

```bash
# Real-data model (Continuous or Continuous-Fragmented).
uv run --frozen python -m statespacecheck_paper.interactive \
    --cache-dir data/cache --model continuous

# Figure-3 simulation.
uv run --frozen python -m statespacecheck_paper.interactive \
    --cache-dir data/cache/simulation --simulation
```

## Controls

| Action | Binding |
| --- | --- |
| Recenter on a point | Click anywhere on a time-axis panel |
| Pin a spike | Click the spike on the raster or a metric panel |
| Unpin | Click the pinned spike again, or `Esc` |
| Step center by one bin | `←` / `→` |
| Step center by one window | `Shift+←` / `Shift+→` |
| Play / pause auto-scroll | `Space` |
| Slower / faster auto-scroll | `,` / `.` |
| Resize window width | Mouse wheel over a time-axis panel, or `[` / `]` |
| Reset to a 20 s context window | `R` |
| Toggle real-data model | `M` (real-data caches only) |

The slice panel's "Overlay" combo switches the population-likelihood
plot's blue overlay between predictive `p(x_t | y_{1:t-1})`, filtered
`p(x_t | y_{1:t})`, and smoothed `p(x_t | y_{1:T})` distributions.
Smoothed is only available for caches that include `acausal_posterior`
(rebuild via `cache build --force` if the entry is greyed out).
