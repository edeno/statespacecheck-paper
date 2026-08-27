# How each paper figure is produced

This document maps every main-text figure from **manuscript claim → reproduction
command → entry point → configuration → computation (module reading order) →
intermediate data → output → the scientific tests that guard it**. It is the
public, tracked complement to the reproduction table in
[`README.md`](../README.md#reproducing-the-paper-figures): the README gets you to
the entry points; this document connects them to the code and the manuscript.

All commands are run from the repository root in the locked environment:

```bash
uv sync --frozen
uv run python scripts/generate_figureNN.py     # one figure
uv run python scripts/generate_all_figures.py  # all four
# Outputs land in manuscript/figures/main/figureNN.{pdf,png} at 450 DPI.
```

## Architecture at a glance

The code separates **general, figure-agnostic layers** from **per-figure
families**:

- **General layers** (reused by every figure): `simulation` (random walks,
  place-field rates, spike simulators), `decoding` (the Bayesian filter
  `decode_with_diagnostics` + its per-window override mechanism), and
  `diagnostics` (the HPD-overlap / rank-based predictive-p-value / KL-divergence
  computation and containers). `diagnostics` is the dependency-graph leaf.
- **Per-figure families**: `figure01_generation`,
  `figure02_{panels,generation}`, `figure03_{protocol,simulation,summary,plotting,generation}` and
  `figure04_{cache,workflow,layout,generation}`. Each figure is a small set of
  single-responsibility modules rather than one monolith, so an outside reader
  can follow the scientific workflow (configure → simulate/load → decode →
  diagnose → summarize → render).

### Module dependency graph (acyclic)

```text
diagnostics            → (external statespacecheck only; leaf)
decoding               → diagnostics, simulation
simulation             → (numpy/scipy only)

figure03_protocol      → (leaf; no sibling paper module)
figure03_simulation    → figure03_protocol, decoding, diagnostics, simulation
figure03_summary       → figure03_protocol, figure03_simulation, diagnostics
figure03_plotting      → figure03_protocol, figure03_summary, diagnostics, plotting, style
figure03_generation    → figure03_protocol, figure03_simulation, figure03_summary, figure03_plotting, style
generate_figure03.py   → figure03_generation

figure04_decoder       → (leaf; nld construction + Figure4Config)
figure04_place_fields  → (leaf; place-field / marginalized-posterior extraction)
figure04_diagnostics   → diagnostics, figure04_place_fields
figure04_plot_primitives → style
figure04_track_plots   → figure04_plot_primitives
figure04_panels        → diagnostics, figure04_diagnostics, figure04_plot_primitives, figure04_track_plots, plotting, style
figure04_cache         → figure04_decoder (Figure4Config only)
figure04_workflow      → figure04_cache, figure04_decoder, figure04_diagnostics, figure04_place_fields, diagnostics, load_local_data
figure04_layout        → figure04_workflow, diagnostics, figure04_panels, figure04_plot_primitives, figure04_track_plots
figure04_generation    → figure04_workflow, figure04_layout, figure04_cache, figure04_decoder, paths, style
generate_figure04.py   → figure04_generation
```

### Not part of figure generation

- **`interactive/`** — an optional pyqtgraph viewer that consumes the same
  diagnostic results but produces no manuscript figure.
- **`scripts/exploratory/`** — window-finding and sanity-check workflows, not
  canonical generators.

Start from the four `scripts/generate_figureNN.py` entry points.

---

## Figure 1 — Schematic and distribution comparisons

- **Reproduction:** `uv run python scripts/generate_figure01.py` (simulated;
  no external data).
- **Manuscript:** the state-space-model schematic and the
  predictive-vs-likelihood distribution comparisons in the Introduction/Methods.
- **Entry point:** `scripts/generate_figure01.py::main`, which calls
  `figure01_generation.generate_figure01`; the separately testable
  `compose_figure01` returns the in-memory figure.
- **Configuration:** named constants / function arguments inside the generation recipe and
  `schematic.py`; no config dataclass.
- **Computation:** `schematic.py` (graphical model + equation boxes) and
  `plotting.create_distribution_comparison_panel` / `compute_hpd_region`.
- **Output:** `manuscript/figures/main/figure01.{pdf,png}`.
- **Tests:** `tests/test_schematic.py`; `tests/test_figures.py` (entry-point
  contract); `tests/test_plotting.py::TestComputeHpdRegion`,
  `TestCreateDistributionComparisonPanel`.

Trace: `generate_figure01` → `compose_figure01` → semantic axes
(`graphical_model`, `filtering_equations`, and four named consistency cases) →
the `schematic` / `plotting` renderers → `save_figure`.

## Figure 2 — Diagnostic demonstrations

- **Reproduction:** `uv run python scripts/generate_figure02.py` (simulated).
- **Manuscript:** the worked demonstrations of the three diagnostics.
- **Entry point:** `scripts/generate_figure02.py::main`, which calls
  `figure02_generation.generate_figure02`; `compose_figure02` accepts an
  injectable random generator and returns the in-memory figure.
- **Configuration:** named constants / arguments in the generation recipe; per-panel
  renderers live in `figure02_panels.py`.
- **Computation:** `figure02_panels.py` → `plotting.plot_likelihood_columns`
  and the `diagnostics` computations.
- **Output:** `manuscript/figures/main/figure02.{pdf,png}`.
- **Tests:** `tests/test_figures.py` (the figure-2 panel/MC-loop tests);
  `tests/test_diagnostics.py`.

Trace: `create_shared_example(rng)` returns one immutable
`Figure2ExampleData` whose named arrays/scalars feed all nine renderers →
`compose_figure02` arranges semantic mosaic axes such as `hpd_predictive`,
`predictive_histogram`, and `kl_pointwise` → `generate_figure02` saves it.

## Figure 3 — Per-spike diagnostics across an 8-phase simulation

- **Reproduction:** `uv run python scripts/generate_figure03.py` (simulated,
  deterministic under the fixed seed).
- **Manuscript:** the simulation figure showing which model misfits each
  diagnostic detects vs. misses across three misfit conditions and two
  specificity controls.
- **Entry point:** `scripts/generate_figure03.py::main` (the CLI), which calls
  the scientific orchestrator
  `figure03_generation.generate_figure03(config, *, n_realizations)`.
- **Configuration:** `Figure3Config` (frozen; in `figure03_protocol.py`).
  the generation recipe uses the canonical `Figure3Config(drift_momentum=0.88)` and
  `N_REALIZATIONS = 100`; both values are load-bearing for the published PNG.
- **Computation (reading order):**
  `figure03_protocol` (config + phase ladder) →
  `figure03_simulation.run_figure03_simulation` (drives the 8-phase trajectory,
  calls `decoding.decode_with_diagnostics`, which calls `diagnostics`) →
  `figure03_summary.estimate_realization_summary` (pools 100 realizations into
  thresholds + median flag percentages) →
  `figure03_plotting.compose_figure03` (the time-series panel + the panel-(b)
  heatmap).
- **Output:** `manuscript/figures/main/figure03.{pdf,png}`.
- **Tests:** `tests/test_figure03_phases.py` (the higher-level scientific
  contract, including the control-integrity checks that the replay and
  sparse-population controls carry no hidden misfit);
  `tests/test_figure03_{protocol,simulation,summary,plotting,contracts}.py`.

### Figure-3 conditions (executable source of truth: `build_summary_conditions`)

In heatmap order, each condition labeled by which part of the model it perturbs:

1. **Well-specified** — pooled clean-recovery windows (out-of-sample
   false-positive reference); *control*.
2. **Remap** — scrambled place-field identities; *observation-model* misfit.
3. **History-dependent firing** — refractory + bursting spikes; *observation
   model* (temporal), largely missed by the per-spike spatial diagnostics.
4. **Replay** — an out-and-back represented sweep while the animal is immobile;
   *control* (benign decoded-vs-true divergence).
5. **Drift** — AR(1) persistent-velocity trajectory; *transition-model* misfit.
6. **Sparse population** — a quiet ordinary ensemble with a few narrow cells
   firing sparsely; *control* (KL responds; HPD/rank-p stay near baseline).

The numeric phase boundaries live in `Figure3Config.phase_boundaries` — see that
dataclass for values rather than duplicating them here.

### Figure-3 traceability walkthrough (following the typed returns)

`Figure3Config` → `run_figure03_simulation(config)` returns a
`Figure3SimulationResult` (`.true_position`, `.spike_counts`, `.diagnostics: DecodingDiagnostics`,
`.position_bins`, `.sparse_place_field_centers`) → `estimate_realization_summary(config, n_realizations=100)`
returns a `Figure3RealizationSummary` (`.diagnostic_thresholds: DiagnosticThresholds`,
`.median_flag_percentages`) → `compose_figure03(true_position=…, spike_counts=…,
diagnostics=…, diagnostic_thresholds=…, config=…, place_field_centers=…,
median_flag_percentages=…)` returns a `matplotlib` `Figure` → `save_figure` writes
`figure03.{pdf,png}`.

### Manuscript ↔ code vocabulary (Figure 3)

| Code name | Manuscript notation | Meaning / shape |
| --- | --- | --- |
| `true_position` | $x_t$ | true position, shape `(n_time,)` |
| `position_bins` | discretized $x$ grid | position bin centers, shape `(n_bins,)` |
| `spike_counts` | $y_{c,t}$ | integer spike counts, shape `(n_time, n_cells)` |
| `place_field_centers` | $\mu_c$ | per-cell place-field centers, shape `(n_cells,)` |
| `place_field_std` | $\sigma_\mathrm{pf}$ | Gaussian place-field standard deviation |
| `place_field_rate_scale` | $\alpha$ | firing-rate scale on the normalized field |
| `prediction_step_std` | $\sigma_\mathrm{pred}$ | decoder baseline dynamics standard deviation |
| `event_hpd_overlap` | HPD overlap | per-spike prediction/likelihood HPD overlap |
| `event_predictive_pvalue` | rank-based predictive $p$-value | per-spike rank statistic |
| `event_kl_divergence` | KL divergence | per-spike prediction→likelihood KL |

## Figure 4 — Real-data decoder diagnostics

- **Reproduction:** `uv run python scripts/generate_figure04.py`. Add
  `--force-recompute` to re-fit and re-decode both models instead of loading the
  cached decoder outputs (this overwrites the cache; a config / data /
  `non_local_detector` change invalidates the cache automatically). The cache
  fingerprint (`figure04_cache.compute_figure04_cache_fingerprint`) hashes the
  schema version, the full `Figure4Config`, the installed `non_local_detector`
  version, and the **content hashes of all five input exports** — so replacing an
  export under the same `animal_date_epoch` invalidates the cache too.
- **Manuscript:** the real hippocampal-recording panels comparing the Continuous
  and Continuous-Fragmented decoders (and the whole-session hexbin summary).
- **Entry point:** `scripts/generate_figure04.py::main` (the CLI), which calls
  `figure04_generation.generate_figure04(*, use_cache)`.
- **Configuration:** `Figure4Config` (in `figure04_decoder.py`) plus the fixed
  `FIGURE4_DIAGNOSTIC_THRESHOLDS = {"hpd_overlap": 0.05, "predictive_pvalue": 0.05}`
  and `FIGURE4_DETAIL_WINDOW = Figure4DetailWindow(center_index=193069,
  half_width_samples=500)` in `figure04_generation.py`. The explicit detail
  window centers the manuscript panels on a KL-divergence spike during
  immobility at a reward well and spans about two seconds total. `Figure4Config`
  is split into three scoped parts:
  a `Figure4DecoderConfig` — `position_std`, `position_bin_size_cm`,
  `sampling_frequency_hz`, threaded into environment/model construction so they
  genuinely drive the decode; a `Figure4Provenance` holding the
  `non_local_detector`-default decode-shaping values (`movement_var`, the ContFrag
  transition/initial-condition/concentration/regularization, and the dependency
  version), recorded and drift-guard pinned but not injected (faithfully injecting
  them would rebuild the nested transition grid and hit the concentration-default
  split); and a `Figure4ExecutionConfig` holding `block_size`, a performance/memory
  knob that does **not** change the decode result (the KDE density is identical for
  any `block_size`) and is therefore excluded from the cache fingerprint. See the
  `Figure4Config` docstring.
- **Computation (reading order):**
  `figure04_generation` (recipe) → `figure04_workflow.prepare_figure04_render_data`
  (loads the recording, loads a fingerprint-matching cache or fits/decodes via
  `figure04_decoder`/`figure04_place_fields`, computes `figure04_diagnostics`) →
  `figure04_workflow.compute_figure04_summary` (typed manuscript scalars) →
  `figure04_layout.compose_figure04`
  (artist arrangement) → `save_figure`.
- **Intermediate data — the honest boundary.** Figure 4 is reproduced **from
  pre-exported derived inputs onward**, not from raw acquisition. The loader
  `load_local_data.load_neural_recording_from_files` reads five files from the
  data directory and returns a validated `NeuralRecordingData`:

  | File | `NeuralRecordingData` field | Contents |
  | --- | --- | --- |
  | `{epoch}_position_info.pkl` | `position_info` | time-indexed position DataFrame (seconds; positions in cm) |
  | `{epoch}_HPC_spike_times.pkl` | `spike_times` | per-cell spike-time arrays (seconds) |
  | `{epoch}_track_graph.pkl` | `track_graph` | `networkx` track-graph structure |
  | `{epoch}_linear_edge_order.pkl` | `linear_edge_order` | linearization edge order |
  | `{epoch}_linear_edge_spacing.pkl` | `linear_edge_spacing` | edge spacing (cm) |

  The raw-recording → these-five-exports step (DANDI / Spyglass / MountainSort /
  linearization) is **not implemented in this repository**; obtain the recording
  per the manuscript's Data Availability statement
  ([Comrie et al. 2024](https://doi.org/10.1101/2024.09.23.613567)) and place the
  exports under `data/` (or set `STATESPACECHECK_DATA_PATH`). The expensive decode
  is cached as a single joblib bundle under `data/intermediates/{epoch}_fig4_cache.joblib`,
  gated by a provenance fingerprint (schema version + `Figure4Config` +
  data identifier + installed `non_local_detector` version).
- **Output:** `manuscript/figures/main/figure04.{pdf,png}`.
- **Tests:** `tests/test_figure04_decoder.py::TestFigure4ConfigMatchesManuscript`
  (config matches the manuscript decoder parameters);
  `tests/test_figure04_{cache,workflow,layout,generation}.py` (orchestration);
  `tests/test_figure04_{diagnostics,place_fields}.py` (the analysis leaves) and
  `tests/test_figure04_{plot_primitives,track_plots,panels}.py` (the plotting
  leaves — panels covers both composite figures and the extracted row renderers);
  `tests/test_load_local_data.py` (the `NeuralRecordingData` contract).

### Figure-4 traceability walkthrough (following the typed returns)

`Figure4Config` + `Figure4Paths` → `prepare_figure04_render_data(config, paths, use_cache=…)`
loads a `NeuralRecordingData` and returns a `Figure4RenderData`
(`.recording`, `.time`, `.head_position`, `.linear_position`,
`.decode_results: Figure4DecodeResults`) — the decode results come from a
fingerprint-matching cache (`Figure4DecodeResults.from_cache_payload`) or a fresh
fit/decode (`_compute_figure04_decode_results`) → `compute_figure04_summary`
returns a `Figure4Summary` (per-decoder `Figure4DiagnosticMeans` plus typed
`FlagConfusion` counts) → `format_figure04_summary` handles CLI text separately
→ `compose_figure04(render_data, diagnostic_thresholds=…,
detail_window=Figure4DetailWindow(…))`
returns a `Figure4Composition` (`.figure`, `.bbox_inches`) → `save_figure` writes
`figure04.{pdf,png}` with that custom crop.

### Manuscript ↔ code vocabulary (Figure 4)

The decoder-parameter names deliberately match the manuscript and the external
`non_local_detector` model: `position_std` and `movement_var` (see `Figure4Config`
and the Methods) keep their manuscript spellings rather than being renamed. The
three diagnostic quantities are the same `event_hpd_overlap` /
`event_predictive_pvalue` / `event_kl_divergence` as in Figure 3.
