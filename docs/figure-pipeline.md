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
uv run python scripts/generate_all_figures.py  # all figures + supplementary summaries
uv run python scripts/emit_reported_values.py  # refresh the manuscript's numbers
make -C manuscript                           # build the paper
# Figures: manuscript/figures/main/figureNN.{pdf,png} at 450 DPI;
# supplementary figure and summaries under manuscript/figures/supplementary/.
# Prose macros: manuscript/reported_values.tex; paper: manuscript/main.pdf.
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
  `figure02_{panels,generation}`,
  `figure03_{protocol,simulation,summary,plotting,generation,sensitivity}`,
  `figure04_{cache,workflow,layout,generation}` and the Figure-4 supplement
  (`figure04_{broadening,supplement_plotting,supplement_generation}`). Each
  figure is a small set of single-responsibility modules rather than one
  monolith, so an outside reader can follow the scientific workflow
  (configure → simulate/load → decode → diagnose → summarize → render).

### Module dependency graph (acyclic)

```text
diagnostics            → (external statespacecheck only; leaf)
decoding               → diagnostics, simulation
simulation             → (numpy/scipy only)
scientific_artifacts   → (standard library + numpy only)
number_format          → (standard library only; leaf)
reported_values        → number_format; reads summary JSONs
emit_reported_values.py → reported_values

figure03_protocol      → (leaf; no sibling paper module)
figure03_simulation    → figure03_protocol, decoding, diagnostics, simulation
figure03_summary       → figure03_protocol, figure03_simulation, diagnostics
figure03_plotting      → figure03_protocol, figure03_summary, diagnostics, number_format, plotting, style
figure03_generation    → figure03_protocol, figure03_simulation, figure03_summary, figure03_plotting, scientific_artifacts, style
figure03_sensitivity   → figure03_generation, figure03_protocol, figure03_summary, scientific_artifacts
generate_figure03.py   → figure03_generation
generate_figure03_sensitivity.py → figure03_sensitivity

figure04_decoder       → (leaf; nld construction + Figure4Config)
figure04_place_fields  → (leaf; place-field / marginalized-posterior extraction)
figure04_diagnostics   → diagnostics, figure04_place_fields
figure04_plot_primitives → style
figure04_track_plots   → figure04_plot_primitives
figure04_panels        → diagnostics, figure04_diagnostics, figure04_plot_primitives, figure04_track_plots, plotting, style
figure04_cache         → figure04_decoder (Figure4Config only)
figure04_workflow      → figure04_cache, figure04_decoder, figure04_diagnostics, figure04_place_fields, diagnostics, load_local_data
figure04_layout        → figure04_workflow, diagnostics, figure04_panels, figure04_plot_primitives, figure04_track_plots
figure04_generation    → figure04_workflow, figure04_layout, figure04_cache, figure04_decoder, paths, scientific_artifacts, style
generate_figure04.py   → figure04_generation
figure04_broadening    → figure04_workflow, figure04_place_fields, diagnostics
figure04_supplement_plotting   → figure04_broadening, style
figure04_supplement_generation → figure04_broadening, figure04_supplement_plotting, figure04_workflow, figure04_cache, figure04_decoder, paths, scientific_artifacts, style
generate_figure04_supplement.py → figure04_supplement_generation
```

### Not part of figure generation

- **`interactive/`** — an optional pyqtgraph viewer that consumes the same
  diagnostic results but produces no manuscript figure.

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

## Figure 3 — Per-spike diagnostics across a 10-phase simulation

- **Reproduction:** `uv run python scripts/generate_figure03.py` (simulated,
  deterministic under the fixed seeds; realizations run in parallel with
  joblib). `uv run python scripts/generate_figure03_sensitivity.py` reruns the
  pipeline for the sensitivity settings.
- **Manuscript:** the simulation figure showing how each diagnostic responds
  to four misfit conditions and two controls under a decoder whose model is
  known exactly, with independently calibrated thresholds, realized null
  rates, and the decoder's accuracy/coverage/region size per condition.
- **Entry point:** `scripts/generate_figure03.py::main` (the CLI), which calls
  the scientific orchestrator
  `figure03_generation.generate_figure03(config, *, n_realizations,
  n_calibration_realizations, n_jobs)`.
- **Configuration:** `Figure3Config` (frozen; in `figure03_protocol.py`).
  The generation recipe uses the default `Figure3Config()`: the exactly
  matched discrete trajectory model, 95% HPD coverage, drift momentum `0.88`,
  the pinned history rate-matching gain `DEFAULT_HISTORY_RATE_MATCHING_GAIN`,
  and the spread, heterogeneous-rate sparse population; `N_REALIZATIONS = 100`
  evaluated realizations (seeds 1–100) and `N_CALIBRATION_REALIZATIONS = 100`
  matched-null calibration sessions (seeds 1001–1100). All are load-bearing
  for the published PNG and summary.
- **Computation (reading order):**
  `figure03_protocol` (config + phase ladder) →
  `figure03_simulation.run_matched_null_simulation` (calibration sessions
  drawn from the decoder's own initial law, transition matrix, and rate
  tables) and `run_figure03_simulation` (the 10-phase session; both call
  `decoding.decode_with_diagnostics`, which calls `diagnostics`) →
  `figure03_summary.estimate_calibration_thresholds` (pooled calibration
  quantiles, realized null rates, HPD tie fraction, rank-tail check) →
  `figure03_summary.estimate_realization_summary` (per-condition flags,
  per-realization distributions, accuracy/coverage/region sizes, replay
  accuracy against the represented trajectory, per-phase spike rates,
  sparse-cell denominators) →
  `figure03_plotting.compose_figure03` (the time-series panel, the panel-(b)
  heatmap with accuracy rows, and the panel-(c) per-realization
  distributions).
- **Output:** `manuscript/figures/main/figure03.{pdf,png}` plus
  `figure03_summary.json` (schema 8), containing the full configuration, seed
  ranges for evaluation and calibration, the calibration record, explicit
  inclusive flag rules, metric/condition order, median and pooled flag
  percentages, per-realization percentages and event counts, the four
  accuracy rows (median and per realization), replay represented-trajectory
  accuracy, per-phase ordinary spike rates, sparse-cell flags and counts, the
  `kl_clip_impact` record (how many events' KL values and flag decisions
  differ between the clipped place-field tables the decoder uses and the
  unclipped Gaussian, for the evaluated realizations and the calibration
  sessions), and source/dependency-lock provenance. The sensitivity recipe writes
  `manuscript/figures/supplementary/figure03_sensitivity_summary.json`
  (schema 1) with a compact summary per setting.
- **Tests:** `tests/test_figure03_phases.py` (the scientific contract: the
  matched phases follow the decoder's transition on the grid, the reflected
  map is coherent and unflagged but wrong, the sparse regime is matched and
  KL-only, the history gain matches the marginal rate, calibration is
  independent and the rank p-value is super-uniform);
  `tests/test_figure03_{protocol,simulation,summary,plotting,contracts,generation}.py`.

### Figure-3 conditions (executable source of truth: `build_summary_conditions`)

In heatmap order, each condition labeled by which part of the model it perturbs:

1. **Matched null** — the opening baseline, drawn exactly from the decoder's
   initial law, transition matrix, and rate tables; *reference*.
2. **Recovery** — the four clean-recovery windows pooled (replay carved out);
   matched dynamics but each begins with a transient; *reference*.
3. **Remap** — one fixed incoherent permutation of place-field identities;
   *observation-model* misfit.
4. **History-dependent firing** — post-spike suppression + bursting at a
   rate-matched gain; *observation model* (temporal), largely missed by the
   per-spike spatial diagnostics.
5. **Replay** — a deterministic out-and-back represented sweep at a fourfold
   rate gain (supplied to the decoder) while the animal is immobile;
   *control* (accuracy is scored against both the physical and the
   represented trajectory).
6. **Drift** — AR(1) persistent-velocity trajectory; *transition-model* misfit.
7. **Reflected map** — every field, sparse cells included, mirrored about
   the track midpoint (the baseline rate table with its rows reversed), a
   coherent wrong map; *observation-model* misfit that is near-null under
   every diagnostic while the decode is a mirror image.
8. **Sparse population** — matched trajectory with a quiet ordinary ensemble
   and sparse narrow cells; *control* under an exactly correct model (KL
   responds; HPD/rank-p rarely).

The numeric phase boundaries live in `Figure3Config.phase_boundaries` — see that
dataclass for values rather than duplicating them here.

### Figure-3 traceability walkthrough (following the typed returns)

`Figure3Config` → `estimate_calibration_thresholds(config, n_calibration_realizations=100)`
returns a `Figure3Calibration` (`.diagnostic_thresholds: DiagnosticThresholds`,
`.pooled_null_flag_percentages`, `.rank_pvalue_tail_percentages`, ...) →
`run_figure03_simulation(config)` returns a `Figure3SimulationResult`
(`.true_position`, `.represented_position`, `.spike_counts`,
`.diagnostics: DecodingDiagnostics`, `.position_bins`,
`.sparse_place_field_centers`) → `estimate_realization_summary(config,
calibration=…, n_realizations=100)` returns a `Figure3RealizationSummary`
(`.median_flag_percentages`, `.flag_percentages_by_realization`,
`.event_counts_by_realization`, `.pooled_flag_percentages`,
`.median_decoding_accuracy`, `.replay_represented_accuracy`, ...) →
`compose_figure03(...)` returns a `matplotlib` `Figure` → `save_figure` writes
`figure03.{pdf,png}` while `write_json_artifact` writes the same summary values
and their configuration to `figure03_summary.json`.

### Manuscript ↔ code vocabulary (Figure 3)

| Code name | Manuscript notation | Meaning / shape |
| --- | --- | --- |
| `true_position` | $x_t$ outside replay; $z_t$ during replay | physical position, shape `(n_time,)`; the physical-position decoding error is measured against this trajectory |
| `represented_position` | $x_t$ | represented position, shape `(n_time,)`; equals `true_position` except during the replay sweep, where replay accuracy is measured against it |
| `position_bins` | discretized $x$ grid | position bin centers, shape `(n_bins,)` |
| `spike_counts` | $y_{c,t}$ | integer spike counts, shape `(n_time, n_cells)` |
| `place_field_centers` | $\mu_c$ | per-cell place-field centers, shape `(n_cells,)` |
| `place_field_std` | $\sigma_\mathrm{pf}$ | Gaussian place-field standard deviation |
| `place_field_rate_scale` | $\alpha$ | expected-count scale multiplying the normalized Gaussian field |
| `prediction_step_std` | $\sigma_\mathrm{pred}$ | decoder baseline dynamics standard deviation |
| `trajectory_model` | matched discrete transition vs reflected walk | `"discrete_matched"` samples the decoder's own grid transition matrix (exact null); `"continuous_reflected"` is the approximate sensitivity benchmark |
| `history_rate_matching_gain` | gain on $\alpha$ in the history window | rescales the history generator so its marginal rate matches the Poisson baseline |
| `hpd_coverage` | $\alpha$ | HPD coverage of the regions the overlap compares |
| `event_hpd_overlap` | HPD overlap | per-spike prediction/likelihood HPD overlap |
| `event_predictive_pvalue` | rank-based predictive $p$-value | per-spike rank statistic |
| `event_kl_divergence` | KL divergence | per-spike prediction→likelihood KL |

### Rates and expected counts

The Methods distinguish the firing rate $\lambda_c(x)$ from the expected count
$m_c(x)=\lambda_c(x)\Delta t$ in a bin of width $\Delta t$. The simulation's
`place_field_rates` and decoder rate tables already contain these expected
counts per step: they go directly into the Poisson distribution. In Figure 3,
$\Delta t=1$ ms and $m_c(x)=\alpha\phi(x\mid\mu_c,\sigma_{\mathrm{pf}}^2)$.
Convert to Hz by dividing by the bin width in seconds; do not multiply these
Poisson inputs by the bin width again.

In the likelihood factorization, the population exposure is
$\exp[-\Lambda(x)\Delta t]$ and each event contributes
$\lambda_c(x)\Delta t$, where $\Lambda$ is the total event rate. The common
bin width cancels when normalizing a single-event likelihood over state and
when forming the event-weighted predictive mark distribution. Thus the
diagnostic normalization can use either rates or expected counts with a common
bin width. The manuscript also states the corresponding clusterless convention:
$\lambda(y,x)$ is a rate per unit mark volume, and its integral over marks is
$\Lambda(x)$.

## Figure 4 — Real-data decoder diagnostics

- **Reproduction:** `uv run python scripts/generate_figure04.py`. Add
  `--force-recompute` to re-fit and re-decode both models instead of loading the
  cached decoder outputs (this overwrites the cache; a config / data /
  `non_local_detector` change invalidates the cache automatically). The cache
  fingerprint (`figure04_cache.compute_figure04_cache_provenance`) hashes the
  schema version, the decoder and provenance parts of `Figure4Config`, the
  data identifier, the installed `non_local_detector`
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
  immobility at a reward well and spans about two seconds total; it was chosen
  by hand as a candidate replay event, not an independently detected one.
  `Figure4Config` is split into four scoped parts:
  a `Figure4DecoderConfig` — `position_std`, `position_bin_size_cm`,
  `sampling_frequency_hz`, threaded into environment/model construction so they
  genuinely drive the decode; a `Figure4Provenance` holding the
  `non_local_detector`-default decode-shaping values (`movement_var`, the ContFrag
  transition/initial-condition/concentration/regularization, and the dependency
  version), recorded and drift-guard pinned but not injected (faithfully injecting
  them would rebuild the nested transition grid and hit the concentration-default
  split); and a `Figure4ExecutionConfig` holding `block_size`, a performance/memory
  knob that does **not** change the decode result (the KDE density is identical for
  any `block_size`) and is therefore excluded from the cache fingerprint; and a
  `Figure4DiagnosticsConfig` (`hpd_coverage`, `event_selection`) hashed into the
  diagnostics fingerprint only. See the `Figure4Config` docstring. The
  supplement's analysis settings (uniform-mixture weights, coverage levels,
  rate groups, speed cutoff) live in `Figure4BroadeningConfig`
  (`figure04_broadening.py`).
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
  from the DANDI Archive as dandiset
  [001942](https://dandiarchive.org/dandiset/001942)
  ([Comrie et al. 2024](https://doi.org/10.1101/2024.09.23.613567)) and place the
  exports under `data/` (or set `STATESPACECHECK_DATA_PATH`). The expensive decode
  is cached as a single joblib bundle under `data/intermediates/{epoch}_fig4_cache.joblib`,
  gated by the provenance fingerprint described above. Execution-only settings
  are excluded; all five input-content hashes are included.
- **Output:** `manuscript/figures/main/figure04.{pdf,png}` plus
  `figure04_summary.json`, containing configuration and dataset identifiers,
  explicit inclusive flag rules, whole-session means, flag-confusion counts,
  rescue rates, source/dependency-lock provenance, the decode-cache fingerprint,
  and SHA-256 checksums for all five input exports.
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
returns a `Figure4Summary` (`.n_units`, per-decoder `Figure4DiagnosticMeans`, and
typed `FlagConfusion` counts) → `format_figure04_summary` handles CLI text separately
→ `compose_figure04(render_data, diagnostic_thresholds=…,
detail_window=Figure4DetailWindow(…))`
returns a `Figure4Composition` (`.figure`, `.bbox_inches`) → `save_figure` writes
`figure04.{pdf,png}` with that custom crop, while `write_json_artifact` writes the
typed summary to `figure04_summary.json`.

The optional interactive viewer derives its Zarr/Parquet/NPZ layout from this
same `Figure4RenderData` via `interactive.cache.build_figure04_viewer_cache`.
It does not require a second set of NetCDF results or fitted-model pickles.
Rebuild it (`--force`) whenever the diagnostics cache changes; the viewer
artifacts carry no fingerprint of their own, so a stale build silently
disagrees with the canonical summary.

### Figure-4 supplement — broadening, coverage, rate, and behavior

- **Reproduction:** `uv run python scripts/generate_figure04_supplement.py`
  (reads the two Figure-4 caches; refits nothing; about six minutes).
- **Computation:** `figure04_supplement_generation.generate_figure04_supplement`
  → `prepare_figure04_render_data` (cached) →
  `figure04_broadening.compute_region_size_and_broadening` (per-event 95/80/50%
  predictive-region sizes for both models, likelihood region sizes, flags and
  rescue at each coverage, and the uniform-mixture counterfactual
  `(1 - w) P + w U` at each weight, all in bounded event chunks from the
  memory-mapped predictions) and
  `compute_rate_and_behavior_association` (per-unit session rates and flag
  fractions, rate-group contributions, a constant mark-frequency baseline,
  low-rate contributions to rescued/newly flagged rank events, and
  immobile-versus-moving strata) →
  `figure04_supplement_plotting.compose_figure04_supplement`.
- **Output:** `manuscript/figures/supplementary/figure04_supplement.{pdf,png}`
  and `figure04_supplement_summary.json` (schema 1), which also carries the
  decode/diagnostics cache provenance.
- **Interpretation guard:** the uniform mixture modifies the diagnostic's
  input on the same events; it is not a refitted alternative decoder, and the
  summary labels it as such.

## Machine-readable summary schema

`figure03_summary.json` uses schema version 6, `figure04_summary.json` schema
version 4, and the two supplementary summaries schema version 1. The Figure-3
schema records the independent calibration (`calibration`: seeds, session
length, pooled event count, realized null flag percentages pooled and per
session, the HPD tie percentage, and rank-tail percentages at 0.01/0.05/0.10),
the accuracy block (`accuracy_metric_order`: median absolute error, filtered
95% HPD coverage percent, filtered and predictive HPD-region sizes;
`median_decoding_accuracy` is `(4, n_conditions)` and
`decoding_accuracy_by_realization` `(n_realizations, 4, n_conditions)`),
`flag_percentages_by_realization` and `event_counts_by_realization` (the
per-realization distributions and denominators behind the medians; a
realization with no events in a column carries `null`),
`pooled_flag_percentages` (event-weighted), `replay_represented_accuracy`,
`phase_ordinary_spike_rates_hz`, the `sparse_population` block (sparse-cell
denominators), and `matched_null_rank_pvalue_tail_percentages`. It also
records `median_flag_percentage_standard_errors` and
`median_decoding_accuracy_standard_errors`: approximate standard errors of the
medians, estimated from order-statistic interval widths. These describe the
uncertainty in the aggregated medians under repeated simulation with the same
configuration, not the spread of individual realizations (which the
per-realization arrays report directly), and do not set reported precision.
The summary also records the calibration-threshold provenance quoted in the
Methods. The Figure-4 schema records `dataset.n_units` alongside the recording
identifier. The
`flag_rules` object binds each numeric threshold to its executable semantics:
`less_than_or_equal` means a value is flagged when `value <= threshold`, and
`greater_than_or_equal` means it is flagged when `value >= threshold`. Keeping
the operator and threshold in one record prevents consumers from guessing
whether a boundary is strict or inclusive.

All summaries contain `provenance.source`, with the installed
`statespacecheck-paper` version, a deterministic SHA-256 digest of every Python
file under `src/statespacecheck_paper`, and the SHA-256 digest of `uv.lock`.
The digest excludes timestamps, generated outputs, and absolute paths, so clean
checkouts of identical source produce the same identity.

The source digest includes comments and docstrings. After a documentation-only
source edit, verify that executable code is unchanged (for example, compare
Python syntax trees with docstrings removed). Then refresh only
`provenance.source` in every committed summary using
`scientific_source_provenance` and `write_json_artifact`, preserving all other
fields, and rerun `uv run python scripts/emit_reported_values.py`. If scientific
code or inputs changed, regenerate the affected figures and summaries through
their canonical entry points instead of relabeling existing results.

Figure 4 and its supplement also contain `provenance.figure04_decode_cache`.
Its `fingerprint_sha256` is the same identity used to accept or reject the
expensive decoder cache, `diagnostics_fingerprint_sha256` the identity of the
diagnostics cache (with the diagnostics configuration and the installed
`statespacecheck` version), and the record includes the installed
`non_local_detector` version plus the content SHA-256 of each of the five
named exports. Canonical artifact generation fails if any input checksum is
missing. Thus a summary can be traced to the exact derived inputs even when
those large files are distributed separately.

### Manuscript ↔ code vocabulary (Figure 4)

The decoder-parameter names deliberately match the manuscript and the external
`non_local_detector` model: `position_std` and `movement_var` (see `Figure4Config`
and the Methods) keep their manuscript spellings rather than being renamed. The
three diagnostic quantities are the same `event_hpd_overlap` /
`event_predictive_pvalue` / `event_kl_divergence` as in Figure 3.

## From summary to prose: the reported-value macros

The manuscript's reported configuration and headline analysis statistics are
generated from the summaries. `main.tex` inputs
`manuscript/reported_values.tex`, a generated file of `\newcommand` definitions
(`\Sim...` for the Figure-3 simulation, `\Rec...` for the Figure-4 recording),
so the chain runs **code → summary JSON → macro file → prose**.
`scripts/emit_reported_values.py` (recipe:
`statespacecheck_paper.reported_values`) reads only the four committed summaries,
so these values reach the paper through an artifact. Upstream acquisition and
sorting parameters, which have no artifact in this repository, remain stated
directly in the Methods.

Prose precision follows the policy defined in the
[`reported_values` module docstring](../src/statespacecheck_paper/reported_values.py):

| Quantity | Prose format | Purpose |
| --- | --- | --- |
| Decoding errors | Two significant figures | Supports the ratios discussed in the Results |
| Flag percentages | Nearest whole percent | Supports comparisons described as substantial, low, or modest |
| Rescue rates | Nearest whole percent, with exact counts alongside | Describes this recording |
| Realized null rates, rank tails, behavior-stratified flag rates, correlations | Two significant figures (correlations: two decimals) | Read against nominal levels such as 1% and 5%, so 1.6% must not print as 2% |
| Sensitivity and coverage tables | Table rows emitted as one macro each | Keeps every cell on the artifact chain |
| Derived constants introduced with “approximately” | Two significant figures | Communicates their approximate scale |
| Exact counts and configured parameters | In full | Preserves the specified count or setting |
| Summary JSON values | Full numerical precision, retaining median SEs | Preserves the analysis detail |

Decoding errors remain bare in the prose, without adding “about” before each
value. Figure 3b's heatmap cells and error row round with the same two
functions the emitter uses (`number_format.whole_percent` and
`number_format.significant`), so a value cannot read differently in the panel
and in the text beside it. Neither a binomial SE nor the published median SEs
controls formatting. A distribution plot of the per-realization Figure-3
results remains a [follow-up](../TODO.md).

`tests/test_reported_values.py` holds the guards: the committed macro file must
byte-match a fresh render of the committed summaries, every macro must actually
appear in `main.tex`, and each mode or condition must keep its own value rather
than borrowing a neighbour's. Tests also cover zero and power-of-ten rounding,
reject lossy rounding of exact parameters and percentile levels, and verify
that changing SEs does not change the formatted values.

`tests/test_reported_statistics_artifacts.py` checks the committed summaries'
schemas, reference values, configuration, and provenance. These checks do not
rerun the full scientific analyses. The manuscript Makefile tracks
`reported_values.tex` as a build prerequisite but does not run the emitter;
regenerate the macros explicitly after changing a summary.
