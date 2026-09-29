# How each paper figure is produced

This document maps every main-text figure from **manuscript claim → reproduction
command → entry point → configuration → computation (module reading order) →
intermediate data → output → the scientific tests that guard it**. It is the
implementation guide. [Reproduction](reproduce.md) covers requirements,
commands, runtime, and storage; this document connects results to the code and
the manuscript.

Run the individual commands below from the repository root after installing
the locked environment. The [complete reproduction recipe](reproduce.md#reproduce-the-full-paper)
downloads the recording input before generating all four figures and building
the manuscript. Figures are exported as PDF and 450-DPI PNG.

## Start here

### The four entry points

Each figure has one script under `scripts/`, a thin command-line adapter that
calls one importable recipe in `src/statespacecheck_paper/`:

| Figure | Script | Recipe | Data |
| --- | --- | --- | --- |
| 1 — schematic and distribution comparisons | `generate_figure01.py` | `figure01_generation.generate_figure01` | none |
| 2 — the three diagnostics on one example | `generate_figure02.py` | `figure02_generation.generate_figure02` | seeded example |
| 3 — simulation study | `generate_figure03.py` | `figure03_generation.generate_figure03` | seeded simulation |
| 4 — hippocampal recording | `generate_figure04.py` (`--force-recompute`) | `figure04_generation.generate_figure04` | archived input file |

### From the Methods to the code

The diagnostics themselves are computed by the separate
[`statespacecheck`](https://github.com/edeno/statespacecheck) package. Both
analyses reach it through one paper-side wrapper,
`diagnostics.compute_spike_event_diagnostics_from_rates`, which calls
`statespacecheck.event_diagnostics` and returns a `SpikeEventDiagnostics` with
one value per spike event. Every per-spike diagnostic in Figures 3 and 4 goes
through it (Figure 2's single worked example calls `statespacecheck` directly).
Figure 3 calls the wrapper from `decoding.decode_with_diagnostics`, once with the
baseline expected-count table and again for the spikes inside each decoder
override window that swaps it; Figure 4 calls it from
`figure04_diagnostics.compute_results_diagnostics`. Equation numbers are those
of `manuscript/main.tex` (labels in parentheses): Eqs. 1–10 are in the Methods,
and Eq. 11 is in the simulation study's generative model.

| Paper quantity | Equation | `statespacecheck` function | Paper-side name |
| --- | --- | --- | --- |
| Event factorization of the Poisson likelihood | Eq. 1 (`eq:poisson_event_factorization`) | — | Figure 3's decoder forms $p(\mathbf{n}_k \mid x_k)$ from a table of expected counts per step (`decoding.decode_with_diagnostics`) |
| One-step predictive distribution $P_k(x_k)=p(x_k\mid y_{1:k-1})$ | unnumbered, after Eq. 1 | input to every function below | `predictive` (see the vocabulary below) |
| Normalized single-event likelihood $Q_{k,j}$ | Eq. 2 (`eq:single_event_likelihood`) | `event_likelihood` | `event_likelihood`, shape `(n_spikes, n_bins)` |
| HPD overlap (overlap coefficient of the two HPD regions) | Eq. 3 (`eq:hpd`) | `hpd_overlap`; regions from `highest_density_region` | `event_hpd_overlap`; coverage `diagnostics.HPD_COVERAGE` (0.95, macro `\HpdCoveragePercent`) |
| Event-weighted predictive $P_k^{\mathrm{event}}$, cell identity $\pi_c$, predictive cell probabilities $f_{\mathrm{pred}}(c)$ | Eqs. 4–6 | `event_weighted_predictive`, `predictive_mark_probabilities` | — |
| Rank-based predictive $p$-value, exact for sorted spikes | Eqs. 8–9 (`eq:predictive_application`, `eq:predictive_application_sorted`) | `mark_predictive_pvalue` | `event_predictive_pvalue` |
| Monte Carlo predictive $p$-value (clusterless form, Eq. 7; Figure 2b) | Eqs. 7–8 | `monte_carlo_mark_pvalue` | used only by `figure02_panels` |
| KL divergence $D_{\mathrm{KL}}(P_k\,\|\,Q_{k,j})$ | Eq. 10 (`eq:kl`) | `kl_divergence` | `event_kl_divergence` |
| Simulated expected counts $m_c(x)=\lambda_c(x)\Delta t=\alpha\phi(x\mid\mu_c,\sigma_{\mathrm{pf}}^2)$ | Eq. 11 (`eq:rate`, simulation study) | — | `simulation.place_field_expected_counts` |

### Flag thresholds

| Figure | Rule | Where it is defined |
| --- | --- | --- |
| 3 | HPD overlap flagged at or below the 1st percentile, KL divergence at or above the 99th percentile, of baseline values pooled over every spike event before step 6000 of all 100 realizations. The committed HPD-overlap threshold is exactly 0, so a spike is flagged only when its two HPD regions do not overlap at all | quantiles `diagnostics.BASELINE_HPD_OVERLAP_QUANTILE` and `BASELINE_KL_DIVERGENCE_QUANTILE`; rule `diagnostics.compute_baseline_diagnostic_thresholds`; pooling `figure03_summary.estimate_realization_summary` |
| 3 | Predictive $p$-value flagged at or below a fixed 0.05 | `diagnostics.FIXED_PREDICTIVE_PVALUE_CUTOFF` |
| 4 | Fixed cutoffs: HPD overlap ≤ 0.05 and predictive $p$-value ≤ 0.05; KL divergence is not thresholded | `figure04_protocol.FIGURE04_DIAGNOSTIC_THRESHOLDS` |

Every comparison is inclusive; `diagnostics.METRIC_FLAG_DIRECTIONS` gives each
metric's worse-fit direction and `diagnostics.flag_mask` applies it. Both
summaries record the numeric thresholds with their comparison under
`flag_rules`. Because ties at a threshold are flagged, a pooled-baseline
threshold can flag more of the baseline than its quantile level: about 1.4% of
the Figure-3 baseline spikes have disjoint HPD regions, so the 1st percentile is
0 and all of them are flagged. The Figure-3 summary records the flagged share of
the pooled baseline for each metric as
`threshold_provenance.<metric>.baseline_flagged_fraction`.

### Decoders

- **Figure 3:** the grid Bayesian filter `decoding.decode_with_diagnostics`,
  with a Gaussian random-walk transition
  (`simulation.gaussian_transition_matrix`, step standard deviation
  `Figure3Config.prediction_step_std`) and Poisson place-field expected counts
  per step from `figure03_simulation.build_figure03_expected_count_tables`.
  Decoder override windows (`decoding.DecoderOverrideWindow`) swap in the
  scrambled fields of the remap misfit and the correct fields of the replay and
  sparse-population controls.
- **Figure 4:** the `non_local_detector` models `SortedSpikesDecoder`
  (Continuous) and `ContFragSortedSpikesClassifier` (Continuous–Fragmented),
  built by `figure04_decoder.build_decoder_models` from `Figure4Config`: the
  values passed to the package are in `Figure4DecoderConfig`, and the package
  defaults the decode relies on are recorded in `Figure4PackageDefaults`.
  `figure04_models` holds each model's ID and display labels.

### Where the manuscript's numbers come from

`manuscript/main.tex` quotes no analysis number literally. It inputs
`manuscript/reported_values.tex`, which `scripts/emit_reported_values.py`
(recipe `reported_values`) renders from the two committed summaries,
`figure03_summary.json` and `figure04_summary.json` in
`manuscript/figures/main/`. Each macro's comment in that file names the summary
field it came from. See [From summary to prose](#from-summary-to-prose-the-reported-value-macros).

### Shared vocabulary

| Paper term | Code name | Notes |
| --- | --- | --- |
| Predictive distribution $P_k$, $p(x_k\mid y_{1:k-1})$ (the Introduction's “prediction distribution”) | `predictive` | Figure 3: `DecodingDiagnostics.predictive`, shape `(n_time, n_bins)`; at $t=0$ it is the initial distribution. Figure 4: `non_local_detector`'s output variable `predictive_posterior` (`figure04_place_fields.DECODER_PREDICTIVE_VAR`), marginalized over the Continuous–Fragmented dynamics mode by `marginal_position_distribution(results, "predictive")` |
| Filtered posterior $p(x_k\mid y_{1:k})$ | `posterior` | Figure 3 (`DecodingDiagnostics.posterior`); its mean gives the decoding error. Figure 4 does not request the filter output |
| Smoothed (acausal) posterior $p(x_k\mid y_{1:K})$ | `acausal_posterior` | Figure 4 decoder output (`figure04_place_fields.DECODER_SMOOTHED_VAR`; `marginal_position_distribution(results, "smoothed")`); used by the desktop viewer's overlay only |
| Normalized single-event likelihood $Q_{k,j}$ | `event_likelihood` | one row per spike; the figures' likelihood rows show its mean over the spikes in each bin |
| Normalized combined likelihood of all cells in a bin | `combined_likelihood` | Figure 3 (`DecodingDiagnostics.combined_likelihood`); the viewer's “population likelihood” |
| Per-spike HPD overlap, predictive $p$-value, KL divergence | `event_hpd_overlap`, `event_predictive_pvalue`, `event_kl_divergence` | with `event_time_ind` (time bin) and `event_cell_ind` (unit); the un-prefixed `hpd_overlap`, … are dense `(n_time, n_cells)` matrices |
| Flag | `flag_mask`, `flag_rules` | inclusive comparison against a threshold |
| Firing rate $\lambda_c$ vs expected count $m_c=\lambda_c\Delta t$ | `expected_counts…` arrays hold expected counts per step; the `…rate_scale` configuration fields scale them | see [Rates and expected counts](#rates-and-expected-counts) |
| Expected-count table | `expected_counts_per_step`, `baseline_expected_counts_per_step`, `figure03_simulation.Figure3ExpectedCountTables`, `simulation.place_field_expected_counts` | shape `(n_bins, n_cells)`; each entry is an expected count per step $m_c(x)$, not a rate in Hz |
| Time bin $k$ ($K$ bins) | `t`, `event_time_ind`, `n_time` | the paper writes $k$; the code writes `t`, counting from 0 |
| The Figure-3 remap, replay, and sparse-population windows | decoder override windows: `decoding.DecoderOverrideWindow`, `DecoderOverrideSchedule` | a half-open step interval `[start, end)` in which the decoder uses another expected-count table; the remap window is the observation misfit, the replay and sparse-population windows are controls |
| Position and decoding error, Figure 3 | “a.u.”; summary `error_units: position_units` | the simulated track is in arbitrary position units (axis label “Position (a.u.)”); Figure 4 positions are in cm |
| Rescued percentage (Figure 4) | `FlagConfusion.rescued_fraction`; summary `flag_confusions[].rescued_fraction` | fraction (0–1) of the spikes the Continuous model flags that the Continuous–Fragmented model does not; the prose prints it as a whole percent (`\RecHpdRescuedPercent`, `\RecPvalueRescuedPercent`) |
| Rank-based predictive $p$-value | `statespacecheck.mark_predictive_pvalue` (called by `event_diagnostics`) | `statespacecheck` also exports `predictive_pvalue`, a Monte Carlo check for a user-supplied replicate generator, which the paper does not use |
| Summary source digest | `provenance.source.source_tree_sha256` | SHA-256 of every `.py` file under `src/statespacecheck_paper`, comments and docstrings included: any source edit changes it, so both summaries' `provenance.source` are refreshed together |
| Cache fingerprints | `fingerprint_sha256` (decode), `diagnostics_fingerprint_sha256` | SHA-256 of the docstring-stripped syntax trees of listed modules plus the relevant configuration, input-file checksum, and package versions; comment and docstring edits leave them unchanged (see [Figure-4 cache behavior](#figure-4-cache-behavior)) |

## Architecture at a glance

The code separates **general, figure-agnostic layers** from **per-figure
families**:

- **General layers**: `simulation` (random walks, place-field expected counts, spike
  simulators), `decoding` (the Bayesian filter `decode_with_diagnostics` and its
  decoder override windows), and `diagnostics` (containers, the HPD coverage,
  and the paper's threshold rule around the per-spike HPD-overlap /
  predictive-p-value / KL-divergence computation, which lives in the external
  `statespacecheck` package). Figure 1 uses `diagnostics` (the HPD coverage);
  Figure 2 uses `diagnostics` and `simulation`; Figure 3 uses all three; and
  Figure 4 uses `diagnostics`. `diagnostics` imports no sibling module, and
  `style` imports it for each metric's flag direction. The package `__init__`
  imports no sibling module, so importing one module loads only its own dependencies.
- **Per-figure families**: `figure01_generation`;
  `figure02_{panels,generation}`;
  `figure03_{protocol,simulation,summary,plotting,generation}`; and
  `figure04_{input,models,protocol,decoder,place_fields,diagnostics,plot_primitives,track_plots,panels,fit,cache,workflow,summary,layout,generation}`.
  `figure04_models` holds each decoder's one machine ID (`continuous`,
  `continuous_fragmented`) and its display labels; the figure, summary
  printout, viewer, and website data take the labels from it, and the
  decode-hashed modules do not import it. `figure04_protocol` holds the fixed
  flag cutoffs and the manuscript's detail window, which the figure, viewer,
  website export, and lab pipeline share; it imports only `diagnostics`, so
  none of them loads the layout or plotting modules for these values. Each figure is a small set of
  single-responsibility modules rather than one monolith, so an outside reader
  can follow the scientific workflow (configure → simulate/load → decode →
  diagnose → summarize → render).

  Some Figure-4 boundaries also protect the expensive decode cache. Refitting
  and decoding both models takes minutes and writes an ~8 GB cache, so the fit
  and decode live in `figure04_fit`, and the decode fingerprint hashes only it
  and the modules it imports (`figure04_decoder`, `figure04_input`,
  `figure04_place_fields`). The diagnostics fingerprint hashes the modules that
  compute the per-spike diagnostics (`diagnostics`, `figure04_diagnostics`,
  `figure04_place_fields`, and `figure04_workflow`, which assembles them).
  Result containers, orchestration, and progress messages (`figure04_workflow`),
  model labels (`figure04_models`), cache I/O (`figure04_cache`), whole-session
  statistics (`figure04_summary`), and all plotting live outside the decode
  fingerprint, so editing them refits nothing.
- **Shared support**: `plotting`, `style`, and `schematic` (figure drawing);
  `paths` (repository and data locations, published identifiers);
  `scientific_artifacts` (summary provenance and flag rules); `number_format`
  and `reported_values` (summary to prose); `figure04_download` (the
  Figure-4 input file's download); `site_export` (website data).

### Module dependency graph

Every module under `src/statespacecheck_paper/`, with the sibling modules it
imports. Names before any `;` are imported at module level. `lazy:` names are
imported inside a function, so they load only when that function runs;
`type-only:` names are imported under `if TYPE_CHECKING` for annotations.
Imports that run only when a module is executed as a script are left out.
`tests/test_import_boundaries.py` derives this graph from the source, checks
that it has no cycle (counting lazy imports), and checks that this listing
matches it exactly.

<!-- module-graph -->
```text
__init__                            → (none)
paths                               → (none)
number_format                       → (none)
diagnostics                         → (none)
simulation                          → (none)
decoding                            → diagnostics, simulation
style                               → diagnostics
plotting                            → style
schematic                           → style
scientific_artifacts                → diagnostics, paths
reported_values                     → number_format, paths

figure01_generation                 → diagnostics, paths, plotting, schematic, style
figure02_panels                     → diagnostics, simulation, style
figure02_generation                 → figure02_panels, paths, style

figure03_protocol                   → (none)
figure03_simulation                 → decoding, diagnostics, figure03_protocol, simulation
figure03_summary                    → diagnostics, figure03_protocol, figure03_simulation
figure03_plotting                   → diagnostics, figure03_protocol, figure03_summary, number_format, plotting, style
figure03_generation                 → diagnostics, figure03_plotting, figure03_protocol, figure03_simulation, figure03_summary, paths, scientific_artifacts, style

figure04_input                      → paths
figure04_download                   → figure04_input, paths
figure04_models                     → (none)
figure04_protocol                   → diagnostics
figure04_decoder                    → diagnostics
figure04_place_fields               → (none)
figure04_diagnostics                → diagnostics, figure04_place_fields
figure04_plot_primitives            → figure04_place_fields, style
figure04_track_plots                → figure04_input, figure04_plot_primitives
figure04_panels                     → diagnostics, figure04_diagnostics, figure04_models, figure04_place_fields, figure04_plot_primitives, figure04_track_plots, plotting, style
figure04_fit                        → figure04_decoder, figure04_input, figure04_place_fields
figure04_cache                      → figure04_decoder, figure04_input
figure04_workflow                   → diagnostics, figure04_cache, figure04_decoder, figure04_diagnostics, figure04_fit, figure04_input
figure04_summary                    → diagnostics, figure04_diagnostics, figure04_models, figure04_workflow
figure04_layout                     → diagnostics, figure04_models, figure04_panels, figure04_protocol, figure04_track_plots, figure04_workflow, plotting, style
figure04_generation                 → figure04_cache, figure04_decoder, figure04_layout, figure04_models, figure04_protocol, figure04_summary, figure04_workflow, paths, scientific_artifacts, style

site_export                         → decoding, diagnostics, figure03_protocol, figure03_simulation, figure03_summary, figure04_cache, figure04_decoder, figure04_diagnostics, figure04_models, figure04_place_fields, figure04_protocol, figure04_workflow, number_format, paths, reported_values, simulation, style

interactive                         → (none)
interactive.__main__                → interactive.app
interactive.app                     → figure04_models, interactive.data_source; lazy: interactive.viewer
interactive.cache                   → diagnostics, figure04_models, figure04_place_fields, paths; lazy: figure03_protocol, figure03_simulation, figure04_cache, figure04_decoder, figure04_workflow
interactive.data_source             → figure04_models, figure04_place_fields, interactive.cache
interactive.panels                  → figure03_plotting, plotting, style
interactive.viewer                  → figure04_models, figure04_protocol, interactive.data_source, interactive.panels, style

spyglass_pipeline                   → (none)
spyglass_pipeline.figure04_input    → figure04_input
spyglass_pipeline.paper_export      → spyglass_pipeline.figure04_input
spyglass_pipeline.figure04_schema   → figure04_decoder, figure04_input, figure04_models, figure04_place_fields, figure04_protocol, spyglass_pipeline.figure04_compute, spyglass_pipeline.figure04_input
spyglass_pipeline.figure04_compute  → figure04_models, spyglass_pipeline.figure04_input; lazy: diagnostics, figure04_decoder, figure04_diagnostics, figure04_generation, figure04_place_fields, figure04_protocol, figure04_summary
spyglass_pipeline.pickle_conversion → figure04_input, spyglass_pipeline.figure04_input
```
<!-- module-graph -->

### Not part of figure generation

- **`interactive/`** — an optional pyqtgraph viewer that consumes the same
  diagnostic results but produces no manuscript figure.
- **`spyglass_pipeline/`** — the lab's Spyglass pipeline for Figure 4, upstream of the
  paper code; no figure or analysis module imports it (`tests/test_import_boundaries.py`).
  - **`figure04_input.py`** (with `scripts/fetch_figure04_inputs.py`) rebuilds the
    Figure-4 input file from Spyglass, writes it (`recording_arrays` defines the
    layout, `write_npz` writes it deterministically), and compares it with the
    archived copy. See [data-lineage.md](data-lineage.md).
  - **`paper_export.py`** (with `scripts/spyglass_export_figure04.py`) records that
    fetch in a Spyglass paper export.
  - **`figure04_schema.py`** (with `scripts/spyglass_pipeline_figure04.py`) defines
    the Figure-4 decode and diagnostics as Spyglass tables, meant to reproduce this
    figure's numbers inside the lab database; **`figure04_compute.py`** holds what
    those tables compute, runnable without a database. See
    [spyglass-pipeline.md](spyglass-pipeline.md).
  - **`pickle_conversion.py`** (with `scripts/convert_figure04_pickles.py`)
    converts the five pickles the recording was first exported as into the
    input file.

---

## Figure 1 — Schematic and distribution comparisons

- **Reproduction:** `uv run python scripts/generate_figure01.py` (simulated;
  no external data).
- **Manuscript:** the state-space-model schematic and the
  predictive-vs-likelihood distribution comparisons in the Introduction/Methods.
- **Entry point:** `scripts/generate_figure01.py::main`, which calls
  `figure01_generation.generate_figure01`; the separately testable
  `compose_figure01` returns the in-memory figure.
- **Configuration:** named constants / function arguments inside the generation
  recipe and `schematic.py`; the HPD coverage is `diagnostics.HPD_COVERAGE`. No
  config dataclass.
- **Computation:** `schematic.py` (graphical model + equation boxes) and
  `figure01_generation.create_distribution_comparison_panel` (HPD regions from
  `statespacecheck.highest_density_region`).
- **Output:** `manuscript/figures/main/figure01.{pdf,png}`.
- **Tests:** `tests/test_schematic.py`; `tests/test_figures.py` (entry-point
  contract); `tests/test_plotting.py::TestCreateDistributionComparisonPanel`.

Trace: `generate_figure01` → `compose_figure01` → semantic axes
(`graphical_model`, `filtering_equations`, and four named consistency cases) →
the `schematic` renderers and `create_distribution_comparison_panel` → `save_figure`.

## Figure 2 — Diagnostic demonstrations

- **Reproduction:** `uv run python scripts/generate_figure02.py` (simulated).
- **Manuscript:** the worked demonstrations of the three diagnostics.
- **Entry point:** `scripts/generate_figure02.py::main`, which calls
  `figure02_generation.generate_figure02`; `compose_figure02` accepts an
  injectable random generator and returns the in-memory figure.
- **Configuration:** named constants / arguments in the generation recipe; per-panel
  renderers live in `figure02_panels.py`.
- **Computation:** `figure02_panels.py` draws the panels; the diagnostics, HPD regions and the Monte Carlo predictive p-value come from
  `statespacecheck` (`kl_divergence`, `hpd_overlap`, `highest_density_region`,
  `monte_carlo_mark_pvalue`).
- **Output:** `manuscript/figures/main/figure02.{pdf,png}`.
- **Tests:** `tests/test_figures.py` (the Figure 2 panel and data tests).

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
  The generation recipe uses the default `Figure3Config()` and
  `figure03_summary.N_REALIZATIONS = 100`; the published figure and summary
  depend on both.
- **Computation (reading order):**
  `figure03_protocol` (config + phase ladder) →
  `figure03_simulation.run_figure03_simulation` (drives the 8-phase trajectory,
  calls `decoding.decode_with_diagnostics`, which calls `diagnostics`) →
  `figure03_summary.estimate_realization_summary` (pools 100 realizations into
  thresholds, median flag percentages, and median decoding error) →
  `figure03_plotting.compose_figure03` (the time-series panel + the panel-(b)
  flag heatmap and decoding-error row).
- **Output:** `manuscript/figures/main/figure03.{pdf,png}` plus
  `figure03_summary.json`, containing the full configuration, seed range,
  explicit inclusive flag rules, the threshold rule with the share of the pooled
  baseline each threshold flags, metric/condition order, reported percentage
  matrix, per-condition decoding error (median absolute error), and
  source/dependency-lock provenance.
- **Thresholds:** pooled over the opening baseline of all realizations; the
  HPD-overlap threshold is exactly 0 (`flag_rules.hpd_overlap.threshold`), so
  Figure 3 flags a spike's HPD overlap only when the predictive and likelihood
  HPD regions are disjoint.
- **Tests:** `tests/test_figure03_phases.py` (the higher-level scientific
  contract, including the control-integrity checks that the replay and
  sparse-population controls carry no hidden observation misfit, because the
  decoder is given the expected-count tables that generated their spikes. Both still have
  a motion-model mismatch: the decoder keeps its random-walk prior for the
  replay's deterministic out-and-back sweep and for the sparse phase's
  stationary animal);
  `tests/test_figure03_{protocol,simulation,summary,plotting,generation,contracts}.py`.

### Figure-3 conditions (executable source of truth: `build_summary_conditions`)

In heatmap order, each condition labeled by which part of the model it perturbs
(the summary's `condition_order` ID is in parentheses; the bold name is the
condition's `title`, the website's tab):

1. **Well-specified** (`well_specified`) — pooled clean-recovery windows
   (out-of-sample false-positive reference); *control*.
2. **Remap** (`remap`) — scrambled place-field identities; *observation-model*
   misfit.
3. **History-dependent firing** (`history_dependent`) — refractory + bursting
   spikes; *observation model* (temporal), largely missed by the per-spike
   spatial diagnostics.
4. **Replay** (`replay`) — an out-and-back represented sweep while the animal
   is immobile; *control* (benign decoded-vs-true divergence). The decoder has
   the correct place fields but keeps its random-walk prior, which does not describe
   the deterministic sweep.
5. **Drift** (`drift`) — AR(1) persistent-velocity trajectory;
   *transition-model* misfit.
6. **Sparse population** (`sparse_population`) — a quiet ordinary ensemble with
   a few narrow cells firing sparsely; *control* (KL responds; HPD/rank-p stay
   near baseline). The animal is stationary, and the decoder keeps its
   random-walk prior.

The simulation's eight phases are the opening baseline, remap, clean recovery 1,
history-dependent firing, clean recovery 2 (containing the replay sweep), drift,
clean recovery 3, and the sparse population. `figure03_protocol.PhaseBoundary`
names their end steps, and the numeric boundaries live in
`Figure3Config.phase_boundaries` — see that dataclass for values rather than
duplicating them here.

### Figure-3 traceability walkthrough (following the typed returns)

`Figure3Config` → `run_figure03_simulation(config)` returns a
`Figure3SimulationResult` (`.physical_position`, `.spike_counts`, `.diagnostics: DecodingDiagnostics`,
`.position_bins`, `.sparse_place_field_centers`) → `estimate_realization_summary(config, n_realizations=100)`
returns a `Figure3RealizationSummary` (`.diagnostic_thresholds: DiagnosticThresholds`,
`.median_flag_percentages`, `.median_decoding_error`,
`.flag_percentage_standard_errors`, `.decoding_error_standard_errors`) →
`compose_figure03(physical_position=…, spike_counts=…,
diagnostics=…, diagnostic_thresholds=…, config=…, place_field_centers=…,
median_flag_percentages=…, median_decoding_error=…)` returns a `matplotlib`
`Figure` → `save_figure` writes
`figure03.{pdf,png}` while `write_json_artifact` writes the same summary values
and their configuration to `figure03_summary.json`.

### Manuscript ↔ code vocabulary (Figure 3)

| Code name | Manuscript notation | Meaning / shape |
| --- | --- | --- |
| `physical_position` | $x_t$ outside replay; $z_t$ during replay | physical position, shape `(n_time,)`; decoding error is measured against this trajectory |
| `position_bins` | discretized $x$ grid | position bin centers, shape `(n_bins,)` |
| `spike_counts` | $y_{c,t}$ | integer spike counts, shape `(n_time, n_cells)` |
| `place_field_centers` | $\mu_c$ | per-cell place-field centers, shape `(n_cells,)` |
| `place_field_std` | $\sigma_\mathrm{pf}$ | Gaussian place-field standard deviation |
| `place_field_rate_scale` | $\alpha$ | expected-count scale multiplying the normalized Gaussian field |
| `prediction_step_std` | $\sigma_\mathrm{pred}$ | decoder baseline dynamics standard deviation |
| `predictive` | $P_k$, the predictive distribution | shape `(n_time, n_bins)` |
| `posterior` | filtered posterior | shape `(n_time, n_bins)`; the decoding-error row uses its mean |
| `event_likelihood` | $Q_{k,j}$ | shape `(n_spikes, n_bins)`; the likelihood row shows its mean per bin |
| `event_hpd_overlap` | HPD overlap | per-spike prediction/likelihood HPD overlap |
| `event_predictive_pvalue` | rank-based predictive $p$-value | per-spike rank statistic |
| `event_kl_divergence` | KL divergence | per-spike prediction→likelihood KL |
| `DecoderOverrideWindow` | the remap, replay, and sparse-population windows | decoder override windows that swap the decoder's expected-count table |
| `well_specified`, `remap`, `history_dependent`, `replay`, `drift`, `sparse_population` | the heatmap's condition columns | `condition_order` in the summary |

### Rates and expected counts

The Methods distinguish the firing rate $\lambda_c(x)$ from the expected count
$m_c(x)=\lambda_c(x)\Delta t$ in a bin of width $\Delta t$. The simulation's
`place_field_expected_counts` and decoder expected-count tables
(`expected_counts_per_step`) contain these expected counts per step: they go
directly into the Poisson distribution. In Figure 3,
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

- **Manuscript:** the real hippocampal-recording panels comparing the Continuous
  and Continuous–Fragmented decoders: (a, b) a two-second window with a
  candidate replay event under each decoder, and (c) whole-session hexbin
  comparisons of each diagnostic between the two decoders, with the counts of
  *rescued* and *newly flagged* spikes quoted in the Results.
- **Reproduction:** `uv run python scripts/generate_figure04.py` after
  downloading the input file (`make download-data`). The fit and decode are
  cached; see [Figure-4 cache behavior](#figure-4-cache-behavior).
- **Entry point:** `scripts/generate_figure04.py::main` (the CLI), which calls
  `figure04_generation.generate_figure04(*, use_cache)`.
- **Decoders and configuration:** `Figure4Config` (in `figure04_decoder.py`).
  Both decoders share one sorted-spikes kernel-density observation model. The
  Continuous model has a single random-walk dynamics mode; the
  Continuous–Fragmented model adds a fragmented mode, keeps the random walk for
  Continuous-to-Continuous transitions, and makes the other three mode
  transitions uniform over track-interior position bins. `Figure4Config` has
  four parts:
  - `decoder`, a `Figure4DecoderConfig`: the values passed to
    `non_local_detector` — `position_std` (KDE bandwidth,
    $\sqrt{12.5}\approx3.54$ cm), `position_bin_size_cm` (2 cm), and
    `sampling_frequency_hz` (500 Hz, i.e. 2 ms bins);
  - `package_defaults`, a `Figure4PackageDefaults`: the `non_local_detector`
    defaults the decode relies on without passing them — the observation
    algorithm, `movement_var` (6 cm²) and `movement_mean`, each model's
    position initial-condition and transition classes, the Continuous–Fragmented
    mode-transition class, diagonal, and mode initial conditions, and the package
    version, which must equal the installed one. They are recorded and checked against
    the built models at decode time (`validate_package_defaults`) and in the
    tests, but not injected: injecting them would mean rebuilding the nested
    transition grid. The discrete-transition concentration and regularization
    are not recorded, since `non_local_detector` reads them only in
    `estimate_parameters`, which Figure 4 never calls;
  - `diagnostics`, a `Figure4DiagnosticsConfig`: the HPD coverage and the
    event-selection rule (every spike of every unit inside the decoded time
    grid);
  - `execution`, a `Figure4ExecutionConfig`: `block_size`, a memory knob that
    does not change the decode (the KDE density is identical for any
    `block_size`).
- **Diagnostics:** the Continuous–Fragmented predictive distribution is
  marginalized over the dynamics mode, so both decoders are diagnosed on the
  same position grid against the same place fields
  (`figure04_diagnostics.compute_results_diagnostics`). Flags use the fixed
  cutoffs `FIGURE04_DIAGNOSTIC_THRESHOLDS = {"hpd_overlap": 0.05,
  "predictive_pvalue": 0.05}` in `figure04_protocol.py`; KL divergence is not
  thresholded. `figure04_diagnostics.FlagConfusion` counts, for each metric,
  the spikes flagged by both decoders, by the Continuous model only (`rescued`),
  by the Continuous–Fragmented model only (`newly_flagged`), and by neither.
- **Detail window:** `FIGURE04_DETAIL_WINDOW = Figure4DetailWindow(center_index=193069,
  half_width_samples=500)` in `figure04_protocol.py` centers panels (a, b) on
  a KL-divergence spike during immobility at a reward well and spans about two
  seconds total.
- **Computation (reading order):**
  `figure04_generation` (recipe) → `figure04_workflow.prepare_figure04_render_data`
  (loads the recording; loads the cached decode or fits and decodes both models
  with `figure04_fit.fit_and_decode`; loads the cached diagnostics or
  computes them with `figure04_diagnostics`) →
  `figure04_summary.compute_figure04_summary` (whole-session means and flag
  confusions) → `figure04_layout.compose_figure04` (artist arrangement) →
  `save_figure`.
- **Intermediate data — the honest boundary.** Figure 4 is reproduced **from
  the Figure-4 input file (derived data) onward**, not from raw acquisition. The loader
  `figure04_input.load_figure04_input` reads one file from the
  data directory, `{epoch}_figure04_inputs.npz`, and returns a validated
  `NeuralRecordingData`:

  | `NeuralRecordingData` field | Arrays in the `.npz` | Contents |
  | --- | --- | --- |
  | `position_info` | `position_time`, `position_columns`, `position/<column>` | time-indexed position (seconds; positions in cm) |
  | `spike_times` | `spike_times`, `spike_offsets` | per-cell spike times (seconds), concatenated with per-cell offsets |
  | `track_graph` | `track_nodes`, `track_node_positions`, `track_edges`, `track_edge_distance`, `track_edge_id` | track graph |
  | `linear_edge_order` | `linear_edge_order` | linearization edge order |
  | `linear_edge_spacing` | `linear_edge_spacing` | edge spacing (cm) |

  The file holds only numeric and string arrays, is read with
  `allow_pickle=False`, and is written deterministically (same content, same
  SHA-256); `spyglass_pipeline.figure04_input.recording_arrays` defines the layout.
  `scripts/convert_figure04_pickles.py` converts the five pickles the recording
  was first exported as and checks the result.

  The input comes from the Frank-lab Spyglass database (the recording of
  [Comrie et al. 2026](https://doi.org/10.1016/j.neuron.2026.08.023)).
  [data-lineage.md](data-lineage.md) records the exact Spyglass entries and
  processing steps, the verification against the files the figure used, and which
  of them are public (DANDI dandiset
  [001942](https://dandiarchive.org/dandiset/001942) has the raw recording).
  `spyglass_pipeline/figure04_input.py` rebuilds the input file from the database
  (`scripts/fetch_figure04_inputs.py`, read-only) and `spyglass_pipeline/paper_export.py`
  logs the fetches in a Spyglass paper export (`scripts/spyglass_export_figure04.py`); both run on a lab server with
  Spyglass and database access. The input file is archived on Zenodo
  ([10.5281/zenodo.23020757](https://doi.org/10.5281/zenodo.23020757));
  `scripts/download_figure04_inputs.py` downloads it and checks its SHA-256.
  It downloads into `data/` by default; to use another directory, set
  `STATESPACECHECK_DATA_PATH` (or pass `--data-path`).
- **Output:** `manuscript/figures/main/figure04.{pdf,png}` plus
  `figure04_summary.json`, containing configuration and dataset identifiers,
  explicit inclusive flag rules, whole-session means, flag-confusion counts,
  source/dependency-lock provenance, the decode- and diagnostics-cache
  fingerprints, and the SHA-256 checksum of the input file.
- **Tests:** `tests/test_figure04_decoder.py::TestFigure4ConfigMatchesManuscript`
  (config matches the manuscript decoder parameters);
  `tests/test_figure04_{cache,workflow,layout,generation}.py` (orchestration);
  `tests/test_figure04_{diagnostics,place_fields,models}.py` (the analysis
  leaves and model registry);
  `tests/test_figure04_{plot_primitives,track_plots,panels}.py` (the plotting
  leaves); `tests/test_figure04_input.py` (the `NeuralRecordingData` contract);
  `tests/test_figure04_download.py` (the input download).

### Figure-4 traceability walkthrough (following the typed returns)

`Figure4Config` + `Figure4Paths` → `prepare_figure04_render_data(config, paths, use_cache=…)`
loads a `NeuralRecordingData` and returns a `Figure4RenderData`
(`.recording`, `.time`, `.linear_position`,
`.analysis_results: Figure4AnalysisResults`, `.cache_provenance`) — the analysis
results (decodes, per-spike diagnostics, spike counts, place fields) are built by `Figure4AnalysisResults.from_cache_payload` from a decode
payload and a diagnostics payload, each loaded from its fingerprint-matching
cache or computed fresh (`figure04_fit.fit_and_decode` for the decode,
`_compute_diagnostics_payload` for the diagnostics) → `compute_figure04_summary`
returns a `Figure4Summary` (`.n_units`, per-decoder `Figure4DiagnosticMeans`, and
typed `FlagConfusion` counts) → `format_figure04_summary` handles CLI text separately
→ `compose_figure04(render_data, diagnostic_thresholds=…,
detail_window=Figure4DetailWindow(…))`
returns a `Figure4Composition` (`.figure`, `.bbox_inches`) → `save_figure` writes
`figure04.{pdf,png}` with that custom crop, while `write_json_artifact` writes the
typed summary to `figure04_summary.json`.

The optional interactive viewer derives its Zarr/Parquet/NPZ layout from this
same `Figure4RenderData` via `interactive.cache.build_figure04_viewer_cache`.

### Manuscript ↔ code vocabulary (Figure 4)

| Paper term | Code name | Notes |
| --- | --- | --- |
| Continuous, Continuous–Fragmented | model IDs `continuous`, `continuous_fragmented` | `figure04_models.CONTINUOUS` / `CONTINUOUS_FRAGMENTED` carry the IDs, full labels, and short labels (“Cont.”, “Cont.–Frag.”) |
| `position_std`, `movement_var` | same names | kept in the spelling of the Methods and `non_local_detector` |
| Predictive distribution, marginalized over dynamics mode | `marginal_position_distribution(results, "predictive")` | reads `predictive_posterior` |
| Rescued, newly flagged | `FlagConfusion.rescued`, `FlagConfusion.newly_flagged` | reference decoder `continuous`, comparison `continuous_fragmented` (`flag_confusion_models` in the summary) |
| Per-spike diagnostics | `event_hpd_overlap`, `event_predictive_pvalue`, `event_kl_divergence` | as in Figure 3 |

### Figure-4 cache behavior

Fitting and decoding both models takes several minutes, so
`prepare_figure04_render_data` caches the results as two joblib files in
`<data path>/intermediates/` (`data/` unless `STATESPACECHECK_DATA_PATH` is set;
`Figure4Paths` defines the locations):

- the **decode cache** `{epoch}_figure04_decode.joblib` (about 8 GB; memory-mapped
  on load) holds both models' decoder outputs, the spike counts, and the place
  fields;
- the **diagnostics cache** `{epoch}_figure04_diagnostics.joblib` holds both models'
  per-spike diagnostics.

Each is used only when its fingerprint matches; otherwise it is rebuilt and
overwritten. `--force-recompute` refits, recomputes, and overwrites both
regardless.

- The **decode fingerprint** (`figure04_cache.compute_figure04_cache_provenance`)
  hashes the decode schema version, the `decoder` and `package_defaults` parts
  of `Figure4Config`, the data identifier, the installed `non_local_detector`
  version, the **content hash of the input file** (so replacing it under the
  same `animal_date_epoch` invalidates the cache), and the docstring-stripped
  syntax trees of the fit-and-decode modules (`figure04_cache._DECODE_SOURCE_FILES`):
  `figure04_fit.py` (`fit_and_decode` and the decode time grid), with the
  modules it imports, `figure04_decoder.py` (model construction, defaults
  check, fit, spike counts), `figure04_input.py` (reading and validating the
  recording), and `figure04_place_fields.py` (the cached place fields). This
  covers helper functions, imports, defaults, and recording preparation.
  The fingerprint also hashes the computational environment: the Python
  version (`major.minor.micro`), the machine architecture (`platform.machine()`),
  and the installed version of every distribution in `non_local_detector`'s
  runtime dependency closure (`figure04_cache.installed_dependency_versions`:
  the requirements in the installed package metadata, followed transitively,
  with environment markers evaluated for the running interpreter and
  extras-only requirements excluded; numpy, scipy, jax, and so on). Upgrading
  any of these, changing Python, or moving a cache between architectures
  (for example arm64 and x86_64) therefore refits.
  `tests/test_import_boundaries.py` checks that every paper module these
  import, directly or through others, is hashed too or is in a commented
  allowlist of decode-neutral imports (`diagnostics.HPD_COVERAGE`, the
  default diagnostics coverage; `paths.FIGURE04_INPUTS_EPOCH`, which only
  chooses the missing-file message).
- The **diagnostics cache** is keyed by the decode fingerprint plus a
  **diagnostics fingerprint**
  (`figure04_cache.compute_figure04_diagnostics_fingerprint`): the diagnostics
  schema version, the `Figure4DiagnosticsConfig`, the installed
  `statespacecheck` version, the same environment components for
  `statespacecheck`'s runtime dependency closure (Python version, machine
  architecture, installed versions), and the docstring-stripped syntax trees of
  `diagnostics.py`, `figure04_diagnostics.py`, `figure04_place_fields.py`, and
  `figure04_workflow.py` (whose `_compute_diagnostics_payload` pairs each
  model's predictions with the spikes and coverage).

Any change that refits therefore also recomputes the diagnostics. A change
confined to `diagnostics.py`, `figure04_diagnostics.py`, `figure04_workflow.py`
(including its result containers and progress messages), or the diagnostics
configuration recomputes only the diagnostics from the cached predictions
(about a minute). The `execution` settings are in neither fingerprint.
Docstring and comment edits invalidate neither cache, and the summary scalars
and their printed text (`figure04_summary.py`) are outside both. Caches are
written to a temporary sibling and renamed into place, so a memory-mapped
cache is never overwritten in place. The schema versions
(`FIGURE04_DECODE_SCHEMA_VERSION`, `FIGURE04_DIAGNOSTICS_SCHEMA_VERSION`) are
manual overrides: bumping one invalidates every cache of that kind.

## Machine-readable summary schema

`figure03_summary.json` uses schema version 9
(`figure03_generation.FIGURE03_SUMMARY_SCHEMA_VERSION`) and `figure04_summary.json`
uses schema version 8 (`figure04_generation.FIGURE04_SUMMARY_SCHEMA_VERSION`). The
Figure-3 configuration block records every
`Figure3Config` field together with the step length in seconds
(`configuration.step_seconds`, the protocol constant `STEP_SECONDS`) and the
HPD coverage (`configuration.hpd_coverage`, `diagnostics.HPD_COVERAGE`). The
Figure-3 schema includes the decoding-error block:
`error_metric_order` (`median_absolute_error`), `error_units`, and
`median_decoding_error`, a `(1, n_conditions)` matrix of the
across-realization median absolute error of the filtered-posterior mean
(position units), in the same column order as `median_flag_percentages`. It also
records `median_flag_percentage_standard_errors` and
`median_decoding_error_standard_errors`: approximate standard errors of the
medians, estimated from order-statistic interval widths. These describe the
uncertainty in the aggregated medians under repeated simulation with the same
configuration, not the spread of individual realizations, and do not set
reported precision. The spread itself is in `realization_flag_percentages`
`(n_realizations, 3, n_conditions)` and `realization_decoding_error`
`(n_realizations, 1, n_conditions)`: every realization's values, in seed order
from `realizations.first_seed`; the medians are their medians over the first
axis. The summary also records the baseline-threshold provenance
(`threshold_provenance`) quoted in the Methods: the baseline's end step, each
metric's rule (quantile or fixed cutoff), and each metric's
`baseline_flagged_fraction`, the fraction (0–1) of pooled baseline spike events
flagged under the recorded threshold.

The Figure-4 schema records `dataset.n_units` alongside the recording
identifier, names the second decoder `continuous_fragmented` throughout, and
keeps the recorded `non_local_detector` defaults under
`configuration.package_defaults`: the observation-model algorithm
(`sorted_spikes_kde`), the random-walk `movement_var` and `movement_mean`,
each model's position initial conditions and position transitions per mode
as `non_local_detector` class names (`UniformInitialConditions`; `RandomWalk`
for Continuous to Continuous and `Uniform` otherwise), its mode initial
conditions, the Continuous–Fragmented mode-transition class and diagonal, and
the package version the manuscript states (`\RecNldVersion`). The decode
refuses to run unless that version is the installed one, which
`provenance.figure04_caches.non_local_detector_version` records, and a test
checks that the committed summary's two values agree. Each `flag_confusions` entry counts
spikes flagged by `both` decoders, by the reference only (`rescued`), by the
comparison only (`newly_flagged`), and by `neither`; `flag_confusion_models`
names the reference (`continuous`) and comparison (`continuous_fragmented`)
decoders.

In both summaries, the `flag_rules` object binds each numeric threshold to its
executable semantics: `less_than_or_equal` means a value is flagged when
`value <= threshold`, and `greater_than_or_equal` means it is flagged when
`value >= threshold`. Keeping the operator and threshold in one record prevents
consumers from guessing whether a boundary is strict or inclusive.

Both summaries contain `provenance.source`, with the installed
`statespacecheck-paper` and `statespacecheck` versions (the diagnostics are
computed in `statespacecheck`, outside the hashed source tree), a deterministic
SHA-256 digest of every Python file under `src/statespacecheck_paper`, and the
SHA-256 digest of `uv.lock`. The emitter refuses summaries whose
`provenance.source` blocks differ in any of these.
The digest excludes timestamps, generated outputs, and absolute paths, so clean
checkouts of identical source produce the same identity.

The source digest includes comments and docstrings. Follow the
[artifact refresh procedure](development.md#refreshing-publication-artifacts)
after changing source code, inputs, or dependencies. Scientific changes require
regenerating the affected results; documentation-only source edits can refresh
the provenance after verifying that executable code is unchanged.

Figure 4 also contains `provenance.figure04_caches`. Its
`fingerprint_sha256` is the same identity used to accept or reject the
expensive decoder cache, `diagnostics_fingerprint_sha256` the identity of the
diagnostics cache (recorded with the diagnostics configuration and the
installed `statespacecheck` version), and the record includes the installed
`non_local_detector` version plus the content
SHA-256 of the named input file. It also records the environment both
fingerprints hash: `python_version`, `machine`, and the installed versions of
the two dependency closures (`decode_dependency_versions` for
`non_local_detector`, `diagnostics_dependency_versions` for `statespacecheck`),
so a reader can see what the decode and diagnostics ran with. Canonical artifact generation fails
if the input checksum is missing. Thus a summary can be traced to the exact
derived inputs even when those large files are distributed separately.

## From summary to prose: the reported-value macros

The manuscript's reported configuration and headline analysis statistics are
generated from the summaries. `main.tex` inputs
`manuscript/reported_values.tex`, a generated file of `\newcommand` definitions
(`\Sim...` for the Figure-3 simulation, `\Rec...` for the Figure-4 recording,
and a few shared ones such as `\HpdCoveragePercent`, the `statespacecheck`
version, and the DOIs), so the chain runs **code → summary JSON → macro file →
prose**. `scripts/emit_reported_values.py` (recipe:
`statespacecheck_paper.reported_values`) takes every analysis value from the two
committed summaries, so these values reach the paper through an artifact. The
emitter also writes three DOI macros: the analysis code's, from the `doi` field
of `CITATION.cff`; the Figure-4 input file's, from `paths.FIGURE04_INPUTS_DOI`;
and that of the `statespacecheck` version the summaries record, read from the
committed `manuscript/software_dois.json`, so emitting runs offline (`--refresh-dois`
adds a newly recorded version's DOI from Zenodo). Upstream acquisition and
sorting parameters, which have no artifact in this repository, remain stated
directly in the Methods.

Prose precision follows the policy defined in the
[`reported_values` module docstring](../src/statespacecheck_paper/reported_values.py):

| Quantity | Prose format | Purpose |
| --- | --- | --- |
| Decoding errors | Two significant figures | Supports the ratios discussed in the Results |
| Flag percentages | Nearest whole percent | Supports comparisons described as substantial, low, or modest |
| Rescued percentages | Nearest whole percent, with exact counts alongside | Describes this recording |
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
