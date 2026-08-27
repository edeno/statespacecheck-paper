# Architecture evaluation — DRY, testability, linear flow

**Date:** 2026-08-26
**Standard:** best-practice architecture — DRY, small single-responsibility testable functions, linear data flow (load → simulate/fit → diagnose → plot → save).
**Method:** import-graph + function-size analysis (verified by hand), two deep-read agents on the god functions (core compute layer; plotting + Fig-4 pipeline).

> **Correction addendum (2026-08-26, post-review).** Three specifics below were superseded by reviewer feedback; the executable plan at `.claude/docs/plans/architecture-cleanup/` is authoritative. (1) `plot_likelihood_columns` (plotting.py:678) is **already** the shared likelihood-rendering primitive used by all three real-data paths — do **not** add a new `_render_likelihood_overlay`; the real duplication is the xarray→numpy prep and the extent geometry. (2) Several "missing" tests already exist (test_analysis.py:296, :382; test_figure03_phases.py:106, :258) — extend them, don't duplicate; only an event-rank-tolerance test and base-rate validation are genuinely missing. (3) The sparse-population baseline gain is intentionally nonzero, so "silent cells" is stochastic, not an invariant — test parameterization equivalence + rate regime instead. Same-environment pickle/pixel parity is temporary refactor scaffolding, not a durable test.

## Executive summary

The **skeleton is healthy; the muscles are overgrown.** Module layering is clean and acyclic, the numerical primitives are already small/pure/tested, and the real-data path *reuses* the core rather than forking it. The architecture debt is concentrated in a handful of **orchestrator god-functions** (300+ lines, 7–9 concerns each) and **localized DRY overlap in the plotting layer** (~250 lines of similar row layout across two real-data plotters that are not exact duplicates). The highest-leverage fixes are small extractions that pull numerically subtle / control-integrity-critical logic into named functions where a failure can be **localized** and key invariants get **focused** tests. (The branches are already exercised through public-function tests; the defect is that failures aren't localizable and some invariants lack their own test — not that the code is unreachable.)

---

## 1. Module architecture & layering — STRONG

**Scope of this claim:** the graph below covers the **core `src/statespacecheck_paper` module slice** analyzed here. It does **not** include the `interactive/` viewer subpackage, `load_local_data`/`paths`, the `interactive` cache construction, or the `scripts/` layer — those were not mapped, so this supports a conclusion about the core modules, not the whole repository.

Within that slice the internal dependency graph is acyclic and correctly layered:

```
style ─┐                 (leaves: no internal deps)
simulation ─┐
            ├─> analysis ──> plotting ──┬─> figure02_panels
            │       │                   └─> figure03_demo
            │       └─> real_data_analysis ──> real_data_plotting
schematic ──> style
```

- No circular imports.
- `real_data_analysis` imports `analysis` and **forwards** to `compute_per_cell_diagnostics_from_rates` (real_data_analysis.py:579) — the real-data path adapts the core engine, does not duplicate it.
- Core KL/HPD math lives in the external `statespacecheck` package (`ssc.kl_divergence`), not reimplemented in-repo.

**One boundary issue:** `scripts/generate_figure04.py` is **610 lines** — 3× the other figure scripts and well over the repo's own "<200-line thin orchestration" rule. It delegates the *science* to imported functions, but holds ~240 lines of layout/pixel-nudging that belong in a module helper (see §4).

---

## 2. DRY — good at module level, poor in the plotting layer

**Good:** core math centralized (external pkg + `analysis`), real-data adapts not forks.

**Localized violations (all in the plotting layer):**

- **Real-data plotters share a similar 6-row layout with several near-duplicate blocks (~250 lines of overlap).** `plot_single_model_diagnostics` (real_data_plotting.py:1350–1634) and `plot_model_comparison_with_posterior` (:1007–1347) render the same row structure (one vs two columns). **They are not exact duplicates** — the single-model plot's likelihood overlay comes from place fields (mean-per-spike, :1480–1494); the comparison plot's comes from decoder `exp(log_likelihood)` (:1181). Only the source-independent blocks should be shared:
  - likelihood-overlay sequence (exp → dropna → unstack → sum → extent → plot) copy-pasted **3×**: :1181–1235, :1512–1558, and the extent math again at :1498–1502.
  - `height_ratios=[2,2,1.5,1,1,1]` grid hard-coded twice (:1101, :1431); `has_spikes_mask` (:1134 vs :1442); predictive row (:1139 vs :1446); track-graph overlay (:1252 vs :1573); raster row (:1269 vs :1589); scatter rows (:1294 vs :1599).
  - **Fix:** share the *rendering primitive and geometry* only — `_render_likelihood_overlay(ax, likelihood_2d, extent, …)` (takes an already-computed likelihood so each caller keeps its own source) and `_halfpixel_extent(...)`. Do **not** extract a full column renderer that hides which likelihood each plotter uses. Retiring the comparison plotter (it feeds only exploratory scripts) is a separate, deliberate workflow-removal decision — not a DRY freebie, since those scripts would go too.
- **`−log(p)` spike_prob transform repeated 5×** in `plot_per_cell_diagnostic_scatter` (:761–763, :775, :848–849) and `plot_per_spike_metric_hexbin_row` (:1714–1715, :1747). Extract `_neglog(x, eps)` / a `MetricTransform` holding the transform + its inverse-on-threshold.
- **Figure-2 axis styling repeated ~9×** (spine-off + first/last tick + fontsize 8) in figure02_panels.py. Noted for awareness, but **not recommended for a shared abstraction** — a broad `style_minimal_axes` helper with mode flags would obscure per-panel intent; leave the styling explicit. (Rejected per reviewer feedback.)
- **Two *distinct* spike-rank computations (not a duplicate to merge)** — the general `simulation.spike_prob_rank` ranks all cells at every step (batched mask `(n_time, n_cells, n_cells)`) and is unused; the inline real-data rank at analysis.py:1495–1507 compares only the *observed* cell against all cells at `(n_events, n_cells)` memory, essential for the hundreds-of-thousands-of-events case. Extract the specialized event-rank (§3 priority #1); do **not** collapse the two into one implementation.

---

## 3. Small testable functions — primitives good, orchestrators are god-functions

**Already good:** `_condition_on` (analysis.py:69), `normalized_single_spike_likelihood` (:859), `softmax_with_shift` (simulation.py:47), `predictive_mark_probabilities` (simulation.py:283) — small, pure, tested. `plot_combined_diagnostics` (plotting.py:1167) is a clean orchestrator over small `_plot_figure3_*` row helpers — this is the model the rest should follow.

**God-functions (the debt):**

### `decode_and_diagnostics` — analysis.py:1027–1363 (~337 lines, ~9 concerns)
Mostly-linear pipeline (validate → allocate → filter → warn → expand events → diagnose → override → package), but two branchy hotspots. Extract:
- `_resolve_base_rates(...)` (:1200–1213) — build-or-validate rate table; two raise paths, untested.
- `_expand_spike_events(spikes) -> (time_ind, cell_ind)` (:1289–1295) — count-expansion + `+1` offset; index arithmetic, easy to get wrong.
- `_step_observation_model(window, base_transition, base_rates)` (:1227–1243) — collapse the two window selectors.
- `_combined_and_spike_likelihoods(spikes_t, rates_t)` (:1248–1263) — per-step likelihood trio; depends only on `spikes[t]`/`rates_t`, **not** on `posterior[t-1]`, so it can even be lifted out of the recursion.
- `_run_filter(...)` (:1222–1287) — the sequential loop reduced to predict + `_condition_on` + writes.
- `_apply_window_rate_overrides(...)` (:1322–1347) — the six-array lockstep mutation; most defect-prone block, with no focused test that would localize a fault to it.
- Dead `rng` param (`_ = rng`, :1167) — drop it.

### `compute_per_cell_diagnostics_from_rates` — analysis.py:1371–1530 (~4 concerns)
Extract **`_event_spike_prob_rank(pred_chunk, rates, cell_ind)`** (:1495–1507) — the FP-tolerant cumulative rank (`rank_atol = eps·n_bins·16·max`). Most numerically subtle, platform-sensitive logic in the file, with **no focused test** on its tolerance invariant. This is a *specialized* per-event, memory-lean computation — keep it distinct from the general `simulation.spike_prob_rank`, do not merge. Optionally also `_batch_event_diagnostics(...)` (:1480–1508) so the batch loop becomes a thin memory wrapper.

### `run_figure03_simulation` — figure03_demo.py:170–506 (~337 lines, ~7 concerns) — TANGLED
(Note: `_add_phase` itself is only 6 lines at :245–250; the size is the enclosing function.) Three nested closures (`_walk`, `_spikes_position_tuned`, `_add_phase`) thread **three mutable accumulators** — `phases`, `phase_labels`, scalar `x_last` — through eight inline phase blocks. `x_last` threads every phase's start to the prior phase's last sample — this is *intended* trajectory continuity, not accidental coupling, and should be preserved. The readability issue is elsewhere: labels are assigned by list position (`PHASE_LABELS[len(phases)]`), a hidden order-dependency that should be replaced by passing the label explicitly; and each phase is exercised only through the whole-simulation test, so a fault in one phase can't be localized. *Normal* cells are generated inline per-phase while *sparse* cells are generated in one post-hoc block (:380–417) keyed to global boundaries — the observation model is assembled in two places kept consistent by hand. Note: keep the eight phases a plainly-written top-level sequence; extract only scientifically-meaningful phase functions (below), **not** a generic accumulator. Extract:
- `simulate_drift_phase(...)` (AR(1) loop :326–332), `simulate_replay_phase(...)` (splice :294–321), `simulate_sparse_approach_phase(...)` (:339–357).
- **`build_sparse_population(...)` (:380–417) and `build_figure03_rate_tables(...)` (:~419–462)** — pull decoder-model construction out so the generative-vs-decoder rate pairing (the control-integrity property: a "clean" control must carry no hidden misfit) gets its own focused test.
- Extract the scientifically-meaningful phases as named top-level functions (`simulate_replay_phase` :294–321, `simulate_history_dependent_phase`, `simulate_drift_phase` :326–332, `simulate_sparse_approach_phase` :339–357); pass each phase label explicitly instead of by list position; keep the eight phases as a plainly-written top-level sequence and preserve `x_last` continuity. **No** generic accumulator/reducer — that would hide which phase is which.

### Real-data plotters + `run_demo` — exercised only end-to-end
`plot_single_model_diagnostics`, `plot_model_comparison_with_posterior`, and `run_demo` are currently exercised only through full-figure rendering, so a fault isn't localizable. Extracting the §2 helpers makes their pure sub-units directly and localizably testable. **Caveat (do not over-consolidate):** the two plotters are *not* exact duplicates — `plot_single_model_diagnostics` builds its likelihood overlay from place fields (mean-per-spike), while the comparison plotter uses decoder `exp(log_likelihood)`. Share only the source-independent pieces (imshow rendering, half-pixel extent geometry, the −log(p) transform); keep each likelihood *source* explicit at the call site.

---

## 4. Linear data flow

- **`analysis.py` functions: mostly-linear.** The `for t in range(1, n_time)` filter loop (analysis.py:1222) is a *legitimate* sequential recursion (`posterior[t] ← posterior[t-1]`), **not** a vectorization smell — do not "fix" it. Per-step likelihood arrays inside it are the only non-sequential part.
- **`run_figure03_simulation`: tangled** (mutable accumulators + position-indexed labels + two-place cell generation). Least linear function in the repo.
- **`run_demo` (generate_figure04.py:154–596): linear first half, concern-mixed second.** Clean load → fit → decode → diagnose → cache → print (171–324). Then ~240 lines (355–595) interleave subfigure scaffolding, plotter calls, y-lim harmonization, GID artist toggling, label placement, and **inline pixel-nudging** (`scale_bar_shift=22.0`, `visual_edge_correction_px=7.0` "measured on the exported PNG", manual `set_position`). Split:
  - `_load_or_compute_fig4_bundle(use_cache) -> Fig4Bundle` (171–283),
  - `_place_track_inset(...)` (445–520), `_layout_hexbin_row(...)` (522–591), with magic pixel constants promoted to named, documented module constants.
  - `run_demo` becomes: bundle → subfigures → render stacks → place inset → layout hexbins → save.

---

## Prioritized refactor plan (payoff = testability + readability ÷ churn)

1. **`_event_spike_prob_rank`** (analysis.py:1495–1507) — tiny, numerically critical, no focused test, near-zero churn. Do first. Delete the general `simulation.spike_prob_rank` as unused (distinct computation, **not** a merge).
2. **`build_figure03_rate_tables` + `build_sparse_population`** (figure03_demo.py:380–462) — gives control-integrity (generative vs decoder rates) its own focused test; highest scientific-risk area.
3. **`_resolve_base_rates`, `_expand_spike_events`** (analysis.py) — pure, branchy, cheap.
4. **`_apply_window_rate_overrides`** (analysis.py:1322–1347) — most defect-prone decoder block.
5. **`_render_likelihood_overlay` (source-explicit) + `_halfpixel_extent` + `_neglog`** (real_data_plotting.py) — share rendering/geometry/transform only. Retiring the comparison plotter + its exploratory scripts is a separate, deliberate decision, not part of this.
6. **`run_demo` layout helpers** + move layout logic out of the 610-line `generate_figure04.py`.
7. **Named scientific phase functions** (`simulate_replay_phase`, `simulate_history_dependent_phase`, `simulate_drift_phase`, …) + explicit phase labels in figure03_demo.py — keep the plain 8-phase top-level sequence and `x_last` continuity; **no** generic accumulator. Biggest readability win, highest churn; do last.

**Deferred / rejected:** a broad `style_minimal_axes` axis-styling abstraction (rejected — keeps per-panel styling explicit); any refactor motivated by line count alone; deleting exploratory plotters without deliberately removing their workflows.

Steps 1–4 are pure-function extractions with existing behavior pinned via before/after `array_equal` tests. Steps 5–7 are plotting/orchestration, verified by regenerating the seeded figures and diffing PNGs.

## Verdict

Layering and the numerical core (within the core-module slice mapped here) are already best-practice; the debt is orchestration-shaped — a few 300+ line functions that fold 7–9 concerns together and bury the most defect-prone logic (the six-array window override, the FP-tolerant rank, the two-place generative-vs-decoder rate pairing) where a failure can't be localized to it. DRY is fine across modules; the plotting-layer overlap is real but must be consolidated only where each row's likelihood *source* stays explicit. Do the small pure-function extractions (steps 1–4) first: low churn, and they give the numerically subtle and control-integrity-critical code its own focused, localizing tests.
