# Code review: readability & flow for a scientific-paper repo

**Date:** 2026-08-26
**Scope:** Whole-repo review against the standard *"readable by a naive scientist, logical/sensible flow, nothing extraneous."*
**Method:** Four parallel review agents (core modules, plotting/figures, real-data pipeline, repo hygiene) plus independent verification of every dead-code claim and the Figure-2 naming claim (grep across `src/`, `scripts/`, `notebooks/`).

## Summary

The **actual figure pipeline is sound** — `src/` modules + thin `generate_figure0{1..4}.py` scripts, seeded sims, documented real-data path, no build artifacts or `.DS_Store` tracked. The problems are almost entirely **accreted cruft and stale docs**, not broken logic. None of this touches scientific correctness. Three patterns account for most of it.

---

## Theme 1 — Dead functions kept alive only by their own tests (~800 lines)

Each confirmed to have **zero callers** in `src/`, `scripts/`, or `notebooks/` — only references are docstrings and `>>>` doctest examples. They pass tests, so they look load-bearing, but nothing uses them.

| Function | Location | ~lines |
|---|---|---|
| `Transformed` + `transform_metrics` | `analysis.py:1969` | ~145 |
| `likelihood_grid_for_counts` | `analysis.py:899` | ~60 |
| `spike_prob_rank` (logic duplicated inline at `analysis.py:1499`) | `simulation.py:350` | ~97 |
| `simulate_spikes_flat_rate` | `simulation.py:556` | ~37 |
| `plot_misfit_examples` | `plotting.py:418` | ~200 |
| `plot_posterior`, `plot_overlap_regions`, `plot_overlap_trace`, `plot_acausal_state_prob` | `real_data_plotting.py:403`+ | ~260 |
| `get_multiunit_population_firing_rate`, `find_sustained_low_overlap` | `real_data_analysis.py:94` | ~90 |

`Transformed`/`transform_metrics` are especially clear-cut: CLAUDE.md says their consumers are `plot_transformed`/`plot_original` in `plotting.py` — **those functions no longer exist**. The whole transform-for-visualization subsystem is orphaned.

**Recommendation:** delete these along with their tests (deleting the tests is correct — they only test dead code). Removes ~800 lines a reader must otherwise rule out.

---

## Theme 2 — Docstrings left behind by the dict → dataclass migration (actively misleading)

The decoder now returns a `PerCellDiagnostics` frozen dataclass, but many docstrings still type the return as `dict[str, np.ndarray]` and tell the reader to index keys that don't exist. `diag["hpd_overlap"]` would raise.

- `analysis.py:1406` — `compute_per_cell_diagnostics_from_rates` Returns section lists dict keys (`spike_time_ind`, …) that aren't fields of the dataclass it returns.
- `real_data_plotting.py:692`, `:1050`, `:1388`, `:1662` — four more `dict`/`Mapping[str, np.ndarray]` docstrings for the same dataclass.

**Recommendation:** rewrite these Returns sections to describe `PerCellDiagnostics` attributes.

---

## Theme 3 — Stale hand-maintained index lists (docs contradict reality)

- `real_data_analysis.py:14` — module "Key Components" lists `compute_per_cell_likelihood` (**doesn't exist**) and omits most functions that do.
- `analysis.py:14–15` — index still advertises the dead `Transformed`/`transform_metrics`.
- `figure02_panels.py:3` — docstring says "11 panel helpers"; there are 9.
- **CLAUDE.md architecture tree** omits the entire `interactive/` subpackage, `generate_figure04.py`, and the scaffolding scripts; **README module list** omits `paths`, `schematic`, `figure02_panels`, `figure03_demo`, `real_data_analysis`, `real_data_plotting`. README's "351 tests" vs actual 335.

**Recommendation:** delete the hand-maintained per-module index blocks (they rot; the code is the index) and re-sync CLAUDE.md/README once the deletions above are done.

---

## Individually notable (not covered by the themes)

- **HIGH — Figure-2 panel naming is a three-way maintenance trap.** Function suffix letters (`_a`/`_d`/`_g`), mosaic axis keys (`A`/`B`/`C`), and printed labels (`a`/`b`/`c`) are three misaligned systems: `plot_kl_panel_a` → axis `"A"` → printed label `"c"`; `plot_hpd_panel_d` → axis `"B"` → label `"a"`; `plot_ppc_panel_g` → axis `"C"` → label `"b"`. The published figure is *correct* (a=HPD, b=Predictive, c=KL, matching the manuscript); the code is confusing. Fix: rename helpers to metric+role (`plot_hpd_distribution`, `plot_hpd_threshold`, …) and drop the vestigial letters. `scripts/generate_figure02.py:108`.
- **MEDIUM — `style.py` documents ~12 `COLORS` entries the code refuses to use.** The phase-background pastels (`phase_baseline`/`phase_remap`/…) are dead — `add_phase_boundaries` uses saturated colors instead and even comments that the pastels "wash out." Only `phase_replay` survives. `style.py:108`.
- **MEDIUM — hardcoded reward-well assumption in a general helper.** `reward_well_nodes=list(range(6))` at `real_data_plotting.py:1266` (and `:1586`) assumes nodes 0–5 are wells, while `generate_figure04.py:458` correctly derives them from `track_graph.degree`. Silently mislabels wells on any other track.
- **LOW — speculative `rng` param** "reserved for future use" in `decode_and_diagnostics` (`analysis.py:1035`); unexplained `× 16` in `rank_atol` (`analysis.py:1502`, `simulation.py:430`); a few magic layout constants in `generate_figure04.py:468–591`.
- **LOW — other doc nits:** `add_scalebar` docstring says `default 7` but signature is `8` (`real_data_plotting.py:71`); `Transformed` docstring says `-log10` but code uses natural log (moot if deleted); `DecodeParams` timeline block omits the replay control (`analysis.py:172`).

---

## Structural / repo-shape

- **`interactive/` (~175 KB, ~19% of the test suite) is NOT cruft.** README documents it as a companion pyqtgraph viewer, it's gated behind an optional extra, and `figure03_demo.py` is deliberately coupled to keep its cache byte-identical. But it's **absent from the CLAUDE.md architecture tree**, so a naive reader can't tell it produces no manuscript figure. Fix = one sentence in the docs labeling it an optional exploration tool — not relocation.
- **Scaffolding scripts beside the real generators** — `sanity_check_figure04{a,b}.py`, `find_{immobile_replay,continuous_wins}_windows.py`, `benchmark_figure04_viewer.py`. One-off window-pickers / benchmarks, not figure generators (none called by `generate_all_figures.py`). `find_immobile_replay` even has its own test. They obscure which scripts are load-bearing. Fix: move to `scripts/exploratory/` with a one-line README.
- **Notebooks (~3.1 MB with embedded outputs committed)** — `fig4.ipynb` alone is 1.8 MB. CLAUDE.md sanctions "messy" notebooks, so this is a soft call, but an `nbstripout` pre-commit hook would stop the bloat.
- **`analysis.py` at 2114 lines** is the one file that genuinely strains readability. Beyond the dead code, ~280 lines of Figure-3 summary-heatmap orchestration (`analysis.py:1684`) could move to a figure-3-scoped module.

---

## Suggested order of attack

1. **Delete the ~800 lines of dead-but-tested code** (Theme 1) + their tests — biggest readability win, lowest risk.
2. **Fix the misleading dataclass docstrings** (Theme 2) and rename the Figure-2 panels — prevents readers being actively misled.
3. **Prune stale index lists + re-sync CLAUDE.md/README** (Theme 3) once 1–2 land.
4. Optional polish: `style.py` dead colors, reward-well hardcoding, relocate scaffolding scripts, strip notebook outputs.

## Verified vs. reported

- **Independently verified:** all 14 dead functions in Theme 1 (zero non-test callers); `plot_transformed`/`plot_original` nonexistence; the Figure-2 three-way naming mismatch; no build artifacts / `.DS_Store` / coverage files tracked in git.
- **Reported by agents, high-confidence but not each line re-checked:** individual docstring line numbers in Themes 2–3 and the LOW nits.
