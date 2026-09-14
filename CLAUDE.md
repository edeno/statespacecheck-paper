# statespacecheck-paper Development Guide

## Project Overview

Source code and supplementary materials for the paper **"Local goodness-of-fit
measures for neural decoding"** — analysis scripts, figure generation code, and
examples demonstrating the `statespacecheck` package.

**Scientific context**: State space models relate neural activity to latent
dynamic brain states. This paper introduces diagnostics — HPD overlap, a
rank-based predictive check, and KL divergence — to assess goodness-of-fit by
comparing one-step predictive state distributions with normalized single-event
likelihoods and checking observed spike marks against their predictive distribution.

**Repository type**: a paper/research repository, not a library. The focus is
reproducible analysis and figure generation. Directory layout, module APIs, and
dependencies are derivable from the source and `pyproject.toml` — read those
rather than duplicating them here.

## Architecture

Reusable code lives in `src/statespacecheck_paper/`; `scripts/` holds thin CLI
adapters whose scientific recipes live in the matching `figureNN_generation.py`
module (so each recipe is importable and testable without executing source text).
The module graph is a DAG enforced in CI; `diagnostics.py` is the leaf.

Key modules and the rationale that the source alone won't tell you:

- **diagnostics.py** — shared goodness-of-fit diagnostics; the dependency-graph
  **leaf** (per-spike-event HPD/KL/rank computation, single-event likelihood,
  predictive-mark probabilities, baseline thresholds).
- **decoding.py** — general Bayesian decoder `decode_with_diagnostics` + the
  per-window override mechanism (`DecoderOverrideWindow`/`DecoderOverrideSchedule`,
  used by Figure 3); depends only on `diagnostics` + `simulation`.
- **figure03_\*** family — protocol (`Figure3Config`, phase ladder) → phased
  simulation → per-condition flag and decoding-error summary → plotting → generation recipe.
- **figure04_\*** family — decoder + config, place-field/marginalized-posterior
  extraction, real-data diagnostics, plotting layers, cache I/O, workflow
  (`Figure4RenderData`), layout, generation recipe. `Figure4Config` (in
  `figure04_decoder.py`) is split into an executable `Figure4DecoderConfig`
  (threaded into decoder construction) and a `Figure4Provenance` (nld-default
  values — `movement_var`, ContFrag transition/initial-condition/concentration/
  regularization — recorded and drift-guard pinned but **not injected**, because
  faithfully injecting them would rebuild the nested transition grid and risk
  changing the decode).
- **load_local_data.py** — `load_neural_recording_from_files` → validated
  `NeuralRecordingData`; loads from pre-exported pickles, no Spyglass DB needed.
- **paths.py** — `DATA_PATH` / `ANIMAL_DATE_EPOCH` constants, env-overridable via
  `STATESPACECHECK_DATA_PATH` / `STATESPACECHECK_ANIMAL_DATE_EPOCH`.
- **style.py / simulation.py / plotting.py / schematic.py** — styling (WONG
  palette), simulation primitives, reusable plotting (HPD regions, likelihood
  columns), and the Figure-1 graphical-model/equation diagrams.
- **reported_values.py** — reads the Figure-3 and Figure-4 summary JSONs and
  emits `manuscript/reported_values.tex`. Its module docstring defines the prose
  reporting policy; standard errors in the summaries do not set printed digits.

## Development Commands

**Always use `uv` for package management and work in the `.venv` environment.**
Never install into a base/global environment. The standard `uv run ruff …`,
`uv run mypy src/`, and `uv run pytest …` invocations apply; `pyproject.toml`
holds the dependency and tooling config. Reproduce the analysis environment with
`uv sync --frozen`; add `--extra dev --extra interactive` for the development
and CI environment. See `README.md` for dependency and lock-update commands.

After changing a figure summary, run `uv run python scripts/emit_reported_values.py`
from the repository root, then `make -C manuscript`. The Makefile tracks the macro
file as an input but does not regenerate it. Source docstrings and comments also
contribute to the provenance hash; see `docs/figure-pipeline.md` for the artifact
refresh procedure for documentation-only source changes.

## Key Design Principles

### Reproducibility first

- Seed with the modern Generator API: `rng = np.random.default_rng(seed)`, and
  thread `rng`/`SeedSequence` through functions rather than the legacy global
  `np.random.seed`.
- Document data sources and preprocessing; save intermediate results when
  computation is expensive.

### Publication-ready figures

- Style through `statespacecheck_paper.style`. Default size (8, 6) in single
  column, (16, 6) double. Axis labels 12pt, ticks 10pt. Colorblind-friendly
  palettes. Export both PNG (preview) and PDF (publication).

### Time-resolved diagnostics

- Arrays are `(n_time, ...)` with time first.
- Vectorize; avoid Python loops over time/space — **with the legitimate
  exception of inherently-sequential recursions** (e.g. the Bayesian filter's
  `for t` loop in `decoding.decode_with_diagnostics`, where step `t` depends on
  `t-1`). Those are not a defect to be vectorized away.
- Handle NaN properly (mark invalid spatial bins); document array shapes.

### Data-structure conventions

- Spatial distributions — 1D: `(n_time, n_position_bins)`; 2D:
  `(n_time, n_x_bins, n_y_bins)`.
- Neural data — spike counts: `(n_time, n_cells)`; the simulation's
  `place_field_rates` table: `(n_bins, n_cells)`.
- State-space outputs — predictive `p(x_t | y_{1:t-1})`, filtered
  `p(x_t | y_{1:t})`, smoothed `p(x_t | y_{1:T})`.
- Manuscript rates are `lambda`; expected counts are `m = lambda * dt`.
  Simulation Poisson inputs are counts per step, even where code names use
  `rate`. Diagnostics normalize rates or expected counts with a common bin
  width equivalently. Do not multiply simulation Poisson inputs by `dt` again.

## Where to Add New Functionality

- **Simulation** (random walks, spike generation, place fields) →
  `simulation.py`. Functions pure and reproducible (take an `rng`).
- **Diagnostics / decoder** → `diagnostics.py` or `decoding.py`; **Figure-3
  code** → the `figure03_*` family. Use dataclasses for config objects.
- **Plotting** → `plotting.py`; return Figure objects; style from `style.py`.
- **Figure styling** → `style.py`.
- **New figures** → `scripts/generate_figureXX.py`: import from shared modules,
  keep the script a thin (<200-line) orchestrator, save to
  `manuscript/figures/main/` or `.../supplementary/`, and add an integration
  test in `tests/test_figures.py`.

**Do not**: add utilities to figure scripts (extract to modules); duplicate code
across scripts; mix simulation/analysis/plotting in one large function; or create
figure-specific versions of general utilities.

## Code Quality Standards

- **Full type hints required; mypy strict mode must pass.** Use `numpy.typing`
  (`NDArray[np.float64]`) for array types.
- **NEVER use `# type: ignore`.** Fix the real type issue — add hints, narrow
  with `isinstance()`, or refactor to be type-safe.
- **Docstrings**: NumPy format with explicit shape specs, e.g.
  `(n_time, n_position_dims)`.
- **Line length** 100. Standard PEP 8 naming. Imports grouped/sorted by ruff.

## Testing

- Small datasets, fixed seeds, `pytest.mark.parametrize` for multiple scenarios;
  test both 1D and 2D spatial arrays and edge cases (empty arrays, NaN, zero
  sums).
- Coverage: aim for >90% on the core modules (diagnostics, simulation, style,
  plotting). The suite is ~81% overall because the interactive GUI and real-data
  plotting modules are exercised less. Regenerate exact per-module percentages
  with `uv run pytest --cov --cov-report=term-missing | tail -1`.
- Run `uv run ruff format . && uv run ruff check . && uv run mypy src/ && uv run pytest`
  before committing; all must pass.
