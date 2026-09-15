# statespacecheck-paper

**Local goodness-of-fit measures for neural decoding**

This repository contains the source code and supplementary materials for the paper demonstrating `statespacecheck`. The paper describes three local diagnostics evaluated at individual spikes: HPD overlap (a geometric relation between the one-step predictive state distribution and a spike's normalized single-event likelihood), a rank-based predictive check on the spike's identity (mark surprise under the model's predictive mark distribution), and KL divergence (distributional difference). Each identifies localized discrepancies of a stated kind; none certifies accurate or calibrated state inference. The paper characterizes what each does and does not detect in a simulation with an exactly matched null and in one hippocampal recording, including how much flag removal broadening a prediction can produce on its own.

## Repository Structure

This is a **paper/research repository** (not a library). The code is organized into:

- **`src/statespacecheck_paper/`**: Reusable modules (styling, simulation, analysis, plotting)
- **`scripts/`**: Thin CLI adapters for the importable figure-generation recipes
- **`manuscript/figures/`**: Generated figure outputs (PDF and PNG)
  - `manuscript/figures/main/`: Main text figures
  - `manuscript/figures/supplementary/`: Supplementary figures
- **`tests/`**: Comprehensive test suite (run `uv run pytest`)

**For developers**: See [CLAUDE.md](CLAUDE.md) for detailed development guide including module organization, coding standards, and where to add new functionality.

## Reproducing the paper figures

Each main-text figure is produced by exactly one script in `scripts/`. Figures 1–3
are fully self-contained (a fixed-seed simulation, no external data); Figure 4
additionally needs the real hippocampal dataset.

For the full manuscript-to-figure map — each figure's entry point, configuration,
module reading order, data boundary, output, and guarding tests — see
[docs/figure-pipeline.md](docs/figure-pipeline.md).

| Figure | Entry point | Input | Output | Data requirement |
| --- | --- | --- | --- | --- |
| 1 | `scripts/generate_figure01.py` | simulated | PDF + PNG | none |
| 2 | `scripts/generate_figure02.py` | simulated | PDF + PNG | none |
| 3 | `scripts/generate_figure03.py` | simulated | PDF + PNG + summary JSON | none |
| 3 (sensitivity) | `scripts/generate_figure03_sensitivity.py` | simulated | summary JSON (supplementary) | none |
| 4 | `scripts/generate_figure04.py` | derived real data | PDF + PNG + summary JSON | documented dataset (see below) |
| 4 (supplement) | `scripts/generate_figure04_supplement.py` | cached Figure-4 decode | PDF + PNG + summary JSON (supplementary) | the Figure-4 caches |

The summary JSON files are reproducibility artifacts, not merely copies of
console output. Figure 3 uses schema version 6, Figure 4 schema version 4, and
the two supplementary summaries schema version 1. All record each flag
threshold with its exact inclusive comparison operator, plus hashes of the
scientific source tree and `uv.lock`. Figure 3 records the independent
matched-null calibration (seeds, pooled event count, realized null flag rates,
rank-tail percentages), per-condition medians and per-realization
distributions of flag percentages and event counts, filtered-posterior
accuracy/coverage/region size, and predictive-region size. Figure 4 records the
unit count, decode-cache and diagnostics-cache fingerprints, installed decoder
and `statespacecheck` versions, and SHA-256 checksums of each of its five
derived input exports; the supplement records predictive-region sizes, the
uniform-mixture counterfactual, coverage sensitivity, and the firing-rate and
behavior stratification. See [the schema notes](docs/figure-pipeline.md#machine-readable-summary-schema).

```bash
# Reproduce the locked environment, then regenerate every figure:
uv sync --frozen
uv run python scripts/generate_all_figures.py
# Outputs land in manuscript/figures/main/ (figures at 450 DPI).
uv run python scripts/emit_reported_values.py
make -C manuscript
```

The emitter reads the four summary JSONs and writes `manuscript/reported_values.tex`,
which supplies the manuscript's reported values. Run it after regenerating any
summary. It can also run directly from the committed summaries without rerunning
the analyses. See [the reporting policy and artifact checks](docs/figure-pipeline.md#from-summary-to-prose-the-reported-value-macros).

Figures 1–3 reproduce deterministically from the seeded simulation. **Figure 4**
uses the real hippocampal recording of [Comrie et al. 2024](https://doi.org/10.1101/2024.09.23.613567),
available on the DANDI Archive as dandiset
[001942](https://dandiarchive.org/dandiset/001942). It is **not** included here
(large). Note that the decoder consumes five derived exports (linearized
position, spike times, track graph, edge order/spacing), not the raw NWB files;
the raw-recording → exports step is external to this repository — see
[docs/figure-pipeline.md](docs/figure-pipeline.md#figure-4--real-data-decoder-diagnostics)
for the export contract. Place the exports under `data/` (or set
`STATESPACECHECK_DATA_PATH`) before running `generate_figure04.py`; the decode is
cached under `data/` on first run.

The `statespacecheck` package this paper *demonstrates* is a separate dependency
with its own [documentation](https://edeno.github.io/statespacecheck) and
[repository](https://github.com/edeno/statespacecheck) — its API, terminology, and
usage examples live there. This README covers reproducing the paper, not using the
library.

## Overview

State space models are powerful tools for relating neural activity to latent dynamic brain states (e.g., memory, attention, spatial navigation). The core assumption is that complex, high-dimensional neural activity can be related to low-dimensional latent states through:

1. **State transition model**: How latent states evolve over time
2. **Observation model**: How neural activity relates to the current latent state

The posterior distribution combines current observations with the prediction
from accumulated history. The diagnostics examine how each spike agrees with
that prediction. A discrepancy can identify model misfit, but it needs context,
and the absence of discrepancies does not establish accuracy: in Figure 3 the
coherent reflected-map misfit is decoded confidently at the wrong location with
near-null flag rates, the sparse regime is flagged by KL divergence under an
exactly correct model, and in Figure 4 adding a few percent of uniform mass to
the Continuous model's predictions removes as many HPD flags as the fitted
alternative model does.

## Installation

For **exact reproduction** (the environment CI uses), sync the locked
dependencies — this installs the precise pinned versions from `uv.lock`:

```bash
uv sync --frozen
```

For **development**, add the locked development tools to the editable install:

```bash
uv sync --frozen --extra dev
```

### Optional extras

```bash
# Interactive decoder viewer (pyqtgraph + PySide6 desktop app, plus
# zarr / pyarrow for the on-disk cache it consumes).
uv sync --frozen --extra interactive

# Development tools plus the viewer dependencies (the CI environment).
uv sync --frozen --extra dev --extra interactive
```

### Installing Dependencies from GitHub

This project may include dependencies installed directly from GitHub repositories. When using `uv`, these require special handling:

1. **Enable direct references** in `pyproject.toml`:

   ```toml
   [tool.hatch.metadata]
   allow-direct-references = true
   ```

2. **Specify GitHub dependencies** in `pyproject.toml`:

   ```toml
   dependencies = [
       "package-name @ git+https://github.com/username/repo.git",
   ]
   ```

3. **Update lock file** when dependencies change:

   ```bash
   # Update specific package from GitHub
   uv lock --upgrade-package package-name

   # Sync environment with updated lock
   uv sync
   ```

**Important**: `uv` uses a lock file (`uv.lock`) to ensure reproducible installs. When a GitHub dependency is updated upstream, you must explicitly update the lock file—`uv` will not automatically fetch the latest commit.

## The `statespacecheck` package

The diagnostics demonstrated here (KL divergence, HPD overlap, rank-based
predictive check) are provided by the standalone
[`statespacecheck`](https://github.com/edeno/statespacecheck) package, a
dependency of this repository. Its API reference, terminology, and standalone
usage examples live in its own
[documentation](https://edeno.github.io/statespacecheck) — they are not
duplicated here to avoid drift. This repository shows how the paper *applies*
those diagnostics; for the decoder-integrated usage, see
[`decoding.decode_with_diagnostics`](src/statespacecheck_paper/decoding.py) and
the figure-generation modules.

## Interactive viewer

A pyqtgraph desktop app (`statespacecheck_paper.interactive`) renders
the decoder's per-time outputs alongside the diagnostics so you can
scrub through a session, click on a spike to inspect its bin, and
swap between the predictive / filtered / smoothed posterior on the
slice column. The viewer reads from a chunked on-disk cache (Zarr +
Parquet + `.npz` sidecars); it never realises the full posterior in
memory.

Two dataset kinds are supported:

- **Real-data decoder caches** (`continuous` / `contfrag` models from
  fitted `non_local_detector` decoders).
- **Figure-3 simulation cache** — the simulated demonstration with a
  matched-null baseline, remap / history-dependent-firing / drift /
  reflected-map misfits, and the replay and sparse-population controls
  (clean-recovery windows between).

### Build a cache

```bash
# Real data (figure 4): derives figure04_continuous.zarr +
# figure04_contfrag.zarr and shared sidecars from the same canonical
# {epoch}_fig4_cache.joblib bundle used by the static figure.
uv run python -m statespacecheck_paper.interactive.cache build \
    --data-dir data \
    --cache-dir data/cache \
    --model both

# Figure-3 simulation: runs the demo simulation + decoder and writes
# simulation.zarr + sidecars.
uv run python -m statespacecheck_paper.interactive.cache build-simulated \
    --cache-dir data/cache/simulation
```

### Open the viewer

```bash
# Real-data model (Continuous or ContFrag).
uv run python -m statespacecheck_paper.interactive \
    --cache-dir data/cache --model continuous

# Figure-3 simulation.
uv run python -m statespacecheck_paper.interactive \
    --cache-dir data/cache/simulation --simulation
```

### Controls

| Action | Binding |
| --- | --- |
| Recenter on a point | Click anywhere on a time-axis panel |
| Pin a spike | Click the spike on the raster or a metric panel |
| Unpin | Click the pinned spike again, or `Esc` |
| Step center by one bin | `←` / `→` |
| Step center by one window | `Shift+←` / `Shift+→` |
| Play / pause auto-scroll | `Space` |
| Scrub auto-scroll speed | `,` / `.` |
| Resize window width | Mouse wheel over a time-axis panel, or `[` / `]` |
| Reset to a 20 s context window | `R` |
| Toggle real-data model | `M` (real-data caches only) |

The slice panel's "Overlay" combo switches the population-likelihood
plot's blue overlay between predictive `p(x_t | y_{1:t-1})`, filtered
`p(x_t | y_{1:t})`, and smoothed `p(x_t | y_{1:T})` distributions.
Smoothed is only available for caches that include `acausal_posterior`
(rebuild via `cache build --force` if the entry is greyed out).

## Development

This repository follows a modular architecture where reusable code lives in `src/statespacecheck_paper/` and figure scripts orchestrate. See [CLAUDE.md](CLAUDE.md) for comprehensive development guide.

### Quick Setup

```bash
# Install UV package manager if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create/sync .venv with the locked development and viewer dependencies
uv sync --frozen --extra dev --extra interactive
```

### Running Tests

```bash
# Run the default suite (slow tests are excluded)
uv run pytest

# Run specific module tests
uv run pytest tests/test_simulation.py -v

# Run with coverage (report the exact totals from the summary line)
uv run pytest --cov --cov-report=term-missing

# Generate HTML coverage report
uv run pytest --cov --cov-report=html
open htmlcov/index.html
```

### Generating Figures

See [Reproducing the paper figures](#reproducing-the-paper-figures) for the
figure→entry-point→data table and the Figure-4 data requirement.

```bash
# Generate all figures
uv run python scripts/generate_all_figures.py

# Or generate an individual figure (one per main-text figure)
uv run python scripts/generate_figure01.py   # Fig 1  (simulated)
uv run python scripts/generate_figure02.py   # Fig 2  (simulated)
uv run python scripts/generate_figure03.py   # Fig 3  (simulated)
uv run python scripts/generate_figure03_sensitivity.py   # Fig 3 sensitivity summary
uv run python scripts/generate_figure04.py   # Fig 4  (needs the real dataset)
uv run python scripts/generate_figure04_supplement.py    # Fig 4 supplement (from the caches)

# Outputs saved to manuscript/figures/main/ directory as PDF and PNG (450 DPI)

# After changing any summary, refresh the prose values and rebuild:
uv run python scripts/emit_reported_values.py
make -C manuscript
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Type checking
uv run mypy src/

# Run all checks
uv run ruff format . && uv run ruff check . && uv run mypy src/ && uv run pytest
```

### Module Organization

- **`style.py`**: Shared styling (WONG palette, figure defaults, save functions)
- **`simulation.py`**: Simulation utilities (random walks, spikes, place fields)
- **`diagnostics.py`**: Goodness-of-fit diagnostics (HPD overlap, rank-based predictive p-value, KL); leaf layer
- **`decoding.py`**: General Bayesian decoder (`decode_with_diagnostics`)
- **`plotting.py`**: Reusable plotting functions (HPD regions, diagnostic plots)
- **`schematic.py`**: Graphical-model and Bayesian-equation diagrams (Figure 1)
- **`figure01_generation.py` / `figure02_{panels,generation}.py`**: Testable composition and generation recipes for Figures 1–2
- **`figure03_{protocol,simulation,summary,plotting,generation,sensitivity}.py`**: Figure-3 protocol, simulation (matched-null generator, phased session, rate-matching calibration), calibration + per-condition summary, plotting, generation recipe, and sensitivity settings
- **`figure04_{decoder,place_fields,diagnostics}.py`**: Figure-4 real-data decoder construction/config, place-field extraction, and diagnostics
- **`figure04_{plot_primitives,track_plots,panels}.py`**: Figure-4 plotting helpers, track-graph rendering, and raster/diagnostic panels
- **`figure04_{cache,workflow,layout,generation}.py`**: Figure-4 decode/diagnostics caches, analysis workflow, composition, and generation recipe
- **`figure04_{broadening,supplement_plotting,supplement_generation}.py`**: Figure-4 supplement analyses (region sizes, uniform-mixture counterfactual, coverage sensitivity, rate/behavior stratification), plotting, and recipe
- **`reported_values.py`**: Summary-to-LaTeX macro generation and the manuscript's reporting policy
- **`load_local_data.py`**: Real data loading utilities
- **`paths.py`**: Shared `DATA_PATH` / `ANIMAL_DATE_EPOCH` constants (env-overridable)

### Standards

- **Python**: 3.10+ (following [SPEC 0](https://scientific-python.org/specs/spec-0000/))
- **Package manager**: UV (recommended) or pip
- **Dependencies**: See [pyproject.toml](pyproject.toml) for full list
- **Docstrings**: NumPy format with shape specifications
- **Type hints**: Full mypy strict mode compliance
- **Style**: ruff for formatting and linting (100 char line length)
- **Testing**: pytest; core modules (analysis, simulation, style, plotting) kept >90% covered, ~81% overall (the interactive GUI and real-data plotting modules are lower)
- **No `# type: ignore`**: Fix type issues by refactoring, not suppressing

### Adding New Functionality

See [CLAUDE.md](CLAUDE.md) for detailed guidance on:

- Where to add simulation/analysis/plotting code
- How to create new figures
- Testing requirements
- Code quality standards

## Scientific Context

The `statespacecheck` package implements goodness-of-fit diagnostics for state space models used in neuroscience, and this repository demonstrates and applies them. The diagnostics compare each spike with the model's one-step prediction: HPD overlap asks whether the prediction's high-density region and the spike's normalized likelihood region overlap (a nesting-tolerant geometric criterion), the rank-based predictive check asks whether the spike's identity is improbable under the predictive mark distribution (which depends on firing rates as well as the state), and KL divergence asks how different the two state distributions are. A flagged spike localizes a discrepancy; attributing it to the transition model, the observation model, or unit definition needs additional evidence such as an alternative model or scientific context, and passing the checks does not establish that the decoded state is accurate or calibrated.

These diagnostics complement but are distinct from:

- **Cross-validation and held-out predictive scores**: Measure predictive generalization; they can also be inspected locally, but they do not compare the prediction with an individual spike's likelihood.
- **Time-rescaling and count checks for point processes**: Assess the exposure/rate structure that the per-event diagnostics here deliberately omit.

## Citation

If you use this package in your research, please cite the paper:

```bibtex
@article{zeng2026local,
  title   = {Local goodness-of-fit measures for neural decoding},
  author  = {Zeng, Sirui and Comrie, Alison E. and Frank, Loren M. and
             Eden, Uri T. and Denovellis, Eric L.},
  year    = {2026},
}
```

A machine-readable citation in CFF format is also provided in
[CITATION.cff](CITATION.cff).

## License

This repository is dual-licensed:

- **Code** (analysis, scripts, figure-generation pipeline, tests): MIT License — see [LICENSE](LICENSE).
- **Manuscript text and figures** (everything under `manuscript/`, including `manuscript/figures/`): Creative Commons Attribution 4.0 International (CC BY 4.0) — see [manuscript/LICENSE](manuscript/LICENSE).

This split keeps the code permissively reusable while licensing the paper content under CC BY 4.0.
