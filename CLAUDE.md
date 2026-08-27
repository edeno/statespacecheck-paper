# statespacecheck-paper Development Guide

## Project Overview

This repository contains the source code and supplementary materials for the paper **"Local goodness-of-fit measures for neural decoding"**. It includes analysis scripts, figure generation code, and examples demonstrating the `statespacecheck` package.

**Scientific Context**: State space models are widely used in neuroscience to relate neural activity to latent dynamic brain states. This paper introduces diagnostics (HPD overlap, a rank-based predictive check, and KL divergence) to assess model goodness-of-fit by examining consistency between posterior distributions and component likelihood distributions.

**Repository Type**: This is a paper/research repository, not a library. The focus is on reproducible analysis, figure generation, and demonstrating the `statespacecheck` package capabilities.

## Architecture

```
statespacecheck-paper/
├── src/statespacecheck_paper/  # Analysis code and utilities
│   ├── __init__.py             # Package initialization
│   ├── load_local_data.py      # Data loading utilities (file-based)
│   ├── paths.py                # DATA_PATH + ANIMAL_DATE_EPOCH constants
│   ├── style.py                # Figure styling (colors, defaults, save)
│   ├── simulation.py           # Simulation utilities
│   ├── diagnostics.py          # Shared goodness-of-fit diagnostics (leaf layer)
│   ├── decoding.py             # General Bayesian decoder + override mechanism
│   ├── plotting.py             # Generic plotting utilities (HPD, likelihood cols, Fig-1 panel)
│   ├── figure01_generation.py  # Figure-1 composition + save recipe
│   ├── figure02_panels.py      # Figure-2 typed example + panel renderers
│   ├── figure02_generation.py  # Figure-2 semantic layout + save recipe
│   ├── figure03_protocol.py    # Figure-3 config (Figure3Config) + phase ladder
│   ├── figure03_simulation.py  # Figure-3 phased simulation + decode
│   ├── figure03_summary.py     # Figure-3b per-condition flag-percentage summary
│   ├── figure03_plotting.py    # Figure-3 rendering (compose_figure03 + panels)
│   ├── figure03_generation.py  # Figure-3 simulation/summary/render/save recipe
│   ├── figure04_decoder.py     # Figure-4 decoder construction + config (Figure4Config lives here)
│   ├── figure04_place_fields.py # Figure-4 place-field / marginalized-posterior extraction
│   ├── figure04_diagnostics.py # Figure-4 real-data goodness-of-fit diagnostics
│   ├── figure04_plot_primitives.py # Figure-4 shared low-level plotting helpers (GIDs, extents)
│   ├── figure04_track_plots.py # Figure-4 track-graph rendering (1D/2D)
│   ├── figure04_panels.py      # Figure-4 raster + diagnostic panels
│   ├── figure04_cache.py       # Figure-4 decode-cache paths, fingerprint, I/O
│   ├── figure04_workflow.py    # Figure-4 load/fit/decode/cache + summary (Figure4RenderData)
│   ├── figure04_layout.py      # Figure-4 artist arrangement (compose_figure04)
│   ├── figure04_generation.py  # Figure-4 generation recipe (generate_figure04)
│   ├── schematic.py            # Graphical model and equation diagrams
│   └── interactive/            # pyqtgraph interactive diagnostic viewer (app, panels, cache)
├── scripts/                     # Figure generation + exploratory scripts
│   ├── generate_figure01.py    # Figure 1: Schematic and distribution comparisons
│   ├── generate_figure02.py    # Figure 2: Diagnostic demonstrations
│   ├── generate_figure03.py    # Figure 3: Per-cell diagnostics across 8-phase simulation
│   ├── generate_figure04.py    # Figure 4: Real-data decoder + diagnostics
│   ├── generate_all_figures.py # Master script to generate all figures
│   └── exploratory/            # Non-canonical scaffolding (sanity checks, window
│                               # selection, viewer benchmark) — see exploratory/README.md
├── manuscript/                  # LaTeX source files + bundled figures (Overleaf-ready)
│   ├── main.tex                 # Self-contained (own inline preamble)
│   ├── Local-GoF-Paper.bib      # Bibliography (BibTeX, from Zotero); built with iopart-num.bst
│   ├── README.md
│   └── figures/                # Generated figure outputs
│       ├── main/               # Main text figures (PDF + PNG)
│       └── supplementary/      # Supplementary figures (PDF + PNG)
├── notebooks/archive/           # Archived exploratory notebooks (dev scratch, not the pipeline)
└── tests/                       # Test suite (unit + property-based + integration)
    ├── test_style.py
    ├── test_simulation.py
    ├── test_diagnostics.py
    ├── test_decoding.py
    ├── test_figure03_protocol.py
    ├── test_figure03_simulation.py
    ├── test_figure03_summary.py
    ├── test_figure03_plotting.py
    ├── test_figure03_phases.py
    ├── test_figure04_decoder.py
    ├── test_figure04_place_fields.py
    ├── test_figure04_diagnostics.py
    ├── test_figure04_plot_primitives.py
    ├── test_figure04_panels.py
    ├── test_figure04_cache.py
    ├── test_figure04_workflow.py
    ├── test_figure04_layout.py
    ├── test_figure04_generation.py
    ├── test_plotting.py
    ├── test_schematic.py       # Tests for schematic module
    ├── test_figures.py         # Integration tests
    └── test_properties.py      # Property-based tests
```

**Key Modules**:

- **style.py**: Shared styling utilities (WONG palette, COLORS dict, figure defaults, save function)
- **simulation.py**: Simulation functions (random walks, spike generation, place fields)
- **diagnostics.py**: Shared goodness-of-fit diagnostics — the dependency-graph leaf (per-spike-event HPD/KL/rank containers + computation, single-spike likelihood, predictive-mark probabilities, baseline thresholds)
- **decoding.py**: General Bayesian decoder `decode_with_diagnostics` + per-window override mechanism (`DecoderOverrideWindow`/`DecoderOverrideSchedule`); depends only on `diagnostics` + `simulation`
- **figure01_generation.py / figure02_panels.py / figure02_generation.py**: Testable composition/generation recipes for Figures 1–2; Figure 2 uses a typed immutable shared example and semantic layout identifiers
- **figure03_protocol.py**: Immutable Figure-3 configuration (`Figure3Config`), phase-ladder enum (`PhaseBoundary`), and replay-window helper — a leaf module
- **figure03_simulation.py**: Figure-3 phased simulation (`run_figure03_simulation` → `Figure3SimulationResult`), phase simulators, rate tables, place-field remapping
- **figure03_summary.py**: Figure-3b per-condition flag-percentage summary (`build_summary_conditions`, `estimate_realization_summary` → `Figure3RealizationSummary`)
- **figure03_plotting.py**: Figure-3 rendering (`compose_figure03` + the time-series/heatmap panels)
- **figure03_generation.py**: Figure-3 simulation → pooled summary → composition → save recipe (`generate_figure03`)
- **plotting.py**: Reusable plotting functions (HPD regions, diagnostic plots)
- **schematic.py**: Graphical model diagrams and Bayesian equation boxes for Figure 1
- **figure04_decoder.py / figure04_place_fields.py / figure04_diagnostics.py**: Figure-4 analysis layers — decoder construction + config, place-field / marginalized-posterior extraction, and the real-data goodness-of-fit diagnostics. `Figure4Config` (in `figure04_decoder.py`) is split into an executable `Figure4DecoderConfig` (threaded into decoder construction — `position_std`, `block_size`, `position_bin_size_cm`, `sampling_frequency_hz`) and a `Figure4Provenance` (nld-default values — `movement_var`, ContFrag transition/initial-condition/concentration/regularization — recorded and drift-guard pinned, but not injected because faithfully injecting them would rebuild the nested transition grid and risk changing the decode).
- **figure04_plot_primitives.py / figure04_track_plots.py / figure04_panels.py**: Figure-4 plotting layers — shared low-level helpers (GID constants, `-log(p)` transform, imshow extents, distribution heatmap), track-graph rendering, and the raster + diagnostic panels.
- **figure04_cache.py / figure04_workflow.py / figure04_layout.py / figure04_generation.py**: Figure-4 family — cache I/O, the load/decode/summary workflow (`Figure4RenderData`), artist arrangement (`compose_figure04`), and the generation recipe (`generate_figure04`).
- **load_local_data.py**: `load_neural_recording_from_files` → validated `NeuralRecordingData` (typed input contract)
- **paths.py**: Shared `DATA_PATH` / `ANIMAL_DATE_EPOCH` constants (env-overridable)

**Figure Scripts** (in `scripts/`):

- **generate_figure01.py / generate_figure02.py / generate_figure03.py / generate_figure04.py**: Thin CLI adapters; the scientific recipes live in the corresponding `figureNN_generation.py` modules
- **generate_all_figures.py**: Master script to generate all figures

## Repository Structure

### Module Organization

The repository follows a clean separation between **reusable code** (in `src/`) and **figure scripts** (in `scripts/`).

#### Core Modules (`src/statespacecheck_paper/`)

**1. style.py** - Figure Styling Utilities

- `WONG`: 8-color colorblind-friendly palette
- `COLORS`: Semantic color dictionary for consistent styling across figures
- `CMAP_POSTERIOR`, `CMAP_DIAGNOSTIC`: Colormaps for heatmaps
- `set_figure_defaults(context='paper')`: Set matplotlib defaults
- `save_figure(basename, dpi=450)`: Save figures as PDF and PNG
- `get_figure_size(width_type='single')`: Get standard figure dimensions

**2. simulation.py** - Data Simulation

- `normalize(x, axis=-1, eps=1e-10)`: Safe array normalization
- `reflect_into_interval(x, xmin, xmax)`: Reflecting boundary conditions
- `gaussian_transition_matrix(position_bins, step_std)`: Random walk transition matrix
- `safe_log(x, eps=1e-10)`: Numerically stable logarithm
- `place_field_rates(position_bins, centers, scale)`: Gaussian place fields
- `simulate_walk(n_time_steps, step_std, initial_position, position_min, position_max, rng)`: Random walk simulation
- `simulate_spikes_position_tuned(position, place_field_centers, place_field_std, place_field_rate_scale, rng)`: Position-tuned Poisson spikes

**3. diagnostics.py** - Shared Goodness-of-Fit Diagnostics (leaf layer)

- `SpikeEventDiagnostics`: Dataclass returned by the per-spike-event diagnostic computation
- `DecodingDiagnostics`: Dataclass returned by `decode_with_diagnostics`
- `DiagnosticThresholds`: Dataclass for diagnostic thresholds
- `compute_spike_event_diagnostics_from_rates(...)`: Per-spike HPD/KL/rank diagnostics
- `compute_normalized_spike_likelihood(firing_rates)`: Normalized single-spike Poisson likelihood
- `compute_predictive_mark_probabilities(predictive_distribution, mark_intensities)`: Predictive mark distribution (moved here from `simulation.py`)
- `compute_baseline_diagnostic_thresholds(diagnostics, *, baseline_end_index)`: Compute baseline thresholds

**4. decoding.py** - General Bayesian Decoder

- `decode_with_diagnostics(spike_counts, position_bins, transition_matrix, place_field_centers, place_field_std, place_field_rate_scale, override_schedule=None, baseline_firing_rates=None)`: Bayesian filter returning `DecodingDiagnostics`
- `DecoderOverrideWindow` / `DecoderOverrideSchedule`: Optional per-window transition/firing-rate overrides (used by Figure 3)

**5. figure03_protocol.py / figure03_simulation.py / figure03_summary.py / figure03_plotting.py** - Figure-3 Family

- `Figure3Config`: Frozen dataclass for the figure-3 simulation protocol (timeline, cells, remapping)
- `remap_place_field_centers(...)`: Compute remapped place field centers
- `build_summary_conditions(config)` / `compute_condition_flag_percentages(...)`: Figure-3b per-phase flag summary
- `compose_figure03(...)`: Assemble the Figure-3 time-series + summary figure

**6. plotting.py** - Reusable Plotting Functions

- `compute_hpd_region(distribution, coverage)`: Highest posterior density region mask
- `extract_contiguous_regions(mask)`: Find contiguous True regions in boolean array
- `create_distribution_comparison_panel(...)`: Create comparison panels for Figure 1
- `plot_likelihood_columns(...)`: Shared likelihood-column rendering primitive

(`add_phase_boundaries` and the Figure-3 composition now live in `figure03_plotting.py`.)

**7. schematic.py** - Graphical Model Diagrams

- `draw_graphical_model(ax)`: Draw state space model graphical representation
- `draw_equation_boxes(ax)`: Draw Bayesian filtering equations
- Used by Figure 1 to create schematic overview

**8. load_local_data.py** - Real Data Loading

- `load_neural_recording_from_files(data_path, animal_date_epoch)`: loads
  position info, spike times, track graph, and linear edge data from
  pre-exported pickle files in the supplied data directory. No Spyglass
  database connection required.

**9. paths.py** - Shared Data Identifiers

- `DATA_PATH`: default `<repo>/data`; override via `STATESPACECHECK_DATA_PATH`.
- `ANIMAL_DATE_EPOCH`: default `j1620210710_02_r1`; override via
  `STATESPACECHECK_ANIMAL_DATE_EPOCH`.
- Imported by `scripts/generate_figure04.py` and the four
  `scripts/exploratory/` sanity-check / window-finder scripts so they share one
  source of truth.

### Figure Scripts

Figure scripts (in `scripts/`) are thin command-line adapters. Scientific
orchestration lives in `src/statespacecheck_paper/figureNN_generation.py`, where
each recipe can be imported and tested without executing source text.

**Example structure**:

```python
from statespacecheck_paper.figure01_generation import generate_figure01

def main() -> None:
    """Generate the canonical Figure 1 artifacts."""
    generate_figure01()

if __name__ == "__main__":
    main()
```

### Testing Structure

Tests are organized by module (regenerate exact percentages with `uv run pytest --cov --cov-report=term-missing | tail -1`):

- **test_style.py**: Style utilities (98% coverage)
- **test_simulation.py**: Simulation functions (100% coverage)
- **test_figure03_*.py**: Figure-3 protocol/simulation/summary/plotting + phases + contracts
- **test_plotting.py**: Plotting functions (96% coverage)
- **test_figures.py**: Integration tests for figure scripts
- **test_properties.py**: Property-based tests using Hypothesis

## Development Commands

### Environment Setup

**CRITICAL**: Always use `uv` for package management and work in `.venv` environment.

```bash
# Install UV if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment with Python 3.11
echo "3.11" > .python-version
uv venv

# Activate environment
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install package in editable mode with dev dependencies
uv pip install -e ".[dev]"
```

### Code Quality

```bash
# Format code (fixes issues automatically)
uv run ruff format .

# Check formatting (CI mode, no modifications)
uv run ruff format --check .

# Lint code (shows issues)
uv run ruff check .

# Fix linting issues automatically where possible
uv run ruff check --fix .

# Type checking
uv run mypy src/
```

### Testing

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_figures.py -v

# Run with coverage report
uv run pytest --cov

# Generate HTML coverage report
uv run pytest --cov --cov-report=html
# Open htmlcov/index.html in browser
```

### Jupyter Notebooks

```bash
# Launch Jupyter
uv run jupyter notebook

# Or use JupyterLab
uv run jupyter lab
```

## Key Design Principles

### 1. Reproducibility First

**All analysis must be reproducible**:

- Seed explicitly with the modern Generator API: `rng = np.random.default_rng(seed)`
  (thread `rng`/`SeedSequence` through functions rather than the legacy global
  `np.random.seed`), matching the repo's actual practice
- Reproduce the locked environment with `uv sync --frozen` (pins come from `uv.lock`
  alongside `pyproject.toml`)
- Document data sources and preprocessing steps
- Save intermediate results when computation is expensive

### 2. Publication-Ready Figures

**Follow principles from Tufte, Gelman, and Heer**:

- Use `statespacecheck_paper.style` module for consistent styling
- Default figure size: (8, 6) inches for single column, (16, 6) for double
- Font sizes: 12pt for axis labels, 10pt for tick labels
- Use colorblind-friendly palettes from seaborn
- Export figures as both PNG (for preview) and PDF/SVG (for publication)
- Include figure captions in docstrings

### 3. Clean Separation of Concerns

- **src/statespacecheck_paper/**: Reusable functions, well-tested
- **notebooks/archive/**: Archived exploratory analysis, can be messy (each carries an archived-notebook banner; not part of the reproducible pipeline)
- **scripts/**: Production scripts to generate final figures/results

### 4. Time-Resolved Diagnostics

**When working with temporal data**:

- Arrays should be `(n_time, ...)` with time as first dimension
- Use vectorized operations, avoid Python loops — with the legitimate exception of
  inherently-sequential recursions (e.g. the Bayesian filter's `for t` loop in
  `decoding.decode_with_diagnostics`, where step `t` depends on step `t-1`); those
  are not a defect to be vectorized away
- Handle NaN values properly (mark invalid spatial bins)
- Document expected array shapes in docstrings

### 5. Data Structure Conventions

**Spatial distributions**:

- 1D: `(n_time, n_position_bins)` - Linear track
- 2D: `(n_time, n_x_bins, n_y_bins)` - Open field

**Neural data**:

- Spike counts: `(n_cells, n_time)`
- Place fields: `(n_cells, n_bins)` or `(n_cells, n_x_bins, n_y_bins)`

**State space model outputs**:

- Predictive: `p(x_t | y_{1:t-1})`
- Filtered: `p(x_t | y_{1:t})`
- Smoothed: `p(x_t | y_{1:T})`

## Code Quality Standards

### Where to Add New Functionality

When adding new features, follow these guidelines:

**Adding simulation functions** → `src/statespacecheck_paper/simulation.py`

- Random walks, spike generation, place field models
- Utility functions for simulation (normalize, boundary conditions)
- Functions should be pure (no side effects) and reproducible (use `rng` parameter)

**Adding diagnostic/decoder functions** → `src/statespacecheck_paper/diagnostics.py` or `decoding.py`; **Figure-3 code** → the `figure03_*` family

- Decoder logic, filtering algorithms
- Diagnostic computations (KL divergence, HPD overlap)
- Data transformations and threshold computations
- Use dataclasses for configuration objects

**Adding plotting functions** → `src/statespacecheck_paper/plotting.py`

- Reusable visualization components
- Diagnostic plots, heatmaps, timeseries
- Functions should return Figure objects for flexibility
- Use consistent styling from `style.py`

**Adding figure styling** → `src/statespacecheck_paper/style.py`

- Color palettes, font configurations
- Figure sizing and layout utilities
- Save/export functions
- Keep consistent across all figures

**Creating new figures** → `scripts/generate_figureXX.py`

- Import from shared modules (don't duplicate code!)
- Keep scripts thin (<200 lines of orchestration)
- Save outputs to `manuscript/figures/main/` or `manuscript/figures/supplementary/`
- Add integration test in `tests/test_figures.py`
- Document what the figure demonstrates

**DO NOT**:

- ❌ Add utilities to figure scripts (extract to modules instead)
- ❌ Duplicate code across figure scripts
- ❌ Mix simulation/analysis/plotting in one large function
- ❌ Create figure-specific versions of general utilities

### Docstrings

Use NumPy format with shape specifications:

```python
def compute_diagnostics(
    state_dist: np.ndarray,
    likelihood: np.ndarray,
    coverage: float = 0.95,
) -> dict[str, np.ndarray]:
    """Compute goodness-of-fit diagnostics.

    Parameters
    ----------
    state_dist : np.ndarray, shape (n_time, n_bins)
        State distribution (predictive or smoothed).
    likelihood : np.ndarray, shape (n_time, n_bins)
        Normalized likelihood distribution.
    coverage : float, default 0.95
        Coverage probability for HPD regions.

    Returns
    -------
    diagnostics : dict[str, np.ndarray]
        Dictionary with 'kl_divergence' and 'hpd_overlap' keys.
        Each value has shape (n_time,).

    Examples
    --------
    >>> state_dist = np.random.dirichlet(np.ones(50), size=100)
    >>> likelihood = np.random.dirichlet(np.ones(50), size=100)
    >>> diag = compute_diagnostics(state_dist, likelihood)
    >>> diag['kl_divergence'].shape
    (100,)
    """
```

### Type Hints

**CRITICAL**: Full type hints required, mypy strict mode must pass.

```python
from typing import Literal

import numpy as np
from numpy.typing import NDArray

def analyze_fit(
    posterior: NDArray[np.float64],
    prior: NDArray[np.float64],
    metric: Literal["kl", "overlap"] = "kl",
) -> NDArray[np.float64]:
    """..."""
```

### No Type Ignores

**NEVER use `# type: ignore` comments**. If mypy complains:

1. Fix the actual type issue
2. Add proper type hints
3. Use type narrowing with `isinstance()` checks
4. Refactor code to be type-safe

### Code Style

- **Line length**: 100 characters max
- **Imports**: Grouped and sorted by ruff (stdlib, third-party, local)
- **Naming**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_SNAKE_CASE`
- **Array operations**: Vectorized NumPy, no Python loops over time/space

## Testing Patterns

### Test Structure

```python
"""Tests for plotting utilities."""

import numpy as np
from statespacecheck_paper.plotting import compute_hpd_region


def test_compute_hpd_region() -> None:
    """Test HPD region computation."""
    # Setup
    n_bins = 50
    x = np.linspace(0, 1, n_bins)
    pdf = np.exp(-((x - 0.5) ** 2) / 0.1)  # Gaussian-like
    pdf = pdf / pdf.sum()

    # Execute
    region = compute_hpd_region(x, pdf, coverage=0.95)

    # Assert
    assert region is not None
    assert len(region) == n_bins
    assert region.dtype == bool
```

### Test Data

- Use small datasets for speed
- Set random seeds for reproducibility
- Use `pytest.mark.parametrize` for multiple scenarios

### Coverage

- Aim for >90% coverage on the core modules (analysis, simulation, style,
  plotting); the suite is ~81% overall because the interactive GUI and
  real-data plotting modules are exercised less
- Focus on edge cases: empty arrays, NaN values, zero sums
- Test both 1D and 2D spatial arrays

## Working with Figures

### Figure Generation Pipeline

The repository uses a modular approach where reusable code lives in `src/` and figure scripts orchestrate:

1. **Extract reusable components** to appropriate modules:
   - Simulation logic → `simulation.py`
   - Diagnostics → `diagnostics.py`; decoder → `decoding.py`
   - Plotting functions → `plotting.py`
   - Write tests for each component

2. **Create thin figure script** in `scripts/`:
   - Import from shared modules
   - Set up parameters
   - Call simulation/analysis/plotting functions
   - Save outputs

3. **Add integration test** in `tests/test_figures.py`:
   - Verify imports work
   - Test figure generation with small parameters
   - Verify output files created

### Example Figure Function

```python
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def plot_kl_over_time(
    kl_divergence: NDArray[np.float64],
    time: NDArray[np.float64] | None = None,
    threshold: float = 1.0,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot KL divergence over time.

    Parameters
    ----------
    kl_divergence : np.ndarray, shape (n_time,)
        KL divergence at each time point.
    time : np.ndarray, shape (n_time,), optional
        Time values. If None, uses indices.
    threshold : float, default 1.0
        Threshold for highlighting high divergence.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    if time is None:
        time = np.arange(len(kl_divergence))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, kl_divergence, color='steelblue', linewidth=1.5)
    ax.axhline(threshold, color='red', linestyle='--', alpha=0.5, label='Threshold')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('KL Divergence', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    return fig, ax
```

## Common Workflows

### Running Analysis Pipeline

1. **Ensure dependencies installed**: `uv pip install -e ".[dev]"`
2. **Generate all figures**: `uv run python scripts/generate_all_figures.py`
3. **Generate individual figure**: `uv run python scripts/generate_figure01.py`
4. **Check outputs**: Figures saved to `manuscript/figures/main/` or `manuscript/figures/supplementary/`
5. **Verify**: Review generated PDF and PNG files

### Before Committing

**Always run these checks**:

```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Type check
uv run mypy src/

# Run tests
uv run pytest
```

**All must pass** before committing.

## Dependencies

### Core Analysis

- **statespacecheck**: Main package with diagnostics
- **numpy**: Array operations
- **scipy**: Statistical functions
- **matplotlib**: Figure generation
- **seaborn**: Statistical visualization
- **pandas**: Data manipulation

### Real Data Analysis (optional)

- **non_local_detector**: GitHub dependency for neural decoding (from LorenFrankLab)
- **spyglass-neuro**: Neural data pipeline framework

### Development

- **ruff**: Fast linter and formatter
- **mypy**: Static type checker
- **pytest**: Testing framework

### Installing from GitHub Repositories

Some dependencies may be installed directly from GitHub rather than PyPI. This is useful for:

- Development versions with unreleased features
- Bug fixes not yet published
- Custom forks with project-specific changes

#### Configuration Required

**1. Enable direct references in `pyproject.toml`:**

Hatchling (the build backend) requires explicit permission for GitHub dependencies:

```toml
[tool.hatch.metadata]
allow-direct-references = true
```

**2. Specify the GitHub dependency:**

```toml
dependencies = [
    "package-name @ git+https://github.com/username/repo.git",
]
```

**Optional**: Pin to specific branch, tag, or commit:

```toml
# Specific branch
"package-name @ git+https://github.com/username/repo.git@branch-name"

# Specific tag
"package-name @ git+https://github.com/username/repo.git@v1.2.3"

# Specific commit (most reproducible)
"package-name @ git+https://github.com/username/repo.git@abc123def456"
```

#### Understanding `uv` Dependency Management

**Key Concept**: `uv` uses a **lock file** (`uv.lock`) to ensure reproducible installations, similar to `npm`'s `package-lock.json` or `poetry`'s `poetry.lock`.

**The Three Environments:**

1. **`pyproject.toml`**: Declares dependency *requirements* (e.g., "latest from GitHub main branch")
2. **`uv.lock`**: Pins *exact commits* for reproducibility (e.g., commit `abc123`)
3. **`.venv/`**: The actual installed packages

**Important Behaviors:**

- `uv pip install -e ".[dev]"` → Creates/updates `.venv` but **does not update** `uv.lock`
- `uv sync` → Installs packages from `uv.lock` into `.venv`
- `uv run python` → Uses environment defined by `uv.lock` (may differ from `.venv`!)

#### Updating GitHub Dependencies

When a GitHub dependency is updated upstream, follow these steps:

**Step 1: Update the lock file**

```bash
# Update specific package
uv lock --upgrade-package package-name

# Update all packages
uv lock --upgrade
```

This fetches the latest commit from GitHub and updates `uv.lock`.

**Step 2: Sync the environment**

```bash
uv sync
```

This installs the newly locked version into your `.venv`.

**Step 3: Verify the update**

```bash
uv run python -c "import package_name; print(package_name.__version__)"
```

#### Common Issues and Solutions

**Issue 1: `uv run` shows old version, but `.venv` has new version**

**Cause**: Lock file (`uv.lock`) not updated

**Solution**:

```bash
uv lock --upgrade-package package-name
uv sync
```

**Issue 2: "Direct reference not allowed" error**

**Cause**: Missing `allow-direct-references` in `pyproject.toml`

**Solution**: Add to `pyproject.toml`:

```toml
[tool.hatch.metadata]
allow-direct-references = true
```

**Issue 3: Package installed from cache instead of latest GitHub**

**Cause**: `uv` caches Git repositories

**Solution**: Force fresh install:

```bash
uv pip install --reinstall --no-cache "package @ git+https://github.com/user/repo.git"
```

Then update lock:

```bash
uv lock --upgrade-package package
uv sync
```

#### Best Practices

1. **Pin production dependencies** to specific commits for reproducibility:

   ```toml
   "non_local_detector @ git+https://github.com/LorenFrankLab/non_local_detector.git@abc123"
   ```

2. **Use branches for development** to automatically get updates:

   ```toml
   "package @ git+https://github.com/user/repo.git@develop"
   ```

3. **Always update lock after changing** `pyproject.toml`:

   ```bash
   uv lock
   uv sync
   ```

4. **Commit `uv.lock`** to version control for reproducibility

5. **Document expected features** if using unreleased versions (e.g., in CHANGELOG or commit message)

#### Example Workflow

Adding a new GitHub dependency:

```bash
# 1. Edit pyproject.toml
cat >> pyproject.toml << 'EOF'
dependencies = [
    "my-package @ git+https://github.com/user/my-package.git",
]

[tool.hatch.metadata]
allow-direct-references = true
EOF

# 2. Update lock file
uv lock

# 3. Install to environment
uv sync

# 4. Verify installation
uv run python -c "import my_package; print(my_package.__version__)"

# 5. Commit changes
git add pyproject.toml uv.lock
git commit -m "Add my-package from GitHub"
```

Updating an existing GitHub dependency:

```bash
# 1. Check current version
uv pip show package-name | grep Version

# 2. Update lock to latest
uv lock --upgrade-package package-name

# 3. Sync environment
uv sync

# 4. Verify new version
uv run python -c "import package_name; print(package_name.__version__)"

# 5. Test that everything works
uv run pytest

# 6. Commit updated lock
git add uv.lock
git commit -m "Update package-name to latest GitHub version"
```

## Performance Considerations

### Vectorization

**Good** (vectorized):

```python
kl_div = kl_divergence(state_dist, likelihood)  # Operates on all time points
```

**Bad** (loop):

```python
kl_div = np.array([
    kl_divergence(state_dist[t:t+1], likelihood[t:t+1])
    for t in range(n_time)
])
```

### Memory Management

- For large datasets, process in chunks
- Use `np.memmap` for very large arrays
- Clear variables when done: `del large_array`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Import error for statespacecheck | Install package: `uv pip install statespacecheck` |
| Mypy errors on numpy types | Import from `numpy.typing`: `from numpy.typing import NDArray` |
| Tests fail with "module not found" | Reinstall: `uv pip install -e ".[dev]"` |
| Jupyter kernel not found | Install kernel: `uv run python -m ipykernel install --user --name statespacecheck-paper` |
| Figure doesn't appear | Use `plt.show()` or save: `fig.savefig('output.png')` |

## Quick Reference

```bash
# Start working
source .venv/bin/activate
uv pip install -e ".[dev]"

# Generate figures
uv run python scripts/generate_all_figures.py

# Check quality
uv run ruff format . && uv run ruff check . && uv run mypy src/ && uv run pytest
```

## Resources

- **statespacecheck docs**: Documentation for the main package
- **NumPy style guide**: <https://numpydoc.readthedocs.io/>
- **Scientific Python SPEC 0**: <https://scientific-python.org/specs/spec-0000/>
- **Matplotlib gallery**: <https://matplotlib.org/stable/gallery/>
