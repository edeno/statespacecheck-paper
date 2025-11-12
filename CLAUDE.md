# statespacecheck-paper Development Guide

## Project Overview

This repository contains the source code and supplementary materials for the paper **"Goodness-of-fit diagnostics for state space models in neuroscience"**. It includes analysis scripts, figure generation code, and examples demonstrating the `statespacecheck` package.

**Scientific Context**: State space models are widely used in neuroscience to relate neural activity to latent dynamic brain states. This paper introduces diagnostics (KL divergence and HPD overlap) to assess model goodness-of-fit by examining consistency between posterior distributions and component likelihood distributions.

**Repository Type**: This is a paper/research repository, not a library. The focus is on reproducible analysis, figure generation, and demonstrating the `statespacecheck` package capabilities.

## Architecture

```
statespacecheck-paper/
├── src/statespacecheck_paper/  # Analysis code and utilities
│   ├── __init__.py             # Package initialization
│   ├── load_data.py            # Data loading utilities
│   ├── style.py                # Figure styling (colors, defaults, save)
│   ├── simulation.py           # Simulation utilities
│   ├── analysis.py             # Analysis logic and diagnostics
│   └── plotting.py             # Plotting utilities
├── scripts/                     # Figure generation scripts
│   ├── generate_figure01.py    # Figure 1: Distribution comparisons
│   ├── generate_figure02.py    # Figure 2: Diagnostic demonstrations
│   ├── generate_figure03.py    # Figure 3: (future)
│   └── generate_all_figures.py # Master script to generate all figures
├── figures/                     # Generated figure outputs
│   ├── main/                   # Main text figures (PDF + PNG)
│   └── supplementary/          # Supplementary figures (PDF + PNG)
├── manuscript/                  # LaTeX source files for paper
│   ├── main.tex
│   ├── supplement.tex
│   ├── references.bib
│   └── README.md
├── notebooks/                   # Jupyter notebooks for exploration
├── tests/                       # Comprehensive test suite (75% coverage)
│   ├── test_style.py
│   ├── test_simulation.py
│   ├── test_analysis.py
│   ├── test_plotting.py
│   ├── test_figures.py         # Integration tests
│   └── test_properties.py      # Property-based tests
└── docs/                        # Documentation and examples
```

**Key Modules**:
- **style.py**: Shared styling utilities (WONG palette, figure defaults, save function)
- **simulation.py**: Simulation functions (random walks, spike generation, place fields)
- **analysis.py**: Analysis logic (decoder, diagnostics, thresholds, transformations)
- **plotting.py**: Reusable plotting functions (HPD regions, diagnostic plots)
- **load_data.py**: Data loading utilities for real datasets

**Figure Scripts** (in `scripts/`):
- **generate_figure01.py**: Thin orchestration layer (~220 lines) for Figure 1
- **generate_figure02.py**: Thin orchestration layer (~200 lines) for Figure 2
- **generate_all_figures.py**: Master script to generate all figures

## Repository Structure

### Module Organization

The repository follows a clean separation between **reusable code** (in `src/`) and **figure scripts** (in `figures/`).

#### Core Modules (`src/statespacecheck_paper/`)

**1. style.py** - Figure Styling Utilities
- `WONG`: 8-color colorblind-friendly palette
- `set_figure_defaults(context='paper')`: Set matplotlib defaults
- `save_figure(basename, dpi=450)`: Save figures as PDF and PNG
- `get_figure_size(width_type='single')`: Get standard figure dimensions

**2. simulation.py** - Data Simulation
- `normalize(x, axis=-1, eps=1e-10)`: Safe array normalization
- `reflect_into_interval(x, xmin, xmax)`: Reflecting boundary conditions
- `gaussian_transition_matrix(n_bins, sigma)`: Random walk transition matrix
- `safe_log(x, eps=1e-10)`: Numerically stable logarithm
- `placefield_rates(position_bins, centers, scale)`: Gaussian place fields
- `spike_prob_rank(rates)`: Cumulative probability ranking
- `simulate_walk(n_time, transition_matrix, x0, rng)`: Random walk simulation
- `simulate_spikes_position_tuned(position, placefield_rates, rng)`: Position-tuned Poisson spikes
- `simulate_spikes_flat_rate(n_time, n_cells, rate, rng)`: Constant-rate Poisson spikes

**3. analysis.py** - Bayesian Decoding and Diagnostics
- `DecodeParams`: Dataclass for decoder configuration (timeline, cells, remapping)
- `Thresholds`: Dataclass for diagnostic thresholds
- `Transformed`: Dataclass for transformed diagnostics
- `likelihood_grid_for_counts(counts, placefield_rates)`: Poisson likelihood computation
- `apply_remap_for_likelihoods(likelihoods, params)`: Cell identity remapping
- `decode_and_diagnostics(spikes, params)`: Main decoder with KL/HPD diagnostics
- `compute_thresholds(baseline_period, quantiles)`: Compute baseline thresholds
- `transform_metrics(diagnostics, thresholds, eps)`: Transform for visualization

**4. plotting.py** - Reusable Plotting Functions
- `compute_hpd_region(distribution, coverage)`: Highest posterior density region mask
- `plot_original(diagnostics, params, thresholds)`: Original diagnostic metrics plot
- `plot_transformed(transformed, params, thresholds)`: Transformed metrics plot
- `plot_misfit_examples(diagnostics, x_true, params)`: Example misfit periods
- `plot_combined_diagnostics(diagnostics, x_true, spikes, params)`: Comprehensive visualization

**5. load_data.py** - Real Data Loading
- Functions to load real neural recording datasets (not covered in detail here)

### Figure Scripts

Figure scripts (in `scripts/`) are thin orchestration layers that:
1. Import from shared modules
2. Set up simulation/analysis parameters
3. Run simulations/analyses
4. Generate and save figures

**Example structure**:
```python
from statespacecheck_paper.style import WONG, set_figure_defaults, save_figure
from statespacecheck_paper.simulation import simulate_walk, simulate_spikes_position_tuned
from statespacecheck_paper.analysis import decode_and_diagnostics, DecodeParams

def create_figure():
    """Generate Figure X showing..."""
    set_figure_defaults()

    # Setup parameters
    params = DecodeParams(...)

    # Run simulation
    x_true = simulate_walk(...)
    spikes = simulate_spikes_position_tuned(...)

    # Run analysis
    results = decode_and_diagnostics(spikes, params)

    # Create plots
    fig, axes = plot_combined_diagnostics(results, x_true, spikes, params)

    # Save to figures/main/
    save_figure("figures/main/figureX")

if __name__ == "__main__":
    create_figure()
```

### Testing Structure

Tests are organized by module with 97.2% coverage:

- **test_style.py**: Style utilities (100% coverage)
- **test_simulation.py**: Simulation functions (100% coverage)
- **test_analysis.py**: Analysis functions (100% coverage)
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
- Set random seeds explicitly: `np.random.seed(42)`
- Pin dependency versions in pyproject.toml
- Document data sources and preprocessing steps
- Save intermediate results when computation is expensive

### 2. Publication-Ready Figures

**Follow principles from Tufte, Gelman, and Heer**:
- Use `statespacecheck_paper.figures` module for consistent styling
- Default figure size: (8, 6) inches for single column, (16, 6) for double
- Font sizes: 12pt for axis labels, 10pt for tick labels
- Use colorblind-friendly palettes from seaborn
- Export figures as both PNG (for preview) and PDF/SVG (for publication)
- Include figure captions in docstrings

### 3. Clean Separation of Concerns

- **src/statespacecheck_paper/**: Reusable functions, well-tested
- **notebooks/**: Exploratory analysis, can be messy
- **scripts/**: Production scripts to generate final figures/results

### 4. Time-Resolved Diagnostics

**When working with temporal data**:
- Arrays should be `(n_time, ...)` with time as first dimension
- Use vectorized operations, avoid Python loops
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

**Adding analysis functions** → `src/statespacecheck_paper/analysis.py`
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
- Save outputs to `figures/main/` or `figures/supplementary/`
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
"""Tests for figure generation."""

import numpy as np
from statespacecheck_paper.figures import create_diagnostic_figure


def test_create_diagnostic_figure() -> None:
    """Test diagnostic figure generation."""
    # Setup
    n_time, n_bins = 100, 50
    state_dist = np.random.dirichlet(np.ones(n_bins), size=n_time)
    likelihood = np.random.dirichlet(np.ones(n_bins), size=n_time)

    # Execute
    fig, axes = create_diagnostic_figure(state_dist, likelihood)

    # Assert
    assert fig is not None
    assert len(axes) == 2  # Two subplots expected
```

### Test Data

- Use small datasets for speed
- Set random seeds for reproducibility
- Use `pytest.mark.parametrize` for multiple scenarios

### Coverage

- Aim for >90% code coverage
- Focus on edge cases: empty arrays, NaN values, zero sums
- Test both 1D and 2D spatial arrays

## Working with Figures

### Figure Generation Pipeline

The repository uses a modular approach where reusable code lives in `src/` and figure scripts orchestrate:

1. **Extract reusable components** to appropriate modules:
   - Simulation logic → `simulation.py`
   - Analysis logic → `analysis.py`
   - Plotting functions → `plotting.py`
   - Write tests for each component

2. **Create thin figure script** in `figures/`:
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
4. **Check outputs**: Figures saved to `figures/main/` or `figures/supplementary/`
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

### Development
- **ruff**: Fast linter and formatter
- **mypy**: Static type checker
- **pytest**: Testing framework

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
- **NumPy style guide**: https://numpydoc.readthedocs.io/
- **Scientific Python SPEC 0**: https://scientific-python.org/specs/spec-0000/
- **Matplotlib gallery**: https://matplotlib.org/stable/gallery/
