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
│   ├── figures.py              # Figure generation code
│   ├── simulations.py          # Simulation utilities
│   └── analysis.py             # Analysis utilities
├── notebooks/                   # Jupyter notebooks for exploration
├── scripts/                     # Scripts to generate figures/results
├── tests/                       # Tests for analysis code
└── docs/                        # Documentation and examples
```

**Key Modules**:
- **figures.py**: Functions to generate publication-ready figures using matplotlib
- **simulations.py**: Code to simulate state space models and generate synthetic data
- **analysis.py**: Analysis pipelines and data processing utilities

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

1. **Create function in `figures.py`**: Reusable, well-tested
2. **Develop in notebook**: Iterate quickly, visualize results
3. **Create script in `scripts/`**: Production version to generate final figure
4. **Document in `docs/`**: Add example with explanation

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

### Adding a New Figure

1. **Explore in notebook**:
   ```bash
   uv run jupyter notebook
   # Create notebooks/figure1_exploration.ipynb
   ```

2. **Extract to module**:
   ```python
   # Add to src/statespacecheck_paper/figures.py
   def create_figure1(...) -> tuple[plt.Figure, ...]:
       """..."""
   ```

3. **Write tests**:
   ```python
   # Add to tests/test_figures.py
   def test_create_figure1() -> None:
       """..."""
   ```

4. **Create production script**:
   ```bash
   # Create scripts/generate_figure1.py
   uv run python scripts/generate_figure1.py
   ```

### Running Analysis Pipeline

1. **Ensure dependencies installed**: `uv pip install -e ".[dev]"`
2. **Run scripts**: `uv run python scripts/run_analysis.py`
3. **Check outputs**: Results saved to `results/` or `figures/`
4. **Verify**: Review generated figures and data files

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

# Run analysis
uv run python scripts/generate_all_figures.py

# Check quality
uv run ruff format . && uv run ruff check . && uv run mypy src/ && uv run pytest
```

## Resources

- **statespacecheck docs**: Documentation for the main package
- **NumPy style guide**: https://numpydoc.readthedocs.io/
- **Scientific Python SPEC 0**: https://scientific-python.org/specs/spec-0000/
- **Matplotlib gallery**: https://matplotlib.org/stable/gallery/
