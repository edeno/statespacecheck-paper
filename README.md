# statespacecheck-paper

**Local goodness-of-fit measures for neural decoding**

This repository contains the source code and supplementary materials for the paper demonstrating `statespacecheck`, a package that provides tools to assess how well Bayesian state space models fit neural data by examining the consistency between posterior distributions and their component likelihood distributions. These diagnostics help identify issues with prior specification and model assumptions, enabling iterative model refinement.

## Repository Structure

This is a **paper/research repository** (not a library). The code is organized into:

- **`src/statespacecheck_paper/`**: Reusable modules (styling, simulation, analysis, plotting)
- **`scripts/`**: Figure generation scripts (thin orchestration layers)
- **`manuscript/figures/`**: Generated figure outputs (PDF and PNG)
  - `manuscript/figures/main/`: Main text figures
  - `manuscript/figures/supplementary/`: Supplementary figures
- **`tests/`**: Comprehensive test suite (326 tests)
- **`notebooks/`**: Jupyter notebooks for exploration

**For developers**: See [CLAUDE.md](CLAUDE.md) for detailed development guide including module organization, coding standards, and where to add new functionality.

## Overview

State space models are powerful tools for relating neural activity to latent dynamic brain states (e.g., memory, attention, spatial navigation). The core assumption is that complex, high-dimensional neural activity can be related to low-dimensional latent states through:

1. **State transition model**: How latent states evolve over time
2. **Observation model**: How neural activity relates to the current latent state

The posterior distribution combines information from both models, weighing current data (normalized likelihood) against accumulated history (prediction distribution). When these distributions agree, the model's prior expectations and data-driven evidence are consistent. When they diverge, the mismatch reveals where and when the model fails to capture the structure of the data.

## Features

- **KL Divergence**: Measure information divergence between posterior and likelihood distributions at each time point
- **HPD Overlap**: Compute spatial overlap between highest posterior density regions
- **Vectorized Operations**: Efficient NumPy-based implementation with no Python loops
- **Flexible Dimensionality**: Supports both 1D `(n_time, n_position_bins)` and 2D `(n_time, n_x_bins, n_y_bins)` spatial arrays
- **Robust Edge Case Handling**: Proper treatment of NaN values, zero sums, and empty distributions

## Terminology

This package uses specific terminology to match standard state space model conventions:

### State Distributions (`state_dist` parameter)

- **One-step-ahead predictive distribution**: p(x_t | y_{1:t-1}) - The distribution over current state given all past observations
- **Smoothed distribution**: p(x_t | y_{1:T}) - The distribution over state at time t given all observations (past and future)
- **Filtered distribution**: p(x_t | y_{1:t}) - The posterior distribution at time t (filtered estimate)

For goodness-of-fit diagnostics, you typically use the **one-step predictive** or **smoothed** distribution as `state_dist`. These represent your model's predictions before (predictive) or after (smoother) incorporating all available data.

### Likelihood (`likelihood` parameter)

- **Normalized likelihood**: p(y_t | x_t) / Σ_x p(y_t | x_t) - The likelihood normalized across spatial positions
- This is mathematically equivalent to the posterior p(x_t | y_t) with a uniform prior
- Represents what your data alone says about the state, without temporal smoothing

### Important Note: Discrete Distributions

All functions expect **discrete probability distributions** represented as histograms over spatial bins. For continuous distributions (e.g., Gaussian), discretize them first:

- Distributions are **automatically normalized** over valid (non-NaN) bins
- Each bin represents the probability mass in that spatial region
- **NaN values** can be used to mark invalid/inaccessible spatial bins (e.g., walls in a maze)
- Finer binning provides better approximation but increases computation

### Interpretation

- **Consistency**: When state distribution and likelihood agree (low KL divergence, high overlap), your model's predictions align with the data
- **Inconsistency**: When they diverge, it indicates:
  - Prior/transition model may be too rigid or misspecified
  - Observation model may not capture the true relationship between states and observations
  - Model capacity may be insufficient

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

### Optional extras

```bash
# Interactive decoder viewer (pyqtgraph + PySide6 desktop app, plus
# zarr / pyarrow for the on-disk cache it consumes).
uv pip install -e ".[interactive]"

# Development tools (ruff, mypy, pytest, hypothesis, jupyter).
uv pip install -e ".[dev,interactive]"
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

## Quick Start

### Basic Example

```python
import numpy as np
from statespacecheck import (
    kl_divergence,
    hpd_overlap,
    highest_density_region,
)

# Example: 1D spatial arrays (time x position)
n_time, n_bins = 100, 50
state_dist = np.random.dirichlet(np.ones(n_bins), size=n_time)  # predictive or smoother
likelihood = np.random.dirichlet(np.ones(n_bins), size=n_time)

# Compute KL divergence at each time point
kl_div = kl_divergence(state_dist, likelihood)
# Returns: (n_time,) array of divergence values

# Compute HPD region overlap
overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)
# Returns: (n_time,) array of overlap proportions (0 = no overlap, 1 = complete)

# Get highest density region mask
hd_mask = highest_density_region(state_dist, coverage=0.95)
# Returns: (n_time, n_bins) boolean mask
```

### Neuroscience Example

```python
import numpy as np
from scipy.stats import norm
from statespacecheck import kl_divergence, hpd_overlap

# Assume you have state space model output for spatial navigation task
# with position bins representing locations in a linear track

# Position bins (e.g., 50 cm track discretized into 100 bins)
position_bins = np.linspace(0, 50, 100)  # cm
n_time = 1000  # Number of time steps

# Example: One-step-ahead predictive distribution from Kalman filter
# predicted_position: (n_time,) array of predicted positions in cm
# predicted_std: (n_time,) array of prediction uncertainty
predicted_position = 25 + 10 * np.sin(np.linspace(0, 4*np.pi, n_time))
predicted_std = np.ones(n_time) * 2.0

# Convert to spatial probability distribution over position bins
# Note: Distributions are automatically normalized, no need to normalize manually
state_dist = np.array([
    norm.pdf(position_bins, loc=pred_pos, scale=pred_std)
    for pred_pos, pred_std in zip(predicted_position, predicted_std)
])

# Example: Likelihood from place cell firing (observation model)
# spike_counts: (n_cells, n_time) array of spike counts
# place_fields: (n_cells, n_bins) array of firing rate maps
# For this example, we'll simulate the likelihood
# Note: Automatically normalized, no manual normalization needed
likelihood = np.array([
    norm.pdf(position_bins, loc=pred_pos + np.random.randn(), scale=3.0)
    for pred_pos in predicted_position
])

# Assess goodness-of-fit
divergence = kl_divergence(state_dist, likelihood)
overlap = hpd_overlap(state_dist, likelihood, coverage=0.95)

# Interpret results
print(f"Mean KL divergence: {np.mean(divergence):.3f}")
print(f"Mean HPD overlap: {np.mean(overlap):.3f}")

# Identify time points with poor fit
high_divergence = divergence > 1.0
low_overlap = overlap < 0.3
print(f"Time points with high divergence: {np.sum(high_divergence)}/{n_time}")
print(f"Time points with low overlap: {np.sum(low_overlap)}/{n_time}")
```

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
- **Figure-3 simulation cache** — the simulated demonstration with
  baseline / remap / history-dependent-firing / drift /
  wide-dynamics-noise phases.

### Build a cache

```bash
# Real data (figure 4): builds figure04_continuous.zarr +
# figure04_contfrag.zarr and shared sidecars from the decoder
# intermediates.
uv run python -m statespacecheck_paper.interactive.cache build \
    --data-dir data \
    --intermediates-dir data/intermediates \
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

## API Reference

### `kl_divergence(state_dist, likelihood)`

Compute Kullback-Leibler divergence between state distribution and likelihood.

**Parameters:**

- `state_dist` (np.ndarray): State distributions (one-step predictive or smoother). Non-negative values, automatically normalized. NaN marks invalid bins. Shape `(n_time, ...)` where `...` represents arbitrary spatial dimensions
- `likelihood` (np.ndarray): Likelihood distributions. Non-negative values, automatically normalized. NaN marks invalid bins. Must have same shape as state_dist

**Returns:**

- `kl_divergence` (np.ndarray): KL divergence at each time point. Shape `(n_time,)`

**Interpretation:**

- **Low divergence (< 0.1)**: State distribution and likelihood agree well, indicating consistency between prior and data
- **Moderate divergence (0.1 - 1.0)**: Some disagreement, worth investigating
- **High divergence (> 1.0)**: Substantial mismatch, suggests issues with prior specification or observation model

### `hpd_overlap(state_dist, likelihood, coverage=0.95)`

Compute overlap between highest posterior density regions.

**Parameters:**

- `state_dist` (np.ndarray): State distributions (one-step predictive or smoother). Non-negative values, automatically normalized. NaN marks invalid bins. Shape `(n_time, ...)` where `...` represents arbitrary spatial dimensions
- `likelihood` (np.ndarray): Likelihood distributions. Non-negative values, automatically normalized. NaN marks invalid bins. Must have same shape as state_dist
- `coverage` (float): Coverage probability for HPD regions (default: 0.95)

**Returns:**

- `overlap` (np.ndarray): Overlap proportion at each time point. Shape `(n_time,)`. Values range from 0 (no overlap) to 1 (complete overlap)

**Interpretation:**

- **High overlap (> 0.7)**: State distribution and likelihood concentrate probability mass in similar regions
- **Moderate overlap (0.3 - 0.7)**: Partial agreement, may indicate transition periods or model uncertainty
- **Low overlap (< 0.3)**: Distributions are spatially inconsistent, suggests model issues

### `highest_density_region(distribution, coverage=0.95)`

Compute boolean mask indicating highest density region membership.

**Parameters:**

- `distribution` (np.ndarray): Probability distributions. Shape `(n_time, ...)` where `...` represents arbitrary spatial dimensions
- `coverage` (float): Desired coverage probability (default: 0.95)

**Returns:**

- `isin_hd` (np.ndarray): Boolean mask. Same shape as input

**Notes:**

- Highest density regions can be multimodal (non-contiguous)
- Regions are defined by selecting positions with highest density until cumulative mass reaches coverage
- NaN values are treated as zero mass

## Development

This repository follows a modular architecture where reusable code lives in `src/statespacecheck_paper/` and figure scripts orchestrate. See [CLAUDE.md](CLAUDE.md) for comprehensive development guide.

### Quick Setup

```bash
# Install UV package manager if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests with coverage (326 tests)
uv run pytest

# Run specific module tests
uv run pytest tests/test_simulation.py -v

# Generate HTML coverage report
uv run pytest --cov --cov-report=html
open htmlcov/index.html
```

### Generating Figures

```bash
# Generate all figures
uv run python scripts/generate_all_figures.py

# Or generate individual figures
uv run python scripts/generate_figure01.py
uv run python scripts/generate_figure02.py

# Outputs saved to manuscript/figures/main/ directory as PDF and PNG (450 DPI)
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
- **`analysis.py`**: Analysis logic (decoder, diagnostics, thresholds)
- **`plotting.py`**: Reusable plotting functions (HPD regions, diagnostic plots)
- **`load_local_data.py`**: Real data loading utilities

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

This package implements goodness-of-fit diagnostics for state space models used in neuroscience. The methods are based on the principle that a well-specified model should have consistent posterior and likelihood distributions. Large divergences or low overlap indicate:

1. **Prior issues**: State transition model too rigid or misspecified
2. **Observation model issues**: Tuning curves or noise assumptions incorrect
3. **Model capacity**: Latent state dimensionality insufficient

These diagnostics complement but are distinct from:

- **Cross-validation**: Measures predictive generalization to new data
- **Permutation tests**: Assess whether model captures structure vs. random patterns

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

MIT License - see LICENSE file for details.
