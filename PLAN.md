# Implementation Plan: Per-Cell Metrics (MATLAB-Style)

## Overview

Replace aggregated metrics with per-cell metrics matching MATLAB implementation:
- `hpd_overlap[t, j]` = overlap between prior HPD and cell j's likelihood HPD
- `kl_divergence[t, j]` = KL(prior || likelihood_cell_j)
- `spike_prob[t, j]` = cumulative prob mass of cells with contribution ≤ cell j

All metrics set to NaN when `spikes[t, j] == 0`.

## Task 1: Update `decode_and_diagnostics` in `analysis.py`

**File:** `src/statespacecheck_paper/analysis.py`

**Changes:**

1. Change return dict shape from `(n_time,)` to `(n_time, n_cells)` for all three metrics

2. Replace the diagnostic computation block (lines 576-600) with per-cell loop:

```python
# Inside the t loop, after likelihood computation:

# Compute per-cell diagnostics
for j in range(n_cells):
    # Skip if no spike from this cell
    if spikes[t, j] == 0:
        # hpd_overlap[t, j], kl_divergence[t, j], spike_prob[t, j] remain NaN
        continue

    # Get cell j's likelihood and normalize
    L_j = likelihood[:, j]  # (n_bins,)
    L_j_norm = L_j / np.maximum(L_j.sum(), 1e-12)  # Normalize

    # Reshape for statespacecheck functions: (1, n_bins)
    prior_t = prior[np.newaxis, :]
    L_j_t = L_j_norm[np.newaxis, :]

    # HPD overlap: prior vs this cell's likelihood
    hpd_overlap[t, j] = ssc.hpd_overlap(prior_t, L_j_t, coverage=0.95)[0]

    # KL divergence: prior vs this cell's likelihood
    kl_divergence[t, j] = ssc.kl_divergence(prior_t, L_j_t)[0]

# spike_prob computed for all cells (even those without spikes for ranking)
spike_prob[t] = spike_prob_rank(prior, cell_fraction_per_bin)
# Then mask cells without spikes
spike_prob[t, spikes[t] == 0] = np.nan
```

3. Remove `conditional_pvalue` computation entirely (lines 612-614)

4. Update return dict:
```python
return {
    "posterior": posterior,
    "predictive": predictive_posterior,
    "likelihood": combined_likelihood_all,
    "hpd_overlap": hpd_overlap,      # Now (n_time, n_cells)
    "kl_divergence": kl_divergence,  # Now (n_time, n_cells)
    "spike_prob": spike_prob,        # Now (n_time, n_cells), replaces conditional_pvalue
}
```

5. Update preallocations at line 527-531:
```python
hpd_overlap = np.full((n_time, n_cells), np.nan)
kl_divergence = np.full((n_time, n_cells), np.nan)
spike_prob = np.full((n_time, n_cells), np.nan)
```

**Verification:**
- Run: `uv run pytest tests/test_analysis.py -v -k decode`
- Check output shapes are `(n_time, n_cells)`

---

## Task 2: Update `Thresholds` dataclass and `compute_thresholds` in `analysis.py`

**File:** `src/statespacecheck_paper/analysis.py`

**Changes:**

1. Update `Thresholds` dataclass (lines 631-661):
```python
@dataclass
class Thresholds:
    """Threshold values for diagnostic metrics.

    Thresholds are computed from baseline period across ALL cells (flattened).
    """
    hpd_overlap: float      # 1st percentile across all cells
    kl_divergence: float    # 99th percentile across all cells
    spike_prob: float       # Fixed at 0.05 per MATLAB
```

2. Update `compute_thresholds` function (lines 663-711):
```python
def compute_thresholds(
    metrics: dict[str, NDArray[np.floating]], baseline_end: int = 60_000
) -> Thresholds:
    """Compute threshold values from baseline period.

    Thresholds computed across ALL cells (flattened), matching MATLAB.
    """
    # Flatten (n_time, n_cells) to 1D for quantile computation
    hpd_baseline = metrics["hpd_overlap"][:baseline_end].ravel()
    hpd_overlap_threshold = np.nanquantile(hpd_baseline, 0.01)

    kl_baseline = metrics["kl_divergence"][:baseline_end].ravel()
    kl_divergence_threshold = np.nanquantile(kl_baseline, 0.99)

    # spike_prob threshold is fixed at 0.05 per MATLAB
    spike_prob_threshold = 0.05

    return Thresholds(
        hpd_overlap=hpd_overlap_threshold,
        kl_divergence=kl_divergence_threshold,
        spike_prob=spike_prob_threshold,
    )
```

**Verification:**
- Run: `uv run pytest tests/test_analysis.py -v -k threshold`

---

## Task 3: Update `Transformed` dataclass and `transform_metrics` in `analysis.py`

**File:** `src/statespacecheck_paper/analysis.py`

**Changes:**

1. Update `Transformed` dataclass (lines 714-756):
```python
@dataclass
class Transformed:
    """Transformed diagnostic metrics and thresholds.

    All metrics have shape (n_time, n_cells).
    """
    hpd_overlap: NDArray[np.floating]      # -log(HPDO + eps1)
    kl_divergence: NDArray[np.floating]    # sqrt(KL)
    spike_prob: NDArray[np.floating]       # -log(spikeProb + eps2)
    hpd_overlap_threshold: float
    kl_divergence_threshold: float
    spike_prob_threshold: float
```

2. Update `transform_metrics` function (lines 759-826):
```python
def transform_metrics(
    metrics: dict[str, NDArray[np.floating]],
    thresholds: Thresholds,
    eps1: float = 1e-2,
    eps2: float = 1e-10,
) -> Transformed:
    """Apply transformations to metrics for better visualization.

    Transformations (matching MATLAB):
    - HPD overlap: -log(HPDO + eps1)
    - KL divergence: sqrt(KL)
    - spike_prob: -log(spikeProb + eps2)
    """
    hpd_overlap_transformed = -safe_log(metrics["hpd_overlap"] + eps1)
    kl_divergence_transformed = np.sqrt(metrics["kl_divergence"])
    spike_prob_transformed = -safe_log(metrics["spike_prob"] + eps2)

    return Transformed(
        hpd_overlap=hpd_overlap_transformed,
        kl_divergence=kl_divergence_transformed,
        spike_prob=spike_prob_transformed,
        hpd_overlap_threshold=-np.log(thresholds.hpd_overlap + eps1),
        kl_divergence_threshold=np.sqrt(thresholds.kl_divergence),
        spike_prob_threshold=-np.log(thresholds.spike_prob + eps2),
    )
```

**Verification:**
- Run: `uv run pytest tests/test_analysis.py -v -k transform`

---

## Task 4: Update plotting functions in `plotting.py`

**File:** `src/statespacecheck_paper/plotting.py`

**Changes:**

### 4a. Update `plot_original` function (lines 342-513)

Change from plotting 1D arrays to plotting 2D arrays as scatter points (all cells overlaid):

```python
# For HPD overlap (line 450-461):
# Old: axes[1].plot(metrics["hpd_overlap"], ".", ...)
# New: Plot all cells as scatter points
n_time, n_cells = metrics["hpd_overlap"].shape
time_indices = np.tile(np.arange(n_time)[:, np.newaxis], (1, n_cells))
axes[1].scatter(
    time_indices.ravel(),
    metrics["hpd_overlap"].ravel(),
    s=1.5,
    alpha=0.6,
    c=COLORS["hpd_overlap"],
    rasterized=True,
)

# Similar changes for KL divergence (lines 463-474)
# Similar changes for spike_prob (lines 476-495) - rename from conditional_pvalue
```

Update ylabel for spike_prob panel:
```python
axes[3].set_ylabel("Spike Prob", fontsize=10, labelpad=8)
```

### 4b. Update `plot_transformed` function (lines 516-659)

Same pattern - use scatter for 2D arrays:

```python
# For each metric, use scatter instead of plot
time_indices = np.tile(np.arange(n_time)[:, np.newaxis], (1, n_cells))
axes[1].scatter(
    time_indices.ravel(),
    transformed.hpd_overlap.ravel(),
    s=0.5,
    alpha=0.3,
    c=COLORS["hpd_overlap"],
    rasterized=True,
)
```

Update ylabel:
```python
axes[3].set_ylabel("-log(Spike Prob)", fontsize=9, labelpad=8)
```

### 4c. Update `plot_misfit_examples` function (lines 662-899)

Update to use `spike_prob` instead of `conditional_pvalue`:

```python
# Line 881: Change from conditional_pvalue to spike_prob
# Get diagnostic values - now per-cell, pick cell with worst value
hpdo_vals = metrics["hpd_overlap"][example_time]  # (n_cells,)
kl_vals = metrics["kl_divergence"][example_time]   # (n_cells,)
spike_prob_vals = metrics["spike_prob"][example_time]  # (n_cells,)

# Use nanmean or worst value for display
hpdo_val = np.nanmean(hpdo_vals)
kl_val = np.nanmean(kl_vals)
spike_prob_val = np.nanmean(spike_prob_vals)

# Update title text (line 884-887)
title_text = f"{phase_name}\nHPD: {hpdo_val:.2g}  KL: {kl_val:.2g}  SP: {spike_prob_val:.2g}"
```

### 4d. Update `plot_combined_diagnostics` function (lines 902-1413)

1. Update time-series panels to use scatter for 2D metrics:

```python
# HPD overlap panel (lines 1038-1052)
n_time, n_cells = metrics["hpd_overlap"].shape
time_indices = np.tile(np.arange(n_time)[:, np.newaxis], (1, n_cells))
ax_hpdo.scatter(
    time_indices.ravel(),
    metrics["hpd_overlap"].ravel(),
    s=0.8,
    alpha=0.6,
    c=COLORS["hpd_overlap"],
    rasterized=True,
)

# Similar for KL divergence (lines 1068-1082)
# Similar for spike_prob (lines 1098-1142)
```

2. Update spike_prob panel labels:
```python
ax_spike.set_ylabel("Spike Prob", fontsize=9, labelpad=7)
```

3. Update example panel metrics display (lines 1338-1345):
```python
# Use mean across cells for display
hpdo_val = np.nanmean(metrics["hpd_overlap"][example_time])
kl_val = np.nanmean(metrics["kl_divergence"][example_time])
spike_prob_val = np.nanmean(metrics["spike_prob"][example_time])

metrics_text = f"HPD: {hpdo_val:.2f}\nKL: {kl_val:.1f}\nSP: {spike_prob_val:.2g}"
```

**Verification:**
- Run: `uv run pytest tests/test_plotting.py -v`
- Run: `uv run python scripts/generate_figure03.py` (visual check)

---

## Task 5: Update `generate_figure03.py`

**File:** `scripts/generate_figure03.py`

**Changes:**

Minimal changes needed since most logic is in analysis.py and plotting.py.

1. If any direct references to `conditional_pvalue` exist, change to `spike_prob`

**Verification:**
- Run: `uv run python scripts/generate_figure03.py`
- Visual inspection of output figure

---

## Task 6: Update tests

**Files:** `tests/test_analysis.py`, `tests/test_plotting.py`, `tests/test_figures.py`

**Changes:**

1. Update test assertions for new shapes:
```python
# Old: assert metrics["hpd_overlap"].shape == (n_time,)
# New: assert metrics["hpd_overlap"].shape == (n_time, n_cells)
```

2. Update test fixtures to use `spike_prob` instead of `conditional_pvalue`

3. Update threshold tests to verify flattened computation

**Verification:**
- Run: `uv run pytest tests/ -v`
- Run: `uv run pytest --cov` for coverage check

---

## Task 7: Run full test suite and generate figure

**Commands:**
```bash
uv run ruff format .
uv run ruff check --fix .
uv run mypy src/
uv run pytest tests/ -v
uv run python scripts/generate_figure03.py
```

**Verification:**
- All tests pass
- No type errors
- Figure 3 generated successfully with per-cell metrics displayed

---

## Summary of Key Changes

| Component | Before | After |
|-----------|--------|-------|
| `hpd_overlap` shape | `(n_time,)` | `(n_time, n_cells)` |
| `kl_divergence` shape | `(n_time,)` | `(n_time, n_cells)` |
| Metric name | `conditional_pvalue` | `spike_prob` |
| `spike_prob` shape | N/A | `(n_time, n_cells)` |
| Comparison | Prior vs combined likelihood | Prior vs each cell's likelihood |
| Threshold computation | Per-timestep | Flattened across all cells |
| NaN masking | When no spikes at timestep | When cell j has no spike at timestep t |
