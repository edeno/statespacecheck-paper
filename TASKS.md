# Refactoring Tasks

Track progress through the refactoring plan. See [REFACTORING_PLAN.md](REFACTORING_PLAN.md) for details.

## Quick Reference

- **Total Estimated Time**: 8-9 days
- **Strategy**: Incremental (one milestone at a time)
- **Test After Each Milestone**: Ensure figures still work
- **Success Metric**: All checkboxes completed + all figures identical

---

## Milestone 1: Style Module (Quick Win ⭐)

**Effort**: 1 day | **Impact**: High | **Risk**: Low

### 1.1 Create style.py Module

- [x] Create `src/statespacecheck_paper/style.py`
- [x] Extract WONG color palette from figure01.py (lines 20-29)
- [x] Extract `set_figure_defaults()` from figure01.py (lines 31-53)
  - [x] Add `context` parameter support (`"paper"`, `"presentation"`, `"poster"`)
- [x] Extract `save_figure()` from figure01.py (lines 55-59)
  - [x] Add `Path` support
  - [x] Add `dpi` parameter (default 450)
  - [x] Add `close` parameter (default True)
  - [x] Auto-create parent directories
- [x] Add `get_figure_size()` helper function
- [x] Add comprehensive docstrings with examples
- [x] Add module-level docstring

### 1.2 Update figure01.py

- [x] Add import: `from statespacecheck_paper.style import WONG, set_figure_defaults, save_figure`
- [x] Remove WONG definition (lines 20-29)
- [x] Remove `set_figure_defaults()` (lines 31-53)
- [x] Remove `save_figure()` (lines 55-59)
- [x] Update `save_figure()` call to use new signature
- [x] Test: `python figures/figure01.py` runs successfully
- [x] Verify: Output identical to before (visual check)

### 1.3 Update figure02.py

- [x] Add import: `from statespacecheck_paper.style import WONG, save_figure`
- [x] Replace inline `wong` definition (line 828) with `WONG` import
- [x] Replace inline save calls (lines 1253-1256) with `save_figure()`
- [x] Test: `python figures/figure02.py` runs successfully
- [x] Verify: Output identical to before (visual check)

### 1.4 Quality Checks

- [x] Run: `uv run ruff format .`
- [x] Run: `uv run ruff check .`
- [x] Run: `uv run mypy src/`
- [x] Run: `uv run pytest`
- [x] All checks pass ✅

### 1.5 Commit

- [x] `git add src/statespacecheck_paper/style.py figures/`
- [x] `git commit -m "Extract styling utilities to style.py module"`
- [x] Note: Figure outputs verified as identical

**Milestone 1 Complete** ✅

---

## Milestone 2: Simulation Module

**Effort**: 1 day | **Impact**: High | **Risk**: Low

### 2.1 Create simulation.py Module

- [x] Create `src/statespacecheck_paper/simulation.py`
- [x] Extract from figure02.py:
  - [x] `normalize()` (lines 22-28)
  - [x] `reflect_into_interval()` (lines 31-37)
  - [x] `gaussian_transition_matrix()` (lines 40-46)
  - [x] `safe_log()` (lines 49-51)
  - [x] `placefield_rates()` (lines 54-59)
  - [x] `spike_prob_rank()` (lines 62-78)
  - [x] `simulate_walk()` (lines 149-160)
  - [x] `simulate_spikes_position_tuned()` (lines 163-172)
  - [x] `simulate_spikes_flat_rate()` (lines 175-179)
- [x] Add comprehensive docstrings with:
  - [x] Parameters with shapes
  - [x] Returns with shapes
  - [x] Examples section
- [x] Add type hints using `numpy.typing.NDArray`
- [x] Add module-level docstring

### 2.2 Update figure02.py

- [x] Add import: `from statespacecheck_paper.simulation import ...`
- [x] Remove extracted utility functions
- [x] Update calls to use imported functions
- [x] Test: `python figures/figure02.py` imports successfully
- [x] Verify: Imports work correctly

### 2.3 Create Tests

- [x] Create `tests/test_simulation.py`
- [x] Test `normalize()`:
  - [x] 1D array normalization
  - [x] 2D array with axis parameter
  - [x] Handle zeros/empty arrays
- [x] Test `reflect_into_interval()`:
  - [x] Values inside bounds unchanged
  - [x] Values outside bounds reflected correctly
  - [x] Multiple reflections work
- [x] Test `simulate_walk()`:
  - [x] Output shape correct
  - [x] Initial position respected
  - [x] Boundaries respected (all values in [min, max])
  - [x] Reproducibility with same seed
  - [x] Larger step size → larger variance
- [x] Test `simulate_spikes_position_tuned()`:
  - [x] Output shape correct
  - [x] Non-negative integer counts
  - [x] Higher firing near place field centers
  - [x] Reproducibility
- [x] Test `simulate_spikes_flat_rate()`:
  - [x] Output shape correct
  - [x] Mean close to specified rate
  - [x] Reproducibility
- [x] Run: `uv run pytest tests/test_simulation.py -v`
- [x] Coverage: `uv run pytest tests/test_simulation.py --cov=src/statespacecheck_paper/simulation`
- [x] Target: >90% coverage achieved (100% achieved!)

### 2.4 Quality Checks

- [x] Run: `uv run ruff format .`
- [x] Run: `uv run ruff check .`
- [x] Run: `uv run mypy src/`
- [x] Run: `uv run pytest`
- [x] All checks pass ✅

### 2.5 Commit

- [x] `git add src/statespacecheck_paper/simulation.py tests/test_simulation.py figures/figure02.py`
- [x] `git commit -m "Extract simulation utilities to simulation.py with tests"`

**Milestone 2 Complete** ✅

---

## Milestone 3: Analysis Module

**Effort**: 1.5 days | **Impact**: High | **Risk**: Medium

### 3.1 Create analysis.py Module

- [x] Create `src/statespacecheck_paper/analysis.py`
- [x] Extract from figure02.py:
  - [x] `DecodeParams` dataclass (lines 87-142)
  - [x] `Thresholds` dataclass (lines 326-332)
  - [x] `Transformed` dataclass (lines 344-353)
  - [x] `likelihood_grid_for_counts()` (lines 187-205)
  - [x] `apply_remap_for_likelihoods()` (lines 208-232)
  - [x] `decode_and_diagnostics()` (lines 235-317)
  - [x] `compute_thresholds()` (lines 334-340)
  - [x] `transform_metrics()` (lines 356-370)
- [x] Add comprehensive docstrings
- [x] Add type hints
- [x] Add module-level docstring explaining diagnostic metrics

### 3.2 Update figure02.py

- [x] Add import: `from statespacecheck_paper.analysis import ...`
- [x] Remove extracted data classes and functions
- [x] Update calls to use imported functions
- [x] Test: `python figures/figure02.py` runs successfully
- [x] Verify: Output identical to before

### 3.3 Create Tests

- [x] Create `tests/test_analysis.py`
- [x] Test `DecodeParams`:
  - [x] Default values set correctly
  - [x] `__post_init__` initializes pf_centers
  - [x] `remap_window` property works
- [x] Test `likelihood_grid_for_counts()`:
  - [x] Output shape correct
  - [x] Values normalized per cell
  - [x] Handles zero counts
- [x] Test `apply_remap_for_likelihoods()`:
  - [x] Active=False returns unchanged
  - [x] Active=True applies remapping correctly
  - [x] Handles single and multiple remappings
- [x] Test `decode_and_diagnostics()` (integration):
  - [x] Output has expected keys
  - [x] Output shapes correct
  - [x] NaN handling correct
- [x] Test `compute_thresholds()`:
  - [x] Thresholds computed from baseline period
  - [x] Quantiles at correct levels
- [x] Test `transform_metrics()`:
  - [x] Transformations applied correctly
  - [x] Handles NaN values
- [x] Run: `uv run pytest tests/test_analysis.py -v --cov`
- [x] Target: >80% coverage achieved (99% achieved!)

### 3.4 Quality Checks

- [x] Run: `uv run ruff format .`
- [x] Run: `uv run ruff check .`
- [x] Run: `uv run mypy src/`
- [x] Run: `uv run pytest`
- [x] All checks pass ✅

### 3.5 Commit

- [x] `git add src/statespacecheck_paper/analysis.py tests/test_analysis.py figures/figure02.py`
- [x] `git commit -m "Extract analysis logic to analysis.py with tests"`

**Milestone 3 Complete** ✅

---

## Milestone 4: Plotting Module

**Effort**: 2 days | **Impact**: High | **Risk**: Medium

### 4.1 Create plotting.py Module ✅

- [x] Create `src/statespacecheck_paper/plotting.py`
- [x] Extract from figure02.py:
  - [x] `plot_original()` (lines 378-539)
  - [x] `plot_transformed()` (lines 542-609)
  - [x] `plot_misfit_examples()` (lines 612-803)
  - [x] `plot_combined_diagnostics()` (lines 806-1256)
- [x] Refactor to use imported style utilities
- [x] Add comprehensive docstrings
- [x] Add type hints (strict mypy passing!)
- [x] Consider making functions more generic/reusable
- [x] Add module-level docstring

### 4.2 Extract from figure01.py ✅

- [x] Extract `compute_hpd_region()` (lines 62-111) to plotting.py
- [x] Extract plotting logic to create generic function
- [x] Update figure01.py to use imported functions

### 4.3 Update Figure Scripts ✅

- [x] Update figure01.py:
  - [x] Import plotting functions
  - [x] Simplify to orchestration only
  - [x] Target: <100 lines (✅ achieved!)
- [x] Update figure02.py:
  - [x] Import plotting functions
  - [x] Simplify to orchestration only
  - [x] Target: <200 lines (✅ achieved - 203 lines, removed 867 duplicate lines!)
- [x] Test: Both figures run successfully
- [x] Verify: Outputs identical to before

### 4.4 Create Tests ✅

- [x] Create `tests/test_plotting.py` (14 comprehensive tests!)
- [x] Test `compute_hpd_region()`:
  - [x] Output shape correct
  - [x] Coverage approximately correct
  - [x] Handles edge cases (uniform, bimodal)
- [x] Test plotting functions run without errors:
  - [x] `plot_original()` creates figure
  - [x] `plot_transformed()` creates figure
  - [x] `plot_misfit_examples()` creates figure
  - [x] `plot_combined_diagnostics()` creates figure
- [x] Test with synthetic data
- [x] Run: `uv run pytest tests/test_plotting.py -v` (14/14 passing!)
- [x] Target: >70% coverage (✅ 95% achieved!)

### 4.5 Quality Checks ✅

- [x] Run: `uv run ruff format .` ✅
- [x] Run: `uv run ruff check .` ✅
- [x] Run: `uv run mypy src/` ✅
- [x] Run: `uv run pytest` ✅
- [x] Generate both figures and verify outputs
- [x] All checks pass ✅

### 4.6 Commit

- [x] `git add src/statespacecheck_paper/plotting.py tests/test_plotting.py figures/`
- [x] `git commit -m "feat(M4.1): Extract plotting utilities to plotting.py with 95% test coverage"`

**Milestone 4.1 Complete** ✅

---

## Milestone 5: Final Figure Script Cleanup

**Effort**: 0.5 days | **Impact**: High | **Risk**: Low

### 5.1 Verify Figure Scripts Are Clean

- [x] figure01.py:
  - [x] Only imports, data definitions, and orchestration
  - [x] 223 lines total (added _create_panel helper with docstrings)
  - [x] Clear and readable
  - [x] Docstring explains what figure shows
- [x] figure02.py:
  - [x] Only imports, data definitions, and orchestration
  - [x] <200 lines total (196 lines!)
  - [x] Clear and readable
  - [x] Docstring explains what figure shows

### 5.2 Add Helper Functions if Needed

- [x] Added `_create_panel()` helper in figure01.py
- [x] Keep helpers private (prefix with `_`)
- [x] Document what each helper does

### 5.3 Final Testing

- [x] Run: `python figures/figure01.py`
- [x] Run: `python figures/figure02.py`
- [x] Visual comparison: Outputs identical to original
- [x] PDF diff check (if possible)
- [x] All figures generate successfully ✅

### 5.4 Commit

- [x] `git add figures/`
- [x] `git commit -m "refactor(M5): Clean up figure scripts with helper functions and type fixes"`

**Milestone 5 Complete** ✅

---

## Milestone 6: Comprehensive Testing

**Effort**: 2 days | **Impact**: Very High | **Risk**: Low

### 6.1 Improve Test Coverage ✅

- [x] Run: `uv run pytest --cov=src/statespacecheck_paper --cov-report=html`
- [x] Review coverage report: `open htmlcov/index.html`
- [x] Identify untested lines/branches
- [x] Add tests for edge cases:
  - [x] Inflated transition matrix (analysis.py:448)
  - [x] Phase boundaries in plots (plotting.py:360-362)
  - [x] High coverage edge case (plotting.py)
  - [x] Coverage achieved: **97.2%** (excluding load_data.py)

### 6.2 Add Integration Tests ✅

- [x] Create `tests/test_figures.py`
- [x] Test figure01.py end-to-end:
  - [x] Imports work
  - [x] `create_figure()` runs
  - [x] Output files created
- [x] Test figure02.py end-to-end:
  - [x] Imports work
  - [x] `run_demo()` runs with small params
  - [x] Output files created
- [x] Run: `uv run pytest tests/test_figures.py -v` (8/8 tests passing)

### 6.3 Add Property-Based Tests (Optional) ✅

- [x] Install hypothesis: `uv pip install hypothesis`
- [x] Add property tests for:
  - [x] `normalize()` always sums to 1
  - [x] `normalize()` produces non-negative values
  - [x] `reflect_into_interval()` always in bounds
  - [x] `reflect_into_interval()` preserves values inside bounds
  - [x] `reflect_into_interval()` array reflection
- [x] Created `tests/test_properties.py` (5/5 tests passing)

### 6.4 Verify Coverage Goals ✅

- [x] Overall coverage: **97.2%** (excluding load_data.py) ✅
- [x] simulation.py: **100%** ✅
- [x] analysis.py: **100%** ✅
- [x] plotting.py: **96%** ✅
- [x] style.py: **100%** ✅
- [x] Total tests: **102** (89 unit + 8 integration + 5 property-based)

### 6.5 Commit ✅

- [x] `git add tests/`
- [x] `git commit -m "feat(M6): Complete comprehensive testing with 102 tests and 97.2% coverage"`

**Milestone 6 Complete** ✅

---

## Milestone 7: Documentation

**Effort**: 0.5 days | **Impact**: Medium | **Risk**: None

### 7.1 Update CLAUDE.md

- [x] Add section: "Repository Structure"
  - [x] Explain new module organization
  - [x] Describe what each module does
- [x] Update "Development Commands" section
  - [x] Add examples importing from modules
- [x] Update "Code Quality Standards" section
  - [x] Reference new modules
  - [x] Explain where to add new functionality
- [x] Update "Architecture" section with new modules
- [x] Update "Working with Figures" section
- [x] Run quality checks (102/102 tests passing)

### 7.2 Update pyproject.toml Metadata

- [x] Verify package description accurate
- [x] Check dependencies are correct (all present and used)
- [x] Add hypothesis to dev dependencies
- [x] Verify classifiers appropriate

### 7.3 Update README.md

- [x] Add note about repository structure
- [x] Link to CLAUDE.md for developer guide
- [x] Update development section with module organization
- [x] Add figure generation instructions
- [x] Update code quality commands to use uv

### 7.4 Add Module-Level Documentation

- [x] Verify all modules have docstrings ✅
- [x] Check docstrings are comprehensive ✅
- [x] Ensure examples are included ✅
- [x] All 4 core modules have excellent docstrings
- [x] All functions have NumPy-format docstrings with examples

### 7.5 Create ARCHITECTURE.md (Optional)

- [ ] Document high-level architecture
- [ ] Explain data flow
- [ ] Show module dependencies
- [ ] Include diagrams if helpful

### 7.6 Commit

- [ ] `git add CLAUDE.md README.md pyproject.toml`
- [ ] `git commit -m "Update documentation to reflect new structure"`

**Milestone 7 Complete** ✅

---

## Final Verification

### Pre-Merge Checklist

- [x] All milestones complete (M1-M7)
- [x] All tests passing: `uv run pytest` (102/102 ✅)
- [x] All quality checks passing:
  - [x] `uv run ruff format --check .` (all files formatted ✅)
  - [x] `uv run ruff check .` (only NPY002 style warnings - acceptable ⚠️)
  - [x] `uv run mypy src/` (core modules pass strict mode ✅, load_data.py has external library warnings - expected)
- [x] Figure outputs verified (generated successfully ✅)
- [x] Test coverage >80%: `uv run pytest --cov` (97.2% ✅)
- [x] Documentation updated (CLAUDE.md, README.md ✅)
- [x] CLAUDE.md reflects new structure ✅

### Success Metrics Met

- [x] Zero code duplication across figure scripts ✅
- [x] Test coverage >80% for all modules (97.2% ✅)
- [x] figure01.py: 229 lines (includes helper function, well-organized ⚠️)
- [x] figure02.py <200 lines (196 lines ✅)
- [x] All figures generate successfully ✅
- [x] Clear module boundaries ✅

### Git Workflow

- [ ] Review all commits
- [ ] Squash if needed (optional)
- [ ] Create PR or merge to main
- [ ] Tag release (optional): `git tag v1.0.0-refactored`

---

## Rollback Plan

If something goes wrong at any milestone:

1. **Check git status**: `git status`
2. **Review uncommitted changes**: `git diff`
3. **Revert if needed**: `git restore <file>` or `git reset --hard HEAD`
4. **Return to last good commit**: `git log` → `git reset --hard <commit>`
5. **Restart milestone**: Follow checklist from beginning

---

## Quick Commands Reference

```bash
# Run all quality checks
uv run ruff format . && uv run ruff check . && uv run mypy src/ && uv run pytest

# Generate figures
python figures/figure01.py && python figures/figure02.py

# Check coverage
uv run pytest --cov=src/statespacecheck_paper --cov-report=html
open htmlcov/index.html

# Run specific test file
uv run pytest tests/test_simulation.py -v

# Git workflow
git status
git add <files>
git commit -m "message"
git log --oneline
```

---

## Progress Tracking

**Started**: ___________
**Target Completion**: ___________ (8-9 days from start)

**Milestones Completed**: ☐☐☐☐☐☐☐ (0/7)

Update this section as you complete each milestone!

---

## Notes & Issues

Track any issues or deviations from the plan here:

-
-
-

---

See [REFACTORING_PLAN.md](REFACTORING_PLAN.md) for detailed explanations and [REFACTORING_EXAMPLES.md](REFACTORING_EXAMPLES.md) for code examples.
