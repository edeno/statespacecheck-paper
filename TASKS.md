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

- [ ] `git add src/statespacecheck_paper/simulation.py tests/test_simulation.py figures/figure02.py`
- [ ] `git commit -m "Extract simulation utilities to simulation.py with tests"`

**Milestone 2 Complete** ✅

---

## Milestone 3: Analysis Module

**Effort**: 1.5 days | **Impact**: High | **Risk**: Medium

### 3.1 Create analysis.py Module

- [ ] Create `src/statespacecheck_paper/analysis.py`
- [ ] Extract from figure02.py:
  - [ ] `DecodeParams` dataclass (lines 87-142)
  - [ ] `Thresholds` dataclass (lines 326-332)
  - [ ] `Transformed` dataclass (lines 344-353)
  - [ ] `likelihood_grid_for_counts()` (lines 187-205)
  - [ ] `apply_remap_for_likelihoods()` (lines 208-232)
  - [ ] `decode_and_diagnostics()` (lines 235-317)
  - [ ] `compute_thresholds()` (lines 334-340)
  - [ ] `transform_metrics()` (lines 356-370)
- [ ] Add comprehensive docstrings
- [ ] Add type hints
- [ ] Add module-level docstring explaining diagnostic metrics

### 3.2 Update figure02.py

- [ ] Add import: `from statespacecheck_paper.analysis import ...`
- [ ] Remove extracted data classes and functions
- [ ] Update calls to use imported functions
- [ ] Test: `python figures/figure02.py` runs successfully
- [ ] Verify: Output identical to before

### 3.3 Create Tests

- [ ] Create `tests/test_analysis.py`
- [ ] Test `DecodeParams`:
  - [ ] Default values set correctly
  - [ ] `__post_init__` initializes pf_centers
  - [ ] `remap_window` property works
- [ ] Test `likelihood_grid_for_counts()`:
  - [ ] Output shape correct
  - [ ] Values normalized per cell
  - [ ] Handles zero counts
- [ ] Test `apply_remap_for_likelihoods()`:
  - [ ] Active=False returns unchanged
  - [ ] Active=True applies remapping correctly
  - [ ] Handles single and multiple remappings
- [ ] Test `decode_and_diagnostics()` (integration):
  - [ ] Output has expected keys
  - [ ] Output shapes correct
  - [ ] NaN handling correct
- [ ] Test `compute_thresholds()`:
  - [ ] Thresholds computed from baseline period
  - [ ] Quantiles at correct levels
- [ ] Test `transform_metrics()`:
  - [ ] Transformations applied correctly
  - [ ] Handles NaN values
- [ ] Run: `uv run pytest tests/test_analysis.py -v --cov`
- [ ] Target: >80% coverage achieved

### 3.4 Quality Checks

- [ ] Run: `uv run ruff format .`
- [ ] Run: `uv run ruff check .`
- [ ] Run: `uv run mypy src/`
- [ ] Run: `uv run pytest`
- [ ] All checks pass ✅

### 3.5 Commit

- [ ] `git add src/statespacecheck_paper/analysis.py tests/test_analysis.py figures/figure02.py`
- [ ] `git commit -m "Extract analysis logic to analysis.py with tests"`

**Milestone 3 Complete** ✅

---

## Milestone 4: Plotting Module

**Effort**: 2 days | **Impact**: High | **Risk**: Medium

### 4.1 Create plotting.py Module

- [ ] Create `src/statespacecheck_paper/plotting.py`
- [ ] Extract from figure02.py:
  - [ ] `plot_original()` (lines 378-539)
  - [ ] `plot_transformed()` (lines 542-609)
  - [ ] `plot_misfit_examples()` (lines 612-803)
  - [ ] `plot_combined_diagnostics()` (lines 806-1256)
- [ ] Refactor to use imported style utilities
- [ ] Add comprehensive docstrings
- [ ] Add type hints
- [ ] Consider making functions more generic/reusable
- [ ] Add module-level docstring

### 4.2 Extract from figure01.py

- [ ] Extract `compute_hpd_region()` (lines 62-111) to plotting.py
- [ ] Extract plotting logic to create generic function
- [ ] Update figure01.py to use imported functions

### 4.3 Update Figure Scripts

- [ ] Update figure01.py:
  - [ ] Import plotting functions
  - [ ] Simplify to orchestration only
  - [ ] Target: <100 lines
- [ ] Update figure02.py:
  - [ ] Import plotting functions
  - [ ] Simplify to orchestration only
  - [ ] Target: <200 lines
- [ ] Test: Both figures run successfully
- [ ] Verify: Outputs identical to before

### 4.4 Create Tests

- [ ] Create `tests/test_plotting.py`
- [ ] Test `compute_hpd_region()`:
  - [ ] Output shape correct
  - [ ] Coverage approximately correct
  - [ ] Handles edge cases
- [ ] Test plotting functions run without errors:
  - [ ] `plot_original()` creates figure
  - [ ] `plot_misfit_examples()` creates figure
  - [ ] `plot_combined_diagnostics()` creates figure
- [ ] Test with synthetic data
- [ ] Run: `uv run pytest tests/test_plotting.py -v`
- [ ] Target: >70% coverage

### 4.5 Quality Checks

- [ ] Run: `uv run ruff format .`
- [ ] Run: `uv run ruff check .`
- [ ] Run: `uv run mypy src/`
- [ ] Run: `uv run pytest`
- [ ] Generate both figures and verify outputs
- [ ] All checks pass ✅

### 4.6 Commit

- [ ] `git add src/statespacecheck_paper/plotting.py tests/test_plotting.py figures/`
- [ ] `git commit -m "Extract plotting utilities to plotting.py with tests"`

**Milestone 4 Complete** ✅

---

## Milestone 5: Final Figure Script Cleanup

**Effort**: 0.5 days | **Impact**: High | **Risk**: Low

### 5.1 Verify Figure Scripts Are Clean

- [ ] figure01.py:
  - [ ] Only imports, data definitions, and orchestration
  - [ ] <100 lines total
  - [ ] Clear and readable
  - [ ] Docstring explains what figure shows
- [ ] figure02.py:
  - [ ] Only imports, data definitions, and orchestration
  - [ ] <200 lines total
  - [ ] Clear and readable
  - [ ] Docstring explains what figure shows

### 5.2 Add Helper Functions if Needed

- [ ] Consider adding `_simulate_all_phases()` helper in figure02.py
- [ ] Keep helpers private (prefix with `_`)
- [ ] Document what each helper does

### 5.3 Final Testing

- [ ] Run: `python figures/figure01.py`
- [ ] Run: `python figures/figure02.py`
- [ ] Visual comparison: Outputs identical to original
- [ ] PDF diff check (if possible)
- [ ] All figures generate successfully ✅

### 5.4 Commit

- [ ] `git add figures/`
- [ ] `git commit -m "Final cleanup of figure scripts - pure orchestration"`

**Milestone 5 Complete** ✅

---

## Milestone 6: Comprehensive Testing

**Effort**: 2 days | **Impact**: Very High | **Risk**: Low

### 6.1 Improve Test Coverage

- [ ] Run: `uv run pytest --cov=src/statespacecheck_paper --cov-report=html`
- [ ] Review coverage report: `open htmlcov/index.html`
- [ ] Identify untested lines/branches
- [ ] Add tests for edge cases:
  - [ ] Empty arrays
  - [ ] NaN values
  - [ ] Zero values
  - [ ] Boundary conditions

### 6.2 Add Integration Tests

- [ ] Create `tests/test_figures.py`
- [ ] Test figure01.py end-to-end:
  - [ ] Imports work
  - [ ] `create_figure()` runs
  - [ ] Output files created
- [ ] Test figure02.py end-to-end:
  - [ ] Imports work
  - [ ] `run_demo()` runs with small params
  - [ ] Output files created
- [ ] Run: `uv run pytest tests/test_figures.py -v`

### 6.3 Add Property-Based Tests (Optional)

- [ ] Install hypothesis: `uv pip install hypothesis`
- [ ] Add property tests for:
  - [ ] `normalize()` always sums to 1
  - [ ] `reflect_into_interval()` always in bounds
  - [ ] `simulate_walk()` boundaries never violated

### 6.4 Verify Coverage Goals

- [ ] Overall coverage >80% ✅
- [ ] simulation.py >90% ✅
- [ ] analysis.py >80% ✅
- [ ] plotting.py >70% ✅
- [ ] style.py >70% ✅

### 6.5 Commit

- [ ] `git add tests/`
- [ ] `git commit -m "Add comprehensive test suite with >80% coverage"`

**Milestone 6 Complete** ✅

---

## Milestone 7: Documentation

**Effort**: 0.5 days | **Impact**: Medium | **Risk**: None

### 7.1 Update CLAUDE.md

- [ ] Add section: "Repository Structure"
  - [ ] Explain new module organization
  - [ ] Describe what each module does
- [ ] Update "Development Commands" section
  - [ ] Add examples importing from modules
- [ ] Add section: "Adding New Figures"
  - [ ] Step-by-step guide
  - [ ] Template/example code
- [ ] Update "Code Quality Standards" section
  - [ ] Reference new modules
  - [ ] Explain where to add new functionality

### 7.2 Update pyproject.toml Metadata

- [ ] Verify package description accurate
- [ ] Check dependencies are correct
- [ ] Verify classifiers appropriate

### 7.3 Update README.md

- [ ] Add note about repository structure
- [ ] Link to CLAUDE.md for developer guide
- [ ] Update development section if needed

### 7.4 Add Module-Level Documentation

- [ ] Verify all modules have docstrings ✅
- [ ] Check docstrings are comprehensive ✅
- [ ] Ensure examples are included ✅

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

- [ ] All milestones complete
- [ ] All tests passing: `uv run pytest`
- [ ] All quality checks passing:
  - [ ] `uv run ruff format --check .`
  - [ ] `uv run ruff check .`
  - [ ] `uv run mypy src/`
- [ ] Figure outputs verified identical to originals
- [ ] Test coverage >80%: `uv run pytest --cov`
- [ ] Documentation updated
- [ ] CLAUDE.md reflects new structure

### Success Metrics Met

- [ ] Zero code duplication across figure scripts ✅
- [ ] Test coverage >80% for all modules ✅
- [ ] figure01.py <100 lines ✅
- [ ] figure02.py <200 lines ✅
- [ ] All figures produce identical output ✅
- [ ] Clear module boundaries ✅

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
