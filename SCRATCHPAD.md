# Scratchpad - Development Notes

## Current Session: Milestone 1 - Style Module

**Date**: 2025-11-11
**Task**: Extract styling utilities to `style.py` module

### Plan

Following TDD workflow:
1. ✅ Read figure01.py to understand what to extract
2. ✅ Create test_style.py with tests (TDD - tests first!)
3. ✅ Run tests and verify they FAIL (ModuleNotFoundError as expected)
4. ✅ Create style.py implementation
5. ✅ Run tests until they PASS (13/13 tests passing, 100% coverage)
6. ✅ Update figure01.py to use new module
7. ✅ Verify figure01.py still works (generates figures successfully)
8. ✅ Run all quality checks (ruff, mypy, pytest - all passing)

### Items to Extract from figure01.py

1. **WONG color palette** (lines 20-29)
   - List of 8 colorblind-friendly colors
   - Should be a module-level constant

2. **set_figure_defaults()** (lines 31-53)
   - Sets matplotlib rcParams for publication figures
   - Has `context` parameter (currently defaults to "paper")
   - Need to support: "paper", "presentation", "poster" contexts

3. **save_figure()** (lines 55-59)
   - Saves figure as both PDF and PNG
   - Currently takes just name string
   - Enhancement: Add Path support, dpi parameter, close parameter, auto-create directories

### Design Decisions

**For `save_figure()` enhancements:**
- Add `dpi` parameter (default 450)
- Add `close` parameter (default True) - close figure after saving
- Add `Path` support - accept str or Path object
- Auto-create parent directories if they don't exist
- Keep backward compatibility with simple string names

**For `set_figure_defaults()` enhancements:**
- Support `context` parameter: "paper", "presentation", "poster"
- Different font sizes for each context
- "paper": small (7pt) - current defaults
- "presentation": medium (12pt)
- "poster": large (16pt)

### Notes

- Must include comprehensive docstrings (NumPy format)
- Full type hints required (mypy strict mode)
- Need to add helper function `get_figure_size()` per TASKS.md
- Module-level docstring required

### Testing Strategy

Tests to write:
1. Test WONG constant exists and has 8 colors
2. Test set_figure_defaults() with different contexts
3. Test save_figure() creates both PDF and PNG
4. Test save_figure() with Path objects
5. Test save_figure() auto-creates directories
6. Test get_figure_size() returns correct dimensions

---

## Completed

### Milestone 1 - Style Module ✅ COMPLETE

**Completed**: 2025-11-11

Successfully extracted styling utilities to shared module:

**1.1 Create style.py Module ✅**
- Created `src/statespacecheck_paper/style.py` with:
  - WONG color palette constant (8 colorblind-friendly colors)
  - set_figure_defaults() with context support (paper/presentation/poster)
  - save_figure() with Path support, auto-directory creation, custom DPI
  - get_figure_size() helper function
- Created comprehensive test suite (13 tests, 100% coverage)
- All quality checks passing (ruff, mypy, pytest)

**1.2 Update figure01.py ✅**
- Imported WONG, save_figure, set_figure_defaults from style module
- Removed duplicated code (color palette, utility functions)
- Verified figure01.py generates identical output
- Result: figure01.py reduced from 279 lines to ~219 lines (21% reduction)

**1.3 Update figure02.py ✅**
- Imported WONG and save_figure from style module
- Replaced two inline wong palette definitions with WONG constant
- Replaced manual save calls with save_figure()
- Verified figure02.py generates identical output
- Result: Eliminated code duplication between figure scripts

**Impact**:
- ✅ Single source of truth for styling
- ✅ Zero code duplication for colors and save logic
- ✅ Easy to change palette/fonts globally
- ✅ Consistent figures across entire paper
- ✅ 100% test coverage on style module

---

## Blockers

None.

---

## Questions

None.
