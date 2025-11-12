# Repository Reorganization Tasks

**Based on**: PLAN.md and CLAUDE.md
**Goal**: Reorganize repository structure for better scientific reproducibility
**Timeline**: ~50 minutes total

---

## Milestone 1: Create New Directory Structure ⏱️ 5 minutes

**Goal**: Set up the new directory organization without breaking existing code

### Tasks

- [ ] Create `scripts/` directory

  ```bash
  mkdir -p scripts
  ```

- [ ] Create `figures/main/` directory

  ```bash
  mkdir -p figures/main
  ```

- [ ] Create `figures/supplementary/` directory

  ```bash
  mkdir -p figures/supplementary
  ```

- [ ] Create `manuscript/` directory

  ```bash
  mkdir -p manuscript
  ```

- [ ] Add `.gitkeep` placeholder for empty supplementary directory

  ```bash
  touch figures/supplementary/.gitkeep
  git add figures/supplementary/.gitkeep
  ```

**Verification**:

```bash
ls -la scripts/
ls -la figures/main/
ls -la figures/supplementary/
ls -la manuscript/
```

---

## Milestone 2: Move Files to New Locations ⏱️ 5 minutes

**Goal**: Relocate files using `git mv` to preserve history

### Tasks: Move Figure Scripts

- [ ] Move `figures/figure01.py` to `scripts/generate_figure01.py`

  ```bash
  git mv figures/figure01.py scripts/generate_figure01.py
  ```

- [ ] Move `figures/figure02.py` to `scripts/generate_figure02.py`

  ```bash
  git mv figures/figure02.py scripts/generate_figure02.py
  ```

- [ ] Move `figures/figure03.py` to `scripts/generate_figure03.py`

  ```bash
  git mv figures/figure03.py scripts/generate_figure03.py
  ```

### Tasks: Move Figure Outputs

- [ ] Move `figures/figure01.pdf` to `figures/main/`

  ```bash
  git mv figures/figure01.pdf figures/main/
  ```

- [ ] Move `figures/figure01.png` to `figures/main/`

  ```bash
  git mv figures/figure01.png figures/main/
  ```

- [ ] Move `figures/figure02.pdf` to `figures/main/`

  ```bash
  git mv figures/figure02.pdf figures/main/
  ```

- [ ] Move `figures/figure02.png` to `figures/main/`

  ```bash
  git mv figures/figure02.png figures/main/
  ```

### Tasks: Clean Up

- [ ] Remove old `figures/__pycache__/` directory

  ```bash
  rm -rf figures/__pycache__
  ```

- [ ] Verify `figures/` directory is now empty except for new subdirectories

  ```bash
  ls -la figures/
  # Should show: main/ supplementary/
  ```

**Verification**:

```bash
git status  # Check that moves are tracked correctly
ls scripts/  # Should show: generate_figure01.py, generate_figure02.py, generate_figure03.py
ls figures/main/  # Should show: figure01.pdf, figure01.png, figure02.pdf, figure02.png
```

---

## Milestone 3: Update Code for New Paths ⏱️ 10 minutes

**Goal**: Update all scripts and tests to use new file locations

### Tasks: Update Figure Script Save Paths

- [ ] Update `scripts/generate_figure01.py` save path
  - Open `scripts/generate_figure01.py`
  - Find: `save_figure("figures/figure01")`
  - Replace with: `save_figure("figures/main/figure01")`
  - Save file

- [ ] Update `scripts/generate_figure02.py` save path
  - Open `scripts/generate_figure02.py`
  - Find: `save_figure("figures/figure02")`
  - Replace with: `save_figure("figures/main/figure02")`
  - Save file

- [ ] Update `scripts/generate_figure03.py` save path (if needed)
  - Open `scripts/generate_figure03.py`
  - Check if it has any `save_figure()` calls
  - If yes, update to: `save_figure("figures/main/figure03")`
  - Save file

### Tasks: Create Master Script

- [ ] Create `scripts/generate_all_figures.py`
  - Copy template from PLAN.md "Step 5: Create Master Script" section
  - Save to `scripts/generate_all_figures.py`
  - Make executable if needed: `chmod +x scripts/generate_all_figures.py`

- [ ] Add master script to git

  ```bash
  git add scripts/generate_all_figures.py
  ```

### Tasks: Update Test Paths

- [ ] Update `tests/test_figures.py` import path
  - Open `tests/test_figures.py`
  - Find: `FIGURES_DIR = Path(__file__).parent.parent / "figures"`
  - Replace with: `FIGURES_DIR = Path(__file__).parent.parent / "scripts"`
  - Save file

- [ ] Verify test updates with linter

  ```bash
  uv run ruff check tests/test_figures.py
  ```

**Verification**:

```bash
# Check that sed worked (or manual edits are correct)
grep "save_figure" scripts/generate_figure01.py
grep "save_figure" scripts/generate_figure02.py

# Check test path
grep "FIGURES_DIR" tests/test_figures.py
```

---

## Milestone 4: Create Manuscript Infrastructure ⏱️ 10 minutes

**Goal**: Add LaTeX source files for paper writing

### Tasks: Create Manuscript Files

- [ ] Create `manuscript/README.md`
  - Copy template from PLAN.md "Step 6: Create Manuscript Structure"
  - Save to `manuscript/README.md`

- [ ] Create `manuscript/main.tex`
  - Copy template from PLAN.md
  - Customize author names if needed
  - Save to `manuscript/main.tex`

- [ ] Create `manuscript/supplement.tex` (optional starter)

  ```latex
  \documentclass[11pt]{article}

  \usepackage{graphicx}
  \usepackage{amsmath}
  \usepackage{hyperref}

  \title{Supplementary Materials: Goodness-of-fit Diagnostics for State Space Models}
  \author{Eric Denovellis, Sirui Zeng, Uri T. Eden}

  \begin{document}
  \maketitle

  \section{Supplementary Methods}

  \section{Supplementary Figures}

  \end{document}
  ```

- [ ] Create `manuscript/references.bib`
  - Copy template from PLAN.md
  - Save to `manuscript/references.bib`

- [ ] Add manuscript files to git

  ```bash
  git add manuscript/
  ```

**Verification**:

```bash
ls -la manuscript/
# Should show: README.md, main.tex, supplement.tex, references.bib

# Optional: Test LaTeX compilation
cd manuscript
pdflatex main.tex  # Should compile (may have warnings)
cd ..
```

---

## Milestone 5: Update Documentation ⏱️ 10 minutes

**Goal**: Update all documentation to reflect new structure

### Tasks: Update README.md

- [ ] Open `README.md`

- [ ] Update "Repository Structure" section
  - Replace old structure with new structure from PLAN.md
  - Ensure accuracy with current files

- [ ] Update "Generating Figures" section
  - Add instructions for master script:

    ```markdown
    ### Generating Figures

    ```bash
    # Generate all figures
    python scripts/generate_all_figures.py

    # Generate individual figure
    python scripts/generate_figure01.py

    # Outputs:
    # - figures/main/figure01.pdf (publication quality, 450 DPI)
    # - figures/main/figure01.png (preview quality, 450 DPI)
    ```

    ```

- [ ] Add "Compiling the Manuscript" section (optional)

  ```markdown
  ### Compiling the Manuscript

  ```bash
  cd manuscript
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex
  ```

  ```

- [ ] Save README.md

### Tasks: Update CLAUDE.md

- [ ] Open `CLAUDE.md`

- [ ] Update "Architecture" section (around line 11-36)
  - Replace directory tree with new structure from PLAN.md
  - Update file counts if changed

- [ ] Update "Figure Scripts" section (around line 45-47)
  - Update paths: `scripts/generate_figure01.py` instead of `figures/figure01.py`

- [ ] Update "Where to Add New Functionality" section (around line 265-302)
  - Update "Creating new figures" to reference `scripts/` directory

- [ ] Update "Example structure" code block (around line 102-130)
  - Update import statements if needed
  - Update save_figure path example

- [ ] Save CLAUDE.md

**Verification**:

```bash
# Check documentation looks correct
head -100 README.md
head -100 CLAUDE.md

# Check for broken links or formatting
grep "scripts/" README.md
grep "figures/main/" README.md
```

---

## Milestone 6: Verification & Testing ⏱️ 10 minutes

**Goal**: Verify everything works correctly after reorganization

### Tasks: Test Figure Generation

- [ ] Test individual figure generation

  ```bash
  python scripts/generate_figure01.py
  ```

  - Check output appears in `figures/main/figure01.{pdf,png}`

- [ ] Test master script

  ```bash
  python scripts/generate_all_figures.py
  ```

  - Should run both figure01 and figure02
  - Check for success messages
  - Verify all outputs exist in `figures/main/`

- [ ] Verify figure outputs are correct

  ```bash
  ls -lh figures/main/
  # Should show recent timestamps and reasonable file sizes
  ```

### Tasks: Run Full Test Suite

- [ ] Run pytest with coverage

  ```bash
  uv run pytest -v
  ```

  - Verify all 100 tests pass
  - Check for any import errors or path issues

- [ ] Check test coverage

  ```bash
  uv run pytest --cov
  ```

  - Should still be ~75% overall coverage

### Tasks: Code Quality Checks

- [ ] Run ruff format check

  ```bash
  uv run ruff format --check .
  ```

  - Should pass (no formatting changes needed)

- [ ] Run ruff linting

  ```bash
  uv run ruff check .
  ```

  - Check for any new errors (there may be pre-existing NPY002 warnings in tests)

- [ ] Run mypy type checking

  ```bash
  uv run mypy src/
  ```

  - Should pass (same pre-existing warnings for external libraries)

### Tasks: Verify Git Status

- [ ] Check git status for correct tracking

  ```bash
  git status
  ```

  - Should show moved files (not deleted + added)
  - Should show new files (master script, manuscript files)
  - Should show modified files (updated paths)

- [ ] Verify file moves preserved history

  ```bash
  git log --follow scripts/generate_figure01.py
  ```

  - Should show history from when it was `figures/figure01.py`

**Verification Checklist**:

- [ ] `python scripts/generate_all_figures.py` runs successfully
- [ ] All 100 tests pass
- [ ] No broken imports or path errors
- [ ] Git history preserved for moved files
- [ ] Output files in correct locations
- [ ] Documentation accurate

---

## Milestone 7: Commit Changes ⏱️ 5 minutes

**Goal**: Create clean, well-documented commit

### Tasks: Stage Changes

- [ ] Review all changes

  ```bash
  git status
  git diff  # Check modifications
  ```

- [ ] Stage all changes

  ```bash
  git add -A
  ```

- [ ] Verify staging

  ```bash
  git status
  ```

### Tasks: Create Commit

- [ ] Commit with descriptive message

  ```bash
  git commit -m "refactor: Reorganize repository structure

- Move figure scripts from figures/ to scripts/
- Organize outputs in figures/main/ and figures/supplementary/
- Add manuscript/ directory with LaTeX templates
- Create master figure generation script (generate_all_figures.py)
- Update test paths to point to scripts/
- Update documentation (README.md, CLAUDE.md)

Implements separation of concerns: scripts (code) vs figures (outputs).
See PLAN.md for full reorganization plan.

Tests: All 100 tests passing
Quality: ruff format ✓, ruff check ✓, mypy ✓"

  ```

- [ ] Verify commit
  ```bash
  git log -1 --stat
  ```

**Verification**:

```bash
# Check commit includes all expected changes
git show --name-status
# Should show R (renamed) for moved files
# Should show A (added) for new files
# Should show M (modified) for updated files
```

---

## Post-Migration Tasks (Optional)

**Goal**: Enhance repository beyond basic reorganization

### Optional: Enhance Documentation

- [ ] Add "Contributing" section to README.md
- [ ] Add architecture diagram to CLAUDE.md
- [ ] Create `.github/PULL_REQUEST_TEMPLATE.md`

### Optional: CI/CD Improvements

- [ ] Add figure regeneration check to CI
  - Create `.github/workflows/regenerate-figures.yml`
  - Compare regenerated vs committed figures

- [ ] Add manuscript compilation to CI
  - Check LaTeX compiles without errors

### Optional: Dependency Management

- [ ] Pin exact versions in pyproject.toml

  ```toml
  dependencies = [
      "statespacecheck==0.2.1",  # Check actual version
      "non_local_detector==1.0.0",
      "spyglass-neuro==0.5.3",
  ]
  ```

- [ ] Test fresh install

  ```bash
  uv pip install -e ".[dev]"
  python scripts/generate_all_figures.py
  ```

---

## Troubleshooting

### If Figure Generation Fails

**Problem**: `python scripts/generate_all_figures.py` fails

**Solutions**:

1. Check import paths in scripts
2. Verify `save_figure()` paths are correct
3. Run individual scripts to isolate issue
4. Check for typos in path strings

### If Tests Fail

**Problem**: `pytest` shows import errors or path issues

**Solutions**:

1. Verify `FIGURES_DIR` in `tests/test_figures.py` points to `scripts/`
2. Check that test is importing from correct module
3. Clear pytest cache: `rm -rf .pytest_cache`
4. Reinstall package: `uv pip install -e ".[dev]"`

### If Git Shows Deletions Instead of Moves

**Problem**: `git status` shows deleted + untracked instead of renamed

**Solution**:

1. Used `git mv` instead of `mv` - redo migration
2. Or stage files and git will detect renames:

   ```bash
   git add -A
   git status  # Should now show renames
   ```

---

## Success Criteria

The reorganization is complete and successful when all of these are true:

**Structure**:

- [x] `scripts/` contains all figure generation scripts
- [x] `figures/main/` contains all figure outputs
- [x] `manuscript/` contains LaTeX source files
- [x] No files remain in old `figures/` location (except subdirs)

**Functionality**:

- [x] `python scripts/generate_all_figures.py` works
- [x] All 100 tests pass
- [x] No import errors or broken paths
- [x] Figures generate to correct locations

**Quality**:

- [x] `uv run ruff format --check .` passes
- [x] `uv run ruff check .` passes (or same errors as before)
- [x] `uv run mypy src/` passes (or same warnings as before)

**Documentation**:

- [x] README.md reflects new structure
- [x] CLAUDE.md reflects new structure
- [x] PLAN.md documents the reorganization

**Git**:

- [x] File moves tracked correctly (not delete+add)
- [x] Commit message is clear and descriptive
- [x] Git history preserved for moved files

---

## Timeline Summary

| Milestone | Tasks | Estimated Time |
|-----------|-------|----------------|
| 1. Directory Structure | 5 | 5 min |
| 2. Move Files | 7 | 5 min |
| 3. Update Code | 5 | 10 min |
| 4. Manuscript | 4 | 10 min |
| 5. Documentation | 4 | 10 min |
| 6. Verification | 10 | 10 min |
| 7. Commit | 3 | 5 min |
| **Total** | **38 tasks** | **55 min** |

---

## Notes

- **CLAUDE.md Standards**: All code changes must maintain 100% test pass rate, type safety, and formatting
- **Git Best Practices**: Use `git mv` to preserve file history
- **Testing**: Run tests after each milestone to catch issues early
- **Documentation**: Update docs before committing to ensure accuracy

---

## References

- **PLAN.md**: Full reorganization plan with rationale
- **CLAUDE.md**: Development standards and architecture
- **README.md**: User-facing documentation
