# Repository Reorganization Plan

**Date**: 2025-11-11
**Purpose**: Reorganize statespacecheck-paper repository for better scientific reproducibility and clarity

---

## Goals

1. **Separate code from outputs**: Move figure generation scripts to `scripts/`, keep outputs in `figures/`
2. **Organize outputs by type**: Main text vs supplementary figures
3. **Add manuscript infrastructure**: LaTeX source files for paper
4. **Create master script**: Single command to regenerate all figures
5. **Maintain simplicity**: No complex build systems, keep what works

---

## Current Structure (Before)

```
statespacecheck-paper/
├── src/statespacecheck_paper/     # Library code (good, keep as-is)
├── figures/
│   ├── figure01.py                # ⚠️ Scripts mixed with outputs
│   ├── figure01.pdf
│   ├── figure01.png
│   ├── figure02.py
│   ├── figure02.pdf
│   └── figure02.png
├── tests/                         # Tests (good, keep as-is)
└── scripts/                       # Empty directory
```

**Problems**:

- Figure generation scripts mixed with output files
- Hard to distinguish "what to run" from "what was generated"
- No clear place for manuscript source
- Empty `scripts/` directory is confusing

---

## Target Structure (After)

```
statespacecheck-paper/
├── src/statespacecheck_paper/     # ✅ No change - library code
│   ├── analysis.py
│   ├── plotting.py
│   ├── simulation.py
│   └── style.py
│
├── scripts/                        # 🔧 NEW: Executable scripts
│   ├── generate_figure01.py       # Moved from figures/
│   ├── generate_figure02.py
│   ├── generate_figure03.py
│   └── generate_all_figures.py    # NEW: Master script
│
├── figures/                        # 📊 Reorganized: Outputs only
│   ├── main/                      # NEW: Main text figures
│   │   ├── figure01.pdf
│   │   ├── figure01.png
│   │   ├── figure02.pdf
│   │   └── figure02.png
│   └── supplementary/             # NEW: Supplementary figures
│       └── .gitkeep
│
├── manuscript/                     # 📝 NEW: Paper LaTeX source
│   ├── main.tex
│   ├── supplement.tex
│   ├── references.bib
│   └── README.md
│
└── tests/                          # ✅ Update paths only
    └── test_figures.py            # Update FIGURES_DIR path
```

---

## Key Principles

### 1. Separation of Concerns

| Directory | Contains | Purpose | Version Controlled |
|-----------|----------|---------|-------------------|
| `src/` | Reusable Python modules | Library code, well-tested | ✅ Yes |
| `scripts/` | Executable `.py` files | Generate figures, run analyses | ✅ Yes |
| `figures/` | PDF and PNG outputs | Results of running scripts | ✅ Yes (for papers) |
| `manuscript/` | LaTeX source files | Paper text and structure | ✅ Yes |
| `tests/` | Test files | Verify code correctness | ✅ Yes |

### 2. Clear Naming Convention

**Scripts**: `scripts/generate_*.py` - clearly executable
**Outputs**: `figures/main/*.{pdf,png}` - clearly generated

### 3. Keep It Simple

- ✅ No complex build systems (Snakemake, Make, etc.)
- ✅ No numbered workflows (01_, 02_, etc.) - only 2-3 figures
- ✅ Version control outputs - small files, useful for reviewers
- ✅ Flat structure - no deep nesting

---

## Migration Steps

### Phase 1: Directory Structure (5 minutes)

1. Create new directories:
   - `scripts/`
   - `figures/main/`
   - `figures/supplementary/`
   - `manuscript/`

2. Add placeholder files:
   - `figures/supplementary/.gitkeep`

### Phase 2: Move Files (5 minutes)

1. Move figure scripts:
   - `figures/figure01.py` → `scripts/generate_figure01.py`
   - `figures/figure02.py` → `scripts/generate_figure02.py`
   - `figures/figure03.py` → `scripts/generate_figure03.py`

2. Move figure outputs:
   - `figures/figure01.{pdf,png}` → `figures/main/`
   - `figures/figure02.{pdf,png}` → `figures/main/`

3. Clean up:
   - Remove `figures/__pycache__/`

### Phase 3: Update Code (10 minutes)

1. Update save paths in scripts:
   - `scripts/generate_figure01.py`: `save_figure("figures/main/figure01")`
   - `scripts/generate_figure02.py`: `save_figure("figures/main/figure02")`
   - `scripts/generate_figure03.py`: `save_figure("figures/main/figure03")`

2. Create master script:
   - `scripts/generate_all_figures.py` - runs all figure scripts

3. Update test paths:
   - `tests/test_figures.py`: Change `FIGURES_DIR` to point to `scripts/`

### Phase 4: Create Manuscript Structure (10 minutes)

1. Create manuscript files:
   - `manuscript/README.md` - Instructions for building
   - `manuscript/main.tex` - Main manuscript template
   - `manuscript/supplement.tex` - Supplementary materials template
   - `manuscript/references.bib` - Bibliography starter

### Phase 5: Update Documentation (10 minutes)

1. Update README.md:
   - New "Repository Structure" section
   - Updated "Generating Figures" instructions

2. Update CLAUDE.md:
   - Update Architecture section with new structure
   - Update "Where to Add New Functionality" section

### Phase 6: Verification (10 minutes)

1. Test figure generation:

   ```bash
   python scripts/generate_all_figures.py
   ```

2. Run test suite:

   ```bash
   uv run pytest
   ```

3. Verify file locations:

   ```bash
   ls scripts/          # Should have 4 .py files
   ls figures/main/     # Should have 4 files (.pdf + .png)
   ls manuscript/       # Should have LaTeX files
   ```

4. Check git status:

   ```bash
   git status           # Verify moves are tracked correctly
   ```

---

## File Changes Summary

### New Files

- `scripts/generate_figure01.py` (moved from `figures/`)
- `scripts/generate_figure02.py` (moved from `figures/`)
- `scripts/generate_figure03.py` (moved from `figures/`)
- `scripts/generate_all_figures.py` (new)
- `manuscript/README.md` (new)
- `manuscript/main.tex` (new)
- `manuscript/supplement.tex` (new)
- `manuscript/references.bib` (new)
- `figures/supplementary/.gitkeep` (new)

### Modified Files

- `scripts/generate_figure01.py` - Update `save_figure()` path
- `scripts/generate_figure02.py` - Update `save_figure()` path
- `scripts/generate_figure03.py` - Update `save_figure()` path
- `tests/test_figures.py` - Update `FIGURES_DIR` path
- `README.md` - Update structure documentation
- `CLAUDE.md` - Update structure documentation

### Moved Files

- `figures/figure01.{pdf,png}` → `figures/main/figure01.{pdf,png}`
- `figures/figure02.{pdf,png}` → `figures/main/figure02.{pdf,png}`

### Deleted Files

- `figures/__pycache__/` (directory)

---

## Expected Benefits

### For Development

- ✅ **Clarity**: Obvious where to find executable code (`scripts/`)
- ✅ **Organization**: Outputs organized by purpose (main vs supplementary)
- ✅ **Efficiency**: Single command regenerates all figures
- ✅ **Standards**: Follows Python scientific computing conventions

### For Reproducibility

- ✅ **Transparency**: Clear what scripts produce what outputs
- ✅ **Simplicity**: Easy for others to regenerate figures
- ✅ **Documentation**: Manuscript source shows how figures are used
- ✅ **Testing**: Tests verify figure generation still works

### For Reviewers

- ✅ **Accessibility**: Can view figures without running code
- ✅ **Traceability**: Can see which script generated which figure
- ✅ **Validation**: Can regenerate and compare to submitted versions
- ✅ **Understanding**: Manuscript source helps understand context

---

## Post-Migration Checklist

- [ ] All files moved correctly (no broken paths)
- [ ] Figure generation works: `python scripts/generate_all_figures.py`
- [ ] All tests pass: `uv run pytest`
- [ ] Documentation updated (README.md, CLAUDE.md)
- [ ] Git history preserved (files moved with `git mv`)
- [ ] Commit message clear and descriptive

---

## Rollback Plan

If issues arise, rollback is straightforward since we used `git mv`:

```bash
# Undo all changes
git reset --hard HEAD

# Or undo specific moves
git mv scripts/generate_figure01.py figures/figure01.py
git mv figures/main/figure01.pdf figures/
git mv figures/main/figure01.png figures/
```

---

## Future Extensions (Out of Scope)

These are **not** part of this reorganization but may be added later:

- ❌ `data/` directory - No data management yet
- ❌ `notebooks/` directory - No exploration notebooks yet
- ❌ `results/` directory - No separate analysis outputs yet
- ❌ `environment.yml` - UV lock file is sufficient
- ❌ `.gitignore` updates - Keep versioning outputs
- ❌ Complex build systems - Keep it simple

---

## Success Criteria

The reorganization is successful when:

1. ✅ Scripts are in `scripts/`, outputs in `figures/`
2. ✅ `python scripts/generate_all_figures.py` works
3. ✅ All tests pass (100/100)
4. ✅ Documentation reflects new structure
5. ✅ Git history shows clean moves (not deletions)
6. ✅ No broken imports or paths
7. ✅ Manuscript infrastructure ready for paper writing

---

## Timeline

**Total Estimated Time**: 50 minutes

- Phase 1 (Directory Structure): 5 min
- Phase 2 (Move Files): 5 min
- Phase 3 (Update Code): 10 min
- Phase 4 (Manuscript): 10 min
- Phase 5 (Documentation): 10 min
- Phase 6 (Verification): 10 min

**Recommended Approach**: Execute phases sequentially, verify at each step.

---

## References

- **CLAUDE.md**: Development guidelines and standards
- **README.md**: User-facing documentation
- **Scientific Python Cookiecutter**: <https://github.com/scientific-python/cookie>
- **Research Compendium**: <https://research-compendium.science/>
