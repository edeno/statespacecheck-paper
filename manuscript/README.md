# Manuscript Source Files

This directory contains the LaTeX source files for the paper, optimized for bioRxiv submission.

## Files

- **main.tex**: Main manuscript text with an inline preamble; inputs `reported_values.tex`
- **reported_values.tex**: Generated `\newcommand` definitions for reported
  analysis statistics and configuration; `main.tex` inputs it. Do not edit by
  hand — regenerate from the repository root with
  `uv run python scripts/emit_reported_values.py` after the figure summaries
  change (a test fails if it is stale).
- **Local-GoF-Paper.bib**: Bibliography database (BibTeX, exported from Zotero/Better BibTeX)
- Bibliography style: `iopart-num` (IOP numbered/Vancouver), used by `main.tex` via `\bibliographystyle{iopart-num}`. It ships with TeX Live / Overleaf, so it is not vendored in this directory.
- **.latexmkrc**: Build configuration for latexmk
- **Makefile**: Convenient build commands
- **LICENSE**: Creative Commons Attribution 4.0 (CC BY 4.0) — covers the manuscript text and figures in this directory

## Quick Start

The committed figures and `reported_values.tex` are sufficient to build the
paper with a LaTeX installation; Python is needed only to regenerate them.
After changing either summary JSON, run the emitter from the repository root
before building. Make tracks the macro file but does not regenerate it.

### Using Make (Recommended)

```bash
cd manuscript

# Build the main manuscript
make

# Clean all build artifacts
make clean

# View PDF
make view
```

### Using latexmk (Recommended)

```bash
cd manuscript

# Build main manuscript
latexmk -pdf main.tex

# Clean build artifacts
latexmk -C

# Continuous preview mode (rebuilds on file change)
latexmk -pvc main.tex
```

### Using pdflatex (Manual)

```bash
cd manuscript

# Build main manuscript (cite + BibTeX, iopart-num style)
pdflatex main.tex
bibtex main          # processes \bibliography with the iopart-num style
pdflatex main.tex
pdflatex main.tex
```

## Figures

Figures are stored in `figures/main/` and `figures/supplementary/` (within `manuscript/`) and referenced with relative paths:

```latex
\includegraphics[width=\textwidth]{figures/main/figure01.pdf}
```

To regenerate figures and reported values, run from the repository root
(Figure 4 requires the [derived data exports](../docs/figure-pipeline.md#figure-4--real-data-decoder-diagnostics)):

```bash
uv sync --frozen
uv run python scripts/generate_all_figures.py
uv run python scripts/emit_reported_values.py
make -C manuscript
```

The [reporting policy](../docs/figure-pipeline.md#from-summary-to-prose-the-reported-value-macros)
uses two significant figures for decoding errors and approximate constants,
whole percentages for flag and rescue rates, and full exact counts and settings.
The JSONs retain full precision; their standard errors do not determine digits.

## Preparing for bioRxiv Submission

Run these commands from the repository root.

1. **Generate all figures and reported values**:

   ```bash
   uv sync --frozen
   uv run python scripts/generate_all_figures.py
   uv run python scripts/emit_reported_values.py
   ```

2. **Build manuscript**:

   ```bash
   make -C manuscript
   ```

3. **Review output**:
   - Check `manuscript/main.pdf` for proper formatting
   - Ensure all figures appear correctly

4. **Package for submission**:
   - bioRxiv accepts PDF uploads directly
   - A source bundle must include `main.tex`, `reported_values.tex`,
     `Local-GoF-Paper.bib`, and the referenced files in `figures/`

## Features

### Preamble

`main.tex` carries its own inline preamble, so its packages and bibliography
setup are defined directly in the file. Its generated number macros live in
the separate, committed `reported_values.tex` file.

## Output

The compiled PDF is created in this directory:

- `main.pdf` - Main manuscript

Build artifacts (`.aux`, `.bbl`, `.log`, etc.) are ignored by Git (see `.gitignore`).

## Troubleshooting

### Missing packages

If you get "Package not found" errors, install TeX Live or MacTeX:

```bash
# macOS
brew install --cask mactex

# Linux (Debian/Ubuntu)
sudo apt-get install texlive-full

# Check installation
pdflatex --version
```

### Bibliography not appearing

Make sure to run the full build sequence:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use `latexmk` which handles this automatically:

```bash
latexmk -pdf main.tex
```

### Figures not found

Generate figures and refresh reported values from the repository root:

```bash
uv run python scripts/generate_all_figures.py
uv run python scripts/emit_reported_values.py
```

Verify figures exist:

```bash
ls manuscript/figures/main/
```

## Contact

For questions about the manuscript, contact:

- Eric L. Denovellis: <eric.denovellis@ucsf.edu>
