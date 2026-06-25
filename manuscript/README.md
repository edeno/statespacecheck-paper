# Manuscript Source Files

This directory contains the LaTeX source files for the paper, optimized for bioRxiv submission.

## Files

- **main.tex**: Main manuscript text (self-contained, with its own inline preamble)
- **Local-GoF-Paper.bib**: Bibliography database (BibTeX, exported from Zotero/Better BibTeX)
- **plos2025.bst**: PLOS numbered-Vancouver BibTeX style used by `main.tex`
- **.latexmkrc**: Build configuration for latexmk
- **Makefile**: Convenient build commands
- **LICENSE**: Creative Commons Attribution 4.0 (CC BY 4.0) — covers the manuscript text and figures in this directory

## Quick Start

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

# Build main manuscript (cite + BibTeX, plos2025.bst; main.tex is self-contained)
pdflatex main.tex
bibtex main          # processes \bibliography with the plos2025 style
pdflatex main.tex
pdflatex main.tex
```

## Figures

Figures are stored in `figures/main/` and `figures/supplementary/` (within `manuscript/`) and referenced with relative paths:

```latex
\includegraphics[width=\textwidth]{figures/main/figure01.pdf}
```

Generate figures before building the manuscript:

```bash
cd ..
python scripts/generate_all_figures.py
cd manuscript
make
```

## Preparing for bioRxiv Submission

1. **Generate all figures**:

   ```bash
   python scripts/generate_all_figures.py
   ```

2. **Build manuscript**:

   ```bash
   cd manuscript
   make
   ```

3. **Review output**:
   - Check `main.pdf` for proper formatting
   - Verify line numbers are present (for peer review)
   - Ensure all figures appear correctly

4. **Package for submission**:
   - bioRxiv accepts PDF uploads directly
   - Optionally include source files (.tex, .bib, figures/)

## Features

### Line Numbers

Line numbers are enabled by default for peer review:

```latex
\linenumbers  % in main.tex
```

To disable for final version, comment it out:

```latex
% \linenumbers
```

### Preamble

`main.tex` is self-contained: it carries its own inline preamble, so its
packages and bibliography setup are defined directly in the file.

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

Generate figures first:

```bash
cd ..
python scripts/generate_all_figures.py
cd manuscript
```

Verify figures exist:

```bash
ls figures/main/
```

## Contact

For questions about the manuscript, contact:

- Eric L. Denovellis: <eric.denovellis@ucsf.edu>
