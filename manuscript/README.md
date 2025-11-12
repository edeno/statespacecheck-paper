# Manuscript Source Files

This directory contains the LaTeX source files for the paper, optimized for bioRxiv submission.

## Files

- **main.tex**: Main manuscript text
- **supplement.tex**: Supplementary materials
- **preamble.tex**: Shared package configuration (included by both main and supplement)
- **references.bib**: Bibliography database
- **.latexmkrc**: Build configuration for latexmk
- **Makefile**: Convenient build commands

## Quick Start

### Using Make (Recommended)

```bash
cd manuscript

# Build both main and supplement
make

# Build only main manuscript
make main

# Build only supplement
make supplement

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

# Build supplement
latexmk -pdf supplement.tex

# Clean build artifacts
latexmk -C

# Continuous preview mode (rebuilds on file change)
latexmk -pvc main.tex
```

### Using pdflatex (Manual)

```bash
cd manuscript

# Build main manuscript (using modern biblatex + biber)
pdflatex main.tex
biber main          # Modern bibliography processor (not bibtex)
pdflatex main.tex
pdflatex main.tex

# Build supplement
pdflatex supplement.tex
biber supplement
pdflatex supplement.tex
pdflatex supplement.tex
```

## Figures

Figures are stored in `../figures/main/` and `../figures/supplementary/` and referenced with relative paths:

```latex
\includegraphics[width=\textwidth]{../figures/main/figure01.pdf}
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
   - Include `preamble.tex` if submitting LaTeX source

## Features

### Line Numbers

Line numbers are enabled by default for peer review:

```latex
\linenumbers  % in main.tex and supplement.tex
```

To disable for final version, comment out in both files:

```latex
% \linenumbers
```

### Shared Preamble

Both `main.tex` and `supplement.tex` use `preamble.tex` for consistent formatting:

```latex
\input{preamble}
```

## Output

Compiled PDFs are created in this directory:

- `main.pdf` - Main manuscript
- `supplement.pdf` - Supplementary materials

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
ls ../figures/main/
```

## Contact

For questions about the manuscript, contact:

- Eric L. Denovellis: <eric.denovellis@ucsf.edu>
