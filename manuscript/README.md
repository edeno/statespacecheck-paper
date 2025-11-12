# Manuscript Source Files

This directory contains the LaTeX source files for the paper.

## Files

- **main.tex**: Main manuscript text
- **supplement.tex**: Supplementary materials
- **references.bib**: Bibliography database

## Compiling

### Using pdflatex

```bash
cd manuscript
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Using latexmk (recommended)

```bash
cd manuscript
latexmk -pdf main.tex
```

## Figures

Figures are stored in `../figures/main/` and `../figures/supplementary/` and can be included with relative paths:

```latex
\includegraphics[width=\textwidth]{../figures/main/figure01.pdf}
```

## Output

Compiled PDFs will be created in this directory:
- `main.pdf` - Main manuscript
- `supplement.pdf` - Supplementary materials
