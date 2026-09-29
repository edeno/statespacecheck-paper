# Manuscript

Read [main.pdf](main.pdf), or build it from the committed source and figures:

```bash
# From the repository root:
make manuscript
```

This needs LaTeX, `latexmk`, and BibTeX with the `iopart-num` bibliography style.
The dependencies and full analysis workflow are in
[the reproduction guide](../docs/reproduce.md).

## Files

| File | Purpose |
| --- | --- |
| `main.tex` | Manuscript and preamble |
| `Local-GoF-Paper.bib` | Bibliography exported from Zotero/Better BibTeX |
| `reported_values.tex` | Generated numerical macros read by `main.tex` |
| `software_dois.json` | Zenodo DOIs of cited software releases, read by the macro emitter |
| `figures/main/` | Four main figures and Figure 3/4 analysis summaries |
| `.latexmkrc`, `Makefile` | LaTeX build settings and commands |
| `LICENSE` | CC BY 4.0 for manuscript text and figures |

`reported_values.tex` is generated from the two figure summaries. After either
summary changes, run `make reported-values` from the repository root, then build
the manuscript. The build tracks the macro file but does not regenerate it.
See the [reporting policy](../docs/figure-pipeline.md#from-summary-to-prose-the-reported-value-macros).

## Direct LaTeX commands

From this directory:

```bash
latexmk -pdf main.tex
latexmk -pvc main.tex  # rebuild as the source changes
```

For a manual build, run `pdflatex main.tex`, `bibtex main`, and then
`pdflatex main.tex` twice more. `make clean` or `latexmk -C` removes the compiled
PDF as well as temporary build files; rebuild before committing the paper.

## Submission files

A source bundle consists of `main.tex`, `reported_values.tex`,
`Local-GoF-Paper.bib`, and the figure PDFs referenced by `main.tex`. The
bibliography style is supplied by TeX Live/Overleaf. Build and inspect the PDF
before preparing a submission. Release follow-ups are tracked in [TODO.md](../TODO.md).
