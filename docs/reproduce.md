# Reproduce the paper

Run commands from the repository root. The four `scripts/generate_figureNN.py`
entry points generate the main figures; the root Makefile calls those scripts
and orders the steps needed to build the paper.

## Requirements and resources

- **Python environment:** [uv](https://docs.astral.sh/uv/getting-started/installation/)
  and the committed `uv.lock`. `.python-version` selects Python 3.11 for
  reproduction; CI also tests Python 3.10 and 3.12. `uv sync --frozen` installs
  into `.venv` and may download Python and dependencies.
- **Commands:** Make and a shell. On Windows, use a Make installation with a
  compatible shell, or the Python commands listed below.
- **Manuscript:** LaTeX, `latexmk`, and BibTeX, including the packages named in
  `manuscript/main.tex` and the `iopart-num` bibliography style. TeX Live/MacTeX
  provide these; a minimal TeX installation may need additional packages.
- **Optional tools:** Node 22+ for website tests and assembly. Desktop-viewer
  dependencies are installed separately; see [the viewer guide](interactive.md).
- **Network:** installation, the initial Figure 4 download, and the reported-value
  emitter's lookup of the cited `statespacecheck` DOI on Zenodo need internet.
  Building the manuscript from committed figures/macros can run offline.

| Work | Approximate cost |
| --- | --- |
| Figures 1–2 | Small simulations and figure rendering |
| Figure 3 | About 5 minutes for its 100 realizations in the reviewed macOS ARM64 environment, plus rendering |
| Figure 4 input | 75 MB download, verified by SHA-256 |
| Figure 4 fresh fit/decode | Several minutes per model, followed by diagnostics and rendering; runtime depends on hardware |
| Figure 4 cached decode | Approximately 8 GB on disk; arrays are memory-mapped when loaded |

Peak RAM for a fresh Figure 4 run has not been benchmarked; the disk-cache size
is not a RAM requirement. Cache replacement is atomic and temporarily keeps the
old and new decode bundles, so allow roughly 16 GB for those two files when
refitting, plus space for inputs, diagnostics, dependencies, and optional viewer
caches. A source change in the fitting/data-preparation modules invalidates the
decode cache; see [cache behavior](figure-pipeline.md#figure-4--real-data-decoder-diagnostics).

## Build from committed artifacts

```bash
make manuscript
```

This invokes `make -C manuscript` and produces `manuscript/main.pdf` using the
committed figure PDFs, bibliography, and `reported_values.tex`. Python and the
recording are unnecessary for this step. See
[manuscript/README.md](../manuscript/README.md) for direct LaTeX commands.

## Reproduce Figures 1–3

```bash
make sync
make figures-simulated
make reported-values
make manuscript
```

Figures 1–3 use seeded simulations. The manuscript build also uses the committed
Figure 4 assets. Figure 3's summary runs 100 realizations (seeds 1–100) and reports the
median across realizations of the per-condition flag percentages and decoding errors.

## Reproduce the full paper

```bash
make reproduce
```

The target performs the following steps in order, including under `make -j`:

```bash
uv sync --frozen
uv run --frozen python scripts/download_figure04_inputs.py
uv run --frozen python scripts/generate_all_figures.py
uv run --frozen python scripts/emit_reported_values.py
make -C manuscript
```

The download saves `{epoch}_figure04_inputs.npz` only if its SHA-256 matches the
published value. It reuses an existing matching file and refuses to overwrite
a different one. The dataset is archived at
[doi:10.5281/zenodo.23020757](https://doi.org/10.5281/zenodo.23020757).

To store inputs and caches elsewhere, set `STATESPACECHECK_DATA_PATH` for the
whole workflow:

```bash
STATESPACECHECK_DATA_PATH=/path/to/paper-data make reproduce
```

The canonical epoch is `j1620210710_02_r1`. Public reproduction needs the archived
input file and no lab database connection. Lab acquisition and export procedures
are documented separately in [data lineage](data-lineage.md) and the
[Spyglass pipeline](spyglass-pipeline.md).

## Individual steps

| Command | Result |
| --- | --- |
| `make download-data` | Download and verify the Figure 4 input |
| `make figures-simulated` | Regenerate Figures 1–3 and the Figure 3 summary |
| `make figures` | Regenerate all four figures and both summaries; input must exist |
| `make reported-values` | Regenerate the manuscript's numerical macros from both summaries |
| `make manuscript` | Build the PDF from the current figures and macros |

An individual figure can be regenerated with
`uv run --frozen python scripts/generate_figure03.py` (substitute `01`, `02`, or
`04` as needed). To refit Figure 4 regardless of its cache, use:

```bash
uv run --frozen python scripts/generate_figure04.py --force-recompute
```

After changing either summary, regenerate the macros before building the PDF.
`make manuscript` tracks the macro file but does not regenerate it. To update
the website after a reproduction run, follow [site/README.md](../site/README.md).

## Outputs and checks

| Committed artifact | Generator | Purpose |
| --- | --- | --- |
| `manuscript/figures/main/figureNN.pdf` and `.png` | `scripts/generate_figureNN.py` | Publication figure and preview |
| `figure03_summary.json` and `figure04_summary.json` in that directory | Figure 3/4 generators | Full-precision statistics, configuration, thresholds, provenance |
| `manuscript/reported_values.tex` | `scripts/emit_reported_values.py` | Numbers used in the manuscript |
| `manuscript/main.pdf` | `make manuscript` | Readable paper |
| `site/data/*.json` and the metric parity fixture | `scripts/export_site_data.py` | Website displays, numbers, and Python/JavaScript agreement checks |

These outputs are committed so readers can inspect the results and build the
paper before rerunning the analyses. Recording inputs, decode/viewer caches,
and temporary build files are ignored by Git.

After installing the development dependencies, run the artifact checks:

```bash
make sync-dev
uv run --frozen pytest tests/test_reported_statistics_artifacts.py tests/test_reported_values.py tests/test_site_export.py
```

These checks validate the committed reference statistics, schemas, macros, and
website data. They do not independently rerun the full analyses. To check a
reproduction, also inspect the generated summary diff against the committed
version: counts and statistics should match; changes in configuration or source
must be explained by provenance. PDF timestamps and small rendering differences
can change binary files even when the scientific outputs agree.

The figure-to-code map and summary schema are in [figure-pipeline.md](figure-pipeline.md).
Code-change validation and artifact refresh rules are in [development.md](development.md).
