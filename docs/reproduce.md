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
- **Network:** installation and the initial Figure 4 download need internet.
  Generating figures, emitting the reported-value macros (the cited
  `statespacecheck` DOI is committed in `manuscript/software_dois.json`), and
  building the manuscript run offline.

| Work | Approximate cost |
| --- | --- |
| Figures 1–2 | Small simulations and figure rendering |
| Figure 3 | About 5 minutes for its 100 realizations in the reviewed macOS ARM64 environment, plus rendering |
| Figure 4 input | 75 MB download, verified by SHA-256 |
| Figure 4 fresh fit/decode | Several minutes per model, followed by diagnostics and rendering; runtime depends on hardware |
| Figure 4 cached decode | Approximately 8 GB on disk; arrays are memory-mapped when loaded |

Peak RAM for a fresh Figure 4 run has not been benchmarked; the disk-cache size
is not a RAM requirement. Cache replacement is atomic and temporarily keeps the
old and new decode caches, so allow roughly 16 GB for those two files when
refitting, plus space for inputs, diagnostics, dependencies, and optional viewer
caches. A source change in the fitting/data-preparation modules invalidates the
decode cache, as do a different Python version, machine architecture, or
installed version of any runtime dependency of `non_local_detector`; see
[cache behavior](figure-pipeline.md#figure-4-cache-behavior).

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
uv sync --frozen --inexact
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

## Check a fresh reproduction

```bash
make reproduce-fresh                              # work in a new temporary directory
make reproduce-fresh FRESH_DIR=/path/to/empty-dir  # or in a chosen empty directory
```

`scripts/reproduce_fresh.py` reruns the committed paper from scratch without
touching the checkout. It exports the committed tree (`git archive HEAD`, so
uncommitted changes are not included) into `<work dir>/repo`, installs that
copy's locked environment, and points `STATESPACECHECK_DATA_PATH` at the empty
`<work dir>/data`, so no existing input file or cache is reused. In the copy it
downloads and verifies the Figure 4 input, generates Figures 1–4 and both
summaries, emits the macros (offline), and builds the manuscript with latexmk.
It then runs `scripts/check_reproduction.py` on the fresh summaries against
HEAD's and compares the fresh `reported_values.tex` with HEAD's byte for byte.
The target exits nonzero on any difference.

The work directory keeps `fresh/` (figures, summaries, macros, and PDF),
`committed/` (HEAD's summaries and macros), `reproduction_report.txt`, and
`steps.tsv` (seconds and the largest child-process resident memory for each
step). Running in a copy, rather than regenerating the committed files in place
and restoring them, leaves a working tree with local edits untouched and needs no
cleanup.

The `Reproduce` GitHub Actions workflow (`.github/workflows/reproduce.yml`) runs
this target on Ubuntu with Python 3.11 monthly, on each published release, on
demand, and on pull requests that change the workflow. It installs the TeX Live
collections the manuscript needs, frees runner disk space for the decode cache,
and uploads the report, step timings, fresh summaries, and fresh macros.

Measured fresh runs (`steps.tsv`; peak memory is the largest child process):

| Resource | macOS, Apple M5 Max, 18 cores, 64 GB |
| --- | --- |
| Peak RAM | 13.8 GB, during the Figure 4 fit and decode |
| Disk | 7.9 GB of data (7.7 GB decode cache, 75 MB input), plus 0.6 GB for the copy and its environment |
| Time | 7 minutes: figures 385 s (Figure 3 about 260 s, the Figure 4 fit about 140 s), download 27 s, manuscript 3 s |

Individual steps on the same machine, with current Figure 4 caches: Figures 1
and 2 take about 1 s each (under 0.6 GB); Figure 4 from its caches 8 s (4.7 GB);
the website export 5 s (3.3 GB).

## Individual steps

| Command | Result |
| --- | --- |
| `make download-data` | Download and verify the Figure 4 input |
| `make figures-simulated` | Regenerate Figures 1–3 and the Figure 3 summary |
| `make figures` | Regenerate all four figures and both summaries; input must exist |
| `make reported-values` | Regenerate the manuscript's numerical macros from both summaries (offline) |
| `make manuscript` | Build the PDF from the current figures and macros |
| `make reproduce-fresh` | Rerun HEAD in a clean copy with empty caches and compare with the committed results |

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
reproduction, compare the regenerated summaries with the committed ones:

```bash
uv run --frozen python scripts/check_reproduction.py FRESH/figure03_summary.json FRESH/figure04_summary.json
```

It requires exact equality for integers, strings, booleans, seeds,
configuration, counts, and the source and `uv.lock` digests, and equality within
the tolerances defined at the top of the script (`RTOL`, `ATOL`) for
floating-point statistics. It skips only the provenance entries that describe
the machine rather than the result (`MACHINE_SPECIFIC_PATHS`: the Figure-4 cache
fingerprints, Python version, architecture, and dependency versions), prints
every difference with its JSON path, and exits nonzero if there is any.
`--committed-dir` compares against summaries elsewhere than
`manuscript/figures/main/`. PDF timestamps and small rendering differences can
change binary files even when the scientific outputs agree.

The figure-to-code map and summary schema are in [figure-pipeline.md](figure-pipeline.md).
Code-change validation and artifact refresh rules are in [development.md](development.md).
