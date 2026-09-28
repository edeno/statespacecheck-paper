# Local goodness-of-fit measures for neural decoding

Source code, analyses, and figures for the paper by Sirui Zeng, Alison E. Comrie,
Loren M. Frank, Uri T. Eden, and Eric L. Denovellis. The paper compares one-step
state predictions with the information in individual spikes using HPD overlap,
a rank-based predictive check, and KL divergence.

- **Read:** [manuscript PDF](manuscript/main.pdf)
- **Explore:** [interactive project website](https://edeno.github.io/statespacecheck-paper/)
- **Reproduce:** [complete instructions](docs/reproduce.md)
- **Apply the diagnostics:** [`statespacecheck` documentation](https://edeno.github.io/statespacecheck/)

## Build the manuscript

The figures and reported-value macros are committed. With LaTeX and `latexmk`
installed, run from the repository root:

```bash
make manuscript
```

This builds `manuscript/main.pdf` without Python or the recording data.

## Reproduce the figures

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and Make.
All commands below run from the repository root. The locked analysis environment
uses Python 3.11; uv can install it if needed. CI runs the tests on Linux, macOS,
and Windows. The reported Figure 4 values were verified on macOS (Apple silicon)
with Python 3.11; Intel Macs cannot run Figure 4 because the locked JAX release
has no build for them.

### Figures 1–3: self-contained simulations

```bash
make sync
make figures-simulated
make reported-values  # needs internet to look up the cited package DOI
make manuscript
```

### All four figures and the manuscript

```bash
make reproduce
```

This installs the locked environment, downloads and verifies the 75 MB Figure 4
input, generates all four figures, updates reported values, and builds the PDF.
Figure 4 creates a decode cache of approximately 8 GB. See the
[requirements, timings, and individual commands](docs/reproduce.md) before running.

The Figure 4 input is archived at
[doi:10.5281/zenodo.23020757](https://doi.org/10.5281/zenodo.23020757).
Public reproduction uses that input file; its origin, preprocessing, and the
availability of the underlying recordings are documented in
[data lineage](docs/data-lineage.md).

## Find your way around

| Location | Contents |
| --- | --- |
| `manuscript/` | LaTeX source, compiled paper, figures, and analysis summaries |
| `scripts/` | Commands for generating figures, downloading data, and maintaining artifacts |
| `src/statespacecheck_paper/` | Simulation, decoding, analysis, and plotting recipes |
| `tests/` | Scientific contracts, artifact checks, and software tests |
| `site/` | Interactive website and its committed data |

| Guide | Purpose |
| --- | --- |
| [Reproduction](docs/reproduce.md) | Requirements, commands, outputs, and verification |
| [Figure pipeline](docs/figure-pipeline.md) | Trace each figure from manuscript to configuration and code |
| [Data lineage](docs/data-lineage.md) | Data sources, preprocessing, and checksums |
| [Desktop viewer](docs/interactive.md) | Install, prepare caches, and inspect individual spikes |
| [Website](site/README.md) | Export data, test, preview, and deploy the site |
| [Development](docs/development.md) | Code conventions, checks, and artifact maintenance |
| [Spyglass pipeline](docs/spyglass-pipeline.md) | Lab-specific database and export operations |

Run `make help` for the root commands. To work on the code:

```bash
make sync-dev
make check  # Python checks and website tests; requires Node 22+
```

## Use the diagnostics in your own work

This repository contains the experiments for the paper. The reusable diagnostics
are developed in the separate [`statespacecheck`](https://github.com/edeno/statespacecheck)
package and installed here as a dependency. To apply them to your own decoder,
use that package directly: `statespacecheck.event_diagnostics` takes a one-step
predictive distribution, per-unit intensities, and each spike's time bin and unit,
and returns all three diagnostics per spike. General computations (the per-spike
diagnostics, HPD regions, the Monte Carlo predictive p-value, threshold
estimation) belong there; this repository holds what is specific to the paper
(the simulations, the real-data decoding, the threshold choices, figures,
reported values, the viewer, and the website).

## Citation and licensing

Use [CITATION.cff](CITATION.cff) for the authors and citation metadata. The analysis
code is archived on [Zenodo](https://doi.org/10.5281/zenodo.23019296).

Code is licensed under [MIT](LICENSE). Manuscript text and figures are licensed
under [CC BY 4.0](manuscript/LICENSE). The Figure 4 input file on Zenodo is
licensed under CC BY 4.0.

## Questions and problems

Open an [issue](https://github.com/edeno/statespacecheck-paper/issues) or contact
Eric Denovellis (eric.denovellis@ucsf.edu).
