# Development guide

This repository holds the experiments and publication artifacts for **Local
goodness-of-fit measures for neural decoding**. Start with
[reproduction](reproduce.md) to run the paper, or the
[figure pipeline](figure-pipeline.md) to trace a result into its implementation.

## Environment and checks

Use uv and the repository's `.venv` for package management. Install the locked
development and viewer dependencies with:

```bash
make sync-dev
make check
```

`make check` checks formatting, lint, strict types, the default Python test suite,
and website metric parity. The website tests need Node 22+; run
`make check-python` or `make check-site` separately when working on one component.
An explicit Node executable can be passed as `make check NODE=/path/to/node`.

For focused checks or formatting:

```bash
uv run --frozen pytest tests/test_simulation.py
uv run --frozen ruff format .
uv run --frozen ruff check .
uv run --frozen mypy src/
```

The default pytest run excludes `slow` tests. Run them deliberately with
`uv run --frozen pytest -m slow`; the real-data viewer-cache test reads
multi-gigabyte files and builds another cache. Coverage reports are written to
`htmlcov/` by the default suite. Use the current report for coverage totals.
Run formatting, lint, type checks, and tests before committing.

### Dependency changes

`uv.lock` pins dependencies, including the Git revision of `non_local_detector`.
To update a dependency intentionally:

```bash
uv lock --upgrade-package PACKAGE_NAME
uv sync --frozen --extra dev --extra interactive
```

Review the lockfile diff and reproduce affected analyses. The configured
`allow-direct-references` setting supports the existing Git dependency; ordinary
reproduction requires no metadata edits. CI sets `UV_PYTHON` for each matrix job
so the local Python 3.11 pin does not override the requested test version.

## Code organization

Reusable implementation lives in `src/statespacecheck_paper/`. Scripts are thin
CLI adapters to importable figure recipes. Keep simulation, analysis, plotting,
and orchestration separate; the dependency graph is tested for cycles.

| Work | Location |
| --- | --- |
| General simulation primitives | `simulation.py` |
| General Bayesian filter | `decoding.py` |
| Paper diagnostic containers and threshold choices | `diagnostics.py` |
| Figure 3 protocol, simulation, summary, rendering | `figure03_*` |
| Figure 4 loading, fitting, diagnostics, caches, rendering | `load_local_data.py`, `write_local_data.py`, `figure04_*` |
| Shared plotting and appearance | `plotting.py`, `style.py`, `schematic.py` |
| Summary-to-prose reporting | `reported_values.py`, `number_format.py` |
| Website export | `site_export.py` |

The general diagnostic computations belong to the separate `statespacecheck`
package. Change them there, release/update that dependency, and regenerate this
paper's affected outputs. `diagnostics.py` imports no sibling paper module.

New figure scripts should call an importable generation recipe, save to
`manuscript/figures/main/` or `supplementary/`, and have an integration check in
`tests/test_figures.py`. Keep utilities in the package and scripts under 200 lines.
Use dataclasses for configurations and scientific result containers.

## Scientific and coding conventions

- Use `np.random.default_rng` and pass an RNG or `SeedSequence` through simulations.
  Preserve draw order when preserving an existing seeded experiment.
- Arrays have time first: spatial distributions are `(n_time, n_position_bins)`
  or `(n_time, n_x_bins, n_y_bins)`; spike counts are `(n_time, n_cells)`;
  the simulated rate table is `(n_bins, n_cells)`.
- Predictions mean `p(x_t | y_{1:t-1})`, filtering means `p(x_t | y_{1:t})`, and
  smoothing means `p(x_t | y_{1:T})`. Keep them distinct in code and prose.
- Manuscript rates are `lambda`; expected counts are `lambda * dt`. Simulation
  Poisson inputs already contain counts per step, even where named `rate`.
  Do not multiply them by `dt` again.
- Handle invalid spatial bins and NaNs explicitly. Vectorize independent
  operations; sequential filtering recursions require their time loop.
- Use `style.py` for fonts, the colorblind-friendly palette, and the
  `METRIC_SPECS` registry (each metric's color, display transform, and label).
  Export both PDF and PNG; the canonical generators use 450 DPI.
- Add full type hints (`NDArray[np.float64]` for arrays) and NumPy-style
  docstrings with array shapes. Strict mypy must pass on Python 3.11, the
  development pin; CI type-checks there only, because the lock installs numpy 2.2
  on Python 3.10, whose stubs disagree with 2.3's. Fix type errors rather than
  adding `# type: ignore`.
- Plotting functions return Figure objects. Put shared utilities in modules, not
  in figure scripts, and do not make figure-specific copies of general helpers.
- Save intermediate results when a computation is expensive.
- Ruff handles imports, formatting, and the 100-character line length.
- Tests should use small seeded examples, cover scientific invariants and edge
  cases (empty arrays, NaN, zero sums), use `pytest.mark.parametrize` for
  scenarios, and include 1D/2D arrays where the implementation supports them.
  Aim for more than 90% coverage of the core modules (diagnostics, simulation,
  style, plotting).

## Refreshing publication artifacts

For changes to scientific code, inputs, or dependencies, regenerate the affected
figures and summaries through their canonical entry points. Compare their
statistics against the previous results, then run:

```bash
make reported-values
make manuscript
uv run --frozen python scripts/export_site_data.py
```

The macro emitter needs internet for the package DOI lookup. Website export
needs the Figure 4 inputs/cache; `--skip-recording` is appropriate only when
the recording outputs are unchanged. Figure 3 and 4 summaries must carry the
same current source provenance before emitting macros; the tests enforce this
(the emitter itself checks only that both record the same `statespacecheck`
version).

The source digest covers all Python files under `src/`, including comments and
docstrings. After a documentation-only source change, first verify that the
executable syntax trees are unchanged. Then refresh only `provenance.source`
in both summaries with `scientific_source_provenance` and `write_json_artifact`,
and run the commands above. Edits confined to `reported_values`, `site_export`,
or `figure04_download` may use this procedure while no figure imports them and
the reported values remain unchanged. Markdown-only edits do not affect the hash.

Figure 4 caches have separate decode and diagnostic fingerprints. Decode source
hashes cover fitting, recording preparation, and shared workflow/place-field code.
Any executable change to those modules refits both models, including an edited
message string. Changes confined to `diagnostics.py`, `figure04_diagnostics.py`,
or the diagnostics configuration recompute diagnostics from cached predictions. Comments and docstrings
are excluded from both cache hashes. Existing caches without the decode source
hash are rebuilt on their next use. See [the cache specification](figure-pipeline.md#figure-4--real-data-decoder-diagnostics).

Keep publication PDFs, previews, summaries, macros, and website data committed
together after a result change. Inspect binary diffs visually: PDF creation
timestamps and small renderer differences do not establish a scientific change.
The [artifact table](reproduce.md#outputs-and-checks) identifies each generator.

## Script audiences and lab operations

- **Public reproduction:** `download_figure04_inputs.py`, `generate_figure*.py`,
  `emit_reported_values.py`, and `export_site_data.py`.
- **Lab acquisition/export:** `fetch_figure04_inputs.py`,
  `spyglass_export_figure04.py`, `spyglass_pipeline_figure04.py`, and
  `datajoint_read_only.py`. See [the lab guide](spyglass-pipeline.md).
- **Legacy conversion:** `convert_figure04_pickles.py`, for the original five
  local pickles. Current reproduction uses the archived `.npz` input. Conversion
  verification and historic checksums remain in [data lineage](data-lineage.md).

Never write to the lab's Spyglass database without the user's explicit approval
of that specific step, including inserts, `populate`, schema creation, deletes,
and exports. Use `scripts/datajoint_read_only.py` for read-only checks.

`spyglass_data.py` keeps Spyglass imports inside functions. Importing
`spyglass_pipeline.py` connects to the database, so figure code and tests must
never import it. Fetches requiring analysis NWB storage must run on a lab server;
the locked Spyglass extra does not satisfy the current lab export requirements.
The lab guide records the environment, status, and blockers.
