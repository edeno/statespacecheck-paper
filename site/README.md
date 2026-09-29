# Project website

Run the commands below from the repository root. Node 22+ (`engines` in
`site/package.json`) is required for assembly and tests; the preview server
also needs `python3`.

The site uses plain HTML, CSS, and JavaScript with a small Make/Node assembly step.
It has four interactive explainers: a time stepper through a short spike train decoded
by the paper's Bayesian filter, a playground that recomputes the three
diagnostics as the reader moves a prediction, a player for the Figure-3
simulation conditions, and the Figure-4 recording window with a switch between
its two decoders. The recording player holds the selected spike fixed while the
model changes and shows that unit's place field from the shared observation model.

- The players and every number in the page text come from the paper's pipeline
  via `statespacecheck_paper.site_export`, which writes `site/data/*.json`. The
  numbers are the same reported-value macros the manuscript uses, plus one
  that only the page states (`site_export.page_values`: the length of the
  Figure-4 window). The flag thresholds beside each readout are printed by the
  manuscript's rounding policy (`site_export.flag_threshold_text`).
- The playground runs a JavaScript port of the per-spike diagnostics
  (`site/js/metrics.js`). `site/tests/metrics.test.mjs` checks it against
  reference cases computed by the Python implementation. Its example buttons
  load `site_export.PLAYGROUND_PRESETS`; `tests/test_site_export.py` checks
  that each flags what its label claims.

```bash
# Regenerate the page data after a figure summary or a diagnostic changes.
# The Figure-4 window needs the Figure-4 input file. It reuses the Figure-4
# caches when current; otherwise it refits both models (several minutes) and
# writes the ~8 GB decode cache. Add --skip-recording to leave recording.json
# untouched when the recording outputs are unchanged.
uv run --frozen python scripts/export_site_data.py

# Check the JavaScript diagnostics against the Python reference
make check-site

# Check the assembled page's accessibility and keyboard behavior in headless
# Chromium (installs the test dependencies and the browser once)
npm --prefix site ci
(cd site && npx playwright install chromium)
make check-site-browser

# Assemble the site (adds Figure 1 and the paper PDF, and writes the reported
# numbers into the HTML) and preview it locally
make -C site serve   # http://localhost:8000
# If `node` is not on make's PATH (e.g., nvm loads lazily), pass the binary:
make -C site serve NODE=/path/to/node
```

The browser tests (`site/tests/browser/`) build the site, wait without
scrolling until every section has loaded, and run axe-core's WCAG 2.2 A/AA
rules in each interactive state: the loaded page at desktop and 320-pixel
widths, each simulated condition, each playground example, and the filter
explainer while playing. They also check the keyboard and screen-reader
semantics that axe cannot: the skip link, the condition tabs (arrow keys move
focus; Enter or Space selects), the playground's native cell radios, the time
tracks as sliders whose value text names the selected spike, the per-player
spike tables, table headers, and the chart descriptions. Automated checks catch
regressions; they do not replace keyboard and screen-reader review.

Every push to `main` deploys the site through `.github/workflows/pages.yml` once
CI (including the website's staleness tests) passes on that commit; a CI run
that finishes after `main` has moved on is not deployed. The
repository's Pages source must be set to **GitHub Actions** (Settings → Pages).

The assembled `site/_build/` directory is ignored by Git. Run
`make -C site clean` to remove it. The exported data and parity fixture are
committed publication artifacts; see the [artifact table](../docs/reproduce.md#outputs-and-checks).
