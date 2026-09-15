# Manuscript TODO

Open follow-ups for **"Local goodness-of-fit measures for neural decoding"**
(`manuscript/main.tex`). Completed work is in git history; this lists only what's left.

## Scientific and manuscript revision

The [revision plan](docs/manuscript-fix-plan.md) is the detailed checklist and
records the status of each item. Real-data models are fitted and checked using
the full recording. The branch `manuscript/scientific-revision` implements
items 1–5 of the plan; what remains is listed here.

- [x] Correct verified implementation errors and clarify initialization, event-binning, and history-process contracts.
- [x] Add full-recording broadening controls, HPD-region sizes, coverage sensitivity, and rate-stratified event summaries (`scripts/generate_figure04_supplement.py`).
- [x] Add matched simulation references and controls, independent Monte Carlo calibration, and posterior uncertainty assessments (`scripts/generate_figure03.py`, `scripts/generate_figure03_sensitivity.py`).
- [x] Complete the first manuscript pass for mathematical corrections, notation, literature positioning, and interpretations supported by existing evidence.
- [x] Integrate final simulation and recording results throughout the Abstract, Methods, Results, Discussion, all captions, and supporting documentation.
- [x] Regenerate summaries, reported-value macros, figures, and viewer artifacts; compile and inspect the complete manuscript.
- [ ] Complete input-export reproducibility and release provenance: the raw-recording → five-export step is still external to the repository (see `docs/figure-pipeline.md`); either archive the five exact exports with the checksums recorded in `figure04_summary.json` or provide an executable export procedure tied to the DANDI assets, sorting/curation settings, and track definitions.
- [ ] Conditional extensions the plan leaves open if broader claims are reinstated: additional transition models (e.g. a uniform-jump control `T_w = (1-w) T_C + w U 1^T` fitted on the full recording), a broader parameter sweep (rates, field widths, neuron counts, several remappings, milder drift), a systematic runtime study, additional recordings, curated-unit / split-merge sensitivity, and a parametric-bootstrap (simulate-refit) calibration of the fitted real-data procedure. The manuscript currently claims none of these.

## Red placeholders in main.tex

- [ ] **Archival DOI** — replace the red DOI placeholder in the Data and Code Availability section of `manuscript/main.tex` with the DOI for the tagged release used in the paper.

## Figure 3

- [x] **Show per-realization distributions** — panel (c) now shows every realization's flag percentage per condition with the median marked; event counts per column are in `figure03_summary.json`.

## Citations

- [x] **Divergence-naming cites** — the weak TV / Pearson-χ² citations were removed; the f-divergence family now cites Liese & Vajda (2006). Overlap coefficient cites Simpson (1943); calibration vs sharpness cites Gneiting, Balabdaoui & Raftery (2007).
