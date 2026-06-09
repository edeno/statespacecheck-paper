# Manuscript TODO

Outstanding follow-ups for **"Local goodness-of-fit measures for neural decoding"**
(`manuscript/main.tex`). Tracked items from the main.tex reconciliation, the base-e
log switch, and the best-practices / grammar-and-argument reviews.

## Required (need data or external facts)

- [x] **1. Regenerate Figure 4.** Done. Regenerated `figure04.{pdf,png}` with the
  base-e `-log(p)` axis (both decoders re-fit from the local data in `data/`) and
  rebuilt `main.pdf`. The figure now matches its caption; build is clean (0 undefined
  citations/refs). Verified visually that the diagnostic rows and the panel-(e) hexbin
  read `-log(p)` rather than `-log10(p)`.

- [x] **2. Add two missing citations** to `manuscript/references.bib`: `wilson1994`
  and `karlsson2009` (cited in the Data-analysis-results subsection for immobility
  replay). Done in commit `934643f`; the build now resolves both (0 undefined
  citations).

- [ ] **3. Verify the real-data counts.** Partially verified.
  - **203 cells: confirmed.** The data file has exactly 203 units and the Figure 4
    pipeline loads/decodes all of them unfiltered (`generate_figure04.py:107`), so
    "203 simultaneously recorded cells" matches what is analyzed. Note: 21 of the 203
    are silent (0 spikes); only 182 actually fire. Decide whether to keep "203" (all
    sorted units) or report "182 active" — optional.
  - **23 tetrodes: not verifiable locally** (the spike-times file carries no tetrode
    labels). Internally consistent (203/23 ≈ 8.8 units/tetrode); confirm the exact
    number against the Comrie2024 methods.
  - Still TODO: surface the cell count in the Methods text (currently only in the
    Fig 4 caption).

## Red TODO placeholders left in main.tex

- [ ] **4. Spike-sorting methods.** Resolve `\textcolor{red}{[Add spike sorting
  information.]}` in the "Data collection and behavioral task" subsection.

- [ ] **6. Funding.** Resolve `\textcolor{red}{[TODO: funding sources / grant
  numbers]}` in the Acknowledgments.

## Citations / notation polish

- [ ] **12. Citation vetting.**
  - Verify `Ref2018` at the replay claim ("hippocampal replay ... not directly
    observable") points to the intended paper — flagged as a possible mismatch.
  - The total-variation (`Ref2023b`) and Pearson-$\chi^2$ (`Ref2018d`) divergence
    citations are weak authorities for naming standard divergences; consider a
    standard reference (e.g. Cover & Thomas, `Ref1999a`, currently unused in the bib).

- [x] **13. Equation (4) notation.** Done. The sampling distribution is now explicit
  (`\Pr_{\widetilde{y}_{k,j} \sim f_{\mathrm{pred}}}`), and the predictive-check
  statistic is renamed `D_{k,j}` → `p_{k,j}` (it is a p-value, avoiding the clash
  with `D_{\mathrm{KL}}`).

## Done

- [x] **Threshold description (Simulation Results).** Resolved. Verified against
  `compute_thresholds` (`src/statespacecheck_paper/analysis.py`) and rewrote the prose
  (removing the red draft): HPD overlap = 1st percentile of the initial baseline
  (flag if ≤), KL divergence = 99th percentile of the same baseline (flag if ≥),
  predictive p-value = fixed 0.05 cutoff (flag if ≤, not data-derived). Also fixed the
  Figure 3b "empirical 99% interval" sentence. Removed the inaccurate blue/red
  spike-coloring claim — Figure 3 plots single-color scatter with a horizontal
  threshold line.
