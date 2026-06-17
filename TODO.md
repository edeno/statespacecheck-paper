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


- [x] **4. Alter Figure3b.** Done. Panel 3b is now pooled over **100 independent
  realizations** of the simulation via `estimate_stable_summary`
  (`src/statespacecheck_paper/figure03_demo.py`): the flag thresholds come from the
  baseline windows pooled across all 100 runs (KL 99th-pct dropped 4.52→4.08; the
  single-seed estimate varied ~17% seed-to-seed), and each cell reports the
  **median** percent flagged across realizations (per a request to make the
  fractions, not just the threshold, the priority). Surfaced a finding: the remap
  column is strongly trajectory-dependent (median 66–70%, but varies widely
  run-to-run); the old single-seed figure showed ~79%, near the high end. Panel 3a
  still shows the single seed-1 realization with the stabilized threshold line.
  Caption + body text (`main.tex` lines ~215/221/223) updated, incl. wide-dynamics
  KL 11%→median 14%. Figure regenerated and `main.pdf` rebuilt (0 undefined
  refs/citations).

- Work on Intro promises

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

- [ ] **14. `Ref2015a` is likely a swapped citation (High).** It is cited in the
  Introduction for "some state-space models smooth emotional movement into a
  low-dimensional trajectory," but `Ref2015a` (Deng et al., *Clusterless decoding of
  position from multiunit activity using a marked point process filter*, Neural
  Comput. 2015) is about spatial-position decoding with no affective content. The
  intended reference is probably `Ref2024a` (Vinograd et al., affective line-attractor,
  Nature 2024), already in the bib. Verify and swap.

- [ ] **15. `Ref2007` metadata is wrong** in `references.bib` (Bayarri & Castellanos,
  "Bayesian Checking of the Second Levels of Hierarchical Models"). The entry has
  `journal = {Institute of Mathematical Statistics}` and `pages = {363--367}`; the
  correct venue is *Statistical Science* 22(3):322--343, 2007 (DOI 10.1214/07-STS235).
  Fix the journal, volume/number, and pages.

- [ ] **16. Bibliography cleanup.** `Ref2014c` (Newman et al., *Modelling Population
  Dynamics*) is a Springer book but is entered as `@article` with the series name as
  the journal — fix the entry type. Also prune the ~43 uncited entries and the
  confusing near-duplicate keys (`Ref2014c` / `Ref2014b` / `Ref2014c_archer`,
  `Ref2024a` / `Ref2024a_chu`) before submission.

## Done

- [x] **Threshold description (Simulation Results).** Resolved. Verified against
  `compute_thresholds` (`src/statespacecheck_paper/analysis.py`) and rewrote the prose
  (removing the red draft): HPD overlap = 1st percentile of the initial baseline
  (flag if ≤), KL divergence = 99th percentile of the same baseline (flag if ≥),
  predictive p-value = fixed 0.05 cutoff (flag if ≤, not data-derived). Also fixed the
  Figure 3b "empirical 99% interval" sentence. Removed the inaccurate blue/red
  spike-coloring claim — Figure 3 plots single-color scatter with a horizontal
  threshold line.

errors that lead to incorrect state estimates, vs ones that show up on classic goodness of fit

## Possible journals
PLOS Comp Bio
NBDT
Neural Comp
IEEE Biomedical Engineering