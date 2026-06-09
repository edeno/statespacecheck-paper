# Manuscript TODO

Outstanding follow-ups for **"Local goodness-of-fit measures for neural decoding"**
(`manuscript/main.tex`). Tracked items from the main.tex reconciliation, the base-e
log switch, and the best-practices / grammar-and-argument reviews.

## Required (need data or external facts)

- [ ] **1. Regenerate Figure 4.** The figure code is now base-e (`-log(p)`, natural
  log), but the committed image `manuscript/figures/main/figure04.{pdf,png}` was last
  built under base-10 and still shows `-log10(p)`. Regenerate on a machine with the
  hippocampal data:
  ```bash
  uv run python scripts/generate_figure04.py
  ```
  (Set `STATESPACECHECK_DATA_PATH` / `STATESPACECHECK_ANIMAL_DATE_EPOCH` if not default.)
  Until then the Figure 4 image is inconsistent with its caption.

- [ ] **2. Add two missing citations** to `manuscript/references.bib`: `wilson1994`
  and `karlsson2009` (cited in the Data-analysis-results subsection for immobility
  replay; currently render as `??`).

- [ ] **3. Verify the real-data counts.** Fig 4 caption says "203 simultaneously
  recorded cells"; the Methods text says "Twenty-three tetrodes." Confirm the two are
  mutually consistent and that 203 is the actual loaded / post-curation cell count.
  Surface the cell count in the Methods text (currently only in the caption).

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

- [ ] **13. Equation (4) notation.**
  - Make the sampling distribution explicit in `\Pr`, e.g.
    `\Pr_{\widetilde{y} \sim f_{\mathrm{pred}}}(\cdots)`, so the probability's sample
    space is unambiguous.
  - Consider renaming the predictive-check statistic `D_{k,j}` to avoid the letter
    clash with the KL divergence `D_{\mathrm{KL}}` (it is interpreted as a p-value).

## Notes (verified, not yet edited in prose)

- **Threshold description (red-draft sentence in Simulation Results).** Verified
  against `compute_thresholds` (`src/statespacecheck_paper/analysis.py`): the current
  wording "empirical 99% interval for that metric" is inaccurate. The code uses:
  HPD overlap = 1st percentile of baseline (flag if ≤), KL divergence = 99th
  percentile of baseline (flag if ≥), predictive p-value = **fixed 0.05 cutoff**
  (flag if ≤, not data-derived). In Figure 3 the baseline is the opening baseline
  window only (steps 0–6000), not all recovery windows. Use accurate per-metric
  wording when finalizing that red-draft sentence.
