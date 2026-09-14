# Manuscript TODO

Open follow-ups for **"Local goodness-of-fit measures for neural decoding"**
(`manuscript/main.tex`). Completed work is in git history; this lists only what's left.

## Red placeholders in main.tex

- [ ] **Archival DOI** — replace the red DOI placeholder in the Data and Code Availability section of `manuscript/main.tex` with the DOI for the tagged release used in the paper.

## Figure 3

- [ ] **Show per-realization distributions** — add a distribution plot of the flag percentages to show trajectory dependence, especially for remapping. The JSON's approximate SEs describe uncertainty in the aggregated medians, not the spread of individual realizations. Any claim comparing metrics should use their paired realization results. Deferred from PR #10.

## Citations

- [ ] **Divergence-naming cites** — reassess whether `bhattacharyyaTotalVariationDistance2024` (TV) and `crackNoteKarlPearsons2018` (Pearson χ²) are strong enough authorities. Note: Cover & Thomas is **not** currently in the bib (only `kullbackInformationSufficiency1951` and `Hastieelementsstatisticallearning2009`); add it if that's the intended reference.
