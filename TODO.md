# Manuscript TODO

Open follow-ups for **"Local goodness-of-fit measures for neural decoding"**
(`manuscript/main.tex`). Completed work is in git history; this lists only what's left.

## Red placeholders in main.tex

- [ ] **Archival DOI** — fill `\textcolor{red}{[TODO: Add the archival DOI (e.g., Zenodo) ...]}` in Data/Code Availability (`main.tex:351`).
- [ ] **Funding** — fill the remaining `\textcolor{red}{[TODO: add funding for U.T. Eden and any support for S. Zeng and E.L. Denovellis]}` in Acknowledgments (`main.tex:357`). NIH/SCGB/HHMI (L.M.F.) and NSF/NIH/UCSF (A.E.C.) are already filled in.
- [x] **MountainSort4 citation** — add Chung et al. 2017, *Neuron* 95(6):1381–1394 to the bibliography; replace the red `[cite: …]` in §4.1 with `\citep{...}`.

## Citations

> The bibliography (`manuscript/Local-GoF-Paper.bib`) was re-exported with
> descriptive Better BibTeX keys; the old `RefYYYY[a-z]` keys below no longer
> exist. Open items are re-anchored to the current keys.

- [x] **Intro examples (line ~92) — verify/swap.** Draft-import placeholders (former `Ref2015a`/`Ref2022b`/`Ref2018`/`Ref2025a`, commit `789e26f`) reconciled during the re-export.
- [x] **Bayarri & Castellanos 2007 metadata** — `bayarriBayesianCheckingSecond2007` now carries *Statist. Sci.* 22(3), 2007; verified against the re-exported entry.
- [ ] **Divergence-naming cites** — reassess whether `bhattacharyyaTotalVariationDistance2024` (TV) and `crackNoteKarlPearsons2018` (Pearson χ²) are strong enough authorities. Note: Cover & Thomas is **not** currently in the bib (only `kullbackInformationSufficiency1951` and `Hastieelementsstatisticallearning2009`); add it if that's the intended reference.

## Bibliography cleanup

- [x] Fix the series-as-journal entry type (`@article` → `@book`/`@incollection`).
- [x] Prune uncited entries and near-duplicate keys during the re-export.


## Cover Letter
- [ ] Create an overleaf cover letter

## Possible journals

*PLOS Comp Bio* ·  Neural Comp · IEEE Biomedical Engineering  · NBDT · Journal of Neural Engineering  ·  J Neurosci Methods
