# Manuscript TODO

Open follow-ups for **"Local goodness-of-fit measures for neural decoding"**
(`manuscript/main.tex`). Completed work is in git history; this lists only what's left.

## Red placeholders in main.tex

- [ ] **Funding** — fill `\textcolor{red}{[TODO: funding sources / grant numbers]}` in Acknowledgments.
- [ ] **MountainSort4 citation** — add Chung et al. 2017, *Neuron* 95(6):1381–1394 to `references.bib`; replace the red `[cite: …]` in §4.1 with `\citep{...}`.

## Citations

- [ ] **Intro examples (line ~92) — verify/swap.** `Ref2015a` ("emotional movement") is a clusterless position-decoding paper, likely meant to be `Ref2024a`; `Ref2022b` ("locally anchored map") and `Ref2018`+`Ref2025a` ("affective trajectories and LFP") also read off. (All from draft-import commit `789e26f`.)
- [ ] **`Ref2007` metadata wrong** — fix to *Statistical Science* 22(3):322–343, 2007 (DOI 10.1214/07-STS235).
- [ ] **Divergence-naming cites** — `Ref2023b` (TV) and `Ref2018d` (Pearson χ²) are weak authorities; consider Cover & Thomas (`Ref1999a`, already in bib).

## Bibliography cleanup

- [ ] Fix `Ref2014c` entry type (`@article` with series-as-journal → `@book`/`@incollection`).
- [ ] Prune uncited entries and near-duplicate keys (`Ref2014c_archer`, `Ref2024a`/`Ref2024a_chu`).

## Prose (optional)

- [ ] Discussion: "HPD overlap … consistently maintaining lower error rates across various conditions" overstates — no formal error-rate comparison was run.
- [ ] Discussion closing ("empower us … deepen our understanding") is generic.

## Possible journals

PLOS Comp Bio · NBDT · Neural Comp · IEEE Biomedical Engineering
