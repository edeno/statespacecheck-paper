# Publication follow-ups

## Release and archival records

- [ ] Tag the exact analysis release used for submission and record its
  version-specific Zenodo DOI. `CITATION.cff` currently points to the concept DOI
  that groups all releases; the old manuscript DOI placeholder has been replaced.
- [ ] Publish a version of DANDI dandiset 001942 and update the data statement
  and Zenodo input record with its DOI. See [data lineage](docs/data-lineage.md).
- [ ] Complete the lab's Spyglass pipeline/export and public sorting archive,
  following the blockers and approval requirements in
  [the lab guide](docs/spyglass-pipeline.md).
- [ ] Upload the Spyglass export for this paper, including the HPC sorting used in
  Figure 4, to DANDI, and cite it in the data statement.
- [ ] Release and archive `non_local_detector` at the commit `uv.lock` pins
  (`956fdcc`; its last release is v0.6.9) on PyPI or Zenodo, then pin the
  release. Figure 4 cannot be rebuilt if that commit becomes unavailable.

## Scientific follow-ups

- [ ] Show the variability of Figure 3's per-realization results, which the summary
  now keeps (`realization_flag_percentages`, `realization_decoding_error`).
- [ ] Evaluate sensitivity to simulation parameters and diagnostic thresholds.

These are separate analysis tasks; they are not prerequisites for reproducing
the current paper from its archived input and committed configuration.
