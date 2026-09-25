# Figure 4 in the lab's Spyglass pipeline

Lab policy is that a paper's analyses run as Spyglass pipelines, so they can be
reproduced and exported with the lab's data. `src/statespacecheck_paper/spyglass_pipeline.py`
does this for Figure 4, and `scripts/spyglass_pipeline_figure04.py` runs it one step
at a time. The figure pipeline itself (`scripts/generate_figure04.py`) does not use
Spyglass; it reads the input file described in [data-lineage.md](data-lineage.md).
The two paths are meant to give identical results, and `--step check` compares them.

**Status (2026-09-24): code only. Nothing has been written to the lab database.**
Dry runs against the live database (read-only) resolve the position entry, the
sort's 22 keys, and both decoding parameter sets; `sorted-spikes-group` correctly
reports that the sort must be registered in `SpikeSortingOutput` first.

## Steps

Every step except `check` writes to the lab database. The script only prints what
a step would write unless given `--write`, and then asks for confirmation.

| Step | Writes | Notes |
| --- | --- | --- |
| `position-group` | `PositionGroup` `statespacecheck_figure04` | the epoch's v0 `IntervalPositionInfo` (`default_decoding`) entry; `head_position_x/y`; no upsampling |
| `spike-sorting-output` | 22 `SpikeSortingOutput.CuratedSpikeSorting` entries | registers the v0 HPC sort (not yet in the merge table); re-running skips existing entries |
| `sorted-spikes-group` | `SortedSpikesGroup` `statespacecheck_figure04` (`all_units`) | the sort's 22 merge entries |
| `decoding-parameters` | `DecodingParameters` `statespacecheck_figure04_continuous` / `_contfrag` | models from `figure04_decoder.build_decoder_models`; `decoding_kwargs` requests `filter`, `predictive_posterior`, `log_likelihood` |
| `decoding-selections` | two `SortedSpikesDecodingSelection` entries | encoding = decoding = `02_r1 noPrePostTrialTimes`; `estimate_decoding_params = 0` |
| `decode` | `SortedSpikesDecodingV1`, `DecodingOutput`, result and model files | needs the analysis store (a lab server); about 3 minutes per model on CPU |
| `diagnostics-schema` | schema `edeno_statespacecheck` and its tables | including the `Figure4DiagnosticsParameters` entry `figure04` |
| `diagnostics-selection` | `Figure4DiagnosticsSelection` | the two decodes' merge IDs |
| `diagnostics` | `Figure4Diagnostics` (+ `Mean`, `FlagConfusion`) and an analysis NWB file | per-spike diagnostics table and the Figure-4 summary |
| `check` | nothing | stored summary vs `manuscript/figures/main/figure04_summary.json` |

The diagnostics are computed by `spyglass_data.figure04_diagnostics_from_decodes`,
which applies the figure pipeline's own functions to the stored decodes and fitted
models. It also checks that the spike trains match the fitted units (count and
order, via each unit's fitted mean rate), so a unit-order mismatch between the
decode and the diagnostics raises instead of pairing spikes with the wrong place
fields.

## Configuration that reproduces Figure 4

These were established offline by emulating `SortedSpikesDecodingV1.make()` (Spyglass
`origin/master` at 0.6.0+3, 2026-09-23) on the Figure-4 input, with the paper's
`non_local_detector`:

| Variant | Figure-4 summary vs committed |
| --- | --- |
| figure pipeline, re-run | identical |
| Spyglass-matched: no upsampling, `estimate_decoding_params = 0` | identical |
| same with `PositionGroup.upsample_rate = 500` | identical (the entry is already on a 500 Hz grid) |
| `estimate_decoding_params = 1` (the Spyglass default) | **different**: EM re-fits the ContFrag transitions (`[[0.9988, 0.0012], [0.133, 0.867]]` vs `[[0.98, 0.02], [0.02, 0.98]]`); ContFrag mean HPD overlap 0.880 → 0.836, KL 2.18 → 2.85; HPD rescue rate 92% → 30% |

Storing the decodes the way Spyglass does (`save_results` / `save_model`), reloading
them, and running `figure04_diagnostics_from_decodes` also reproduces the committed
summary exactly.

## Blockers

Both concern the single-state Continuous decoder; the two-state ContFrag decoder
works. `non_local_detector` squeezes the length-1 `states` dimension out of the
results, leaving scalar `states`, `environments`, and `encoding_groups` coordinates.

1. **Spyglass `SortedSpikesDecodingV1.make()`** then fails assembling
   `discrete_state_transitions` (`ValueError: dimension 'states_from' already exists
   as a scalar variable`): it builds the `states_from` / `states_to` coordinates
   from `results.coords["states"]`, which is 0-d.
2. **`non_local_detector` `load_results`** (and so Spyglass `fetch_results`) fails
   to reload such a file: it rebuilds the `state_bins` index from every coordinate
   of `state_bins`, including the scalar ones (`PandasMultiIndex only accepts
   1-dimensional variables`). Using only the one-dimensional coordinates works.

No existing fix was found in either repository (searched 2026-09-24). Until one is
released, the Continuous decode cannot go through `SortedSpikesDecodingV1`.

## Environment

The pipeline must run with this repository's dependencies, including the locked
`non_local_detector` (the decode has to be the figure's decode) and `statespacecheck`
(the diagnostics), plus a Spyglass that (a) declares every column of the lab
database's export tables, (b) has `AnalysisNwbfile.build`, and (c) has the
single-state fix. The Spyglass in `uv.lock` (0.5.5) meets none of these, so running
the pipeline needs the locked Spyglass upgraded once a release has the fix
(check the resulting `uv.lock` diff: no package used by the figures may change).
`decode` and `diagnostics` also need the lab's analysis NWB store, i.e. a lab server.

The input-file fetch (`scripts/fetch_figure04_inputs.py`) is lighter: it runs in a
lab conda environment with the lab's Spyglass and this repository's `src/` on
`PYTHONPATH`, without this package's other dependencies.

## Next steps

Each needs the user's go-ahead before anything is written.

1. Get the single-state fix into Spyglass and `non_local_detector`, and upgrade the
   locked Spyglass.
2. Run the steps in order, dry run first, then `--step check`.
3. Log and package a Spyglass export of the Figure-4 analysis (see
   [data-lineage.md](data-lineage.md) for `scripts/spyglass_export_figure04.py`,
   which exports the input fetch; a fuller export would log fetches of
   `Figure4Diagnostics` instead), then add the sorted units to DANDI.

## Notes for working on this

- **Never write to the lab database without the user's explicit approval** of that
  specific step: inserts, `populate`, schema creation, deletes, and exports. The
  scripts dry-run by default and ask before writing.
- To check Spyglass code against the live database without writing, run it through
  `scripts/datajoint_read_only.py`: it stops DataJoint from creating schemas or tables
  and makes every insert, delete, and drop raise (including the default rows
  DataJoint inserts into `Lookup` tables on import).
- Importing Spyglass connects to the database. Keep Spyglass imports inside
  functions in modules the figures or tests import (`spyglass_data` is tested for
  this); `spyglass_pipeline` imports Spyglass at the top and must stay out of the
  figure pipeline and the tests.
- The analysis NWB files exist only on the lab's storage. A laptop can reach the
  database (via VPN) but `fetch_nwb` fails there, and reads over the VPN can stall;
  run fetches on a lab server. Server load varies; check it and use an idle one.
- After any change under `src/` or to `uv.lock`, refresh `provenance.source` in both
  figure summaries (procedure in [figure-pipeline.md](figure-pipeline.md)) and re-emit
  the reported values; `tests/test_reported_statistics_artifacts.py` fails otherwise.
  After a Figure-4 summary change, also update `site/data/replay.json` (its
  decode-cache fingerprint); other re-exported site files that differ only in the
  last floating-point digit need not be committed.
- Figure-4 renders are not byte-stable: the "Cont.-Frag. Model" panel title can
  land about a pixel differently between runs (seen with fresh and cache-backed
  renders alike). Keep the committed PNG when that title is the only difference
  (check pixels, e.g. with PIL), and compare PDFs by rendering them (their bytes
  also carry a creation date).
