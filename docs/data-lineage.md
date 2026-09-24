# Figure 4 data lineage

Figure 4 reads one derived input file for one recording epoch, `j1620210710_02_r1`
(rat j16, 2021-07-10, run epoch `02_r1`; the spatial-bandit data of
[Comrie et al. 2024](https://doi.org/10.1101/2024.09.23.613567)). This page records
which Frank-lab Spyglass database entries it comes from, how that was verified,
and what is publicly available. The fetch code is
`src/statespacecheck_paper/spyglass_data.py`.

## The input file

`j1620210710_02_r1_figure04_inputs.npz`, SHA-256
`60383b394b597e2900545548ecac7c53a8601038ace9dbeb42d7f9a5fe1c93b3` (75.1 MB; not in
the repository, `data/` is ignored). It matches
`provenance.figure04_decode_cache.export_file_sha256` in
`manuscript/figures/main/figure04_summary.json`. It holds only numeric and string
arrays (loaded with `allow_pickle=False`) and is written deterministically, so the
same content always has the same SHA-256; `load_local_data.recording_arrays`
defines the layout.

### Originally: five pickles

The recording was first exported as five pickles (the earliest copy found, a local
download, is dated 2025-02-04), and the figure was first made from them.
`scripts/convert_figure04_pickles.py` converts them to the `.npz` above and checks
that it loads back identical (values, dtypes, index, column order, every unit's
spike times, and the track graph in the same node and edge order).

| File | SHA-256 |
| --- | --- |
| `j1620210710_02_r1_position_info.pkl` | `8c96c7ace899db447d546e63a4e196188f14ab81ed66ec39c7939954b3775927` |
| `j1620210710_02_r1_HPC_spike_times.pkl` | `2d3e87fddf2e76237d5cbcd8773fb754d873b96dd41820f4c6a07b0fc9802905` |
| `j1620210710_02_r1_track_graph.pkl` | `147bfb7edb0e100cb56a09a00bca8104e789b5f462c970789ddd5d5adb45b84f` |
| `j1620210710_02_r1_linear_edge_order.pkl` | `ccdd3cf89080a08d6b670c77eb635174fb801c44cb040e4f6aba7c3da5967646` |
| `j1620210710_02_r1_linear_edge_spacing.pkl` | `10fe7c1a5cf8e7ec589293a3ce47a20764bc918c193ae6736a1e3df783c83c8e` |

## Spyglass sources

All entries are for NWB file `j1620210710_.nwb`.

**Track graph, edge order, edge spacing.** `TrackGraph` (`spyglass.linearization.v0`)
entry `j1620210706`, found through `IntervalLinearizedPosition` for
`pos 1 valid times` / `default_decoding` / `linearization_param_name="default"`.
The graph has 10 nodes and 9 edges; edge spacing is 15 cm. (The track graph is
shared across this animal's sessions, hence the earlier date in its name.)

**Position.** `IntervalPositionInfo` (`spyglass.common.common_position`) for
interval `pos 1 valid times` (the `PositionIntervalMap` entry for epoch `02_r1`)
with `position_info_param_name="default_decoding"`: LED-tracked head position,
smoothed over 0.125 s, speed smoothed with a 0.1 s s.d. kernel, and upsampled
linearly to 500 Hz. The loader then

1. drops rows with any NaN,
2. keeps samples within the `02_r1 noPrePostTrialTimes` interval,
3. linearizes onto the track graph (`track_linearization.get_linearized_position`),
4. adds `patch_id` from the track segment.

DLC position exists for this session only in epochs 10, 12, and 14, so there is no
DLC alternative for this epoch.

**HPC spike times.** v0 `CuratedSpikeSorting` (`spyglass.spikesorting.v0`):

| Field | Value |
| --- | --- |
| `sort_interval_name` | `runs_noPrePostTrialTimes raw data valid times` |
| `preproc_params_name` | `franklab_tetrode_hippocampus` |
| `team_name` | `ac_em_xs` |
| `sorter` / `sorter_params_name` | `mountainsort4` / `franklab_tetrode_hippocampus_30KHz` |
| `artifact_removed_interval_list_name` | one per sort group, artifact parameters `ampl_2000_z_30_prop_075_1ms` |
| `curation_id` | `1` |

- 22 tetrode sort groups, all in `BrainRegion` "hippocampus" (11 left, 11 right).
  Sort groups 10, 21, and 22 have no units after curation.
- Sorted 2022-07-18 to 2022-07-22. Curation 1 is derived from curation 0 and
  described as "auto curated"; no `AutomaticCuration` entry records the rule that
  produced it. It labels 101 units `noise`/`reject`, and v0 `CuratedSpikeSorting`
  drops units labelled `reject`. **The remaining 203 units carry no labels**
  (none `accept` or `mua`): curation 1 has no manual labels.
- The sort interval spans all run epochs. The export clips each unit to the first
  and last position timestamps (inclusive) and keeps units with no spikes in
  this epoch: **21 of the 203 units have no spikes**. 870,018 spikes remain, of
  11,394,298 over the whole sort interval.
- Units are in `(sort_group_id, unit_id)` order; the fetch checks each analysis
  file's units against `CuratedSpikeSorting.Unit`.

## Verification

Checked on 2026-09-24 against the Frank-lab database, reading only:

| File | How it was checked | Result |
| --- | --- | --- |
| track graph, edge order, edge spacing | compared with the `TrackGraph` entry | identical (node positions exact) |
| `position_info` | rebuilt from the public DANDI copy of the `IntervalPositionInfo` analysis file (`sub-j16/sub-j16_ses-j16-20210710_obj-g0r7gi_behavior.nwb` in dandiset 001942) plus the steps above | identical: `pandas.testing.assert_frame_equal(check_exact=True)` |
| `HPC_spike_times` | `CuratedSpikeSorting.fetch_nwb()` on a lab server (Spyglass 0.5.6.dev16) | identical, array by array in order; all 22 analysis files match the `contents_hash` recorded in the database |

The code in this repository was then checked end to end. `scripts/fetch_figure04_inputs.py`
ran on a lab server in a conda environment with Python 3.11.8, Spyglass
0.5.6.dev16, NumPy 1.26.4, pandas 1.5.3, networkx 3.4, and track-linearization
2.3.2. **All five pickles it wrote were byte-identical to the ones the figure was
made from** (same SHA-256 as above). It was run again, on a second lab server,
after the fetch gained its data checks (one sort per sort group, unit IDs per sort
group, patch coverage, and the loader's checks before writing), with the same
result.

After the switch to the `.npz`, the same script wrote the `.npz` directly on a lab
server (NumPy 1.26), and it has **the same SHA-256 as the one converted from the
pickles** on a laptop (NumPy 2.3). Figure 4 regenerated from the `.npz` gives an
identical summary (every mean and flag count; only the provenance checksums and
cache fingerprints change). The PNG from a cache-backed render is byte-identical to
the committed one, and the PDF renders to identical pixels. (A render in the same
process right after a fresh decode shifts one panel title by about a pixel; this is
a rendering quirk, not a data difference.)

The script that originally wrote the files was not found in version control. The
code here reproduces its output exactly. Position follows `continuum-swr-replay`'s
`get_position_info`, and spike times follow the pattern of its sorted-unit loader;
the differences (for example, units with no spikes are kept here) are listed in
each function's docstring.

## Public availability

DANDI dandiset [001942](https://dandiarchive.org/dandiset/001942) is the Spyglass
export `comrie2026` (export 135): for this session the export lists 154 files and
the dandiset has 154 assets. It contains the raw recording and the position
analysis file above. **The HPC sorting above is not part of that export or of any
other Spyglass export, and is not on DANDI.** The recording could be re-sorted from
the raw data, but the exact units used in Figure 4 are available only from the lab
database until they are exported.

## Regenerating the input file

Both scripts need lab database credentials and a lab server with the analysis NWB
store mounted. `REF` below is a directory holding the input file the figure used;
`--output-dir` must not already hold one (the fetch script accepts `--overwrite`).

```bash
# Read-only: rebuild the input file and compare it with the one the figure used.
PYTHONPATH=src python scripts/fetch_figure04_inputs.py \
    --output-dir /tmp/figure04_inputs --compare-to REF

# Writes to the lab database: log the same fetches in a Spyglass export and
# package it. Asks for confirmation; refuses an existing paper_id.
PYTHONPATH=src python scripts/spyglass_export_figure04.py \
    --paper-id <new-paper-id> --output-dir /tmp/figure04_export --compare-to REF --populate
```

The fetch script was verified as shown, in a lab conda environment with the
lab's Spyglass and this repository's `src/` on the import path. (That
environment's pandas 1.5.3 is below this package's `pandas>=2.0`, so the package
is not installed there.) The export script has not been run yet. Before it writes
anything it checks the paths, asks for confirmation, and checks that the
installed Spyglass declares every column of the database's export tables; the
version in `uv.lock` does not, so the export must use the lab's current Spyglass.
Spyglass also requires packaging with the same `x.y.z` version that logged the
selection, which is why `--populate` packages in the same run and requires
`--compare-to`.

The `spyglass` extra (`uv sync --extra spyglass`) installs the Spyglass in
`uv.lock`; it has not been used for a full fetch.

`--compare-to` compares array by array and names any array that differs. Because
the file is written deterministically, a matching SHA-256 already means identical
content.

From the original pickles (no database needed):

```bash
uv run python scripts/convert_figure04_pickles.py --pickle-dir data --output-dir data
```

## Open items

- Run the Spyglass export for this paper (`scripts/spyglass_export_figure04.py`).
- Make the HPC sorting publicly available, e.g. by adding the export's analysis
  files to DANDI.
- The manuscript's data statement cites dandiset 001942, which does not include
  the sorted units.
