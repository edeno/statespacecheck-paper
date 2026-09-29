"""The Figure-4 session explorer's files: container format, content, and currency."""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import io
import json
import tarfile
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from statespacecheck_paper.figure04_place_fields import marginal_position_distribution
from statespacecheck_paper.figure04_workflow import Figure4RenderData
from statespacecheck_paper.paths import (
    FIGURE04_SUMMARY_PATH,
    SITE_DATA_DIR,
    SITE_EXPLORER_ARCHIVE_DIR,
)
from statespacecheck_paper.site_explorer_export import (
    MODELS,
    RELEASE_URL,
    archive_bytes,
    block_bounds,
    decode_container,
    encode_container,
    explorer_files,
    export_recording_explorer,
    export_source_digest,
)
from statespacecheck_paper.site_export import (
    EVENT_VALUE_SIGNIFICANT_FIGURES,
    FIGURE04_PREDICTIVE_PERCENTILES,
    display_rows,
    flag_events,
    rounded_significant,
)
from statespacecheck_paper.style import METRIC_NAMES
from tests.test_figure04_layout import _compose_render_data
from tests.test_site_export import _recording_summary

CONTAINER_FIXTURE = SITE_DATA_DIR.parent / "tests" / "fixtures" / "container.bin"
BLOCK = 10


def container_fixture_bytes() -> bytes:
    """The reference container that site/tests/container.test.mjs decodes."""
    return encode_container(
        {"example": "reference", "values": [1, 2.5]},
        {
            "u1": np.array([0, 7, 255], dtype=np.uint8),
            "u2": np.array([[1, 65535], [300, 2]], dtype=np.uint16),
            "u4": np.array([4_000_000_000, 5], dtype=np.uint32),
            "i4": np.array([-1, 2**31 - 1], dtype=np.int32),
            "f4": np.array([0.5, -2.25, 1e-14], dtype=np.float32),
            "f8": np.array([np.pi], dtype=np.float64),
            "delta_u1": np.array([[10, 12, 5, 250], [0, 0, 1, 255]], dtype=np.uint8),
            "delta_u4": np.array([3, 3, 9, 4_000_000_000], dtype=np.uint32),
            "empty": np.zeros((0, 3), dtype=np.uint8),
        },
        delta=("delta_u1", "delta_u4"),
    )


def _explorer_render_data() -> Figure4RenderData:
    """Synthetic render data whose spikes sit in the bins their times fall in."""
    render_data = _compose_render_data()
    analysis = render_data.analysis_results
    reference = analysis.continuous_diagnostics
    assert reference.event_time is not None
    # The synthetic time grid is np.arange(n_time), so a spike's bin is its floor.
    time_ind = np.floor(reference.event_time).astype(np.intp)
    return dataclasses.replace(
        render_data,
        analysis_results=dataclasses.replace(
            analysis,
            continuous_diagnostics=dataclasses.replace(reference, event_time_ind=time_ind),
            continuous_fragmented_diagnostics=dataclasses.replace(
                analysis.continuous_fragmented_diagnostics, event_time_ind=time_ind
            ),
        ),
    )


def _explorer_summary(render_data: Figure4RenderData) -> dict[str, Any]:
    return {**_recording_summary(render_data), "flag_confusions": [{"metric": "hpd_overlap"}]}


def _decoded(files: dict[str, bytes], name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return decode_container(gzip.decompress(files[name]))


@pytest.mark.parametrize(
    "arrays",
    [
        {"a": np.arange(5, dtype=np.uint8)},
        {"a": np.arange(12, dtype=np.float32).reshape(3, 4), "b": np.array([-3], dtype=np.int32)},
        {"a": np.zeros((0, 2), dtype=np.uint16)},
    ],
)
def test_container_round_trips(arrays: dict[str, NDArray[Any]]) -> None:
    meta, decoded = decode_container(encode_container({"k": 1}, arrays, delta=()))
    assert meta == {"k": 1}
    for name, values in arrays.items():
        np.testing.assert_array_equal(decoded[name], values)
        assert decoded[name].dtype == values.dtype


def test_delta_filter_round_trips_through_wraparound() -> None:
    rows = np.array([[250, 3, 0, 255], [1, 200, 7, 7]], dtype=np.uint8)
    encoded = encode_container({}, {"rows": rows}, delta=("rows",))
    np.testing.assert_array_equal(decode_container(encoded)[1]["rows"], rows)
    with pytest.raises(TypeError, match="unsigned"):
        encode_container({}, {"x": np.array([1.0], dtype=np.float32)}, delta=("x",))


def test_committed_container_fixture_is_current() -> None:
    """The JavaScript reader's reference file is what the Python writer produces."""
    assert CONTAINER_FIXTURE.read_bytes() == container_fixture_bytes()


def test_block_bounds_cover_the_session_once() -> None:
    assert block_bounds(25, block=10) == [(0, 10), (10, 20), (20, 25)]
    with pytest.raises(ValueError, match="at least 2 blocks"):
        block_bounds(15, block=10)


def test_explorer_files_hold_every_spike_once_as_exported() -> None:
    render_data = _explorer_render_data()
    summary = _explorer_summary(render_data)
    overview, files = explorer_files(render_data, summary, grid_size=4, block=BLOCK)
    analysis = render_data.analysis_results
    diagnostics = dict(
        zip(
            MODELS,
            (analysis.continuous_diagnostics, analysis.continuous_fragmented_diagnostics),
            strict=True,
        )
    )
    results = dict(
        zip(
            MODELS,
            (analysis.continuous_results, analysis.continuous_fragmented_results),
            strict=True,
        )
    )
    n_events = overview["n_events"]
    index_meta, index = _decoded(files, "explorer/index.bin.gz")
    assert index_meta["n_events"] == n_events
    np.testing.assert_array_equal(index["time_bin"], diagnostics[MODELS[0]].event_time_ind)
    for metric in METRIC_NAMES:
        counts = overview["metrics"][metric]["counts"]
        assert sum(counts) == n_events
        np.testing.assert_array_equal(np.bincount(index[f"square_{metric}"], minlength=16), counts)

    bounds = block_bounds(render_data.time.size, BLOCK)
    assert overview["n_blocks"] == len(bounds)
    next_event = 0
    for number, (start, stop) in enumerate(bounds):
        meta, block = _decoded(files, f"explorer/blocks/{number:04d}.bin.gz")
        assert (meta["start"], meta["stop"], meta["first_event"]) == (start, stop, next_event)
        ids = slice(next_event, next_event + block["event_bin"].size)
        next_event = ids.stop
        np.testing.assert_array_equal(
            block["event_bin"].astype(np.int64) + start, index["time_bin"][ids]
        )
        np.testing.assert_array_equal(block["event_cell"], index["cell"][ids])
        event_time = diagnostics[MODELS[0]].event_time
        assert event_time is not None
        np.testing.assert_allclose(
            (block["event_tick"] / 1e4) + meta["start_time"],
            event_time[ids] - render_data.time[0],
            atol=6e-5,
        )
        for model in MODELS:
            predictive = np.nan_to_num(
                marginal_position_distribution(
                    results[model].isel(time=slice(start, stop)), "predictive"
                )
            )
            np.testing.assert_array_equal(block[f"predictive_{model}"], display_rows(predictive))
            np.testing.assert_allclose(
                block[f"predictive_max_{model}"], predictive.max(axis=1), rtol=1e-6
            )
            for metric in METRIC_NAMES:
                values = getattr(diagnostics[model], f"event_{metric}")[ids]
                np.testing.assert_allclose(
                    block[f"{metric}_{model}"],
                    rounded_significant(values, EVENT_VALUE_SIGNIFICANT_FIGURES),
                    rtol=1e-6,
                )
        # One bit per (model, flagged metric), in the order the block names.
        for bit, (model, metric) in enumerate(meta["flag_bits"]):
            values = getattr(diagnostics[model], f"event_{metric}")[ids]
            np.testing.assert_array_equal(
                (block["flags"] >> bit) & 1, flag_events(values, summary["flag_rules"][metric])
            )
    assert next_event == n_events

    # Each window's color range: the paper's percentiles over its two blocks.
    for model in MODELS:
        ranges = overview["window_ranges"][model]
        assert len(ranges) == len(bounds) - 1
        start, stop = bounds[1][0] - BLOCK, bounds[1][1]
        window = np.nan_to_num(
            marginal_position_distribution(
                results[model].isel(time=slice(start, stop)), "predictive"
            )
        )
        np.testing.assert_allclose(
            ranges[0], np.percentile(window, FIGURE04_PREDICTIVE_PERCENTILES)
        )


def test_explorer_rejects_another_decode_and_unordered_spikes() -> None:
    render_data = _explorer_render_data()
    summary = _explorer_summary(render_data)
    summary["provenance"]["figure04_caches"]["fingerprint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="fingerprints differ"):
        explorer_files(render_data, summary, block=BLOCK)
    unordered = _compose_render_data()  # random, unordered spike bins
    with pytest.raises(ValueError, match="time order"):
        explorer_files(unordered, _explorer_summary(unordered), block=BLOCK)


def test_archive_is_deterministic_and_named_by_content(tmp_path: Any) -> None:
    render_data = _explorer_render_data()
    summary = _explorer_summary(render_data)
    overview_path, archive_path = export_recording_explorer(
        render_data, summary, tmp_path / "data", tmp_path / "archive", grid_size=4, block=BLOCK
    )
    overview = json.loads(overview_path.read_text())
    data = archive_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    assert overview["archive"] == {
        "name": f"recording-explorer-{digest[:12]}.tar",
        "url": RELEASE_URL + f"recording-explorer-{digest[:12]}.tar",
        "sha256": digest,
        "bytes": len(data),
    }
    _, files = explorer_files(render_data, summary, grid_size=4, block=BLOCK)
    assert archive_bytes(files) == data
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        names = archive.getnames()
    assert names == sorted(files)


def test_committed_overview_matches_the_paper_and_the_export_code() -> None:
    """The overview describes the summary's decode and was written by today's code.

    A change to the export modules' executable code changes the digest: export
    again and publish the new archive.
    """
    summary = json.loads(FIGURE04_SUMMARY_PATH.read_text(encoding="utf-8"))
    overview = json.loads((SITE_DATA_DIR / "recording_explorer.json").read_text(encoding="utf-8"))
    caches = summary["provenance"]["figure04_caches"]
    assert overview["export_source_sha256"] == export_source_digest()
    assert overview["decode_cache_fingerprint"] == caches["fingerprint_sha256"]
    assert overview["diagnostics_fingerprint"] == caches["diagnostics_fingerprint_sha256"]
    assert overview["flag_rules"] == summary["flag_rules"]
    assert overview["flag_confusions"] == summary["flag_confusions"]
    assert overview["n_events"] == summary["flag_confusions"][0]["n"]
    for metric in METRIC_NAMES:
        assert sum(overview["metrics"][metric]["counts"]) == overview["n_events"]
    for model in MODELS:
        assert len(overview["window_ranges"][model]) == overview["n_blocks"] - 1
    archive = overview["archive"]
    assert archive["name"] == f"recording-explorer-{archive['sha256'][:12]}.tar"
    assert archive["url"] == RELEASE_URL + archive["name"]


def test_local_archive_matches_the_committed_overview() -> None:
    """Check the exported archive, when present (the site build downloads it)."""
    overview = json.loads((SITE_DATA_DIR / "recording_explorer.json").read_text(encoding="utf-8"))
    path = SITE_EXPLORER_ARCHIVE_DIR / overview["archive"]["name"]
    if not path.exists():
        pytest.skip(f"{path.name} is not in {SITE_EXPLORER_ARCHIVE_DIR}")
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == overview["archive"]["sha256"]
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        for name in archive.getnames():
            member = archive.extractfile(name)
            assert member is not None, name
            files[name] = member.read()
    assert len(files) == overview["n_blocks"] + 1
    _, index = _decoded(files, "explorer/index.bin.gz")
    assert index["time_bin"].size == overview["n_events"]
    for metric in METRIC_NAMES:
        np.testing.assert_array_equal(
            np.bincount(index[f"square_{metric}"], minlength=overview["grid_size"] ** 2),
            overview["metrics"][metric]["counts"],
        )
    for number in (0, overview["n_blocks"] // 2, overview["n_blocks"] - 1):
        meta, block = _decoded(files, f"explorer/blocks/{number:04d}.bin.gz")
        assert meta["decode_cache_fingerprint"] == overview["decode_cache_fingerprint"]
        ids = slice(meta["first_event"], meta["first_event"] + block["event_bin"].size)
        np.testing.assert_array_equal(
            block["event_bin"].astype(np.int64) + meta["start"], index["time_bin"][ids]
        )
        np.testing.assert_array_equal(block["event_cell"], index["cell"][ids])
