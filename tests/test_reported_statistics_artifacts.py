"""Golden contracts for the machine-readable statistics reported by the paper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from statespacecheck_paper.diagnostics import DiagnosticThresholds
from statespacecheck_paper.figure03_generation import figure03_summary_payload
from statespacecheck_paper.figure03_protocol import Figure3Config
from statespacecheck_paper.figure03_summary import Figure3RealizationSummary
from statespacecheck_paper.figure04_cache import Figure4CacheProvenance, Figure4Paths
from statespacecheck_paper.figure04_decoder import Figure4Config
from statespacecheck_paper.figure04_diagnostics import FlagConfusion
from statespacecheck_paper.figure04_generation import figure04_summary_payload
from statespacecheck_paper.figure04_workflow import (
    Figure4DiagnosticMeans,
    Figure4Summary,
)
from statespacecheck_paper.load_local_data import EXPORT_FILE_SUFFIXES
from statespacecheck_paper.scientific_artifacts import write_json_artifact

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "manuscript" / "figures" / "main"


def _load(name: str) -> dict[str, Any]:
    with open(FIGURE_DIR / name, encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def _round_trip_live_payload(tmp_path: Path, payload: dict[str, object]) -> dict[str, Any]:
    path = write_json_artifact(tmp_path / "live.json", payload)
    with open(path, encoding="utf-8") as handle:
        result: dict[str, Any] = json.load(handle)
    return result


def test_figure03_reported_statistics_match_canonical_run(tmp_path: Path) -> None:
    payload = _load("figure03_summary.json")

    assert payload["schema_version"] == 5
    assert payload["realizations"] == {
        "count": 100,
        "first_seed": 1,
        "last_seed": 100,
    }
    assert payload["metric_order"] == [
        "hpd_overlap",
        "predictive_pvalue",
        "kl_divergence",
    ]
    assert payload["condition_order"] == [
        "well_specified",
        "remap",
        "history_dependent",
        "replay",
        "drift",
        "sparse_population",
    ]
    np.testing.assert_allclose(
        np.asarray(payload["median_flag_percentages"]),
        np.array(
            [
                [1.766, 40.799, 1.192, 2.083, 10.733, 0.000],
                [2.761, 43.026, 1.764, 2.723, 13.741, 0.000],
                [1.178, 36.771, 0.668, 1.301, 8.532, 42.857],
            ]
        ),
        atol=5e-4,
        rtol=0.0,
    )
    assert payload["accuracy_metric_order"] == ["median_absolute_error"]
    np.testing.assert_allclose(
        np.asarray(payload["median_decoding_accuracy"]),
        np.array([[1.792, 42.169, 1.678, 36.358, 7.985, 0.990]]),
        atol=5e-4,
        rtol=0.0,
    )
    assert payload["flag_rules"] == {
        "hpd_overlap": {"comparison": "less_than_or_equal", "threshold": 0.0},
        "kl_divergence": {
            "comparison": "greater_than_or_equal",
            "threshold": 4.13792649205148,
        },
        "predictive_pvalue": {
            "comparison": "less_than_or_equal",
            "threshold": 0.05,
        },
    }
    assert payload["condition_labels"][-1] == "Sparse population"
    # Dispersion is published because it sets the manuscript's printed precision.
    assert payload["standard_error_method"] == "order_statistic_interval_95"
    flag_errors = np.asarray(payload["median_flag_percentage_standard_errors"])
    accuracy_errors = np.asarray(payload["median_decoding_accuracy_standard_errors"])
    assert flag_errors.shape == np.asarray(payload["median_flag_percentages"]).shape
    assert accuracy_errors.shape == np.asarray(payload["median_decoding_accuracy"]).shape
    assert np.all(flag_errors >= 0.0) and np.all(accuracy_errors >= 0.0)
    # The remap column is trajectory-dependent, so its median is the least
    # certain of the flag percentages by an order of magnitude.
    remap = payload["condition_order"].index("remap")
    assert np.all(flag_errors[:, remap] > 1.0)
    # The Methods quote the rule behind the thresholds, not just their values.
    assert payload["threshold_provenance"] == {
        "baseline_end_index": 6000,
        "hpd_overlap": {"rule": "pooled_baseline_quantile", "quantile": 0.01},
        "kl_divergence": {"rule": "pooled_baseline_quantile", "quantile": 0.99},
        "predictive_pvalue": {"rule": "fixed_cutoff", "cutoff": 0.05},
    }
    # The history-dependence misfit parameters the Methods report, at 1 ms/step:
    # a 1 ms hard refractory, a 2-10 ms burst window, a threefold rate increase.
    configuration = payload["configuration"]
    assert configuration["history_refractory_steps"] == 1
    assert configuration["history_burst_window"] == [2, 10]
    assert configuration["history_burst_factor"] == 3.0

    summary = Figure3RealizationSummary(
        diagnostic_thresholds=DiagnosticThresholds(
            hpd_overlap=payload["flag_rules"]["hpd_overlap"]["threshold"],
            kl_divergence=payload["flag_rules"]["kl_divergence"]["threshold"],
            predictive_pvalue=payload["flag_rules"]["predictive_pvalue"]["threshold"],
        ),
        median_flag_percentages=np.asarray(payload["median_flag_percentages"]),
        median_decoding_accuracy=np.asarray(payload["median_decoding_accuracy"]),
        flag_percentage_standard_errors=np.asarray(
            payload["median_flag_percentage_standard_errors"]
        ),
        decoding_accuracy_standard_errors=np.asarray(
            payload["median_decoding_accuracy_standard_errors"]
        ),
        n_realizations=payload["realizations"]["count"],
    )
    live = figure03_summary_payload(Figure3Config(), summary)
    assert _round_trip_live_payload(tmp_path, live) == payload


def test_figure04_reported_statistics_counts_partition_events(tmp_path: Path) -> None:
    payload = _load("figure04_summary.json")

    assert payload["schema_version"] == 3
    # 203 units is the count reported in the Figure-4 caption.
    assert payload["dataset"] == {
        "animal_date_epoch": "j1620210710_02_r1",
        "n_units": 203,
    }
    assert payload["diagnostic_means"]["continuous"] == pytest.approx(
        {
            "hpd_overlap": 0.8312001903462088,
            "kl_divergence": 3.014811995428145,
            "predictive_pvalue": 0.5246446577216307,
        }
    )

    expected = {
        "hpd_overlap": (870_018, 1_501, 17_289, 176, 851_052),
        "predictive_pvalue": (870_018, 24_581, 9_373, 1_706, 834_358),
    }
    for confusion in payload["flag_confusions"]:
        counts = (
            confusion["n"],
            confusion["both"],
            confusion["a_only"],
            confusion["b_only"],
            confusion["neither"],
        )
        assert counts == expected[confusion["metric"]]
        assert sum(counts[1:]) == counts[0]
        assert confusion["rescue_rate"] == pytest.approx(
            confusion["a_only"] / (confusion["a_only"] + confusion["both"])
        )

    assert payload["flag_rules"] == {
        "hpd_overlap": {"comparison": "less_than_or_equal", "threshold": 0.05},
        "predictive_pvalue": {
            "comparison": "less_than_or_equal",
            "threshold": 0.05,
        },
    }
    source = payload["provenance"]["source"]
    assert len(source["source_tree_sha256"]) == 64
    assert len(source["uv_lock_sha256"]) == 64

    epoch = payload["dataset"]["animal_date_epoch"]
    cache_payload = payload["provenance"]["figure04_decode_cache"]
    cache_provenance = Figure4CacheProvenance(
        fingerprint_sha256=cache_payload["fingerprint_sha256"],
        schema_version=cache_payload["schema_version"],
        animal_date_epoch=epoch,
        export_checksums=tuple(
            (suffix, cache_payload["export_file_sha256"][f"{epoch}{suffix}"])
            for suffix in EXPORT_FILE_SUFFIXES
        ),
        non_local_detector_version=cache_payload["non_local_detector_version"],
    )
    means = payload["diagnostic_means"]
    summary = Figure4Summary(
        continuous=Figure4DiagnosticMeans(**means["continuous"]),
        continuous_fragmented=Figure4DiagnosticMeans(**means["continuous_fragmented"]),
        flag_confusions=tuple(
            FlagConfusion(
                metric=item["metric"],
                threshold=item["threshold"],
                n=item["n"],
                both=item["both"],
                a_only=item["a_only"],
                b_only=item["b_only"],
                neither=item["neither"],
            )
            for item in payload["flag_confusions"]
        ),
        n_units=payload["dataset"]["n_units"],
    )
    live = figure04_summary_payload(
        config=Figure4Config(),
        paths=Figure4Paths(REPO_ROOT / "data", epoch),
        summary=summary,
        cache_provenance=cache_provenance,
    )
    assert _round_trip_live_payload(tmp_path, live) == payload
