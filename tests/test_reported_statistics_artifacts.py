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
from statespacecheck_paper.figure03_summary import Figure3Calibration, Figure3RealizationSummary
from statespacecheck_paper.figure04_cache import Figure4CacheProvenance, Figure4Paths
from statespacecheck_paper.figure04_decoder import Figure4Config, Figure4DiagnosticsConfig
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


SUPPLEMENTARY_DIR = REPO_ROOT / "manuscript" / "figures" / "supplementary"


def _load(name: str) -> dict[str, Any]:
    with open(FIGURE_DIR / name, encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def _load_supplementary(name: str) -> dict[str, Any]:
    with open(SUPPLEMENTARY_DIR / name, encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def _round_trip_live_payload(tmp_path: Path, payload: dict[str, object]) -> dict[str, Any]:
    path = write_json_artifact(tmp_path / "live.json", payload)
    with open(path, encoding="utf-8") as handle:
        result: dict[str, Any] = json.load(handle)
    return result


def test_figure03_reported_statistics_match_canonical_run(tmp_path: Path) -> None:
    payload = _load("figure03_summary.json")

    assert payload["schema_version"] == 6
    assert payload["realizations"] == {"count": 100, "first_seed": 1, "last_seed": 100}
    calibration = payload["calibration"]
    assert calibration["count"] == 100
    assert calibration["first_seed"] == 1001 and calibration["last_seed"] == 1100
    assert calibration["steps_per_session"] == 6000
    assert calibration["n_pooled_events"] == 289_278
    assert calibration["pooled_null_flag_percentages"] == pytest.approx(
        {"hpd_overlap": 1.623, "predictive_pvalue": 2.626, "kl_divergence": 1.000}, abs=5e-4
    )
    # Super-uniform under the matched null: each tail is at most its alpha.
    for alpha, tail in zip(
        calibration["rank_pvalue_tail_alphas"],
        calibration["rank_pvalue_tail_percentages"],
        strict=True,
    ):
        assert tail <= 100.0 * alpha
    assert payload["metric_order"] == ["hpd_overlap", "predictive_pvalue", "kl_divergence"]
    assert payload["condition_order"] == [
        "matched_null",
        "recovery",
        "remap",
        "history_dependent",
        "replay",
        "drift",
        "reflected_map",
        "sparse_population",
    ]
    np.testing.assert_allclose(
        np.asarray(payload["median_flag_percentages"]),
        np.array(
            [
                [1.507, 1.835, 22.595, 0.895, 2.103, 10.659, 2.152, 0.000],
                [2.603, 2.876, 23.925, 1.501, 2.686, 13.672, 3.088, 0.000],
                [0.949, 1.163, 18.689, 0.506, 1.178, 8.097, 1.390, 56.250],
            ]
        ),
        atol=5e-4,
        rtol=0.0,
    )
    assert payload["accuracy_metric_order"] == [
        "median_absolute_error",
        "filtered_hpd_coverage_percent",
        "median_filtered_hpd_size",
        "median_predictive_hpd_size",
    ]
    np.testing.assert_allclose(
        np.asarray(payload["median_decoding_accuracy"]),
        np.array(
            [
                [1.755, 1.788, 41.822, 1.906, 36.998, 7.951, 46.846, 4.249],
                [96.183, 95.111, 1.238, 94.200, 5.200, 36.013, 2.487, 97.588],
                [11.0, 11.0, 11.0, 11.0, 8.0, 11.0, 11.0, 26.0],
                [11.0, 11.0, 11.0, 11.0, 8.0, 11.0, 11.0, 26.0],
            ]
        ),
        atol=5e-4,
        rtol=0.0,
    )
    assert payload["flag_rules"] == {
        "hpd_overlap": {"comparison": "less_than_or_equal", "threshold": 0.0},
        "kl_divergence": {"comparison": "greater_than_or_equal", "threshold": 4.220212564130426},
        "predictive_pvalue": {"comparison": "less_than_or_equal", "threshold": 0.05},
    }
    assert payload["condition_labels"][-1] == "Sparse population"
    np.testing.assert_allclose(
        payload["replay_represented_accuracy"]["median"], [1.301, 97.450], atol=5e-4
    )
    assert payload["sparse_population"]["median_sparse_cell_event_count"] == 12.0
    # Per-realization arrays carry the distributions behind the medians.
    per_realization = np.asarray(payload["flag_percentages_by_realization"], dtype=float)
    assert per_realization.shape == (100, 3, 8)
    np.testing.assert_allclose(
        np.nanmedian(per_realization, axis=0), payload["median_flag_percentages"], atol=1e-9
    )
    assert np.asarray(payload["event_counts_by_realization"]).shape == (100, 8)
    # Approximate median uncertainty is retained independently of prose precision.
    assert payload["standard_error_method"] == "order_statistic_interval_95"
    flag_errors = np.asarray(payload["median_flag_percentage_standard_errors"], dtype=float)
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
        "calibration_steps_per_session": 6000,
        "calibration_sample": "independent_matched_null_sessions",
        "hpd_overlap": {"rule": "pooled_calibration_quantile", "quantile": 0.01},
        "kl_divergence": {"rule": "pooled_calibration_quantile", "quantile": 0.99},
        "predictive_pvalue": {"rule": "fixed_cutoff", "cutoff": 0.05},
    }
    # The Methods' configuration: exact matched null, 95% regions, the
    # history-dependence parameters at 1 ms/step, and the pinned rate gain.
    configuration = payload["configuration"]
    assert configuration["trajectory_model"] == "discrete_matched"
    assert configuration["hpd_coverage"] == 0.95
    assert configuration["history_refractory_steps"] == 1
    assert configuration["history_burst_window"] == [2, 10]
    assert configuration["history_burst_factor"] == 3.0
    assert configuration["history_rate_matching_gain"] == 0.512

    calibration_record = Figure3Calibration(
        diagnostic_thresholds=DiagnosticThresholds(
            hpd_overlap=payload["flag_rules"]["hpd_overlap"]["threshold"],
            kl_divergence=payload["flag_rules"]["kl_divergence"]["threshold"],
            predictive_pvalue=payload["flag_rules"]["predictive_pvalue"]["threshold"],
        ),
        n_calibration_realizations=calibration["count"],
        first_calibration_seed=calibration["first_seed"],
        calibration_steps_per_session=calibration["steps_per_session"],
        n_pooled_events=calibration["n_pooled_events"],
        pooled_null_flag_percentages=np.asarray(
            [calibration["pooled_null_flag_percentages"][m] for m in payload["metric_order"]]
        ),
        per_realization_null_flag_percentages=np.asarray(
            calibration["per_realization_null_flag_percentages"]
        ),
        hpd_threshold_tie_percent=calibration["hpd_threshold_tie_percent"],
        rank_pvalue_tail_percentages=np.asarray(calibration["rank_pvalue_tail_percentages"]),
    )

    def _nan(values: object) -> np.ndarray:
        return np.asarray(values, dtype=float)

    summary = Figure3RealizationSummary(
        calibration=calibration_record,
        median_flag_percentages=np.asarray(payload["median_flag_percentages"]),
        flag_percentages_by_realization=_nan(payload["flag_percentages_by_realization"]),
        event_counts_by_realization=np.asarray(payload["event_counts_by_realization"]),
        pooled_flag_percentages=np.asarray(payload["pooled_flag_percentages"]),
        median_decoding_accuracy=np.asarray(payload["median_decoding_accuracy"]),
        decoding_accuracy_by_realization=np.asarray(payload["decoding_accuracy_by_realization"]),
        flag_percentage_standard_errors=_nan(payload["median_flag_percentage_standard_errors"]),
        decoding_accuracy_standard_errors=np.asarray(
            payload["median_decoding_accuracy_standard_errors"]
        ),
        replay_represented_accuracy=np.asarray(
            payload["replay_represented_accuracy"]["by_realization"]
        ),
        phase_spike_rates_by_realization=np.asarray(
            payload["phase_ordinary_spike_rates_hz"]["by_realization"]
        ),
        sparse_cell_flag_percentages_by_realization=_nan(
            payload["sparse_population"]["sparse_cell_flag_percentages_by_realization"]
        ),
        sparse_cell_event_counts_by_realization=np.asarray(
            payload["sparse_population"]["sparse_cell_event_counts_by_realization"]
        ),
        matched_null_rank_pvalue_tail_percentages=np.asarray(
            payload["matched_null_rank_pvalue_tail_percentages"]
        ),
        n_realizations=payload["realizations"]["count"],
    )
    live = figure03_summary_payload(Figure3Config(), summary)
    assert _round_trip_live_payload(tmp_path, live) == payload


def test_figure04_reported_statistics_counts_partition_events(tmp_path: Path) -> None:
    payload = _load("figure04_summary.json")

    assert payload["schema_version"] == 4
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
        diagnostics_fingerprint_sha256=cache_payload["diagnostics_fingerprint_sha256"],
        diagnostics_schema_version=cache_payload["diagnostics_schema_version"],
        statespacecheck_version=cache_payload["statespacecheck_version"],
        diagnostics_config=Figure4DiagnosticsConfig(**cache_payload["diagnostics_config"]),
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
