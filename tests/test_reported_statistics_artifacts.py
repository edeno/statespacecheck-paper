"""Golden contracts for the machine-readable statistics reported by the paper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = REPO_ROOT / "manuscript" / "figures" / "main"


def _load(name: str) -> dict[str, Any]:
    with open(FIGURE_DIR / name, encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def test_figure03_reported_statistics_match_canonical_run() -> None:
    payload = _load("figure03_summary.json")

    assert payload["schema_version"] == 1
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
                [1.512, 39.413, 0.938, 1.150, 9.877, 0.000],
                [2.761, 43.026, 1.764, 2.723, 13.741, 0.000],
                [1.184, 36.771, 0.672, 0.820, 8.532, 44.949],
            ]
        ),
        atol=5e-4,
        rtol=0.0,
    )
    assert payload["diagnostic_thresholds"] == pytest.approx(
        {
            "hpd_overlap": 0.0,
            "kl_divergence": 4.003728364400552,
            "predictive_pvalue": 0.05,
        }
    )


def test_figure04_reported_statistics_counts_partition_events() -> None:
    payload = _load("figure04_summary.json")

    assert payload["schema_version"] == 1
    assert payload["dataset"] == {"animal_date_epoch": "j1620210710_02_r1"}
    assert payload["diagnostic_means"]["continuous"] == pytest.approx(
        {
            "hpd_overlap": 0.8363245225528076,
            "kl_divergence": 3.0032265820064956,
            "predictive_pvalue": 0.5246446577216307,
        }
    )

    expected = {
        "hpd_overlap": (870_018, 1_456, 16_881, 173, 851_508),
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
