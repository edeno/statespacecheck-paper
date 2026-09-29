"""Tests for ``scripts/check_reproduction.py``, the fresh-run summary comparison."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from statespacecheck_paper.paths import FIGURE_DIR
from tests._scripts import load_script


@pytest.fixture(scope="module")
def check() -> ModuleType:
    return load_script("check_reproduction")


def _summary() -> dict[str, Any]:
    """A small summary with every JSON type, nesting, and the machine-specific keys."""
    return {
        "schema_version": 8,
        "configuration": {"seed": 1001, "label": "continuous", "flag": True, "none": None},
        "counts": [870_018, 1_501],
        "means": {"hpd_overlap": 0.8312001903462088, "matrix": [[1.5, 2.25], [0.0, 4.0]]},
        "provenance": {
            "source": {"source_tree_sha256": "a" * 64, "uv_lock_sha256": "b" * 64},
            "figure04_caches": {
                "fingerprint_sha256": "c" * 64,
                "diagnostics_fingerprint_sha256": "d" * 64,
                "python_version": "3.11.15",
                "machine": "arm64",
                "decode_dependency_versions": {"numpy": "2.3.4"},
                "diagnostics_dependency_versions": {"numpy": "2.3.4"},
                "input_file_sha256": {"inputs.npz": "e" * 64},
            },
        },
    }


def test_identical_summaries_match(check: ModuleType) -> None:
    assert check.compare_json(_summary(), _summary()) == []


def test_machine_specific_provenance_is_ignored(check: ModuleType) -> None:
    fresh = _summary()
    caches = fresh["provenance"]["figure04_caches"]
    caches.update(
        fingerprint_sha256="0" * 64,
        diagnostics_fingerprint_sha256="1" * 64,
        python_version="3.11.9",
        machine="x86_64",
        decode_dependency_versions={"numpy": "2.3.5", "jax": "0.8.0"},
        diagnostics_dependency_versions={},
    )
    assert check.compare_json(_summary(), fresh) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda s: s["provenance"]["source"].update(uv_lock_sha256="f" * 64),
            "provenance.source.uv_lock_sha256: ",
        ),
        (
            lambda s: s["provenance"]["figure04_caches"]["input_file_sha256"].update(
                {"inputs.npz": "0" * 64}
            ),
            "provenance.figure04_caches.input_file_sha256.inputs.npz: ",
        ),
        (lambda s: s["configuration"].update(seed=1002), "configuration.seed: 1001 committed"),
        (lambda s: s["configuration"].update(label="other"), "configuration.label: "),
        (lambda s: s["configuration"].update(flag=False), "configuration.flag: True committed"),
        (lambda s: s["counts"].__setitem__(1, 1_502), "counts[1]: 1501 committed, 1502 fresh"),
        (lambda s: s["counts"].append(3), "counts: length 2 committed, 3 fresh"),
        (lambda s: s["means"]["matrix"][1].__setitem__(0, 1e-6), "means.matrix[1][0]: "),
        (lambda s: s["counts"].__setitem__(0, 870_018.0), "counts[0]: type integer committed"),
        (lambda s: s["configuration"].update(flag=1), "configuration.flag: type boolean"),
        (lambda s: s["configuration"].pop("none"), "configuration.none: missing from the fresh"),
        (lambda s: s.update(extra=1), "extra: not in the committed summary"),
    ],
)
def test_each_difference_is_reported_with_its_path(
    check: ModuleType, mutate: Any, expected: str
) -> None:
    fresh = _summary()
    mutate(fresh)
    differences = check.compare_json(_summary(), fresh)
    assert len(differences) == 1
    assert differences[0].startswith(expected)


def test_float_tolerance_is_relative_plus_absolute(check: ModuleType) -> None:
    committed = {"x": 1000.0, "zero": 0.0}
    within = {"x": 1000.0 * (1 + 0.5 * check.RTOL), "zero": 0.5 * check.ATOL}
    assert check.compare_json(committed, within) == []
    beyond = {"x": 1000.0 * (1 + 2 * check.RTOL), "zero": 2 * check.ATOL}
    assert [message.split(":")[0] for message in check.compare_json(committed, beyond)] == [
        "x",
        "zero",
    ]


def test_all_differences_are_reported(check: ModuleType) -> None:
    fresh = copy.deepcopy(_summary())
    fresh["counts"][0] += 1
    fresh["means"]["hpd_overlap"] += 0.01
    fresh["provenance"]["source"]["source_tree_sha256"] = "0" * 64
    assert len(check.compare_json(_summary(), fresh)) == 3


def _write(directory: Path, name: str, payload: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_cli_exit_status_and_report(
    check: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    committed_dir, fresh_dir = tmp_path / "committed", tmp_path / "fresh"
    for name in check.SUMMARY_NAMES:
        _write(committed_dir, name, _summary())
    fresh03 = _write(fresh_dir, "figure03_summary.json", _summary())
    changed = _summary()
    changed["counts"][1] = 0
    fresh04 = _write(fresh_dir, "figure04_summary.json", changed)
    report = tmp_path / "report.txt"

    args = [str(fresh03), str(fresh04), "--committed-dir", str(committed_dir)]
    assert check.main([*args, "--report", str(report)]) == 1
    assert "figure04_summary.json: counts[1]: 1501 committed, 0 fresh" in report.read_text()
    assert "1 difference(s)" in capsys.readouterr().out

    _write(fresh_dir, "figure04_summary.json", _summary())
    assert check.main(args) == 0
    assert "Reproduced: no differences" in capsys.readouterr().out


def test_committed_summaries_reproduce_themselves(check: ModuleType) -> None:
    """The comparison accepts the committed summaries as their own fresh run."""
    for name in check.SUMMARY_NAMES:
        assert check.compare_summary_files(FIGURE_DIR / name, FIGURE_DIR / name) == []
