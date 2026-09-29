"""Repository paths resolve to the checkout, whatever the working directory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from statespacecheck_paper import paths
from statespacecheck_paper.interactive.cache import figure03_flag_thresholds

REPOSITORY_PATHS = (
    "FIGURE_DIR",
    "FIGURE03_SUMMARY_PATH",
    "FIGURE04_SUMMARY_PATH",
    "MACRO_FILE_PATH",
    "CITATION_PATH",
    "SITE_DATA_DIR",
    "PARITY_FIXTURE_PATH",
)


@pytest.mark.parametrize("name", REPOSITORY_PATHS)
def test_repository_paths_are_committed_files_under_the_root(name: str) -> None:
    path = getattr(paths, name)
    assert path.is_absolute()
    assert path.is_relative_to(paths.REPO_ROOT)
    assert path.exists(), path


def test_figure03_thresholds_load_outside_the_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``interactive.cache build-simulated`` reads Figure 3's flag rules from any directory."""
    monkeypatch.chdir(tmp_path)
    rules = json.loads(paths.FIGURE03_SUMMARY_PATH.read_text(encoding="utf-8"))["flag_rules"]
    assert figure03_flag_thresholds() == {
        metric: rule["threshold"] for metric, rule in rules.items()
    }
