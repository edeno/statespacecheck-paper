"""Wiring checks for ``scripts/reproduce_fresh.py`` without running the analyses.

The steps themselves (sync, download, figures, macros, LaTeX, and the summary
comparison) are replaced by stand-ins; the export of HEAD, the work-directory
layout, the environment, and the macro comparison run for real.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests._scripts import load_script


@pytest.fixture
def script() -> ModuleType:
    return load_script("reproduce_fresh")


def _stub_steps(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, *, edit_macros: bool = False
) -> list[tuple[str, list[str], Path, dict[str, str]]]:
    """Record the steps instead of running them; stub the ``uv`` summary check."""
    steps: list[tuple[str, list[str], Path, dict[str, str]]] = []

    def run_step(name: str, command: list[str], cwd: Path, env: dict[str, str], log: Path) -> None:
        steps.append((name, list(command), cwd, env))
        if name == "manuscript":
            (cwd / script.PDF).write_bytes(b"%PDF stand-in")
        if name == "macros" and edit_macros:
            macros = cwd / script.MACROS
            macros.write_text(macros.read_text(encoding="utf-8") + "% changed\n", encoding="utf-8")

    real_run = subprocess.run

    def run(command: list[str], *args: Any, **kwargs: Any) -> Any:
        if command[0] == "uv":  # the summary comparison inside the exported repo
            steps.append(("check", list(command), kwargs["cwd"], kwargs["env"]))
            return subprocess.CompletedProcess(command, 0)
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(script, "_run", run_step)
    monkeypatch.setattr(script.subprocess, "run", run)
    return steps


def test_runs_every_step_in_an_export_of_head_with_empty_data(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/elsewhere/.venv")
    monkeypatch.setenv("STATESPACECHECK_DATA_PATH", "/real/data/with/caches")
    steps = _stub_steps(script, monkeypatch)
    work = tmp_path / "work"

    assert script.main(["--work-dir", str(work)]) == 0

    assert [name for name, *_ in steps] == [
        "sync",
        "download",
        "figures",
        "macros",
        "manuscript",
        "check",
    ]
    for _name, _command, cwd, env in steps:
        assert cwd == work / "repo"
        assert env["STATESPACECHECK_DATA_PATH"] == str(work / "data")
        assert "VIRTUAL_ENV" not in env
    check_command = steps[-1][1]
    assert check_command[-4:] == [
        str(work / "fresh" / "figure03_summary.json"),
        str(work / "fresh" / "figure04_summary.json"),
        f"--committed-dir={work / 'committed'}",
        f"--report={work / 'reproduction_report.txt'}",
    ]
    # The export is the committed tree, not the checkout.
    assert (work / "repo" / "pyproject.toml").is_file()
    assert not (work / "repo" / ".git").exists()
    assert list((work / "data").iterdir()) == []
    for name in script.SUMMARIES:
        assert (work / "committed" / name).is_file()
        assert (work / "fresh" / name).is_file()
    assert (work / "fresh" / "main.pdf").read_bytes() == b"%PDF stand-in"
    report = (work / "reproduction_report.txt").read_text(encoding="utf-8")
    assert "reported_values.tex: identical" in report


def test_a_changed_macro_file_fails(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_steps(script, monkeypatch, edit_macros=True)
    assert script.main(["--work-dir", str(tmp_path / "work")]) == 1
    report = (tmp_path / "work" / "reproduction_report.txt").read_text(encoding="utf-8")
    assert "reported_values.tex: differs" in report
    assert "+% changed" in report


def test_refuses_a_non_empty_work_directory(script: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "leftover").write_text("cache", encoding="utf-8")
    with pytest.raises(SystemExit):
        script.main(["--work-dir", str(tmp_path)])
