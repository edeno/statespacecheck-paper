"""Reproduce the committed paper from a clean export of HEAD with empty caches (CLI).

Exports the committed tree (``git archive HEAD``) into ``<work dir>/repo`` and
runs every step there with its own locked environment and an empty data
directory (``STATESPACECHECK_DATA_PATH=<work dir>/data``): download and verify
the Figure-4 input, generate Figures 1-4 and their summaries, emit the macros
(offline), and build the manuscript. It then compares the fresh summaries with
the committed ones (``check_reproduction.py``) and the fresh macro file with
the committed one, byte for byte. The checkout itself is never written, so no
committed artifact needs restoring; uncommitted changes are not reproduced.

Outputs, kept after the run: ``fresh/`` (figures, summaries, macros, PDF),
``committed/`` (HEAD's summaries and macros), ``reproduction_report.txt``, and
``steps.tsv`` (seconds and peak child-process memory per step).
"""

from __future__ import annotations

import argparse
import difflib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES = Path("manuscript") / "figures" / "main"
SUMMARIES = ("figure03_summary.json", "figure04_summary.json")
MACROS = Path("manuscript") / "reported_values.tex"
PDF = Path("manuscript") / "main.pdf"
# Environment variables that would point the fresh run at another environment,
# data directory, or dataset; the run sets or clears them.
_CLEARED_VARIABLES = ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "STATESPACECHECK_ANIMAL_DATE_EPOCH")


def _peak_child_rss_mb() -> float | None:
    """Largest resident set of any finished child process so far, in MB."""
    try:
        import resource
    except ImportError:  # Windows has no resource module
        return None
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # ru_maxrss is in bytes on macOS and kilobytes on Linux.
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def _run(name: str, command: Sequence[str], cwd: Path, env: dict[str, str], log: Path) -> None:
    """Run one step, append its duration and peak memory to ``log``; raise on failure."""
    print(f"\n=== {name}: {' '.join(command)}", flush=True)
    start = time.perf_counter()
    subprocess.run(command, cwd=cwd, env=env, check=True)
    elapsed = time.perf_counter() - start
    peak = _peak_child_rss_mb()
    with log.open("a", encoding="utf-8") as stream:
        stream.write(f"{name}\t{elapsed:.1f}\t{'' if peak is None else f'{peak:.0f}'}\n")


def _directory_size_gb(path: Path) -> float:
    """Total size of the regular files under ``path``, in GB."""
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e9


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fresh reproduction; return 0 if it matches the committed results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Empty or missing directory for the run (default: a new temporary directory).",
    )
    args = parser.parse_args(argv)
    work = args.work_dir or Path(tempfile.mkdtemp(prefix="statespacecheck-fresh-"))
    if work.exists() and any(work.iterdir()):
        parser.error(f"{work} is not empty; the fresh run needs empty caches")
    repo, data, committed, fresh = (work / name for name in ("repo", "data", "committed", "fresh"))
    for directory in (repo, data, committed, fresh):
        directory.mkdir(parents=True)
    log = work / "steps.tsv"
    log.write_text("step\tseconds\tpeak_child_rss_mb\n", encoding="utf-8")

    head = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"Reproducing commit {head} in {work}")
    archive = work / "head.tar"
    subprocess.run(
        ["git", "-C", str(REPO_ROOT), "archive", "--format=tar", f"--output={archive}", head],
        check=True,
    )
    with tarfile.open(archive) as tar:
        tar.extractall(repo, filter="data")
    archive.unlink()
    for name in SUMMARIES:
        shutil.copy2(repo / FIGURES / name, committed / name)
    shutil.copy2(repo / MACROS, committed / MACROS.name)
    (repo / PDF).unlink()  # so the manuscript build cannot report "up to date"

    env = {key: value for key, value in os.environ.items() if key not in _CLEARED_VARIABLES}
    env["STATESPACECHECK_DATA_PATH"] = str(data)
    python = ["uv", "run", "--frozen", "python"]
    _run("sync", ["uv", "sync", "--frozen"], repo, env, log)
    _run("download", [*python, "scripts/download_figure04_inputs.py"], repo, env, log)
    _run("figures", [*python, "scripts/generate_all_figures.py"], repo, env, log)
    _run("macros", [*python, "scripts/emit_reported_values.py"], repo, env, log)
    _run("manuscript", ["make", "-C", "manuscript"], repo, env, log)

    for path in sorted((repo / FIGURES).iterdir()):
        shutil.copy2(path, fresh / path.name)
    shutil.copy2(repo / MACROS, fresh / MACROS.name)
    shutil.copy2(repo / PDF, fresh / PDF.name)
    print(f"Data directory after the run: {_directory_size_gb(data):.1f} GB")

    report = work / "reproduction_report.txt"
    summaries_match = (
        subprocess.run(
            [
                *python,
                "scripts/check_reproduction.py",
                *(str(fresh / name) for name in SUMMARIES),
                f"--committed-dir={committed}",
                f"--report={report}",
            ],
            cwd=repo,
            env=env,
            check=False,
        ).returncode
        == 0
    )
    committed_macros = (committed / MACROS.name).read_text(encoding="utf-8").splitlines(True)
    fresh_macros = (fresh / MACROS.name).read_text(encoding="utf-8").splitlines(True)
    macro_diff = "".join(difflib.unified_diff(committed_macros, fresh_macros, "committed", "fresh"))
    with report.open("a", encoding="utf-8") as stream:
        stream.write(
            f"{MACROS.name}: " + (f"differs\n{macro_diff}" if macro_diff else "identical\n")
        )
    print(f"{MACROS.name}: {'differs' if macro_diff else 'identical'}\n{macro_diff}", end="")
    print(f"\nReport: {report}\nStep timings: {log}\nFresh outputs: {fresh}")
    return 0 if summaries_match and not macro_diff else 1


if __name__ == "__main__":
    raise SystemExit(main())
