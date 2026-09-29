"""Compare freshly generated Figure-3/4 summaries with the committed ones (CLI).

Every value must match: integers, strings, booleans, and nulls (seeds,
configuration, counts, source and lock digests) exactly, and floating-point
values within ``RTOL``/``ATOL``. Mapping keys, list lengths, and JSON types must
agree. The keys in ``MACHINE_SPECIFIC_PATHS`` are skipped because they
legitimately differ between machines. Every difference is reported with its
JSON path; the exit status is 1 if there is any, else 0.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from statespacecheck_paper.paths import FIGURE_DIR

# Floating-point tolerances: a fresh value ``f`` matches the committed ``c`` when
# ``|f - c| <= ATOL + RTOL * |c|``. The summaries come from seeded, deterministic
# code, so these start near double-precision round-off (a few ulps of
# accumulated arithmetic). They are provisional: set them from a measured
# cross-platform fresh run (for example macOS arm64 against Linux x86_64 CI),
# and record that run's observed maximum differences here.
RTOL = 1e-9
ATOL = 1e-12

# Provenance values that describe the machine rather than the result, so they may
# differ between two correct runs: the Figure-4 cache fingerprints (which hash
# the architecture and installed dependency versions) and the recorded
# environment. The source-tree and uv.lock digests, the input checksum, and the
# package versions are compared exactly.
MACHINE_SPECIFIC_PATHS: frozenset[str] = frozenset(
    {
        "provenance.figure04_caches.fingerprint_sha256",
        "provenance.figure04_caches.diagnostics_fingerprint_sha256",
        "provenance.figure04_caches.python_version",
        "provenance.figure04_caches.machine",
        "provenance.figure04_caches.decode_dependency_versions",
        "provenance.figure04_caches.diagnostics_dependency_versions",
    }
)

SUMMARY_NAMES = ("figure03_summary.json", "figure04_summary.json")


def _type_name(value: object) -> str:
    """JSON type of a parsed value (``bool`` is not a number)."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if value is None:
        return "null"
    if isinstance(value, list):
        return "array"
    return "object"


def compare_json(
    committed: object,
    fresh: object,
    path: str = "",
    *,
    rtol: float = RTOL,
    atol: float = ATOL,
    ignored: frozenset[str] = MACHINE_SPECIFIC_PATHS,
) -> list[str]:
    """Return one message per difference between two parsed JSON values.

    Parameters
    ----------
    committed, fresh : object
        Parsed JSON values (``json.load`` output) to compare.
    path : str
        JSON path of these values, used as the prefix of each message.
    rtol, atol : float
        Tolerances for floating-point values, ``|f - c| <= atol + rtol * |c|``.
    ignored : frozenset[str]
        Dotted paths of mapping entries that are not compared.

    Returns
    -------
    list[str]
        Messages of the form ``"<path>: <description>"``, in document order;
        empty when the values match.
    """
    if path in ignored:
        return []
    where = path or "<root>"
    committed_type, fresh_type = _type_name(committed), _type_name(fresh)
    if committed_type != fresh_type:
        return [f"{where}: type {committed_type} committed, {fresh_type} fresh"]
    if isinstance(committed, dict) and isinstance(fresh, dict):
        differences = [
            f"{path + '.' if path else ''}{key}: missing from the fresh summary"
            for key in sorted(committed.keys() - fresh.keys())
        ]
        differences += [
            f"{path + '.' if path else ''}{key}: not in the committed summary"
            for key in sorted(fresh.keys() - committed.keys())
        ]
        for key in sorted(committed.keys() & fresh.keys()):
            differences += compare_json(
                committed[key],
                fresh[key],
                f"{path}.{key}" if path else key,
                rtol=rtol,
                atol=atol,
                ignored=ignored,
            )
        return differences
    if isinstance(committed, list) and isinstance(fresh, list):
        if len(committed) != len(fresh):
            return [f"{where}: length {len(committed)} committed, {len(fresh)} fresh"]
        differences = []
        for index, (committed_item, fresh_item) in enumerate(zip(committed, fresh, strict=True)):
            differences += compare_json(
                committed_item,
                fresh_item,
                f"{path}[{index}]",
                rtol=rtol,
                atol=atol,
                ignored=ignored,
            )
        return differences
    if isinstance(committed, float) and isinstance(fresh, float):
        difference = abs(fresh - committed)
        if difference > atol + rtol * abs(committed):
            return [
                f"{where}: {committed!r} committed, {fresh!r} fresh (|difference| "
                f"{difference:.3g} exceeds atol {atol:g} + rtol {rtol:g} * |committed|)"
            ]
        return []
    if committed != fresh:
        return [f"{where}: {committed!r} committed, {fresh!r} fresh"]
    return []


def compare_summary_files(committed_path: Path, fresh_path: Path) -> list[str]:
    """Compare two summary files; each message is prefixed with the file name."""
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    return [f"{fresh_path.name}: {message}" for message in compare_json(committed, fresh)]


def main(argv: Sequence[str] | None = None) -> int:
    """Compare the given fresh summaries with the committed ones; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("figure03", type=Path, help="Freshly generated figure03_summary.json.")
    parser.add_argument("figure04", type=Path, help="Freshly generated figure04_summary.json.")
    parser.add_argument(
        "--committed-dir",
        type=Path,
        default=FIGURE_DIR,
        help="Directory holding the committed summaries (default: manuscript/figures/main).",
    )
    parser.add_argument("--report", type=Path, help="Also write the report to this file.")
    args = parser.parse_args(argv)

    differences: list[str] = []
    for name, fresh_path in zip(SUMMARY_NAMES, (args.figure03, args.figure04), strict=True):
        differences += compare_summary_files(args.committed_dir / name, fresh_path)
    lines = [
        f"Compared {', '.join(SUMMARY_NAMES)} with {args.committed_dir}",
        f"Floats: atol {ATOL:g}, rtol {RTOL:g}",
        f"Skipped: {', '.join(sorted(MACHINE_SPECIFIC_PATHS))}",
        *differences,
        f"{len(differences)} difference(s)" if differences else "Reproduced: no differences",
    ]
    report = "\n".join(lines) + "\n"
    print(report, end="")
    if args.report is not None:
        args.report.write_text(report, encoding="utf-8")
    return 1 if differences else 0


if __name__ == "__main__":
    raise SystemExit(main())
