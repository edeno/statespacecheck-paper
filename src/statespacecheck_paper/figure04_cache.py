"""Figure-4 decoder-output cache: paths, provenance fingerprints, and I/O.

The Figure-4 decode (fit + decode both models) is expensive, and the per-spike
diagnostics derived from it are comparatively cheap, so the two are cached in
**separate** joblib bundles under ``data/intermediates``:

- the *decode* bundle (``{epoch}_fig4_cache.joblib``) holds the fitted models'
  predictive/filter outputs, spike counts, and place fields, gated by the
  **decode fingerprint** (:func:`compute_figure04_cache_provenance`): schema,
  decode-affecting configuration, input-data identity and content hashes, and
  the installed ``non_local_detector`` version;
- the *diagnostics* bundle (``{epoch}_fig4_diagnostics.joblib``) holds the
  per-spike diagnostics for both models, gated by the decode fingerprint **and**
  a **diagnostics fingerprint** (:func:`compute_figure04_diagnostics_fingerprint`):
  the diagnostics schema, the :class:`~figure04_decoder.Figure4DiagnosticsConfig`
  (HPD coverage, event-selection rule), the installed ``statespacecheck``
  version, and a digest of the *executable* source of the diagnostic modules
  (docstrings and comments excluded).

A diagnostic implementation or configuration change therefore recomputes only
the diagnostics from the cached predictions; a decoder or data change refits.
This module owns the cache locations (:class:`Figure4Paths`), both fingerprints,
the machine-readable provenance record stored in the summary, and the load/save
helpers with explicit invalid-cache behavior. It imports ``Figure4Config`` from
:mod:`figure04_decoder` and the input-file suffix list (``EXPORT_FILE_SUFFIXES``)
from :mod:`load_local_data`, whose loader owns it.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import warnings
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TypedDict

import joblib

from statespacecheck_paper.figure04_decoder import Figure4Config, Figure4DiagnosticsConfig
from statespacecheck_paper.load_local_data import EXPORT_FILE_SUFFIXES

# Decode-bundle schema. Version 5 changed the cached HPD/KL event likelihood
# from normalized Poisson(1; lambda) to normalized event intensity while the
# diagnostics still lived in the decode bundle; the decode payload itself has
# not changed since, so the decode fingerprint is unchanged by the split.
FIGURE04_CACHE_SCHEMA_VERSION = 5

# Diagnostics-bundle schema. Version 1 is the first separately cached
# diagnostics payload (events binned with the decoder's ``digitize`` rule).
FIGURE04_DIAGNOSTICS_SCHEMA_VERSION = 1

# Source files whose executable content shapes the cached diagnostics. Their
# docstring-stripped syntax trees are hashed into the diagnostics fingerprint,
# so an implementation change recomputes the diagnostics while a comment or
# docstring edit does not.
_DIAGNOSTIC_SOURCE_FILES: tuple[str, ...] = (
    "diagnostics.py",
    "figure04_diagnostics.py",
    "figure04_place_fields.py",
)

# The decode payload keys (the expensive, fitted part).
_FIGURE04_DECODE_PAYLOAD_KEYS = (
    "continuous_results",
    "contfrag_results",
    "spike_counts",
    "place_field_peaks",
    "diagnostic_place_fields",
    "diagnostic_position_bins",
)
# The diagnostics payload keys (derived from the decode payload).
_FIGURE04_DIAGNOSTICS_PAYLOAD_KEYS = (
    "continuous_diagnostics",
    "contfrag_diagnostics",
)
# The full in-memory payload consumed by :class:`Figure4DecodeResults`. These
# are the serialized key spellings and MUST NOT change without a schema bump
# (``contfrag_*`` is retained as the serialized name even though the in-memory
# render-data fields are spelled ``continuous_fragmented_*``).
_FIGURE04_CACHE_PAYLOAD_KEYS = _FIGURE04_DECODE_PAYLOAD_KEYS + _FIGURE04_DIAGNOSTICS_PAYLOAD_KEYS


class Figure4CacheArtifactProvenance(TypedDict):
    """Path-independent cache and input identities stored in the summary."""

    schema_version: int
    fingerprint_sha256: str
    non_local_detector_version: str
    export_file_sha256: dict[str, str | None]
    diagnostics_schema_version: int
    diagnostics_fingerprint_sha256: str
    statespacecheck_version: str
    diagnostics_config: dict[str, object]


# The pre-exported input files are named by ``load_local_data`` (which owns
# ``EXPORT_FILE_SUFFIXES``); their content hashes go into the fingerprint so that
# replacing an export under the same ``{epoch}`` prefix invalidates the cache
# instead of silently reusing a decode of the old data.


def _export_file_checksums(paths: Figure4Paths) -> dict[str, str | None]:
    """sha256 of each pre-exported input file (``None`` when a file is absent).

    A missing file hashes to ``None`` rather than raising, so the fingerprint
    stays well-defined for synthetic/test paths that have no real exports; a
    real run hashes the actual bytes so any data-content change invalidates.
    """
    checksums: dict[str, str | None] = {}
    for suffix in EXPORT_FILE_SUFFIXES:
        file_path = paths.data_path / f"{paths.animal_date_epoch}{suffix}"
        if not file_path.exists():
            checksums[suffix] = None
            continue
        digest = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        checksums[suffix] = digest.hexdigest()
    return checksums


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove docstring statements from every module/class/function body."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and (
            node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            node.body = node.body[1:] or [ast.Pass()]
    return tree


def executable_source_digest(paths: tuple[Path, ...]) -> str:
    """SHA-256 of the docstring-stripped syntax trees of ``paths``, in order.

    Comments never reach the syntax tree and docstrings are removed before
    dumping, so the digest changes only when executable code changes. Line
    endings do not affect the parse, so the digest is stable across checkouts.
    """
    digest = hashlib.sha256()
    for path in paths:
        tree = _strip_docstrings(ast.parse(path.read_text(encoding="utf-8")))
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(ast.dump(tree, annotate_fields=True, include_attributes=False).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _diagnostic_source_digest() -> str:
    """Executable-source digest of the modules that compute the diagnostics."""
    package_root = Path(__file__).resolve().parent
    return executable_source_digest(tuple(package_root / name for name in _DIAGNOSTIC_SOURCE_FILES))


@dataclasses.dataclass(frozen=True)
class Figure4CacheProvenance:
    """Identity of the validated Figure-4 decode and diagnostics caches and inputs."""

    fingerprint_sha256: str
    schema_version: int
    animal_date_epoch: str
    export_checksums: tuple[tuple[str, str | None], ...]
    non_local_detector_version: str
    diagnostics_fingerprint_sha256: str
    diagnostics_schema_version: int
    statespacecheck_version: str
    diagnostics_config: Figure4DiagnosticsConfig

    def artifact_payload(
        self, *, require_complete_inputs: bool = True
    ) -> Figure4CacheArtifactProvenance:
        """Return path-independent cache provenance for a summary artifact."""
        checksum_by_suffix = dict(self.export_checksums)
        if len(checksum_by_suffix) != len(self.export_checksums):
            raise ValueError("Figure 4 provenance contains duplicate export suffixes.")
        missing_suffixes = set(EXPORT_FILE_SUFFIXES) - set(checksum_by_suffix)
        unexpected_suffixes = set(checksum_by_suffix) - set(EXPORT_FILE_SUFFIXES)
        if missing_suffixes or unexpected_suffixes:
            raise ValueError(
                "Figure 4 provenance must identify the canonical input exports; "
                f"missing {sorted(missing_suffixes)}, unexpected {sorted(unexpected_suffixes)}."
            )
        export_file_sha256 = {
            f"{self.animal_date_epoch}{suffix}": checksum
            for suffix, checksum in checksum_by_suffix.items()
        }
        missing = [name for name, checksum in export_file_sha256.items() if checksum is None]
        if require_complete_inputs and missing:
            raise ValueError(
                "Canonical Figure 4 provenance requires every exported input; "
                f"missing checksums for {missing}."
            )
        return {
            "schema_version": self.schema_version,
            "fingerprint_sha256": self.fingerprint_sha256,
            "non_local_detector_version": self.non_local_detector_version,
            "export_file_sha256": export_file_sha256,
            "diagnostics_schema_version": self.diagnostics_schema_version,
            "diagnostics_fingerprint_sha256": self.diagnostics_fingerprint_sha256,
            "statespacecheck_version": self.statespacecheck_version,
            "diagnostics_config": dataclasses.asdict(self.diagnostics_config),
        }


@dataclasses.dataclass(frozen=True)
class Figure4Paths:
    """Injected data-location identifiers for the Figure-4 workflow.

    Threaded into :func:`prepare_figure04_render_data` instead of reading the
    module-global ``DATA_PATH`` / ``ANIMAL_DATE_EPOCH`` so the compute/load
    layer is testable with synthetic inputs and a temporary cache directory.
    """

    data_path: Path
    animal_date_epoch: str

    @property
    def cache_path(self) -> Path:
        """Path of the cached Figure-4 *decode* bundle (under data/intermediates).

        A single joblib bundle is used rather than netCDF because the decoder
        results carry a ``state_bins`` MultiIndex coordinate, which netCDF cannot
        serialize; joblib (pickle) preserves it exactly.
        """
        return self.data_path / "intermediates" / f"{self.animal_date_epoch}_fig4_cache.joblib"

    @property
    def diagnostics_cache_path(self) -> Path:
        """Path of the cached Figure-4 per-spike *diagnostics* bundle."""
        return (
            self.data_path / "intermediates" / f"{self.animal_date_epoch}_fig4_diagnostics.joblib"
        )


def _installed_version(distribution: str) -> str:
    """Return an installed distribution's version for provenance."""
    try:
        return version(distribution)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"Cannot fingerprint Figure 4 without an installed {distribution} "
            "distribution; an 'unknown' provenance value could accept the wrong cache."
        ) from exc


def _installed_non_local_detector_version() -> str:
    """Return the installed ``non_local_detector`` version for provenance."""
    return _installed_version("non_local_detector")


def _installed_statespacecheck_version() -> str:
    """Return the installed ``statespacecheck`` version for provenance."""
    return _installed_version("statespacecheck")


def compute_figure04_diagnostics_fingerprint(config: Figure4DiagnosticsConfig) -> str:
    """Return the fingerprint gating the Figure-4 *diagnostics* cache.

    Hashes the diagnostics schema version, the diagnostics configuration (HPD
    coverage and event-selection rule), the installed ``statespacecheck``
    version, and the executable-source digest of the diagnostic modules. It
    deliberately excludes the decode identity: the diagnostics bundle stores
    the decode fingerprint alongside this one, and both must match.
    """
    payload = {
        "diagnostics_schema_version": FIGURE04_DIAGNOSTICS_SCHEMA_VERSION,
        "diagnostics_config": dataclasses.asdict(config),
        "statespacecheck_version": _installed_statespacecheck_version(),
        "diagnostic_source_digest": _diagnostic_source_digest(),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def compute_figure04_cache_provenance(
    config: Figure4Config,
    paths: Figure4Paths,
) -> Figure4CacheProvenance:
    """Return both fingerprints and their path-independent provenance components.

    The *decode* fingerprint hashes the decode schema version, the
    decode-affecting parameters (the :class:`Figure4Config` ``decoder`` and
    ``provenance`` parts -- but **not** ``execution``, which is performance-only
    and leaves the decode identical, nor ``diagnostics``, which does not touch
    the fit), the input-data identifier *and the content hashes of the
    pre-exported input files*, and the *installed* ``non_local_detector``
    revision. Any change forces a refit; the cached bundle stores this
    fingerprint so a stale cache cannot silently produce a figure that no longer
    matches the current method, input data, or dependency. Hashing the file
    contents (not just ``animal_date_epoch``) is what makes replacing an export
    under the same epoch invalidate the cache rather than reuse a decode of the
    old bytes.

    The *diagnostics* fingerprint is :func:`compute_figure04_diagnostics_fingerprint`.

    Bumping :data:`FIGURE04_CACHE_SCHEMA_VERSION` remains the manual override ---
    it is part of the hashed payload, so a bump invalidates every existing cache.
    """
    export_checksums = _export_file_checksums(paths)
    non_local_detector_version = _installed_non_local_detector_version()
    fingerprint_payload = {
        "schema_version": FIGURE04_CACHE_SCHEMA_VERSION,
        "config": {
            "decoder": dataclasses.asdict(config.decoder),
            "provenance": dataclasses.asdict(config.provenance),
        },
        "animal_date_epoch": paths.animal_date_epoch,
        "export_checksums": export_checksums,
        "non_local_detector_version": non_local_detector_version,
    }
    blob = json.dumps(fingerprint_payload, sort_keys=True, default=str).encode()
    return Figure4CacheProvenance(
        fingerprint_sha256=hashlib.sha256(blob).hexdigest(),
        schema_version=FIGURE04_CACHE_SCHEMA_VERSION,
        animal_date_epoch=paths.animal_date_epoch,
        export_checksums=tuple(export_checksums.items()),
        non_local_detector_version=non_local_detector_version,
        diagnostics_fingerprint_sha256=compute_figure04_diagnostics_fingerprint(config.diagnostics),
        diagnostics_schema_version=FIGURE04_DIAGNOSTICS_SCHEMA_VERSION,
        statespacecheck_version=_installed_statespacecheck_version(),
        diagnostics_config=config.diagnostics,
    )


def compute_figure04_cache_fingerprint(config: Figure4Config, paths: Figure4Paths) -> str:
    """Return the decode fingerprint gating the Figure-4 decode cache.

    See :func:`compute_figure04_cache_provenance` for the fingerprint inputs.
    """
    return compute_figure04_cache_provenance(config, paths).fingerprint_sha256


def _load_wrapper(path: Path, *, mmap_mode: str | None) -> Mapping[str, object] | None:
    """Read a joblib cache wrapper, treating any read failure as a miss."""
    if not path.exists():
        return None
    try:
        cached = joblib.load(path, mmap_mode=mmap_mode)
    except Exception as exc:
        warnings.warn(
            f"Figure-4 cache at {path} could not be read ({exc!r}); treating it "
            "as a miss and recomputing (the recompute will overwrite it).",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
    if not isinstance(cached, Mapping):
        return None
    return cached


def load_figure04_cache(
    path: Path,
    expected_fingerprint: str,
    *,
    mmap_mode: str | None = "r",
) -> dict[str, object] | None:
    """Load a Figure-4 *decode* payload from ``path``, or ``None`` on any miss.

    Returns ``None`` (a cache miss) when the file is absent or unreadable, the
    wrapper is not a mapping, its schema/fingerprint does not match, or it does
    not carry every decode payload key. A valid load returns only the decode
    payload (the six :data:`_FIGURE04_DECODE_PAYLOAD_KEYS`).

    Legacy bundles that also embed the diagnostics keys are accepted as a
    decode payload (their embedded diagnostics are ignored; the diagnostics
    bundle is the sole diagnostics source), so a pre-split cache continues to
    serve the expensive decode while its diagnostics are recomputed once.

    Large arrays are memory-mapped by default (``mmap_mode="r"``) so a
    multi-gigabyte bundle can be inspected without materializing it; pass
    ``mmap_mode=None`` to load everything into memory.

    Any failure to read/unpickle the file is treated as a miss, but a
    ``RuntimeWarning`` is emitted so the cause is visible instead of a silent,
    repeating recompute.
    """
    cached = _load_wrapper(path, mmap_mode=mmap_mode)
    if cached is None:
        return None
    required_keys = {"schema_version", "fingerprint", *_FIGURE04_DECODE_PAYLOAD_KEYS}
    allowed_keys = required_keys | set(_FIGURE04_DIAGNOSTICS_PAYLOAD_KEYS)
    keys = set(cached.keys())
    if not required_keys <= keys or not keys <= allowed_keys:
        return None
    if cached.get("schema_version") != FIGURE04_CACHE_SCHEMA_VERSION:
        return None
    if cached.get("fingerprint") != expected_fingerprint:
        return None
    return {key: cached[key] for key in _FIGURE04_DECODE_PAYLOAD_KEYS}


def save_figure04_cache(path: Path, fingerprint: str, payload: Mapping[str, object]) -> None:
    """Write a Figure-4 *decode* payload to ``path`` with its provenance wrapper.

    Raises ``ValueError`` unless the payload keys are exactly
    :data:`_FIGURE04_DECODE_PAYLOAD_KEYS`, then creates the parent directory and
    stores the ``schema_version`` / ``fingerprint`` wrapper plus the payload.
    The bundle is written to a temporary sibling and atomically renamed into
    place, so a bundle being read (possibly memory-mapped) is never overwritten
    in place.
    """
    if set(payload.keys()) != set(_FIGURE04_DECODE_PAYLOAD_KEYS):
        raise ValueError(
            "save_figure04_cache payload keys must be exactly "
            f"{sorted(_FIGURE04_DECODE_PAYLOAD_KEYS)}; got {sorted(payload.keys())}."
        )
    _atomic_dump(
        path,
        {
            "schema_version": FIGURE04_CACHE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            **payload,
        },
    )


def load_figure04_diagnostics_cache(
    path: Path,
    expected_fingerprint: str,
    expected_diagnostics_fingerprint: str,
) -> dict[str, object] | None:
    """Load a Figure-4 *diagnostics* payload from ``path``, or ``None`` on a miss.

    The bundle must carry the diagnostics schema version, the decode
    fingerprint of the predictions it was derived from, and the diagnostics
    fingerprint; all three must match, and the keys must be exactly the wrapper
    plus :data:`_FIGURE04_DIAGNOSTICS_PAYLOAD_KEYS`.
    """
    cached = _load_wrapper(path, mmap_mode=None)
    if cached is None:
        return None
    expected_keys = {
        "diagnostics_schema_version",
        "fingerprint",
        "diagnostics_fingerprint",
        *_FIGURE04_DIAGNOSTICS_PAYLOAD_KEYS,
    }
    if set(cached.keys()) != expected_keys:
        return None
    if cached.get("diagnostics_schema_version") != FIGURE04_DIAGNOSTICS_SCHEMA_VERSION:
        return None
    if cached.get("fingerprint") != expected_fingerprint:
        return None
    if cached.get("diagnostics_fingerprint") != expected_diagnostics_fingerprint:
        return None
    return {key: cached[key] for key in _FIGURE04_DIAGNOSTICS_PAYLOAD_KEYS}


def save_figure04_diagnostics_cache(
    path: Path,
    fingerprint: str,
    diagnostics_fingerprint: str,
    payload: Mapping[str, object],
) -> None:
    """Write a Figure-4 *diagnostics* payload to ``path`` with both fingerprints.

    Raises ``ValueError`` unless the payload keys are exactly
    :data:`_FIGURE04_DIAGNOSTICS_PAYLOAD_KEYS`.
    """
    if set(payload.keys()) != set(_FIGURE04_DIAGNOSTICS_PAYLOAD_KEYS):
        raise ValueError(
            "save_figure04_diagnostics_cache payload keys must be exactly "
            f"{sorted(_FIGURE04_DIAGNOSTICS_PAYLOAD_KEYS)}; got {sorted(payload.keys())}."
        )
    _atomic_dump(
        path,
        {
            "diagnostics_schema_version": FIGURE04_DIAGNOSTICS_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "diagnostics_fingerprint": diagnostics_fingerprint,
            **payload,
        },
    )


def _atomic_dump(path: Path, payload: Mapping[str, object]) -> None:
    """``joblib.dump`` to a temporary sibling, then rename over ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    joblib.dump(dict(payload), tmp_path)
    tmp_path.replace(path)
