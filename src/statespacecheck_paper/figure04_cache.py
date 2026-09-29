"""Figure-4 decoder-output cache: paths, provenance fingerprints, and I/O.

The Figure-4 decode (fit + decode both models) is expensive, and the per-spike
diagnostics derived from it are comparatively cheap, so the two are cached in
**separate** joblib bundles under ``<data path>/intermediates`` (the data path
is ``data/`` unless ``STATESPACECHECK_DATA_PATH`` is set; see :class:`Figure4Paths`):

- the *decode* bundle (``{epoch}_figure04_decode.joblib``) holds the fitted models'
  decoder outputs (smoothed posterior, predictive distribution, log-likelihood), spike
  counts, and place fields, gated by the
  **decode fingerprint** (:func:`compute_figure04_cache_provenance`): schema,
  decode-affecting configuration, input-data identity and content hashes,
  executable source of the fitting/data-preparation modules, and the installed
  ``non_local_detector`` version;
- the *diagnostics* bundle (``{epoch}_figure04_diagnostics.joblib``) holds the
  per-spike diagnostics for both models, gated by the decode fingerprint **and**
  a **diagnostics fingerprint** (:func:`compute_figure04_diagnostics_fingerprint`):
  the diagnostics schema, the :class:`~figure04_decoder.Figure4DiagnosticsConfig`
  (HPD coverage, event-selection rule), the installed ``statespacecheck``
  version, and a digest of the *executable* source of the diagnostic modules
  (docstrings and comments excluded).

Changes confined to the diagnostic modules or configuration recompute diagnostics
from cached predictions. Changes to decoding, data preparation, or their shared
workflow/place-field modules refit both models.
This module owns the cache locations (:class:`Figure4Paths`), both fingerprints,
the machine-readable provenance record stored in the summary, and the load/save
helpers with explicit invalid-cache behavior. It imports ``Figure4Config`` from
:mod:`figure04_decoder` and the input-file name (``input_file_path``,
``INPUT_FILE_SUFFIX``) from :mod:`figure04_input`, whose loader owns it.
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
from statespacecheck_paper.figure04_input import (
    INPUT_FILE_SUFFIX,
    file_sha256,
    input_file_path,
)

# Decode-bundle schema version, hashed into the decode fingerprint. Bump it to
# invalidate every decode cache when the payload layout changes in a way the
# decode-source digest does not capture.
FIGURE04_DECODE_SCHEMA_VERSION = 6

# Diagnostics-bundle schema version, hashed into the diagnostics fingerprint;
# bump it to invalidate every diagnostics cache.
FIGURE04_DIAGNOSTICS_SCHEMA_VERSION = 2

# Hash entire modules so changes to helpers, imports, defaults, or recording
# preparation cannot silently reuse an old decode. Shared workflow/place-field
# edits conservatively invalidate both caches; diagnostic-only modules below
# remain independent of the expensive decode. Hashing files by name means this
# module need not import figure04_workflow (which imports it).
_DECODE_SOURCE_FILES: tuple[str, ...] = (
    "figure04_decoder.py",
    "figure04_place_fields.py",
    "figure04_workflow.py",
    "figure04_input.py",
)

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
    "continuous_fragmented_results",
    "spike_counts",
    "place_field_peaks",
    "diagnostic_place_fields",
    "diagnostic_position_bins",
)
# The diagnostics payload keys (derived from the decode payload).
_FIGURE04_DIAGNOSTICS_PAYLOAD_KEYS = (
    "continuous_diagnostics",
    "continuous_fragmented_diagnostics",
)
# The full in-memory payload consumed by :class:`Figure4DecodeResults`. These
# are the serialized key spellings, equal to the in-memory field names, and
# MUST NOT change without a schema bump.
_FIGURE04_DECODE_AND_DIAGNOSTICS_PAYLOAD_KEYS = (
    _FIGURE04_DECODE_PAYLOAD_KEYS + _FIGURE04_DIAGNOSTICS_PAYLOAD_KEYS
)


class Figure4CacheArtifactProvenance(TypedDict):
    """Path-independent cache and input identities stored in the summary."""

    schema_version: int
    fingerprint_sha256: str
    non_local_detector_version: str
    input_file_sha256: dict[str, str]
    diagnostics_schema_version: int
    diagnostics_fingerprint_sha256: str
    statespacecheck_version: str
    diagnostics_config: dict[str, object]


# The Figure-4 input file is named by ``figure04_input`` (which owns
# ``INPUT_FILE_SUFFIX``); its content hash goes into the fingerprint so that
# replacing the file under the same ``{epoch}`` prefix invalidates the cache
# instead of silently reusing a decode of the old data.


def _input_file_checksum(paths: Figure4Paths) -> str | None:
    """sha256 of the Figure-4 input file (``None`` when the file is absent).

    A missing file hashes to ``None`` rather than raising, so the fingerprint
    stays well-defined for synthetic/test paths that have no real input file; a
    real run hashes the actual bytes so any data-content change invalidates.
    """
    file_path = input_file_path(paths.data_path, paths.animal_date_epoch)
    if not file_path.exists():
        return None
    return file_sha256(file_path)


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


def _decode_source_digest() -> str:
    """Executable-source digest of the modules that prepare and decode the recording."""
    package_root = Path(__file__).resolve().parent
    return executable_source_digest(tuple(package_root / name for name in _DECODE_SOURCE_FILES))


@dataclasses.dataclass(frozen=True)
class Figure4CacheProvenance:
    """Identity of the validated Figure-4 decode and diagnostics caches and inputs."""

    fingerprint_sha256: str
    schema_version: int
    animal_date_epoch: str
    input_file_sha256: str | None
    non_local_detector_version: str
    diagnostics_fingerprint_sha256: str
    diagnostics_schema_version: int
    statespacecheck_version: str
    diagnostics_config: Figure4DiagnosticsConfig

    def artifact_payload(self) -> Figure4CacheArtifactProvenance:
        """Return path-independent cache provenance for a summary artifact.

        Raises ``ValueError`` unless the input file has a checksum.
        """
        input_file = f"{self.animal_date_epoch}{INPUT_FILE_SUFFIX}"
        if self.input_file_sha256 is None:
            raise ValueError(
                "Canonical Figure 4 provenance requires the input file's checksum; "
                f"{input_file} has none."
            )
        return {
            "schema_version": self.schema_version,
            "fingerprint_sha256": self.fingerprint_sha256,
            "non_local_detector_version": self.non_local_detector_version,
            "input_file_sha256": {input_file: self.input_file_sha256},
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
    def decode_cache_path(self) -> Path:
        """Path of the cached Figure-4 *decode* bundle (under ``data_path/intermediates``).

        A single joblib bundle is used rather than netCDF because the decoder
        results carry a ``state_bins`` MultiIndex coordinate, which netCDF cannot
        serialize; joblib (pickle) preserves it exactly.
        """
        return self.data_path / "intermediates" / f"{self.animal_date_epoch}_figure04_decode.joblib"

    @property
    def diagnostics_cache_path(self) -> Path:
        """Path of the cached Figure-4 per-spike *diagnostics* bundle."""
        return (
            self.data_path
            / "intermediates"
            / f"{self.animal_date_epoch}_figure04_diagnostics.joblib"
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
    ``package_defaults`` parts -- but **not** ``execution``, which is performance-only
    and leaves the decode identical, nor ``diagnostics``, which does not touch
    the fit), the input-data identifier *and the content hash of the
    Figure-4 input file*, the executable source of the fitting and
    data-preparation modules, and the *installed* ``non_local_detector``
    revision. Any change forces a refit; the cached bundle stores this
    fingerprint so a stale cache cannot silently produce a figure that no longer
    matches the current method, input data, or dependency. Hashing the file
    contents (not just ``animal_date_epoch``) is what makes replacing the input
    file under the same epoch invalidate the cache rather than reuse a decode of
    the old bytes.

    Comments and docstrings do not affect the source digest. Edits to shared
    workflow or place-field code conservatively refit; edits confined to
    ``diagnostics.py`` or ``figure04_diagnostics.py`` only recompute diagnostics.

    The *diagnostics* fingerprint is :func:`compute_figure04_diagnostics_fingerprint`.

    Bumping :data:`FIGURE04_DECODE_SCHEMA_VERSION` is the manual override ---
    it is part of the hashed payload, so a bump invalidates every existing cache.
    """
    input_file_sha256 = _input_file_checksum(paths)
    non_local_detector_version = _installed_non_local_detector_version()
    fingerprint_payload = {
        "schema_version": FIGURE04_DECODE_SCHEMA_VERSION,
        "config": {
            "decoder": dataclasses.asdict(config.decoder),
            "package_defaults": dataclasses.asdict(config.package_defaults),
        },
        "animal_date_epoch": paths.animal_date_epoch,
        "input_file_sha256": input_file_sha256,
        "non_local_detector_version": non_local_detector_version,
        "decode_source_digest": _decode_source_digest(),
    }
    blob = json.dumps(fingerprint_payload, sort_keys=True, default=str).encode()
    return Figure4CacheProvenance(
        fingerprint_sha256=hashlib.sha256(blob).hexdigest(),
        schema_version=FIGURE04_DECODE_SCHEMA_VERSION,
        animal_date_epoch=paths.animal_date_epoch,
        input_file_sha256=input_file_sha256,
        non_local_detector_version=non_local_detector_version,
        diagnostics_fingerprint_sha256=compute_figure04_diagnostics_fingerprint(config.diagnostics),
        diagnostics_schema_version=FIGURE04_DIAGNOSTICS_SCHEMA_VERSION,
        statespacecheck_version=_installed_statespacecheck_version(),
        diagnostics_config=config.diagnostics,
    )


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


def load_figure04_decode_cache(path: Path, expected_fingerprint: str) -> dict[str, object] | None:
    """Load a Figure-4 *decode* payload from ``path``, or ``None`` on any miss.

    Returns ``None`` (a cache miss) when the file is absent or unreadable, the
    wrapper is not a mapping, its schema/fingerprint does not match, or its
    keys are not exactly the wrapper keys plus the decode payload keys. A valid
    load returns the decode payload (the six :data:`_FIGURE04_DECODE_PAYLOAD_KEYS`).

    Large arrays are memory-mapped read-only (``mmap_mode="r"``) so a
    multi-gigabyte bundle can be used without materializing it.

    Any failure to read/unpickle the file is treated as a miss, but a
    ``RuntimeWarning`` is emitted so the cause is visible instead of a silent,
    repeating recompute.
    """
    cached = _load_wrapper(path, mmap_mode="r")
    if cached is None:
        return None
    if set(cached.keys()) != {"schema_version", "fingerprint", *_FIGURE04_DECODE_PAYLOAD_KEYS}:
        return None
    if cached.get("schema_version") != FIGURE04_DECODE_SCHEMA_VERSION:
        return None
    if cached.get("fingerprint") != expected_fingerprint:
        return None
    return {key: cached[key] for key in _FIGURE04_DECODE_PAYLOAD_KEYS}


def save_figure04_decode_cache(path: Path, fingerprint: str, payload: Mapping[str, object]) -> None:
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
            "save_figure04_decode_cache payload keys must be exactly "
            f"{sorted(_FIGURE04_DECODE_PAYLOAD_KEYS)}; got {sorted(payload.keys())}."
        )
    _atomic_dump(
        path,
        {
            "schema_version": FIGURE04_DECODE_SCHEMA_VERSION,
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
