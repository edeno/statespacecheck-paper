"""Figure-4 decoder-output cache: paths, provenance fingerprint, and I/O.

The Figure-4 decode (fit + decode both models + diagnostics) is expensive, so
its results are cached to a single joblib bundle under ``data/intermediates``.
This module owns the cache location (:class:`Figure4Paths`), the provenance
fingerprint that gates a stale cache (:func:`compute_figure04_cache_fingerprint`),
the corresponding machine-readable input provenance
(:func:`compute_figure04_cache_provenance`), and the load/save helpers with
explicit invalid-cache behavior. It imports
``Figure4Config`` from :mod:`figure04_decoder` and the input-file suffix list
(``EXPORT_FILE_SUFFIXES``) from :mod:`load_local_data`, whose loader owns it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import warnings
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TypedDict

import joblib

from statespacecheck_paper.figure04_decoder import Figure4Config
from statespacecheck_paper.load_local_data import EXPORT_FILE_SUFFIXES

# Version 5 changes the cached HPD/KL event likelihood from normalized
# Poisson(1; lambda) to normalized event intensity. The repository source is
# not part of the cache fingerprint, so this explicit bump is required.
FIGURE04_CACHE_SCHEMA_VERSION = 5

# The decode payload keys shared by the cache dict and :class:`Figure4RenderData`.
# These are the on-disk serialized keys and MUST NOT change without a schema bump
# (``contfrag_*`` is retained as the serialized name even though the in-memory
# render-data fields are spelled ``continuous_fragmented_*``).
_FIGURE04_CACHE_PAYLOAD_KEYS = (
    "continuous_results",
    "contfrag_results",
    "continuous_diagnostics",
    "contfrag_diagnostics",
    "spike_counts",
    "place_field_peaks",
    "diagnostic_place_fields",
    "diagnostic_position_bins",
)


class Figure4CacheArtifactProvenance(TypedDict):
    """Path-independent cache and input identities stored in the summary."""

    schema_version: int
    fingerprint_sha256: str
    non_local_detector_version: str
    export_file_sha256: dict[str, str | None]


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


@dataclasses.dataclass(frozen=True)
class Figure4CacheProvenance:
    """Identity of the validated Figure-4 decode cache and its exported inputs."""

    fingerprint_sha256: str
    schema_version: int
    animal_date_epoch: str
    export_checksums: tuple[tuple[str, str | None], ...]
    non_local_detector_version: str

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
        """Path for the cached Figure-4 decoder outputs (under data/intermediates).

        A single joblib bundle is used rather than netCDF because the decoder
        results carry a ``state_bins`` MultiIndex coordinate, which netCDF cannot
        serialize; joblib (pickle) preserves it exactly.
        """
        return self.data_path / "intermediates" / f"{self.animal_date_epoch}_fig4_cache.joblib"


def _installed_non_local_detector_version() -> str:
    """Return the installed ``non_local_detector`` version for provenance."""
    try:
        return version("non_local_detector")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            "Cannot fingerprint Figure 4 without an installed non_local_detector "
            "distribution; an 'unknown' provenance value could accept the wrong cache."
        ) from exc


def compute_figure04_cache_provenance(
    config: Figure4Config,
    paths: Figure4Paths,
) -> Figure4CacheProvenance:
    """Return the fingerprint and its path-independent provenance components.

    Hashes the schema version, the decode-affecting parameters (the
    :class:`Figure4Config` ``decoder`` and ``provenance`` parts -- but **not**
    ``execution``, which is performance-only and leaves the decode identical),
    the input-data identifier *and the content hashes of the pre-exported input
    files*, and the *installed* ``non_local_detector`` revision. Any change forces
    a recompute; the cached bundle stores this
    fingerprint so a stale cache cannot silently produce a figure that no longer
    matches the current method, input data, or dependency. Hashing the file
    contents (not just ``animal_date_epoch``) is what makes replacing an export
    under the same epoch invalidate the cache rather than reuse a decode of the
    old bytes.

    Bumping :data:`FIGURE04_CACHE_SCHEMA_VERSION` remains the manual override ---
    it is part of the hashed payload, so a bump invalidates every existing cache.
    """
    # Hash only the parameters that change the decode result: the decoder-science
    # config and the recorded provenance. ``config.execution`` (block_size) is a
    # performance-only knob that leaves the decode identical, so it is
    # deliberately excluded -- changing it must not invalidate a cached decode.
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
    )


def compute_figure04_cache_fingerprint(config: Figure4Config, paths: Figure4Paths) -> str:
    """Return the provenance fingerprint gating the Figure-4 decode cache.

    See :func:`compute_figure04_cache_provenance` for the fingerprint inputs.
    """
    return compute_figure04_cache_provenance(config, paths).fingerprint_sha256


def load_figure04_cache(path: Path, expected_fingerprint: str) -> dict[str, object] | None:
    """Load a Figure-4 decode payload from ``path``, or ``None`` on any miss.

    Returns ``None`` (a cache miss) when the file is absent or unreadable, the
    wrapper is not a mapping, its schema/fingerprint does not match, or its keys
    are not exactly ``schema_version`` + ``fingerprint`` + the payload keys. A
    valid load returns only the payload mapping (the eight
    :data:`_FIGURE04_CACHE_PAYLOAD_KEYS`).

    Any failure to read/unpickle the file is treated as a miss, but a
    ``RuntimeWarning`` is emitted so the cause is visible instead of a silent,
    repeating recompute. This deliberately includes the case where an older
    cache references a since-renamed or removed pickled class (which raises
    ``ModuleNotFoundError`` / ``AttributeError`` inside ``joblib.load``): the run
    still proceeds by recomputing, and the recompute overwrites the stale cache
    so the next run loads cleanly.
    """
    if not path.exists():
        return None
    try:
        cached = joblib.load(path)
    except Exception as exc:
        warnings.warn(
            f"Figure-4 cache at {path} could not be read ({exc!r}); treating it "
            "as a miss and recomputing (the recompute will overwrite it).",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if not isinstance(cached, Mapping):
        return None
    expected_keys = {"schema_version", "fingerprint", *_FIGURE04_CACHE_PAYLOAD_KEYS}
    if set(cached.keys()) != expected_keys:
        return None
    if cached.get("schema_version") != FIGURE04_CACHE_SCHEMA_VERSION:
        return None
    if cached.get("fingerprint") != expected_fingerprint:
        return None
    return {key: cached[key] for key in _FIGURE04_CACHE_PAYLOAD_KEYS}


def save_figure04_cache(path: Path, fingerprint: str, payload: Mapping[str, object]) -> None:
    """Write a Figure-4 decode payload to ``path`` with its provenance wrapper.

    Raises ``ValueError`` unless the payload keys are exactly
    :data:`_FIGURE04_CACHE_PAYLOAD_KEYS`, then creates the parent directory and
    stores the ``schema_version`` / ``fingerprint`` wrapper plus the payload.
    """
    if set(payload.keys()) != set(_FIGURE04_CACHE_PAYLOAD_KEYS):
        raise ValueError(
            "save_figure04_cache payload keys must be exactly "
            f"{sorted(_FIGURE04_CACHE_PAYLOAD_KEYS)}; got {sorted(payload.keys())}."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "schema_version": FIGURE04_CACHE_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            **payload,
        },
        path,
    )
