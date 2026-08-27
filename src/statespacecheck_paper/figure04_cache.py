"""Figure-4 decoder-output cache: paths, provenance fingerprint, and I/O.

The Figure-4 decode (fit + decode both models + diagnostics) is expensive, so
its results are cached to a single joblib bundle under ``data/intermediates``.
This module owns the cache location (:class:`Figure4Paths`), the provenance
fingerprint that gates a stale cache (:func:`compute_figure04_cache_fingerprint`),
and the load/save helpers with explicit invalid-cache behavior. It imports only
``Figure4Config`` from :mod:`real_data_analysis`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import warnings
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib

from statespacecheck_paper.real_data_analysis import Figure4Config

FIGURE04_CACHE_SCHEMA_VERSION = 4

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
    """Return the installed ``non_local_detector`` version, or ``"unknown"``."""
    try:
        return version("non_local_detector")
    except PackageNotFoundError:
        return "unknown"


def compute_figure04_cache_fingerprint(config: Figure4Config, paths: Figure4Paths) -> str:
    """Provenance fingerprint gating the Figure-4 cache.

    Hashes the schema version, the manuscript decoder parameters
    (:class:`Figure4Config`), the input-data identifier, and the *installed*
    ``non_local_detector`` revision. Any change forces a recompute; the cached
    bundle stores this fingerprint so a stale cache cannot silently produce a
    figure that no longer matches the current method or dependency.

    Bumping :data:`FIGURE04_CACHE_SCHEMA_VERSION` remains the manual override ---
    it is part of the hashed payload, so a bump invalidates every existing cache.
    """
    payload = {
        "schema_version": FIGURE04_CACHE_SCHEMA_VERSION,
        "config": dataclasses.asdict(config),
        "animal_date_epoch": paths.animal_date_epoch,
        "non_local_detector_version": _installed_non_local_detector_version(),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()


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
