"""Shared synthetic Figure-4 values for the orchestration tests."""

from __future__ import annotations

from statespacecheck_paper.figure04_cache import (
    FIGURE04_CACHE_SCHEMA_VERSION,
    FIGURE04_DIAGNOSTICS_SCHEMA_VERSION,
    Figure4CacheProvenance,
)
from statespacecheck_paper.figure04_decoder import Figure4DiagnosticsConfig
from statespacecheck_paper.load_local_data import EXPORT_FILE_SUFFIXES


def synthetic_cache_provenance(animal_date_epoch: str = "epoch_x") -> Figure4CacheProvenance:
    """Cache provenance with placeholder fingerprints and a checksum per input file.

    The decode fingerprint is ``"c" * 64``, the diagnostics fingerprint
    ``"e" * 64``, and every input checksum ``"d" * 64``.
    """
    return Figure4CacheProvenance(
        fingerprint_sha256="c" * 64,
        schema_version=FIGURE04_CACHE_SCHEMA_VERSION,
        animal_date_epoch=animal_date_epoch,
        export_checksums=tuple((suffix, "d" * 64) for suffix in EXPORT_FILE_SUFFIXES),
        non_local_detector_version="1.2.3",
        diagnostics_fingerprint_sha256="e" * 64,
        diagnostics_schema_version=FIGURE04_DIAGNOSTICS_SCHEMA_VERSION,
        statespacecheck_version="0.1.0",
        diagnostics_config=Figure4DiagnosticsConfig(),
    )
