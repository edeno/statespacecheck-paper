"""Shared synthetic Figure-4 values for the orchestration tests."""

from __future__ import annotations

from statespacecheck_paper.figure04_cache import (
    FIGURE04_DECODE_SCHEMA_VERSION,
    FIGURE04_DIAGNOSTICS_SCHEMA_VERSION,
    Figure4CacheProvenance,
)
from statespacecheck_paper.figure04_decoder import Figure4DiagnosticsConfig


def synthetic_cache_provenance(animal_date_epoch: str = "epoch_x") -> Figure4CacheProvenance:
    """Cache provenance with placeholder fingerprints and input-file checksum.

    The decode fingerprint is ``"c" * 64``, the diagnostics fingerprint
    ``"e" * 64``, and the input-file checksum ``"d" * 64``.
    """
    return Figure4CacheProvenance(
        fingerprint_sha256="c" * 64,
        schema_version=FIGURE04_DECODE_SCHEMA_VERSION,
        animal_date_epoch=animal_date_epoch,
        input_file_sha256="d" * 64,
        non_local_detector_version="1.2.3",
        diagnostics_fingerprint_sha256="e" * 64,
        diagnostics_schema_version=FIGURE04_DIAGNOSTICS_SCHEMA_VERSION,
        statespacecheck_version="0.1.0",
        diagnostics_config=Figure4DiagnosticsConfig(),
        python_version="3.11.0",
        machine="x86_64",
        decode_dependency_versions={"non-local-detector": "1.2.3", "numpy": "2.0.0"},
        diagnostics_dependency_versions={"numpy": "2.0.0", "statespacecheck": "0.1.0"},
    )
