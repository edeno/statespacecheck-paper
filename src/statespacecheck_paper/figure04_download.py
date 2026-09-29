"""Download the published Figure-4 input file from Zenodo and verify it.

The file is version 1.0 of the Zenodo record ``paths.FIGURE04_INPUTS_DOI``; it is
saved only if its SHA-256 is ``paths.FIGURE04_INPUTS_SHA256``, the value the
Figure-4 summary records. This is the package's one network download. No figure
code imports this module, so editing it does not touch the Figure-4 pipeline.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
import uuid
from pathlib import Path

from statespacecheck_paper.figure04_input import FIGURE04_INPUTS_FILE, file_sha256
from statespacecheck_paper.paths import FIGURE04_INPUTS_DOI, FIGURE04_INPUTS_SHA256

# The download URL is built from the DOI, so the two cannot name different records.
FIGURE04_INPUTS_RECORD_URL = (
    f"https://zenodo.org/api/records/{FIGURE04_INPUTS_DOI.removeprefix('10.5281/zenodo.')}"
)
_FIGURE04_INPUTS_URL = f"{FIGURE04_INPUTS_RECORD_URL}/files/{FIGURE04_INPUTS_FILE}/content"


def _keep_existing(output: Path, sha256: str) -> Path:
    """Return ``output`` if it is the verified file; otherwise refuse to replace it."""
    if file_sha256(output) != sha256:
        raise FileExistsError(
            f"{output} exists with a different SHA-256 than the published file; "
            "move it aside to download the published one"
        )
    return output


def download_figure04_inputs(
    data_path: str | Path,
    *,
    url: str = _FIGURE04_INPUTS_URL,
    sha256: str = FIGURE04_INPUTS_SHA256,
) -> Path:
    """Download the Figure-4 input file from Zenodo into ``data_path`` and verify it.

    The file (``FIGURE04_INPUTS_FILE``, 75 MB) is version 1.0 of Zenodo record
    ``FIGURE04_INPUTS_DOI``. It is saved under its final name only if its SHA-256
    matches, so a partial or corrupted download never appears under that name,
    and an existing file is never overwritten.

    Parameters
    ----------
    data_path : str or Path
        Directory to save the file in (created if needed).
    url, sha256 : str, keyword-only
        Source and expected SHA-256; the defaults are the published file.

    Returns
    -------
    Path
        The verified file.

    Raises
    ------
    FileExistsError
        If a different file (another SHA-256) has that name, before or after
        the download.
    ValueError
        If the downloaded file's SHA-256 does not match.
    OSError
        If the source cannot be reached (``urllib.error.URLError``, including
        ``HTTPError`` for an HTTP error status), the transfer fails or times out
        (60 s without data), or the connection closes before the announced
        length (``ConnectionError``; Python does not report that itself).
    """
    data_path = Path(data_path)
    output = data_path / FIGURE04_INPUTS_FILE
    if output.exists():
        return _keep_existing(output, sha256)
    data_path.mkdir(parents=True, exist_ok=True)
    # A unique name opened with "xb" gets the default file mode (tempfile's are
    # owner-only). It is removed only after it is closed: Windows cannot delete
    # an open file.
    partial = data_path / f"{FIGURE04_INPUTS_FILE}.{uuid.uuid4().hex}.part"
    try:
        with urllib.request.urlopen(url, timeout=60) as reply, partial.open("xb") as stream:
            shutil.copyfileobj(reply, stream)
            announced = reply.headers.get("Content-Length")
        received = partial.stat().st_size
        if announced is not None and received != int(announced):
            raise ConnectionError(
                f"Download from {url} stopped after {received} of {announced} bytes; run it again"
            )
        downloaded = file_sha256(partial)
        if downloaded != sha256:
            raise ValueError(
                f"File downloaded from {url} has SHA-256 {downloaded}; expected {sha256}"
            )
        # A hard link publishes the file atomically and, unlike a rename, refuses
        # a file that appeared under that name during the download.
        try:
            os.link(partial, output)
        except FileExistsError:
            return _keep_existing(output, sha256)
    finally:
        partial.unlink(missing_ok=True)
    return output
