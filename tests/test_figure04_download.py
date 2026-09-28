"""Tests for the Zenodo download of the Figure-4 input file."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, BinaryIO

import pytest

from statespacecheck_paper.figure04_download import (
    FIGURE04_INPUTS_RECORD_URL,
    download_figure04_inputs,
)
from statespacecheck_paper.load_local_data import FIGURE04_INPUTS_FILE
from statespacecheck_paper.paths import DATA_PATH, FIGURE04_INPUTS_DOI, FIGURE04_INPUTS_SHA256

from ._scripts import load_script


@pytest.fixture
def published_copy(tmp_path: Path) -> tuple[str, str]:
    """A stand-in for the Zenodo file: its ``file://`` URL and SHA-256."""
    source = tmp_path / "source.npz"
    source.write_bytes(b"figure 4 inputs")
    return source.as_uri(), hashlib.sha256(b"figure 4 inputs").hexdigest()


def test_download_saves_the_verified_file(tmp_path: Path, published_copy: tuple[str, str]) -> None:
    url, sha256 = published_copy
    output = download_figure04_inputs(tmp_path / "data", url=url, sha256=sha256)
    assert output == tmp_path / "data" / FIGURE04_INPUTS_FILE
    assert output.read_bytes() == b"figure 4 inputs"
    assert list(output.parent.iterdir()) == [output]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission bits")
def test_downloaded_file_gets_the_default_mode(
    tmp_path: Path, published_copy: tuple[str, str]
) -> None:
    """Readable by the group and others on a shared data directory, as the umask allows."""
    url, sha256 = published_copy
    previous = os.umask(0o022)
    try:
        output = download_figure04_inputs(tmp_path, url=url, sha256=sha256)
    finally:
        os.umask(previous)
    assert stat.S_IMODE(output.stat().st_mode) == 0o644


def _fail_mid_transfer(source: BinaryIO, destination: BinaryIO) -> None:
    destination.write(source.read(3))
    raise ConnectionResetError("connection reset")


@pytest.mark.parametrize(
    ("failure", "error"),
    [("unreachable source", urllib.error.URLError), ("mid-transfer", ConnectionResetError)],
)
def test_failed_download_raises_its_error_and_leaves_nothing(
    tmp_path: Path,
    published_copy: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    error: type[Exception],
) -> None:
    """The partial file is closed before it is removed, which Windows requires."""
    url, sha256 = published_copy
    if failure == "unreachable source":
        url = (tmp_path / "missing.npz").as_uri()
    else:
        monkeypatch.setattr(shutil, "copyfileobj", _fail_mid_transfer)
    data = tmp_path / "data"
    with pytest.raises(error):
        download_figure04_inputs(data, url=url, sha256=sha256)
    assert list(data.iterdir()) == []


def test_download_with_the_wrong_checksum_leaves_nothing(
    tmp_path: Path, published_copy: tuple[str, str]
) -> None:
    url, _ = published_copy
    data = tmp_path / "data"
    with pytest.raises(ValueError, match="SHA-256"):
        download_figure04_inputs(data, url=url, sha256="0" * 64)
    assert list(data.iterdir()) == []


class _TruncatingHandler(BaseHTTPRequestHandler):
    """Announces 100 bytes, sends 10, and closes the connection."""

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "100")
        self.end_headers()
        self.wfile.write(b"x" * 10)

    def log_message(self, format: str, *args: Any) -> None:
        pass


def test_interrupted_download_is_reported_as_interrupted(tmp_path: Path) -> None:
    """Python returns the short body without an error; the download must not."""
    server = HTTPServer(("127.0.0.1", 0), _TruncatingHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/"
        with pytest.raises(ConnectionError, match="stopped after 10 of 100 bytes"):
            download_figure04_inputs(tmp_path, url=url, sha256="0" * 64)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert list(tmp_path.iterdir()) == []


def test_existing_verified_file_is_kept_without_downloading(
    tmp_path: Path, published_copy: tuple[str, str]
) -> None:
    _, sha256 = published_copy
    (tmp_path / FIGURE04_INPUTS_FILE).write_bytes(b"figure 4 inputs")
    missing_source = (tmp_path / "missing.npz").as_uri()
    assert download_figure04_inputs(tmp_path, url=missing_source, sha256=sha256).is_file()


def test_existing_different_file_is_not_overwritten(
    tmp_path: Path, published_copy: tuple[str, str]
) -> None:
    url, sha256 = published_copy
    existing = tmp_path / FIGURE04_INPUTS_FILE
    existing.write_bytes(b"something else")
    with pytest.raises(FileExistsError, match="different SHA-256"):
        download_figure04_inputs(tmp_path, url=url, sha256=sha256)
    assert existing.read_bytes() == b"something else"


@pytest.mark.parametrize(
    ("appearing", "kept"), [(b"figure 4 inputs", True), (b"something else", False)]
)
def test_file_appearing_during_the_download_is_not_overwritten(
    tmp_path: Path,
    published_copy: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
    appearing: bytes,
    kept: bool,
) -> None:
    """Another writer's file is kept if it is the verified one, else refused, never replaced."""
    url, sha256 = published_copy
    data = tmp_path / "data"
    data.mkdir()
    output = data / FIGURE04_INPUTS_FILE
    copy = shutil.copyfileobj

    def copy_while_another_writer_saves(source: BinaryIO, destination: BinaryIO) -> None:
        output.write_bytes(appearing)
        copy(source, destination)

    monkeypatch.setattr(shutil, "copyfileobj", copy_while_another_writer_saves)
    if kept:
        assert download_figure04_inputs(data, url=url, sha256=sha256) == output
    else:
        with pytest.raises(FileExistsError, match="different SHA-256"):
            download_figure04_inputs(data, url=url, sha256=sha256)
    assert output.read_bytes() == appearing
    assert list(data.iterdir()) == [output]


@pytest.mark.parametrize("given", [True, False])
def test_download_script_saves_into_the_data_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    given: bool,
) -> None:
    """``--data-path`` reaches the download; without it the file goes to ``DATA_PATH``."""
    script = load_script("download_figure04_inputs")
    requested: list[Path] = []

    def download(data_path: Path) -> Path:
        requested.append(data_path)
        return data_path / FIGURE04_INPUTS_FILE

    monkeypatch.setattr(script, "download_figure04_inputs", download)
    script.main(["--data-path", str(tmp_path)] if given else [])
    expected = tmp_path if given else DATA_PATH
    assert requested == [expected]
    assert capsys.readouterr().out == f"Verified {expected / FIGURE04_INPUTS_FILE}\n"


def test_zenodo_record_is_the_cited_version_with_the_file() -> None:
    """The DOI names one version (not all versions), and that record holds the file."""
    try:
        with urllib.request.urlopen(FIGURE04_INPUTS_RECORD_URL, timeout=30) as reply:
            record = json.load(reply)
    except (urllib.error.URLError, TimeoutError) as err:  # offline, or Zenodo unavailable
        pytest.skip(f"Zenodo cannot be reached: {err}")
    assert record["doi"] == FIGURE04_INPUTS_DOI
    assert record["conceptdoi"] != FIGURE04_INPUTS_DOI
    assert [file["key"] for file in record["files"]] == [FIGURE04_INPUTS_FILE]


def test_published_checksum_is_the_one_figure4_records() -> None:
    """The file the download verifies is the file the Figure-4 summary was made from."""
    summary = json.loads(
        (
            Path(__file__).resolve().parents[1] / "manuscript/figures/main/figure04_summary.json"
        ).read_text(encoding="utf-8")
    )
    recorded = summary["provenance"]["figure04_decode_cache"]["export_file_sha256"]
    assert recorded == {FIGURE04_INPUTS_FILE: FIGURE04_INPUTS_SHA256}
