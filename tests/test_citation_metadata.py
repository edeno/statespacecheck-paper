"""The paper's title, authors, ORCIDs, and version must agree wherever they are stated.

They are written by hand in four places: the manuscript (``manuscript/main.tex``:
title, PDF metadata, author block), the website (``site/index.html``: page
title, link preview, heading, author links, BibTeX entry), the citation file
(``CITATION.cff``), and the package metadata (``pyproject.toml``, the version
only). Nothing generates one from another, so these tests hold them together.
"""

from __future__ import annotations

import re
from typing import Any

import pytest
import yaml

from statespacecheck_paper.paths import CITATION_PATH, REPO_ROOT

MAIN_TEX = (REPO_ROOT / "manuscript" / "main.tex").read_text(encoding="utf-8")
INDEX_HTML = (REPO_ROOT / "site" / "index.html").read_text(encoding="utf-8")
PYPROJECT = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def citation() -> dict[str, Any]:
    payload: dict[str, Any] = yaml.safe_load(CITATION_PATH.read_text(encoding="utf-8"))
    return payload


def _one(pattern: str, text: str) -> str:
    """The single capture of ``pattern`` in ``text``; ``.`` spans lines, ``^``/``$`` match lines."""
    matches = re.findall(pattern, text, flags=re.DOTALL | re.MULTILINE)
    assert len(matches) == 1, f"expected one match for {pattern!r}; got {matches}"
    return str(matches[0])


def _full_name(author: dict[str, str]) -> str:
    return f"{author['given-names']} {author['family-names']}"


def _orcid(author: dict[str, str]) -> str:
    return author["orcid"].removeprefix("https://orcid.org/")


def _manuscript_authors() -> list[tuple[str, str]]:
    """``(name, ORCID)`` of each author in ``main.tex``'s author block, in order."""
    block = _one(r"\\author\{%(.*?)^\}", MAIN_TEX)
    return re.findall(r"^\s*(.+?)\\,\\orcidlink\{([^}]*)\}", block, flags=re.MULTILINE)


def _site_authors() -> list[tuple[str, str]]:
    """``(name, ORCID)`` of each author linked in the page's heading, in order."""
    return [
        (name, orcid)
        for orcid, name in re.findall(
            r'<span class="author"><a href="https://orcid.org/([^"]*)">([^<]*)</a>', INDEX_HTML
        )
    ]


@pytest.fixture(scope="module")
def paper_title(citation: dict[str, Any]) -> str:
    title: str = citation["preferred-citation"]["title"]
    return title


@pytest.fixture(scope="module")
def paper_authors(citation: dict[str, Any]) -> list[str]:
    """Author names in paper order, from ``CITATION.cff``'s preferred citation."""
    return [_full_name(author) for author in citation["preferred-citation"]["authors"]]


def test_citation_software_authors_are_the_paper_authors(
    citation: dict[str, Any], paper_authors: list[str], paper_title: str
) -> None:
    assert [_full_name(author) for author in citation["authors"]] == paper_authors
    assert f"'{paper_title}'" in citation["title"]


def test_manuscript_title_and_authors(paper_title: str, paper_authors: list[str]) -> None:
    assert _one(r"\\title\{\\bfseries ([^}]*)\}", MAIN_TEX) == paper_title
    assert _one(r"pdftitle=\{([^}]*)\}", MAIN_TEX) == paper_title
    assert _one(r"pdfauthor=\{([^}]*)\}", MAIN_TEX).split(", ") == paper_authors
    assert [name for name, _ in _manuscript_authors()] == paper_authors


def test_site_title_and_authors(paper_title: str, paper_authors: list[str]) -> None:
    assert _one(r"<title>([^<]*)</title>", INDEX_HTML) == paper_title
    assert _one(r'<meta property="og:title" content="([^"]*)"', INDEX_HTML) == paper_title
    assert _one(r"<h1>([^<]*)</h1>", INDEX_HTML) == paper_title
    assert [name for name, _ in _site_authors()] == paper_authors


def test_site_bibtex_entry(paper_title: str, citation: dict[str, Any]) -> None:
    bibtex = _one(r'<pre class="bibtex" id="bibtex">(.*?)</pre>', INDEX_HTML)
    assert _one(r"title\s*=\s*\{([^}]*)\}", bibtex) == paper_title
    authors = " ".join(_one(r"author\s*=\s*\{([^}]*)\}", bibtex).split())
    assert authors.split(" and ") == [
        f"{author['family-names']}, {author['given-names']}"
        for author in citation["preferred-citation"]["authors"]
    ]
    assert _one(r"year\s*=\s*\{(\d+)\}", bibtex) == str(citation["preferred-citation"]["year"])


def test_orcids_agree(citation: dict[str, Any]) -> None:
    orcids = [(_full_name(author), _orcid(author)) for author in citation["authors"]]
    assert _manuscript_authors() == orcids
    assert _site_authors() == orcids


def test_citation_version_is_the_package_version(citation: dict[str, Any]) -> None:
    """``CITATION.cff`` restates the version that ``pyproject.toml`` sets."""
    project = _one(r"^\[project\]$(.*?)^\[", PYPROJECT)
    assert citation["version"] == _one(r'^version = "([^"]*)"$', project)
