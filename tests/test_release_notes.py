"""Tests for the release-notes helper used by the release workflow."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tools import release_notes

CHANGELOG = """# Changelog

## [Unreleased]

## [0.2.0] - 2026-12-01

### Added
- Something new.

## [0.1.0] - 2026-11-17

### Added
- First release.
"""


def make_repo(tmp_path: Path, version: str) -> Path:
    (tmp_path / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "{version}"\n', encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG, encoding="utf-8")
    return tmp_path


def test_section_stops_at_the_next_version():
    notes = release_notes.changelog_section(CHANGELOG, "0.2.0")
    assert notes == "### Added\n- Something new."
    assert "First release" in release_notes.changelog_section(CHANGELOG, "0.1.0")
    with pytest.raises(ValueError, match="no notes"):
        release_notes.changelog_section(CHANGELOG, "9.9.9")


def test_tag_must_match_pyproject(tmp_path: Path):
    repo = make_repo(tmp_path, "0.2.0")
    assert "Something new" in release_notes.release_notes("v0.2.0", repo)
    with pytest.raises(ValueError, match="does not match"):
        release_notes.release_notes("v0.1.0", repo)


def test_main_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    repo = make_repo(tmp_path, "0.2.0")
    assert release_notes.main(["v0.2.0"], root=repo) == 0
    assert "Something new" in capsys.readouterr().out
    assert release_notes.main(["v0.3.0"], root=repo) == 1
    assert "does not match" in capsys.readouterr().err


def test_citation_must_name_the_release(tmp_path: Path):
    repo = make_repo(tmp_path, "0.2.0")
    citation = repo / "CITATION.cff"
    citation.write_text('cff-version: 1.2.0\nversion: 0.2.0\ndate-released: "2026-12-01"\n', encoding="utf-8")
    assert "Something new" in release_notes.release_notes("v0.2.0", repo)
    citation.write_text('cff-version: 1.2.0\nversion: 0.1.0\ndate-released: "2026-12-01"\n', encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape("CITATION.cff says version 0.1.0")):
        release_notes.release_notes("v0.2.0", repo)
    citation.write_text("cff-version: 1.2.0\nversion: 0.2.0\ndate-released: 2026-11-17\n", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape("date-released 2026-11-17, but CHANGELOG.md says 2026-12-01")):
        release_notes.release_notes("v0.2.0", repo)


def test_the_real_changelog_has_notes_for_the_current_version():
    """Guards the next release: bump pyproject.toml, CHANGELOG.md and CITATION.cff together."""
    version = release_notes.project_version(release_notes.ROOT / "pyproject.toml")
    assert release_notes.release_notes(f"v{version}")
