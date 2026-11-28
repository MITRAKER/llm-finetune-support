"""Print the release notes for one version from CHANGELOG.md.

Used by .github/workflows/release.yml when a tag such as v0.2.0 is pushed, and
before the workflow creates the tag when it is run by hand. It also refuses to
continue if the tag is not "v" followed by the version in pyproject.toml, or if
CITATION.cff names another version or release date than CHANGELOG.md, so a
release can never carry the wrong number.

Usage:
    python tools/release_notes.py v0.2.0 > release-notes.md
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADING_RE = re.compile(r"^## \[(?P<version>[^\]]+)\](?: - (?P<date>\d{4}-\d{2}-\d{2}))?")


def project_version(pyproject: Path) -> str:
    """The version in pyproject.toml, the single source of truth for the version number."""
    with pyproject.open("rb") as file:
        return str(tomllib.load(file)["project"]["version"])


def changelog_section(changelog: str, version: str) -> str:
    """The text under '## [version]' up to the next '## [' heading (Keep a Changelog format)."""
    notes: list[str] = []
    inside = False
    for line in changelog.splitlines():
        match = HEADING_RE.match(line)
        if match:
            if inside:
                break
            inside = match.group("version") == version
            continue
        if inside:
            notes.append(line)
    text = "\n".join(notes).strip()
    if not text:
        raise ValueError(f"CHANGELOG.md has no notes for version {version}")
    return text


def changelog_date(changelog: str, version: str) -> str | None:
    """The date on the '## [version] - YYYY-MM-DD' heading, or None if it has none."""
    for line in changelog.splitlines():
        match = HEADING_RE.match(line)
        if match and match.group("version") == version:
            return match.group("date")
    return None


def citation_field(citation: str, key: str) -> str | None:
    """A top-level field of CITATION.cff, such as version or date-released (no YAML parser needed)."""
    match = re.search(rf"^{re.escape(key)}:\s*[\"']?([^\"'\s]+)[\"']?\s*$", citation, flags=re.MULTILINE)
    return match.group(1) if match else None


def check_citation(citation: str, version: str, date: str | None) -> None:
    """GitHub's "Cite this repository" box reads CITATION.cff: it must name this release."""
    cited = citation_field(citation, "version")
    if cited != version:
        raise ValueError(f"CITATION.cff says version {cited}, but this release is {version}: update it")
    released = citation_field(citation, "date-released")
    if date is not None and released != date:
        raise ValueError(f"CITATION.cff says date-released {released}, but CHANGELOG.md says {date}: update it")


def release_notes(tag: str, root: Path = ROOT) -> str:
    version = project_version(root / "pyproject.toml")
    if tag != f"v{version}":
        raise ValueError(f"tag {tag} does not match the version in pyproject.toml: it must be v{version}")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    citation = root / "CITATION.cff"
    if citation.exists():
        check_citation(citation.read_text(encoding="utf-8"), version, changelog_date(changelog, version))
    return changelog_section(changelog, version)


def main(argv: list[str] | None = None, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag", help="git tag, e.g. v0.2.0")
    args = parser.parse_args(argv)
    try:
        print(release_notes(args.tag, root))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
