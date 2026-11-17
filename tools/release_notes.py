"""Print the release notes for one version from CHANGELOG.md.

Used by .github/workflows/release.yml when a tag such as v0.2.0 is pushed. It
also refuses to continue if the tag does not match the version in
pyproject.toml, so a release can never carry the wrong number.

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
HEADING_RE = re.compile(r"^## \[(?P<version>[^\]]+)\]")


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


def release_notes(tag: str, root: Path = ROOT) -> str:
    version = tag.removeprefix("v")
    expected = project_version(root / "pyproject.toml")
    if version != expected:
        raise ValueError(f"tag {tag} does not match the version in pyproject.toml ({expected})")
    return changelog_section((root / "CHANGELOG.md").read_text(encoding="utf-8"), version)


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
