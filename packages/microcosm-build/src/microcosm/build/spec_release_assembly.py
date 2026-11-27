"""Assemble a spec country's transport build into a release build.

Placeholder for tracker 5b · 13: the assembler is written in the next commit.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "AssembledRelease",
    "SpecReleaseAssemblyError",
    "assemble_spec_release",
    "main",
]


class SpecReleaseAssemblyError(ValueError):
    """The build cannot be assembled into a publishable release build."""


@dataclass(frozen=True)
class AssembledRelease:
    release: str
    build: str
    release_dir: Path
    artifact_root: Path
    publish_command: str
    promote_command: str


def assemble_spec_release(
    build_dir: Path | str,
    *,
    country: str,
    out_dir: Path | str,
    created_at: str | None = None,
) -> AssembledRelease:
    return AssembledRelease("", "", Path(out_dir), Path(), "", "")


def main(argv: Sequence[str] | None = None) -> int:
    print(json.dumps({"release_dir": "", "publish_command": "", "promote_command": ""}))
    return 0
