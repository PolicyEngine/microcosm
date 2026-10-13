"""Stable repository paths for tests and shared test support modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TestPaths:
    """Locations associated with one workspace package's tests."""

    repository: Path
    package: Path
    tests: Path


def paths_for(package_name: str) -> TestPaths:
    """Return stable paths for ``package_name`` independent of the caller."""

    package = REPOSITORY_ROOT / "packages" / package_name
    if not package.is_dir():
        raise ValueError(f"unknown workspace package: {package_name}")
    return TestPaths(
        repository=REPOSITORY_ROOT,
        package=package,
        tests=package / "tests",
    )
