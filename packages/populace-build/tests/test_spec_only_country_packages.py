"""Country packages are declarative resource packages, not Python runtimes."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
COUNTRY_PACKAGE_ROOT = ROOT / "packages/populace-build/src/populace/build"
SPEC_SUFFIXES = {".json", ".jsonld"}
FORBIDDEN_EXECUTABLE_TOKENS = frozenset(
    {
        "callable",
        "callback",
        "entry",
        "entry_point",
        "entrypoint",
        "function",
        "handler",
        "import",
        "loader",
        "module",
        "python",
    }
)


def test__given_country_packages__then_no_python_files_are_present() -> None:
    # Given
    country_roots = _country_package_roots()

    # When
    offenders = [
        path.relative_to(ROOT).as_posix()
        for root in country_roots
        for path in root.rglob("*.py")
    ]

    # Then
    assert offenders == []


def test__given_country_packages__then_only_spec_files_are_present() -> None:
    # Given
    country_roots = _country_package_roots()

    # When
    offenders = [
        path.relative_to(ROOT).as_posix()
        for root in country_roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix not in SPEC_SUFFIXES
    ]

    # Then
    assert offenders == []


def test__given_country_package_manifests__then_resources_are_local_specs() -> None:
    # Given
    country_roots = _country_package_roots()

    # When
    offenders: list[str] = []
    for root in country_roots:
        country = root.name
        manifest_path = root / "country_package.json"
        if not manifest_path.exists():
            offenders.append(f"{country}: missing country_package.json")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("country") != country:
            offenders.append(f"{country}: country mismatch")
        resources = set(manifest.get("resources", ()))
        expected_files = resources | {"country_package.json"}
        actual_files = {path.name for path in root.iterdir() if path.is_file()}
        for extra in sorted(actual_files - expected_files):
            offenders.append(f"{country}: unlisted resource {extra}")
        for resource in sorted(resources):
            resource_path = root / resource
            if resource_path.parent != root:
                offenders.append(f"{country}: non-local resource {resource}")
            elif resource_path.suffix not in SPEC_SUFFIXES:
                offenders.append(f"{country}: non-spec resource {resource}")
            elif not resource_path.exists():
                offenders.append(f"{country}: missing resource {resource}")

    # Then
    assert offenders == []


# Shared package-data directories under build/ that are not country packages.
NON_COUNTRY_DIRECTORIES = frozenset({"schemas"})


def test__given_country_like_directories__then_each_has_a_manifest() -> None:
    # Given
    roots = [
        path
        for path in COUNTRY_PACKAGE_ROOT.iterdir()
        if path.is_dir()
        and not path.name.startswith("__")
        and not path.name.endswith("_runtime")
        and path.name not in NON_COUNTRY_DIRECTORIES
    ]

    # When
    offenders = [
        root.relative_to(ROOT).as_posix()
        for root in roots
        if not (root / "country_package.json").exists()
    ]

    # Then
    assert offenders == []


def test__given_country_json_specs__then_no_python_entrypoints_are_declared() -> None:
    # Given
    json_specs = [
        path
        for country_root in _country_package_roots()
        for path in country_root.rglob("*.json")
    ]

    # When
    offenders: list[str] = []
    for path in json_specs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        _collect_executable_content(
            payload, path.relative_to(ROOT).as_posix(), offenders
        )

    # Then
    assert offenders == []


def _collect_executable_content(
    value: object,
    location: str,
    offenders: list[str],
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_location = f"{location}.{key}"
            if _is_executable_key(key):
                offenders.append(child_location)
            _collect_executable_content(child, child_location, offenders)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_executable_content(child, f"{location}[{index}]", offenders)
    elif isinstance(value, str) and _looks_like_python_entrypoint(value):
        offenders.append(location)


def _country_package_roots() -> list[Path]:
    return sorted(
        path.parent for path in COUNTRY_PACKAGE_ROOT.glob("*/country_package.json")
    )


def _is_executable_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
    if normalized in FORBIDDEN_EXECUTABLE_TOKENS:
        return True
    tokens = normalized.split("_")
    return any(token in FORBIDDEN_EXECUTABLE_TOKENS for token in tokens)


def _looks_like_python_entrypoint(value: str) -> bool:
    if "://" in value:
        return False
    return bool(
        re.search(
            r"\b[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+:[a-zA-Z_]\w*\b",
            value,
        )
    )
