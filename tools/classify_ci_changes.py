#!/usr/bin/env python3
"""Classify changed paths for country-specific GitHub Actions jobs."""

from __future__ import annotations

import json
import os
import subprocess
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

API_ROOT = "https://api.github.com"
DOC_PREFIXES = ("docs/",)
DOC_FILES = {
    "AGENTS.md",
    "CLAUDE.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
}


def _scope_for_test(path: PurePosixPath) -> str | None:
    parts = path.parts
    try:
        tests_index = parts.index("tests")
        environment, scope = parts[tests_index + 1 : tests_index + 3]
    except (ValueError, IndexError):
        return None
    if environment == "engine_free" and scope == "shared":
        return "shared"
    if environment in {"engine_free", "engine", "integration"} and scope in {
        "us",
        "uk",
    }:
        return scope
    return None


def scope_for_path(value: str) -> str | None:
    """Return ``shared``, ``us``, ``uk``, or ``None`` for documentation."""

    path = PurePosixPath(value)
    normalized = path.as_posix()
    if normalized in DOC_FILES or normalized.startswith(DOC_PREFIXES):
        return None

    test_scope = _scope_for_test(path)
    if test_scope is not None:
        return test_scope
    if normalized.startswith("build/us/"):
        return "us"
    if normalized.startswith("build/uk/"):
        return "uk"
    if "/src/microcosm/build/us_runtime/" in f"/{normalized}":
        return "us"
    if "/src/microcosm/build/uk_runtime/" in f"/{normalized}":
        return "uk"
    if normalized.startswith("tools/"):
        name = path.name.lower()
        if "us" in name and "uk" not in name:
            return "us"
        if "uk" in name and "us" not in name:
            return "uk"
    return "shared"


def classify_paths(paths: Iterable[str]) -> dict[str, bool]:
    """Return the CI environments affected by an exhaustive changed-path set."""

    values = tuple(dict.fromkeys(paths))
    if not values:
        return {"shared": True, "us": True, "uk": True}
    scopes = {scope_for_path(path) for path in values}
    return {name: name in scopes for name in ("shared", "us", "uk")}


def api_json(url: str, token: str) -> Any:
    """Read one GitHub API response as JSON."""

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def _file_paths(files: Iterable[Mapping[str, object]]) -> list[str]:
    paths: list[str] = []
    for file in files:
        paths.append(str(file["filename"]))
        previous = file.get("previous_filename")
        if previous:
            paths.append(str(previous))
    return paths


def pull_request_paths(
    event: Mapping[str, Any],
    repository: str,
    token: str,
    *,
    request_json: Callable[[str, str], Any] = api_json,
) -> list[str]:
    """Return every current and previous path from a pull request diff."""

    number = event["pull_request"]["number"]
    paths: list[str] = []
    page = 1
    while True:
        files = request_json(
            f"{API_ROOT}/repos/{repository}/pulls/{number}/files"
            f"?per_page=100&page={page}",
            token,
        )
        if not isinstance(files, list):
            raise TypeError("pull request files response must be a list")
        paths.extend(_file_paths(files))
        if len(files) < 100:
            return paths
        page += 1


def push_paths(
    event: Mapping[str, Any],
    repository: str,
    token: str,
    *,
    request_json: Callable[[str, str], Any] = api_json,
) -> list[str]:
    """Return current and previous paths from the push comparison."""

    before = str(event.get("before", ""))
    after = str(event["after"])
    if before and set(before) != {"0"}:
        comparison = request_json(
            f"{API_ROOT}/repos/{repository}/compare/{before}...{after}", token
        )
        if not isinstance(comparison, Mapping):
            raise TypeError("comparison response must be an object")
        files = comparison.get("files", [])
        if not isinstance(files, list):
            raise TypeError("comparison files must be a list")
        return _file_paths(files)
    output = subprocess.check_output(
        ["git", "diff-tree", "--no-commit-id", "--name-only", "-r", after],
        text=True,
    )
    return [line for line in output.splitlines() if line]


def changed_paths(
    event_name: str,
    event: Mapping[str, Any],
    repository: str,
    token: str,
    *,
    request_json: Callable[[str, str], Any] = api_json,
) -> list[str]:
    """Resolve paths from the GitHub event without relying on a shallow clone."""

    if event_name == "pull_request":
        return pull_request_paths(event, repository, token, request_json=request_json)
    if event_name == "push":
        return push_paths(event, repository, token, request_json=request_json)
    raise ValueError(f"unsupported GitHub event: {event_name}")


def main() -> int:
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    paths = changed_paths(
        os.environ["GITHUB_EVENT_NAME"],
        event,
        os.environ["GITHUB_REPOSITORY"],
        os.environ["GITHUB_TOKEN"],
    )
    result = classify_paths(paths)
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
        for name, value in result.items():
            print(f"{name}={str(value).lower()}", file=output)
    print("changed paths:")
    for path in paths:
        print(path)
    print("classification:", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
