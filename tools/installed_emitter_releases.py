#!/usr/bin/env python3
"""Name the microcosm-emitter releases the installed-emitter CI job installs.

Prints, one per line, the minimum release builds accept
(``MINIMUM_EMITTER_VERSION`` in ``microcosm.build.telemetry_emitter``) and then
the newest release pinned in ``tools/installed-emitter/requirements.txt``, or
that release once when the two are the same. It refuses a pin that does not
name exactly one ``X.Y.Z`` release, or names one older than the minimum.

It also asks the package index uv uses for the newest published releases,
and never fails for what it learns:

- a warning when the newest ``microcosm-emitter`` is newer than the pin, and
  a notice when it is older (the pinned release may have been yanked);
- a warning when the newest ``huggingface-hub`` 1.x release, which a new
  emitter install takes, is newer than the release
  ``tools/installed-emitter/constraints.txt`` pins for the stand-in Hub;
- a notice for each comparison it could not make, so that a change in uv's
  answer cannot turn the comparison off unseen.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_FILE = ROOT / "tools" / "installed-emitter" / "requirements.txt"
CONSTRAINTS_FILE = ROOT / "tools" / "installed-emitter" / "constraints.txt"
PACKAGE = "microcosm-emitter"
HUB_PACKAGE = "huggingface-hub"
#: The huggingface-hub releases the emitter accepts (``<2`` in its metadata).
HUB_REQUIREMENT = f"{HUB_PACKAGE}<2"
#: The emitter's lowest supported Python, for resolving its newest release.
PYTHON_VERSION = "3.13"
INDEX_TIMEOUT_SECONDS = 60
_RELEASE = re.compile(r"\d+\.\d+\.\d+")
_PIN = re.compile(rf"{re.escape(PACKAGE)}==(?P<release>\S+)")
_PACKAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

type Run = Callable[..., subprocess.CompletedProcess[str]]


def release_number(release: str) -> tuple[int, ...]:
    """``(major, minor, patch)`` of an ``X.Y.Z`` release."""
    if not _RELEASE.fullmatch(release):
        raise ValueError(f"not an X.Y.Z release: {release!r}")
    return tuple(int(part) for part in release.split("."))


def pinned_release(text: str) -> str:
    """The one release a requirements file pins for ``microcosm-emitter``."""
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    found = _PIN.fullmatch(lines[0]) if len(lines) == 1 else None
    if found is None or not _RELEASE.fullmatch(found["release"]):
        raise SystemExit(
            "::error::tools/installed-emitter/requirements.txt must pin exactly one "
            f"release, as {PACKAGE}==X.Y.Z"
        )
    return found["release"]


def releases_to_install(minimum: str, pinned: str) -> list[str]:
    """The minimum, then the pinned release; one release when they are equal."""
    if release_number(pinned) < release_number(minimum):
        raise SystemExit(
            f"::error::The pinned {PACKAGE} {pinned} is older than "
            f"MINIMUM_EMITTER_VERSION {minimum}; raise it in "
            "tools/installed-emitter/requirements.txt"
        )
    return [minimum] if pinned == minimum else [minimum, pinned]


def constrained_release(text: str, package: str) -> str:
    """The ``X.Y.Z`` release a constraints file pins for ``package``."""
    found = [
        line.strip().split("==", 1)[1]
        for line in text.splitlines()
        if line.strip().startswith(f"{package}==")
    ]
    if len(found) != 1 or not _RELEASE.fullmatch(found[0]):
        raise SystemExit(
            "::error::tools/installed-emitter/constraints.txt must pin exactly one "
            f"release, as {package}==X.Y.Z"
        )
    return found[0]


def newest_release(requirement: str, run: Run = subprocess.run) -> str | None:
    """The newest ``X.Y.Z`` release meeting ``requirement`` on uv's package
    index, or None if it cannot say.

    ``uv pip compile`` resolves the package alone, from the index uv is
    configured with, and refreshes its cached answer for it.
    """
    package = _PACKAGE_NAME.match(requirement)
    if package is None:
        raise ValueError(f"not a requirement: {requirement!r}")
    name = package.group()
    try:
        answer = run(
            [
                "uv",
                "pip",
                "compile",
                "-",
                "--no-deps",
                "--no-header",
                "--no-annotate",
                "--quiet",
                "--python-version",
                PYTHON_VERSION,
                "--refresh-package",
                name,
            ],
            input=f"{requirement}\n",
            capture_output=True,
            text=True,
            timeout=INDEX_TIMEOUT_SECONDS,
            check=False,
            env={**os.environ, "UV_HTTP_TIMEOUT": "10"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if answer.returncode != 0:
        return None
    pinned = re.compile(rf"{re.escape(name)}==(?P<release>\S+)")
    for line in answer.stdout.splitlines():
        found = pinned.fullmatch(line.strip())
        if found is not None and _RELEASE.fullmatch(found["release"]):
            return found["release"]
    return None


def _not_compared(requirement: str, file_name: str) -> str:
    return (
        f"::notice::Could not ask the package index for the newest {requirement}; "
        f"tools/installed-emitter/{file_name} was not compared with it."
    )


def index_messages(
    pinned: str,
    constrained_hub: str,
    newest: Callable[[str], str | None],
) -> list[str]:
    """What the package index says about the pin and the hub constraint."""
    messages = []
    published = newest(PACKAGE)
    if published is None:
        messages.append(_not_compared(PACKAGE, "requirements.txt"))
    elif release_number(published) > release_number(pinned):
        messages.append(
            f"::warning::tools/installed-emitter/requirements.txt pins {PACKAGE} "
            f"{pinned}, but the package index's newest release is {published}; "
            "raise the pin so that the installed-emitter job tests it."
        )
    elif release_number(published) < release_number(pinned):
        messages.append(
            f"::notice::The package index's newest {PACKAGE} is {published}, older "
            f"than the pin {pinned}: the pinned release may have been yanked."
        )
    hub = newest(HUB_REQUIREMENT)
    if hub is None:
        messages.append(_not_compared(HUB_REQUIREMENT, "constraints.txt"))
    elif release_number(hub) > release_number(constrained_hub):
        messages.append(
            "::warning::tools/installed-emitter/constraints.txt pins "
            f"{HUB_PACKAGE} {constrained_hub}, but the package index's newest 1.x "
            f"release is {hub}, which a new emitter install takes. Check the "
            "stand-in Hub against its requests, then raise the constraint."
        )
    return messages


def _minimum_emitter_version() -> str:
    from microcosm.build.telemetry_emitter import MINIMUM_EMITTER_VERSION

    return MINIMUM_EMITTER_VERSION


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minimum", help="the minimum release (default: MINIMUM_EMITTER_VERSION)"
    )
    parser.add_argument("--pin-file", type=Path, default=PIN_FILE)
    parser.add_argument("--constraints-file", type=Path, default=CONSTRAINTS_FILE)
    parser.add_argument(
        "--no-index-check",
        action="store_true",
        help="do not compare the pin and the constraint with the package index",
    )
    args = parser.parse_args(argv)
    minimum = args.minimum or _minimum_emitter_version()
    pinned = pinned_release(args.pin_file.read_text(encoding="utf-8"))
    constrained_hub = constrained_release(
        args.constraints_file.read_text(encoding="utf-8"), HUB_PACKAGE
    )
    releases = releases_to_install(minimum, pinned)
    if not args.no_index_check:
        for message in index_messages(pinned, constrained_hub, newest_release):
            print(message, file=sys.stderr)
    for release in releases:
        print(release)
    return 0


if __name__ == "__main__":
    sys.exit(main())
