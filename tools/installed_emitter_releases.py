#!/usr/bin/env python3
"""Name the microcosm-emitter releases the installed-emitter CI job installs.

Prints, one per line, the minimum release builds accept
(``MINIMUM_EMITTER_VERSION`` in ``microcosm.build.telemetry_emitter``) and then
the newest release pinned in ``tools/installed-emitter/requirements.txt``, or
that release once when the two are the same. It refuses a pin that does not
name exactly one ``X.Y.Z`` release, or names one older than the minimum.

It also asks the package index uv uses for the newest published release, and
prints a warning, never a failure, when that differs from the pin. When the
index cannot be asked, it skips that check without a message.
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
PACKAGE = "microcosm-emitter"
#: The emitter's lowest supported Python, for resolving its newest release.
PYTHON_VERSION = "3.13"
INDEX_TIMEOUT_SECONDS = 60
_RELEASE = re.compile(r"\d+\.\d+\.\d+")
_PIN = re.compile(rf"{re.escape(PACKAGE)}==(?P<release>\S+)")

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


def published_release(run: Run = subprocess.run) -> str | None:
    """The newest release on uv's package index, or None if it cannot say.

    ``uv pip compile`` resolves the package alone, from the index uv is
    configured with, and refreshes its cached answer for it.
    """
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
                PACKAGE,
            ],
            input=f"{PACKAGE}\n",
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
    for line in answer.stdout.splitlines():
        found = _PIN.fullmatch(line.strip())
        if found is not None and _RELEASE.fullmatch(found["release"]):
            return found["release"]
    return None


def lag_warning(pinned: str, published: str | None) -> str | None:
    """A warning when the index's newest release is not the pinned one."""
    if published is None or published == pinned:
        return None
    return (
        f"::warning::tools/installed-emitter/requirements.txt pins {PACKAGE} "
        f"{pinned}, but the package index's newest release is {published}; raise "
        "the pin so that the installed-emitter job tests it."
    )


def _minimum_emitter_version() -> str:
    from microcosm.build.telemetry_emitter import MINIMUM_EMITTER_VERSION

    return MINIMUM_EMITTER_VERSION


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--minimum", help="the minimum release (default: MINIMUM_EMITTER_VERSION)"
    )
    parser.add_argument("--pin-file", type=Path, default=PIN_FILE)
    parser.add_argument(
        "--no-index-check",
        action="store_true",
        help="do not compare the pin with the package index",
    )
    args = parser.parse_args(argv)
    minimum = args.minimum or _minimum_emitter_version()
    pinned = pinned_release(args.pin_file.read_text(encoding="utf-8"))
    releases = releases_to_install(minimum, pinned)
    if not args.no_index_check:
        warning = lag_warning(pinned, published_release())
        if warning is not None:
            print(warning, file=sys.stderr)
    for release in releases:
        print(release)
    return 0


if __name__ == "__main__":
    sys.exit(main())
