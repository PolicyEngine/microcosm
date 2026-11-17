"""Report the emitter a Route A release would start, against a commit's floor.

The release starts ``microcosm-emitter`` through
``microcosm.build.telemetry_emitter._start_emitter``. This repeats its checks
without importing microcosm, so ``--resolve`` can run before the build
worktree exists:

* the command is ``MICROCOSM_EMITTER_COMMAND`` (stripped), else
  ``microcosm-emitter``, resolved with ``shutil.which``;
* ``<command> version`` must exit 0 within 15 seconds and print a JSON object;
* its ``emitter_version`` must start with a release number at or above the
  floor (``minimum_emitter_version.json``, which must hold exactly ``X.Y.Z``);
* a build that requires telemetry (``MICROCOSM_TELEMETRY_REQUIRED=1``) also
  needs ``protocol_versions`` to be a list of integers holding 1.

The stages run in the build worktree, so a relative command path, which
``shutil.which`` resolves against the working directory, is refused here, and
so is a bare command when ``PATH`` reaches a relative or empty entry (the
working directory) before a directory that holds it: either could find a
different emitter, or none, in the worktree.

usage: check_emitter.py < minimum_emitter_version.json
Prints one line, ``NOTE <text>`` or ``BLOCKER <text>``, and exits 0.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys

COMMAND_VARIABLE = "MICROCOSM_EMITTER_COMMAND"
REQUIRED_VARIABLE = "MICROCOSM_TELEMETRY_REQUIRED"
DEFAULT_COMMAND = "microcosm-emitter"
PROTOCOL_VERSION = 1
TIMEOUT_SECONDS = 15.0
_RELEASE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def _release(text: object) -> tuple[int, ...] | None:
    found = _RELEASE.match(text) if isinstance(text, str) else None
    return None if found is None else tuple(int(part) for part in found.groups())


def _floor(text: str) -> str | None:
    try:
        data = json.loads(text)
    except ValueError:
        return None
    floor = data.get("minimum_emitter_version") if isinstance(data, dict) else None
    if isinstance(floor, str) and _RELEASE.fullmatch(floor):
        return floor
    return None


def _relative_entry(candidate: str) -> str | None:
    """The first relative or empty ``PATH`` entry ``shutil.which`` searches
    before an absolute one that holds ``candidate``, or None."""

    path = os.environ.get("PATH")
    if not path:
        # Unset: which falls back to CS_PATH or os.defpath, both absolute.
        # Empty: which finds nothing.
        return None
    for entry in path.split(os.pathsep):
        if not os.path.isabs(entry):
            return entry
        if shutil.which(candidate, path=entry) is not None:
            return None
    return None


def _problem(floor: str, required: bool) -> tuple[str | None, str]:
    """Why the release would not start an emitter it accepts (None if it
    would), and what the accepted emitter is."""

    named = os.environ.get(COMMAND_VARIABLE, "").strip()
    candidate = named or DEFAULT_COMMAND
    if os.sep in candidate and not os.path.isabs(candidate):
        return (
            f"{COMMAND_VARIABLE} is the relative path {candidate!r}; the stages "
            "run in the build worktree, so give an absolute path"
        ), ""
    if os.sep not in candidate:
        entry = _relative_entry(candidate)
        if entry is not None:
            where = (
                "an empty PATH entry (the working directory)"
                if entry == ""
                else f"the relative PATH entry {entry!r}"
            )
            return (
                f"{where} comes before any directory holding {candidate}; the "
                "stages run in the build worktree, where it could find a "
                "different emitter or none, so make PATH's entries absolute or "
                f"set {COMMAND_VARIABLE} to an absolute path"
            ), ""
    command = shutil.which(candidate)
    if command is None:
        looked = (
            f"{candidate!r}, named by {COMMAND_VARIABLE}"
            if named
            else f"{DEFAULT_COMMAND} on PATH"
        )
        return (
            f"no microcosm-emitter found ({looked}); the release needs {floor} or later"
        ), ""
    try:
        answer = subprocess.run(
            [command, "version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"`{command} version` did not answer within {TIMEOUT_SECONDS:g} s", ""
    except Exception as error:  # as the release does: undecodable output too
        return f"`{command} version` could not be run ({type(error).__name__})", ""
    if answer.returncode != 0:
        return f"`{command} version` exited {answer.returncode}", ""
    try:
        description = json.loads(answer.stdout)
    except ValueError:
        description = None
    if not isinstance(description, dict):
        return f"`{command} version` printed no JSON object", ""
    version = description.get("emitter_version")
    installed = _release(version)
    if installed is None:
        return (
            f"`{command} version` reported no release version ({version!r}); "
            f"the release needs {floor} or later"
        ), ""
    if installed < _release(floor):
        return (
            f"microcosm-emitter {version} at {command} is older than the commit's "
            f"floor {floor} (uv tool install --upgrade 'microcosm-emitter>={floor}', "
            f"or point {COMMAND_VARIABLE} at a newer one)"
        ), ""
    if required:
        versions = description.get("protocol_versions")
        if not isinstance(versions, list) or not all(type(v) is int for v in versions):
            return f"`{command} version` printed no list of protocol versions", ""
        if PROTOCOL_VERSION not in versions:
            return (
                f"`{command} version` lists protocol versions {versions}, not "
                f"protocol version {PROTOCOL_VERSION}, the one the release sends"
            ), ""
    return None, f"microcosm-emitter {version} at {command}"


def check(floor_text: str) -> str:
    """The one line ``check_emitter`` in ``route_a.sh`` reports."""

    setting = os.environ.get(REQUIRED_VARIABLE)
    if setting not in (None, "0", "1"):
        return (
            f"BLOCKER {REQUIRED_VARIABLE} is {setting!r}; the release accepts only "
            "1, 0 or unset and refuses anything else"
        )
    required = setting == "1"
    floor = _floor(floor_text)
    if floor is None:
        return (
            "BLOCKER the commit's minimum_emitter_version.json does not hold "
            '{"minimum_emitter_version": "X.Y.Z"}, so the release would fail to '
            "import its telemetry module"
        )
    problem, accepted = _problem(floor, required)
    if problem is None:
        return f"NOTE {accepted} meets the commit's floor {floor}"
    if required:
        return (
            f"BLOCKER {problem}; {REQUIRED_VARIABLE}=1, so the release would stop "
            "before any work"
        )
    return (
        f"NOTE {problem}; {REQUIRED_VARIABLE} is not 1, so the release would run "
        "without collector telemetry or staging hand-overs"
    )


def main() -> int:
    print(check(sys.stdin.read()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
