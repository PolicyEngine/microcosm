"""Dry-run a US fiscal-refresh release's gates before launching it.

Give it the release config, meaning the release tool's own argv. It runs
``tools/build_us_fiscal_refresh_release.py`` from this checkout with
``--dry-run-gates-report``. The release's own base load, input stages and
pre-solve gates run as they would in the build. Where the release would then
spend hours on target materialization and the solve, the dry run instead
grades the staged frame at its base weights. It covers every data-dependent
register and every pre-export gate computable from the base: the QRF
tail-concentration waiver register, the export input-mass register, the
degenerate-input register, eCPS parity, input coverage, stored inputs, SPM
composition, zero support and the pre-solve battery. Each one runs through the
release tool's own gate function.

Route A run 310842b986d7 (2026-09-26) failed after 13,707 s on nothing but a
stale QRF tail register. By that run's own ``build.timing``, what this dry run
replays (base load, input stages, pre-solve gates) took at most 2,049 s. What it
skips, target compilation (9,951 s) and calibration (1,668 s), took the rest.

Examples::

    # The argv a supervisor saved (route A's release-config.json "argv"):
    uv run python tools/dry_run_us_release_gates.py \\
        --release-config run/release-config.json \\
        --json-out run/dry-run-gates.json

    # A new base or a candidate register against the same config:
    uv run python tools/dry_run_us_release_gates.py \\
        --release-config run/release-config.json \\
        --base-h5 base-out/base_populace_us_2024_puf_support.h5 \\
        --qrf-tail-concentration-exclusions candidate_register.json \\
        --json-out dry-run-gates.json

    # Or the release arguments themselves, after "--":
    uv run python tools/dry_run_us_release_gates.py --json-out gates.json -- \\
        --base-h5 base.h5 --ledger-facts facts.jsonl --out release \\
        --dense-default-dataset --qrf-tail-concentration-exclusions register.json

Exit code: 1 on any certain failure, 2 on AT-RISK only, 0 clean, and 64 when
the release tool wrote no report because its arguments did not parse. The
report (JSON, plus a table on stdout) states what each verdict rests on. The
certainty model and the evidence behind its default margins are in
``microcosm.build.us_runtime.release_gate_dry_run``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "packages" / "microcosm-build" / "src")
)

_RELEASE_TOOL_NAME = "build_us_fiscal_refresh_release.py"
_REPORT_FLAG = "--dry-run-gates-report"
#: Exit code when the release tool wrote no report (its arguments did not
#: parse), distinct from the report's own 0/1/2 (EX_USAGE).
_EXIT_NO_REPORT = 64

#: Release flags this wrapper can replace in a saved config, and why.
_OVERRIDABLE_FLAGS: tuple[tuple[str, str], ...] = (
    ("--base-h5", "Grade this base H5 instead of the config's."),
    (
        "--qrf-tail-concentration-exclusions",
        "Grade this QRF tail register instead of the config's.",
    ),
    (
        "--release-id",
        "Use this release id (the dry run refuses a certified one, as the "
        "release would).",
    ),
)

#: Dry-run margin options, passed through to the release tool.
_MARGIN_FLAGS: tuple[str, ...] = (
    "--dry-run-tail-share-rise-margin",
    "--dry-run-tail-share-fall-margin",
    "--dry-run-mass-drift-margin",
    "--dry-run-support-nonzero-share-margin",
    "--dry-run-support-carrier-retention",
    "--dry-run-l0-tail-share-rise-margin",
    "--dry-run-l0-tail-share-fall-margin",
)


def release_argv_from_config(payload: object) -> list[str]:
    """The release tool arguments in a saved config.

    Accepts a JSON list of arguments or an object with an ``argv`` list (the
    supervisor's ``release-config.json``). Everything up to and including the
    release tool's script path (the interpreter, ``-B`` and so on) is dropped.
    A list without that path must already be bare release arguments.

    Raises:
        ValueError: On any other shape, or an argv naming a different script.
    """

    argv = payload.get("argv") if isinstance(payload, dict) else payload
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        raise ValueError(
            "A release config must be a JSON list of arguments or an object "
            "whose 'argv' is one."
        )
    for index, token in enumerate(argv):
        if Path(token).name == _RELEASE_TOOL_NAME:
            return list(argv[index + 1 :])
    if argv and not argv[0].startswith("-"):
        raise ValueError(
            f"The config's argv starts with {argv[0]!r} and never names "
            f"{_RELEASE_TOOL_NAME}; pass the release tool's own arguments."
        )
    return list(argv)


def _flag_positions(argv: Sequence[str], flag: str) -> list[int]:
    return [
        index
        for index, token in enumerate(argv)
        if token == flag or token.startswith(f"{flag}=")
    ]


def replace_flag(argv: Sequence[str], flag: str, value: str) -> list[str]:
    """``argv`` with every ``flag`` occurrence replaced by one ``flag value``.

    Handles both ``--flag value`` and ``--flag=value`` spellings.
    """

    kept: list[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        if token == flag:
            skip_next = True
            continue
        if token.startswith(f"{flag}="):
            continue
        kept.append(token)
    return [*kept, flag, value]


def build_release_argv(
    release_argv: Sequence[str],
    *,
    json_out: Path,
    overrides: dict[str, str],
    margins: dict[str, str],
) -> list[str]:
    """The release tool argv for the dry run.

    Raises:
        ValueError: If the release arguments already ask for a dry run.
    """

    if _flag_positions(release_argv, _REPORT_FLAG):
        raise ValueError(
            f"The release arguments already carry {_REPORT_FLAG}; this wrapper "
            "sets it from --json-out."
        )
    argv = list(release_argv)
    for flag, value in overrides.items():
        argv = replace_flag(argv, flag, value)
    for flag, value in margins.items():
        argv = replace_flag(argv, flag, value)
    return [*argv, _REPORT_FLAG, str(json_out)]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dry-run a US fiscal-refresh release: the release's own base load, "
            "input stages and pre-solve gates, then every data-dependent "
            "register and base-computable pre-export gate on the staged frame "
            "at base weights, with no target materialization and no solve."
        ),
        epilog=(
            "Release arguments can follow '--' instead of --release-config. "
            "Exit: 1 on any certain failure, 2 on AT-RISK only, 0 clean, 64 "
            "when the release arguments did not parse (no report)."
        ),
    )
    parser.add_argument(
        "--release-config",
        type=Path,
        help=(
            "JSON release config: a list of release-tool arguments, or an "
            "object whose 'argv' holds the full command (interpreter and "
            "script path are dropped)."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        required=True,
        help="Where to write the machine-readable dry-run report.",
    )
    for flag, meaning in _OVERRIDABLE_FLAGS:
        parser.add_argument(flag, help=meaning)
    for flag in _MARGIN_FLAGS:
        parser.add_argument(
            flag,
            help=(
                "Passed through to the release tool (see its --help for the "
                "default and meaning)."
            ),
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--" in arguments:
        split = arguments.index("--")
        own, trailing = arguments[:split], arguments[split + 1 :]
    else:
        own, trailing = arguments, []
    parser = _parser()
    args = parser.parse_args(own)
    if (args.release_config is None) == (not trailing):
        parser.error(
            "give the release arguments exactly once: --release-config FILE or "
            "'-- <release arguments>'"
        )
    try:
        release_argv = (
            release_argv_from_config(json.loads(args.release_config.read_text()))
            if args.release_config is not None
            else trailing
        )
        overrides = {
            flag: value
            for flag, _meaning in _OVERRIDABLE_FLAGS
            if (value := getattr(args, flag[2:].replace("-", "_"))) is not None
        }
        margins = {
            flag: value
            for flag in _MARGIN_FLAGS
            if (value := getattr(args, flag[2:].replace("-", "_"))) is not None
        }
        dry_run_argv = build_release_argv(
            release_argv,
            json_out=args.json_out,
            overrides=overrides,
            margins=margins,
        )
    except ValueError as error:
        parser.error(str(error))

    from microcosm.build.us_runtime.release_gate_preflight import (
        _release_tool_module,
    )

    release = _release_tool_module()
    print(
        f"Dry-running {release.__file__} (this checkout) with "
        f"{len(dry_run_argv)} arguments; the report goes to {args.json_out}."
    )
    # The report is how a dry-run exit code is told apart from an argparse
    # error, which also exits 2: remove a stale one first.
    args.json_out.unlink(missing_ok=True)
    try:
        release.main(dry_run_argv)
    except SystemExit as exit_:
        if not args.json_out.exists():
            print(
                "The release tool stopped before writing a report (its "
                f"arguments did not parse: exit {exit_.code!r}).",
                file=sys.stderr,
            )
            return _EXIT_NO_REPORT
        return int(exit_.code or 0)
    raise RuntimeError(
        f"{release.__file__} returned without a dry-run exit; it does not "
        f"honor {_REPORT_FLAG}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
