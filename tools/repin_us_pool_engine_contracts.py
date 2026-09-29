#!/usr/bin/env python3
"""Re-derive the multispine pool's PolicyEngine-US engine contracts.

``microcosm.build.us_runtime.multispine_pool`` pins five engine-dependent
quantities, each of which fails the pool closed when the installed engine moves:

* ``POOL_SSI_DEPENDENCY_CONTRACT`` — the exact static SSI graph the terminal
  agreement simulation consumes (leaf/node/edge counts and a content digest);
* ``POOL_ENGINE_INPUT_PROJECTION_CONTRACT`` — every installed simulation input
  and its declared default (counts plus two digests);
* ``POOL_SSI_INPUT_PROVISION_COUNTS`` — how each SSI input leaf is provisioned;
* ``POOL_REMAINING_STAGE_INPUT_MANIFEST_SHA256`` — the content digest of every
  post-transfer consumer/input row; and
* the row count quoted in that digest's docstring.

Each is computed by the same code that asserts it, so the only honest way to
move one is to recompute it against the installed wheel.  In-process patching
cannot do that: ``microcosm.build.us_runtime`` builds its spine-agreement
registry at import time, which loads the take-up contract, which asserts the
engine ABI lock, which walks the SSI closure — so the first stale contract
raises before any Python in this process can rebind it.

So the re-derivation is done the way the guards themselves are written: run a
child process that imports the package and computes, read the ``observed=``
payload out of whichever guard failed, write that value into the module, and
run again.  Every value written is one the installed engine produced; none is
transcribed from anywhere else.  The loop converges when the child completes,
and each round prints the old -> new transition so review sees the evidence.

A moved count always has a cause in the engine's changelog, and the bump lane
must name it; this tool supplies the numbers, not the justification.

    uv run python tools/repin_us_pool_engine_contracts.py --check
    uv run python tools/repin_us_pool_engine_contracts.py
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPOSITORY_ROOT
    / "packages"
    / "microcosm-build"
    / "src"
    / "microcosm"
    / "build"
    / "us_runtime"
    / "multispine_pool.py"
)
#: A stale contract can need at most one round per pinned quantity; the input
#: projection is guarded twice (digest, then defaults), and one round is spent
#: regenerating the engine ABI lock.
MAX_ROUNDS = 12


# --------------------------------------------------------------------------
# Child process: import the package and compute everything the guards allow.
# --------------------------------------------------------------------------


def _emit_json() -> int:
    from importlib.metadata import version

    from microcosm.build.us_runtime import multispine_pool as pool
    from microcosm.frame.adapters.policyengine_us import (
        PolicyEngineUSVariableMetadataIndex,
    )

    index = PolicyEngineUSVariableMetadataIndex()
    # Both guarded receipts run here so that a stale pin still in the file
    # raises now, with its own ``observed=`` payload, rather than silently
    # leaving a quantity unrefreshed.
    pool.pool_engine_input_projection_receipt()
    manifest = pool.pool_remaining_stage_input_manifest(index)
    rows = [
        {
            "stage": entry.stage,
            "consumer": entry.consumer,
            "entity": entry.entity,
            "variable": entry.variable,
            "execution_scope": entry.execution_scope,
            "provision": entry.provision,
            "available_by": entry.available_by,
            "fallback": entry.fallback,
        }
        for entry in manifest
    ]
    manifest_sha256 = hashlib.sha256(
        json.dumps(rows, allow_nan=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()
    sys.stdout.write(
        json.dumps(
            {
                "engine_version": version("policyengine-us"),
                "manifest_sha256": manifest_sha256,
                "manifest_rows": len(manifest),
            }
        )
    )
    return 0


# --------------------------------------------------------------------------
# Parent process: run the child, absorb one observation per round.
# --------------------------------------------------------------------------


#: Each fail-closed guard's message prefix -> the literal it pins. Dispatching
#: on the message rather than on the observed payload's shape is load-bearing:
#: two different guards report provision counts, and the SSI closure and the
#: input projection both report an ``engine_version``.
_GUARDS: tuple[tuple[str, str], ...] = (
    ("SSI dependency closure drifted", "ssi_contract"),
    ("SSI input-leaf provisioning drifted", "ssi_provisions"),
    (
        "Simulation input-projection provisioning drifted",
        "projection_provisions",
    ),
    ("simulation input projection drifted", "projection_contract"),
    ("simulation input defaults drifted", "projection_contract"),
    ("Remaining-stage input manifest drifted", "manifest_digest"),
)

_PROVISION_BLOCKS = {
    "ssi_provisions": "POOL_SSI_INPUT_PROVISION_COUNTS",
    "projection_provisions": "POOL_PROJECTION_INPUT_PROVISION_COUNTS",
}
_CONTRACT_BLOCKS = {
    "ssi_contract": "POOL_SSI_DEPENDENCY_CONTRACT = PoolSsiDependencyContract",
    "projection_contract": (
        "POOL_ENGINE_INPUT_PROJECTION_CONTRACT = PoolEngineInputProjectionContract"
    ),
}


def _observed(stderr: str) -> tuple[str, object] | None:
    """Pull one guard's identity and ``observed=`` payload from its message."""

    text = stderr.strip()
    kind = next((name for prefix, name in _GUARDS if prefix in text), None)
    if kind is None:
        return None
    match = re.search(r"observed=(\{.*\})\.\s*$", text, re.DOTALL)
    if match is not None:
        try:
            return kind, ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError):  # pragma: no cover - defensive
            return None
    match = re.search(r"observed=([0-9a-f]{64})\.\s*$", text)
    return (kind, match.group(1)) if match is not None else None


def _field(text: str, block: str, field: str) -> str:
    """Return one keyword argument's rendered value inside a constant block."""

    match = re.search(rf"{re.escape(block)}\((?s:.*?)\)\n", text)
    if match is None:
        raise SystemExit(f"Cannot locate the {block} literal in {MODULE_PATH}.")
    value = re.search(rf"\n    {field}=(.*?),\n", match.group(0))
    if value is None:
        raise SystemExit(f"{block} has no {field} argument.")
    return value.group(1)


def _set_field(text: str, block: str, field: str, rendered: str) -> str:
    match = re.search(rf"{re.escape(block)}\((?s:.*?)\)\n", text)
    assert match is not None
    updated_block = re.sub(
        rf"(\n    {field}=)(.*?)(,\n)",
        lambda item: item.group(1) + rendered + item.group(3),
        match.group(0),
        count=1,
    )
    return text[: match.start()] + updated_block + text[match.end() :]


def _render(value: object) -> str:
    return f'"{value}"' if isinstance(value, str) else str(value)


def _apply(text: str, kind: str, observed: object) -> tuple[str, list[str]]:
    """Write one guard's observation into the module, naming what moved."""

    notes: list[str] = []
    if kind == "manifest_digest":
        pattern = r'POOL_REMAINING_STAGE_INPUT_MANIFEST_SHA256 = \(\n    "(\w+)"\n\)\n'
        match = re.search(pattern, text)
        if match is None:
            raise SystemExit("Cannot locate the remaining-stage manifest digest.")
        if match.group(1) != observed:
            notes.append(
                f"remaining-stage manifest digest {match.group(1)[:12]}... -> "
                f"{str(observed)[:12]}..."
            )
        return (
            text[: match.start()]
            + "POOL_REMAINING_STAGE_INPUT_MANIFEST_SHA256 = (\n"
            + f'    "{observed}"\n)\n'
            + text[match.end() :],
            notes,
        )
    if not isinstance(observed, dict):
        raise SystemExit(f"Unrecognized {kind} observation: {observed!r}")

    if kind in _PROVISION_BLOCKS:
        symbol = _PROVISION_BLOCKS[kind]
        pattern = rf"{symbol}: tuple\[tuple\[str, int\], \.\.\.\] = \((?s:.*?)\)\n"
        match = re.search(pattern, text)
        if match is None:
            raise SystemExit(f"Cannot locate {symbol}.")
        rendered = "\n".join(
            f'    ("{name}", {count}),' for name, count in sorted(observed.items())
        )
        replacement = f"{symbol}: tuple[tuple[str, int], ...] = (\n{rendered}\n)\n"
        if match.group(0) != replacement:
            notes.append(f"{symbol} -> {dict(sorted(observed.items()))}")
        return text[: match.start()] + replacement + text[match.end() :], notes

    block = _CONTRACT_BLOCKS[kind]
    for field, value in observed.items():
        current = _field(text, block, field)
        rendered = _render(value)
        if current != rendered:
            notes.append(f"{block.split(' = ')[0]}.{field} {current} -> {rendered}")
            text = _set_field(text, block, field, rendered)
    return text, notes


#: The engine ABI lock is generated from the same fresh manifest these
#: contracts feed, and ``microcosm.build.us_runtime`` asserts the lock at import
#: time, so a stale lock blocks the child from ever reaching the manifest
#: digest. It is not this tool's pin, but it sits inside this tool's dependency
#: chain, so the loop invokes the lock's own generator when it hits it rather
#: than handing the operator a half-converged tree.
_STALE_ENGINE_ABI_LOCK = "generated lock is stale or non-canonical"
_ENGINE_ABI_LOCK_GENERATOR = (
    "tools/generate_us_bundle_from_constants.py",
    "--engine-lock-only",
)


def _regenerate_engine_abi_lock() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *(_ENGINE_ABI_LOCK_GENERATOR)],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    )


def _run_child() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--emit-json"],
        capture_output=True,
        text=True,
        cwd=REPOSITORY_ROOT,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="refuse if the checked-in contracts differ from the installed engine",
    )
    parser.add_argument("--emit-json", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.emit_json:
        return _emit_json()

    original = MODULE_PATH.read_text()
    text = original
    notes: list[str] = []
    for _round in range(MAX_ROUNDS):
        MODULE_PATH.write_text(text)
        completed = _run_child()
        if completed.returncode == 0:
            derived = json.loads(completed.stdout)
            break
        if _STALE_ENGINE_ABI_LOCK in completed.stderr:
            MODULE_PATH.write_text(text)
            regenerated = _regenerate_engine_abi_lock()
            if regenerated.returncode != 0:
                MODULE_PATH.write_text(original)
                sys.stderr.write(regenerated.stderr)
                raise SystemExit(
                    "The engine ABI lock generator failed; nothing was written."
                )
            notes.append("regenerated the US engine ABI lock")
            continue
        observation = _observed(completed.stderr)
        if observation is None:
            MODULE_PATH.write_text(original)
            sys.stderr.write(completed.stderr)
            raise SystemExit(
                "The pool contract child failed for a reason that is not a "
                "fail-closed engine drift; nothing was written."
            )
        text, round_notes = _apply(text, *observation)
        notes.extend(round_notes)
    else:
        MODULE_PATH.write_text(original)
        raise SystemExit(
            f"Pool engine contracts did not converge in {MAX_ROUNDS} rounds; "
            "nothing was written."
        )

    rows = derived["manifest_rows"]
    row_pattern = (
        r'"""Pinned content digest of all [\d,]+ post-transfer consumer/input '
        r'rows\."""'
    )
    replacement = (
        f'"""Pinned content digest of all {rows:,} post-transfer '
        'consumer/input rows."""'
    )
    match = re.search(row_pattern, text)
    if match is None:
        MODULE_PATH.write_text(original)
        raise SystemExit("Cannot locate the remaining-stage manifest row count.")
    if match.group(0) != replacement:
        notes.append(f"remaining-stage manifest rows -> {rows:,}")
        text = text[: match.start()] + replacement + text[match.end() :]

    if text == original:
        MODULE_PATH.write_text(original)
        print("US pool engine contracts are current")
        return 0
    for note in notes:
        print(f"re-derived: {note}")
    if args.check:
        MODULE_PATH.write_text(original)
        raise SystemExit(
            f"US pool engine contracts are stale for policyengine-us "
            f"{derived['engine_version']}: regenerate with "
            "`uv run python tools/repin_us_pool_engine_contracts.py`."
        )
    MODULE_PATH.write_text(text)
    print(f"updated {MODULE_PATH.relative_to(REPOSITORY_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
