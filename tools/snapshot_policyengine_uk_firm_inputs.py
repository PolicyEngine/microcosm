"""Snapshot policyengine-uk's firm variables for engine-free tests.

Reads a policyengine-uk checkout's source with ``ast`` (it never imports
``policyengine_uk``), through ``microcosm.build.firms.read_entity_variables``,
and writes, as sorted JSON:

- every ``Variable`` subclass whose ``entity`` is ``Firm``, anywhere in the
  package except its ``tests/`` directory, with its value type, unit, default
  value, enum members (the ``possible_values`` class resolved through the
  defining module's imports, so ``Region`` comes from
  ``variables/household/demographic/geography.py``), definition period,
  whether it has a formula (a ``formula`` or ``formula_*`` method or
  attribute, ``adds`` or ``subtracts``) and its file relative to the package;
- the formula-less firm variables, which a dataset sets as inputs;
- every policyengine-uk variable name, and which of Microcosm's own firm
  columns (``MICROCOSM_ONLY_COLUMNS``) collide with one;
- the checkout's HEAD commit, whether its working tree is dirty, the pull
  request the firm entity comes from and the snapshot date.

A variable created at runtime (by a reform or a loop) is not in the source
and so not in the snapshot.

``--check`` regenerates the snapshot and fails if anything but the snapshot
date differs from ``--out``, so a moved policyengine-uk HEAD or a changed firm
variable shows up before the fixture is refreshed.

The tool refuses a checkout with a dirty working tree, since its HEAD commit
would not identify the source it read; ``--allow-dirty`` overrides that for a
local look (the snapshot then records ``dirty`` true).

Usage::

    uv run --no-sync python tools/snapshot_policyengine_uk_firm_inputs.py \\
        --policyengine-uk-root ../policyengine-uk \\
        --out packages/microcosm-build/tests/fixtures/uk/policyengine_uk_firm_inputs.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from microcosm.build.firms import read_entity_variables

SCHEMA_VERSION = 1
PACKAGE_NAME = "policyengine_uk"
FIRM_ENTITY = "Firm"
#: The policyengine-uk pull request that adds the firm entity.
POLICYENGINE_UK_PULL_REQUEST = "PolicyEngine/policyengine-uk#2940"

#: Columns Microcosm's firm table carries that are not policyengine-uk inputs.
#: None may share a name with a policyengine-uk variable, or a dataset load
#: would read it as one.
MICROCOSM_ONLY_COLUMNS = (
    "in_frame",
    "vat_scope",
    "vat_registered",
    "base_weight",
    "data_vintage",
    "source_firm_id",
    "source_firm_key",
    "ons_legal_status",
    "ons_region_code",
)

#: Fields ``--check`` ignores: the date the snapshot was taken.
_VOLATILE_FIELDS = ("snapshot_date",)


def _git(root: Path, *args: str) -> str:
    environment = dict(os.environ)
    local = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    for name in local:
        environment.pop(name, None)
    return subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=True,
        env=environment,
    ).stdout


def build_snapshot(
    root: Path, *, snapshot_date: str, allow_dirty: bool = False
) -> dict[str, Any]:
    """The firm-input snapshot of the policyengine-uk checkout at ``root``.

    Raises ``ValueError`` when the checkout's working tree is dirty, unless
    ``allow_dirty``.
    """

    package_dir = root / PACKAGE_NAME
    if not package_dir.is_dir():
        raise ValueError(f"{root} holds no {PACKAGE_NAME} package.")
    dirty = bool(_git(root, "status", "--porcelain").strip())
    if dirty and not allow_dirty:
        raise ValueError(
            f"{root} has a dirty working tree; commit or stash the changes, "
            "or pass --allow-dirty."
        )
    read = read_entity_variables(package_dir, entity=FIRM_ENTITY)
    variable_names = set(read.variable_names)
    firm_variables = read.variables
    inputs = read.input_names
    return {
        "collisions": sorted(set(MICROCOSM_ONLY_COLUMNS) & variable_names),
        "firm_input_names": inputs,
        "firm_variables": firm_variables,
        "microcosm_only_columns_checked": sorted(MICROCOSM_ONLY_COLUMNS),
        "policyengine_uk": {
            "dirty": dirty,
            "head_commit": _git(root, "rev-parse", "HEAD").strip(),
            "pull_request": POLICYENGINE_UK_PULL_REQUEST,
        },
        "policyengine_uk_variable_names": sorted(variable_names),
        "schema_version": SCHEMA_VERSION,
        "snapshot_date": snapshot_date,
    }


def render(snapshot: dict[str, Any]) -> str:
    return json.dumps(snapshot, indent=2, sort_keys=True) + "\n"


def _stable(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in snapshot.items() if key not in _VOLATILE_FIELDS
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policyengine-uk-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if --out differs from a fresh snapshot (ignoring its date).",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Read a policyengine-uk checkout with a dirty working tree.",
    )
    args = parser.parse_args(argv)
    try:
        snapshot = build_snapshot(
            args.policyengine_uk_root.resolve(),
            snapshot_date=datetime.now(UTC).date().isoformat(),
            allow_dirty=args.allow_dirty,
        )
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    if args.check:
        committed = json.loads(args.out.read_text(encoding="utf-8"))
        if _stable(committed) != _stable(snapshot):
            changed = sorted(
                key
                for key in set(_stable(committed)) | set(_stable(snapshot))
                if committed.get(key) != snapshot.get(key)
            )
            print(f"{args.out} is stale; changed fields: {changed}", file=sys.stderr)
            return 1
        print(f"{args.out} matches the policyengine-uk checkout.")
        return 0
    args.out.write_text(render(snapshot), encoding="utf-8")
    inputs = snapshot["firm_input_names"]
    print(
        f"Wrote {args.out}: {len(snapshot['firm_variables'])} firm variables, "
        f"{len(inputs)} inputs, HEAD {snapshot['policyengine_uk']['head_commit']}"
        f"{' (dirty)' if snapshot['policyengine_uk']['dirty'] else ''}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
