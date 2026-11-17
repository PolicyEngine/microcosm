"""Grade a staging run's nulls against R2's reviewed-fill rules (pre-check).

The ACS local release (R2, ``tools/build_us_acs_local_release.py``) fills
each reviewed engine-input null of the staging H5 with the policyengine-us
default (``fill_reviewed_nulls``). Its materialize stage refuses a null in a
``NEVER_DEFAULT_FILLED`` column and a null ESI employer premium. Its finalize
stage, hours later, grades every applied fill against
``acs_local_reviewed_fill_consumers.yaml`` with
``acs_local_reviewed_fill_consumer_gate``: it refuses a fill without a
register entry, a fill value or spines that differ from the entry, and a
policyengine-us other than the register's ``reviewed_against``.

This check runs in seconds after the staging (R1), before R2:

- It reads the staging summary's ``reviewed_engine_input_nulls`` (column
  names and null counts by spine; no row values).
- A null in a ``NEVER_DEFAULT_FILLED`` column on any spine but the ACS one
  is refused, as materialize refuses it; R2's own stages fill the ACS cells.
- A null in an ESI premium output is refused, as materialize refuses it.
- Every other null becomes the fill record R2 writes, with the fill value R2
  resolves from the installed engine, and R2's own gate grades the records.
  A null that one of R2's stages fills before the fill step counts as a fill
  here, so the check errs towards a refusal.

Exit status 0: R2's fill rules pass. Exit status 2: R2 would refuse; the
JSON report (``--out``) says why. Add the missing register entries (each a
reviewed ruling) and commit them before R2's calibrate stage.

Usage::

    python tools/check_us_acs_local_reviewed_fills.py \\
        --staging-summary <run>/acs_multispine_staging.summary.json \\
        --out <run>/reviewed_fills_precheck.json
"""

# No ``from __future__ import annotations``: test loaders run this file
# without registering it in ``sys.modules``.
import argparse
import importlib.util
import json
import sys
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from microcosm.build.us_runtime.acs_local_reviewed_fill_consumers import (
    acs_local_reviewed_fill_consumer_gate,
    load_reviewed_fill_consumer_register,
)
from microcosm.build.us_runtime.esi_premiums import US_ESI_PREMIUMS_OUTPUT_COLUMNS

TOOL = "tools/check_us_acs_local_reviewed_fills.py"
ISSUE = "microcosm#922"
ACS_SPINE = "acs_2024_1yr"
EXIT_REFUSED = 2
_RELEASE_TOOL = Path(__file__).resolve().parent / "build_us_acs_local_release.py"


def _release_tool():
    """R2's module, for its own fill constants and default resolver."""

    tools = str(_RELEASE_TOOL.parent)
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "build_us_acs_local_release_for_fill_check", _RELEASE_TOOL
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expected_fills(
    nulls: Sequence[Mapping[str, Any]],
    *,
    never_default_filled: Collection[tuple[str, str]],
    resolve_default: Callable[[str], tuple[object, str]],
) -> dict[str, list]:
    """Split the staged nulls as R2's fill step treats them.

    Returns ``fills`` (the records R2 writes, in its manifest's shape),
    ``denied`` (never-default nulls off the ACS spine), ``esi`` (null ESI
    premium outputs) and ``natively_filled`` (never-default nulls on the ACS
    spine only, which R2's stages fill).
    """

    never = set(never_default_filled)
    result: dict[str, list] = {
        "fills": [],
        "denied": [],
        "esi": [],
        "natively_filled": [],
    }
    for item in nulls:
        entity, column = str(item["entity"]), str(item["column"])
        by_spine = {
            str(spine): int(rows)
            for spine, rows in dict(item.get("missing_rows_by_spine") or {}).items()
            if int(rows)
        }
        name = f"{entity}.{column}"
        if entity == "person" and column in US_ESI_PREMIUMS_OUTPUT_COLUMNS:
            result["esi"].append({"column": name, "missing_rows_by_spine": by_spine})
            continue
        if (entity, column) in never:
            off_acs = {
                spine: rows for spine, rows in by_spine.items() if spine != ACS_SPINE
            }
            target = "denied" if off_acs or not by_spine else "natively_filled"
            result[target].append({"column": name, "missing_rows_by_spine": by_spine})
            continue
        fill_value, kind = resolve_default(column)
        missing = int(item["missing_rows"])
        result["fills"].append(
            {
                "entity": entity,
                "column": column,
                "filled_rows": missing,
                "rows": int(item.get("rows", missing)),
                "fill_value": repr(fill_value),
                "fill_kind": kind,
                "register_missing_rows": missing,
                "missing_rows_by_spine": by_spine,
            }
        )
    return result


def grade(
    split: Mapping[str, list],
    *,
    manifest_names: Sequence[str],
    never_default_filled: Collection[tuple[str, str]],
    installed_engine_version: str,
    document: object | None = None,
) -> dict[str, Any]:
    """Run R2's finalize gate on the expected fills; add materialize's refusals."""

    manifest = {"fills": list(split["fills"]), "unregistered_violations": []}
    gate = acs_local_reviewed_fill_consumer_gate(
        {name: manifest for name in manifest_names},
        document=document
        if document is not None
        else load_reviewed_fill_consumer_register(),
        never_default_filled=never_default_filled,
        installed_engine_version=installed_engine_version,
    )
    refusals = [
        *(
            f"{item['column']}: null on {sorted(item['missing_rows_by_spine'])}, "
            "a column R2 never default-fills (materialize refuses it)"
            for item in split["denied"]
        ),
        *(
            f"{item['column']}: null ESI employer premium (materialize refuses it)"
            for item in split["esi"]
        ),
        *(f"finalize gate: {failure}" for failure in gate.failures),
    ]
    return {
        "passed": not refusals,
        "refusals": refusals,
        "expected_fills": len(split["fills"]),
        "natively_filled": split["natively_filled"],
        "gate": {
            "name": gate.name,
            "passed": gate.passed,
            "failures": list(gate.failures),
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Grade a staging summary's reviewed nulls against R2's fill "
            "rules and its finalize reviewed-fill gate, before R2 runs."
        )
    )
    parser.add_argument("--staging-summary", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="JSON report path.")
    return parser.parse_args(argv)


def _tax_benefit_system():
    from policyengine_us import CountryTaxBenefitSystem

    return CountryTaxBenefitSystem()


def _engine_version(release) -> str:
    return release._installed_engine_version()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = json.loads(args.staging_summary.read_text())
    nulls = summary.get("reviewed_engine_input_nulls")
    if not isinstance(nulls, list):
        raise SystemExit(
            f"{args.staging_summary} carries no reviewed_engine_input_nulls list."
        )
    release = _release_tool()
    system = _tax_benefit_system()
    split = expected_fills(
        nulls,
        never_default_filled=release.NEVER_DEFAULT_FILLED,
        resolve_default=lambda column: release._resolve_engine_default(system, column),
    )
    report = {
        "tool": TOOL,
        "issue": ISSUE,
        "staging_summary": str(args.staging_summary),
        "staged_null_columns": len(nulls),
        **grade(
            split,
            manifest_names=release.REVIEWED_FILL_MANIFESTS,
            never_default_filled=release.NEVER_DEFAULT_FILLED,
            installed_engine_version=_engine_version(release),
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in ("passed", "refusals", "expected_fills")},
            indent=2,
        )
    )
    return 0 if report["passed"] else EXIT_REFUSED


if __name__ == "__main__":
    raise SystemExit(main())
