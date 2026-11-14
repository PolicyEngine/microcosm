"""Differential test: the builder's tax units against the archived construction.

    uv run python experiments/us-asec-processed-source-builder/scripts/differential_tax_units.py \\
        --archived-python <archived checkout>/.venv/bin/python \\
        --archived-root <archived checkout at 9f177ddd> \\
        --archived-policyengine-us <unpacked policyengine-us 1.729.0 wheel> \\
        --households 20000 --seed 20260928 \\
        --receipt experiments/us-asec-processed-source-builder/receipts/tax_unit_differential.json \\
        [--fixture packages/microcosm-build/tests/fixtures/us_asec_tax_unit_golden_cases.csv]

Generates seeded synthetic households (``household_cases.py``), then assigns
tax units twice for each income year 2022-2025:

* the archived loader's ``tax_unit_construction.construct_tax_units`` (commit
  ``40314a75``; ``9f177ddd`` leaves it unchanged), in an interpreter with the
  loader's pandas 2.3.3. Its rule helpers come from
  ``policyengine_us.tools.tax_unit_construction`` in the PolicyEngine-US
  1.729.0 wheel, as in the environment that built the 2025 file. Both
  packages' modules are loaded without their ``__init__`` files;
* :func:`microcosm.build.us_runtime.asec_processed_source.construct_asec_tax_units`,
  the builder's path (microunit), in Microcosm's environment (pandas 3).

It records per-year agreement of ``TAX_ID`` and role, row by row, and runs
two precondition probes through both. With ``--fixture`` it writes golden
cases for CI (``test_us_asec_processed_source.py``):
- every household whose units change between years;
- the lowest-numbered households with each tolerated spouse-pointer anomaly;
- then the lowest-numbered households, up to ``--fixture-households``.

Each case carries the archived construction's ``TAX_ID`` and role, computed
on that subset.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from importlib import metadata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from household_cases import generate_cases, probe_cases  # noqa: E402

from microcosm.build.us_runtime.asec_processed_source import (  # noqa: E402
    audit_tax_unit_inputs,
    construct_asec_tax_units,
    dependent_gross_income_limit_for_year,
)

YEARS = (2022, 2023, 2024, 2025)
_ARCHIVED_POLICYENGINE_US = Path()

_ARCHIVED_RUNNER = r"""
import hashlib, importlib, json, os, sys, types
from pathlib import Path
import pandas as pd
# Load the two construction modules from the checkout, and the rule helpers
# they import from the PolicyEngine-US wheel the loader ran with, without
# running either package's __init__ files (which import the rest of each
# package).
package = Path.cwd() / "policyengine_us_data"
wheel = Path(os.environ["ARCHIVED_POLICYENGINE_US"])
for name, path in (
    ("policyengine_us_data", package),
    ("policyengine_us_data.datasets", package / "datasets"),
    ("policyengine_us_data.datasets.cps", package / "datasets" / "cps"),
    ("policyengine_us", wheel / "policyengine_us"),
    ("policyengine_us.tools", wheel / "policyengine_us" / "tools"),
):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
construct_tax_units = importlib.import_module(
    "policyengine_us_data.datasets.cps.tax_unit_construction"
).construct_tax_units
helpers = importlib.import_module("policyengine_us_data.datasets.cps.tax_unit_rule_helpers")
cases = pd.read_csv(sys.argv[1])
out = pd.DataFrame(index=cases.index)
if sys.argv[2] == "--probe":
    try:
        construct_tax_units(cases.copy(), year=2024, mode="policyengine")
        print(json.dumps({"raised": None}))
    except Exception as error:
        print(json.dumps({"raised": type(error).__name__}))
    sys.exit(0)
for year in [int(value) for value in sys.argv[3].split(",")]:
    assignments, _ = construct_tax_units(cases.copy(), year=year, mode="policyengine")
    out[f"TAX_ID_{year}"] = assignments["TAX_ID"].to_numpy()
    out[f"ROLE_{year}"] = [value.decode() for value in assignments["tax_unit_role_input"]]
out.to_csv(sys.argv[2], index=False)
print(json.dumps({
    "pandas": pd.__version__,
    "policyengine_us": next(
        line.split(":", 1)[1].strip()
        for path in wheel.glob("policyengine_us-*.dist-info/METADATA")
        for line in path.read_text().splitlines()
        if line.startswith("Version:")
    ),
    "policyengine_us_files_sha256": {
        name: hashlib.sha256((wheel / "policyengine_us" / name).read_bytes()).hexdigest()
        for name in (
            "tools/tax_unit_construction.py",
            "parameters/gov/irs/income/exemption/amount.yaml",
        )
    },
    "helpers_module": helpers.dependent_gross_income_limit.__module__,
    "limits": {year: helpers.dependent_gross_income_limit(int(year)) for year in sys.argv[3].split(",")},
}))
"""


def _run_archived(python: Path, root: Path, *args: str) -> dict:
    completed = subprocess.run(
        [str(python), "-c", _ARCHIVED_RUNNER, *args],
        cwd=root,
        # The archived checkout's own interpreter and packages only: drop
        # the calling environment's virtualenv markers.
        env={
            **{
                key: value
                for key, value in os.environ.items()
                if key not in {"VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"}
                and not key.startswith("UV_")
            },
            "PYTHONPATH": str(root),
            "ARCHIVED_POLICYENGINE_US": str(_ARCHIVED_POLICYENGINE_US),
        },
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise SystemExit(f"archived construction failed:\n{completed.stderr}")
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _archived_assignments(
    cases: pd.DataFrame, python: Path, root: Path
) -> tuple[pd.DataFrame, dict]:
    with tempfile.TemporaryDirectory() as scratch:
        cases_path = Path(scratch) / "cases.csv"
        out_path = Path(scratch) / "archived.csv"
        cases.to_csv(cases_path, index=False)
        info = _run_archived(
            python,
            root,
            str(cases_path),
            str(out_path),
            ",".join(str(year) for year in YEARS),
        )
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        info["checkout_commit"] = commit
        return pd.read_csv(out_path), info


def _builder_assignments(cases: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=cases.index)
    for year in YEARS:
        tax_ids, _, assignments = construct_asec_tax_units(cases.copy(), year)
        out[f"TAX_ID_{year}"] = tax_ids
        out[f"ROLE_{year}"] = [
            value.decode() for value in assignments["tax_unit_role_input"]
        ]
    return out


def _changed_partition_households(
    cases: pd.DataFrame, first: pd.Series, second: pd.Series
) -> set[int]:
    changed = set()
    for household, rows in cases.groupby("PH_SEQ", sort=False).groups.items():
        first_units = {
            frozenset(rows[first.loc[rows] == unit]) for unit in first.loc[rows]
        }
        second_units = {
            frozenset(rows[second.loc[rows] == unit]) for unit in second.loc[rows]
        }
        if first_units != second_units:
            changed.add(int(household))
    return changed


def _anomalous_spouse_households(cases: pd.DataFrame) -> dict[str, set[int]]:
    """Households with a spouse pointer the construction tolerates but that
    Census files never carry, by kind: to an absent line, to oneself, to a
    member who is not married spouse-present, or one-sided."""

    rows = {(row.PH_SEQ, row.A_LINENO): row for row in cases.itertuples(index=False)}
    kinds: dict[str, set[int]] = {
        "spouse_line_absent": set(),
        "self_spouse_pointer": set(),
        "spouse_not_married_spouse_present": set(),
        "one_sided_spouse_pointer": set(),
    }
    for row in rows.values():
        if row.A_MARITL not in (1, 2) or row.A_SPOUSE <= 0:
            continue
        spouse = rows.get((row.PH_SEQ, row.A_SPOUSE))
        if spouse is None:
            kind = "spouse_line_absent"
        elif spouse is row:
            kind = "self_spouse_pointer"
        elif spouse.A_MARITL not in (1, 2) or spouse.A_SPOUSE <= 0:
            kind = "spouse_not_married_spouse_present"
        elif spouse.A_SPOUSE != row.A_LINENO:
            kind = "one_sided_spouse_pointer"
        else:
            continue
        kinds[kind].add(int(row.PH_SEQ))
    return kinds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived-python", type=Path, required=True)
    parser.add_argument("--archived-root", type=Path, required=True)
    parser.add_argument(
        "--archived-policyengine-us",
        type=Path,
        required=True,
        help="Unpacked PolicyEngine-US wheel the loader ran with (1.729.0).",
    )
    parser.add_argument("--households", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument("--fixture-households", type=int, default=400)
    parser.add_argument("--fixture-per-anomaly", type=int, default=20)
    args = parser.parse_args(argv)
    global _ARCHIVED_POLICYENGINE_US
    _ARCHIVED_POLICYENGINE_US = args.archived_policyengine_us.resolve()

    cases = generate_cases(args.households, args.seed)
    archived, archived_info = _archived_assignments(
        cases, args.archived_python, args.archived_root
    )
    builder = _builder_assignments(cases)

    per_year = {}
    for year in YEARS:
        tax_equal = (
            builder[f"TAX_ID_{year}"].to_numpy()
            == archived[f"TAX_ID_{year}"].to_numpy()
        )
        role_equal = (
            builder[f"ROLE_{year}"].to_numpy() == archived[f"ROLE_{year}"].to_numpy()
        )
        roles = archived[f"ROLE_{year}"].value_counts().to_dict()
        units = archived[f"TAX_ID_{year}"].nunique()
        per_year[str(year)] = {
            "dependent_gross_income_limit": dependent_gross_income_limit_for_year(year),
            "tax_units": int(units),
            "roles": {key: int(value) for key, value in sorted(roles.items())},
            "tax_id_rows_equal": int(tax_equal.sum()),
            "role_rows_equal": int(role_equal.sum()),
            "equal": bool(tax_equal.all() and role_equal.all()),
            "first_disagreeing_households": sorted(
                {int(value) for value in cases.loc[~(tax_equal & role_equal), "PH_SEQ"]}
            )[:10],
        }
    probes = {}
    for name, probe in probe_cases().items():
        with tempfile.TemporaryDirectory() as scratch:
            probe_path = Path(scratch) / "probe.csv"
            probe.to_csv(probe_path, index=False)
            archived_probe = _run_archived(
                args.archived_python, args.archived_root, str(probe_path), "--probe", ""
            )
        try:
            construct_asec_tax_units(probe.copy(), 2024)
            builder_probe = {"raised": None, "message": None}
        except ValueError as error:
            builder_probe = {"raised": type(error).__name__, "message": str(error)}
        probes[name] = {
            "input_audit": audit_tax_unit_inputs(probe),
            "archived": archived_probe,
            "builder": builder_probe,
        }
    changed_households = {
        f"{earlier}-{later}": _changed_partition_households(
            cases, archived[f"TAX_ID_{earlier}"], archived[f"TAX_ID_{later}"]
        )
        for earlier, later in zip(YEARS, YEARS[1:], strict=False)
    }
    changed_by_limit = {key: len(value) for key, value in changed_households.items()}
    receipt = {
        "purpose": (
            "Row-by-row agreement of TAX_ID and role between the archived "
            "construction (commit 40314a75) and the builder's microunit path "
            "on seeded synthetic households."
        ),
        "generator": "experiments/us-asec-processed-source-builder/scripts/household_cases.py",
        "households": args.households,
        "persons": int(len(cases)),
        "seed": args.seed,
        "archived": archived_info,
        "builder": {
            "pandas": pd.__version__,
            "microunit": metadata.version("microunit"),
        },
        "case_input_audit": audit_tax_unit_inputs(cases),
        "years": per_year,
        "precondition_probes": {
            "cases": "household_cases.probe_cases",
            **probes,
        },
        # Only the dependent income limit differs between years, so these
        # households exercise the qualifying-relative income test.
        "households_whose_partition_changes_between_years": changed_by_limit,
        "equal": all(entry["equal"] for entry in per_year.values()),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=1))

    if args.fixture is not None:
        # Every household whose units change with the income limit, the
        # lowest-numbered households with each kind of tolerated spouse-pointer
        # anomaly (up to --fixture-per-anomaly each), then the lowest-numbered
        # households up to the fixture size. The archived construction runs
        # again on the subset, whose dense ids start from 1.
        chosen = set().union(*changed_households.values())
        for households in _anomalous_spouse_households(cases).values():
            chosen |= set(sorted(households)[: args.fixture_per_anomaly])
        for household in sorted(cases["PH_SEQ"].unique()):
            if len(chosen) >= args.fixture_households:
                break
            chosen.add(int(household))
        subset = cases[cases["PH_SEQ"].isin(chosen)].reset_index(drop=True)
        expected, _ = _archived_assignments(
            subset, args.archived_python, args.archived_root
        )
        fixture = pd.concat([subset, expected], axis=1)
        fixture = fixture.rename(
            columns={
                **{f"TAX_ID_{year}": f"EXPECTED_TAX_ID_{year}" for year in YEARS},
                **{f"ROLE_{year}": f"EXPECTED_ROLE_{year}" for year in YEARS},
            }
        )
        args.fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.to_csv(args.fixture, index=False)
    return 0 if receipt["equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
