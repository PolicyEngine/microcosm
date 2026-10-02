"""Differential check: microcosm's EMPSTATI mapping against policyengine-uk-data#526.

policyengine-uk-data is not a microcosm dependency and #526 is unreleased, so
this runs once, by hand, rather than in CI. It loads uk-data's
``FRS_EMPSTATI_EMPLOYMENT_STATUS`` and ``derive_employment_status_from_frs``
from a checkout of the PR head (by AST, so the rest of ``frs.py`` and its
imports are never executed), then checks the two implementations agree:

1. on every code 0-11 and a set of unknown codes, as adult and child rows;
2. on Hypothesis-generated mixes of adult and child rows, raising together;
3. on the licensed FRS 2024-25 person set (``adult.tab`` plus ``child.tab``,
   read through microcosm's sha-pinned reader), element by element.

Only aggregates are printed: per-status counts, with cells under 10 people
suppressed, and the mismatch count. Usage (UK engine environment)::

    uv run --no-sync python \
        experiments/uk-frs-empstati-other-inactive/differential_vs_uk_data_526.py \
        --uk-data-checkout <policyengine-uk-data at #526's head> \
        --frs-dir <licensed FRS 2024-25 tab directory>
"""

from __future__ import annotations

import argparse
import ast
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from policyengine_uk.variables.household.income.employment_status import (
    EmploymentStatus,
)

from microcosm.build.country_spec import load_country_spec
from microcosm.build.uk_runtime.frs_employment import (
    FRS_EMPSTATI_EMPLOYMENT_STATUS,
    derive_employment_status_from_frs,
    derive_frs_employment,
)
from microcosm.build.uk_runtime.frs_spine import normalize_ids, read_pinned_tab

UK_DATA_526_HEAD = "fb0266593cab1dbe4464c4c6db63e4868af43db5"
UK_DATA_NAMES = ("FRS_EMPSTATI_EMPLOYMENT_STATUS", "derive_employment_status_from_frs")


def load_uk_data_mapping(checkout: Path):
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != UK_DATA_526_HEAD:
        raise SystemExit(f"{checkout} is at {head}, not #526's {UK_DATA_526_HEAD}.")
    source = (checkout / "policyengine_uk_data/datasets/frs.py").read_text()
    tree = ast.parse(source)
    wanted = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in UK_DATA_NAMES for t in node.targets)
        )
        or (isinstance(node, ast.FunctionDef) and node.name in UK_DATA_NAMES)
    ]
    if len(wanted) != len(UK_DATA_NAMES):
        raise SystemExit(f"Expected {UK_DATA_NAMES} in uk-data frs.py.")
    namespace = {"EmploymentStatus": EmploymentStatus, "np": np, "pd": pd}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), "frs.py", "exec"), namespace)
    return namespace[UK_DATA_NAMES[0]], namespace[UK_DATA_NAMES[1]]


REFUSAL_TYPES: dict[str, set[str]] = {"microcosm": set(), "uk-data": set()}


def outcome(function, codes, is_adult, *, side):
    """Statuses, or "refused" for any exception (its type is recorded).

    Either side refusing stops the build. The refusal's exception type can
    differ by environment: uk-data's message formatting depends on the pandas
    version, so it is recorded and reported rather than compared.
    """
    try:
        return ("ok", list(function(codes, is_adult)))
    except Exception as error:  # noqa: BLE001 - every exception refuses a build
        REFUSAL_TYPES[side].add(type(error).__name__)
        return ("refused", None)


def suppressed(counts: pd.Series) -> dict[str, object]:
    return {
        str(key): (int(value) if value >= 10 else "<10")
        for key, value in counts.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uk-data-checkout", type=Path, required=True)
    parser.add_argument("--frs-dir", type=Path, required=True)
    args = parser.parse_args()
    uk_data_table, uk_data_derive = load_uk_data_mapping(args.uk_data_checkout)

    assert dict(FRS_EMPSTATI_EMPLOYMENT_STATUS) == uk_data_table
    print("code tables identical: 11 codes")

    fixed = [*range(0, 13), -1, 11.5, math.nan, 99]
    for code in fixed:
        for is_adult in (True, False):
            ours = outcome(
                derive_employment_status_from_frs, [code], [is_adult], side="microcosm"
            )
            theirs = outcome(uk_data_derive, [code], [is_adult], side="uk-data")
            assert ours == theirs, (code, is_adult, ours, theirs)
    print(f"fixed cases agree: {len(fixed) * 2}")

    code_values = st.one_of(
        st.integers(-5, 15), st.just(math.nan), st.floats(0.5, 11.5)
    )
    rows = st.lists(st.tuples(st.booleans(), code_values), max_size=40)

    @settings(max_examples=2_000, deadline=None)
    @given(rows)
    def agree(sample):
        codes = [code for _, code in sample]
        is_adult = [adult for adult, _ in sample]
        assert outcome(
            derive_employment_status_from_frs, codes, is_adult, side="microcosm"
        ) == outcome(uk_data_derive, codes, is_adult, side="uk-data")

    agree()
    print("hypothesis cases agree: 2000 examples")
    print(
        f"refusal exception types (pandas {pd.__version__}):",
        {side: sorted(types) for side, types in REFUSAL_TYPES.items()},
    )
    assert REFUSAL_TYPES["microcosm"] <= {"ValueError"}

    stages = {stage.stage: stage for stage in load_country_spec("uk").sources.stages}
    employment = {a["table"]: a for a in stages["frs_employment"].artifacts}
    spine = {a["table"]: a for a in stages["frs_spine"].artifacts}
    adult = normalize_ids(
        read_pinned_tab(args.frs_dir / "adult.tab", employment["adult"])
    )
    child = normalize_ids(
        read_pinned_tab(
            args.frs_dir / "child.tab", spine["child"], columns=("sernum", "person")
        )
    )
    person = pd.DataFrame(
        {"person_id": np.concatenate([adult["person_id"], child["person_id"]])}
    )
    ours = derive_frs_employment(person, adult)["employment_status"].to_numpy()
    # uk-data's person table fills the child table's absent EMPSTATI with 0.
    uk_data_codes = person["person_id"].map(adult.set_index("person_id")["empstati"])
    theirs = uk_data_derive(
        uk_data_codes.fillna(0).to_numpy(),
        person["person_id"].isin(adult["person_id"]).to_numpy(),
    )
    mismatches = int((ours != theirs).sum())
    print(f"licensed FRS 2024-25 people: {len(person)}; mismatches: {mismatches}")
    print(
        "employment_status counts (unweighted, cells under 10 suppressed):",
        suppressed(pd.Series(ours).value_counts().sort_index()),
    )
    assert mismatches == 0


if __name__ == "__main__":
    main()
