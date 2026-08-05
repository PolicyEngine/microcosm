"""Preflight UK H5 artifacts against populace Frame linkage invariants (#612).

The #612 carrier swap makes ``populace.frame.Frame`` validate the UK national
tables at every stage boundary. Frame enforces invariants the UK path never
checked — group ids sorted ascending (Frame raises, it does not reorder), and
membership equality in *both* directions (a benunit or household with no
member persons fails, not just a dangling reference) — so this tool answers,
before any migration code runs, whether the real artifacts already satisfy
them. Run it against the raw single-year input, the latest staging output,
and the certified candidate.

Disclosure control: the output is written to be publishable on the tracking
issue under the UK Data Service End User Licence (CD137 v16.00 clause 8 /
CD171 §5.2.1). It reports classifications, booleans, aggregates, and
threshold-guarded counts — never unit-record values. Frame's own exception
messages embed real ids, so construction failures are reported as a boolean
plus the pre-computed violation classes, and exception text is never echoed.

Read-only by construction: tables are read with ``pandas.HDFStore`` in mode
``"r"`` and root attributes with ``h5py`` in mode ``"r"``. Never open these
artifacts through ``UKSingleYearDataset(file_path=...)`` — that path opens
mode ``"a"`` (a write open) and would invalidate the certified-candidate
file fingerprint on licensed data.

Example::

    uv run python tools/preflight_uk_frame_linkage.py \
        /path/to/populace_uk_2023.h5 \
        /path/to/staging_uk_2023.h5 \
        --json-out preflight_uk_frame_linkage.json

Exit code: 1 if any artifact fails Frame construction or the split-benunit
nesting check, 2 for an unsafe CLI configuration, and 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "packages" / "populace-frame" / "src")
)
sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "packages" / "populace-build" / "src")
)

from populace.build.uk_runtime.rowwise_dataset import (  # noqa: E402
    read_uk_single_year_weight_metadata,
)
from populace.frame import EntitySchema, Frame, Weights  # noqa: E402

GROUP_ENTITIES = ("benunit", "household")
UK_SCHEMA = EntitySchema(group_entities=GROUP_ENTITIES)
_TABLES = ("person", *GROUP_ENTITIES)
DEFAULT_SDC_MINIMUM_COUNT = 10
# CD171-ResearchDataHandling §5.2.1: cells based on one or two cases are
# never reportable. Keep this as the single authority for every entry point;
# callers may raise the threshold (including to a study-specific 30), but
# cannot lower it.
MINIMUM_SDC_COUNT = 3


def _validated_sdc_minimum(minimum: int) -> int:
    minimum = int(minimum)
    if minimum < MINIMUM_SDC_COUNT:
        raise ValueError(f"SDC minimum count must be at least {MINIMUM_SDC_COUNT}.")
    return minimum


def sdc_count(count: int, *, minimum: int) -> int | str:
    """Mask small nonzero counts: zero and counts >= minimum are safe."""

    minimum = _validated_sdc_minimum(minimum)
    count = int(count)
    if count == 0 or count >= minimum:
        return count
    return f"< {minimum}"


def _column_missing(table: pd.DataFrame, column: str) -> bool:
    return column not in table.columns


def classify_group_linkage(
    person: pd.DataFrame,
    group_table: pd.DataFrame,
    group: str,
    *,
    minimum: int,
) -> dict[str, Any]:
    """Classify one group entity against Frame's linkage invariants."""

    id_column = f"{group}_id"
    membership_column = f"person_{group}_id"
    missing = [
        column
        for table, column in ((group_table, id_column), (person, membership_column))
        if _column_missing(table, column)
    ]
    if missing:
        return {"missing_columns": missing}

    ids = group_table[id_column]
    membership_values = person[membership_column]
    memberships = membership_values.dropna()
    id_set = set(ids.dropna())
    membership_set = set(memberships)
    non_na = ids.dropna()
    return {
        "missing_columns": [],
        "membership_na": sdc_count(membership_values.isna().sum(), minimum=minimum),
        "id_na": sdc_count(ids.isna().sum(), minimum=minimum),
        "id_duplicated": sdc_count(non_na.duplicated().sum(), minimum=minimum),
        "ids_sorted_ascending": bool(non_na.is_monotonic_increasing),
        "orphaned_group_rows": sdc_count(len(id_set - membership_set), minimum=minimum),
        "dangling_memberships": sdc_count(
            len(membership_set - id_set), minimum=minimum
        ),
    }


def classify_linkage(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    *,
    minimum: int,
) -> dict[str, Any]:
    """Classify all Frame linkage invariants over the three UK tables."""

    report: dict[str, Any] = {
        group: classify_group_linkage(
            person,
            {"benunit": benunit, "household": household}[group],
            group,
            minimum=minimum,
        )
        for group in GROUP_ENTITIES
    }
    if _column_missing(person, "person_id"):
        report["person"] = {"missing_columns": ["person_id"]}
    else:
        person_ids = person["person_id"]
        report["person"] = {
            "missing_columns": [],
            "id_na": sdc_count(person_ids.isna().sum(), minimum=minimum),
            "id_duplicated": sdc_count(
                person_ids.dropna().duplicated().sum(), minimum=minimum
            ),
        }
    if not _column_missing(person, "person_benunit_id") and not _column_missing(
        person, "person_household_id"
    ):
        households_per_benunit = person.groupby("person_benunit_id")[
            "person_household_id"
        ].nunique()
        report["split_benunits"] = sdc_count(
            int((households_per_benunit > 1).sum()), minimum=minimum
        )
    return report


def classify_household_weights(
    household: pd.DataFrame, *, minimum: int
) -> dict[str, Any]:
    """Classify the weight vector against Frame's weight invariants."""

    minimum = _validated_sdc_minimum(minimum)
    if _column_missing(household, "household_weight"):
        return {"missing_columns": ["household_weight"]}
    weights = household["household_weight"]
    numeric = pd.to_numeric(weights, errors="coerce")
    conversion_failures = int((weights.notna() & numeric.isna()).sum())
    values = numeric.to_numpy(dtype="float64", na_value=np.nan)
    non_finite = int((~np.isfinite(values)).sum())
    finite = values[np.isfinite(values)]
    carriers = int(np.count_nonzero(finite))
    total_mass_suppressed = 0 < carriers < minimum
    total_mass: float | None = None
    if non_finite == 0 and not total_mass_suppressed:
        candidate_total = float(values.sum())
        if np.isfinite(candidate_total):
            total_mass = candidate_total
    return {
        "missing_columns": [],
        "dtype": str(weights.dtype),
        "empty": bool(values.size == 0),
        "conversion_failures": sdc_count(conversion_failures, minimum=minimum),
        "non_finite": sdc_count(non_finite, minimum=minimum),
        "negative": sdc_count(int((values < 0).sum()), minimum=minimum),
        "all_zero": bool(values.size > 0 and non_finite == 0 and carriers == 0),
        # A weight total can reveal one or two nonzero unit records. Publish it
        # only when it aggregates zero or at least the configured minimum.
        "total_mass": total_mass,
        "total_mass_suppressed": total_mass_suppressed,
    }


def column_collisions(
    person: pd.DataFrame, benunit: pd.DataFrame, household: pd.DataFrame
) -> list[str]:
    """Column names appearing on more than one table (schema, not records)."""

    tables = (person, benunit, household)
    seen: dict[str, int] = {}
    for table in tables:
        for column in table.columns:
            seen[column] = seen.get(column, 0) + 1
    return sorted(name for name, uses in seen.items() if uses > 1)


def reserved_weight_column_collisions(
    person: pd.DataFrame, benunit: pd.DataFrame, household: pd.DataFrame
) -> list[str]:
    """Reserved weight columns Frame would reject, fully qualified.

    The preflight constructs typed weights only for ``household``. Therefore
    ``household.household_weight`` is the one permitted materialized column;
    every other ``{entity}_weight`` placement is a kernel-name collision.
    """

    tables = {"person": person, "benunit": benunit, "household": household}
    collisions: list[str] = []
    for entity in UK_SCHEMA.entities:
        reserved = f"{entity}_weight"
        for table_entity, table in tables.items():
            if reserved not in table.columns:
                continue
            if table_entity == entity == "household":
                continue
            collisions.append(f"{table_entity}.{reserved}")
    return sorted(collisions)


def payload_probe(table: pd.DataFrame) -> dict[str, Any]:
    """Shape and dtype histogram — the payload facts a rewrite must preserve."""

    dtypes: dict[str, int] = {}
    for dtype in table.dtypes:
        dtypes[str(dtype)] = dtypes.get(str(dtype), 0) + 1
    return {
        "rows": int(len(table)),
        "columns": int(table.shape[1]),
        "dtypes": dict(sorted(dtypes.items())),
        "index": type(table.index).__name__,
        "is_range_index": bool(isinstance(table.index, pd.RangeIndex)),
    }


def attempt_frame_construction(
    person: pd.DataFrame,
    benunit: pd.DataFrame,
    household: pd.DataFrame,
    *,
    weight_kind: Any,
) -> bool:
    """True iff Frame construction succeeds. Exception text is never surfaced:
    Frame's messages embed real ids, and the classification above already
    names every violation class this could fail on."""

    try:
        weights = Weights(
            values=household["household_weight"].to_numpy(dtype="float64"),
            kind=weight_kind,
        )
        Frame(
            tables={"person": person, "benunit": benunit, "household": household},
            schema=UK_SCHEMA,
            weights={"household": weights},
        )
    except Exception:
        return False
    return True


def _count_is_nonzero(value: Any) -> bool:
    """Whether an SDC-count field proves at least one violation."""

    return value is not None and value != 0


def _classified_frame_failure_reason(
    linkage: dict[str, Any],
    household_weights: dict[str, Any],
    collisions: list[str],
    reserved_collisions: list[str],
) -> str:
    """Return a static failure category without surfacing unit-record values."""

    person = linkage.get("person", {})
    if person.get("missing_columns"):
        return "Frame requires the classified person id column."
    if _count_is_nonzero(person.get("id_na")):
        return "Frame rejects missing person ids."
    if _count_is_nonzero(person.get("id_duplicated")):
        return "Frame rejects duplicated person ids."
    for group in GROUP_ENTITIES:
        classified = linkage.get(group, {})
        if classified.get("missing_columns"):
            return "Frame requires the classified group linkage columns."
        if _count_is_nonzero(classified.get("membership_na")):
            return "Frame rejects missing group memberships."
        if _count_is_nonzero(classified.get("id_na")):
            return "Frame rejects missing group ids."
        if _count_is_nonzero(classified.get("id_duplicated")):
            return "Frame rejects duplicated group ids."
        if classified.get("ids_sorted_ascending") is False:
            return "Frame requires group ids sorted ascending."
        if _count_is_nonzero(classified.get("orphaned_group_rows")):
            return "Frame rejects group rows without member persons."
        if _count_is_nonzero(classified.get("dangling_memberships")):
            return "Frame rejects person memberships without a group row."
    if collisions:
        return "Frame rejects column names duplicated across entity tables."
    if household_weights.get("missing_columns"):
        return "Frame construction requires household weights."
    if household_weights.get("empty"):
        return "Frame construction requires a nonempty household weight vector."
    if _count_is_nonzero(household_weights.get("conversion_failures")):
        return "Frame rejects non-numeric household weights."
    if _count_is_nonzero(household_weights.get("non_finite")):
        return "Frame rejects non-finite household weights."
    if _count_is_nonzero(household_weights.get("negative")):
        return "Frame rejects negative household weights."
    if household_weights.get("all_zero"):
        return "Frame rejects an all-zero household weight vector."
    if reserved_collisions:
        return "Frame rejects reserved weight-column collisions."
    return "Frame rejected an unclassified invariant; exception text was suppressed."


def _split_benunits_reported(linkage: dict[str, Any]) -> bool:
    return _count_is_nonzero(linkage.get("split_benunits"))


def preflight_artifact(path: Path, *, minimum: int) -> dict[str, Any]:
    """Run every check against one UK single-year H5 artifact."""

    minimum = _validated_sdc_minimum(minimum)
    weight_kind, _mass_log = read_uk_single_year_weight_metadata(path)
    with pd.HDFStore(path, mode="r") as store:
        keys = {key.lstrip("/") for key in store.keys()}
        missing = sorted(set(_TABLES) - keys)
        if missing:
            raise ValueError(f"UK artifact {path.name} is missing table(s): {missing}.")
        person = store["person"]
        benunit = store["benunit"]
        household = store["household"]
        time_period = (
            str(store["time_period"].iloc[0]) if "time_period" in keys else None
        )
    linkage = classify_linkage(person, benunit, household, minimum=minimum)
    household_weights = classify_household_weights(household, minimum=minimum)
    collisions = column_collisions(person, benunit, household)
    reserved_collisions = reserved_weight_column_collisions(person, benunit, household)
    frame_constructed = attempt_frame_construction(
        person, benunit, household, weight_kind=weight_kind
    )
    frame_failure_reason = (
        None
        if frame_constructed
        else _classified_frame_failure_reason(
            linkage,
            household_weights,
            collisions,
            reserved_collisions,
        )
    )
    split_benunits = _split_benunits_reported(linkage)
    failure_reasons = [frame_failure_reason] if frame_failure_reason else []
    if split_benunits:
        failure_reasons.append("Benunits span multiple households.")
    return {
        "path": str(path),
        "audit_completed": True,
        "time_period": time_period,
        "household_weight_kind": weight_kind.value,
        "tables": {
            "person": payload_probe(person),
            "benunit": payload_probe(benunit),
            "household": payload_probe(household),
        },
        "linkage": linkage,
        "household_weights": household_weights,
        "column_collisions": collisions,
        "reserved_weight_column_collisions": reserved_collisions,
        "frame_constructed": frame_constructed,
        "frame_construction_failure_reason": frame_failure_reason,
        "preflight_passed": frame_constructed and not split_benunits,
        "preflight_failure_reasons": failure_reasons,
    }


def _paths_alias(left: Path, right: Path) -> bool:
    """Compare lexical/resolved identity and filesystem inode identity."""

    resolved_alias = False
    try:
        resolved_alias = left.resolve(strict=False) == right.resolve(strict=False)
    except (OSError, RuntimeError):
        pass
    filesystem_alias = False
    try:
        filesystem_alias = left.samefile(right)
    except OSError:
        pass
    return resolved_alias or filesystem_alias


def _json_output_aliases_input(json_out: Path, h5_paths: list[Path]) -> bool:
    return any(_paths_alias(json_out, path) for path in h5_paths)


def _incomplete_artifact_report(path: Path) -> dict[str, Any]:
    reason = "Artifact could not be audited; exception text was suppressed."
    return {
        "path": str(path),
        "audit_completed": False,
        "frame_constructed": False,
        "frame_construction_failure_reason": reason,
        "preflight_passed": False,
        "preflight_failure_reasons": [reason],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify UK H5 artifacts against populace Frame linkage "
            "invariants (#612), with SDC-safe output."
        )
    )
    parser.add_argument(
        "h5",
        nargs="+",
        type=Path,
        help="UK single-year H5 artifacts (opened read-only).",
    )
    parser.add_argument(
        "--sdc-minimum-count",
        type=int,
        default=DEFAULT_SDC_MINIMUM_COUNT,
        help=(
            "Counts below this (other than zero) are reported as a "
            "threshold, never exactly "
            f"(default: {DEFAULT_SDC_MINIMUM_COUNT}; minimum: "
            f"{MINIMUM_SDC_COUNT})."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Also write the report JSON to this path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        minimum = _validated_sdc_minimum(args.sdc_minimum_count)
    except ValueError:
        print(
            f"error: --sdc-minimum-count must be at least {MINIMUM_SDC_COUNT}.",
            file=sys.stderr,
        )
        return 2
    if args.json_out is not None and _json_output_aliases_input(args.json_out, args.h5):
        print(
            "error: --json-out must not alias any H5 input.",
            file=sys.stderr,
        )
        return 2
    artifacts: list[dict[str, Any]] = []
    for path in args.h5:
        try:
            artifact = preflight_artifact(path, minimum=minimum)
        except Exception:
            artifact = _incomplete_artifact_report(path)
        artifacts.append(artifact)
    report = {
        "sdc_minimum_count": minimum,
        "artifacts": artifacts,
    }
    if args.json_out is not None and _json_output_aliases_input(args.json_out, args.h5):
        print(
            "error: --json-out became an alias of an H5 input.",
            file=sys.stderr,
        )
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    failed = [
        artifact["path"]
        for artifact in report["artifacts"]
        if not artifact["preflight_passed"]
    ]
    if failed:
        print(
            f"FAIL: {len(failed)} artifact(s) do not pass linkage preflight.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
