"""List counties whose county SOI AGI row R2 would build empty (pre-check).

The ACS local release (R2, ``tools/build_us_acs_local_release.py``) builds a
county's IRS SOI rows in ``attach_surface`` from the stored entries of the
state SOI rows for the county's households (``split_state_rows``). The state
AGI row stores no zero, so when every household of a county has an AGI of
exactly 0 the county's AGI row is empty. ``attach_surface`` then refuses the
run, after the materialize stage's engine pass (microcosm#1630 review,
finding 1). Minimum county support fills only counties with no household, so
it does not help here.

This check runs on the staging H5 (R1's output) and its summary, before R2.
It computes the value R2 stores, with R2's own functions and the engine:

- It loads the staging frame as R2's materialize stage does
  (``_load_staging_frame``), holds back the formula-owned columns
  (``project_input_only``) and fills the reviewed nulls with the engine
  defaults (``fill_reviewed_nulls``, which refuses what R2 refuses).
- It builds each household chunk's policyengine-us dataset as R2's engine
  pass does (``_dataset_from_frame``) and computes the household sum of
  ``adjusted_gross_income``. That is the value R2's SOI materializer stores
  in the state AGI row: it counts every tax unit, filer or not (the
  engine-tier test compares the two).
- The engine counts only the positive income sources of a tax unit's head
  and spouse (``irs_gross_income``), and only an adult can be a head or a
  spouse. A household whose income is all on dependents, or on a tax unit
  without an adult, has an AGI of 0 unless a deduction makes it negative.
- R2's stages fill some ``NEVER_DEFAULT_FILLED`` columns on the ACS rows
  before its fill step (``stage_filled_never_default``, shared with the
  reviewed-fill pre-check). The check holds back such a column while its
  nulls are on ACS rows only and the AGI does not read it. It refuses every
  other null in a never-default column, as R2's fill step does: a null on a
  donor row, a null in a column that no R2 stage fills, and a null the AGI
  reads. What the AGI reads is the engine's static dependency closure of
  ``adjusted_gross_income`` in the check's year: the walk stops at an
  above-the-line deduction that year's ``gov.irs.ald.deductions`` does not
  list (:data:`YEAR_LISTED_DEDUCTIONS`), whose formula then reads nothing.
- With ``--puma-ladder`` it applies R2's own minimum-support moves
  (``plan_minimum_support`` with ``--county-min-support``, default R2's 3)
  and takes the county list from the ladder, as R2 does.
- A county is **at risk** when it has households but none with a nonzero AGI
  as R2 stores it (float32). Return-count rows count every tax unit, so they
  are never empty.
- R2's ``check_support`` (materialize, after the staging load and the
  SSI/Medicaid pre-pass, before its engine pass) refuses the run when a
  household names a county outside
  the ladder or a ladder county has no household-universe record after the
  minimum-support moves. With ``--puma-ladder`` the check lists both
  (``counties_outside_ladder``, ``unsupported_counties``) with R2's set
  rules, and, as R2 does, moves no household when a county is outside the
  ladder.

The check needs memory for the full staging frame, as R2's materialize stage
does, and runs the engine for the AGI alone (minutes, not R2's full pass).

Exit status 0: no county at risk and none that R2 refuses. Exit status 2: at
least one county at risk; the JSON report (``--out``) lists them. R2 would
refuse each one's empty AGI row after its engine pass, so each needs a fix
before R2 runs. Exit status 3: R2 would refuse the run in materialize before
its engine pass (a county outside the ladder, or without a household-universe
record); fix the staging file or the ladder first. Status 3 takes precedence
over 2 and needs ``--puma-ladder``.

Usage::

    python tools/check_us_acs_county_agi_support.py \\
        --h5 <run>/acs_multispine_staging.h5 \\
        --county-column county_fips_2023 \\
        --puma-ladder build/us/us_puma_ladder_2020.npz \\
        --out <run>/county_agi_support.json

``--staging-summary`` defaults to the H5 path with ``.summary.json``, as in
R2.
"""

# No ``from __future__ import annotations``: test loaders run this file
# without registering it in ``sys.modules``.
import argparse
import gc
import importlib.util
import json
import sys
import time
from collections import defaultdict
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_TOOLS = Path(__file__).resolve().parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import check_us_acs_local_reviewed_fills as reviewed_fills  # noqa: E402
import us_acs_local_county_surface as county_surface  # noqa: E402

from microcosm.frame import Frame  # noqa: E402

TOOL = "tools/check_us_acs_county_agi_support.py"
ISSUE = "microcosm#922"
#: The engine variable whose household sum R2's state SOI AGI row stores.
AGI_VARIABLE = "adjusted_gross_income"
EXIT_AT_RISK = 2
#: R2's ``check_support`` refuses the run, before its engine pass.
EXIT_R2_REFUSES = 3
#: Households per engine chunk; the AGI closure alone is light.
DEFAULT_HH_CHUNK = 100_000
_RELEASE_TOOL = _TOOLS / "build_us_acs_local_release.py"


def _release_tool():
    """R2's module, for its loader, projection, fill and constants."""

    spec = importlib.util.spec_from_file_location(
        "build_us_acs_local_release_for_agi_check", _RELEASE_TOOL
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


#: Above-the-line deductions whose formula reads nothing in a year whose
#: ``gov.irs.ald.deductions`` does not list them. On policyengine-us 2.145.3
#: the 2020-only charitable deduction for non-itemizers reads the itemization
#: choice in 2020, and through it the inputs of every itemized deduction,
#: among them take-up flags R2's stages fill. The static closure follows that
#: path in every year, so in another year it would refuse those flags' nulls,
#: which cannot change the AGI then.
YEAR_LISTED_DEDUCTIONS = ("charitable_contributions_ald",)


def unread_deductions(period: int, *, system: Any = None) -> tuple[str, ...]:
    """The :data:`YEAR_LISTED_DEDUCTIONS` that ``period`` does not list.

    ``system`` is a policyengine-us ``CountryTaxBenefitSystem``; it is built
    when not given.
    """

    if system is None:
        from policyengine_us import CountryTaxBenefitSystem

        system = CountryTaxBenefitSystem()
    listed = set(system.parameters(f"{int(period)}-01-01").gov.irs.ald.deductions)
    return tuple(name for name in YEAR_LISTED_DEDUCTIONS if name not in listed)


def agi_engine_inputs(index: Any, *, unread: Collection[str] = ()) -> frozenset[str]:
    """The engine inputs ``adjusted_gross_income`` can read (static closure).

    ``index`` is a ``PolicyEngineUSVariableMetadataIndex`` (or a stand-in
    with ``variable_dependency_closure``). The walk does not go on past a
    variable in ``unread`` (:func:`unread_deductions`), whose formula reads
    nothing in the check's year.
    """

    closure = index.variable_dependency_closure(AGI_VARIABLE)
    unread = frozenset(unread)
    if not unread:
        return frozenset(closure.input_leaves)
    targets: dict[str, set[str]] = defaultdict(set)
    for consumer, target in closure.edges:
        targets[consumer].add(target)
    reached: set[str] = set()
    pending = [AGI_VARIABLE]
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        if name not in unread:
            pending.extend(targets[name])
    return frozenset(reached.intersection(closure.input_leaves))


def hold_back_null_runtime_columns(
    frame: Frame,
    *,
    never_default_filled: Collection[tuple[str, str]],
    stage_filled: Collection[tuple[str, str]],
    agi_inputs: Collection[str],
) -> tuple[Frame, dict[str, list[str]]]:
    """Drop the null never-default columns that R2's stages fill first.

    R2's stages fill the ``stage_filled`` columns on the ACS rows before its
    fill step, and the fill step refuses any null left in a never-default
    column. So the check holds back a null column only when an R2 stage fills
    it, every null is on an ACS row (the ``<entity>_spine`` tag) and the AGI
    does not read it. Every other null in a never-default column is refused,
    as R2 refuses it.
    """

    from microcosm.build.us_runtime.base_pool import spine_column

    agi_inputs = frozenset(agi_inputs)
    stage_filled = frozenset(stage_filled)
    held: dict[str, list[str]] = {}
    tables = {}
    refused = []
    for entity in frame.entities:
        table = frame.table(entity)
        tag = spine_column(entity)
        null = []
        for column in sorted(
            column
            for owner, column in never_default_filled
            if owner == entity
            and column in table.columns
            and table[column].isna().any()
        ):
            name = f"{entity}.{column}"
            missing = table[column].isna()
            if (entity, column) not in stage_filled:
                refused.append(f"{name}: no R2 stage fills it (re-run staging)")
            elif tag not in table.columns:
                refused.append(f"{name}: no {tag} column places its null rows")
            elif off := int((missing & table[tag].ne(county_surface.ACS_SPINE)).sum()):
                refused.append(
                    f"{name}: null on {off} non-ACS row(s); R2's stages fill "
                    "the ACS rows only"
                )
            elif column in agi_inputs:
                refused.append(f"{name}: the AGI reads it")
            else:
                null.append(column)
        if null:
            held[entity] = null
            table = table.drop(columns=null)
        tables[entity] = table
    if refused:
        raise SystemExit(
            "Null never-default-filled column(s) that R2's materialize stage "
            f"refuses too (fill_reviewed_nulls): {'; '.join(refused)}."
        )
    if not held:
        return frame, held
    weights = {entity: frame.weights_for(entity) for entity in frame.weighted_entities}
    held_frame = Frame(
        tables, frame.schema, weights, frame.strata, mass_log=frame.mass_log
    )
    return held_frame, held


def household_agi(
    frame: Frame,
    summary_path: Path,
    *,
    release: Any,
    period: int | None = None,
    hh_chunk: int = DEFAULT_HH_CHUNK,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Each household's AGI as R2's SOI materializer stores it.

    Returns the household sum of ``adjusted_gross_income`` in the order of
    ``frame``'s household table, and a JSON-ready receipt. ``release`` is
    R2's module (``_release_tool()``).
    """

    import build_us_fiscal_refresh_release as fiscal
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    from microcosm.frame.adapters.policyengine_us import (
        PolicyEngineUSVariableMetadataIndex,
    )

    period = release.PERIOD if period is None else int(period)
    system = CountryTaxBenefitSystem()
    unread = unread_deductions(period, system=system)
    projected, formula_owned = release.project_input_only(frame, period=period)
    projected, held = hold_back_null_runtime_columns(
        projected,
        never_default_filled=release.NEVER_DEFAULT_FILLED,
        stage_filled=reviewed_fills.stage_filled_never_default(release),
        agi_inputs=agi_engine_inputs(
            PolicyEngineUSVariableMetadataIndex(), unread=unread
        ),
    )
    fills, drift = release.fill_reviewed_nulls(
        projected, summary_path, manifest_path=None, period=period, system=system
    )
    household_ids = projected.table("household")["household_id"].to_numpy()
    n_households = len(household_ids)
    positions = pd.Index(household_ids)
    if not positions.is_unique:
        raise ValueError("Household ids must be unique.")
    person_position = positions.get_indexer(
        projected.table("person")["person_household_id"].to_numpy()
    )
    if (person_position < 0).any():
        raise ValueError("Persons name households the household table lacks.")
    if n_households > hh_chunk:
        # A chunk keeps every group its persons reference, so a group across
        # two chunks would be simulated twice, each time in part.
        fiscal._assert_group_entities_nest_in_households(projected)
    agi = np.zeros(n_households, dtype=np.float64)
    chunks = 0
    for low in range(0, n_households, hh_chunk):
        high = min(low + hh_chunk, n_households)
        chunk = projected.select((person_position >= low) & (person_position < high))
        simulation = Microsimulation(
            dataset=fiscal._dataset_from_frame(
                chunk, assert_no_formula_owned_columns=False, period_only=True
            )
        )
        got = np.asarray(simulation.calculate("household_id", period).values)
        if not np.array_equal(got, household_ids[low:high]):
            raise RuntimeError(
                f"Engine chunk {chunks} does not carry its households in frame "
                "order; refusing to place its AGI."
            )
        agi[low:high] = np.asarray(
            simulation.calculate(AGI_VARIABLE, period, map_to="household").values,
            dtype=np.float64,
        )
        chunks += 1
        del simulation, chunk
        gc.collect()
    return agi, {
        "variable": AGI_VARIABLE,
        "aggregation": "household sum over tax units",
        "period": period,
        "engine_version": release._installed_engine_version(),
        "households_per_chunk": int(hh_chunk),
        "chunks": chunks,
        "formula_owned_columns_held_back": sum(len(c) for c in formula_owned.values()),
        "null_runtime_columns_held_back": held,
        "agi_closure_stops_at": list(unread),
        "reviewed_fill_columns": len(fills),
        "reviewed_fill_count_drift": len(drift),
    }


def county_agi_report(
    households: pd.DataFrame,
    nonzero_agi: np.ndarray,
    *,
    county_column: str,
    county_populations: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """Per-county household counts, the counties at risk and R2's refusals.

    ``nonzero_agi`` flags the households (in ``households`` order) whose AGI
    R2 would store as nonzero; the ``households_with_agi_income`` keys count
    them. ``county_populations`` (the ladder's, as R2 uses) sets the county
    list; without it the list is the counties the households name, and
    ``counties_outside_ladder`` and ``unsupported_counties`` are None (not
    checked). With it, they are what R2's ``check_support`` refuses: counties
    the households name that are not in the ladder, and ladder counties
    without a household-universe record (``support_universe``).
    """

    codes = county_surface.county_codes(households, county_column=county_column)
    support = county_surface.support_universe(households)
    totals, supporting, with_income = (defaultdict(int) for _ in range(3))
    for code, supports, earns in zip(
        codes.tolist(), support.tolist(), np.asarray(nonzero_agi).tolist(), strict=True
    ):
        totals[code] += 1
        supporting[code] += int(supports)
        with_income[code] += int(earns)
    if county_populations is None:
        counties = sorted(totals)
        ladder = None
    else:
        counties = sorted(
            int(code) for code, value in county_populations.items() if float(value) > 0
        )
        ladder = {int(code) for code in county_populations}
    at_risk, no_household, thin = [], [], []
    rows = {}
    for code in counties:
        record = {
            "households": totals.get(code, 0),
            "supporting_households": supporting.get(code, 0),
            "households_with_agi_income": with_income.get(code, 0),
        }
        rows[f"{code:05d}"] = record
        if record["households"] == 0:
            no_household.append(f"{code:05d}")
        elif record["households_with_agi_income"] == 0:
            at_risk.append(f"{code:05d}")
        elif (
            record["households_with_agi_income"]
            < county_surface.DEFAULT_COUNTY_MIN_SUPPORT
        ):
            thin.append(f"{code:05d}")
    # R2's check_support compares with every ladder county, whatever its
    # population.
    outside = unsupported = None
    if ladder is not None:
        outside = sorted(set(totals) - ladder)
        unsupported = sorted(ladder - {code for code, n in supporting.items() if n})
    return {
        "county_column": county_column,
        "counties": len(counties),
        "households": int(len(households)),
        "households_with_agi_income": int(np.asarray(nonzero_agi).sum()),
        "at_risk": at_risk,
        "at_risk_detail": {code: rows[code] for code in at_risk},
        "no_household": no_household,
        "fewer_than_min_support_with_agi_income": thin,
        "counties_outside_ladder": (
            None if outside is None else [f"{code:05d}" for code in outside]
        ),
        "unsupported_counties": (
            None if unsupported is None else [f"{code:05d}" for code in unsupported]
        ),
        "by_county": rows,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List counties whose county SOI AGI row R2 would build empty: "
            "counties with households but none with a nonzero AGI, computed "
            "with R2's own fill and engine path."
        )
    )
    parser.add_argument(
        "--h5", required=True, type=Path, help="Staging H5 (R1 output)."
    )
    parser.add_argument(
        "--staging-summary",
        type=Path,
        help=(
            "R1's staging summary, for the reviewed-null fills (default: the "
            "H5 path with .summary.json, as in R2)."
        ),
    )
    parser.add_argument(
        "--county-column",
        default=county_surface.COUNTY_COLUMN_2023,
        choices=county_surface.COUNTY_FIPS_COLUMNS,
        help="R2's --county-column (default county_fips_2023).",
    )
    parser.add_argument(
        "--puma-ladder",
        type=Path,
        help=(
            "R2's PUMA ladder. With it, the check takes the county list from "
            "the ladder, applies R2's minimum-support moves first and lists "
            "the counties R2 refuses (exit status 3)."
        ),
    )
    parser.add_argument(
        "--county-min-support",
        type=int,
        default=county_surface.DEFAULT_COUNTY_MIN_SUPPORT,
        help=(
            "R2's --county-min-support (default "
            f"{county_surface.DEFAULT_COUNTY_MIN_SUPPORT}); used only with --puma-ladder."
        ),
    )
    parser.add_argument(
        "--hh-chunk",
        type=int,
        default=DEFAULT_HH_CHUNK,
        help=f"Households per engine chunk (default {DEFAULT_HH_CHUNK:,}).",
    )
    parser.add_argument("--out", type=Path, help="JSON report path.")
    return parser.parse_args(argv)


def ladder_support(
    households: pd.DataFrame,
    ladder_path: Path,
    *,
    county_column: str,
    county_min_support: int,
) -> tuple[pd.DataFrame, dict[int, float], dict[str, Any] | None]:
    """R2's ladder county populations and minimum-support moves.

    Returns ``(households, county_populations, support)``: the households
    after the moves, as ``check_support`` rewrites them, and the moves'
    receipt (None when nothing was planned). As in R2, no household moves
    when one names a county outside the ladder; R2 then refuses the run.
    """

    from microcosm.build.us_runtime.puma_ladder import load_us_puma_ladder

    ladder = load_us_puma_ladder(ladder_path)
    populations = county_surface.with_county_populations(
        ladder, {}, {"county": True}, county_column=county_column
    )["county"]
    codes = county_surface.county_codes(households, county_column=county_column)
    outside = {int(code) for code in codes.tolist()} - {
        int(code) for code in populations
    }
    needed = county_surface.unsupported_counties(
        households, populations, county_column=county_column
    )
    support = None
    if county_min_support > 0 and needed and not outside:
        overlaps = county_surface.county_puma_overlaps(ladder_path, county_column)
        moves, support = county_surface.plan_minimum_support(
            households,
            populations,
            overlaps,
            county_column=county_column,
            per_county=county_min_support,
        )
        households = county_surface.apply_support_moves(
            households,
            moves,
            county_column=county_column,
            columns=support["rewritten_columns"],
        )
    return households, populations, support


def exit_status(report: Mapping[str, Any]) -> int:
    """3 when R2's ``check_support`` refuses the run, else 2 when one is at risk."""

    if report.get("counties_outside_ladder") or report.get("unsupported_counties"):
        return EXIT_R2_REFUSES
    return EXIT_AT_RISK if report["at_risk"] else 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.time()
    summary_path = args.staging_summary or args.h5.with_suffix(".summary.json")
    if not summary_path.exists():
        raise SystemExit(
            f"Staging summary not found at {summary_path}; the reviewed-null "
            "fills need its reviewed_engine_input_nulls register."
        )
    release = _release_tool()
    frame = release._load_staging_frame(args.h5)
    households = frame.table("household")
    agi, engine = household_agi(
        frame, summary_path, release=release, hh_chunk=args.hh_chunk
    )
    del frame
    gc.collect()
    # R2 stores the household measures as float32 and keeps only nonzeros.
    nonzero = agi.astype(np.float32) != 0
    populations = None
    support = None
    if args.puma_ladder is not None:
        households, populations, support = ladder_support(
            households,
            args.puma_ladder,
            county_column=args.county_column,
            county_min_support=args.county_min_support,
        )
    report = county_agi_report(
        households,
        nonzero,
        county_column=args.county_column,
        county_populations=populations,
    )
    report.update(
        {
            "tool": TOOL,
            "issue": ISSUE,
            "h5": str(args.h5),
            "staging_summary": str(summary_path),
            "agi_engine": engine,
            "minimum_support": (
                None
                if support is None
                else {
                    "households_per_county": args.county_min_support,
                    "moves": len(support.get("moves", [])),
                    "moves_sha256": support.get("moves_sha256"),
                    "moved_counties": support.get("moved_counties", []),
                }
            ),
            "seconds": round(time.time() - started, 1),
        }
    )
    status = exit_status(report)
    report["exit_status"] = status
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    summary = {
        key: report[key]
        for key in (
            "county_column",
            "counties",
            "households",
            "households_with_agi_income",
            "at_risk",
            "at_risk_detail",
            "no_household",
            "fewer_than_min_support_with_agi_income",
            "counties_outside_ladder",
            "unsupported_counties",
            "exit_status",
            "seconds",
        )
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
