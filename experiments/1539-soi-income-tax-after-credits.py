#!/usr/bin/env python3
"""Score a release's SOI income tax after credits rows before and after #1539.

Before microcosm#1539 the release tool built SOI's A06500 amount rows as
``max(income_tax, 0)`` per tax unit and the N06500 return rows as
``income_tax > 0``. After it, both read Form 1040 line 22:
``income_tax_before_credits - income_tax_capped_non_refundable_credits``,
positive part. This script scores both definitions on a published release H5
with the installed policyengine-us, in batches of whole households, under the
release's initial and final household weights. It covers every income tax
after credits row the release's calibration diagnostics list (all-AGI,
all-filing-status rows, national or one state) and writes aggregates only:
weighted sums and counts, never a record.

It first rebuilds the old definition's initial and final estimates and
compares them with the ones the diagnostics record, which checks that the
reconstruction matches the release tool's.

    uv run --no-sync python experiments/1539-soi-income-tax-after-credits.py \\
        --snapshot ~/.cache/huggingface/hub/datasets--policyengine--populace-us/snapshots/<sha> \\
        --release-id populace-us-2024-spm-20260915 \\
        --output experiments/1539-soi-income-tax-after-credits.json

``--summarize <json> --output <md>`` renders the receipts table from a
finished run without the engine.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from importlib import metadata
from pathlib import Path

import numpy as np

PERIOD = 2024
BATCH_HOUSEHOLDS = 2_000
SPM = {"geography_kind": "county"}  # reform_validation.US_RELEASE_SPM_SELECTION
ENTITIES = ("person", "household", "tax_unit", "spm_unit", "family", "marital_unit")
VARIABLES = (
    "income_tax",
    "income_tax_before_credits",
    "income_tax_capped_non_refundable_credits",
    "income_tax_non_refundable_credits",
    "income_tax_refundable_credits",
    "net_investment_income_tax",
    "income_tax_before_refundable_credits",
)
STATE_FIPS = {
    "al": 1, "ak": 2, "az": 4, "ar": 5, "ca": 6, "co": 8, "ct": 9, "de": 10,
    "dc": 11, "fl": 12, "ga": 13, "hi": 15, "id": 16, "il": 17, "in": 18,
    "ia": 19, "ks": 20, "ky": 21, "la": 22, "me": 23, "md": 24, "ma": 25,
    "mi": 26, "mn": 27, "ms": 28, "mo": 29, "mt": 30, "ne": 31, "nv": 32,
    "nh": 33, "nj": 34, "nm": 35, "ny": 36, "nc": 37, "nd": 38, "oh": 39,
    "ok": 40, "or": 41, "pa": 42, "ri": 44, "sc": 45, "sd": 46, "tn": 47,
    "tx": 48, "ut": 49, "vt": 50, "va": 51, "wa": 53, "wv": 54, "wi": 55,
    "wy": 56,
}  # fmt: skip


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(diagnostics: dict) -> list[dict]:
    """The income tax after credits rows, with the slice each one selects."""

    rows = []
    for target in diagnostics["targets"]:
        metadata_ = target.get("metadata") or {}
        if metadata_.get("source_measure_id") not in {
            "income_tax_liability_amount",
            "income_tax_liability_returns",
        }:
            continue
        if (
            metadata_.get("agi_lower_bound") != "-inf"
            or metadata_.get("agi_upper_bound") != "inf"
            or metadata_.get("filing_status") != "All"
        ):
            raise ValueError(f"{target['name']}: not an all-AGI, all-status row")
        state = metadata_.get("state_fips")
        rows.append(
            {
                "name": target["name"],
                "state_fips": None if state is None else int(state),
                "indicator": metadata_.get("measure_mode") == "indicator_sum",
                "source_variable": metadata_.get("source_variable"),
                "target": float(target["target"]),
                "recorded_initial_estimate": float(target["initial_estimate"]),
                "recorded_final_estimate": float(target["final_estimate"]),
            }
        )
    return rows


def _batches(tables: dict, household_ids: np.ndarray):
    person = tables["person"]
    for start in range(0, len(household_ids), BATCH_HOUSEHOLDS):
        ids = household_ids[start : start + BATCH_HOUSEHOLDS]
        people = person.loc[person["person_household_id"].isin(ids)]
        batch = {"person": people.reset_index(drop=True)}
        for group in ENTITIES[1:]:
            frame = tables[group]
            members = people[f"person_{group}_id"].unique()
            batch[group] = frame.loc[frame[f"{group}_id"].isin(members)].reset_index(
                drop=True
            )
        yield batch


def _tax_unit_values(dataset_path: Path, max_batches: int | None):
    """Each tax unit's engine values and household id, and the H5's tables."""

    from policyengine_us import Microsimulation
    from policyengine_us.data import USSingleYearDataset

    data = USSingleYearDataset(file_path=str(dataset_path))
    tables = {entity: getattr(data, entity) for entity in ENTITIES}
    household_ids = tables["household"]["household_id"].to_numpy()
    out: dict[str, list[np.ndarray]] = {
        name: [] for name in (*VARIABLES, "household_id")
    }
    for index, batch in enumerate(_batches(tables, household_ids)):
        if max_batches is not None and index >= max_batches:
            break
        simulation = Microsimulation(
            dataset=USSingleYearDataset(**batch, time_period=PERIOD), spm=dict(SPM)
        )
        table_units = batch["tax_unit"]["tax_unit_id"].to_numpy()
        engine_units = np.asarray(simulation.calculate("tax_unit_id", PERIOD))
        engine_units = engine_units.astype(table_units.dtype)
        if not np.array_equal(engine_units, table_units):
            raise ValueError("engine tax unit order differs from the batch table")
        person = batch["person"]
        unit_household = (
            person.drop_duplicates("person_tax_unit_id")
            .set_index("person_tax_unit_id")["person_household_id"]
            .reindex(engine_units)
            .to_numpy()
        )
        out["household_id"].append(unit_household)
        for variable in VARIABLES:
            out[variable].append(
                np.asarray(simulation.calculate(variable, PERIOD), dtype=np.float64)
            )
        del simulation
        gc.collect()
        print(f"batch {index + 1}: {len(engine_units)} tax units", file=sys.stderr)
    return {name: np.concatenate(parts) for name, parts in out.items()}, tables


_POSTAL = {fips: postal.upper() for postal, fips in STATE_FIPS.items()}


def _relative(value: float, target: float) -> str:
    return f"{(value - target) / target:+.2%}"


def summarize(report: dict) -> str:
    """The receipts table: national rows, engine totals and every state."""

    rows = report["rows"]
    national = {row["indicator"]: row for row in rows if row["state_fips"] is None}
    lines = [
        "# SOI income tax after credits on the published release (microcosm#1539)",
        "",
        "This run scores the published Microcosm US release whose H5 SHA-256 "
        f"begins `{report['dataset_sha256'][:12]}` (calibration diagnostics "
        f"`{report['diagnostics_sha256'][:12]}`) with policyengine-us "
        f"{report['policyengine_us']} and policyengine-core "
        f"{report['policyengine_core']}, the versions that built the release, "
        f"over {report['households']:,} households and "
        f"{report['tax_units']:,} tax units for {report['period']}. "
        "**Before** is the old row definition, `max(income_tax, 0)` per tax "
        "unit (`income_tax > 0` for return rows); **after** is Form 1040 line "
        "22, `max(income_tax_before_credits - "
        "income_tax_capped_non_refundable_credits, 0)`. **Initial** and "
        "**final** are the release's calibration weights before and after "
        "fitting. All figures are weighted sums; no record leaves the run.",
        "",
        "## Reconstruction check",
        "",
    ]
    initial = [
        abs(row["before_initial"] / row["recorded_initial_estimate"] - 1)
        for row in rows
    ]
    final = [
        abs(row["before_final"] / row["recorded_final_estimate"] - 1) for row in rows
    ]
    lines += [
        "On the published H5, the old definition comes within "
        f"{max(initial):.2%} of every initial estimate and {max(final):.2%} of "
        "every final estimate the release's diagnostics record (national "
        f"amount {_relative(national[False]['before_final'], national[False]['recorded_final_estimate'])}, "
        f"national returns {_relative(national[True]['before_final'], national[True]['recorded_final_estimate'])} "
        "at final weights). Both definitions below come from the same H5 run, "
        "so their difference compares like with like.",
        "",
        "## National rows",
        "",
        "| Row | SOI target | Weights | Before | Error | After | Error | After / before |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for indicator, label, scale, unit in (
        (False, "A06500 amount", 1e9, "$B"),
        (True, "N06500 returns", 1e6, "M"),
    ):
        row = national[indicator]
        for weighting in ("initial", "final"):
            before = row[f"before_{weighting}"]
            after = row[f"after_{weighting}"]
            lines.append(
                f"| {label} ({unit}) | {row['target'] / scale:,.2f} | {weighting} | "
                f"{before / scale:,.2f} | {_relative(before, row['target'])} | "
                f"{after / scale:,.2f} | {_relative(after, row['target'])} | "
                f"{after / before - 1:+.2%} |"
            )
    totals = report["final_weighted_national_totals"]
    lines += [
        "",
        "## Engine totals at final weights ($B)",
        "",
        "| Variable | Total |",
        "|---|---:|",
        *(f"| `{name}` | {totals[name] / 1e9:,.2f} |" for name in sorted(totals)),
        "",
        "Line 22 is `income_tax_before_credits` less "
        "`income_tax_capped_non_refundable_credits`; "
        "`income_tax_before_refundable_credits` adds the net investment "
        "income tax and three other Schedule 2 Part II taxes (the recapture "
        "of investment credit, unreported payroll tax and the qualified "
        "retirement penalty) to it, and `income_tax` subtracts the "
        "refundable credits.",
        "",
        "## States at final weights",
        "",
        "| State | Amount target ($B) | Before | After | Returns target (M) | Before | After |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    states = {}
    for row in rows:
        if row["state_fips"] is not None:
            states.setdefault(row["state_fips"], {})[row["indicator"]] = row
    for fips in sorted(states, key=lambda code: _POSTAL[code]):
        amount, count = states[fips][False], states[fips][True]
        lines.append(
            f"| {_POSTAL[fips]} | {amount['target'] / 1e9:,.2f} | "
            f"{_relative(amount['before_final'], amount['target'])} | "
            f"{_relative(amount['after_final'], amount['target'])} | "
            f"{count['target'] / 1e6:,.3f} | "
            f"{_relative(count['before_final'], count['target'])} | "
            f"{_relative(count['after_final'], count['target'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summarize", type=Path, help="a finished run's JSON")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--release-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="smoke test: score only this many household batches",
    )
    args = parser.parse_args(argv)
    if args.summarize is not None:
        args.output.write_text(summarize(json.loads(args.summarize.read_text())))
        return 0
    if args.snapshot is None or args.release_id is None:
        parser.error("--snapshot and --release-id are required to score")

    snapshot = args.snapshot.expanduser()
    release = snapshot / "releases" / args.release_id
    dataset_path = snapshot / "populace_us_2024.h5"
    diagnostics_path = release / "calibration_diagnostics.json"
    manifest = json.loads((release / "release_manifest.json").read_text())
    artifacts = manifest["artifacts"]
    dataset_sha = _sha256(dataset_path)
    diagnostics_sha = _sha256(diagnostics_path)
    if dataset_sha != artifacts["populace_us_2024"]["sha256"]:
        raise ValueError("the H5 is not the release manifest's")
    if diagnostics_sha != artifacts["calibration_diagnostics"]["sha256"]:
        raise ValueError("the diagnostics are not the release manifest's")
    weights_path = snapshot / "populace_us_2024_calibration.npz"
    weights = np.load(weights_path, allow_pickle=False)
    diagnostics = json.loads(diagnostics_path.read_text())
    rows = _rows(diagnostics)

    values, tables = _tax_unit_values(dataset_path, args.max_batches)
    household = tables["household"]
    household_ids = household["household_id"].to_numpy()
    if args.max_batches is not None:
        covered = np.isin(household_ids, values["household_id"])
        household = household.loc[covered].reset_index(drop=True)
        household_ids = household["household_id"].to_numpy()
    final_weight = household["household_weight"].to_numpy(np.float64)
    initial_weight = np.asarray(weights["initial_household_weight"], np.float64)
    npz_final = np.asarray(weights["household_weight"], np.float64)
    if len(initial_weight) != len(tables["household"]):
        raise ValueError("the npz weights do not cover the H5 households")
    if args.max_batches is not None:
        initial_weight = initial_weight[covered]
        npz_final = npz_final[covered]
    position = {value: index for index, value in enumerate(household_ids)}
    unit_position = np.asarray([position[h] for h in values["household_id"]])
    unit_state = household["state_fips"].to_numpy()[unit_position].astype(int)
    line_22 = (
        values["income_tax_before_credits"]
        - values["income_tax_capped_non_refundable_credits"]
    )
    definitions = {
        "before": np.maximum(values["income_tax"], 0.0),
        "after": np.maximum(line_22, 0.0),
    }
    weightings = {
        "initial": initial_weight[unit_position],
        "final": final_weight[unit_position],
    }

    scored = []
    for row in rows:
        mask = (
            np.ones(len(unit_state), dtype=bool)
            if row["state_fips"] is None
            else unit_state == row["state_fips"]
        )
        entry = {key: row[key] for key in row}
        for definition, component in definitions.items():
            value = (
                (component > 0).astype(np.float64) if row["indicator"] else component
            )
            for weighting, unit_weight in weightings.items():
                entry[f"{definition}_{weighting}"] = float(
                    np.dot(unit_weight, value * mask)
                )
        scored.append(entry)

    def total(name):
        return float(np.dot(weightings["final"], name))

    national_identity = {variable: total(values[variable]) for variable in VARIABLES}
    report = {
        "release_id": args.release_id,
        "dataset_sha256": dataset_sha,
        "diagnostics_sha256": diagnostics_sha,
        "weights_npz_sha256": _sha256(weights_path),
        "npz_final_equals_h5_weight_max_abs_difference": float(
            np.max(np.abs(npz_final - final_weight))
        ),
        "policyengine_us": metadata.version("policyengine-us"),
        "policyengine_core": metadata.version("policyengine-core"),
        "period": PERIOD,
        "tax_units": int(len(line_22)),
        "households": int(len(household_ids)),
        "final_weighted_national_totals": national_identity,
        "rows": scored,
    }
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.output} ({len(scored)} rows)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
