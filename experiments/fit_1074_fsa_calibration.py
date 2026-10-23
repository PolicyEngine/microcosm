#!/usr/bin/env python
"""Fit Federal Student Aid portfolio targets on the published US default (#1074).

What it measures. The published default ``populace-us-2024-spm-20260915`` with
the SCF 2025 education-debt stage run on it (seed 0, as microcosm#1073's host
receipt), calibrated on its own 57,240-household support:

- **release**: the published weights, which never saw the student loan
  columns (the "before");
- **A, no FSA**: a dense refit from those weights to the release's
  ``national_state`` surface, compiled from the candidate feed without the
  FSA family. It separates what re-solving on the current surface moves from
  what the FSA family moves;
- **B, with FSA**: the same refit with the 132 FSA targets added;
- **A + L2, B + L2**: both again with an L2 anchor to the published weights
  (``l2_lambda = 0.03``). The release default has none, and a dense refit of
  an L0-selected support without one concentrates weight, so these show how
  much of the movement the anchor holds back.

Each arm uses the release tool's own compile (``compile_us_fiscal_target_registry``
with ``student_loan_targets``), Medicaid substitutions, ``national_state``
surface, target materializer (``_materialize_target_frame``, the FSA rows
through ``fsa_student_loan_household_values``), loss weights and loss cap, and
``calibrate`` with the release's epochs, learning rate, weight cap and
conserved mass. It is not a release: it runs no release gate, uses a
candidate feed (the pinned ``51aa40f`` feed plus the merged chronicle#347 FSA
rows) and publishes nothing.

Outputs: the fit table per family and per FSA target, the borrower and
household composition by race before and after (and in the SCF), the weight
movement, every FSA fact that does not bind with its reason, and a schema-8
``calibration_diagnostics.json`` for arm B (left on the run volume; the
dashboard's staging page reads that format).

    python experiments/fit_1074_fsa_calibration.py --h5 populace_us_2024.h5 \\
        --feed candidate_feed.jsonl --scf-cache-dir dta
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ISSUE = "PolicyEngine/microcosm#1074"
HOST_RELEASE = "populace-us-2024-spm-20260915"
TARGET_SURFACE = "national_state"
ARMS = ("release", "a_no_fsa", "b_with_fsa", "a_no_fsa_l2", "b_with_fsa_l2")
#: The L2 anchor of the sensitivity arms: ``l2_lambda`` times
#: ``mean((w / w_release) ** 2)``, the configuration
#: ``docs/calibration-l2-basis.md`` studies. The release default is 0.
L2_SENSITIVITY_LAMBDA = 0.03
#: solved arm -> (with the FSA family, l2_lambda or None for the release's).
SOLVED_ARMS = {
    "a_no_fsa": (False, None),
    "b_with_fsa": (True, None),
    "a_no_fsa_l2": (False, L2_SENSITIVITY_LAMBDA),
    "b_with_fsa_l2": (True, L2_SENSITIVITY_LAMBDA),
}
ARM_LABELS = {
    "release": "Release",
    "a_no_fsa": "A, no FSA",
    "b_with_fsa": "B, with FSA",
    "a_no_fsa_l2": "A + L2",
    "b_with_fsa_l2": "B + L2",
}
#: The release's per-target loss cap (``US_FISCAL_TARGET_LOSS_CAP``): a miss
#: counts at most 100% of the target's scale, as the loss counts it.
ERROR_CAP = 1.0
_RACE_LABELS = {
    1: "white_non_hispanic",
    2: "black_non_hispanic",
    3: "hispanic",
    4: "other_or_multiple",
}


def _log(message: str) -> None:
    print(f"[{datetime.now(UTC).strftime('%H:%M:%S')}] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _load_release():
    path = REPO / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def engine_ready_frame(release, frame):
    """The host frame without the columns the locked engine computes itself.

    The published H5 was written on an older policyengine-us, so a few of its
    stored columns are formula-owned on the locked 2.2.1, and the release
    materializer refuses them. ``has_marketplace_health_coverage`` carries the
    reported Marketplace coverage the ACA enrollment target reads under its
    2.2.1 name, ``has_marketplace_health_coverage_at_interview``; it moves
    there, as the release's ACA source stage does (``_aca_source_person_table``).
    Every other formula-owned column is dropped, and both lists are recorded.
    """

    from microcosm.frame import Frame

    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    owned = sorted(
        release._formula_owned_gate_adapter()._engine_computed_columns(
            tables, period=release.PERIOD
        )
    )
    person = tables["person"]
    renamed = {}
    reported = release.US_ACA_REPORTED_MARKETPLACE_COVERAGE
    if "has_marketplace_health_coverage" in owned and reported not in person:
        person[reported] = person["has_marketplace_health_coverage"]
        renamed["has_marketplace_health_coverage"] = reported
    dropped = {}
    for entity, table in tables.items():
        present = [column for column in owned if column in table.columns]
        if present:
            tables[entity] = table.drop(columns=present)
            dropped[entity] = present
    ready = Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
    )
    return ready, {"formula_owned_dropped": dropped, "renamed": renamed}


def _read_facts(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _relative_error(estimate: float, target: float) -> float:
    return (estimate - target) / max(abs(target), 1.0)


def _family(spec) -> str:
    """The dashboard's family (provider/category), with the FSA rows by role."""

    if spec.family == "ed_fsa":
        return f"ed_fsa/{spec.metadata['target_role']}"
    hierarchy = spec.hierarchy
    if hierarchy is None:
        return f"{spec.family}/(no hierarchy)"
    return f"{hierarchy.provider.id}/{hierarchy.category.id}"


def _summary(errors: Sequence[float]) -> dict[str, float]:
    values = np.abs(np.asarray(errors, dtype=np.float64))
    capped = np.minimum(values, ERROR_CAP)
    return {
        "targets": len(values),
        "mean_capped_abs_rel_error": float(capped.mean()) if len(values) else math.nan,
        "mean_abs_rel_error": float(values.mean()) if len(values) else math.nan,
        "median_abs_rel_error": float(np.median(values)) if len(values) else math.nan,
        "max_abs_rel_error": float(values.max()) if len(values) else math.nan,
        "share_within_5pct": float((values <= 0.05).mean())
        if len(values)
        else math.nan,
        "share_within_10pct": float((values <= 0.10).mean())
        if len(values)
        else math.nan,
    }


def _person_weights(frame, household_weights: np.ndarray) -> np.ndarray:
    import pandas as pd

    household_ids = frame.table("household")["household_id"].to_numpy()
    by_household = pd.Series(np.asarray(household_weights), index=household_ids)
    return by_household.reindex(
        frame.table("person")["person_household_id"].to_numpy()
    ).to_numpy(dtype=np.float64)


def _person_balances(frame) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    person = frame.table("person")
    own = pd.to_numeric(person["federal_student_loan_balance"], errors="coerce")
    plus = pd.to_numeric(person["federal_parent_plus_loan_balance"], errors="coerce")
    return (
        own.fillna(0.0).to_numpy(dtype=np.float64),
        plus.fillna(0.0).to_numpy(dtype=np.float64),
    )


def race_composition(frame, household_weights: np.ndarray) -> dict[str, object]:
    """Borrowers by race (person and reference-person household), at weights."""

    from microcosm.build.us_runtime.scf_education_debt import _racecl4_from_cps

    person = frame.table("person")
    own, plus = _person_balances(frame)
    total = own + plus
    weights = _person_weights(frame, household_weights)
    race = _racecl4_from_cps(person)
    age = person["age"].to_numpy(dtype=np.float64)
    borrower = total > 0
    adult = age >= 18

    def shares(mask: np.ndarray, value: np.ndarray | None = None) -> dict[str, float]:
        base = weights * (value if value is not None else 1.0)
        denominator = float(base[mask].sum())
        return {
            label: float(base[mask & (race == code)].sum()) / denominator
            for code, label in _RACE_LABELS.items()
        }

    return {
        "borrowers": shares(borrower),
        "balance": shares(borrower, total),
        "adults": shares(adult),
        "borrower_share_of_adults_by_race": {
            label: float(weights[borrower & adult & (race == code)].sum())
            / float(weights[adult & (race == code)].sum())
            for code, label in _RACE_LABELS.items()
        },
    }


def borrower_support(frame) -> dict[str, object]:
    """Unweighted borrower records in each FSA age band: the carriers a band's
    count can be reweighted from (the stage gives balances to reference
    persons, spouses and co-resident adult children, so young borrowers living
    on their own are missed)."""

    own, plus = _person_balances(frame)
    borrower = (own + plus) > 0
    age = frame.table("person")["age"].to_numpy(dtype=np.float64)
    bands = (
        ("24_and_younger", -math.inf, 25.0),
        ("25_to_34", 25.0, 35.0),
        ("35_to_49", 35.0, 50.0),
        ("50_to_61", 50.0, 62.0),
        ("62_and_older", 62.0, math.inf),
    )
    return {
        "borrower_records": int(borrower.sum()),
        "people": int(len(age)),
        "borrower_records_by_age_band": {
            label: int((borrower & (age >= low) & (age < high)).sum())
            for label, low, high in bands
        },
        "people_by_age_band": {
            label: int(((age >= low) & (age < high)).sum())
            for label, low, high in bands
        },
    }


def household_race_composition(fit_table: Mapping[str, object]) -> dict[str, float]:
    """Borrower households by the reference person's race, from a fit table."""

    by_race = fit_table["by_race"]
    holders = {
        label: float(row["household_weight"])
        * float(row["share_with_federal_education_debt"])
        for label, row in by_race.items()
    }
    total = sum(holders.values())
    return {label: value / total for label, value in holders.items()}


def weight_movement(
    frame, initial: np.ndarray, final: np.ndarray, cap: float
) -> dict[str, object]:
    import pandas as pd

    ratio = final / initial
    own, plus = _person_balances(frame)
    household_ids = frame.table("household")["household_id"].to_numpy()
    debt = (
        pd.Series(own + plus, index=frame.table("person")["person_household_id"])
        .groupby(level=0)
        .sum()
        .reindex(household_ids)
        .fillna(0.0)
        .to_numpy()
    )
    holds = debt > 0

    def ess(weights: np.ndarray) -> float:
        return float(weights.sum() ** 2 / np.square(weights).sum())

    return {
        "ratio_quantiles": {
            str(q): float(np.quantile(ratio, q))
            for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
        },
        "share_at_cap": float((ratio >= cap * 0.999).mean()),
        "borrower_households_weighted_mean_ratio": float(
            final[holds].sum() / initial[holds].sum()
        ),
        "other_households_weighted_mean_ratio": float(
            final[~holds].sum() / initial[~holds].sum()
        ),
        "borrower_household_weight_share": {
            "initial": float(initial[holds].sum() / initial.sum()),
            "final": float(final[holds].sum() / final.sum()),
        },
        "effective_sample_size": {"initial": ess(initial), "final": ess(final)},
        "total_weight": {"initial": float(initial.sum()), "final": float(final.sum())},
    }


def repayment_plan_diagnostic(facts: Sequence[Mapping]) -> dict[str, object]:
    """FSA's Direct Loan portfolio by repayment plan, latest quarter (no model
    counterpart: the build carries no plan column)."""

    rows = [
        fact
        for fact in facts
        if str(fact.get("layout", {}).get("record_set_id", "")).startswith(
            "ed_fsa.dl_by_repayment_plan."
        )
    ]
    if not rows:
        return {"status": "absent"}
    latest = max(str(fact["period"]["value"]) for fact in rows)
    table: dict[str, dict[str, float]] = {}
    for fact in rows:
        if str(fact["period"]["value"]) != latest:
            continue
        layout = fact["layout"]
        entry = table.setdefault(
            layout["groupby_value_id"], {"label": layout["groupby_value_label"]}
        )
        entry[layout["measure_id"]] = float(fact["value"])
    dollars = sum(row.get("dollars_outstanding", 0.0) for row in table.values())
    for row in table.values():
        if "dollars_outstanding" in row and dollars:
            row["share_of_dollars"] = row["dollars_outstanding"] / dollars
    return {"status": "diagnostic_only", "period": latest, "plans": table}


def _fmt_pct(value: float, digits: int = 1) -> str:
    return "" if value != value else f"{100 * value:.{digits}f}%"


def _fmt_signed_pct(value: float) -> str:
    return "" if value != value else f"{100 * value:+.1f}%"


def _fmt_money(value: float) -> str:
    if value != value:
        return ""
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.1f}B"
    return f"${value:,.0f}"


def _fmt_ratio(value: float) -> str:
    return f"{value:.1e}" if abs(value) < 0.01 else f"{value:.2f}"


def _fmt_count(value: float) -> str:
    if value != value:
        return ""
    if abs(value) >= 1e6:
        return f"{value / 1e6:,.2f}M"
    return f"{value:,.0f}"


def markdown(receipt: Mapping[str, object]) -> str:
    lines = [
        "# microcosm#1074 receipts: Federal Student Aid portfolio targets",
        "",
        f"Generated {receipt['generated_at']} by "
        "`experiments/fit_1074_fsa_calibration.py`, commit "
        f"`{receipt['provenance']['commit']}`, Modal run "
        f"`{receipt['provenance']['run']}` (`experiments/modal_1074.py`). "
        "Machine-readable copy: `experiments/receipts/1074-fsa-calibration-fit.json`.",
        "",
        "Host: the published default "
        f"`{receipt['host']['release']}` (sha256 `{receipt['host']['sha256'][:12]}…`, "
        f"{receipt['host']['households']:,} households), with the SCF 2025 "
        "education-debt stage run at seed 0. Feed: the pinned "
        f"`51aa40f` US feed plus the merged chronicle#347 FSA rows "
        f"(`{receipt['feed']['candidate_sha256'][:12]}…`, "
        f"{receipt['feed']['rows']:,} rows). Arms: **release** = the published "
        "weights; **A** = a dense refit to the `national_state` surface without "
        "the FSA family; **B** = the same refit with it. Same epochs "
        f"({receipt['solver']['epochs']}), learning rate, 5x weight cap and "
        "conserved mass as a release. Not a release: no gate ran and nothing "
        "was published.",
        "",
        "## FSA targets: the reconciled levels",
        "",
    ]
    controls = receipt["fsa"]["controls"]
    lines += [
        "| | Balance | Borrowers |",
        "|---|---:|---:|",
    ]
    for key, label in (
        ("published_states_and_dc", "50 states and DC, as published"),
        ("published_out_of_scope", "Puerto Rico and Other (leave)"),
        ("published_not_reported", "Not Reported (allocated)"),
        ("published_total", "Location table total"),
        ("not_reported_allocation_factor", "Allocation factor"),
        ("in_scope_control", "In-scope control (every family sums to it)"),
    ):
        balance = controls["total_balance"][key]
        borrowers = controls["borrowers"][key]
        if key == "not_reported_allocation_factor":
            lines.append(f"| {label} | {balance:.4f} | {borrowers:.4f} |")
        else:
            lines.append(
                f"| {label} | {_fmt_money(balance)} | {_fmt_count(borrowers)} |"
            )
    lines += [
        "",
        f"Reference date: {receipt['fsa']['reference_label']} "
        f"(`{receipt['fsa']['reference_period']}`), the latest quarter all four "
        "bound tables publish; unaged at the 2024 build period "
        "(`period_alignment = stock_at_source_reference_date`).",
        "",
        "## Fit by FSA family",
        "",
        "Mean absolute relative error (share of targets within 5%).",
        "",
        "| FSA family | Targets | "
        + " | ".join(ARM_LABELS[arm] for arm in ARMS)
        + " |",
        "|---|---:|" + "---:|" * len(ARMS),
    ]
    for family, row in receipt["fit_by_family"].items():
        if not family.startswith("ed_fsa/"):
            continue
        cells = [
            f"{_fmt_pct(row[arm]['mean_abs_rel_error'])} "
            f"({_fmt_pct(row[arm]['share_within_5pct'], 0)})"
            for arm in ARMS
        ]
        lines.append(
            f"| {family.removeprefix('ed_fsa/')} | {row['release']['targets']} | "
            + " | ".join(cells)
            + " |"
        )
    lines += [
        "",
        "## Headline levels",
        "",
        "| | FSA target | " + " | ".join(ARM_LABELS[arm] for arm in ARMS) + " |",
        "|---|---:|" + "---:|" * len(ARMS),
    ]
    for row in receipt["headline"]:
        formatter = _fmt_count if row["kind"] == "count" else _fmt_money
        if row["kind"] == "share":
            formatter = _fmt_pct
        lines.append(
            f"| {row['label']} | {formatter(row['target'])} | "
            + " | ".join(formatter(row[arm]) for arm in ARMS)
            + " |"
        )
    lines += [
        "",
        "## Every other family: does adding FSA cost the rest of the surface?",
        "",
        "Mean absolute relative error over the `national_state` surface's "
        "non-FSA targets, by provider/category (the dashboard's family), each "
        "miss capped at 100% of its target as the loss caps it, sorted by how "
        "much adding the FSA family moves it:",
        "",
        "| Family | Targets | "
        + " | ".join(ARM_LABELS[arm] for arm in ARMS)
        + " | B - A (pp) | B+L2 - A+L2 (pp) |",
        "|---|---:|" + "---:|" * (len(ARMS) + 2),
    ]
    others = [
        (family, row)
        for family, row in receipt["fit_by_family"].items()
        if not family.startswith("ed_fsa/")
    ]
    metric = "mean_capped_abs_rel_error"

    def delta(row, with_arm: str, without_arm: str) -> float:
        return row[with_arm][metric] - row[without_arm][metric]

    others.sort(key=lambda item: -abs(delta(item[1], "b_with_fsa", "a_no_fsa")))
    for family, row in others:
        lines.append(
            f"| {family} | {row['release']['targets']} | "
            + " | ".join(_fmt_pct(row[arm][metric]) for arm in ARMS)
            + f" | {100 * delta(row, 'b_with_fsa', 'a_no_fsa'):+.2f}"
            + f" | {100 * delta(row, 'b_with_fsa_l2', 'a_no_fsa_l2'):+.2f} |"
        )
    totals = receipt["fit_overall"]["non_fsa"]
    lines += [
        f"| **all non-FSA targets** | {totals['release']['targets']} | "
        + " | ".join(_fmt_pct(totals[arm][metric]) for arm in ARMS)
        + f" | {100 * delta(totals, 'b_with_fsa', 'a_no_fsa'):+.2f}"
        + f" | {100 * delta(totals, 'b_with_fsa_l2', 'a_no_fsa_l2'):+.2f} |",
        "",
        "## Racial composition of borrowers, before and after calibration",
        "",
        "Borrower households by the reference person's race (the SCF's "
        "`racecl4`; the SCF column pools the five implicates):",
        "",
        "| | SCF 2025 | "
        + " | ".join(ARM_LABELS[arm] for arm in ARMS)
        + " | B - release (pp) |",
        "|---|---:|" + "---:|" * (len(ARMS) + 1),
    ]
    race = receipt["race"]
    for label in _RACE_LABELS.values():
        lines.append(
            f"| {label} | {_fmt_pct(race['households']['scf_2025'][label])} | "
            + " | ".join(_fmt_pct(race["households"][arm][label]) for arm in ARMS)
            + f" | {100 * (race['households']['b_with_fsa'][label] - race['households']['release'][label]):+.1f} |"
        )
    lines += [
        "",
        "Borrowers (people) by their own race, and the share of adults who borrow:",
        "",
        "| | "
        + " | ".join(ARM_LABELS[arm] for arm in ARMS)
        + " | Adults with a balance: release | B | B + L2 |",
        "|---|" + "---:|" * (len(ARMS) + 3),
    ]
    for label in _RACE_LABELS.values():
        lines.append(
            f"| {label} | "
            + " | ".join(
                _fmt_pct(race["people"][arm]["borrowers"][label]) for arm in ARMS
            )
            + f" | {_fmt_pct(race['people']['release']['borrower_share_of_adults_by_race'][label])}"
            + f" | {_fmt_pct(race['people']['b_with_fsa']['borrower_share_of_adults_by_race'][label])}"
            + f" | {_fmt_pct(race['people']['b_with_fsa_l2']['borrower_share_of_adults_by_race'][label])} |"
        )
    weights = receipt["weights"]
    lines += [
        "",
        "## Weight movement against the published weights",
        "",
        "| | " + " | ".join(ARM_LABELS[arm] for arm in SOLVED_ARMS) + " |",
        "|---|" + "---:|" * len(SOLVED_ARMS),
    ]
    rows_movement = (
        (
            "Borrower households' weight ratio",
            lambda m: f"{m['borrower_households_weighted_mean_ratio']:.2f}",
        ),
        (
            "Other households' weight ratio",
            lambda m: f"{m['other_households_weighted_mean_ratio']:.3f}",
        ),
        (
            "Borrower households' share of weight",
            lambda m: _fmt_pct(m["borrower_household_weight_share"]["final"]),
        ),
        (
            "Median household weight ratio",
            lambda m: _fmt_ratio(m["ratio_quantiles"]["0.5"]),
        ),
        (
            "99th percentile weight ratio",
            lambda m: _fmt_ratio(m["ratio_quantiles"]["0.99"]),
        ),
        (
            "Effective sample size (release "
            f"{weights['b_with_fsa']['effective_sample_size']['initial']:,.0f})",
            lambda m: f"{m['effective_sample_size']['final']:,.0f}",
        ),
    )
    for label, getter in rows_movement:
        lines.append(
            f"| {label} | "
            + " | ".join(getter(weights[arm]) for arm in SOLVED_ARMS)
            + " |"
        )
    lines += [
        "",
        "## FSA facts that do not bind",
        "",
        "| Disposition | Facts | Reason |",
        "|---|---:|---|",
    ]
    reasons: dict[tuple[str, str], int] = {}
    for item in receipt["fsa"]["dispositions"]:
        if item["disposition"] == "bound":
            continue
        key = (item["disposition"], item["reason"])
        reasons[key] = reasons.get(key, 0) + 1
    for (disposition, reason), count in sorted(reasons.items()):
        lines.append(f"| {disposition} | {count} | {reason} |")
    dropped = receipt["fsa"]["materialization_dropped"]
    lines += [
        "",
        f"Materialization dropped {len(dropped)} FSA targets"
        + (f": {', '.join(dropped)}." if dropped else "."),
        "",
        "## Every FSA target",
        "",
        "| Target | FSA (reconciled) | "
        + " | ".join(ARM_LABELS[arm] for arm in ARMS)
        + " |",
        "|---|---:|" + "---:|" * len(ARMS),
    ]
    for row in receipt["fsa_targets"]:
        formatter = _fmt_count if row["measure_mode"] == "indicator_sum" else _fmt_money
        lines.append(
            f"| {row['label']} | {formatter(row['target'])} | "
            + " | ".join(
                f"{formatter(row[arm])} ({_fmt_signed_pct(row[arm + '_error'])})"
                for arm in ARMS
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--feed", type=Path, required=True)
    parser.add_argument("--feed-provenance", type=Path)
    parser.add_argument("--scf-cache-dir", type=Path, required=True)
    parser.add_argument("--run", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--diagnostics-dir", type=Path)
    parser.add_argument(
        "--materialization-cache-dir",
        type=Path,
        help="Cache the JCT reform passes here, so a restarted run reuses them.",
    )
    parser.add_argument(
        "--maximum-microsim-batch-size",
        type=int,
        default=0,
        help=(
            "Households per engine batch; 0 (default) runs the whole support "
            "in one simulation, the batch-invariant reading."
        ),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO / "experiments" / "receipts" / "1074-fsa-calibration-fit.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=REPO / "experiments" / "1074-fsa-calibration-receipts.md",
    )
    args = parser.parse_args(argv)
    started = time.perf_counter()

    release = _load_release()
    from microcosm.build.us_runtime import compile_us_fiscal_target_registry
    from microcosm.build.us_runtime.fsa_student_loan_targets import (
        FSA_TARGET_FAMILY,
        fsa_student_loan_target_receipt,
    )
    from microcosm.build.us_runtime.scf_education_debt import (
        SCF_2025_FULL_EXTRACT,
        SCF_2025_SUMMARY_EXTRACT,
        fetch_scf_public_file,
        load_scf_2025_education_debt_donor,
        scf_education_debt_donor_fit_table,
        us_scf_education_debt_fit_table,
        us_scf_education_debt_signal_gate,
        with_us_scf_education_debt_inputs,
    )
    from microcosm.calibrate import (
        TargetRegistry,
        build_constraint_matrix,
        calibrate,
        write_calibration_diagnostics,
    )
    from microcosm.frame import Frame, WeightKind, Weights

    # The release's own defaults (epochs, learning rate, weight cap, aging,
    # batch size), read from its parser rather than restated here.
    cli = release._parse_args(
        [
            "--base-h5",
            str(args.h5),
            "--ledger-facts",
            str(args.feed),
            "--out",
            str(Path("/tmp") / "fit-1074-release-args"),
            "--no-staging",
        ]
    )
    epochs = int(args.epochs or cli.epochs)

    _log(f"reading feed {args.feed}")
    facts = _read_facts(args.feed)
    feed_sha = _sha256(args.feed)
    crosswalk_path = release.default_congressional_district_vintage_crosswalk_path()
    crosswalk = release.load_congressional_district_vintage_crosswalk(crosswalk_path)

    _log("compiling the target registry with and without the FSA family")
    registry = compile_us_fiscal_target_registry(
        facts,
        target_period=release.PERIOD,
        congressional_district_vintage_crosswalk=crosswalk,
        age_targets=cli.age_targets,
        allow_unaged_dollar_targets=cli.allow_unaged_dollar_targets,
        student_loan_targets=True,
    )
    without_flag = compile_us_fiscal_target_registry(
        facts,
        target_period=release.PERIOD,
        congressional_district_vintage_crosswalk=crosswalk,
        age_targets=cli.age_targets,
        allow_unaged_dollar_targets=cli.allow_unaged_dollar_targets,
    )
    non_fsa_names = [s.name for s in registry.specs if s.family != FSA_TARGET_FAMILY]
    if non_fsa_names != [s.name for s in without_flag.specs]:
        raise RuntimeError("The FSA flag changed the non-FSA surface.")
    registry, _ = release.apply_us_medicaid_enrollment_substitutions(registry)
    specs, surface_receipt = release._select_target_surface(
        registry.specs, TARGET_SURFACE
    )
    fsa_receipt = fsa_student_loan_target_receipt(facts, target_period=release.PERIOD)
    _log(f"{len(specs)} targets on the {TARGET_SURFACE} surface")

    _log("loading the host and running the SCF education-debt stage")
    summary_path = fetch_scf_public_file(SCF_2025_SUMMARY_EXTRACT, args.scf_cache_dir)
    full_path = fetch_scf_public_file(SCF_2025_FULL_EXTRACT, args.scf_cache_dir)
    donor = load_scf_2025_education_debt_donor(summary_path, full_path)
    host_sha = _sha256(args.h5)
    base = release._load_frame(args.h5)
    frame = with_us_scf_education_debt_inputs(
        base,
        seed=args.seed,
        time_period=release.PERIOD,
        scf_education_debt_donor=donor,
    )
    del base
    gate = us_scf_education_debt_signal_gate(frame)
    _log(f"stage gate passed={gate.passed}")
    frame, engine_adaptation = engine_ready_frame(release, frame)
    _log(f"engine adaptation: {engine_adaptation}")

    _log("materializing the target frame")
    batch_size = args.maximum_microsim_batch_size or None
    cache_context = None
    if args.materialization_cache_dir is not None:
        args.materialization_cache_dir.mkdir(parents=True, exist_ok=True)
        # Binds the reform caches to these exact staged inputs and engine.
        cache_context = {
            "experiment": ISSUE,
            "base_dataset_sha256": host_sha,
            "staged_frame_sha256": release._staged_frame_sha256(frame),
            "policyengine_us_version": release._package_or_workspace_version(
                "policyengine-us"
            ),
            "target_period": release.PERIOD,
            "maximum_microsim_batch_size": batch_size,
            # Not a release producer: the release binds its materializer
            # identity here, and declares None otherwise.
            "target_frame_materializer_identity_sha256": None,
        }
    target_frame, materialized, compilation = release._materialize_target_frame(
        frame,
        tuple(specs),
        maximum_microsim_batch_size=batch_size,
        target_materialization_cache_dir=args.materialization_cache_dir,
        target_materialization_cache_context=cache_context,
    )
    dropped = list(compilation.get("dropped_target_names") or [])
    _log(f"materialized {len(materialized)} targets; dropped {len(dropped)}")
    registry_b = materialized
    registry_a = TargetRegistry(
        tuple(s for s in materialized.specs if s.family != FSA_TARGET_FAMILY),
        country="us",
    )
    initial = np.asarray(target_frame.weights_for("household").values, dtype=np.float64)

    results = {}
    timings = {}
    for arm, (with_fsa, l2_lambda) in SOLVED_ARMS.items():
        arm_registry = registry_b if with_fsa else registry_a
        _log(f"calibrating arm {arm} ({len(arm_registry)} targets, {epochs} epochs)")
        t0 = time.perf_counter()
        results[arm] = calibrate(
            target_frame,
            arm_registry.to_target_set(),
            epochs=epochs,
            learning_rate=cli.learning_rate,
            max_weight_ratio=cli.max_weight_ratio,
            seed=args.seed,
            mass="conserve",
            l2_lambda=cli.l2_lambda if l2_lambda is None else l2_lambda,
            target_loss_weights=release._fiscal_target_loss_weights(
                arm_registry, cli.target_family_loss_multipliers
            ),
            target_loss_cap=release.US_FISCAL_TARGET_LOSS_CAP,
        )
        timings[arm] = time.perf_counter() - t0
        _log(
            f"arm {arm}: loss {results[arm].initial_loss:.5f} -> "
            f"{results[arm].final_loss:.5f} in {timings[arm]:.0f}s"
        )
    weights = {
        "release": initial,
        **{
            arm: np.asarray(results[arm].weights, dtype=np.float64)
            for arm in SOLVED_ARMS
        },
    }

    problem = build_constraint_matrix(target_frame, registry_b.to_target_set())
    estimates = {arm: problem.estimates(w) for arm, w in weights.items()}
    specs_by_row = {f"{spec.name}@{spec.period}": spec for spec in registry_b.specs}
    rows = []
    for index, name in enumerate(problem.names):
        spec = specs_by_row[name]
        target = float(problem.target_vector[index])
        row = {
            "name": spec.name,
            "family": _family(spec),
            "is_fsa": spec.family == FSA_TARGET_FAMILY,
            "target": target,
        }
        for arm in ARMS:
            value = float(estimates[arm][index])
            row[arm] = value
            row[f"{arm}_error"] = _relative_error(value, target)
        if row["is_fsa"]:
            row["label"] = spec.hierarchy.target.label
            row["measure_mode"] = spec.metadata["measure_mode"]
            row["target_role"] = spec.metadata["target_role"]
        rows.append(row)

    if args.diagnostics_dir is not None:
        # Every target's row, kept on the run volume beside the diagnostics.
        args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        (args.diagnostics_dir / "targets.json").write_text(
            json.dumps(rows, default=float), encoding="utf-8"
        )
    fit_by_family: dict[str, dict[str, object]] = {}
    for family in sorted({row["family"] for row in rows}):
        members = [row for row in rows if row["family"] == family]
        fit_by_family[family] = {
            arm: _summary([row[f"{arm}_error"] for row in members]) for arm in ARMS
        }
    fit_overall = {
        group: {
            arm: _summary(
                [row[f"{arm}_error"] for row in rows if row["is_fsa"] == is_fsa]
            )
            for arm in ARMS
        }
        for group, is_fsa in (("fsa", True), ("non_fsa", False))
    }

    _log("tabulating composition and fit tables")

    def with_weights(values: np.ndarray):
        return Frame(
            {entity: target_frame.table(entity) for entity in target_frame.entities},
            target_frame.schema,
            {"household": Weights(values, WeightKind.CALIBRATED)},
        )

    fit_tables = {
        arm: us_scf_education_debt_fit_table(with_weights(w))
        for arm, w in weights.items()
    }
    scf_fit = scf_education_debt_donor_fit_table(donor)
    race = {
        "households": {
            "scf_2025": household_race_composition(scf_fit),
            **{arm: household_race_composition(fit_tables[arm]) for arm in ARMS},
        },
        "people": {
            arm: race_composition(target_frame, w) for arm, w in weights.items()
        },
    }
    own, plus = _person_balances(target_frame)
    person_weights = {
        arm: _person_weights(target_frame, w) for arm, w in weights.items()
    }
    controls = fsa_receipt["controls"]
    pp_target = next(
        row["target"]
        for row in rows
        if row["is_fsa"] and row["target_role"] == "fsa_parent_plus_balance"
    )
    headline = [
        {
            "label": "Federal balance, own plus Parent PLUS",
            "kind": "amount",
            "target": controls["total_balance"]["in_scope_control"],
            **{arm: float(np.dot(own + plus, person_weights[arm])) for arm in ARMS},
        },
        {
            "label": "Borrowers (people with a balance)",
            "kind": "count",
            "target": controls["borrowers"]["in_scope_control"],
            **{arm: float(person_weights[arm][(own + plus) > 0].sum()) for arm in ARMS},
        },
        {
            "label": "Parent PLUS balance",
            "kind": "amount",
            "target": pp_target,
            **{arm: float(np.dot(plus, person_weights[arm])) for arm in ARMS},
        },
        {
            "label": "Parent PLUS share of the balance",
            "kind": "share",
            "target": pp_target / controls["total_balance"]["in_scope_control"],
            **{
                arm: float(np.dot(plus, person_weights[arm]))
                / float(np.dot(own + plus, person_weights[arm]))
                for arm in ARMS
            },
        },
        {
            "label": "Mean balance per borrower",
            "kind": "amount",
            "target": controls["total_balance"]["in_scope_control"]
            / controls["borrowers"]["in_scope_control"],
            **{
                arm: float(np.dot(own + plus, person_weights[arm]))
                / float(person_weights[arm][(own + plus) > 0].sum())
                for arm in ARMS
            },
        },
        {
            "label": "Households with federal education debt",
            "kind": "share",
            "target": math.nan,
            **{
                arm: float(
                    fit_tables[arm]["households"]["share_with_federal_education_debt"]
                )
                for arm in ARMS
            },
        },
    ]

    if args.diagnostics_dir is not None:
        args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        try:
            outcome = write_calibration_diagnostics(
                results["b_with_fsa"],
                args.diagnostics_dir / "calibration_diagnostics.json",
                target_registry=registry_b,
                target_surface=release.target_surface_payload(results["b_with_fsa"]),
                build={
                    "base_dataset_sha256": host_sha,
                    "experiment": ISSUE,
                    "candidate_feed_sha256": feed_sha,
                    "target_compilation": compilation,
                },
            )
            _log(f"diagnostics written: {outcome}")
        except Exception as error:  # noqa: BLE001 - a receipt still ships
            _log(f"diagnostics not written: {error!r}")

    fsa_dropped = sorted(
        name for name in dropped if name.startswith(f"{FSA_TARGET_FAMILY}.")
    )
    receipt = {
        "schema_version": 1,
        "issue": ISSUE,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%d"),
        "provenance": {"commit": _git_commit(), "run": args.run},
        "host": {
            "release": HOST_RELEASE,
            "sha256": host_sha,
            "households": int(target_frame.n("household")),
            "people": int(target_frame.n("person")),
            "scf_stage_seed": args.seed,
            "scf_stage_gate_passed": bool(gate.passed),
            "engine_adaptation": engine_adaptation,
        },
        "feed": {
            "candidate_sha256": feed_sha,
            "rows": len(facts),
            "provenance": (
                json.loads(args.feed_provenance.read_text())
                if args.feed_provenance is not None
                else None
            ),
        },
        "solver": {
            "kernel": "microcosm.calibrate.calibrate (dense refit, no L0)",
            "epochs": epochs,
            "learning_rate": cli.learning_rate,
            "max_weight_ratio": cli.max_weight_ratio,
            "l2_lambda": cli.l2_lambda,
            "mass": "conserve",
            "target_surface": TARGET_SURFACE,
            "age_targets": cli.age_targets,
            "loss": {
                arm: {
                    "initial": float(results[arm].initial_loss),
                    "final": float(results[arm].final_loss),
                    "seconds": timings[arm],
                }
                for arm in results
            },
            "arms": {
                arm: {
                    "with_fsa": with_fsa,
                    "targets": len(registry_b if with_fsa else registry_a),
                    "l2_lambda": cli.l2_lambda if l2_lambda is None else l2_lambda,
                }
                for arm, (with_fsa, l2_lambda) in SOLVED_ARMS.items()
            },
            "surface": surface_receipt,
        },
        "fsa": {
            **{
                key: fsa_receipt[key]
                for key in (
                    "status",
                    "reference_period",
                    "reference_label",
                    "period_alignment",
                    "controls",
                    "disposition_counts",
                    "dispositions",
                )
            },
            "materialization_dropped": fsa_dropped,
            "repayment_plan_diagnostic": repayment_plan_diagnostic(facts),
        },
        "headline": headline,
        "fit_by_family": fit_by_family,
        "fit_overall": fit_overall,
        "fsa_targets": [row for row in rows if row["is_fsa"]],
        "race": race,
        "support": borrower_support(target_frame),
        "scf_fit_tables": {"scf_2025": scf_fit, **fit_tables},
        "weights": {
            arm: weight_movement(
                target_frame, initial, weights[arm], cli.max_weight_ratio
            )
            for arm in SOLVED_ARMS
        },
        "seconds": time.perf_counter() - started,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(receipt, indent=2, default=float, allow_nan=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.out_md.write_text(markdown(receipt), encoding="utf-8", newline="\n")
    _log(f"wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
