#!/usr/bin/env python
"""Fit Federal Student Aid portfolio targets on the published US default (#1074).

What it measures. The published default ``populace-us-2024-spm-20260915`` with
the SCF 2025 education-debt stage run on it (seed 0, as microcosm#1073's host
receipt), calibrated on its own 57,240-household support:

- **release**: the published weights, which never saw the student loan
  columns (the "before");
- **A**: a dense refit from those weights to the release's ``national_state``
  surface, compiled from the feed without the FSA family. It separates what
  re-solving on the current surface moves from what the FSA family moves;
- **B**: the same refit with the 130 FSA targets added;
- **A and B + record L2** (``l2_lambda = 0.03``, ``l2_basis = "record"``):
  ``calibrate``'s default concentration penalty, ``mean((w / w_release) ** 2)``,
  which spreads weight without anchoring to the design (``solve.py``). It is
  the penalty the release's ``--l2-lambda`` passes today;
- **A and B + chi-square L2** (``l2_lambda = 0.03``, ``l2_basis =
  "chi_square"``): the anchor-weighted chi-square distance from the published
  weights, the setting the ACS local L2 study recommends on its own loss. The
  release has no flag for this basis. ``--extra-chi-square-lambda`` adds A and
  B at another lambda, for when 0.03 barely moves the fit;
- **B, counts x0** (at lambda 0 and with record L2): B with the 65 FSA balance
  targets at full loss weight and the 65 borrower-count targets at none of
  theirs, renormalized to mean 1, so the counts only validate. No release flag
  reproduces it: ``--target-family-loss-multiplier ed_fsa=`` scales balances
  and counts together.

The release's L2 setting depends on these arms' FSA fit, non-FSA fit,
effective sample size and income tables, which the receipt reports for each.

Each arm uses the release tool's own compile (``compile_us_fiscal_target_registry``
with ``student_loan_targets``), Medicaid substitutions, ``national_state``
surface, target materializer (``_materialize_target_frame``, the FSA rows
through ``fsa_student_loan_household_values``), loss weights and loss cap, and
``calibrate`` with the release's epochs, learning rate, weight cap and
conserved mass. It is not a release: it runs no release gate, reads the feed
``--feed`` names (``experiments/modal_1074.py`` builds the US feed the branch
pins, or the earlier candidate splice; ``--feed-provenance`` says which) and
publishes nothing. These arms re-solve the published 57,240 households, while a
Route A build solves the whole base.

Outputs: the fit table per family and per FSA target, the share of households
with debt by household income quintile (each arm's own cut points, and the
release arm's applied to every arm), the borrower and household composition by
race before and after (and in the SCF), the weight movement, every FSA fact
that does not bind with its reason, and a schema-8
``calibration_diagnostics.json`` for arm B (left on the run volume; the
dashboard's staging page reads that format). The JSON receipt writes a missing
value as ``null``.

    python experiments/fit_1074_fsa_calibration.py --h5 populace_us_2024.h5 \\
        --feed consumer_facts.jsonl --scf-cache-dir dta
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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
ISSUE = "PolicyEngine/microcosm#1074"
HOST_RELEASE = "populace-us-2024-spm-20260915"
TARGET_SURFACE = "national_state"
RELEASE_ARM = "release"
#: The concentration penalty of the L2 arms, in units of the loss: the setting
#: the ACS local L2 study recommends on its own loss
#: (``experiments/us-acs-local-l2-basis-20260928/README.md``). The release
#: default is 0.
L2_SENSITIVITY_LAMBDA = 0.03
L2_BASIS_RECORD = "record"
L2_BASIS_CHI_SQUARE = "chi_square"


@dataclass(frozen=True)
class Arm:
    """One solved arm: what it binds and how ``calibrate`` penalizes it.

    ``l2_lambda`` None takes the release's ``--l2-lambda`` default (0), where
    the basis does nothing. ``count_weight`` is the fraction of its loss weight
    each FSA borrower-count target keeps, before the vector is renormalized to
    mean 1, as a family multiplier is; at 0 the counts only validate.
    """

    label: str
    with_fsa: bool
    l2_lambda: float | None = None
    l2_basis: str = L2_BASIS_RECORD
    count_weight: float = 1.0


#: The grid the next receipt runs (the #1074 review of 2026-11-05): A and B at
#: lambda 0, record 0.03 and chi-square 0.03, and B with the counts at x0, at
#: lambda 0 and record 0.03. The release arm, the published weights, comes
#: first in every table.
SOLVED_ARMS: dict[str, Arm] = {
    "a_no_fsa": Arm("A", with_fsa=False),
    "b_with_fsa": Arm("B", with_fsa=True),
    "a_no_fsa_l2": Arm("A + record L2", False, L2_SENSITIVITY_LAMBDA),
    "b_with_fsa_l2": Arm("B + record L2", True, L2_SENSITIVITY_LAMBDA),
    "a_no_fsa_chi2": Arm(
        "A + chi-square L2", False, L2_SENSITIVITY_LAMBDA, L2_BASIS_CHI_SQUARE
    ),
    "b_with_fsa_chi2": Arm(
        "B + chi-square L2", True, L2_SENSITIVITY_LAMBDA, L2_BASIS_CHI_SQUARE
    ),
    "b_counts_w0": Arm("B, counts x0", True, count_weight=0.0),
    "b_counts_w0_l2": Arm(
        "B + record L2, counts x0", True, L2_SENSITIVITY_LAMBDA, count_weight=0.0
    ),
}


def arm_grid(extra_chi_square_lambdas: Sequence[float] = ()) -> dict[str, Arm]:
    """:data:`SOLVED_ARMS`, plus A and B at each extra chi-square lambda."""

    grid = dict(SOLVED_ARMS)
    for value in extra_chi_square_lambdas:
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"A chi-square lambda must be positive, got {value!r}.")
        tag = f"{value:g}".replace(".", "p")
        for prefix, label, with_fsa in (
            ("a_no_fsa", "A", False),
            ("b_with_fsa", "B", True),
        ):
            name = f"{prefix}_chi2_{tag}"
            if name in grid or value == L2_SENSITIVITY_LAMBDA:
                raise ValueError(f"Chi-square lambda {value:g} is already in the grid.")
            grid[name] = Arm(
                f"{label} + chi-square L2 {value:g}",
                with_fsa,
                value,
                L2_BASIS_CHI_SQUARE,
            )
    return grid


def arm_pairs(grid: Mapping[str, Arm]) -> list[tuple[str, str]]:
    """(with FSA, without) pairs at one penalty and full count weight: what
    adding the FSA family costs the rest of the surface."""

    pairs = []
    for name, arm in grid.items():
        if not arm.with_fsa or arm.count_weight != 1.0:
            continue
        for other, control in grid.items():
            if (
                not control.with_fsa
                and control.l2_lambda == arm.l2_lambda
                and control.l2_basis == arm.l2_basis
            ):
                pairs.append((name, other))
    return pairs


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


def _is_borrower_count_role(role: str) -> bool:
    """FSA's state, age-band and debt-size-band borrower counts."""

    return role.endswith("_borrowers")


def scaled_borrower_count_loss_weights(
    specs: Sequence, loss_weights: np.ndarray, fraction: float
) -> np.ndarray:
    """``loss_weights`` with each FSA borrower count's weight times ``fraction``.

    Every other target keeps its weight, then the vector is renormalized to
    mean 1, as a family multiplier is. A ``fraction`` of 0 leaves the counts
    in the target set as validation only.
    """

    mask = np.asarray(
        [
            spec.family == "ed_fsa"
            and _is_borrower_count_role(str(spec.metadata["target_role"]))
            for spec in specs
        ],
        dtype=bool,
    )
    if not mask.any():
        raise RuntimeError("No FSA borrower-count target to down-weight.")
    scaled = np.asarray(loss_weights, dtype=np.float64).copy()
    scaled[mask] *= fraction
    return scaled / scaled.mean()


def arm_comparison(
    rows: Sequence[Mapping[str, object]],
    headline: Sequence[Mapping[str, object]],
    fit_overall: Mapping[str, Mapping[str, Mapping[str, float]]],
    weights: Mapping[str, Mapping[str, object]],
    arms: Sequence[str],
) -> dict[str, object]:
    """Per arm: the FSA fit by balance and count, the rest of the surface, the
    borrower levels, the Parent PLUS mix, the borrower age profile and the
    weight concentration (the release's are the published weights')."""

    fsa = [row for row in rows if row["is_fsa"]]
    balances = [row for row in fsa if not _is_borrower_count_role(row["target_role"])]
    counts = [row for row in fsa if _is_borrower_count_role(row["target_role"])]
    ages = [row for row in fsa if row["target_role"] == "fsa_age_band_borrowers"]
    by_label = {row["label"]: row for row in headline}
    fsa_age_total = sum(row["target"] for row in ages)
    result: dict[str, object] = {
        "targets": {
            "balance": len(balances),
            "borrower_count": len(counts),
            "age_bands": len(ages),
        },
        "non_fsa_targets": sum(1 for row in rows if not row["is_fsa"]),
        "fsa_age_distribution": {
            row["label"]: row["target"] / fsa_age_total for row in ages
        },
        "arms": {},
    }
    any_solved = next(iter(weights.values()))
    for arm in arms:
        age_total = sum(row[arm] for row in ages)
        age_shares = {row["label"]: row[arm] / age_total for row in ages}
        movement = weights.get(arm)
        result["arms"][arm] = {
            "balance_targets": _summary([row[f"{arm}_error"] for row in balances]),
            "borrower_count_targets": _summary([row[f"{arm}_error"] for row in counts]),
            "fsa_targets": fit_overall["fsa"][arm],
            "non_fsa_targets": fit_overall["non_fsa"][arm],
            "non_fsa_mean_capped_abs_rel_error": fit_overall["non_fsa"][arm][
                "mean_capped_abs_rel_error"
            ],
            "households_with_federal_education_debt": by_label[
                "Households with federal education debt"
            ][arm],
            "borrowers": by_label["Borrowers (people with a balance)"][arm],
            "mean_balance_per_borrower": by_label["Mean balance per borrower"][arm],
            "parent_plus_share": by_label["Parent PLUS share of the balance"][arm],
            "borrower_age_distribution": age_shares,
            "borrower_age_tvd_vs_fsa": 0.5
            * sum(
                abs(age_shares[label] - share)
                for label, share in result["fsa_age_distribution"].items()
            ),
            "effective_sample_size": (
                movement["effective_sample_size"]["final"]
                if movement is not None
                else any_solved["effective_sample_size"]["initial"]
            ),
            "median_household_weight_ratio": (
                movement["ratio_quantiles"]["0.5"] if movement is not None else 1.0
            ),
        }
    return result


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


def parent_plus_diagnostic(
    facts: Sequence[Mapping], fsa_receipt: Mapping[str, object]
) -> dict[str, object]:
    """FSA's Parent PLUS row at the reference quarter, as a diagnostic fact.

    The loan-type table binds no target (microcosm#1525), so FSA's Parent PLUS
    figure comes from its dollar rows at the reference quarter, which the plan
    marks diagnostic: the Parent PLUS share of their sum, and that share of the
    in-scope control. Under 34 CFR 685.220(f) the Secretary pays off each loan
    a Direct Consolidation Loan consolidates, so FSA likely files a
    consolidated Parent PLUS loan under Consolidation, and the share is likely a
    lower bound on the stage's column (an inference; FSA's notes don't say).
    """

    reference = fsa_receipt["reference_period"]
    ids = {
        item["source_record_id"]
        for item in fsa_receipt["dispositions"]
        if item["table"] == "portfolio_by_loan_type"
        and item["measure"] == "dollars_outstanding"
        and item["period"] == reference
        and item["disposition"] == "diagnostic"
    }
    rows = [fact for fact in facts if fact["lineage"]["source_record_id"] in ids]
    by_type = {
        fact["layout"]["groupby_value_id"]: float(fact["value"]) for fact in rows
    }
    if not ids or len(rows) != len(ids) or len(by_type) != len(rows):
        raise RuntimeError(
            f"FSA's loan-type dollar rows at {reference} are missing or repeated."
        )
    if "parent_plus" not in by_type:
        raise RuntimeError(
            f"FSA's loan-type table has no Parent PLUS row at {reference}."
        )
    partition = math.fsum(by_type.values())
    share = by_type["parent_plus"] / partition
    control = fsa_receipt["controls"]["total_balance"]["in_scope_control"]
    return {
        "status": "diagnostic",
        "reference_period": reference,
        "published_by_loan_type": dict(sorted(by_type.items())),
        "published_loan_type_total": partition,
        "parent_plus_share": share,
        "parent_plus_share_of_in_scope_control": share * control,
        "reading": (
            "likely a lower bound: FSA likely files a consolidated Parent PLUS "
            "loan under Consolidation (34 CFR 685.220(f)); FSA's notes don't say"
        ),
    }


_QUINTILES = ("1", "2", "3", "4", "5")


def _quintile_summary(groups: Mapping[str, Mapping[str, float]]) -> dict[str, object]:
    """Debt shares, weight shares, Q5 - Q1 and the bottom two quintiles' share
    of borrower households, from five ``{household_weight, share}`` groups."""

    shares = {q: groups[q]["share_with_federal_education_debt"] for q in _QUINTILES}
    weight = {q: groups[q]["household_weight"] for q in _QUINTILES}
    holders = {q: weight[q] * shares[q] for q in _QUINTILES}
    return {
        "share_with_federal_education_debt": shares,
        "share_of_household_weight": {
            q: weight[q] / sum(weight.values()) for q in _QUINTILES
        },
        "q5_minus_q1": shares["5"] - shares["1"],
        "bottom_two_share_of_borrower_households": (holders["1"] + holders["2"])
        / sum(holders.values()),
    }


def _household_debt(frame) -> np.ndarray:
    """Each household's federal balance (own plus Parent PLUS), in table order."""

    import pandas as pd

    own, plus = _person_balances(frame)
    return (
        pd.Series(own + plus, index=frame.table("person")["person_household_id"])
        .groupby(level=0)
        .sum()
        .reindex(frame.table("household")["household_id"].to_numpy())
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def income_quintile_tables(
    frame,
    weights: Mapping[str, np.ndarray],
    fit_tables: Mapping[str, Mapping[str, object]],
    scf_fit: Mapping[str, object],
) -> dict[str, object]:
    """Households with federal education debt by household income quintile.

    ``own_cut_points``: each arm's own weighted quintiles (its fit table's
    ``by_income_quintile``), and the SCF's. ``release_cut_points``: every
    household keeps the quintile it has at the release's weights, which applies
    the release arm's cut points to every arm (a tie at a cut keeps the
    release's split), so a column compares the same households across arms.
    The income is the one the imputation conditions on, over the SCF primary
    economic unit. The student debt pilot cuts tax-unit AGI over all tax units
    and counts borrowers as people, so these rows are not its income tables.
    """

    from microcosm.build.us_runtime.scf_education_debt import (
        _income_quintiles,
        _recipients,
    )

    income = _recipients(frame).features["household_income"].to_numpy(np.float64)
    release = np.asarray(weights[RELEASE_ARM], dtype=np.float64)
    fixed = _income_quintiles(income, release).astype(str)
    holds = _household_debt(frame) > 0
    by_fixed = {}
    for arm, values in weights.items():
        w = np.asarray(values, dtype=np.float64)
        by_fixed[arm] = _quintile_summary(
            {
                q: {
                    "household_weight": float(w[fixed == q].sum()),
                    "share_with_federal_education_debt": float(
                        w[(fixed == q) & holds].sum() / w[fixed == q].sum()
                    ),
                }
                for q in _QUINTILES
            }
        )
    own = {
        arm: _quintile_summary(table["by_income_quintile"])
        for arm, table in {"scf_2025": scf_fit, **fit_tables}.items()
    }
    release_shares = own[RELEASE_ARM]["share_with_federal_education_debt"]
    fixed_shares = by_fixed[RELEASE_ARM]["share_with_federal_education_debt"]
    return {
        "income": "household_income of the SCF primary economic unit",
        "own_cut_points": own,
        "release_cut_points": by_fixed,
        "release_income_range_by_quintile": {
            q: [float(income[fixed == q].min()), float(income[fixed == q].max())]
            for q in _QUINTILES
        },
        "release_fixed_cuts_reproduce_own": bool(
            np.allclose(
                [release_shares[q] for q in _QUINTILES],
                [fixed_shares[q] for q in _QUINTILES],
            )
        ),
    }


def json_ready(value: object) -> object:
    """``value`` as strict JSON: a NaN or infinite float becomes None (``null``)
    and a numpy scalar or array its Python value, at any depth."""

    if isinstance(value, Mapping):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _missing(value: float | None) -> bool:
    """A value the JSON receipt writes as ``null`` (None or not a number)."""

    return value is None or value != value


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    return "" if _missing(value) else f"{100 * value:.{digits}f}%"


def _fmt_signed_pct(value: float | None) -> str:
    return "" if _missing(value) else f"{100 * value:+.1f}%"


def _fmt_points(value: float | None) -> str:
    return "" if _missing(value) else f"{100 * value:+.1f}"


def _fmt_money(value: float | None) -> str:
    if _missing(value):
        return ""
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.1f}B"
    return f"${value:,.0f}"


def _fmt_ratio(value: float | None) -> str:
    if _missing(value):
        return ""
    return f"{value:.1e}" if abs(value) < 0.01 else f"{value:.2f}"


def _fmt_count(value: float | None) -> str:
    if _missing(value):
        return ""
    if abs(value) >= 1e6:
        return f"{value / 1e6:,.2f}M"
    return f"{value:,.0f}"


def _arms(receipt: Mapping[str, object], *, solved: bool = False) -> list[str]:
    """The receipt's arms in table order, the release first (or only the solved)."""

    return [
        name
        for name, arm in receipt["arms"].items()
        if not solved or name != RELEASE_ARM
    ]


def _labels(receipt: Mapping[str, object], arms: Sequence[str]) -> str:
    return " | ".join(receipt["arms"][arm]["label"] for arm in arms)


def arm_summary_markdown(receipt: Mapping[str, object]) -> list[str]:
    """Every arm's FSA fit, non-FSA fit, levels, age profile and concentration."""

    comparison = receipt["arm_comparison"]
    arms = comparison["arms"]
    targets = comparison["targets"]
    controls = receipt["fsa"]["controls"]
    scf_share = receipt["scf_fit_tables"]["scf_2025"]["households"][
        "share_with_federal_education_debt"
    ]
    shown = _arms(receipt)
    lines = [
        "",
        "## Every arm at a glance",
        "",
        "The fit, the levels and the weight concentration the release's L2 "
        "setting depends on. Reference: FSA, except the household share, which "
        "is the SCF's; FSA's Parent PLUS share is diagnostic (no target binds "
        "it). The x0 arms keep the "
        f"{targets['borrower_count']} borrower-count targets at none of their "
        "loss weight, so those counts only validate.",
        "",
        "| | Reference | " + _labels(receipt, shown) + " |",
        "|---|---:|" + "---:|" * len(shown),
    ]
    rows = (
        (
            f"All FSA targets within 5% ({targets['balance'] + targets['borrower_count']})",
            "",
            lambda m: _fmt_pct(m["fsa_targets"]["share_within_5pct"], 0),
        ),
        (
            f"FSA balance targets within 5% ({targets['balance']})",
            "",
            lambda m: _fmt_pct(m["balance_targets"]["share_within_5pct"], 0),
        ),
        (
            f"FSA borrower-count targets within 5% ({targets['borrower_count']})",
            "",
            lambda m: _fmt_pct(m["borrower_count_targets"]["share_within_5pct"], 0),
        ),
        (
            "FSA balance targets: mean error",
            "",
            lambda m: _fmt_pct(m["balance_targets"]["mean_abs_rel_error"]),
        ),
        (
            "FSA borrower-count targets: mean error",
            "",
            lambda m: _fmt_pct(m["borrower_count_targets"]["mean_abs_rel_error"]),
        ),
        (
            "Non-FSA capped error",
            "",
            lambda m: _fmt_pct(m["non_fsa_mean_capped_abs_rel_error"], 2),
        ),
        (
            f"Non-FSA targets within 5% ({comparison['non_fsa_targets']:,})",
            "",
            lambda m: _fmt_pct(m["non_fsa_targets"]["share_within_5pct"]),
        ),
        (
            "Households with federal education debt",
            _fmt_pct(scf_share),
            lambda m: _fmt_pct(m["households_with_federal_education_debt"]),
        ),
        (
            "Borrowers",
            _fmt_count(controls["borrowers"]["in_scope_control"]),
            lambda m: _fmt_count(m["borrowers"]),
        ),
        (
            "Mean balance per borrower",
            _fmt_money(
                controls["total_balance"]["in_scope_control"]
                / controls["borrowers"]["in_scope_control"]
            ),
            lambda m: _fmt_money(m["mean_balance_per_borrower"]),
        ),
        (
            "Parent PLUS share of the balance (FSA: diagnostic)",
            _fmt_pct(receipt["fsa"]["parent_plus"]["parent_plus_share"]),
            lambda m: _fmt_pct(m["parent_plus_share"]),
        ),
        (
            "Effective sample size",
            "",
            lambda m: f"{m['effective_sample_size']:,.0f}",
        ),
        (
            "Median household's share of its published weight",
            "",
            lambda m: _fmt_pct(m["median_household_weight_ratio"], 2),
        ),
    )
    for label, reference, getter in rows:
        lines.append(
            f"| {label} | {reference} | "
            + " | ".join(getter(arms[arm]) for arm in shown)
            + " |"
        )
    for band, share in comparison["fsa_age_distribution"].items():
        short = band.removeprefix("Federal student loan borrowers, borrower ")
        lines.append(
            f"| Borrowers {short} | {_fmt_pct(share)} | "
            + " | ".join(
                _fmt_pct(arms[arm]["borrower_age_distribution"][band]) for arm in shown
            )
            + " |"
        )
    lines.append(
        "| Borrower age profile: distance from FSA (TVD) | | "
        + " | ".join(f"{arms[arm]['borrower_age_tvd_vs_fsa']:.3f}" for arm in shown)
        + " |"
    )
    return lines


def _feed_sentence(receipt: Mapping[str, object]) -> str:
    """Which feed the arms compiled, from the runner's provenance."""

    feed = receipt["feed"]
    provenance = feed.get("provenance") or {}
    digest = f"`{feed['sha256'][:12]}…`, {feed['rows']:,} rows"
    if provenance.get("kind") == "committed pin":
        pin = provenance["pin"]
        return (
            f"the US feed the branch pins, rebuilt at Chronicle "
            f"`{pin['source_commit'][:8]}` ({digest}, as pinned), which "
            "carries the FSA rows."
        )
    if "pinned_feed" in provenance:
        return (
            f"a candidate, the `{provenance['pinned_feed']['commit'][:7]}` US "
            "feed plus the chronicle#347 FSA rows built at "
            f"`{provenance['fsa_packages']['commit'][:7]}` ({digest}); no pin "
            "names it."
        )
    return f"`--feed` ({digest})."


def markdown(receipt: Mapping[str, object]) -> str:
    arms = _arms(receipt)
    solved = _arms(receipt, solved=True)
    pairs = [tuple(pair) for pair in receipt["arm_pairs"]]
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
        f"education-debt stage run at seed 0. Feed: {_feed_sentence(receipt)} "
        "Arms: **release** = the published weights; **A** = a dense refit to "
        "the `national_state` surface without the FSA family; **B** = the same "
        "refit with it; **+ record L2** and **+ chi-square L2** = A and B with "
        "a concentration penalty at the stated `l2_lambda`, on `calibrate`'s "
        "default `record` basis (the mean squared weight ratio, which spreads "
        "weight without anchoring to the design; the only basis the release's "
        "`--l2-lambda` passes) or its `chi_square` basis (the anchor-weighted "
        "distance from the published weights); **counts x0** = B with the "
        "balance targets at full loss weight and the borrower counts at none. "
        f"Same epochs ({receipt['solver']['epochs']}), learning rate, 5x weight "
        "cap and conserved mass as a release, on the published support rather "
        "than the whole base a Route A build solves. Not a release: no gate "
        "ran and nothing was published.",
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
        "tables publish. The targets bind that stock unaged "
        "(`period_alignment = stock_at_source_reference_date`), and the build "
        "writes the calibrated balances at its period, 2024, since the H5 "
        "records one period for every column. policyengine-us#10415 (merged "
        "2026-10-30) holds the balances at their stored value in every year, so "
        "a 2024 or a 2026 read returns each person's March 2026 stock. "
        "policyengine-us also raises household weights by population growth, "
        "x1.0165 from 2024 to 2026 at the student debt pilot's pin, so a 2026 "
        "read of these weights shows every level here 1.65% higher. The age "
        "bands group the stored, unaged `age`.",
        "",
        f"FSA's Parent PLUS row ({_fmt_money(receipt['fsa']['parent_plus']['published_by_loan_type']['parent_plus'])}, "
        f"{_fmt_pct(receipt['fsa']['parent_plus']['parent_plus_share'])} of the "
        "loan-type table's dollar rows) is diagnostic: 34 CFR 685.220(f) has "
        "the Secretary pay off each loan a Direct Consolidation Loan "
        "consolidates, so FSA likely files a consolidated Parent PLUS loan under "
        "Consolidation and the share is likely a lower bound on the stage's "
        "column. FSA's notes don't say.",
        "",
        *arm_summary_markdown(receipt),
        "",
        "## Fit by FSA family",
        "",
        "Mean absolute relative error (share of targets within 5%).",
        "",
        "| FSA family | Targets | " + _labels(receipt, arms) + " |",
        "|---|---:|" + "---:|" * len(arms),
    ]
    for family, row in receipt["fit_by_family"].items():
        if not family.startswith("ed_fsa/"):
            continue
        cells = [
            f"{_fmt_pct(row[arm]['mean_abs_rel_error'])} "
            f"({_fmt_pct(row[arm]['share_within_5pct'], 0)})"
            for arm in arms
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
        "Reference: FSA's in-scope control; the Parent PLUS rows are FSA's "
        "diagnostic loan-type row, and the household share has no FSA figure.",
        "",
        "| | FSA | " + _labels(receipt, arms) + " |",
        "|---|---:|" + "---:|" * len(arms),
    ]
    for row in receipt["headline"]:
        formatter = _fmt_count if row["kind"] == "count" else _fmt_money
        if row["kind"] == "share":
            formatter = _fmt_pct
        lines.append(
            f"| {row['label']} | {formatter(row['target'])} | "
            + " | ".join(formatter(row[arm]) for arm in arms)
            + " |"
        )
    lines += income_markdown(receipt)
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
        + _labels(receipt, arms)
        + "".join(
            f" | {receipt['arms'][b]['label']} - {receipt['arms'][a]['label']} (pp)"
            for b, a in pairs
        )
        + " |",
        "|---|---:|" + "---:|" * (len(arms) + len(pairs)),
    ]
    others = [
        (family, row)
        for family, row in receipt["fit_by_family"].items()
        if not family.startswith("ed_fsa/")
    ]
    metric = "mean_capped_abs_rel_error"

    def delta(row, with_arm: str, without_arm: str) -> float:
        return row[with_arm][metric] - row[without_arm][metric]

    def deltas(row) -> str:
        return "".join(f" | {100 * delta(row, b, a):+.2f}" for b, a in pairs)

    others.sort(key=lambda item: -abs(delta(item[1], *pairs[0])))
    for family, row in others:
        lines.append(
            f"| {family} | {row['release']['targets']} | "
            + " | ".join(_fmt_pct(row[arm][metric]) for arm in arms)
            + deltas(row)
            + " |"
        )
    totals = receipt["fit_overall"]["non_fsa"]
    lines += [
        f"| **all non-FSA targets** | {totals['release']['targets']} | "
        + " | ".join(_fmt_pct(totals[arm][metric]) for arm in arms)
        + deltas(totals)
        + " |",
        "",
        "## Racial composition of borrowers, before and after calibration",
        "",
        "Borrower households by the reference person's race (the SCF's "
        "`racecl4`; the SCF column pools the five implicates):",
        "",
        "| | SCF 2025 | " + _labels(receipt, arms) + " | B - release (pp) |",
        "|---|---:|" + "---:|" * (len(arms) + 1),
    ]
    race = receipt["race"]
    households = race["households"]
    for label in _RACE_LABELS.values():
        lines.append(
            f"| {label} | {_fmt_pct(households['scf_2025'][label])} | "
            + " | ".join(_fmt_pct(households[arm][label]) for arm in arms)
            + f" | {_fmt_points(households['b_with_fsa'][label] - households['release'][label])} |"
        )
    lines += [
        "",
        "Borrowers (people) by their own race:",
        "",
        "| | " + _labels(receipt, arms) + " |",
        "|---|" + "---:|" * len(arms),
    ]
    for label in _RACE_LABELS.values():
        lines.append(
            f"| {label} | "
            + " | ".join(
                _fmt_pct(race["people"][arm]["borrowers"][label]) for arm in arms
            )
            + " |"
        )
    lines += [
        "",
        "Adults with a balance, by race:",
        "",
        "| | " + _labels(receipt, arms) + " |",
        "|---|" + "---:|" * len(arms),
    ]
    for label in _RACE_LABELS.values():
        lines.append(
            f"| {label} | "
            + " | ".join(
                _fmt_pct(race["people"][arm]["borrower_share_of_adults_by_race"][label])
                for arm in arms
            )
            + " |"
        )
    weights = receipt["weights"]
    lines += [
        "",
        "## Weight movement against the published weights",
        "",
        "| | " + _labels(receipt, solved) + " |",
        "|---|" + "---:|" * len(solved),
    ]
    initial_ess = weights[solved[0]]["effective_sample_size"]["initial"]
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
            f"Effective sample size (release {initial_ess:,.0f})",
            lambda m: f"{m['effective_sample_size']['final']:,.0f}",
        ),
    )
    for label, getter in rows_movement:
        lines.append(
            f"| {label} | " + " | ".join(getter(weights[arm]) for arm in solved) + " |"
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
        "| Target | FSA (reconciled) | " + _labels(receipt, arms) + " |",
        "|---|---:|" + "---:|" * len(arms),
    ]
    for row in receipt["fsa_targets"]:
        formatter = _fmt_count if row["measure_mode"] == "indicator_sum" else _fmt_money
        lines.append(
            f"| {row['label']} | {formatter(row['target'])} | "
            + " | ".join(
                f"{formatter(row[arm])} ({_fmt_signed_pct(row[arm + '_error'])})"
                for arm in arms
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def income_markdown(receipt: Mapping[str, object]) -> list[str]:
    """Households with debt by income quintile, at each arm's own cut points and
    at the release arm's, with the A controls beside each B arm."""

    income = receipt["income_quintiles"]
    arms = _arms(receipt)
    header = "| | " + " | ".join(f"Q{q}" for q in _QUINTILES)
    header += (
        " | Q5 - Q1, points | Bottom two quintiles' share of borrower households |"
    )
    rule = "|---|" + "---:|" * (len(_QUINTILES) + 2)

    def row(label: str, table: Mapping[str, object]) -> str:
        shares = table["share_with_federal_education_debt"]
        return (
            f"| {label} | "
            + " | ".join(_fmt_pct(shares[q]) for q in _QUINTILES)
            + f" | {_fmt_points(table['q5_minus_q1'])}"
            + f" | {_fmt_pct(table['bottom_two_share_of_borrower_households'])} |"
        )

    own = income["own_cut_points"]
    fixed = income["release_cut_points"]
    lines = [
        "",
        "## Households with federal education debt by household income quintile",
        "",
        "The income is the household income the imputation conditions on, over "
        "the SCF primary economic unit. The student debt pilot counts borrowers "
        "as people and cuts tax-unit AGI over all tax units, so these rows are "
        "not its income tables. Where Q4 tops a row, Q5 - Q1 is an endpoint "
        "contrast rather than a slope.",
        "",
        "Each arm's own weighted quintiles (the SCF's are its own families'):",
        "",
        header,
        rule,
        row("SCF 2025", own["scf_2025"]),
        *(row(receipt["arms"][arm]["label"], own[arm]) for arm in arms),
    ]
    ranges = income["release_income_range_by_quintile"]
    lines += [
        "",
        "At the release arm's cut points: every household keeps the quintile it "
        "has at the published weights, so each column holds the same households "
        "in every arm. The release row reproduces its own-cut row"
        + ("" if income["release_fixed_cuts_reproduce_own"] else " (it does not here)")
        + ". Income range of each release quintile: "
        + "; ".join(
            f"Q{q} {_fmt_money(ranges[q][0])} to {_fmt_money(ranges[q][1])}"
            for q in _QUINTILES
        )
        + ".",
        "",
        header,
        rule,
        *(row(receipt["arms"][arm]["label"], fixed[arm]) for arm in arms),
        "",
        "Each fixed quintile's share of household weight:",
        "",
        "| | " + " | ".join(f"Q{q}" for q in _QUINTILES) + " |",
        "|---|" + "---:|" * len(_QUINTILES),
        *(
            f"| {receipt['arms'][arm]['label']} | "
            + " | ".join(
                _fmt_pct(fixed[arm]["share_of_household_weight"][q]) for q in _QUINTILES
            )
            + " |"
            for arm in arms
        ),
    ]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument("--feed", type=Path, required=True)
    parser.add_argument("--feed-provenance", type=Path)
    parser.add_argument("--scf-cache-dir", type=Path, required=True)
    parser.add_argument("--run", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--extra-chi-square-lambda",
        type=float,
        action="append",
        default=[],
        help=(
            "Also solve A and B with the chi-square penalty at this lambda "
            f"(beside {L2_SENSITIVITY_LAMBDA:g}); repeatable."
        ),
    )
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
    grid = arm_grid(args.extra_chi_square_lambda)
    all_arms = (RELEASE_ARM, *grid)
    pairs = arm_pairs(grid)

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
    if fsa_receipt["status"] != "bound":
        raise RuntimeError(
            f"The feed binds no FSA target (status {fsa_receipt['status']!r}); "
            "the fit needs a feed that carries the ed_fsa facts."
        )
    parent_plus = parent_plus_diagnostic(facts, fsa_receipt)
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
    for arm, spec in grid.items():
        arm_registry = registry_b if spec.with_fsa else registry_a
        loss_weights = release._fiscal_target_loss_weights(
            arm_registry, cli.target_family_loss_multipliers
        )
        if spec.count_weight != 1.0:
            loss_weights = scaled_borrower_count_loss_weights(
                arm_registry.specs, loss_weights, spec.count_weight
            )
        l2_lambda = cli.l2_lambda if spec.l2_lambda is None else spec.l2_lambda
        _log(
            f"calibrating arm {arm} ({len(arm_registry)} targets, {epochs} epochs, "
            f"l2 {spec.l2_basis} {l2_lambda:g}, "
            f"borrower-count weight x{spec.count_weight:g})"
        )
        t0 = time.perf_counter()
        results[arm] = calibrate(
            target_frame,
            arm_registry.to_target_set(),
            epochs=epochs,
            learning_rate=cli.learning_rate,
            max_weight_ratio=cli.max_weight_ratio,
            seed=args.seed,
            mass="conserve",
            l2_lambda=l2_lambda,
            l2_basis=spec.l2_basis,
            target_loss_weights=loss_weights,
            target_loss_cap=release.US_FISCAL_TARGET_LOSS_CAP,
        )
        timings[arm] = time.perf_counter() - t0
        _log(
            f"arm {arm}: loss {results[arm].initial_loss:.5f} -> "
            f"{results[arm].final_loss:.5f} in {timings[arm]:.0f}s"
        )
    weights = {
        RELEASE_ARM: initial,
        **{arm: np.asarray(results[arm].weights, dtype=np.float64) for arm in grid},
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
        for arm in all_arms:
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
            arm: _summary([row[f"{arm}_error"] for row in members]) for arm in all_arms
        }
    fit_overall = {
        group: {
            arm: _summary(
                [row[f"{arm}_error"] for row in rows if row["is_fsa"] == is_fsa]
            )
            for arm in all_arms
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
            **{arm: household_race_composition(fit_tables[arm]) for arm in all_arms},
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
    # FSA's Parent PLUS figure is a diagnostic fact since #1525: its share of
    # the loan-type table's dollar rows, applied to the in-scope control.
    pp_target = parent_plus["parent_plus_share_of_in_scope_control"]
    headline = [
        {
            "label": "Federal balance, own plus Parent PLUS",
            "kind": "amount",
            "target": controls["total_balance"]["in_scope_control"],
            **{arm: float(np.dot(own + plus, person_weights[arm])) for arm in all_arms},
        },
        {
            "label": "Borrowers (people with a balance)",
            "kind": "count",
            "target": controls["borrowers"]["in_scope_control"],
            **{
                arm: float(person_weights[arm][(own + plus) > 0].sum())
                for arm in all_arms
            },
        },
        {
            "label": "Parent PLUS balance (FSA: diagnostic)",
            "kind": "amount",
            "target": pp_target,
            **{arm: float(np.dot(plus, person_weights[arm])) for arm in all_arms},
        },
        {
            "label": "Parent PLUS share of the balance",
            "kind": "share",
            "target": pp_target / controls["total_balance"]["in_scope_control"],
            **{
                arm: float(np.dot(plus, person_weights[arm]))
                / float(np.dot(own + plus, person_weights[arm]))
                for arm in all_arms
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
                for arm in all_arms
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
                for arm in all_arms
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
            "sha256": feed_sha,
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
            "surface": surface_receipt,
        },
        # Table order: the release, then the solved grid.
        "arms": {
            RELEASE_ARM: {"label": "Release", "weights": "published"},
            **{
                arm: {
                    "label": spec.label,
                    "with_fsa": spec.with_fsa,
                    "targets": len(registry_b if spec.with_fsa else registry_a),
                    "l2_lambda": (
                        cli.l2_lambda if spec.l2_lambda is None else spec.l2_lambda
                    ),
                    "l2_basis": spec.l2_basis,
                    "borrower_count_loss_weight": spec.count_weight,
                }
                for arm, spec in grid.items()
            },
        },
        "arm_pairs": [list(pair) for pair in pairs],
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
            "parent_plus": parent_plus,
            "repayment_plan_diagnostic": repayment_plan_diagnostic(facts),
        },
        "headline": headline,
        "fit_by_family": fit_by_family,
        "fit_overall": fit_overall,
        "fsa_targets": [row for row in rows if row["is_fsa"]],
        "race": race,
        "support": borrower_support(target_frame),
        "scf_fit_tables": {"scf_2025": scf_fit, **fit_tables},
        "income_quintiles": income_quintile_tables(
            target_frame, weights, fit_tables, scf_fit
        ),
        "weights": {
            arm: weight_movement(
                target_frame, initial, weights[arm], cli.max_weight_ratio
            )
            for arm in grid
        },
        "seconds": time.perf_counter() - started,
    }
    receipt["arm_comparison"] = arm_comparison(
        rows, headline, fit_overall, receipt["weights"], all_arms
    )
    # Strict JSON: a missing value (the household share's FSA target, an
    # empty group's share) is null, never a bare NaN.
    receipt = json_ready(receipt)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(receipt, indent=2, default=float, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    args.out_md.write_text(markdown(receipt), encoding="utf-8", newline="\n")
    _log(f"wrote {args.out_json} and {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
