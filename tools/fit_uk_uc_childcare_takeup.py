"""Fit the Universal Credit childcare-element take-up rates by family type.

The engine pays the childcare element to every benefit unit that meets the
work condition and reports childcare costs; DWP publishes the households that
receive it, split single / couple (Universal Credit childcare element
statistics, Table 1). The spine's ``would_claim_uc_childcare`` flag is drawn
at one rate per family type, and the quantity a rate controls is the
expectation ``rate * sum(weight * paid_uc * element > 0)`` over the units of
that family type, so each rate is the published calendar-year mean divided by
that weighted base at the input's weights (the #834 expected-count objective, here
with one target per rate and no cross-scheme interaction). The receipt lands
in ``take_up_contract.json`` as the entries' ``fitting_receipt``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.uk_runtime.take_up_contract import load_uk_take_up_contract

TOOL_VERSION = 1
CONCEPT = "dwp.uc_households_receiving_childcare_element"
MEASURES = {"single": "single_benefit_units", "couple": "couple_benefit_units"}
RATE_KEYS = {"single": "uc_childcare_single", "couple": "uc_childcare_couple"}
GB_REGIONS = (
    "NORTH_EAST",
    "NORTH_WEST",
    "YORKSHIRE",
    "EAST_MIDLANDS",
    "WEST_MIDLANDS",
    "EAST_OF_ENGLAND",
    "LONDON",
    "SOUTH_EAST",
    "SOUTH_WEST",
    "WALES",
    "SCOTLAND",
)


ACCEPTANCE_ROOT_MARKER = "data/ukds/acceptance"


def _portable_path(path: Path) -> str:
    """Record the input relative to the licensed acceptance root, never a home path."""

    text = str(path)
    marker = text.find(ACCEPTANCE_ROOT_MARKER)
    return text[marker:] if marker >= 0 else path.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def published_means(
    ledger_facts: Path, *, expected_sha256: str | None, year: int
) -> dict:
    """Calendar-year means of the published single and couple counts."""

    artifact = load_ledger_consumer_artifact(
        ledger_facts, expected_facts_sha256=expected_sha256
    )
    months = {f"{year}-{month:02d}" for month in range(1, 13)}
    by_measure: dict[str, list[float]] = {measure: [] for measure in MEASURES.values()}
    for fact in artifact.facts:
        measure = fact.get("observed_measure", {})
        if measure.get("source_concept") != CONCEPT:
            continue
        if fact.get("period", {}).get("value") not in months:
            continue
        measure_id = measure.get("source_measure_id")
        if measure_id in by_measure:
            by_measure[measure_id].append(float(fact["value"]))
    means = {}
    for family, measure_id in MEASURES.items():
        values = by_measure[measure_id]
        if len(values) != 12:
            raise ValueError(
                f"expected 12 monthly {measure_id} facts for {year}, found {len(values)}"
            )
        means[family] = float(np.mean(values))
    return {"means": means, "facts_sha256": artifact.facts_sha256}


def weighted_bases(input_h5: Path, year: int) -> dict:
    """Weighted paid-UC GB units with a positive childcare element, by family type."""

    from policyengine_uk import Microsimulation

    simulation = Microsimulation(dataset=str(input_h5))
    benunit = simulation.populations["benunit"]
    weight = np.asarray(
        simulation.calculate("benunit_weight", year).values, dtype=float
    )
    labels = np.asarray(simulation.calculate("region", year, map_to="person")).astype(
        str
    )
    codes, uniques = pd.factorize(pd.Series(labels))
    region = np.asarray(uniques)[benunit.value_from_first_person(codes)].astype(str)
    universal_credit = np.asarray(
        simulation.calculate("universal_credit", year).values, dtype=float
    )
    element = np.asarray(
        simulation.calculate("uc_childcare_element", year).values, dtype=float
    )
    couple = np.asarray(simulation.calculate("is_married", year).values, dtype=bool)
    paid = (universal_credit > 0) & (element > 0) & np.isin(region, GB_REGIONS)
    bases = {
        "single": float(weight[paid & ~couple].sum()),
        "couple": float(weight[paid & couple].sum()),
    }
    amount = {
        "single": float((element * weight)[paid & ~couple].sum()),
        "couple": float((element * weight)[paid & couple].sum()),
    }
    return {
        "bases": bases,
        "element_gbp": amount,
        "records": {
            "single": int((paid & ~couple).sum()),
            "couple": int((paid & couple).sum()),
        },
    }


def fit(
    input_h5: Path,
    ledger_facts: Path,
    *,
    ledger_facts_sha256: str | None,
    year: int,
    generated_at: str | None = None,
) -> dict:
    published = published_means(
        ledger_facts, expected_sha256=ledger_facts_sha256, year=year
    )
    model = weighted_bases(input_h5, year)
    rates = {}
    achieved = {}
    for family in MEASURES:
        base = model["bases"][family]
        target = published["means"][family]
        rate = min(1.0, target / base) if base > 0 else 1.0
        rates[RATE_KEYS[family]] = round(rate, 4)
        achieved[f"dwp.uc.households_childcare_element_{family}"] = {
            "target": round(target, 3),
            "base_at_rate_one": round(base, 3),
            "value": round(rates[RATE_KEYS[family]] * base, 3),
            "ratio": round(rates[RATE_KEYS[family]] * base / target, 4)
            if target
            else None,
        }
    try:
        engine_version = metadata.version("policyengine-uk")
    except metadata.PackageNotFoundError:
        engine_version = None
    receipt = {
        "tool": "fit_uk_uc_childcare_takeup",
        "tool_version": TOOL_VERSION,
        "input_h5": _portable_path(input_h5),
        "input_sha256": _sha256(input_h5),
        "seed": 0,
        "objective": "expected_count",
        "generated_at": generated_at or datetime.now(UTC).date().isoformat(),
        "model_period": year,
        "engine_version": engine_version,
        "ledger_facts_sha256": published["facts_sha256"],
        "stochastic_contract_sha256_at_fit": load_uk_take_up_contract().resource_sha256,
        "basis": (
            "rate = published calendar-year mean / paid-UC GB benefit units with a positive "
            "childcare element of that family type, weighted as the input H5 carries them (a "
            "calibrated candidate fits at its calibrated weights, a spine at design weights); "
            "one target per rate, so the expected-count objective is exact and needs no optimizer"
        ),
        "published_means": published["means"],
        "model": model,
        "rates": rates,
        "achieved": achieved,
        "status": "fitted at the input's weights; realized counts are measured on the twins in the receipts",
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in receipt.items() if k != "receipt_sha256"}, sort_keys=True
        ).encode()
    ).hexdigest()
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5",
        type=Path,
        required=True,
        help="an H5 the engine loads; the rates are fitted at the weights it carries",
    )
    parser.add_argument(
        "--ledger-facts",
        type=Path,
        required=True,
        help="Chronicle consumer artifact directory",
    )
    parser.add_argument("--ledger-facts-sha256")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = fit(
        args.input_h5,
        args.ledger_facts,
        ledger_facts_sha256=args.ledger_facts_sha256,
        year=args.year,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "rates": receipt["rates"],
                "achieved": receipt["achieved"],
                "published": receipt["published_means"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
