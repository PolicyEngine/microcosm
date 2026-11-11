#!/usr/bin/env python
"""The gainers of the trading allowance raise on a UK release, and a re-fit of
the claimant rates (microcosm#1439).

The trading_allowance_receipts stage stores each trader's
``trading_allowance_claim_status`` and leaves ``self_employment_gross_receipts``
at zero. This tool reads one release H5 (final weights) on the locked
policyengine-uk and applies the statuses and zero receipts in memory. It reads
the statuses the stage stored where the release carries them. On a release built
before the stage stored them, it makes the stage's draw again on the build's
person ids (``person_build_id``): the release renumbers ``person_id``, and a draw
keyed on the renumbered ids is a different draw. It then runs one simulation
pair at ``--period`` (2028, inside 2027-28): the allowance at
GBP 3,000, current law from 6 April 2027, against a reform holding it at GBP 1,000.
A gainer is a person whose income tax is lower under the raise. It reports the
weighted gainers and their mean gain, overall and by the stage's profit bands.

The check is judged on the expected count (María's ruling of 2026-11-09): a
second simulation pair sets every trader outside the full-relief group to
CLAIMANT, which gives each band's would-be gainers, and the expected gainers are
each band's rate times its would-be gainers. That count is held to HMRC's
400,000 within 5%. The realised count, from the release's own draw, is reported
beside it with the draw's spread: by band, rate times one less the rate, times
the would-be gainers squared, over their Kish effective records. On a release's
few effective records that spread can exceed the 5% tolerance, so the realised
count of one draw cannot decide the check.

With ``--fit`` every trader outside the full-relief group is set to CLAIMANT,
each band's weighted gainers and total gain are summed, and the two free band
rates are solved for HMRC's policy paper on the Budget 2026 raise (about
400,000 taxpayers with a lower bill, by about GBP 380 on average). Gainers and
gains are linear in the rates, so one pair is enough. The solved rates go into
the stage's module and spec before the arm (María's answer of 2026-11-06).

The evidence written under ``docs/evidence/uk-trading-allowance-1439/`` holds
aggregates only: band totals with their support in distinct source households,
withheld where fewer than three households carry them (the unit-record guard).
It needs the UK engine environment and about 5 to 8 GB: run it alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.uk_runtime.national_frame import UK_BUILD_IDENTIFIER_COLUMNS
from microcosm.build.uk_runtime.receipt_counts import support_band
from microcosm.build.uk_runtime.trading_allowance_receipts import (
    CLAIMANT,
    TRADING_ALLOWANCE_CLAIMANT_BANDS,
    TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT,
    TRADING_ALLOWANCE_STATUS_OUTPUT,
    assign_trading_allowance_claim_status,
)

ALLOWANCE = "gov.hmrc.income_tax.allowances.trading_allowance"
#: The raise's start, and the allowance the reform holds from it.
RAISE_START = "2027-04-06"
UNRAISED_ALLOWANCE = 1_000
TARGET_GAINERS = 400_000.0
TARGET_MEAN_GAIN = 380.0
#: The check's tolerance on the expected gainers (#1439's post-build check).
GAINERS_TOLERANCE = 0.05
EXPECTED_COUNT_RULE = (
    "each band's claimant rate times its would-be gainers (every trader outside "
    "the full-relief group a claimant), held to HMRC's 400,000 within 5%; the "
    "realised count of the release's draw is reported beside it with the draw's "
    "spread (María's ruling of 2026-11-09)"
)
DISCLOSURE_FLOOR = 3
#: The UK release's survey year.
FIRST_DATA_YEAR = 2024
EVIDENCE_DIR = Path("docs/evidence/uk-trading-allowance-1439")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _band_index(profit: np.ndarray) -> np.ndarray:
    """Each person's stage profit band (0, 1, ...), or -1 with no positive profit."""

    index = np.full(profit.shape, -1, dtype=int)
    lower = 0.0
    for position, (upper, _rate) in enumerate(TRADING_ALLOWANCE_CLAIMANT_BANDS):
        in_band = profit > lower
        if upper is not None:
            in_band &= profit <= upper
        index[in_band & (index < 0)] = position
        if upper is None:
            break
        lower = upper
    return index


def _stored_people(h5: Path) -> pd.DataFrame:
    person = pd.read_hdf(h5, "person")
    household = pd.read_hdf(h5, "household")
    columns = {"household_id", "household_is_spi_synthetic", "household_source_id"}
    missing = sorted(columns - set(household.columns))
    if missing:
        raise ValueError(f"{h5}: the household table carries no {missing}.")
    household = household.loc[:, sorted(columns)].set_index("household_id")
    optional = [
        UK_BUILD_IDENTIFIER_COLUMNS["person"],
        TRADING_ALLOWANCE_STATUS_OUTPUT,
        TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT,
    ]
    wanted = ["person_id", "person_household_id", "self_employment_income"]
    joined = person.loc[
        :, [*wanted, *(column for column in optional if column in person.columns)]
    ].join(household, on="person_household_id")
    if joined["household_source_id"].isna().any():
        raise ValueError(f"{h5}: a person has no household row.")
    return joined


def read_claim_statuses(people: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, str]:
    """Each person's claim status and full-relief flag, and where they came from.

    The statuses the stage stored, where the release carries them. Otherwise the
    stage's draw, made again on the ids it was keyed on: the build's, which a
    release keeps in ``person_build_id`` beside its renumbered ``person_id``. A
    frame with no build ids was never renumbered, so its ``person_id`` is the
    build's.
    """

    stored = [
        column
        for column in (
            TRADING_ALLOWANCE_STATUS_OUTPUT,
            TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT,
        )
        if column in people.columns
    ]
    if len(stored) == 1:
        raise ValueError(f"The release stores {stored} without its partner.")
    if stored:
        return (
            people[TRADING_ALLOWANCE_STATUS_OUTPUT].astype(str).to_numpy(object),
            people[TRADING_ALLOWANCE_FULL_RELIEF_OUTPUT].to_numpy(dtype=bool),
            "stored",
        )
    build_id = UK_BUILD_IDENTIFIER_COLUMNS["person"]
    keyed_on = build_id if build_id in people.columns else "person_id"
    statuses = assign_trading_allowance_claim_status(
        people[keyed_on].to_numpy(),
        people["self_employment_income"].to_numpy(dtype=float),
        spi_row=people["household_is_spi_synthetic"].to_numpy(dtype=bool),
    )
    return statuses.status, statuses.full_relief, f"drawn on {keyed_on}"


def _income_tax(h5: Path, period: int, status: np.ndarray, *, raised: bool):
    from policyengine_uk import Microsimulation

    reform = (
        None
        if raised
        else {ALLOWANCE: {f"{RAISE_START}.2100-12-31": UNRAISED_ALLOWANCE}}
    )
    simulation = Microsimulation(dataset=str(h5), reform=reform)
    # Every year from the data's first to the period simulated, so a release
    # that stores #1428's receipts for its projected years reads zero.
    for year in range(FIRST_DATA_YEAR, period + 1):
        simulation.set_input("trading_allowance_claim_status", year, status)
        simulation.set_input(
            "self_employment_gross_receipts", year, np.zeros(len(status))
        )
    tax = np.asarray(simulation.calculate("income_tax", period).values, dtype=float)
    weight = np.asarray(
        simulation.calculate("household_weight", period, map_to="person").values,
        dtype=float,
    )
    return tax, weight


def _gains(h5: Path, period: int, status: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raised, weight = _income_tax(h5, period, status, raised=True)
    unraised, _ = _income_tax(h5, period, status, raised=False)
    return unraised - raised, weight


def _cell(weight, gain, mask, sources) -> dict[str, object]:
    gainer = mask & (gain > 0.005)
    support = int(pd.unique(sources[gainer]).size)
    if support < DISCLOSURE_FLOOR:
        return {"support": support_band(support), "withheld": True}
    weighted_gainers = float(weight[gainer].sum())
    total_gain = float((weight * gain)[gainer].sum())
    return {
        "support": support_band(support),
        "weighted_gainers": round(weighted_gainers, -2),
        "total_gain": round(total_gain, -3),
        "mean_gain": round(total_gain / weighted_gainers, 0),
    }


def measure(h5: Path, period: int, *, fit: bool) -> dict[str, object]:
    people = _stored_people(h5)
    profit = people["self_employment_income"].to_numpy(dtype=float)
    status, full_relief, statuses_read = read_claim_statuses(people)
    if fit:
        status = np.where((profit > 0.0) & ~full_relief, CLAIMANT, status)
    gain, weight = _gains(h5, period, status.astype(str))
    sources = people["household_source_id"].to_numpy()
    band = _band_index(profit)
    bands = []
    for position, (upper, rate) in enumerate(TRADING_ALLOWANCE_CLAIMANT_BANDS):
        bands.append(
            {
                "upper_inclusive": upper,
                "rate": rate,
                **_cell(weight, gain, band == position, sources),
            }
        )
    evidence: dict[str, object] = {
        "tool": "tools/measure_uk_trading_allowance_gainers.py",
        "release": h5.name,
        "release_sha256": _sha256(h5),
        "policyengine_uk": version("policyengine-uk"),
        "period": period,
        "mode": "fit" if fit else "measure",
        "statuses": statuses_read,
        "overall": _cell(weight, gain, np.ones(len(gain), dtype=bool), sources),
        "bands": bands,
    }
    if fit:
        evidence["fit"] = _solve_rates(weight, gain, band)
    else:
        every_claimant = np.where((profit > 0.0) & ~full_relief, CLAIMANT, status)
        would_be, _ = _gains(h5, period, every_claimant.astype(str))
        evidence["check"] = expected_gainers_check(
            weight=weight,
            would_be_gain=would_be,
            band=band,
            sources=sources,
            realised=evidence["overall"],
        )
    return evidence


def expected_gainers_check(
    *,
    weight: np.ndarray,
    would_be_gain: np.ndarray,
    band: np.ndarray,
    sources: np.ndarray,
    realised: dict[str, object],
) -> dict[str, object]:
    """The post-build check on the expected gainers, with the realised beside it.

    ``would_be_gain`` is each person's gain with every trader outside the
    full-relief group a claimant, so a band's would-be gainers are its people
    with a gain. The expected gainers sum each band's rate times its would-be
    gainers' weight; the draw's variance sums, by band, the rate times one less
    the rate times that weight squared over the band's Kish effective records.
    A band whose would-be gainers rest on fewer than three source households
    is withheld, and so is every total that would give it back.
    """

    weight = np.asarray(weight, dtype=np.float64)
    would_be_gain = np.asarray(would_be_gain, dtype=np.float64)
    band = np.asarray(band)
    sources = np.asarray(sources)
    expected = 0.0
    expected_gain = 0.0
    variance = 0.0
    bands: list[dict[str, object]] = []
    withheld = False
    for position, (upper, rate) in enumerate(TRADING_ALLOWANCE_CLAIMANT_BANDS):
        pool = (band == position) & (would_be_gain > 0.005)
        support = int(pd.unique(sources[pool]).size)
        entry: dict[str, object] = {
            "upper_inclusive": upper,
            "rate": rate,
            "support": support_band(support),
        }
        if rate > 0.0 and support < DISCLOSURE_FLOOR:
            withheld = True
            entry["withheld"] = True
            bands.append(entry)
            continue
        mass = float(weight[pool].sum())
        squares = float((weight[pool] ** 2).sum())
        effective = mass**2 / squares if squares > 0.0 else 0.0
        expected += rate * mass
        expected_gain += rate * float((weight * would_be_gain)[pool].sum())
        if effective > 0.0:
            variance += rate * (1.0 - rate) * mass**2 / effective
        if rate > 0.0:
            entry.update(
                {
                    "would_be_gainers": round(mass, -2),
                    "expected_gainers": round(rate * mass, -2),
                    "effective_records": round(effective, 0),
                }
            )
        bands.append(entry)
    check: dict[str, object] = {
        "judged_on": "expected_count",
        "rule": EXPECTED_COUNT_RULE,
        "target_gainers": TARGET_GAINERS,
        "tolerance": GAINERS_TOLERANCE,
        "bands": bands,
    }
    if withheld:
        check["withheld"] = True
        return check
    spread = float(np.sqrt(variance))
    check.update(
        {
            "expected_gainers": round(expected, -2),
            "expected_mean_gain": round(expected_gain / expected, 0)
            if expected
            else None,
            "relative_gap": round(expected / TARGET_GAINERS - 1.0, 4),
            "within_tolerance": bool(
                abs(expected / TARGET_GAINERS - 1.0) <= GAINERS_TOLERANCE
            ),
            "draw_spread_sd": round(spread, -2),
        }
    )
    realised_gainers = realised.get("weighted_gainers")
    if isinstance(realised_gainers, (int, float)):
        check["realised_gainers"] = realised_gainers
        check["realised_minus_expected_in_sd"] = (
            round((float(realised_gainers) - expected) / spread, 2) if spread else None
        )
    return check


def _solve_rates(weight, gain, band) -> dict[str, object]:
    """The two free band rates for HMRC's gainers and mean gain."""

    free = [
        position
        for position, (_, rate) in enumerate(TRADING_ALLOWANCE_CLAIMANT_BANDS)
        if rate > 0.0
    ]
    if len(free) != 2:
        raise ValueError(f"the fit solves two band rates, the stage has {free}.")
    gainer = gain > 0.005
    gainers = [float(weight[gainer & (band == position)].sum()) for position in free]
    gains = [
        float((weight * gain)[gainer & (band == position)].sum()) for position in free
    ]
    matrix = np.array([gainers, gains])
    targets = np.array([TARGET_GAINERS, TARGET_GAINERS * TARGET_MEAN_GAIN])
    rates = np.linalg.solve(matrix, targets)
    return {
        "target_gainers": TARGET_GAINERS,
        "target_mean_gain": TARGET_MEAN_GAIN,
        "rates": [round(float(rate), 4) for rate in rates],
        "within_unit_interval": bool(((rates >= 0.0) & (rates <= 1.0)).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--h5", type=Path, required=True, help="A UK release H5.")
    parser.add_argument("--period", type=int, default=2028)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--label", required=True, help="The evidence file's name.")
    args = parser.parse_args()
    evidence = measure(args.h5, args.period, fit=args.fit)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE_DIR / f"{args.label}.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence.get("fit") or evidence["check"], indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
