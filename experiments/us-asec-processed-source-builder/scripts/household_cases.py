"""Seeded synthetic CPS households that stress tax-unit construction.

Imported by ``differential_tax_units.py`` in two environments (Microcosm's and
the archived loader's), so it needs only NumPy and pandas. Every household has
unique line numbers, as Census households do. The generator mixes:

* mutual married spouse-present pairs (``A_MARITL`` 1 or 2), one-sided and
  dangling spouse pointers, self-pointers and spouse-absent marital codes.
  A one-sided pointer that would put someone in two married pairs is
  removed: both constructions mishandle that structure (see
  :func:`probe_cases`), and no Census file contains it;
* parent pointers to present lines, absent lines, 0 and -1;
* every ``A_EXPRRP`` code, including invalid ones (0, 6, 99);
* incomes on both sides of the 2024 and 2025 qualifying-relative limits
  ($5,050 and $5,200), negative self-employment income, Social Security, and
  ``PTOTVAL`` independent of the components;
* disability, enrollment, full-time and school-level codes 0, 1 and 2.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

INCOME_COLUMNS = (
    "WSAL_VAL",
    "SEMP_VAL",
    "FRSE_VAL",
    "INT_VAL",
    "DIV_VAL",
    "RNT_VAL",
    "CAP_VAL",
    "UC_VAL",
    "OI_VAL",
    "ANN_VAL",
    "PNSN_VAL",
)
DISABILITY_COLUMNS = (
    "PEDISDRS",
    "PEDISEAR",
    "PEDISEYE",
    "PEDISOUT",
    "PEDISPHY",
    "PEDISREM",
)
CASE_COLUMNS = (
    "PH_SEQ",
    "A_LINENO",
    "A_AGE",
    "A_MARITL",
    "A_SPOUSE",
    "PEPAR1",
    "PEPAR2",
    "A_EXPRRP",
    *INCOME_COLUMNS,
    "SS_VAL",
    "PTOTVAL",
    *DISABILITY_COLUMNS,
    "A_ENRLW",
    "A_FTPT",
    "A_HSCOL",
)
_LIMIT_EDGES = (
    0,
    1,
    4_399,
    4_400,
    4_699,
    4_700,
    5_049,
    5_050,
    5_051,
    5_199,
    5_200,
    5_201,
)


def _income(rng: np.random.Generator) -> int:
    draw = rng.random()
    if draw < 0.45:
        return 0
    if draw < 0.70:
        return int(rng.choice(_LIMIT_EDGES))
    if draw < 0.90:
        return int(rng.integers(1, 9_000))
    return int(rng.integers(9_000, 250_000))


def _household(rng: np.random.Generator, household_id: int) -> list[dict[str, int]]:
    size = int(
        rng.choice(
            np.arange(1, 11), p=np.array([22, 24, 16, 14, 9, 6, 4, 2, 2, 1]) / 100
        )
    )
    if rng.random() < 0.8:
        lines = list(range(1, size + 1))
    else:
        lines = sorted(
            int(line) for line in rng.choice(np.arange(1, 17), size=size, replace=False)
        )
    people = []
    for line in lines:
        band = rng.random()
        if band < 0.30:
            age = int(rng.integers(0, 18))
        elif band < 0.45:
            age = int(rng.integers(18, 25))
        elif band < 0.85:
            age = int(rng.integers(25, 65))
        else:
            age = int(rng.integers(65, 86))
        marital = 7 if age < 15 and rng.random() < 0.97 else int(rng.integers(1, 8))
        people.append(
            {
                "PH_SEQ": household_id,
                "A_LINENO": line,
                "A_AGE": age,
                "A_MARITL": marital,
                "A_SPOUSE": 0,
                "PEPAR1": -1,
                "PEPAR2": -1,
                "A_EXPRRP": 0,
            }
        )
    order = rng.permutation(len(people))
    reference = people[int(order[0])]
    reference["A_EXPRRP"] = int(rng.choice([1, 2]))
    adults = [person for person in people if person["A_AGE"] >= 15]
    rng.shuffle(adults)
    # Mutual married spouse-present pairs.
    while len(adults) >= 2 and rng.random() < 0.55:
        first, second = adults.pop(), adults.pop()
        first["A_SPOUSE"], second["A_SPOUSE"] = second["A_LINENO"], first["A_LINENO"]
        first["A_MARITL"] = second["A_MARITL"] = int(rng.choice([1, 2]))
        if first is reference or second is reference:
            other = second if first is reference else first
            other["A_EXPRRP"] = int(rng.choice([3, 4]))
    # Malformed spouse pointers: one-sided, absent line, self, spouse-absent.
    for person in people:
        draw = rng.random()
        if person["A_SPOUSE"] == 0 and draw < 0.06:
            person["A_SPOUSE"] = int(rng.choice([p["A_LINENO"] for p in people]))
        elif person["A_SPOUSE"] == 0 and draw < 0.09:
            person["A_SPOUSE"] = 17
        elif person["A_SPOUSE"] != 0 and draw < 0.04:
            person["A_MARITL"] = int(rng.choice([3, 4, 5, 6]))
    parents = [p["A_LINENO"] for p in people if p["A_AGE"] >= 14]
    for person in people:
        if person["A_EXPRRP"] == 0:
            person["A_EXPRRP"] = int(
                rng.choice([5, 5, 5, 7, 8, 9, 10, 11, 12, 13, 14, 3, 4, 0, 6, 99])
            )
        draw = rng.random()
        if parents and person["A_AGE"] < 30 and draw < 0.7:
            candidates = [line for line in parents if line != person["A_LINENO"]]
            if candidates:
                person["PEPAR1"] = int(rng.choice(candidates))
                if rng.random() < 0.5:
                    person["PEPAR2"] = int(rng.choice(candidates))
        elif draw < 0.75:
            person["PEPAR1"] = 17  # an absent line: lines run 1-16
        elif draw < 0.78:
            person["PEPAR1"] = 0
        for column in INCOME_COLUMNS:
            person[column] = _income(rng) if rng.random() < 0.25 else 0
        if rng.random() < 0.05:
            person["SEMP_VAL"] = -int(rng.integers(1, 20_000))
        person["SS_VAL"] = _income(rng) if rng.random() < 0.2 else 0
        components = sum(person[column] for column in INCOME_COLUMNS) + person["SS_VAL"]
        person["PTOTVAL"] = (
            components if rng.random() < 0.7 else int(rng.integers(-5_000, 120_000))
        )
        for column in DISABILITY_COLUMNS:
            person[column] = int(
                rng.choice([0, 2, 2, 2, 1] if rng.random() < 0.3 else [0, 2])
            )
        person["A_ENRLW"] = int(rng.choice([0, 1, 2]))
        person["A_FTPT"] = int(rng.choice([0, 1, 2]))
        person["A_HSCOL"] = int(rng.choice([0, 1, 2]))
    _break_spouse_chains(people)
    return [people[int(index)] for index in rng.permutation(len(people))]


def _married_spouse_present(person: dict[str, int]) -> bool:
    return person["A_MARITL"] in (1, 2) and person["A_SPOUSE"] > 0


def _pairs(people: list[dict[str, int]]) -> set[tuple[int, int]]:
    """Married pairs exactly as the construction forms them."""

    by_line = {person["A_LINENO"]: person for person in people}
    pairs = set()
    for person in people:
        if not _married_spouse_present(person):
            continue
        spouse = by_line.get(person["A_SPOUSE"])
        if spouse is None or spouse is person or not _married_spouse_present(spouse):
            continue
        pairs.add(tuple(sorted((person["A_LINENO"], spouse["A_LINENO"]))))
    return pairs


def _break_spouse_chains(people: list[dict[str, int]]) -> None:
    by_line = {person["A_LINENO"]: person for person in people}
    while True:
        counts = Counter(line for pair in _pairs(people) for line in pair)
        over = {line for line, count in counts.items() if count > 1}
        if not over:
            return
        for person in people:
            target = person["A_SPOUSE"]
            if (
                _married_spouse_present(person)
                and target in over
                and by_line[target]["A_SPOUSE"] != person["A_LINENO"]
            ):
                person["A_SPOUSE"] = 0
                break


def probe_cases() -> dict[str, pd.DataFrame]:
    """Households that break the construction's preconditions.

    ``spouse_chain``: line 1 names 2, 2 names 3 and 3 names 2, all married
    spouse-present, so line 2 lands in pairs (1, 2) and (2, 3) and line 1 is
    the (1, 2) pair's spouse. ``self_parent``: a disabled adult names their
    own line as a parent, passes the qualifying-child age test and claims
    their own unit.
    """

    chain = pd.DataFrame(
        [
            {"PH_SEQ": 1, "A_LINENO": 1, "A_AGE": 30, "A_MARITL": 1, "A_SPOUSE": 2},
            {"PH_SEQ": 1, "A_LINENO": 2, "A_AGE": 40, "A_MARITL": 1, "A_SPOUSE": 3},
            {"PH_SEQ": 1, "A_LINENO": 3, "A_AGE": 35, "A_MARITL": 1, "A_SPOUSE": 2},
        ]
    )
    chain["A_EXPRRP"] = [12, 1, 3]
    self_parent = pd.DataFrame(
        [
            {"PH_SEQ": 1, "A_LINENO": 1, "A_AGE": 44, "A_MARITL": 7, "A_SPOUSE": 0},
            {"PH_SEQ": 1, "A_LINENO": 2, "A_AGE": 40, "A_MARITL": 6, "A_SPOUSE": 0},
        ]
    )
    self_parent["A_EXPRRP"] = [1, 10]
    probes = {}
    for name, frame in (("spouse_chain", chain), ("self_parent", self_parent)):
        frame = frame.reindex(columns=list(CASE_COLUMNS), fill_value=0)
        frame["PEPAR1"] = frame["PEPAR2"] = -1
        probes[name] = frame
    probes["self_parent"].loc[1, "PEPAR1"] = 2
    probes["self_parent"].loc[1, "PEDISPHY"] = 1
    return {name: frame.astype("int64") for name, frame in probes.items()}


def generate_cases(households: int, seed: int) -> pd.DataFrame:
    """Return ``households`` synthetic households as CPS person rows."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, int]] = []
    for household_id in range(1, households + 1):
        rows.extend(_household(rng, household_id))
    return pd.DataFrame(rows, columns=list(CASE_COLUMNS)).astype("int64")
