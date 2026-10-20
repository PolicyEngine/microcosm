#!/usr/bin/env python
"""Diagnose the SCF-vs-CPS-host gap of the education-debt stage (microcosm#1073).

The first receipt imputed 14.4% of CPS host households with federal education
debt against 18.2% in SCF 2025 (17.3% in the SCF hold-out), short mostly at
reference-person ages 25-49 and income quintiles 2-5, and a Parent PLUS share
of 21.5% against 13.8%. A QRF reproduces its donor's conditional distribution,
so a host-only gap means the host's predictors are distributed or coded
differently from the donor's. This script measures that, weighted, before any
change:

1. who the reference person is (SCF: the economically dominant individual or
   couple, then the male or older partner; CPS candidates: the current
   oldest-adult rule and the householder-anchored rule);
2. the predictors by reference-person age band: married-or-partnered (SCF
   ``married`` counts partners), children of the reference person or partner
   (SCF ``kids``), race, and income summed over candidate CPS units (household,
   family, SPM unit, couple plus own children, couple only), against the SCF
   primary economic unit's ``wageinc`` and ``income - kginc``.

    .venv/Scripts/python.exe experiments/diagnose_1073_host_fit.py \
        --h5 populace_us_2024.h5
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from microcosm.build.us_runtime.scf_education_debt import (
    _RECIPIENT_HOUSEHOLD_INCOME_COLUMNS,
    load_scf_2025_education_debt_donor,
)

AGE_BANDS = (("18_to_24", 18, 24), ("25_to_34", 25, 34), ("35_to_49", 35, 49),
             ("50_to_61", 50, 61), ("62_plus", 62, 200))  # fmt: skip
_HOST_COLUMNS = (
    "person_household_id",
    "person_family_id",
    "person_spm_unit_id",
    "age",
    "is_female",
    "is_household_head",
    "A_LINENO",
    "A_SPOUSE",
    "PECOHAB",
    "A_MARITL",
    "PEPAR1",
    "PEPAR2",
    "PRDTRACE",
    "PRDTHSP",
    "employment_income_before_lsr",
)


def wq(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="mergesort")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, q * cumulative[-1], side="left"))
    return float(values[order][min(index, len(values) - 1)])


def summarize(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        "p25": wq(values, weights, 0.25),
        "p50": wq(values, weights, 0.50),
        "p75": wq(values, weights, 0.75),
        "p90": wq(values, weights, 0.90),
        "mean": float(np.average(values, weights=weights)),
    }


def share(flag: np.ndarray, weights: np.ndarray) -> float:
    return float(weights[flag].sum() / weights.sum())


def load_host(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    with pd.HDFStore(path, "r") as store:
        available = set(store.get_storer("/person").data_columns)
        columns = [c for c in _HOST_COLUMNS if c in available] + [
            c for c in _RECIPIENT_HOUSEHOLD_INCOME_COLUMNS if c in available
        ]
        person = store.select("/person", columns=columns).reset_index(drop=True)
        household = store.select(
            "/household", columns=["household_id", "household_weight"]
        )
    weights = pd.Series(
        household["household_weight"].to_numpy(dtype=float),
        index=household["household_id"].to_numpy(),
    )
    return person, weights


def _num(series: pd.Series, fill: float = 0.0) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").fillna(fill).to_numpy(dtype=float)


def host_units(person: pd.DataFrame) -> pd.DataFrame:
    """One row per household: reference person under two rules, and units."""

    hh = person["person_household_id"].to_numpy()
    line = _num(person["A_LINENO"], -1).astype(int)
    age = _num(person["age"])
    female = person["is_female"].astype(bool).to_numpy()
    lookup = {(h, n_): i for i, (h, n_) in enumerate(zip(hh, line, strict=True))}
    spouse_ptr = _num(person["A_SPOUSE"]).astype(int)
    partner_ptr = _num(person["PECOHAB"]).astype(int)
    married = np.isin(_num(person["A_MARITL"]).astype(int), (1, 2))

    def partner_of(i: int) -> int:
        for pointer in (spouse_ptr[i], partner_ptr[i]):
            if pointer > 0:
                j = lookup.get((hh[i], int(pointer)))
                if j is not None and j != i:
                    return j
        return -1

    # Current rule (scf_auto_loans._scf_reference_person_mask): the only adult,
    # the man of a two-adult married couple (older if same sex), else the
    # oldest adult. Householder rule: the CPS householder (line 1) or, when the
    # householder lives with a spouse or partner, the SCF's man-or-older of the
    # two.
    groups = pd.Series(np.arange(len(person))).groupby(hh, sort=False)
    rows = []
    for household_id, positions in groups:
        idx = positions.to_numpy()
        adults = idx[age[idx] >= 18]
        if len(adults) == 0:
            current = idx[np.argmax(age[idx])]
        elif len(adults) == 1:
            current = adults[0]
        elif len(adults) == 2 and married[adults].all():
            if female[adults[0]] == female[adults[1]]:
                current = adults[np.argmax(age[adults])]
            else:
                current = adults[~female[adults]][0]
        else:
            current = adults[np.argmax(age[adults])]
        householder = idx[np.argmin(line[idx])]
        partner = partner_of(householder)
        anchored = householder
        if partner >= 0:
            pair = np.array([householder, partner])
            if female[pair[0]] != female[pair[1]]:
                anchored = pair[~female[pair]][0]
            else:
                anchored = pair[np.argmax(age[pair])]
        rows.append((household_id, current, anchored, householder))
    return pd.DataFrame(
        rows, columns=["household_id", "current", "anchored", "householder"]
    )


def partner_rows(person: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    """Positional row of each given row's spouse (A_SPOUSE) or partner (PECOHAB)."""

    hh = person["person_household_id"].to_numpy()
    line = _num(person["A_LINENO"], -1).astype(int)
    lookup = {(h, n_): i for i, (h, n_) in enumerate(zip(hh, line, strict=True))}
    pointers = (
        _num(person["A_SPOUSE"]).astype(int),
        _num(person["PECOHAB"]).astype(int),
    )
    out = np.full(len(rows), -1, dtype=np.int64)
    for k, r in enumerate(rows):
        for ptr in pointers:
            if ptr[r] > 0:
                j = lookup.get((hh[r], int(ptr[r])))
                if j is not None and j != r:
                    out[k] = j
                    break
    return out


def unit_income(
    person: pd.DataFrame,
    reference: np.ndarray,
    unit: str,
    column_sets: Mapping[str, Sequence[str]],
) -> dict[str, np.ndarray]:
    """Income sums over a candidate unit around each household's reference person."""

    hh = person["person_household_id"].to_numpy()
    line = _num(person["A_LINENO"], -1).astype(int)
    {(h, n_): i for i, (h, n_) in enumerate(zip(hh, line, strict=True))}
    n = len(person)
    member = np.zeros(n, dtype=bool)
    owner = np.full(n, -1, dtype=np.int64)  # household row index of the unit
    ref_rows = reference
    if unit == "household":
        member[:] = True
        index_of = pd.Series(np.arange(len(ref_rows)), index=hh[ref_rows])
        owner = index_of.reindex(hh).to_numpy()
    else:
        ref_set = {}
        for k, r in enumerate(ref_rows):
            ref_set[r] = k
        partner = partner_rows(person, ref_rows)
        if unit in ("family", "spm_unit"):
            key = "person_family_id" if unit == "family" else "person_spm_unit_id"
            ids = person[key].to_numpy()
            unit_of_ref = pd.Series(np.arange(len(ref_rows)), index=ids[ref_rows])
            unit_of_ref = unit_of_ref[~unit_of_ref.index.duplicated()]
            owner = unit_of_ref.reindex(ids).to_numpy(dtype=float)
            owner = np.where(np.isnan(owner), -1, owner).astype(np.int64)
            # a unit id can belong to another household's reference person only
            # if ids repeat across households; guard on household
            same = owner >= 0
            same[same] = hh[same] == hh[ref_rows[owner[same]]]
            owner = np.where(same, owner, -1)
            member = owner >= 0
            if unit == "family":
                # a cohabiting partner is a separate CPS family; SCF counts
                # partners in the PEU, so add the partner explicitly
                for k, j in enumerate(partner):
                    if j >= 0 and owner[j] < 0:
                        owner[j] = k
                        member[j] = True
        else:
            for k, r in enumerate(ref_rows):
                owner[r] = k
                if partner[k] >= 0:
                    owner[partner[k]] = k
            if unit == "couple_and_children":
                parents = {(hh[r], line[r]): k for k, r in enumerate(ref_rows)}
                for k, j in enumerate(partner):
                    if j >= 0:
                        parents[(hh[j], line[j])] = k
                for pointer in ("PEPAR1", "PEPAR2"):
                    ptr = _num(person[pointer]).astype(int)
                    for i in range(n):
                        if owner[i] < 0 and ptr[i] > 0:
                            k = parents.get((hh[i], ptr[i]))
                            if k is not None:
                                owner[i] = k
            member = owner >= 0
    out = {}
    for name, columns in column_sets.items():
        values = np.zeros(n)
        for column in columns:
            if column in person.columns:
                values += _num(person[column])
        if name == "employment_income":
            values = np.maximum(values, 0.0)
        sums = np.zeros(len(ref_rows))
        np.add.at(sums, owner[member], values[member])
        out[name] = sums
    return out


def children_counts(
    person: pd.DataFrame, reference: np.ndarray
) -> dict[str, np.ndarray]:
    hh = person["person_household_id"].to_numpy()
    line = _num(person["A_LINENO"], -1).astype(int)
    {(h, n_): i for i, (h, n_) in enumerate(zip(hh, line, strict=True))}
    age = _num(person["age"])
    ref_lines = {(hh[r], line[r]): k for k, r in enumerate(reference)}
    partners = partner_rows(person, reference)
    partner_lines = {(hh[j], line[j]): k for k, j in enumerate(partners) if j >= 0}
    own = np.zeros(len(reference))
    either = np.zeros(len(reference))
    either_under_18 = np.zeros(len(reference))
    ptr1 = _num(person["PEPAR1"]).astype(int)
    ptr2 = _num(person["PEPAR2"]).astype(int)
    for i in range(len(person)):
        keys = [(hh[i], p) for p in (ptr1[i], ptr2[i]) if p > 0]
        if not keys:
            continue
        k_ref = next((ref_lines[x] for x in keys if x in ref_lines), None)
        k_any = (
            k_ref
            if k_ref is not None
            else next((partner_lines[x] for x in keys if x in partner_lines), None)
        )
        if k_ref is not None:
            own[k_ref] += 1
        if k_any is not None:
            either[k_any] += 1
            if age[i] < 18:
                either_under_18[k_any] += 1
    return {"own_of_reference": own, "of_reference_or_partner": either,
            "of_reference_or_partner_under_18": either_under_18}  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, required=True)
    parser.add_argument(
        "--scf-cache-dir", type=Path, default=Path.home() / ".cache/microcosm/scf"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/receipts/1073-host-fit-diagnosis.json"),
    )
    args = parser.parse_args(argv)

    donor = load_scf_2025_education_debt_donor(
        args.scf_cache_dir / "rscfp2025.dta", args.scf_cache_dir / "p25i6.dta"
    )
    scf_w = donor["scf_weight"].to_numpy(float)
    scf_age = donor["age"].to_numpy(float)

    person, hh_weights = load_host(args.h5)
    units = host_units(person)
    weights = hh_weights.reindex(units["household_id"]).to_numpy(float)
    age = _num(person["age"])
    report: dict[str, object] = {}

    # 1. Who is the reference person, and how old.
    report["reference_person_age_bands"] = {
        "scf": {
            b: share((scf_age >= lo) & (scf_age <= hi), scf_w)
            for b, lo, hi in AGE_BANDS
        },
        **{
            f"cps_{rule}": {
                b: share((age[units[rule]] >= lo) & (age[units[rule]] <= hi), weights)
                for b, lo, hi in AGE_BANDS
            }
            for rule in ("current", "anchored")
        },
    }
    report["rules_disagree_share"] = share(
        units["current"].to_numpy() != units["anchored"].to_numpy(), weights
    )

    column_sets = {
        "employment_income": ("employment_income_before_lsr",),
        "household_income": _RECIPIENT_HOUSEHOLD_INCOME_COLUMNS,
    }
    for rule in ("current", "anchored"):
        reference = units[rule].to_numpy()
        ref_age = age[reference]
        has_partner = partner_rows(person, reference) >= 0
        married_code = np.isin(_num(person["A_MARITL"]).astype(int)[reference], (1, 2))
        kids = children_counts(person, reference)
        incomes = {
            unit: unit_income(person, reference, unit, column_sets)
            for unit in (
                "household",
                "spm_unit",
                "family",
                "couple_and_children",
                "couple",
            )
        }
        by_band = {}
        for band, lo, hi in AGE_BANDS:
            host = (ref_age >= lo) & (ref_age <= hi)
            scf = (scf_age >= lo) & (scf_age <= hi)
            entry = {
                "scf_married_or_partnered": share(
                    donor["is_married"].to_numpy()[scf] == 1, scf_w[scf]
                ),
                "cps_A_MARITL_1_2": share(married_code[host], weights[host]),
                "cps_spouse_or_partner_present": share(
                    has_partner[host], weights[host]
                ),
                "scf_kids_positive": share(
                    donor["own_children_in_household"].to_numpy()[scf] > 0,
                    scf_w[scf],
                ),
                **{
                    f"cps_kids_{k}_positive": share(v[host] > 0, weights[host])
                    for k, v in kids.items()
                },
                "scf_employment_income": summarize(
                    donor["employment_income"].to_numpy()[scf], scf_w[scf]
                ),
                "scf_household_income": summarize(
                    donor["household_income"].to_numpy()[scf], scf_w[scf]
                ),
            }
            for unit, sums in incomes.items():
                for name, values in sums.items():
                    entry[f"cps_{name}_{unit}"] = summarize(values[host], weights[host])
            by_band[band] = entry
        report[f"predictors_by_reference_age_{rule}_rule"] = by_band

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Console digest
    print("ref age bands", json.dumps(report["reference_person_age_bands"], indent=0))
    print("rules disagree", report["rules_disagree_share"])
    for rule in ("current", "anchored"):
        for band in ("25_to_34", "35_to_49", "50_to_61"):
            e = report[f"predictors_by_reference_age_{rule}_rule"][band]
            print(f"\n[{rule}] {band}")
            for key in (
                "scf_married_or_partnered",
                "cps_A_MARITL_1_2",
                "cps_spouse_or_partner_present",
                "scf_kids_positive",
                "cps_kids_own_of_reference_positive",
                "cps_kids_of_reference_or_partner_positive",
            ):
                print(f"  {key}: {e[key]:.3f}")
            for name in ("employment_income", "household_income"):
                s = e[f"scf_{name}"]
                print(
                    f"  scf {name}: p25 {s['p25']:,.0f} p50 {s['p50']:,.0f} p75 {s['p75']:,.0f} p90 {s['p90']:,.0f}"
                )
                for unit in (
                    "household",
                    "spm_unit",
                    "family",
                    "couple_and_children",
                    "couple",
                ):
                    s = e[f"cps_{name}_{unit}"]
                    print(
                        f"  cps {name} {unit}: p25 {s['p25']:,.0f} p50 {s['p50']:,.0f} p75 {s['p75']:,.0f} p90 {s['p90']:,.0f}"
                    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
