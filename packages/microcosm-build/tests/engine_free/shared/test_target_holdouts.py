"""The shared hold-out passes (microcosm.build.target_holdouts, microcosm#1728).

The tests use a made-up country: regions hold districts, and a district target
of the ``tax`` family is held out with the other targets of its concept family
in its region. They check the reader, the record tests, the concept families,
the role pass and its receipt, the training target set, and the scoring
against the pro-rata baseline. No country code is read.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
from scipy import sparse

from microcosm.build.holdout import hash_holdout_unit
from microcosm.build.target_holdouts import (
    ROLE_HOLDOUT,
    ROLE_TRAIN,
    ConceptFamilyRule,
    RecordTest,
    TargetHoldoutError,
    assign_holdout_roles,
    attach_population_shares,
    error_summary,
    load_target_holdouts,
    missing_population_shares,
    pro_rata_baseline,
    score_holdout,
    training_target_set,
)


def _section() -> dict:
    return {
        "concept_families": {
            "source_field": "measure",
            "families": ["tax"],
            "merge_prefixes": ["credit"],
            "strip_suffixes": ["_amount", "_count"],
            "rationale": "Counts and amounts of one concept are one family.",
        },
        "passes": [
            {
                "id": "district",
                "label": "District",
                "unit": "region_x_family",
                "unit_fields": ["region", "concept_family"],
                "salt": "test.district.v1",
                "default_fraction": 0.25,
                "max_fraction": 0.5,
                "scope": [],
                "eligible": [
                    {"key": "geography_level", "kind": "equals", "value": "district"},
                    {"key": "parent", "kind": "truthy"},
                ],
                "receipt": ["unit", "fraction", "held_units", "hash"],
                "hash_description": "a salted hash",
                "scoring": {
                    "level": "district",
                    "parent_name_field": "parent",
                    "child_population_field": "child_pop",
                    "parent_population_field": "parent_pop",
                    "population_check": "nonzero",
                    "populations": {
                        "families": ["tax"],
                        "code_field": "code",
                        "parent_code_divisor": 100,
                    },
                    "by_family": True,
                    "target_fields": {"unit": "holdout_unit"},
                    "baseline": "parent x population share",
                },
                "rationale": "Whole blocks, so no sibling pins a held row.",
            },
            {
                "id": "ward",
                "label": "Ward",
                "unit": "region_x_family",
                "unit_fields": ["region", "concept_family"],
                "salt": "test.ward.v1",
                "default_fraction": 0.1,
                "max_fraction": 0.5,
                "scope": [
                    {"key": "geography_level", "kind": "equals", "value": "ward"}
                ],
                "eligible": [{"key": "parent", "kind": "truthy"}],
                "receipt": ["unit", "eligible_targets", "held_targets"],
                "hash_description": None,
                "scoring": None,
                "rationale": "Ward rows only.",
            },
        ],
    }


def _records() -> list[dict]:
    records = [
        {
            "name": "r1_income",
            "geography_level": "region",
            "family": "tax",
            "value": 100.0,
        },
        {
            "name": "r2_income",
            "geography_level": "region",
            "family": "tax",
            "value": 80.0,
        },
    ]
    for region, parent in (("1", "r1_income"), ("2", "r2_income")):
        for district in range(1, 5):
            for measure in ("income_amount", "income_count", "credit_x", "other"):
                records.append(
                    {
                        "name": f"d{region}{district}_{measure}",
                        "geography_level": "district",
                        "family": "tax",
                        "measure": measure,
                        "region": region,
                        "code": f"{region}{district:02d}",
                        "parent": parent if measure != "other" else "",
                        "value": float(10 * district),
                    }
                )
    records.append(
        {
            "name": "w1",
            "geography_level": "ward",
            "family": "pop",
            "region": "1",
            "value": 7.0,
        }
    )
    return records


def test_the_reader_reads_a_declaration() -> None:
    holdouts = load_target_holdouts(_section())
    district = holdouts.holdout_pass("district")
    assert district.unit_fields == ("region", "concept_family")
    assert district.eligible[1] == RecordTest("parent", "truthy")
    assert district.scoring is not None
    assert district.scoring.target_fields == (("unit", "holdout_unit"),)
    assert holdouts.holdout_pass("ward").scoring is None
    with pytest.raises(TargetHoldoutError, match="No hold-out pass"):
        holdouts.holdout_pass("county")


def _mutations():
    def drop(path, key):
        def mutate(section):
            target = section
            for step in path:
                target = target[step]
            del target[key]

        return mutate

    def set_(path, key, value):
        def mutate(section):
            target = section
            for step in path:
                target = target[step]
            target[key] = value

        return mutate

    first = ("passes", 0)
    return {
        "unknown top key": set_((), "extra", 1),
        "missing passes": drop((), "passes"),
        "empty passes": set_((), "passes", []),
        "unknown pass key": set_(first, "extra", 1),
        "missing salt": drop(first, "salt"),
        "empty salt": set_(first, "salt", ""),
        "fraction above one": set_(first, "max_fraction", 1.5),
        "default above max": set_(first, "default_fraction", 0.75),
        "boolean fraction": set_(first, "default_fraction", True),
        "unknown receipt key": set_(first, "receipt", ["unit", "nope"]),
        "hash without description": set_(first, "hash_description", None),
        "unknown test kind": set_(first, "eligible", [{"key": "a", "kind": "near"}]),
        "equals without value": set_(
            first, "eligible", [{"key": "a", "kind": "equals"}]
        ),
        "truthy with value": set_(
            first, "eligible", [{"key": "a", "kind": "truthy", "value": 1}]
        ),
        "empty unit fields": set_(first, "unit_fields", []),
        "bad population check": set_(
            (*first, "scoring"), "population_check", "sometimes"
        ),
        "bad divisor": set_(
            (*first, "scoring", "populations"), "parent_code_divisor", 0
        ),
        "bad target fields": set_((*first, "scoring"), "target_fields", ["unit"]),
        "repeated pass id": set_(("passes", 1), "id", "district"),
        "bad family list": set_(("concept_families",), "families", "tax"),
    }


@pytest.mark.parametrize("case", sorted(_mutations()))
def test_the_reader_refuses_a_malformed_declaration(case: str) -> None:
    section = _section()
    _mutations()[case](section)
    with pytest.raises(TargetHoldoutError):
        load_target_holdouts(section)


def test_record_tests() -> None:
    equals = RecordTest("level", "equals", "district")
    truthy = RecordTest("parent", "truthy")
    assert equals.accepts({"level": "district"})
    assert not equals.accepts({"level": "region"})
    assert not equals.accepts({})
    assert truthy.accepts({"parent": "r1"})
    for value in ("", 0, None, []):
        assert not truthy.accepts({"parent": value})
    assert not truthy.accepts({})


def test_concept_families() -> None:
    rule = ConceptFamilyRule("measure", frozenset({"tax"}), ("credit",), ("_amount",))
    assert rule.of_measure("credit_one_child_count") == "credit"
    assert rule.of_measure("income_amount") == "income"
    assert rule.of_measure("income_count") == "income_count"
    assert rule.of_record({"family": "tax", "measure": "income_amount"}) == "income"
    assert rule.of_record({"family": "pop", "measure": "income_amount"}) == "pop"


def test_the_role_pass() -> None:
    holdouts = load_target_holdouts(_section())
    district = holdouts.holdout_pass("district")
    records = _records()
    receipt = assign_holdout_roles(records, district, holdouts.concept_families)

    assert list(receipt) == ["unit", "fraction", "held_units", "hash"]
    assert receipt["fraction"] == 0.25
    assert receipt["hash"] == "a salted hash"
    units = {}
    for record in records:
        assert list(record)[-2:] == ["holdout_unit", "role"]
        eligible = record["geography_level"] == "district" and bool(
            record.get("parent")
        )
        if not eligible:
            assert (record["holdout_unit"], record["role"]) == (None, ROLE_TRAIN)
            continue
        family = {
            "income_amount": "income",
            "income_count": "income",
            "credit_x": "credit",
        }[record["measure"]]
        assert record["holdout_unit"] == f"{record['region']}|{family}"
        held = hash_holdout_unit(
            record["holdout_unit"], fraction=0.25, salt="test.district.v1"
        )
        assert record["role"] == (ROLE_HOLDOUT if held else ROLE_TRAIN)
        units[record["holdout_unit"]] = held
    assert receipt["held_units"] == sum(units.values())

    # The ward pass writes ward rows only; district rows keep their roles.
    before = copy.deepcopy(records)
    ward = assign_holdout_roles(
        records, holdouts.holdout_pass("ward"), holdouts.concept_families
    )
    assert records[:-1] == before[:-1]
    assert records[-1]["holdout_unit"] is None
    assert ward == {"unit": "region_x_family", "eligible_targets": 0, "held_targets": 0}


def test_the_role_pass_refuses_a_bad_fraction_or_salt() -> None:
    holdouts = load_target_holdouts(_section())
    district = holdouts.holdout_pass("district")
    for fraction in (-0.01, 0.51):
        with pytest.raises(TargetHoldoutError, match="fraction must be in"):
            assign_holdout_roles(
                _records(), district, holdouts.concept_families, fraction=fraction
            )
    with pytest.raises(TargetHoldoutError, match="salt"):
        assign_holdout_roles(_records(), district, holdouts.concept_families, salt="")


def _scored():
    holdouts = load_target_holdouts(_section())
    district = holdouts.holdout_pass("district")
    records = _records()
    assign_holdout_roles(records, district, holdouts.concept_families, fraction=0.5)
    attach_population_shares(
        records,
        {101: 10.0, 102: 30.0, 103: 20.0, 104: 40.0, 201: 5.0},
        district.scoring,
    )
    return records, district.scoring


def test_population_shares() -> None:
    records, scoring = _scored()
    by_name = {record["name"]: record for record in records}
    district = by_name["d12_income_amount"]
    assert (district["child_pop"], district["parent_pop"]) == (30.0, 100.0)
    # Districts the ladder does not know get no population and are named.
    missing = missing_population_shares(records, scoring)
    assert "d22_income_amount" in missing
    assert "d12_income_amount" not in missing
    assert "r1_income" not in by_name["r1_income"]


def test_the_training_target_set_keeps_training_rows_only() -> None:
    records, _scoring = _scored()
    for record in records:
        record.setdefault("period", 2024)
        record.setdefault("source", "test")
    n = 4
    matrix = sparse.csr_array(
        np.arange(len(records) * n, dtype=np.float32).reshape(len(records), n)
    )
    target_set = training_target_set(records, matrix, n)
    trained = [record["name"] for record in records if record["role"] == ROLE_TRAIN]
    assert [target.name for target in target_set.targets] == trained
    first = target_set.targets[0]
    row = [record["name"] for record in records].index(first.name)
    assert np.array_equal(first.measure(None), matrix.toarray()[row].astype(np.float64))
    with pytest.raises(ValueError, match="does not match"):
        training_target_set(records, matrix, n + 1)
    with pytest.raises(ValueError, match="row-aligned"):
        training_target_set(records, matrix, n, specs=[])


def test_scoring_against_the_pro_rata_baseline() -> None:
    records, scoring = _scored()
    held = [
        index
        for index, record in enumerate(records)
        if record["role"] == ROLE_HOLDOUT and record.get("child_pop")
    ]
    assert held, "the fixture must hold out some covered rows"
    keep = {records[index]["name"] for index in held}
    # Rows the ladder does not cover cannot be scored; train them here.
    for record in records:
        if record["role"] == ROLE_HOLDOUT and record["name"] not in keep:
            record["role"] = ROLE_TRAIN
    n = 3
    rng = np.random.default_rng(0)
    matrix = sparse.csr_array(rng.uniform(0, 2, size=(len(records), n)))
    design, final = np.ones(n), rng.uniform(0.5, 1.5, size=n)
    result = score_holdout(
        records,
        matrix,
        scoring,
        design_weights=design,
        final_weights=final,
        cap=10.0,
        receipt={"unit": "region_x_family"},
    )
    assert list(result)[:3] == ["report_only", "unit", "n_targets"]
    assert result["n_targets"] == len(held)
    assert list(result)[-2:] == ["by_family", "targets"]
    first = result["targets"][0]
    assert list(first) == [
        "name",
        "unit",
        "target",
        "calibrated",
        "design",
        "pro_rata",
    ]
    record = records[held[0]]
    parent = next(r for r in records if r["name"] == record["parent"])
    assert (
        first["pro_rata"]
        == parent["value"] * record["child_pop"] / record["parent_pop"]
    )
    json.dumps(result)

    for record in records:
        record["role"] = ROLE_TRAIN
    empty = score_holdout(
        records, matrix, scoring, design_weights=design, final_weights=final, cap=10.0
    )
    assert empty == {"report_only": True, "n_targets": 0}


def test_the_population_checks() -> None:
    _records_, scoring = _scored()
    parent = {"name": "p", "value": 10.0}
    row = {"name": "c", "parent": "p", "child_pop": float("nan"), "parent_pop": 4.0}
    nonzero = pro_rata_baseline([parent, row], [1], scoring)
    assert math.isnan(nonzero[0])
    strict = type(scoring)(
        **{**scoring.__dict__, "population_check": "finite_positive"}
    )
    with pytest.raises(TargetHoldoutError):
        pro_rata_baseline([parent, row], [1], strict)
    row["child_pop"] = 0.0
    with pytest.raises(TargetHoldoutError):
        pro_rata_baseline([parent, row], [1], scoring)
    row["child_pop"], row["parent"] = 2.0, "missing"
    with pytest.raises(TargetHoldoutError):
        pro_rata_baseline([parent, row], [1], scoring)


def test_error_summary() -> None:
    summary = error_summary([1.0, 2.0, 4.0], [1.0, 1.0, 2.0], cap=10.0)
    assert summary["max_abs_rel_error"] == 1.0
    assert summary["fraction_within_10pct"] == pytest.approx(1 / 3)
