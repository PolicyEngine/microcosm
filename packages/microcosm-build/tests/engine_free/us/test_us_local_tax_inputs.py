"""US local-tax inputs: registry pin, derivation properties and gates.

Invariants exercised here (stated for every input, then checked by
Hypothesis):

- a Philadelphia resident's taxable wages are ``max(wages, 0)`` and every
  other person's are 0, so ``0 <= taxable <= max(wages, 0)``, taxable wages
  never fall when wages rise, and a nonresident's are 0 whatever they earn;
- derivation is deterministic, commutes with any reordering of person rows,
  and applying it twice changes nothing; the input tables are never
  modified;
- an input the registry marks unavailable is never written, and a stale
  column is removed; without the residence flag the taxable wages are not
  written either;
- the receipt's weighted taxable wages equal the person-weighted sum of the
  written column, and its counts match the rows;
- a null residence flag, an infinite wage, a missing column or (unless the
  line propagates nulls) a resident's null wage raises instead of writing 0;
- the gate accepts exactly the column the rule writes and fails any other.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.us_annual_static_aging import _rederive_local_tax_inputs
from microcosm.build.us_runtime.local_jurisdictions import (
    JURISDICTION_FLAG_DERIVED,
    JURISDICTION_FLAG_UNAVAILABLE,
    LOCAL_TAX_INPUT_RESIDENT_WAGES,
    LOCAL_TAX_INPUT_UNAVAILABLE,
    NULL_WAGES_PROPAGATE,
    NULL_WAGES_REFUSE,
    US_LOCAL_TAX_INPUT_REGISTRY_RESOURCE,
    US_LOCAL_TAX_INPUT_REGISTRY_SHA256,
    apply_us_local_tax_inputs,
    load_us_local_tax_input_registry,
    packaged_us_local_jurisdiction_registry,
    packaged_us_local_tax_input_registry,
    rederive_us_local_jurisdiction_flags,
    us_local_tax_input_failures,
    us_local_tax_input_output_columns,
    us_resident_wage_tax_values,
    with_us_local_jurisdiction_flags,
    with_us_local_tax_inputs,
)
from microcosm.build.us_runtime.operator_boundary import (
    PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
)
from microcosm.frame import US_SCHEMA
from test_support.microcosm_build.us_local_jurisdictions import us_frame
from test_support.paths import paths_for

REGISTRY_PATH = (
    paths_for("microcosm-build").package
    / "src/microcosm/build/us_runtime"
    / US_LOCAL_TAX_INPUT_REGISTRY_RESOURCE
)
REGISTRY = packaged_us_local_tax_input_registry()
TAXABLE = "pa_philadelphia_wage_tax_taxable_wages"
RESIDENT = "pa_philadelphia_wage_tax_resident"
WAGES = "employment_income_before_lsr"
TAXABLE_ITEM = next(item for item in REGISTRY.inputs if item.variable == TAXABLE)

# The decisions, written out independently of the registry file: a provision
# or basis mistyped there fails this oracle.
EXPECTED_PROVISIONS = {
    "co_denver_employee_occupational_privilege_tax_months": (
        LOCAL_TAX_INPUT_UNAVAILABLE,
        "work_location",
    ),
    "de_wilmington_nonresident_earnings": (
        LOCAL_TAX_INPUT_UNAVAILABLE,
        "work_location",
    ),
    "ny_yonkers_nonresident_earnings": (LOCAL_TAX_INPUT_UNAVAILABLE, "work_location"),
    "pa_philadelphia_wage_tax_reduced_rate_eligible": (
        LOCAL_TAX_INPUT_UNAVAILABLE,
        "refund_claim",
    ),
    TAXABLE: (LOCAL_TAX_INPUT_RESIDENT_WAGES, RESIDENT),
}
UNAVAILABLE = sorted(
    variable
    for variable, (provision, _) in EXPECTED_PROVISIONS.items()
    if provision == LOCAL_TAX_INPUT_UNAVAILABLE
)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_packaged_registry_bytes_match_the_pinned_sha256() -> None:
    payload = REGISTRY_PATH.read_bytes()

    assert hashlib.sha256(payload).hexdigest() == US_LOCAL_TAX_INPUT_REGISTRY_SHA256
    assert REGISTRY.sha256 == US_LOCAL_TAX_INPUT_REGISTRY_SHA256


def test_registry_provisions_each_input_as_decided() -> None:
    observed = {
        item.variable: (
            item.provision,
            item.residence_flag
            if item.provision == LOCAL_TAX_INPUT_RESIDENT_WAGES
            else item.basis,
        )
        for item in REGISTRY.inputs
    }

    assert observed == EXPECTED_PROVISIONS
    assert {item.entity for item in REGISTRY.inputs} == {"person"}
    assert TAXABLE_ITEM.wage_input == WAGES
    person_flags = {
        item.flag
        for item in packaged_us_local_jurisdiction_registry().for_entity("person")
    }
    assert TAXABLE_ITEM.residence_flag in person_flags


def test_only_the_resident_wage_input_is_ever_an_output_column() -> None:
    assert us_local_tax_input_output_columns() == {"person": frozenset({TAXABLE})}
    assert PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES["local_tax_inputs"] == {
        "person": frozenset({TAXABLE})
    }


def test_registry_pins_every_downloaded_source_document() -> None:
    document = json.loads(REGISTRY_PATH.read_text())
    cited = {name for item in document["inputs"] for name in item["sources"]}

    assert cited == set(document["sources"])
    pinned = [
        entry
        for source in document["sources"].values()
        for entry in source["documents"]
        if isinstance(entry, dict)
    ]
    assert pinned
    for entry in pinned:
        assert entry["url"].startswith("https://")
        assert len(entry["sha256"]) == 64
        int(entry["sha256"], 16)


def test_packaged_registry_refuses_bytes_that_differ_from_the_pin(tmp_path) -> None:
    edited = tmp_path / "registry.json"
    edited.write_bytes(REGISTRY_PATH.read_bytes() + b" ")

    with pytest.raises(ValueError, match="do not match the pinned SHA-256"):
        load_us_local_tax_input_registry(
            edited, expected_sha256=US_LOCAL_TAX_INPUT_REGISTRY_SHA256
        )


def _document() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def _write(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document))
    return path


def _edit_input(variable: str, **changes: object) -> dict:
    document = _document()
    for item in document["inputs"]:
        if item["variable"] == variable:
            for key, value in changes.items():
                if value is None:
                    item.pop(key)
                else:
                    item[key] = value
    return document


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ({**_document(), "kind": "other"}, "kind must be"),
        ({**_document(), "schema_version": 2}, "schema_version must be"),
        ({**_document(), "sources": {}}, "has no sources"),
        ({**_document(), "inputs": []}, "has no inputs"),
        (
            {**_document(), "inputs": list(reversed(_document()["inputs"]))},
            "must be sorted",
        ),
        (
            {**_document(), "inputs": _document()["inputs"][:1] * 2},
            "repeats a variable",
        ),
        (_edit_input(TAXABLE, provision="imputed"), "provision must be one of"),
        (_edit_input(TAXABLE, entity="household"), "is a person input"),
        (_edit_input(TAXABLE, entity="tax_unit"), "entity must be one of"),
        (
            _edit_input(TAXABLE, residence_flag=None),
            "is missing \\['residence_flag'\\]",
        ),
        (_edit_input(TAXABLE, basis="work_location"), "may not name \\['basis'\\]"),
        (_edit_input(TAXABLE, wage_input="Wages"), "lowercase identifier"),
        (_edit_input(TAXABLE, sources=["census_blog"]), "unknown source"),
        (_edit_input(TAXABLE, engine_consumers=[]), "must list engine variables"),
        (
            _edit_input("ny_yonkers_nonresident_earnings", basis="survey_gap"),
            "basis must be one of",
        ),
        (
            _edit_input("ny_yonkers_nonresident_earnings", residence_flag="in_yonkers"),
            "may not name \\['residence_flag'\\]",
        ),
        (
            _edit_input("ny_yonkers_nonresident_earnings", reason=""),
            "reason must be a non-empty string",
        ),
    ],
)
def test_registry_loader_refuses_malformed_registries(
    tmp_path, document: dict, message: str
) -> None:
    path = _write(tmp_path / "registry.json", document)

    with pytest.raises(ValueError, match=message):
        load_us_local_tax_input_registry(path)


def test_a_resident_wage_input_must_name_a_person_residence_flag(tmp_path) -> None:
    # in_denver is a household flag; the rule needs a person's own residence.
    registry = load_us_local_tax_input_registry(
        _write(
            tmp_path / "registry.json",
            _edit_input(TAXABLE, residence_flag="in_denver"),
        )
    )
    person = pd.DataFrame({"in_denver": [True], WAGES: [1.0]})

    with pytest.raises(ValueError, match="are not person flags"):
        apply_us_local_tax_inputs(
            {"person": person, "household": pd.DataFrame()}, registry=registry
        )


# --------------------------------------------------------------------------- #
# Derivation properties
# --------------------------------------------------------------------------- #

_WAGE = st.one_of(
    st.floats(min_value=-1e7, max_value=1e9, allow_nan=False, allow_infinity=False),
    st.sampled_from([0.0, -0.0, 1.0, -1.0, 50_000.0]),
)
_PEOPLE = st.lists(st.tuples(st.booleans(), _WAGE), min_size=1, max_size=40)


def _person(rows: list[tuple[bool, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "person_id": np.arange(1, len(rows) + 1),
            "person_household_id": np.arange(1, len(rows) + 1),
            RESIDENT: [resident for resident, _ in rows],
            WAGES: [wages for _, wages in rows],
        }
    )


def _tables(person: pd.DataFrame) -> dict[str, pd.DataFrame]:
    household = pd.DataFrame({"household_id": person["person_household_id"].unique()})
    return {"person": person, "household": household}


@settings(max_examples=200, deadline=None)
@given(_PEOPLE)
def test_taxable_wages_are_a_residents_nonnegative_wages(rows) -> None:
    person = _person(rows)

    values = us_resident_wage_tax_values(person, TAXABLE_ITEM)

    for (resident, wages), value in zip(rows, values, strict=True):
        assert value == (max(wages, 0.0) if resident else 0.0)
        assert 0.0 <= value <= max(wages, 0.0)


@settings(max_examples=200, deadline=None)
@given(
    _PEOPLE,
    st.lists(
        st.floats(min_value=0, max_value=1e8, allow_nan=False), min_size=40, max_size=40
    ),
)
def test_taxable_wages_never_fall_when_wages_rise(rows, raises) -> None:
    person = _person(rows)
    higher = person.assign(**{WAGES: person[WAGES] + np.asarray(raises[: len(rows)])})

    low = us_resident_wage_tax_values(person, TAXABLE_ITEM)
    high = us_resident_wage_tax_values(higher, TAXABLE_ITEM)

    assert (high >= low).all()
    # Nonresidents owe nothing whatever they earn.
    assert (high[~person[RESIDENT].to_numpy()] == 0).all()


@settings(max_examples=100, deadline=None)
@given(_PEOPLE, st.randoms(use_true_random=False))
def test_derivation_commutes_with_person_row_order(rows, random) -> None:
    person = _person(rows)
    order = list(range(len(person)))
    random.shuffle(order)
    shuffled = person.iloc[order].reset_index(drop=True)

    written, _ = apply_us_local_tax_inputs(_tables(person))
    shuffled_written, _ = apply_us_local_tax_inputs(_tables(shuffled))

    np.testing.assert_array_equal(
        shuffled_written["person"][TAXABLE].to_numpy(),
        written["person"][TAXABLE].to_numpy()[order],
    )


@settings(max_examples=100, deadline=None)
@given(_PEOPLE)
def test_applying_twice_changes_nothing_and_leaves_the_inputs_untouched(rows) -> None:
    person = _person(rows)
    before = person.copy()

    once, receipt = apply_us_local_tax_inputs(_tables(person))
    twice, again = apply_us_local_tax_inputs(once)

    pd.testing.assert_frame_equal(person, before)
    pd.testing.assert_frame_equal(twice["person"], once["person"])
    assert again == receipt
    assert us_local_tax_input_failures(once["person"], require_derivable=True) == []


@settings(max_examples=100, deadline=None)
@given(_PEOPLE, st.lists(st.floats(min_value=0, max_value=1e6), min_size=40))
def test_receipt_counts_and_weighted_sums_match_the_written_rows(rows, weights) -> None:
    person = _person(rows)
    household_weights = np.asarray(weights[: len(rows)])

    written, receipt = apply_us_local_tax_inputs(
        _tables(person), household_weights=household_weights
    )

    entry = receipt["inputs"][TAXABLE]
    values = written["person"][TAXABLE].to_numpy()
    resident = person[RESIDENT].to_numpy()
    wages = person[WAGES].to_numpy()
    assert entry["status"] == JURISDICTION_FLAG_DERIVED
    assert entry["residents"] == int(resident.sum())
    assert entry["residents_with_taxable_wages"] == int((values > 0).sum())
    assert entry["residents_with_negative_wages_set_to_zero"] == int(
        (resident & (wages < 0)).sum()
    )
    assert entry["residents_with_null_wages_left_null"] == 0
    assert entry["weighted_residents"] == pytest.approx(
        household_weights[resident].sum()
    )
    assert entry["weighted_taxable_wages"] == pytest.approx(household_weights @ values)
    assert entry["nonresident_component"]["status"] == JURISDICTION_FLAG_UNAVAILABLE
    assert receipt["registry_sha256"] == US_LOCAL_TAX_INPUT_REGISTRY_SHA256


def test_person_weights_follow_each_persons_household() -> None:
    # Two residents share household 7 and one lives alone in household 3;
    # the household table lists them in another order.
    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3],
            "person_household_id": [7, 7, 3],
            RESIDENT: [True, True, True],
            WAGES: [10.0, 20.0, 40.0],
        }
    )
    household = pd.DataFrame({"household_id": [3, 7]})

    _, receipt = apply_us_local_tax_inputs(
        {"person": person, "household": household},
        household_weights=np.asarray([2.0, 5.0]),
    )

    entry = receipt["inputs"][TAXABLE]
    assert entry["weighted_residents"] == 5.0 + 5.0 + 2.0
    assert entry["weighted_taxable_wages"] == 5.0 * 30.0 + 2.0 * 40.0


def test_household_weights_must_align_with_the_household_table() -> None:
    person = _person([(True, 1.0), (False, 2.0)])

    with pytest.raises(ValueError, match="must align with the household table"):
        apply_us_local_tax_inputs(_tables(person), household_weights=np.ones(3))


def test_unavailable_inputs_are_never_written_and_stale_columns_are_removed() -> None:
    person = _person([(True, 100.0), (False, 200.0)]).assign(
        **{variable: 1.0 for variable in UNAVAILABLE}
    )

    written, receipt = apply_us_local_tax_inputs(_tables(person))

    assert not set(UNAVAILABLE) & set(written["person"])
    assert written["person"][TAXABLE].tolist() == [100.0, 0.0]
    for variable in UNAVAILABLE:
        entry = receipt["inputs"][variable]
        assert entry["status"] == JURISDICTION_FLAG_UNAVAILABLE
        assert entry["basis"] == EXPECTED_PROVISIONS[variable][1]
        assert entry["reason"]
    # The caller's table still holds what it held.
    assert set(UNAVAILABLE) <= set(person)


def test_without_the_residence_flag_the_taxable_wages_are_not_written() -> None:
    person = _person([(True, 100.0)]).drop(columns=[RESIDENT]).assign(**{TAXABLE: 5.0})

    written, receipt = apply_us_local_tax_inputs(_tables(person))

    assert TAXABLE not in written["person"]
    entry = receipt["inputs"][TAXABLE]
    assert entry["status"] == JURISDICTION_FLAG_UNAVAILABLE
    assert f"wrote no {RESIDENT}" in entry["reason"]


def test_other_tables_pass_through_unchanged() -> None:
    tables = {**_tables(_person([(True, 1.0)])), "tax_unit": pd.DataFrame({"a": [1]})}

    written, _ = apply_us_local_tax_inputs(tables)

    assert written["tax_unit"] is tables["tax_unit"]
    assert written["household"] is tables["household"]


# --------------------------------------------------------------------------- #
# Refusals and null handling
# --------------------------------------------------------------------------- #


def test_a_residents_null_wage_raises_unless_the_line_propagates_it() -> None:
    person = _person([(True, 1.0), (True, 2.0), (False, 3.0)])
    person.loc[[1, 2], WAGES] = np.nan

    with pytest.raises(ValueError, match="null on 1 resident row"):
        apply_us_local_tax_inputs(_tables(person))
    written, receipt = apply_us_local_tax_inputs(
        _tables(person), null_wages=NULL_WAGES_PROPAGATE
    )

    values = written["person"][TAXABLE].to_numpy()
    # The resident's null stays null; a nonresident's null wage still gives 0.
    assert values[0] == 1.0
    assert np.isnan(values[1])
    assert values[2] == 0.0
    entry = receipt["inputs"][TAXABLE]
    assert entry["null_wages"] == NULL_WAGES_PROPAGATE
    assert entry["residents_with_null_wages_left_null"] == 1


def test_a_nonresidents_null_wage_is_decided_in_either_mode() -> None:
    person = _person([(False, 1.0)])
    person[WAGES] = np.nan

    for mode in (NULL_WAGES_REFUSE, NULL_WAGES_PROPAGATE):
        assert us_resident_wage_tax_values(person, TAXABLE_ITEM, null_wages=mode) == [
            0.0
        ]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({WAGES: [np.inf]}, "1 infinite row"),
        ({WAGES: [-np.inf]}, "1 infinite row"),
        ({RESIDENT: [None]}, "1 null row"),
        ({RESIDENT: ["yes"]}, "non-boolean"),
    ],
)
def test_undecidable_rows_raise_instead_of_writing_zero(change, message) -> None:
    person = _person([(True, 1.0)]).astype({RESIDENT: object}).assign(**change)

    for mode in (NULL_WAGES_REFUSE, NULL_WAGES_PROPAGATE):
        with pytest.raises(ValueError, match=message):
            us_resident_wage_tax_values(person, TAXABLE_ITEM, null_wages=mode)


def test_a_missing_wage_column_raises() -> None:
    person = _person([(True, 1.0)]).drop(columns=[WAGES])

    with pytest.raises(ValueError, match=f"no '{WAGES}'"):
        apply_us_local_tax_inputs(_tables(person))


def test_an_unknown_null_mode_or_an_unavailable_item_raises() -> None:
    person = _person([(True, 1.0)])
    unavailable = next(
        item for item in REGISTRY.inputs if item.provision == "unavailable"
    )

    with pytest.raises(ValueError, match="null_wages must be one of"):
        us_resident_wage_tax_values(person, TAXABLE_ITEM, null_wages="fill")
    with pytest.raises(ValueError, match="is not a resident_wages input"):
        us_resident_wage_tax_values(person, unavailable)


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #


def _derived(rows: list[tuple[bool, float]]) -> pd.DataFrame:
    return apply_us_local_tax_inputs(_tables(_person(rows)))[0]["person"]


def test_gate_accepts_the_derived_column_or_its_absence() -> None:
    derived = _derived([(True, 10.0), (False, 20.0)])

    assert us_local_tax_input_failures(derived, require_derivable=True) == []
    assert us_local_tax_input_failures(derived.drop(columns=[TAXABLE])) == []
    # Neither flag nor input: nothing could decide it, and nothing claims to.
    assert (
        us_local_tax_input_failures(
            derived.drop(columns=[TAXABLE, RESIDENT]), require_derivable=True
        )
        == []
    )


def test_gate_requires_the_input_its_residence_flag_decides() -> None:
    derived = _derived([(True, 10.0)]).drop(columns=[TAXABLE])

    (failure,) = us_local_tax_input_failures(derived, require_derivable=True)

    assert failure.startswith(f"{TAXABLE}: absent, but {RESIDENT} is present")


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda p: p.assign(**{TAXABLE: [10.0, 1.0]}), "1/2 person row"),
        (lambda p: p.assign(**{TAXABLE: [9.0, 0.0]}), "1/2 person row"),
        (lambda p: p.assign(**{TAXABLE: [np.nan, 0.0]}), "missing or non-finite"),
        (lambda p: p.assign(**{TAXABLE: [np.inf, 0.0]}), "missing or non-finite"),
        (lambda p: p.drop(columns=[RESIDENT]), f"{RESIDENT} is absent"),
        (lambda p: p.drop(columns=[WAGES]), "cannot be checked"),
        (
            lambda p: p.assign(ny_yonkers_nonresident_earnings=0.0),
            "ny_yonkers_nonresident_earnings: present, but no source decides it "
            "\\(work_location\\)",
        ),
        (
            lambda p: p.assign(pa_philadelphia_wage_tax_reduced_rate_eligible=False),
            "\\(refund_claim\\)",
        ),
    ],
)
def test_gate_fails_any_column_the_rule_did_not_write(edit, message) -> None:
    derived = _derived([(True, 10.0), (False, 20.0)])

    failures = us_local_tax_input_failures(edit(derived))

    assert len(failures) == 1
    assert pd.Series(failures).str.contains(message).all()


def test_gate_accepts_a_propagated_null_only_in_propagate_mode() -> None:
    person = _person([(True, np.nan), (False, 1.0)])
    written, _ = apply_us_local_tax_inputs(
        _tables(person), null_wages=NULL_WAGES_PROPAGATE
    )

    assert (
        us_local_tax_input_failures(written["person"], null_wages=NULL_WAGES_PROPAGATE)
        == []
    )
    (failure,) = us_local_tax_input_failures(written["person"])
    assert "cannot be checked" in failure
    # A filled value where the wage is null is not what propagation writes.
    filled = written["person"].assign(**{TAXABLE: [0.0, 0.0]})
    (failure,) = us_local_tax_input_failures(filled, null_wages=NULL_WAGES_PROPAGATE)
    assert "1/2 person row" in failure


# --------------------------------------------------------------------------- #
# Frames, existing artifacts and annual aging
# --------------------------------------------------------------------------- #


def _philadelphia_frame():
    frame = us_frame(
        [
            {"state_fips": 42, "county_fips": "42101"},
            {"state_fips": 42, "county_fips": "42045"},
        ],
        person_inputs=[
            [{WAGES: 60_000.0}, {WAGES: -500.0}],
            [{WAGES: 70_000.0}],
        ],
        weights=[3.0, 4.0],
    )
    return with_us_local_jurisdiction_flags(frame, rungs=("county",))[0]


def test_frame_helper_writes_the_input_and_keeps_weights_and_links() -> None:
    frame = _philadelphia_frame()

    built, receipt = with_us_local_tax_inputs(frame)

    assert built.table("person")[TAXABLE].tolist() == [60_000.0, 0.0, 0.0]
    np.testing.assert_array_equal(
        built.weights_for("household").values, frame.weights_for("household").values
    )
    assert built.entities == frame.entities
    entry = receipt["inputs"][TAXABLE]
    assert entry["weighted_residents"] == 6.0
    assert entry["weighted_taxable_wages"] == 180_000.0
    assert TAXABLE not in frame.table("person")


def test_rederiving_refuses_a_carried_taxable_wage_that_disagrees() -> None:
    built, _ = with_us_local_tax_inputs(_philadelphia_frame())
    tables = {entity: built.table(entity) for entity in built.entities}
    tables["person"] = tables["person"].assign(**{TAXABLE: [1.0, 0.0, 0.0]})
    corrupt = type(built)(
        tables, built.schema, {"household": built.weights_for("household")}
    )

    with pytest.raises(ValueError, match=f"residence flags and wages: {TAXABLE}"):
        rederive_us_local_jurisdiction_flags(corrupt)


def test_rederiving_refuses_a_carried_unavailable_input() -> None:
    frame = _philadelphia_frame()
    tables = {entity: frame.table(entity) for entity in frame.entities}
    tables["person"] = tables["person"].assign(de_wilmington_nonresident_earnings=1.0)
    carried = type(frame)(
        tables, frame.schema, {"household": frame.weights_for("household")}
    )

    with pytest.raises(ValueError, match="de_wilmington_nonresident_earnings"):
        rederive_us_local_jurisdiction_flags(carried)


@settings(max_examples=60, deadline=None)
@given(
    st.lists(
        st.floats(min_value=0.5, max_value=3.0, allow_nan=False), min_size=3, max_size=3
    )
)
def test_annual_aging_rederives_taxable_wages_from_the_aged_wages(factors) -> None:
    built, _ = with_us_local_tax_inputs(_philadelphia_frame())
    tables = {entity: built.table(entity) for entity in built.entities}
    aged_person = tables["person"].assign(
        **{WAGES: tables["person"][WAGES] * np.asarray(factors)}
    )
    aged = {**tables, "person": aged_person}

    rederived, names = _rederive_local_tax_inputs(aged)

    assert names == [TAXABLE]
    expected = np.where(aged_person[RESIDENT], np.maximum(aged_person[WAGES], 0.0), 0.0)
    np.testing.assert_array_equal(rederived["person"][TAXABLE].to_numpy(), expected)
    assert us_local_tax_input_failures(rederived["person"]) == []
    # The caller's table keeps its base-year values.
    assert aged_person[TAXABLE].tolist() == [60_000.0, 0.0, 0.0]


def test_annual_aging_adds_no_input_a_base_did_not_carry() -> None:
    frame = _philadelphia_frame()
    tables = {entity: frame.table(entity) for entity in frame.entities}

    rederived, names = _rederive_local_tax_inputs(tables)

    assert names == []
    assert TAXABLE not in rederived["person"]
    assert US_SCHEMA.person_id_column in rederived["person"]
