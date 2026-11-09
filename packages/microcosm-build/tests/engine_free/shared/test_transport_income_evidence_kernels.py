"""R2C-28's income evidence kernels (G8-x1).

``transport.income_identity@1``, ``transport.omission_register@1`` and
``transport.native_band_report@1``. Every column, table, reference and fact
here is synthetic. The NZ wiring has its own tests in
``test_nz_transport_declarations.py``; the feed-gated test at the end reads
the real TY2024 IRD tables against the pinned feed.
"""

from __future__ import annotations

import copy
import json
import os
import random
import sys
from decimal import Decimal
from fractions import Fraction
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy import sparse

from microcosm.build.ledger_artifact import load_ledger_consumer_artifact
from microcosm.build.reference_intervals import (
    boundary_diagnostics,
    interval_table,
    scale_interval_table,
    source_interval_table,
)
from microcosm.build.transport import income_evidence_kernels
from microcosm.build.transport.artifact_types import CALIBRATION_PREPARATION_TYPE
from microcosm.build.transport.income_evidence_kernels import (
    IDENTITY_RECEIPTS_TYPE,
    INCOME_IDENTITIES,
    INCOME_IDENTITY,
    INCOME_IDENTITY_REF,
    NATIVE_BAND_REPORT,
    NATIVE_BAND_REPORT_REF,
    NATIVE_BAND_REPORT_TYPE,
    NIL_MODEL_REASON,
    OMISSION_REGISTER,
    OMISSION_REGISTER_REF,
    OMISSION_REGISTER_TYPE,
    WEIGHT_BASES,
    IncomeIdentityError,
    TilingError,
    aged_native_table,
    coarse_members,
    native_band_report,
    omission_register_totals,
    register_income_evidence_kernels,
)
from microcosm.build.transport.measure_kernels import (
    MEASURE_COMBINE,
    MEASURE_COMBINE_REF,
)
from microcosm.calibrate import CalibrationProblem, Target
from microcosm.calibrate.artifacts import (
    PROBLEM_TYPE,
    SOLUTION_TYPE,
    decode_problem,
    encode_problem,
    encode_solution,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    KernelRegistry,
    Node,
    Owned,
    Slice,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.executor import Population, _project_context
from microcosm.graph.kernel import ArtifactValue, NumericScope
from test_support.microcosm_build.chronicle_feed import write_pinned_artifact
from test_support.microcosm_build.first_party_imports import hashed_module_names
from test_support.microcosm_build.transport_graph import toy_fact
from test_support.microcosm_build.transport_population import digest
from test_support.paths import paths_for

SCHEMA = EntitySchema(group_entities=("household",))
OUTPUTS = {
    "measure": "toy_measure_cents",
    "unit": "toy_measure_units",
    "universe_status": "toy_measure_universe_status",
}
IDENTITIES = {
    "description": "Toy identities.",
    "identities": [
        {"id": identity, "statement": f"Toy statement for {identity}."}
        for identity in INCOME_IDENTITIES
    ],
}
REGISTER = {
    "description": "Toy register: items the toy measure omits.",
    "items": [
        {"item": "toy_pension", "size": "unknown", "reason": "No toy producer."},
        {"item": "toy_gains", "size": "unknown", "reason": "Rarely taxed."},
    ],
}
WAGES = "toy_wages_cents"
BUSINESS = "toy_business_cents"
RENT = "toy_rent_cents"
PROPERTY = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture,),
)


def _declaration(components, add_backs=(), *, floor=0, scale=100):
    return {
        "entity": "person",
        "components": [{"column": c, "sign": s} for c, s in components],
        "add_backs": list(add_backs),
        "floor": floor,
        "scale": scale,
        "outputs": dict(OUTPUTS),
    }


def _frame(person: pd.DataFrame, weights=None) -> Frame:
    size = len(person)
    household = pd.DataFrame({"household_id": np.arange(1, size + 1, dtype=np.int64)})
    values = np.linspace(1, 2, size) if weights is None else np.asarray(weights)
    return Frame(
        {"person": person, "household": household},
        SCHEMA,
        {"household": Weights(values, WeightKind.DESIGN)},
    )


def _people(columns: dict[str, list]) -> pd.DataFrame:
    size = len(next(iter(columns.values())))
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, size + 1, dtype=np.int64),
            "person_household_id": np.arange(1, size + 1, dtype=np.int64),
        }
    )
    for name, values in columns.items():
        person[name] = pd.array(values, dtype="Int64")
    return person


def _context(node: Node, frame: Frame):
    return _project_context(
        node,
        Population.from_frame(frame, "toy.population"),
        key="0" * 64,
        sources={},
        tolerances={},
        numerics={},
    )


def _measured(columns, declaration, weights=None):
    """The person table after ``measure.combine@1``, and that kernel's receipt."""

    person = _people(columns)
    frame = _frame(person, weights)
    node = Node(
        "toy.measure",
        MEASURE_COMBINE_REF,
        inputs=(
            Slice("person", tuple(row["column"] for row in declaration["components"])),
        ),
        outputs=(
            Owned("person", OUTPUTS["measure"], "Int64"),
            Owned("person", OUTPUTS["unit"], "float64"),
            Owned("person", OUTPUTS["universe_status"], "string"),
        ),
        params={
            "measure": canonical_json(declaration).decode(),
            "measure_sha256": digest(declaration),
        },
    )
    result = MEASURE_COMBINE.run(_context(node, frame))
    for name in OUTPUTS.values():
        person[name] = result.columns["person", name].array
    return person, result.receipt


def _measure_params(declaration):
    return {
        "measure": canonical_json(declaration).decode(),
        "measure_sha256": digest(declaration),
        "measurement_id": "toy_measurement_v1",
    }


def _identity_node(declaration, *, identities=IDENTITIES, columns=None):
    names = columns or (
        *(row["column"] for row in declaration["components"]),
        *OUTPUTS.values(),
    )
    return Node(
        "toy.evidence.identity",
        INCOME_IDENTITY_REF,
        inputs=(Slice("person", tuple(names)),),
        params={
            **_measure_params(declaration),
            "identities": canonical_json(identities).decode(),
        },
        artifact_outputs=(ArtifactOutput("identity_receipts", IDENTITY_RECEIPTS_TYPE),),
    )


def _register_node(declaration, *, register=REGISTER, columns=None):
    names = columns or (
        *(row["column"] for row in declaration["components"]),
        OUTPUTS["measure"],
    )
    return Node(
        "toy.evidence.register",
        OMISSION_REGISTER_REF,
        inputs=(Slice("person", tuple(names)),),
        params={
            "register": canonical_json(register).decode(),
            "income_measurements_sha256": "c" * 64,
            **_measure_params(declaration),
        },
        artifact_outputs=(ArtifactOutput("omission_register", OMISSION_REGISTER_TYPE),),
    )


def _identities(person, declaration, weights=None, **node):
    result = INCOME_IDENTITY.run(
        _context(_identity_node(declaration, **node), _frame(person, weights))
    )
    return result.artifacts["identity_receipts"], result.receipt


def _register(person, declaration, weights=None, **node):
    result = OMISSION_REGISTER.run(
        _context(_register_node(declaration, **node), _frame(person, weights))
    )
    return result.artifacts["omission_register"], result.receipt


TOY = _declaration([(WAGES, 1), (BUSINESS, 1), (RENT, 1)], add_backs=[RENT])
TOY_COLUMNS = {
    WAGES: [10_000, 0, 500, None, 0, 7],
    BUSINESS: [-5_000, -200, 0, 4, 0, 0],
    RENT: [-300, 100, 0, 0, 0, -9],
}


# ---------------------------------------------------------------------------
# transport.income_identity@1
# ---------------------------------------------------------------------------


def test_the_identities_hold_on_measure_combines_own_output() -> None:
    person, receipt = _measured(TOY_COLUMNS, TOY)
    payload, kernel_receipt = _identities(person, TOY)
    document = json.loads(payload)
    assert document["kind"] == "transport_income_identity"
    assert document["producer"]["kernel"] == INCOME_IDENTITY_REF
    assert document["identities"] == IDENTITIES["identities"]
    assert document["measurement_id"] == "toy_measurement_v1"
    assert document["n"] == 6
    assert document["universe_statuses"] == receipt["universe_statuses"]
    assert document["unresolved"] == 1
    assert document["sum_measure"] == receipt["sum_measure"]
    assert [
        (row["column"], row["resolved_signed_total"]) for row in document["components"]
    ] == [
        (row["column"], row["unweighted_signed_total"]) for row in receipt["components"]
    ]
    assert document["add_backs"] == receipt["add_backs"]
    assert kernel_receipt == {
        "n": 6,
        "unresolved": 1,
        "artifact_sha256": sha256(payload).hexdigest(),
    }


@pytest.mark.parametrize(
    ("column", "edit", "identity"),
    [
        (OUTPUTS["measure"], lambda value: value + 1, "component_sum"),
        (OUTPUTS["unit"], lambda value: np.nextafter(value, np.inf), "unit_rule"),
        (
            OUTPUTS["universe_status"],
            lambda value: "excluded" if value == "included" else "included",
            "universe_partition",
        ),
    ],
)
def test_one_edited_record_refuses_and_names_the_identity(column, edit, identity):
    person, _ = _measured(TOY_COLUMNS, TOY)
    person.loc[2, column] = edit(person.loc[2, column])
    with pytest.raises(
        IncomeIdentityError, match=f"identity {identity} fails at record 2"
    ):
        _identities(person, TOY)


def test_a_measure_beside_a_null_component_refuses() -> None:
    person, _ = _measured(TOY_COLUMNS, TOY)
    person.loc[3, OUTPUTS["measure"]] = 4
    with pytest.raises(IncomeIdentityError, match="component_sum fails at record 3"):
        _identities(person, TOY)


def test_an_unresolved_record_with_a_status_refuses() -> None:
    person, _ = _measured(TOY_COLUMNS, TOY)
    person.loc[3, OUTPUTS["universe_status"]] = "excluded"
    with pytest.raises(
        IncomeIdentityError, match="universe_partition fails at record 3"
    ):
        _identities(person, TOY)


def test_the_identities_must_be_the_kernels_own_in_order() -> None:
    person, _ = _measured(TOY_COLUMNS, TOY)
    reordered = copy.deepcopy(IDENTITIES)
    reordered["identities"].reverse()
    with pytest.raises(ValueError, match="must name exactly"):
        _identities(person, TOY, identities=reordered)
    blank = copy.deepcopy(IDENTITIES)
    blank["identities"][0]["statement"] = ""
    with pytest.raises(ValueError, match=r"list of \{id, statement\}"):
        _identities(person, TOY, identities=blank)


def test_the_identity_inputs_are_exactly_the_components_and_outputs() -> None:
    person, _ = _measured(TOY_COLUMNS, TOY)
    with pytest.raises(ValueError, match="inputs must slice exactly"):
        _identities(person, TOY, columns=(WAGES, BUSINESS, *OUTPUTS.values()))


def test_the_identity_artifact_moves_with_the_runs_columns() -> None:
    first, _ = _identities(_measured(TOY_COLUMNS, TOY)[0], TOY)
    changed = copy.deepcopy(TOY_COLUMNS)
    changed[WAGES][0] += 1
    second, _ = _identities(_measured(changed, TOY)[0], TOY)
    assert first != second
    assert json.loads(first)["sum_measure"] + 1 == json.loads(second)["sum_measure"]


_CENTS = st.one_of(st.integers(min_value=-(10**12), max_value=10**12), st.none())


@st.composite
def measures(draw):
    width = draw(st.integers(min_value=1, max_value=4))
    size = draw(st.integers(min_value=1, max_value=10))
    names = [f"toy_component_{index}_cents" for index in range(width)]
    signs = [draw(st.sampled_from((1, -1))) for _ in names]
    backs = [name for name in names if draw(st.booleans())]
    columns = {name: [draw(_CENTS) for _ in range(size)] for name in names}
    floor = draw(st.integers(min_value=-(10**6), max_value=10**6))
    scale = draw(st.integers(min_value=1, max_value=10**4))
    weights = [
        draw(st.floats(min_value=0.01, max_value=1e4, allow_nan=False))
        for _ in range(size)
    ]
    declaration = _declaration(
        list(zip(names, signs, strict=True)), backs, floor=floor, scale=scale
    )
    return columns, declaration, weights


@PROPERTY
@given(case=measures(), data=st.data())
def test_property_the_identities_hold_and_any_one_edit_refuses(case, data) -> None:
    columns, declaration, weights = case
    person, receipt = _measured(columns, declaration, weights)
    document = json.loads(_identities(person, declaration, weights)[0])
    assert document["sum_measure"] == receipt["sum_measure"]
    assert document["universe_statuses"] == receipt["universe_statuses"]
    index = data.draw(st.integers(min_value=0, max_value=len(person) - 1))
    status = person.loc[index, OUTPUTS["universe_status"]]
    if status == "unresolved":
        person.loc[index, OUTPUTS["universe_status"]] = "excluded"
        expected = "universe_partition"
    else:
        role = data.draw(st.sampled_from(("measure", "unit", "universe_status")))
        column = OUTPUTS[role]
        if role == "measure":
            person.loc[index, column] = person.loc[index, column] + 1
            expected = "component_sum"
        elif role == "unit":
            person.loc[index, column] = np.nextafter(person.loc[index, column], np.inf)
            expected = "unit_rule"
        else:
            person.loc[index, column] = (
                "excluded" if status == "included" else "included"
            )
            expected = "universe_partition"
    with pytest.raises(IncomeIdentityError, match=f"identity {expected} fails"):
        _identities(person, declaration, weights)


# ---------------------------------------------------------------------------
# transport.omission_register@1
# ---------------------------------------------------------------------------


def test_the_register_binds_the_declared_items_verbatim() -> None:
    person, _ = _measured(TOY_COLUMNS, TOY)
    payload, receipt = _register(person, TOY)
    document = json.loads(payload)
    assert document["kind"] == "transport_omission_register"
    assert document["register"] == REGISTER
    assert document["register_sha256"] == sha256(canonical_json(REGISTER)).hexdigest()
    assert document["income_measurements_sha256"] == "c" * 64
    assert document["weights"]["entity"] == "person"
    assert document["weights"]["basis"] == "pre_fit"
    assert receipt == {"items": 2, "artifact_sha256": sha256(payload).hexdigest()}


def test_the_register_refuses_repeated_items_and_extra_inputs() -> None:
    person, _ = _measured(TOY_COLUMNS, TOY)
    repeated = copy.deepcopy(REGISTER)
    repeated["items"].append(dict(repeated["items"][0]))
    with pytest.raises(ValueError, match="must be distinct"):
        _register(person, TOY, register=repeated)
    with pytest.raises(ValueError, match="inputs must slice exactly"):
        _register(
            person,
            TOY,
            columns=(WAGES, BUSINESS, RENT, OUTPUTS["measure"], OUTPUTS["unit"]),
        )


def _included_matches_receipt(included, receipt) -> None:
    """Differential: the register's totals against ``measure.combine@1``'s."""

    assert [
        (
            row["column"],
            row["sign"],
            row["unweighted_signed_total"],
            row["weighted_signed_total"],
        )
        for row in included["components"]
    ] == [
        (
            row["column"],
            row["sign"],
            row["unweighted_signed_total"],
            row["weighted_signed_total"],
        )
        for row in receipt["components"]
    ]
    assert [(row["column"], row["total"]) for row in included["add_backs"]] == [
        (row["column"], row["total"]) for row in receipt["add_backs"]
    ]
    assert included["below_zero"] == receipt["below_zero"]
    assert included["at_zero"] == receipt["at_zero"]


def test_the_included_totals_equal_measure_combines_receipt() -> None:
    person, receipt = _measured(TOY_COLUMNS, TOY)
    document = json.loads(_register(person, TOY)[0])
    _included_matches_receipt(document["included"], receipt)
    assert document["included"]["unresolved"]["n"] == 1


@PROPERTY
@given(case=measures(), seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_property_register_totals_match_and_ignore_record_order(case, seed) -> None:
    columns, declaration, weights = case
    person, receipt = _measured(columns, declaration, weights)
    document = json.loads(_register(person, declaration, weights)[0])
    _included_matches_receipt(document["included"], receipt)
    names = [row["column"] for row in declaration["components"]]
    order = list(range(len(person)))
    random.Random(seed).shuffle(order)
    shuffled = person.iloc[order].reset_index(drop=True)
    totals = omission_register_totals(
        declaration,
        {name: shuffled[name] for name in (*names, OUTPUTS["measure"])},
        np.asarray(weights, dtype=np.float64)[order],
    )
    assert totals == document["included"]


# ---------------------------------------------------------------------------
# transport.native_band_report@1: the pure report
# ---------------------------------------------------------------------------

INCOME = "toy_income_cents"
UNIVERSE = "toy_income_universe_status"
RECORD_SET = "toy.income_bands"
FACTOR = {"numerator": 107, "denominator": 100}
FACTOR_KEYS = {
    "factor_id": "toy_factor",
    "factor_declaration_sha256": "a" * 64,
    "factor_receipt_sha256": "b" * 64,
}


def _table(table_id, cuts, *, first, nil, prefix, companions=False, **fields):
    bands = []
    if nil:
        bands.append(
            {
                "id": f"{prefix}_nil",
                "kind": "nil",
                "indicator": f"{prefix}_nil_indicator",
                "source_label": "nil",
            }
        )
    for index, upper in enumerate([*cuts, None]):
        band = {
            "id": f"{prefix}_{index:02d}",
            "kind": "positive",
            "upper_cent": upper,
            "indicator": f"{prefix}_{index:02d}_indicator",
            "source_label": f"{prefix} band {index}",
        }
        if companions:
            band["count_reference"] = f"{prefix}_{index:02d}_count"
            band["amount_reference"] = f"{prefix}_{index:02d}_amount"
        bands.append(band)
    return {
        "id": table_id,
        "entity": "person",
        "currency": "NZD",
        "input_unit": "integer_cent",
        "income_column": INCOME,
        "universe_column": UNIVERSE,
        "source_period": 2024,
        "target_period": 2027,
        "positive_band_closure": "right",
        "nil_band": "present" if nil else "absent",
        "negative_income_policy": "floor_zero",
        "first_lower_cut_cents": first,
        "source_locator": "toy",
        "diagnostics": {
            "boundary_window_cents": 100,
            "alternative_positive_band_closure": "left",
        },
        "bands": bands,
        **fields,
    }


def _intervals(native_cuts, coarse_cuts, coarse_first, *, factor=FACTOR):
    native = _table("toy_native", native_cuts, first=0, nil=True, prefix="nat")
    coarse = _table(
        "toy_coarse",
        coarse_cuts,
        first=coarse_first,
        nil=False,
        prefix="crs",
        companions=True,
    )
    return {
        "schema_version": 1,
        "tables": [native, scale_interval_table(coarse, factor, **FACTOR_KEYS)],
    }


def _fact(key, value, *, money):
    fact = {"aggregate_fact_key": key, "layout": {"record_set_id": RECORD_SET}}
    fact["value"] = value
    if money:
        fact["observed_measure"] = {"unit": "nzd"}
    return fact


def _references(intervals, counts):
    """Diagnostic rows for each native band and coarse rows for each coarse band.

    ``counts`` maps each native band id to its published count; its amount is
    the count times 1,000.5 NZD, as exact decimal text.
    """

    native, coarse = intervals["tables"]
    diagnostics, facts = [], []
    for band in native["bands"]:
        for kind, value in (
            ("count", counts[band["id"]]),
            ("money", str(Decimal(counts[band["id"]]) * Decimal("1000.5"))),
        ):
            key = f"fact:{band['id']}:{kind}"
            diagnostics.append(
                {
                    "name": f"{band['id']}_{kind}",
                    "filter": band["indicator"],
                    "metadata": {"reference_value_kind": kind},
                    "ledger_selector": {
                        "record_set_id": RECORD_SET,
                        "aggregate_fact_key": key,
                    },
                }
            )
            facts.append(_fact(key, value, money=kind == "money"))
    members = coarse_members(
        source_interval_table(native), source_interval_table(coarse)
    )
    targets = []
    for band in coarse["bands"]:
        for kind, name in (
            ("count", band["count_reference"]),
            ("money", band["amount_reference"]),
        ):
            targets.append(
                {
                    "name": name,
                    "ledger_selector": {
                        "record_set_id": RECORD_SET,
                        "aggregate_fact_key": [
                            f"fact:{member}:{kind}" for member in members[band["id"]]
                        ],
                    },
                }
            )
    return (
        {"target_references": targets},
        {"diagnostic_references": diagnostics},
        facts,
    )


def _report(intervals, references, records, *, post_fit=None):
    targets, diagnostics, facts = references
    ids, amounts, statuses, households, weights = records
    post = weights if post_fit is None else post_fit
    return native_band_report(
        intervals,
        native_table="toy_native",
        coarse_table="toy_coarse",
        references=targets,
        diagnostic_references=diagnostics,
        facts=facts,
        person_ids=ids,
        amounts=amounts,
        statuses=statuses,
        source_households=households,
        weights={
            "pre_fit": (np.asarray(weights, dtype=np.float64), "design"),
            "post_fit": (np.asarray(post, dtype=np.float64), "design"),
        },
        measurement_id="toy_measurement_v1",
    )


NATIVE_CUTS = [10_000, 20_000, 30_000, 40_000, 50_000]
COARSE_CUTS = [30_000, 50_000]
COARSE_FIRST = 20_000


def _toy_counts(intervals):
    return {
        band["id"]: 3 * (index + 1)
        for index, band in enumerate(intervals["tables"][0]["bands"])
    }


def _toy_records():
    # Aged cuts are 1.07 times the source cuts: 10,700, 21,400, 32,100, ...
    amounts = [0, 5_000, 10_700, 10_701, 21_400, 32_101, 53_500, 60_000, 7]
    statuses = ["excluded", *["included"] * 7, "excluded"]
    ids = list(range(101, 101 + len(amounts)))
    households = [1, 1, 2, 3, 4, 4, 5, 6, 7]
    weights = [1.0, 1.5, 2.0, 2.5, 0.5, 1.25, 3.0, 0.75, 4.0]
    return ids, amounts, statuses, households, weights


def test_the_report_sets_each_native_row_beside_the_publisher() -> None:
    intervals = _intervals(NATIVE_CUTS, COARSE_CUTS, COARSE_FIRST)
    references = _references(intervals, _toy_counts(intervals))
    report = _report(intervals, references, _toy_records())
    rows = {row["band"]: row for row in report["rows"]}
    assert list(rows) == [band["id"] for band in intervals["tables"][0]["bands"]]
    assert report["scaling"]["factor_id"] == "toy_factor"
    # The second positive row's cut, 20,000 cents, aged by 107/100.
    assert rows["nat_01"]["source_upper_cent"] == 20_000
    assert rows["nat_01"]["aged_upper_cent"] == {"numerator": 21_400, "denominator": 1}
    assert rows["nat_01"]["publisher"]["count"] == 9
    assert rows["nat_01"]["publisher"]["amount"] == {
        "numerator": 18_009,
        "denominator": 2,
    }
    assert rows["nat_01"]["publisher"]["aged_amount"] == {
        "numerator": 18_009 * 107,
        "denominator": 200,
    }
    assert [rows[band]["coarse_band"] for band in ("nat_00", "nat_01", "nat_02")] == [
        None,
        None,
        "crs_00",
    ]
    # 10,700 sits in nat_00 (right-closed), 10,701 and 21,400 in nat_01.
    model = rows["nat_01"]["model"]["pre_fit"]
    assert model["records"] == 2
    assert model["weighted_count"] == {"numerator": 3, "denominator": 1}
    assert rows["nat_01"]["source_households"] == 2
    assert [row["band"] for row in report["coarse"]] == ["crs_00", "crs_01", "crs_02"]
    assert report["coarse"][0]["members"] == ["nat_02"]
    assert report["coarse"][1]["members"] == ["nat_03", "nat_04"]
    assert report["coarse"][1]["publisher"]["count"] == 15 + 18
    assert "R2C-4" in report["disclosure"]


def test_an_empty_nil_row_carries_null_not_zero() -> None:
    intervals = _intervals(NATIVE_CUTS, COARSE_CUTS, COARSE_FIRST)
    references = _references(intervals, _toy_counts(intervals))
    report = _report(intervals, references, _toy_records())
    nil = report["rows"][0]
    assert (nil["band"], nil["kind"]) == ("nat_nil", "nil")
    assert nil["model"] is None
    assert nil["model_reason"] == NIL_MODEL_REASON
    assert nil["source_households"] is None
    assert nil["publisher"]["count"] == 3
    ids, amounts, statuses, households, weights = _toy_records()
    statuses = ["included", *statuses[1:]]
    report = _report(
        intervals, references, (ids, amounts, statuses, households, weights)
    )
    nil = report["rows"][0]
    assert nil["model"]["pre_fit"]["records"] == 1
    assert nil["model_reason"] is None


def test_a_native_band_across_a_coarse_cut_refuses() -> None:
    intervals = _intervals(NATIVE_CUTS, [35_000, 50_000], COARSE_FIRST)
    native, coarse = intervals["tables"]
    with pytest.raises(TilingError, match="crosses a coarse cut"):
        coarse_members(source_interval_table(native), source_interval_table(coarse))


def test_a_coarse_reference_that_misses_a_member_fact_refuses() -> None:
    intervals = _intervals(NATIVE_CUTS, COARSE_CUTS, COARSE_FIRST)
    targets, diagnostics, facts = _references(intervals, _toy_counts(intervals))
    row = next(r for r in targets["target_references"] if r["name"] == "crs_01_count")
    row["ledger_selector"]["aggregate_fact_key"] = ["fact:nat_03:count"]
    with pytest.raises(TilingError, match="not exactly its 2 member native rows"):
        _report(intervals, (targets, diagnostics, facts), _toy_records())


def test_a_native_reference_must_pin_exactly_one_fact() -> None:
    intervals = _intervals(NATIVE_CUTS, COARSE_CUTS, COARSE_FIRST)
    targets, diagnostics, facts = _references(intervals, _toy_counts(intervals))
    facts.append(copy.deepcopy(facts[0]))
    with pytest.raises(ValueError, match="matches 2 facts"):
        _report(intervals, (targets, diagnostics, facts), _toy_records())


def test_an_amount_in_another_currency_refuses() -> None:
    intervals = _intervals(NATIVE_CUTS, COARSE_CUTS, COARSE_FIRST)
    targets, diagnostics, facts = _references(intervals, _toy_counts(intervals))
    facts[1]["observed_measure"]["unit"] = "aud"
    with pytest.raises(ValueError, match="not the table's 'NZD'"):
        _report(intervals, (targets, diagnostics, facts), _toy_records())


def test_the_native_table_ages_by_the_coarse_tables_own_factor_only() -> None:
    intervals = _intervals(NATIVE_CUTS, COARSE_CUTS, COARSE_FIRST)
    aged, block = aged_native_table(intervals, "toy_native", "toy_coarse")
    assert block["operational_factor"] == FACTOR
    assert interval_table(aged).cuts == tuple(
        Fraction(cut) * Fraction(107, 100) for cut in (0, *NATIVE_CUTS)
    )
    unscaled = copy.deepcopy(intervals)
    unscaled["tables"][1] = _table(
        "toy_coarse",
        COARSE_CUTS,
        first=COARSE_FIRST,
        nil=False,
        prefix="crs",
        companions=True,
    )
    with pytest.raises(ValueError, match="carries no scaling"):
        aged_native_table(unscaled, "toy_native", "toy_coarse")
    other = copy.deepcopy(intervals)
    other["tables"][0] = scale_interval_table(
        other["tables"][0], {"numerator": 11, "denominator": 10}, **FACTOR_KEYS
    )
    with pytest.raises(ValueError, match="scaled by another factor"):
        aged_native_table(other, "toy_native", "toy_coarse")
    currency = copy.deepcopy(intervals)
    currency["tables"][0]["currency"] = "AUD"
    with pytest.raises(ValueError, match=r"differ in \['currency'\]"):
        aged_native_table(currency, "toy_native", "toy_coarse")


@st.composite
def tilings(draw):
    """Native cuts, a coarse table on a subset of them, records and weights."""

    gaps = draw(
        st.lists(st.integers(min_value=1, max_value=5_000), min_size=2, max_size=8)
    )
    native_cuts = list(np.cumsum(gaps).tolist())
    picks = sorted(draw(st.sets(st.sampled_from(range(len(native_cuts))), min_size=1)))
    coarse_first = native_cuts[picks[0]]
    coarse_cuts = [native_cuts[index] for index in picks[1:]]
    numerator = draw(st.integers(min_value=1, max_value=400))
    denominator = draw(st.integers(min_value=1, max_value=400))
    size = draw(st.integers(min_value=1, max_value=14))
    top = int(native_cuts[-1] * 3)
    amounts = [draw(st.integers(min_value=-500, max_value=top)) for _ in range(size)]
    statuses = [draw(st.sampled_from(("included", "excluded"))) for _ in range(size)]
    households = [draw(st.integers(min_value=1, max_value=4)) for _ in range(size)]
    weights = [
        draw(st.floats(min_value=0.01, max_value=1e3, allow_nan=False))
        for _ in range(size)
    ]
    ids = list(range(1, size + 1))
    return (
        (
            native_cuts,
            coarse_cuts,
            coarse_first,
            {
                "numerator": numerator,
                "denominator": denominator,
            },
        ),
        (ids, amounts, statuses, households, weights),
    )


def _rational(value):
    return Fraction(value["numerator"], value["denominator"])


def _prepare(case):
    (native_cuts, coarse_cuts, coarse_first, factor), records = case
    intervals = _intervals(native_cuts, coarse_cuts, coarse_first, factor=factor)
    counts = {
        band["id"]: index * 3
        for index, band in enumerate(intervals["tables"][0]["bands"])
    }
    return intervals, _references(intervals, counts), records


@PROPERTY
@given(case=tilings())
def test_property_each_basis_conserves_the_included_mass(case) -> None:
    intervals, references, records = _prepare(case)
    report = _report(intervals, references, records)
    ids, amounts, statuses, households, weights = records
    included = [index for index, status in enumerate(statuses) if status == "included"]
    for basis in WEIGHT_BASES:
        mass = report["mass"][basis]
        assert mass["included_records"] == len(included)
        below = mass["below_first_cut"]["declared"]
        models = [row["model"][basis] for row in report["rows"] if row["model"]]
        assert sum(m["records"] for m in models) + below["records"] == len(included)
        assert sum(
            (_rational(m["weighted_count"]) for m in models), Fraction(0)
        ) + _rational(below["weighted_count"]) == sum(
            (Fraction(weights[index]) for index in included), Fraction(0)
        )


@PROPERTY
@given(case=tilings())
def test_property_coarse_bands_equal_their_members_against_the_coarse_table(case):
    """Differential: member sums against the coarse table's own diagnostics."""

    intervals, references, records = _prepare(case)
    report = _report(intervals, references, records)
    ids, amounts, statuses, _, weights = records
    coarse = boundary_diagnostics(
        interval_table(intervals["tables"][1]),
        ids,
        amounts,
        statuses,
        weights,
        weight_record_ids=ids,
        weight_kind="design",
    )
    rows = {row["band"]: row for row in report["rows"]}
    for band, expected in zip(report["coarse"], coarse["bands"], strict=True):
        assert band["band"] == expected["band"]
        members = [rows[name]["model"]["pre_fit"] for name in band["members"]]
        for field in ("weighted_count", "weighted_amount_cents"):
            assert sum(
                (_rational(member[field]) for member in members), Fraction(0)
            ) == _rational(expected["declared"][field])
        assert (
            sum(member["records"] for member in members)
            == (expected["declared"]["records"])
        )
        assert band["publisher"]["count"] == sum(
            rows[name]["publisher"]["count"] for name in band["members"]
        )


@PROPERTY
@given(case=tilings())
def test_property_post_fit_equals_pre_fit_at_equal_weights(case) -> None:
    intervals, references, records = _prepare(case)
    report = _report(intervals, references, records)
    for row in report["rows"]:
        if row["model"] is not None:
            assert row["model"]["pre_fit"] == row["model"]["post_fit"]
    assert report["mass"]["pre_fit"] == report["mass"]["post_fit"]


def _order_free(report):
    report = copy.deepcopy(report)
    for mass in report["mass"].values():
        del mass["digests"]["weights_sha256"]
        del mass["digests"]["record_order_sha256"]
    return report


@PROPERTY
@given(case=tilings(), seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_property_record_order_changes_nothing(case, seed) -> None:
    intervals, references, records = _prepare(case)
    order = list(range(len(records[0])))
    random.Random(seed).shuffle(order)
    shuffled = tuple([column[index] for index in order] for column in records)
    assert _order_free(_report(intervals, references, shuffled)) == _order_free(
        _report(intervals, references, records)
    )


@PROPERTY
@given(case=tilings(), data=st.data())
def test_property_a_coarse_cut_off_the_native_cuts_refuses(case, data) -> None:
    (native_cuts, _, _, factor), _ = case
    wide = [
        index
        for index in range(len(native_cuts) - 1)
        if native_cuts[index + 1] - native_cuts[index] >= 2
    ]
    if not wide:
        return
    index = data.draw(st.sampled_from(wide))
    cut = data.draw(
        st.integers(
            min_value=native_cuts[index] + 1, max_value=native_cuts[index + 1] - 1
        )
    )
    intervals = _intervals(native_cuts, [cut], native_cuts[0], factor=factor)
    native, coarse = intervals["tables"]
    with pytest.raises(TilingError, match="crosses a coarse cut"):
        coarse_members(source_interval_table(native), source_interval_table(coarse))


def test_the_kernels_register_once_and_idempotently() -> None:
    registry = KernelRegistry()
    register_income_evidence_kernels(registry)
    register_income_evidence_kernels(registry)
    assert set(registry.refs()) == {
        INCOME_IDENTITY_REF,
        OMISSION_REGISTER_REF,
        NATIVE_BAND_REPORT_REF,
    }
    assert registry.get(INCOME_IDENTITY_REF) is INCOME_IDENTITY
    assert registry.get(OMISSION_REGISTER_REF) is OMISSION_REGISTER
    assert registry.get(NATIVE_BAND_REPORT_REF) is NATIVE_BAND_REPORT


# ---------------------------------------------------------------------------
# transport.native_band_report@1's run(), on calibration artifacts and a feed
# ---------------------------------------------------------------------------

#: Four people in three households. The three included people all fall in
#: nat_01 (10,700 < x <= 21,400 once aged), and households 11 and 13 clone one
#: donor household, so that row holds three records from two source households.
NATIVE_PEOPLE = {
    "person_id": [101, 102, 103, 104],
    "person_household_id": [11, 11, 12, 13],
    INCOME: [0, 21_400, 10_701, 15_000],
    UNIVERSE: ["excluded", "included", "included", "included"],
}
NATIVE_HOUSEHOLDS = {
    "household_id": [11, 12, 13],
    "source_household_id": ["donor-a", "donor-b", "donor-a"],
}
DESIGN_WEIGHTS = (2.0, 1.0, 3.0)
CALIBRATED_WEIGHTS = (2.5, 0.5, 4.0)
FEED = "toy_feed"


def _feed_facts(facts):
    """The pure tests' facts as full consumer facts with numeric values."""

    rows = []
    for fact in facts:
        key = fact["aggregate_fact_key"]
        value = fact["value"]
        row = toy_fact(key.replace(":", "_"), value, entity="person")
        row["aggregate_fact_key"] = key
        row["layout"] = dict(fact["layout"])
        if "observed_measure" in fact:
            # Exact halves, so the JSON number carries the decimal text.
            row["value"] = float(Decimal(value))
            row["observed_measure"] = {
                **row["observed_measure"],
                **fact["observed_measure"],
            }
        rows.append(row)
    return rows


def _native_world(tmp_path, *, frame_weights=CALIBRATED_WEIGHTS, intervals_sha256=None):
    """The node, a frame, a pinned feed and the three artifacts run() reads.

    The solution holds ``CALIBRATED_WEIGHTS``; ``frame_weights`` and
    ``intervals_sha256`` let a test put the frame or the preparation off it.
    """

    intervals = _intervals(NATIVE_CUTS, COARSE_CUTS, COARSE_FIRST)
    targets, diagnostics, facts = _references(intervals, _toy_counts(intervals))
    path, pin = write_pinned_artifact(tmp_path / "feed", _feed_facts(facts))
    person = pd.DataFrame(
        {
            "person_id": np.asarray(NATIVE_PEOPLE["person_id"], dtype=np.int64),
            "person_household_id": np.asarray(
                NATIVE_PEOPLE["person_household_id"], dtype=np.int64
            ),
            INCOME: pd.array(NATIVE_PEOPLE[INCOME], dtype="Int64"),
            UNIVERSE: pd.array(NATIVE_PEOPLE[UNIVERSE], dtype="string"),
        }
    )
    households = pd.DataFrame(
        {
            "household_id": np.asarray(
                NATIVE_HOUSEHOLDS["household_id"], dtype=np.int64
            ),
            "source_household_id": pd.array(
                NATIVE_HOUSEHOLDS["source_household_id"], dtype="string"
            ),
        }
    )
    frame = Frame(
        {"person": person, "household": households},
        SCHEMA,
        {
            "household": Weights(
                np.asarray(frame_weights, dtype=np.float64), WeightKind.CALIBRATED
            )
        },
    )
    ids = tuple(NATIVE_HOUSEHOLDS["household_id"])
    target = Target(
        "count", "household", lambda f: np.ones(len(ids)), value=6, period=2027
    )
    problem = encode_problem(
        CalibrationProblem(
            sparse.csr_array([[1.0] * len(ids)]),
            np.array([6.0]),
            (target.row_name,),
            Weights(np.asarray(DESIGN_WEIGHTS, dtype=np.float64), WeightKind.DESIGN),
            "household",
            (target,),
        ),
        entity_ids=ids,
    )
    preparation = canonical_json(
        {
            "effective_intervals": intervals,
            "effective_intervals_sha256": intervals_sha256
            or sha256(canonical_json(intervals)).hexdigest(),
        }
    )
    payloads = {
        "preparation": (preparation, CALIBRATION_PREPARATION_TYPE),
        "problem": (problem, PROBLEM_TYPE),
        "solution": (
            encode_solution(
                np.asarray(CALIBRATED_WEIGHTS, dtype=np.float64),
                entity_ids=ids,
                problem_sha256=decode_problem(problem).sha256,
            ),
            SOLUTION_TYPE,
        ),
    }
    node = Node(
        "toy.evidence.native_bands",
        NATIVE_BAND_REPORT_REF,
        inputs=(
            Slice("person", (INCOME, UNIVERSE)),
            Slice("household", ("source_household_id",)),
        ),
        sources=(FEED,),
        params={
            "native_table": "toy_native",
            "coarse_table": "toy_coarse",
            "references": canonical_json(targets).decode(),
            "diagnostic_references": canonical_json(diagnostics).decode(),
            "diagnostic_references_sha256": digest(diagnostics),
            "chronicle_feed_pin": canonical_json(pin).decode(),
            "weight_entity": "household",
            "measurement_id": "toy_measurement_v1",
        },
        artifact_inputs=tuple(
            ArtifactInput(alias, f"toy.{alias}", alias, kind)
            for alias, (_, kind) in payloads.items()
        ),
        artifact_outputs=(
            ArtifactOutput("native_band_report", NATIVE_BAND_REPORT_TYPE),
        ),
    )
    artifacts = {
        alias: ArtifactValue(
            payload,
            kind,
            sha256(alias.encode()).hexdigest(),
            sha256(payload).hexdigest(),
            NumericScope(),
        )
        for alias, (payload, kind) in payloads.items()
    }
    context = _project_context(
        node,
        Population.from_frame(frame, "toy.terminal"),
        key="0" * 64,
        sources={FEED: path},
        tolerances={},
        numerics={},
        artifacts=artifacts,
    )
    return context, intervals, (targets, diagnostics), path, pin


def test_the_kernel_reports_the_frame_at_the_problems_and_solutions_weights(
    tmp_path,
) -> None:
    """run() equals the pure report on the frame's records (a differential).

    Each person takes its household's design weight from the problem and its
    calibrated weight from the solution, and its household's donor id; the
    facts come from the pinned feed.
    """

    context, intervals, (targets, diagnostics), path, pin = _native_world(tmp_path)
    result = NATIVE_BAND_REPORT.run(context)
    document = json.loads(result.artifacts["native_band_report"])
    loaded = load_ledger_consumer_artifact(path, feed_pin=pin, lexical_floats=True)
    rows = [0, 0, 1, 2]
    expected = native_band_report(
        intervals,
        native_table="toy_native",
        coarse_table="toy_coarse",
        references=targets,
        diagnostic_references=diagnostics,
        facts=loaded.facts,
        person_ids=NATIVE_PEOPLE["person_id"],
        amounts=NATIVE_PEOPLE[INCOME],
        statuses=NATIVE_PEOPLE[UNIVERSE],
        source_households=[NATIVE_HOUSEHOLDS["source_household_id"][r] for r in rows],
        weights={
            "pre_fit": (np.asarray(DESIGN_WEIGHTS)[rows], "design"),
            "post_fit": (np.asarray(CALIBRATED_WEIGHTS)[rows], "calibrated"),
        },
        measurement_id="toy_measurement_v1",
    )
    for key, value in json.loads(canonical_json(expected)).items():
        assert document[key] == value, key
    assert document["kind"] == "transport_native_band_report"
    assert document["measurement_id"] == "toy_measurement_v1"
    assert document["feed"] == {
        "facts_sha256": loaded.facts_sha256,
        "manifest_sha256": loaded.manifest_sha256,
    }
    assert result.receipt["rows"] == len(intervals["tables"][0]["bands"])
    assert (
        result.receipt["artifact_sha256"]
        == sha256(result.artifacts["native_band_report"]).hexdigest()
    )
    # Two of nat_01's three people sit in clones of donor-a.
    (row,) = [row for row in document["rows"] if row["band"] == "nat_01"]
    assert row["model"]["post_fit"]["records"] == 3
    assert row["source_households"] == 2


def test_the_kernel_refuses_frame_weights_off_the_solution(tmp_path) -> None:
    context, *_ = _native_world(tmp_path, frame_weights=(2.5, 0.5, 4.5))
    with pytest.raises(ValueError, match="weights differ from the calibration"):
        NATIVE_BAND_REPORT.run(context)


def test_the_kernel_refuses_a_preparation_without_authenticated_intervals(
    tmp_path,
) -> None:
    context, *_ = _native_world(tmp_path, intervals_sha256="0" * 64)
    with pytest.raises(ValueError, match="no authenticated effective intervals"):
        NATIVE_BAND_REPORT.run(context)


# ---------------------------------------------------------------------------
# Each kernel hashes every module its run() executes (as the evidence kernels'
# traced lists do)
# ---------------------------------------------------------------------------

#: The executor's interface to every kernel; an edit there re-keys every node.
_TRACE_EXCLUSIONS = frozenset({"microcosm.graph.kernel"})


def _microcosm_module_files() -> dict[str, str]:
    files = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("microcosm.") and path:
            files[str(Path(path).resolve())] = name
    return files


def _traced(run) -> set[str]:
    """Every ``microcosm.*`` module whose Python functions ``run()`` calls."""

    called: set[str] = set()

    def profile(frame, event, arg):
        if event == "call":
            called.add(frame.f_code.co_filename)

    sys.setprofile(profile)
    try:
        run()
    finally:
        sys.setprofile(None)
    files = _microcosm_module_files()
    return {
        files[path]
        for path in (str(Path(name).resolve()) for name in called if name[:1] != "<")
        if path in files
    }


def test_each_kernel_hashes_every_module_its_run_executes(tmp_path) -> None:
    # Under-hashing serves stale results: an edit to an unhashed module leaves
    # implementation_hash() unchanged, so a stored artifact outlives it.
    person, _ = _measured(TOY_COLUMNS, TOY)
    native, *_ = _native_world(tmp_path)
    runs = {
        INCOME_IDENTITY: lambda: _identities(person, TOY),
        OMISSION_REGISTER: lambda: _register(person, TOY),
        NATIVE_BAND_REPORT: lambda: NATIVE_BAND_REPORT.run(native),
    }
    for kernel, run in runs.items():
        hashed = hashed_module_names(kernel, income_evidence_kernels)
        assert income_evidence_kernels.__name__ in hashed, kernel.ref
        traced = _traced(run)
        assert income_evidence_kernels.__name__ in traced, kernel.ref
        assert traced - hashed - _TRACE_EXCLUSIONS == set(), kernel.ref
        assert not hashed & _TRACE_EXCLUSIONS, kernel.ref


def test_the_module_stays_outside_the_batterys_binding_closure() -> None:
    # The registry imports this module and the battery's bindings never do,
    # so adding the kernels re-keys no gates.battery@1 node.
    from microcosm.build.transport import binding_identity
    from microcosm.build.transport.gate_bindings import TRANSPORT_GATE_REGISTRY

    modules = set()
    for name, binding in TRANSPORT_GATE_REGISTRY.items():
        _, digests = binding_identity.describe(binding, path=name)
        modules |= set(digests)
    assert "microcosm.build.transport.evidence_contract" in modules
    assert income_evidence_kernels.__name__ not in modules
    assert "microcosm.build.transport.registry" not in modules
    # SKELETON_KERNELS names the three refs as literals, and the composer
    # sits outside the closure too, so that edit re-keys no battery either.
    assert "microcosm.build.transport.compose" not in modules


# ---------------------------------------------------------------------------
# The real TY2024 tables against the pinned feed (feed-gated)
# ---------------------------------------------------------------------------

_NZ = paths_for("microcosm-build").package / "src/microcosm/build/nz"
NATIONAL = "ird_taxable_income_national_ty2024"
STANDIN = "ird_taxable_income_standin_ty2024"


def _nz(name):
    return json.loads((_NZ / name).read_text())


@pytest.fixture(scope="module")
def nz_facts():
    path = os.environ.get("NZ_CHRONICLE_ARTIFACT")
    if not path:
        pytest.skip("NZ_CHRONICLE_ARTIFACT is unset; public NZ feed audit unavailable")
    return load_ledger_consumer_artifact(
        Path(path), feed_pin=_nz("chronicle_feed.json"), lexical_floats=True
    ).facts


def test_the_ird_native_rows_tile_the_twelve_standin_bands_in_the_feed(nz_facts):
    tables = {row["id"]: row for row in _nz("reference_intervals.json")["tables"]}
    # Any factor ages both tables alike, so the tiling holds or fails at
    # every factor; this one is arbitrary.
    intervals = {
        "schema_version": 1,
        "tables": [
            tables[NATIONAL],
            scale_interval_table(tables[STANDIN], FACTOR, **FACTOR_KEYS),
        ],
    }
    ids = [1, 2, 3]
    report = native_band_report(
        intervals,
        native_table=NATIONAL,
        coarse_table=STANDIN,
        references=_nz("target_references.json"),
        diagnostic_references=_nz("income_diagnostic_references.json"),
        facts=nz_facts,
        person_ids=ids,
        amounts=[0, 4_500_000, 200_000_000],
        statuses=["excluded", "included", "included"],
        source_households=[1, 2, 3],
        weights={
            basis: (np.array([1.0, 2.0, 3.0]), "design") for basis in WEIGHT_BASES
        },
        measurement_id="ird_taxable_income_ty2027_v1",
    )
    assert len(report["rows"]) == 237
    assert len(report["coarse"]) == 12
    members = [name for band in report["coarse"] for name in band["members"]]
    assert len(members) == len(set(members))
    below = [
        row["band"]
        for row in report["rows"]
        if row["kind"] == "positive" and row["coarse_band"] is None
    ]
    assert len(members) + len(below) + 1 == 237
    assert report["rows"][0]["model"] is None
    assert report["rows"][0]["model_reason"] == NIL_MODEL_REASON
