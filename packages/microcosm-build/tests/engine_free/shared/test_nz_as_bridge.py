"""The packaged reg 17 base rate and reg 18 cutout, run on labelled fixture engines.

``JobseekerShapeFixture`` copies only the shape of rulespec-nz's
``jobseeker_support_gross_weekly_rate``, ``..._weekly_income_test_reduction``
and ``..._net_weekly_payment`` formulas. Its default numbers are fixture
values: three of them (the single-with-children rate 521.52, the lower
threshold 160 and the Income Test 3 rate 0.7) and the eldest-child tax credit
7921 reproduce golden-08's compiled quantities; every other number is
invented so the branches differ. None is a statement of NZ law. The tests
execute the packaged node declarations through the real kernels.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from microcosm.build.transport.compose import (
    TransportGraphConfig,
    _artifact_outputs,
    load_transport_spec,
)
from microcosm.build.transport.entitlement import _parameter, _rows
from microcosm.build.transport.target_kernels import CalibrationPrepareKernel
from microcosm.frame import ExportContract, Frame, VariableMetadata, WeightKind, Weights
from microcosm.frame.adapters.axiom import NZ_SCHEMA
from microcosm.frame.rules_kernels import (
    SimulateCounterfactualKernel,
    SimulateSolveZeroKernel,
)
from microcosm.graph import Node, Owned, Slice
from test_support.microcosm_frame.kernels import _context
from test_support.paths import paths_for

_FIXTURES = paths_for("microcosm-build").tests / "fixtures/nz"
_SPEC = load_transport_spec("nz")
_BRIDGE = _SPEC["resources"]["as_rate_bridge"]
_BASE = "accommodation_supplement_base_rate_weekly_amount"
_CUTOUT = "accommodation_supplement_non_beneficiary_income_cutout_weekly_amount"
_HAS_CHILD = "accommodation_supplement_bridge_has_dependent_child"
_GROSS = "accommodation_supplement_bridge_sole_parent_gross_rate"
_YOUNGEST = "jobseeker_support_youngest_dependent_child_age"
_JS_BINDINGS = (
    "main_benefits_rates",
    "family_tax_credits",
    "nz_super",
    "accommodation_supplement",
)
_REFS = {name: f"fixture:{name}" for name in _JS_BINDINGS}
_CONFIG = TransportGraphConfig(
    engine_refs=_REFS, create_outputs=(Owned("person", "age", "int64"),)
)


@dataclass
class JobseekerShapeFixture:
    """Fixture numbers for the Jobseeker Support formula shape; not NZ law."""

    single_with_children: float = 521.52
    single_youngest_14_or_older: float = 480.10
    single_under_20_living_with_parent: float = 250.0
    single_under_25: float = 300.0
    single_transferred_2013: float = 340.0
    single_standard: float = 360.0
    partner_ineligible_with_children: float = 410.0
    partner_ineligible_without_children: float = 330.0
    both_main_with_children: float = 430.0
    both_main_without_children: float = 310.0
    super_partner_with_children: float = 420.0
    super_partner_without_children: float = 300.0
    no_benefit_partner_with_children: float = 450.0
    no_benefit_partner_without_children: float = 320.0
    lower: float = 160.0
    upper: float = 250.0
    test_1_lower_rate: float = 0.3
    test_1_upper_rate: float = 0.7
    test_3_rate: float = 0.7
    test_4_rate: float = 0.35

    def gross(self, row):
        children = row["jobseeker_support_dependent_children_count"] >= 1
        if row["jobseeker_support_single"]:
            if children:
                if row[_YOUNGEST] >= 14:
                    return self.single_youngest_14_or_older
                return self.single_with_children
            if (
                row["jobseeker_support_applicant_age"] < 20
                and row["jobseeker_support_living_with_parent"]
                and row["jobseeker_support_benefit_commenced_on_or_after_1998_07_01"]
            ):
                return self.single_under_20_living_with_parent
            if row["jobseeker_support_applicant_age"] < 25:
                return self.single_under_25
            if row["jobseeker_support_transferred_2013_no_dependent_children"]:
                return self.single_transferred_2013
            return self.single_standard
        if row["jobseeker_support_partner_ineligible_due_to_sanction_or_strike"]:
            return (
                self.partner_ineligible_with_children
                if children
                else self.partner_ineligible_without_children
            )
        if row["jobseeker_support_partner_receives_main_benefit"]:
            return (
                self.both_main_with_children
                if children
                else self.both_main_without_children
            )
        if row[
            "jobseeker_support_partner_receives_new_zealand_superannuation_or_veterans_pension"
        ]:
            return (
                self.super_partner_with_children
                if children
                else self.super_partner_without_children
            )
        return (
            self.no_benefit_partner_with_children
            if children
            else self.no_benefit_partner_without_children
        )

    def _test_1(self, income):
        return (
            min(max(0, income - self.lower), max(0, self.upper - self.lower))
            * self.test_1_lower_rate
            + max(0, income - self.upper) * self.test_1_upper_rate
        )

    def reduction(self, row):
        income = row["jobseeker_support_total_income"]
        if row["jobseeker_support_single"]:
            if row["jobseeker_support_dependent_children_count"] >= 1:
                return self._test_1(income)
            if row["jobseeker_support_transferred_2013_no_dependent_children"]:
                return self._test_1(income)
            return max(0, income - self.lower) * self.test_3_rate
        if (
            row["jobseeker_support_partner_ineligible_due_to_sanction_or_strike"]
            or row["jobseeker_support_partner_receives_main_benefit"]
            or row[
                "jobseeker_support_partner_receives_new_zealand_superannuation_or_veterans_pension"
            ]
        ):
            return max(0, income - self.lower) * self.test_4_rate
        return max(0, income - self.lower) * self.test_3_rate

    def net(self, row):
        return max(0, self.gross(row) - self.reduction(row))


_JS_INPUTS = (
    "jobseeker_support_applicant_age",
    "jobseeker_support_benefit_commenced_on_or_after_1998_07_01",
    "jobseeker_support_dependent_children_count",
    "jobseeker_support_living_with_parent",
    "jobseeker_support_partner_ineligible_due_to_sanction_or_strike",
    "jobseeker_support_partner_receives_main_benefit",
    "jobseeker_support_partner_receives_new_zealand_superannuation_or_veterans_pension",
    "jobseeker_support_single",
    "jobseeker_support_total_income",
    "jobseeker_support_transferred_2013_no_dependent_children",
    _YOUNGEST,
)


class _FixtureEngine:
    """Engine protocol double that records each input column it receives."""

    entity = "person"
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    def __init__(self):
        self.received: list[pd.DataFrame] = []

    def entity_schema(self):
        return NZ_SCHEMA

    def variables(self):
        return self.inputs

    def variable_metadata(self, name):
        if name not in self.outputs:
            raise ValueError("Input variables have no engine output metadata")
        return VariableMetadata(
            name=name, entity=self.entity, dtype="float", period="year"
        )

    def export_contract(self):
        return ExportContract.empty()

    def write_dataset(self, bundle, path, period):
        return None

    def _batch(self, table):
        """Read inputs by column dtype, as the Axiom adapter's batch builder does."""
        batch = {}
        for name in self.inputs:
            column = table[name]
            kind = column.dtype.kind
            if kind == "b":
                batch[name] = column.to_numpy(dtype=bool)
            elif kind in ("i", "u"):
                batch[name] = column.to_numpy(dtype=np.int64)
            elif kind == "f":
                batch[name] = column.to_numpy(dtype=np.float64)
            else:
                raise ValueError(f"Column {name!r} has dtype kind {kind!r}.")
        return batch


class JobseekerEngine(_FixtureEngine):
    inputs = _JS_INPUTS
    outputs = (
        "jobseeker_support_gross_weekly_rate",
        "jobseeker_support_weekly_income_test_reduction",
        "jobseeker_support_net_weekly_payment",
    )

    def __init__(self, schedule=None):
        super().__init__()
        self.schedule = JobseekerShapeFixture() if schedule is None else schedule

    def materialize(self, bundle, variables, period):
        assert period == "2026-27"
        table = bundle.table("person")
        self.received.append(table.copy())
        batch = self._batch(table)
        rows = [
            {name: batch[name][index] for name in self.inputs}
            for index in range(len(table))
        ]
        functions = {
            "jobseeker_support_gross_weekly_rate": self.schedule.gross,
            "jobseeker_support_weekly_income_test_reduction": self.schedule.reduction,
            "jobseeker_support_net_weekly_payment": self.schedule.net,
        }
        return {
            name: np.asarray([functions[name](row) for row in rows], dtype=np.float64)
            for name in variables
        }


class TaxCreditEngine(_FixtureEngine):
    """Prescribed amount = care units times fixture annual amounts."""

    entity = "family"
    inputs = (
        "family_tax_credit_eldest_dependent_child_care_units",
        "family_tax_credit_subsequent_dependent_child_care_units",
    )
    outputs = ("family_tax_credit_prescribed_amount",)
    eldest = 7921.0
    subsequent = 6438.0

    def materialize(self, bundle, variables, period):
        table = bundle.table("family")
        self.received.append(table.copy())
        batch = self._batch(table)
        return {
            "family_tax_credit_prescribed_amount": batch[self.inputs[0]] * self.eldest
            + batch[self.inputs[1]] * self.subsequent
        }


class SuperannuationEngine(_FixtureEngine):
    """A fixture weekly amount before tax for adults aged 65 or older."""

    inputs = ("nz_super_person_age",)
    outputs = ("nz_super_weekly_amount_before_tax",)
    weekly = 538.0

    def materialize(self, bundle, variables, period):
        table = bundle.table("person")
        self.received.append(table.copy())
        age = self._batch(table)["nz_super_person_age"]
        return {
            "nz_super_weekly_amount_before_tax": np.where(age >= 65, self.weekly, 0.0)
        }


class AccommodationSupplementEngine(_FixtureEngine):
    """Pays half the rent, rounded, and judges a fixture cash-asset limit.

    As in rulespec-nz's core.yaml, the rounded payment never reads the
    non-rate judgment: only the host's gate can withhold it.
    """

    entity = "family"
    outputs = (
        "accommodation_supplement_rounded_weekly_payment",
        "accommodation_supplement_non_rate_entitlement_requirements",
    )
    cash_limit = 8100.0

    def __init__(self, undetermined=()):
        super().__init__()
        self.undetermined = set(undetermined)

    def variables(self):
        return tuple(
            entry["input"]
            for entry in _SPEC["resources"]["axiom_input_closure"]["entries"]
            if entry["entity"] == "Family"
            and entry["rulespec_path"].endswith("accommodation_supplement/core.yaml")
        )

    def materialize(self, bundle, variables, period):
        table = bundle.table("family")
        self.received.append(table.copy())
        payment = np.round(
            table["accommodation_supplement_weekly_rent_paid"].to_numpy(float) / 2
        )
        judgment = np.where(
            table["accommodation_supplement_cash_assets"].to_numpy(float)
            > self.cash_limit,
            -1,
            1,
        )
        judgment = np.where(
            table["family_id"].isin(self.undetermined).to_numpy(), 0, judgment
        )
        values = {
            "accommodation_supplement_rounded_weekly_payment": payment,
            "accommodation_supplement_non_rate_entitlement_requirements": judgment,
        }
        return {name: values[name] for name in variables}


def _row(node_id):
    for row, scenario in _rows(_SPEC)[0]:
        if row["id"] == node_id:
            return copy.deepcopy(row), scenario
    raise AssertionError(f"No packaged node {node_id!r}.")


def _node(row, scenario):
    return Node(
        row["id"],
        row["kernel"],
        inputs=tuple(
            Slice(item["entity"], tuple(item["columns"])) for item in row["inputs"]
        ),
        outputs=tuple(
            Owned(item["entity"], item["column"], item["dtype"])
            for item in row["outputs"]
        ),
        params={
            key: _parameter(_SPEC, _CONFIG, value, scenario)
            for key, value in row["params"].items()
        },
        artifact_outputs=_artifact_outputs(row),
    )


@dataclass
class Unit:
    """One invented benefit unit; every amount is fixture data."""

    partnered: bool = False
    children: tuple[int, ...] = ()
    receipt_group: str = "beneficiary"
    adult_age: int = 30
    cash_assets: float = 0.0
    rent: float = 400.0


_RECEIPT_FLAGS = {
    "superannuitant": "receives_nz_super",
    "beneficiary": "receives_jobseeker_support",
}


def _closure_inputs(binding, entity):
    path = next(
        row["rulespec_path"]
        for row in _SPEC["resources"]["axiom_rules_bindings"]["bindings"]
        if row["id"] == binding
    )
    return {
        entry["input"]: entry.get("value", 0)
        for entry in _SPEC["resources"]["axiom_input_closure"]["entries"]
        if entry["rulespec_path"] == path and entry["entity"] == entity
    }


def _frame(units):
    flags = {
        flag
        for predicate in _SPEC["resources"]["reporting_classifiers"]["receipt_group"][
            "predicates"
        ].values()
        for flag in predicate["columns"]
    }
    person_defaults = {
        **_closure_inputs("main_benefits_rates", "Person"),
        **_closure_inputs("nz_super", "Person"),
    }
    family_defaults = _closure_inputs("accommodation_supplement", "Family")
    person, family = [], []
    next_person = 1
    for index, unit in enumerate(units):
        family_id = 100 + index
        adults = 2 if unit.partnered else 1
        for member in range(adults + len(unit.children)):
            adult = member < adults
            age = unit.adult_age if adult else unit.children[member - adults]
            row = {
                "person_id": next_person,
                "person_household_id": family_id,
                "person_family_id": family_id,
                **person_defaults,
                "age": age,
                "nz_super_person_age": age,
                **{
                    flag: adult and _RECEIPT_FLAGS.get(unit.receipt_group) == flag
                    for flag in sorted(flags)
                },
            }
            person.append(row)
            next_person += 1
        youngest = min(unit.children) if unit.children else None
        family_type = (
            ("couple_with_children" if unit.children else "couple")
            if unit.partnered
            else ("sole_parent" if unit.children else "single")
        )
        family.append(
            {
                "family_id": family_id,
                **{
                    name: (0.0 if value == 0 else value)
                    for name, value in family_defaults.items()
                },
                "accommodation_supplement_cash_assets": unit.cash_assets,
                "accommodation_supplement_weekly_rent_paid": unit.rent,
                "n_dependent_children": len(unit.children),
                "is_partnered": unit.partnered,
                "is_single": not unit.partnered,
                "is_sole_parent": not unit.partnered and bool(unit.children),
                "youngest_dependent_child_age": youngest,
                "family_type": family_type,
                "receipt_group": unit.receipt_group,
            }
        )
    family = pd.DataFrame(family)
    family["youngest_dependent_child_age"] = family[
        "youngest_dependent_child_age"
    ].astype("Int64")
    family["receipt_group"] = family["receipt_group"].astype("string")
    family["family_type"] = family["family_type"].astype("string")
    ids = family["family_id"].to_numpy()
    return Frame(
        {
            "person": pd.DataFrame(person),
            "household": pd.DataFrame({"household_id": ids}),
            "family": family,
        },
        NZ_SCHEMA,
        {"household": Weights(np.ones(len(ids)), WeightKind.DESIGN)},
    )


def _engines(schedule=None, *, undetermined=()):
    return {
        _REFS["main_benefits_rates"]: JobseekerEngine(schedule),
        _REFS["family_tax_credits"]: TaxCreditEngine(),
        _REFS["nz_super"]: SuperannuationEngine(),
        _REFS["accommodation_supplement"]: AccommodationSupplementEngine(undetermined),
    }


def _run(frame, row, scenario, engines):
    node = _node(row, scenario)
    if row["kernel"] == "calibration.prepare@1":
        kernel = CalibrationPrepareKernel()
    elif row["kernel"] == "simulate.solve_zero@1":
        kernel = SimulateSolveZeroKernel(engines)
    else:
        kernel = SimulateCounterfactualKernel(engines)
    result = kernel.run(_context(frame, node, weighted_entities=("household",)))
    for (entity, column), values in result.columns.items():
        frame.table(entity)[column] = values.to_numpy()
    return result


def _bridge(units, *, schedule=None, prefix="nz.as", branch="nz.scn.S0"):
    """Run masks, gross rate, cutout and the branch's base rate in order."""
    frame = _frame(units)
    engines = _engines(schedule)
    for node_id in (
        f"{prefix}.bridge.masks",
        f"{prefix}.bridge.sole_parent_gross_rate",
        f"{prefix}.bridge.cutout",
        f"{branch}.bridge.base_rate",
    ):
        _run(frame, *_row(node_id), engines)
    family = frame.table("family")
    ids = family["family_id"]
    base = dict(zip(ids, family[_BASE], strict=True))
    return frame, engines, base, dict(zip(ids, family[_CUTOUT], strict=True))


# --- Golden-08 --------------------------------------------------------------


def test_fixture_run_reproduces_golden_08_base_rate_and_cutout():
    """Golden-08: a sole parent with children aged 0, 1 and 10 and no income.

    Tolerance: 1e-9 absolute, the solve's declared tolerance; the base rate's
    float64 sum carries representation error near 1e-13.
    """
    golden = json.loads(
        (_FIXTURES / "axiom_oracle_golden_08_accommodation_supplement.json").read_text()
    )
    (case,) = _BRIDGE["differential_expectations"]["cases"]
    assert case["request_id"] == "golden-08" and case["branch"] == "sole_parent"
    _, _, base, cutout = _bridge([Unit(children=(0, 1, 10))])
    for column, actual in ((_BASE, base[100]), (_CUTOUT, cutout[100])):
        expected = Decimal(case[column])
        assert Decimal(golden["inputs"][column]["value"]) == expected
        assert abs(Decimal(repr(actual)) - expected) <= Decimal("1e-9"), column


# --- Reg 18 -----------------------------------------------------------------

_BOUNDED = settings(max_examples=40, deadline=None, derandomize=True)


@_BOUNDED
@given(
    lower=st.floats(min_value=0, max_value=400),
    slope=st.floats(min_value=0.05, max_value=1),
    rate=st.floats(min_value=1, max_value=1500),
    children=st.lists(st.integers(min_value=0, max_value=17), min_size=1, max_size=3),
)
def test_solved_sole_parent_cutout_equals_lower_plus_rate_over_slope(
    lower, slope, rate, children
):
    assume(lower + rate / slope < 4999)
    schedule = JobseekerShapeFixture(
        single_with_children=rate, lower=lower, test_3_rate=slope
    )
    _, _, _, cutout = _bridge([Unit(children=tuple(children))], schedule=schedule)
    assert abs(cutout[100] - (lower + rate / slope)) <= 1e-9


@_BOUNDED
@given(
    lower=st.floats(min_value=0, max_value=450),
    test_3=st.floats(min_value=0.05, max_value=1),
    test_4=st.floats(min_value=0.05, max_value=1),
    single=st.floats(min_value=1, max_value=1500),
    couple=st.floats(min_value=1, max_value=1500),
    couple_children=st.lists(
        st.integers(min_value=0, max_value=17), min_size=0, max_size=2
    ),
)
def test_solved_other_unit_cutout_equals_the_harness_anchor_extrapolation(
    lower, test_3, test_4, single, couple, couple_children
):
    schedule = JobseekerShapeFixture(
        single_standard=single,
        both_main_with_children=couple,
        both_main_without_children=couple,
        lower=lower,
        test_3_rate=test_3,
        test_4_rate=test_4,
    )

    def payment(gross, rate, income):
        return max(0.0, gross - max(0.0, income - lower) * rate)

    expected = []
    for gross, rate in ((single, test_3), (couple, test_4)):
        at, after = payment(gross, rate, 500), payment(gross, rate, 501)
        assume(after > 0 and lower + gross / rate < 4999)
        expected.append(500 + at / (at - after))
    _, _, _, cutout = _bridge(
        [Unit(), Unit(partnered=True, children=tuple(couple_children))],
        schedule=schedule,
    )
    for actual, harness in zip((cutout[100], cutout[101]), expected, strict=True):
        assert abs(actual - harness) <= 1e-9 * max(1.0, harness)


def test_solve_covers_a_zero_anchor_payment_where_the_extrapolation_refuses():
    schedule = JobseekerShapeFixture(single_standard=40.0, lower=160, test_3_rate=0.7)
    anchor = {
        **{name: 0 for name in _JS_INPUTS},
        "jobseeker_support_single": True,
        "jobseeker_support_applicant_age": 25,
        "jobseeker_support_total_income": 500,
    }
    assert schedule.net(anchor) == 0
    _, _, _, cutout = _bridge([Unit()], schedule=schedule)
    assert abs(cutout[100] - (160 + 40 / 0.7)) <= 1e-9


def test_sole_parent_reaches_income_test_3_not_income_test_1():
    # Income Test 1's two tiers give 250 + (521.52 - 27) / 0.7 = 956.46.
    _, _, _, cutout = _bridge([Unit(children=(4,))])
    assert abs(cutout[100] - (160 + 521.52 / 0.7)) <= 1e-9
    assert abs(cutout[100] - (250 + (521.52 - 27) / 0.7)) > 1


def test_sole_parent_cutout_uses_the_gross_rate_overrides_not_the_units_schedule():
    # A youngest child of 15 selects the 14-or-older rate on the unit's own
    # schedule; reg 18(3)(b) reads the single-with-children rate.
    frame, _, _, cutout = _bridge([Unit(children=(15,)), Unit()])
    assert frame.table("family")[_GROSS].tolist() == [521.52, 0.0]
    assert abs(cutout[100] - (160 + 521.52 / 0.7)) <= 1e-9
    assert abs(cutout[100] - (160 + 480.10 / 0.7)) > 1


def test_upper_bracket_refuses_a_positive_residual():
    schedule = JobseekerShapeFixture(single_with_children=5000.0)
    with pytest.raises(ValueError, match="upper endpoint has a positive residual"):
        _bridge([Unit(children=(1,))], schedule=schedule)


def test_cutout_declaration_matches_the_ruling():
    row, _ = _row("nz.as.bridge.cutout")
    params = row["params"]
    assert (params["bracket"], params["tolerance"], params["iterations"]) == (
        [0, 5000],
        1e-9,
        43,
    )
    assert np.ldexp(5000.0, -43) <= params["tolerance"]
    assert params["solve_input"] == {
        "entity": "person",
        "column": "jobseeker_support_total_income",
    }
    assert not {"quantum", "certificate_margin"} & set(params)
    terms = params["output_coefficients"]
    assert [(term.get("variable"), term["coefficient"]) for term in terms] == [
        ("jobseeker_support_net_weekly_payment", 1),
        ("jobseeker_support_weekly_income_test_reduction", -1),
        (None, 1),
    ]
    assert terms[0]["mask"]["equals"] is False
    assert terms[1]["mask"]["equals"] is True
    assert terms[2]["source_column"] == _GROSS
    overrides = {item["column"]: item for item in params["input_overrides"]}
    assert overrides["jobseeker_support_single"]["source_column"] == (
        "accommodation_supplement_bridge_single_without_dependent_child"
    )
    assert "jobseeker_support_total_income" not in overrides
    assert _BRIDGE["outputs"][1]["method"] == "solve_zero"
    assert "awaiting D1" not in json.dumps(_BRIDGE)


# --- Reg 17 -----------------------------------------------------------------


def test_base_rate_sums_jobseeker_once_or_twice_and_the_credit_only_with_a_child():
    units = [
        Unit(),
        Unit(partnered=True),
        Unit(partnered=True, children=(3, 9)),
        Unit(children=(0, 1, 10)),
    ]
    _, engines, base, _ = _bridge(units)
    schedule = engines[_REFS["main_benefits_rates"]].schedule
    credit = TaxCreditEngine.eldest / 52
    expected = {
        100: schedule.single_standard,
        101: 2 * schedule.both_main_without_children,
        102: 2 * schedule.both_main_with_children + credit,
        103: schedule.single_with_children + credit,
    }
    for family_id, value in expected.items():
        assert abs(base[family_id] - value) <= 1e-9, family_id


def test_base_rate_declaration_matches_the_ruling():
    row, _ = _row("nz.scn.S0.bridge.base_rate")
    params = row["params"]
    terms = params["output_coefficients"]
    shape = [
        (
            term["variable"],
            term["coefficient"],
            term.get("aggregation"),
            term["mask"]["column"],
            term["mask"]["equals"],
        )
        for term in terms
    ]
    assert shape == [
        ("jobseeker_support_net_weekly_payment", 1, "first", "is_partnered", False),
        ("jobseeker_support_net_weekly_payment", 2, "first", "is_partnered", True),
        ("family_tax_credit_prescribed_amount", 1 / 52, None, _HAS_CHILD, True),
    ]
    assert terms[2]["engine_ref"] == {
        "binding": "family_tax_credits",
        "field": "engine_ref",
    }
    (component,) = params["components"]
    assert component["engine_ref"]["binding"] == "family_tax_credits"


def _youngest_received(units, *, masked=True):
    row, scenario = _row("nz.scn.S0.bridge.base_rate")
    if not masked:
        for item in row["params"]["input_overrides"]:
            if item["column"] == _YOUNGEST:
                item.pop("mask")
    frame = _frame(units)
    engines = _engines()
    _run(frame, *_row("nz.as.bridge.masks"), engines)
    try:
        _run(frame, row, scenario, engines)
    finally:
        received = engines[_REFS["main_benefits_rates"]].received
    return received[-1][_YOUNGEST]


def test_a_childless_family_keeps_the_closure_default_youngest_child_age():
    units = [Unit(), Unit(children=(4, 7))]
    received = _youngest_received(units)
    # 0 for the childless family's adult, 4 for the parent and both children.
    assert pd.Series(received).to_numpy(dtype=np.int64).tolist() == [0, 4, 4, 4]


def test_unmasked_youngest_child_override_carries_a_missing_age_to_the_engine():
    """Without the mask the childless unit's null age reaches the engine input.

    pandas 3 reads the nullable Int64 source as float64 with NaN, so the
    adapter's int64 conversion raises under numpy's invalid-value check and
    otherwise yields int64's minimum.
    """
    units = [Unit(), Unit(children=(4, 7))]
    received = _youngest_received(units, masked=False)
    assert received.isna().tolist() == [True, False, False, False]
    with np.errstate(invalid="raise"), pytest.raises(FloatingPointError):
        pd.Series(received).to_numpy(dtype=np.int64)


def test_bridge_scratch_columns_stay_out_of_the_export_contract():
    contract = _SPEC["resources"]["export_contract"]
    persisted = set(contract["required"]) | set(contract["optional"])
    persisted |= {
        column
        for roster in contract["entity_slices"].values()
        for columns in roster.values()
        for column in columns
    }
    scratch = {column["name"] for column in _BRIDGE["mask_bindings"]["columns"]}
    assert not persisted & (scratch | {_GROSS})


def test_every_as_population_runs_its_own_bridge_and_every_branch_its_base_rate():
    rows = {row["id"]: row for row, _ in _rows(_SPEC)[0]}
    scenarios = _SPEC["resources"]["scenarios"]["scenarios"]
    for node_id in ("masks", "sole_parent_gross_rate", "cutout"):
        assert rows[f"nz.as.bridge.{node_id}"]["population"] == "nz.as.bridge"
    shared = rows["nz.as.bridge"]
    assert (shared["structural"], shared["base"]) == ("filter", "nz.as")
    for scenario in scenarios:
        if scenario["tier"] == "entitlement":
            prefix, population = f"nz.scn.{scenario['id']}", f"nz.scn.{scenario['id']}"
            assert rows[prefix]["base"] == "nz.as.bridge"
        else:
            prefix = f"nz.v.{scenario['id']}"
            population = prefix + ".housing"
            for node_id in ("masks", "sole_parent_gross_rate", "cutout"):
                assert rows[f"{prefix}.bridge.{node_id}"]["population"] == population
        assert rows[f"{prefix}.bridge.base_rate"]["population"] == population


# --- Entitlement gate and MC11 (H25, H26) -------------------------------------

_PAYMENT = "accommodation_supplement_rounded_weekly_payment"
_JUDGMENT = "accommodation_supplement_non_rate_entitlement_requirements"
_POSITIVE = "accommodation_supplement_positive_payment_indicator"
_ENTITLED = "accommodation_supplement_entitled_weekly_payment"
_BRANCHES = (
    *(f"nz.scn.{name}" for name in ("S0", "S1", "S2", "S3", "S5")),
    *(f"nz.v.{name}" for name in ("V1", "V2", "V3", "V4", "VHF")),
)


def _as_engine_outputs(frame, engines):
    """Write the AS engine's payment and judgment as the branch's .as node would."""
    engine = engines[_REFS["accommodation_supplement"]]
    values = engine.materialize(frame, engine.outputs, "2026-27")
    for name, data in values.items():
        frame.table("family")[name] = data


def _entitle(units, branch, *, undetermined=(), frame=None):
    """Run the branch's packaged positive-payment and entitled nodes."""
    frame = _frame(units) if frame is None else frame
    engines = _engines(undetermined=undetermined)
    _as_engine_outputs(frame, engines)
    for suffix in (".positive_payment", ".as.entitled"):
        _run(frame, *_row(branch + suffix), engines)
    family = frame.table("family")
    return frame, dict(zip(family["family_id"], family[_ENTITLED], strict=True))


def test_a_unit_failing_the_non_rate_judgment_reads_zero():
    _, entitled = _entitle([Unit(), Unit(cash_assets=20000.0)], "nz.scn.S0")
    assert entitled == {100: 200.0, 101: 0.0}


def test_an_undetermined_judgment_reads_zero():
    frame, entitled = _entitle([Unit(), Unit()], "nz.scn.S0", undetermined={101})
    assert frame.table("family")[_JUDGMENT].tolist() == [1, 0]
    assert entitled == {100: 200.0, 101: 0.0}


def test_s1_differs_from_s0_for_a_unit_over_the_asset_limit():
    from microcosm.frame.input_closure import InputClosure

    unit = Unit(cash_assets=20000.0)
    _, central = _entitle([unit], "nz.scn.S0")
    # S1's encoding knob zeroes the cash-asset input the encoder supplies.
    s1 = next(
        row for row in _SPEC["resources"]["scenarios"]["scenarios"] if row["id"] == "S1"
    )
    closure = InputClosure.from_dict(_SPEC["resources"]["axiom_input_closure"])
    zeroed = closure.group_knobs(s1["encoding_knobs"]).zeroed
    assert zeroed == frozenset({"accommodation_supplement_cash_assets"})
    frame = _frame([unit])
    for column in zeroed:
        frame.table("family")[column] = 0.0
    _, asset_test_off = _entitle([unit], "nz.scn.S1", frame=frame)
    assert central == {100: 0.0}
    assert asset_test_off == {100: 200.0}


@pytest.mark.parametrize("branch", [b for b in _BRANCHES if not b.endswith("S5")])
def test_a_superannuitant_unit_reads_zero_outside_s5(branch):
    _, entitled = _entitle(
        [Unit(receipt_group="superannuitant", adult_age=70), Unit()], branch
    )
    assert entitled == {100: 0.0, 101: 200.0}


def test_s5_keeps_a_superannuitant_units_payment():
    _, entitled = _entitle(
        [Unit(receipt_group="superannuitant", adult_age=70)], "nz.scn.S5"
    )
    assert entitled == {100: 200.0}


def test_the_excluded_set_equals_the_superannuitant_receipt_group_cell():
    from microcosm.build.transport.reporting_kernels import (
        register_reporting_kernels,
    )
    from microcosm.graph import KernelRegistry

    units = [
        Unit(receipt_group="superannuitant", adult_age=70),
        Unit(receipt_group="superannuitant", adult_age=68, partnered=True),
        Unit(receipt_group="beneficiary"),
        Unit(receipt_group="other", children=(3,)),
    ]
    frame = _frame(units)
    frame.table("family").drop(columns="receipt_group", inplace=True)
    registry = register_reporting_kernels(KernelRegistry())
    row, scenario = _row("nz.scn.S0.receipt_group")
    node = _node(row, scenario)
    result = registry.get(node.kernel).run(
        _context(frame, node, weighted_entities=("household",))
    )
    for (entity, column), values in result.columns.items():
        frame.table(entity)[column] = values.to_numpy()
    _, entitled = _entitle(units, "nz.scn.S0", frame=frame)
    family = frame.table("family")
    excluded = set(
        family.loc[
            (family[_PAYMENT] > 0) & (family[_JUDGMENT] == 1) & ~family[_POSITIVE],
            "family_id",
        ]
    )
    cell = set(family.loc[family["receipt_group"] == "superannuitant", "family_id"])
    assert excluded == cell == {100, 101}
    assert all(entitled[unit] == 0 for unit in cell)


def _s5_base_rate(units):
    frame = _frame(units)
    engines = _engines()
    for node_id in (
        "nz.as.bridge.masks",
        "nz.scn.S5.bridge.superannuitant_masks",
        "nz.scn.S5.bridge.base_rate",
    ):
        _run(frame, *_row(node_id), engines)
    family = frame.table("family")
    return dict(zip(family["family_id"], family[_BASE], strict=True))


def test_s5_pays_a_superannuitant_unit_from_its_nzs_base_rate():
    base = _s5_base_rate(
        [Unit(receipt_group="superannuitant", adult_age=70), Unit(adult_age=30)]
    )
    schedule = JobseekerShapeFixture()
    assert base[100] == SuperannuationEngine.weekly
    assert base[101] == schedule.single_standard


def test_a_partnered_superannuitant_unit_takes_coefficient_two_in_s5():
    base = _s5_base_rate(
        [
            Unit(receipt_group="superannuitant", adult_age=70, partnered=True),
            Unit(receipt_group="superannuitant", adult_age=70, children=(12,)),
        ]
    )
    assert base[100] == 2 * SuperannuationEngine.weekly
    credit = TaxCreditEngine.eldest / 52
    assert abs(base[101] - (SuperannuationEngine.weekly + credit)) <= 1e-9


def test_the_central_base_rate_keeps_jobseeker_for_superannuitant_units():
    _, _, base, _ = _bridge(
        [Unit(receipt_group="superannuitant", adult_age=70, partnered=True)]
    )
    assert base[100] == 2 * JobseekerShapeFixture().both_main_without_children


def test_reporting_and_gap_read_the_entitled_payment():
    resources = _SPEC["resources"]
    conventions = resources["scenarios"]["gap_conventions"]
    assert conventions["reporting_cells"]["measure"] == _ENTITLED
    rows = {
        row["metadata"]["holdout_use"]: row
        for row in resources["holdout_references"]["target_references"]
        if row["metadata"].get("activation_status") == "active"
    }
    assert rows["accommodation_supplement_recipient_comparator"]["measure"] == _POSITIVE
    assert rows["accommodation_supplement_dollar_comparator"]["measure"] == _ENTITLED
    for branch in _BRANCHES:
        (family,) = [
            item
            for item in _row(branch + ".gap")[0]["inputs"]
            if item["entity"] == "family"
        ]
        assert _ENTITLED in family["columns"] and _PAYMENT not in family["columns"]
    excluded = resources["export_contract"]["formula_owned_excluded"]
    assert _ENTITLED in excluded
