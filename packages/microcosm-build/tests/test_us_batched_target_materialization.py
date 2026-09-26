"""Household-batched base target materialization.

Fake-engine tests cover household partitions, column placement, engine
release and group nesting. Real-engine tests compare household-local targets
across batch sizes and demonstrate why Medicaid-cost targets require one
whole-pool batch. Multi-batch passes refuse known population aggregates.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.calibrate import TargetSpec
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights


def _load_builder_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "tools" / "build_us_fiscal_refresh_release.py"
    spec = importlib.util.spec_from_file_location(
        "build_us_fiscal_refresh_release", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fake engine: every value is a function of one record's own id
# ---------------------------------------------------------------------------

_VARIABLE_ENTITIES = {
    "income_tax": "tax_unit",
    "taxable_income": "tax_unit",
    "adjusted_gross_income": "tax_unit",
    "filing_status": "tax_unit",
    "state_income_tax": "tax_unit",
    "eitc": "tax_unit",
    "eitc_child_count": "tax_unit",
    "tax_unit_itemizes": "tax_unit",
    "itemized_taxable_income_deductions": "tax_unit",
    "ctc": "tax_unit",
    "ctc_limiting_tax_liability": "tax_unit",
    "tax_unit_size": "tax_unit",
    "mock_credit": "tax_unit",
    "age": "person",
    "ssi": "person",
    "employment_income": "person",
    "rental_income": "person",
    "farm_rent_income": "person",
    "snap": "spm_unit",
    "tanf": "spm_unit",
    "spm_unit_net_income": "spm_unit",
    "household_benefits": "household",
    "family_mock_income": "family",
    "marital_unit_mock_income": "marital_unit",
    # A per-record value whose formula also reads a population aggregate
    # when a test asks it to (``aggregate_reads``).
    "aggregate_probe": "person",
}
_FILING_STATUSES = ("SINGLE", "JOINT", "HEAD_OF_HOUSEHOLD", "SEPARATE")


def _record_value(variable: str, record_id: int):
    """One record's value; it reads nothing but the record's own id."""

    if variable == "filing_status":
        return _FILING_STATUSES[(record_id + record_id // 10) % len(_FILING_STATUSES)]
    if variable == "tax_unit_itemizes":
        return record_id % 3 == 0
    if variable == "eitc_child_count":
        return float(record_id % 4)
    if variable == "age":
        return float((record_id * 13) % 90)
    salt = sum(ord(character) for character in variable)
    # Signed, with zeros, so positive-part, loss-part and indicator slices
    # all see non-trivial rows.
    value = float(((record_id * 7919 + salt * 104729) % 2_000) - 500)
    return 0.0 if (record_id * 31 + salt) % 6 == 0 else value


class _FakeEngineLedger:
    """Records every engine the materializer builds, in order."""

    def __init__(self) -> None:
        self.simulations: list[object] = []
        self.constructions: list[dict[str, object]] = []
        self.reform_systems: list[object] = []


class _FakeHolders:
    """policyengine-core's holder view: the periods each variable is known for."""

    def __init__(self, known: dict[str, tuple[str, ...]] | None = None) -> None:
        self.known = {name: list(periods) for name, periods in (known or {}).items()}
        self.branches: dict[str, _FakeHolders] = {}
        self.clone_branches = False

    def get_branch(self, name: str):
        if name not in self.branches:
            branch = copy.copy(self) if self.clone_branches else _FakeHolders()
            branch.branches = {}
            branch.known = {}
            self.branches[name] = branch
        return self.branches[name]

    def get_holder(self, variable: str):
        periods = tuple(self.known.get(variable, ()))
        return SimpleNamespace(get_known_periods=lambda: list(periods))

    def record(self, variable: str, period) -> None:
        self.known.setdefault(variable, []).append(str(period))


def _install_fake_engine(
    builder,
    monkeypatch,
    *,
    reform_specs,
    aggregate_reads: dict[str, tuple[str, ...]] | None = None,
    aggregate_branch: str | None = None,
    aggregate_delete_branch: bool = False,
    aggregate_household: int | None = None,
    aggregate_reform_only: bool = False,
    aggregate_in_baseline: bool = False,
    clone_branches: bool = False,
    stored_inputs: dict[str, tuple[str, ...]] | None = None,
):
    """``aggregate_reads`` maps a variable to the population aggregates its
    formula computes (in the slash-delimited ``aggregate_branch`` if given);
    ``stored_inputs`` are known from construction, as dataset inputs are."""

    ledger = _FakeEngineLedger()
    aggregate_reads = dict(aggregate_reads or {})

    class FakeVariable:
        def __init__(self, entity: str) -> None:
            self.entity = SimpleNamespace(key=entity)

    class FakeSystem:
        variables = {
            name: FakeVariable(entity) for name, entity in _VARIABLE_ENTITIES.items()
        }

        def __init__(self, reform=None) -> None:
            self.reform = reform
            ledger.reform_systems.append(self)

    class FakeMicrosimulation(_FakeHolders):
        default_tax_benefit_system = FakeSystem

        def __init__(self, *, dataset, reform=None, tax_benefit_system=None):
            super().__init__(stored_inputs)
            alive = [
                simulation
                for simulation in ledger.simulations
                if simulation.dataset is not None
            ]
            frame = dataset["frame"]
            ledger.constructions.append(
                {
                    "reform": reform,
                    "households": tuple(
                        int(value) for value in frame.table("household")["household_id"]
                    ),
                    "alive_before": len(alive),
                }
            )
            self.dataset = dataset
            self.reform = reform
            self.tax_benefit_system = tax_benefit_system
            self.branches: dict[str, _FakeHolders] = {}
            self.clone_branches = clone_branches
            self.baseline = _FakeHolders() if reform is not None else None
            if self.baseline is not None:
                self.baseline.clone_branches = clone_branches
            ledger.simulations.append(self)

        def _ids(self, entity: str) -> np.ndarray:
            return self.dataset["frame"].table(entity)[f"{entity}_id"].to_numpy()

        def _values(self, variable: str) -> np.ndarray:
            entity = _VARIABLE_ENTITIES[variable]
            ids = self._ids(entity)
            if self.reform is not None:
                assert variable == "income_tax"
                return np.asarray(
                    [
                        _record_value(variable, int(id_)) + 10.0 * (id_ % 7)
                        for id_ in ids
                    ]
                )
            return np.asarray([_record_value(variable, int(id_)) for id_ in ids])

        def calculate(self, variable, *, period, map_to=None):
            assert self.dataset is not None, "calculate on a released engine"
            self.record(variable, period)
            for aggregate in aggregate_reads.get(variable, ()):
                if aggregate_reform_only and self.reform is None:
                    continue
                if (
                    aggregate_household is not None
                    and aggregate_household not in self._ids("household")
                ):
                    continue
                holders = self.baseline if aggregate_in_baseline else self
                branch_parent = holders
                if aggregate_branch is not None:
                    for branch_name in aggregate_branch.split("/"):
                        holders = holders.get_branch(branch_name)
                holders.record(aggregate, period)
                if aggregate_delete_branch:
                    del branch_parent.branches[aggregate_branch.split("/")[0]]
            values = self._values(variable)
            entity = _VARIABLE_ENTITIES[variable]
            if map_to is None or map_to == entity:
                return values
            # Project the record values onto persons, then sum persons into
            # the requested entity: still within-household by construction.
            frame = self.dataset["frame"]
            person = frame.table("person")
            if entity == "person":
                person_values = values.astype(np.float64)
            else:
                positions = pd.Series(
                    np.arange(len(values)), index=self._ids(entity)
                ).reindex(person[f"person_{entity}_id"].to_numpy())
                person_values = values.astype(np.float64)[positions.to_numpy()]
            if map_to == "person":
                return person_values
            target_ids = self._ids(map_to)
            target_positions = pd.Series(
                np.arange(len(target_ids)), index=target_ids
            ).reindex(person[f"person_{map_to}_id"].to_numpy())
            out = np.zeros(len(target_ids), dtype=np.float64)
            np.add.at(out, target_positions.to_numpy(), person_values)
            return out

    def fake_dataset_from_frame(
        frame_arg,
        *,
        zero_variables=(),
        system=None,
        assert_no_formula_owned_columns=True,
    ):
        return {"frame": frame_arg, "zero_variables": tuple(zero_variables)}

    monkeypatch.setitem(
        sys.modules,
        "policyengine_us",
        SimpleNamespace(
            CountryTaxBenefitSystem=FakeSystem,
            Microsimulation=FakeMicrosimulation,
        ),
    )
    monkeypatch.setattr(builder, "_assert_no_formula_owned_columns", lambda frame: None)
    monkeypatch.setattr(builder, "_dataset_from_frame", fake_dataset_from_frame)
    monkeypatch.setattr(
        builder,
        "_make_zero_variable_reform",
        lambda system, variable_name: variable_name,
    )
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", reform_specs)
    monkeypatch.setattr(
        builder,
        "SOI_VARIABLE_MAP",
        {
            "adjusted_gross_income": "adjusted_gross_income",
            "eitc": "eitc",
            "ctc": "ctc",
            "employment_income": "employment_income",
            "itemized_taxable_income_deductions": (
                "itemized_taxable_income_deductions"
            ),
            "rent_and_royalty_net_income": "rent_and_royalty_net_income",
            "rent_and_royalty_net_losses": "rent_and_royalty_net_income",
            "tax_filer_individual_count": "tax_unit_size",
        },
    )
    return ledger


# ---------------------------------------------------------------------------
# A small nested frame: five households, multi-member units, three states
# ---------------------------------------------------------------------------

# (household_id, state_fips, congressional_district_geoid) and, per person:
# (person_id, household, tax_unit, spm_unit, family, marital_unit).
_HOUSEHOLDS = (
    (1, 6, "0601"),
    (2, 36, "3601"),
    (3, 6, "0602"),
    (4, 24, "2401"),
    (5, 36, "3602"),
)
_PERSONS = (
    (1, 1, 10, 100, 1000, 10000),
    (2, 1, 10, 100, 1000, 10000),
    (3, 1, 11, 100, 1000, 10001),
    (4, 2, 20, 200, 2000, 20000),
    (5, 3, 30, 300, 3000, 30000),
    (6, 3, 30, 300, 3000, 30001),
    (7, 4, 40, 400, 4000, 40000),
    (8, 4, 40, 400, 4000, 40000),
    (9, 4, 41, 400, 4000, 40001),
    (10, 4, 42, 401, 4001, 40002),
    (11, 5, 50, 500, 5000, 50000),
)


def _nested_frame(persons=_PERSONS) -> Frame:
    person = pd.DataFrame(
        persons,
        columns=[
            "person_id",
            "person_household_id",
            "person_tax_unit_id",
            "person_spm_unit_id",
            "person_family_id",
            "person_marital_unit_id",
        ],
    ).astype("int64")
    tables = {"person": person}
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        ids = np.unique(person[f"person_{entity}_id"].to_numpy())
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    tables["household"] = pd.DataFrame(
        {
            "household_id": np.asarray([row[0] for row in _HOUSEHOLDS], "int64"),
            "state_fips": np.asarray([row[1] for row in _HOUSEHOLDS], "int64"),
            "congressional_district_geoid": np.asarray(
                [row[2] for row in _HOUSEHOLDS], dtype=object
            ),
            "household_input": np.asarray([3.0, 1.0, 4.0, 1.0, 5.0]),
        }
    )
    return Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]), WeightKind.DESIGN
            )
        },
    )


def _spec(name, family, metadata=None, *, measure=None, signed=True):
    return TargetSpec(
        name=name,
        entity="household",
        measure=measure or name,
        value=1.0,
        source="fixture",
        family=family,
        signed=signed,
        metadata=dict(metadata or {}),
    )


def _soi(name, variable, **metadata):
    return _spec(
        name,
        "irs_soi",
        {
            "variable": variable,
            "agi_lower_bound": "-inf",
            "agi_upper_bound": "inf",
            "filing_status": "All",
            "source_measure_id": name,
            **metadata,
        },
    )


def _variable(name, **metadata):
    return _spec(name, "fixture", {"materializer": "policyengine_variable", **metadata})


def _population_age(name, lower, upper, **metadata):
    return _spec(
        name,
        "census_population",
        {
            "materializer": "population_age",
            "measure_mode": "indicator_sum",
            "age_lower_bound": str(lower),
            "age_upper_bound": str(upper),
            **metadata,
        },
    )


# One target per branch of the base simulation, plus one JCT reform.
_TARGETS = (
    _soi("agi_amount", "adjusted_gross_income"),
    _soi(
        "agi_single_ca_bracket_count",
        "count",
        agi_lower_bound="-100",
        agi_upper_bound="900",
        filing_status="Single",
        state_fips="06",
    ),
    _soi(
        "agi_joint_count",
        "count",
        filing_status="Married Filing Jointly/Surviving Spouse",
    ),
    _soi("taxable_returns", "count", taxable_only="true"),
    _soi("eitc_two_children_amount", "eitc"),
    _soi(
        "eitc_returns_with_credit",
        "count",
        ledger_domain="individual_income_tax_returns_with_earned_income_credit",
    ),
    _soi("itemized_amount", "itemized_taxable_income_deductions", itemized_only="true"),
    _soi("ctc_amount", "ctc"),
    _soi("ctc_claims", "ctc", measure_mode="indicator_sum"),
    _soi("rent_royalty_net", "rent_and_royalty_net_income"),
    _soi("rent_royalty_losses", "rent_and_royalty_net_losses"),
    _soi(
        "cd_0601_wages",
        "employment_income",
        congressional_district_geoid="0601",
    ),
    _soi("filer_individuals", "tax_filer_individual_count"),
    _population_age("pop_under_18", 0, 18),
    _population_age("ca_pop_18_to_65", 18, 65, state_fips="06"),
    _population_age("cd_0601_pop", 0, "inf", congressional_district_geoid="0601"),
    _variable("ny_snap", base_variable="snap", state_fips="36"),
    _variable("ssi_recipients", base_variable="ssi", measure_mode="indicator_sum"),
    _variable(
        "low_net_income_snap_units",
        base_variable="spm_unit_net_income",
        measure_mode="less_than_indicator_sum",
        indicator_less_than="100",
        indicator_filter_variable="snap",
    ),
    _variable("rent_combined", base_variables="rental_income,farm_rent_income"),
    _variable(
        "ssi_household_indicator",
        base_variable="ssi",
        measure_mode="indicator_sum",
        indicator_map_to="household",
    ),
    _variable(
        "child_ssi_recipients",
        base_variable="ssi",
        measure_mode="indicator_sum",
        age_lower_bound="0",
        age_upper_bound="18",
    ),
    _variable(
        "cd_3601_household_benefits",
        base_variable="household_benefits",
        congressional_district_geoid="3601",
    ),
    _variable("family_income", base_variable="family_mock_income"),
    _variable("marital_income", base_variable="marital_unit_mock_income"),
    _spec("tanf", "hhs_tanf"),
    _spec("household_input", "fixture"),
    _spec("ca_state_income_tax", "state_income_tax", {"state_fips": "06"}),
    _spec("ny_state_income_tax", "state_income_tax", {"state_fips": "36"}),
    _spec("jct_mock_credit", "jct"),
)
_REFORMS = (
    SimpleNamespace(measure="jct_mock_credit", neutralized_variable="mock_credit"),
)


def _materialize(builder, monkeypatch, frame, batch_size):
    ledger = _install_fake_engine(builder, monkeypatch, reform_specs=_REFORMS)
    target_frame, registry, compilation = builder._materialize_target_frame(
        frame,
        _TARGETS,
        maximum_microsim_batch_size=batch_size,
    )
    return target_frame, registry, compilation, ledger


@pytest.mark.parametrize("batch_size", [1, 2, 3, 0])
def test_batched_base_simulation_matches_unbatched_target_frame(
    monkeypatch, batch_size
) -> None:
    """Identical target columns, column order and dtypes at every batch size."""

    builder = _load_builder_module()
    frame = _nested_frame()
    unbatched, unbatched_registry, unbatched_compilation, _ = _materialize(
        builder, monkeypatch, frame, None
    )
    batched, batched_registry, batched_compilation, _ = _materialize(
        builder, monkeypatch, frame, batch_size
    )

    # Every declared target materialized, so the comparison covers every
    # branch of the base simulation rather than an empty surface.
    assert unbatched_compilation["dropped_target_names"] == []
    unbatched_household = unbatched.table("household")
    assert set(spec.measure for spec in _TARGETS) <= set(unbatched_household)
    # The fixture exercises non-trivial values, not columns of zeros.
    for spec in _TARGETS:
        assert unbatched_household[spec.measure].to_numpy().any(), spec.measure

    pd.testing.assert_frame_equal(
        batched.table("household"), unbatched_household, check_exact=True
    )
    for entity in frame.entities:
        pd.testing.assert_frame_equal(
            batched.table(entity), unbatched.table(entity), check_exact=True
        )
    np.testing.assert_array_equal(
        batched.weights_for("household").values,
        unbatched.weights_for("household").values,
    )
    assert batched_registry.version == unbatched_registry.version
    assert (
        batched_compilation["dropped_target_names"]
        == unbatched_compilation["dropped_target_names"]
    )


def test_batched_base_simulation_reproduces_the_hand_computed_columns(
    monkeypatch,
) -> None:
    """Spot-check that the batched columns are the right numbers, not only
    equal to the unbatched ones. Household 4 has three tax units (a single, a
    joint and a head-of-household filer) and two SPM units."""

    builder = _load_builder_module()
    target_frame, _, _, _ = _materialize(builder, monkeypatch, _nested_frame(), 1)
    household = target_frame.table("household").set_index("household_id")

    tax_units = (40, 41, 42)
    expected_income_tax = sum(_record_value("income_tax", id_) for id_ in tax_units)
    assert household.loc[4, "income_tax"] == expected_income_tax
    assert household.loc[4, "jct_mock_credit"] == sum(
        10.0 * (id_ % 7) for id_ in tax_units
    )
    expected_agi = sum(_record_value("adjusted_gross_income", id_) for id_ in tax_units)
    assert household.loc[4, "agi_amount"] == expected_agi
    # MD is neither CA nor NY: the state-sliced rows are zero there.
    assert household.loc[4, "ca_state_income_tax"] == 0.0
    assert household.loc[4, "ny_snap"] == 0.0
    assert household.loc[2, "ny_snap"] == _record_value("snap", 200)
    assert household.loc[1, "ca_state_income_tax"] == sum(
        _record_value("state_income_tax", id_) for id_ in (10, 11)
    )
    persons = [row for row in _PERSONS if row[1] == 4]
    assert household.loc[4, "pop_under_18"] == sum(
        _record_value("age", row[0]) < 18 for row in persons
    )


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_batched_base_engines_partition_the_pool_and_are_released(
    monkeypatch, batch_size
) -> None:
    builder = _load_builder_module()
    frame = _nested_frame()
    household_ids = tuple(
        int(value) for value in frame.table("household")["household_id"]
    )
    _, _, compilation, ledger = _materialize(builder, monkeypatch, frame, batch_size)

    base = [entry for entry in ledger.constructions if entry["reform"] is None]
    reform = [entry for entry in ledger.constructions if entry["reform"] is not None]
    expected_batches = -(-len(household_ids) // batch_size)
    # The base simulation and the JCT family use the same partition: disjoint,
    # contiguous, in pool order, never larger than the batch size.
    for constructions in (base, reform):
        assert len(constructions) == expected_batches
        assert all(len(entry["households"]) <= batch_size for entry in constructions)
        assert (
            tuple(id_ for entry in constructions for id_ in entry["households"])
            == household_ids
        )
    # microcosm#456: each engine is released before the next is built, and
    # none survives the materializer.
    assert [entry["alive_before"] for entry in ledger.constructions] == [0] * len(
        ledger.constructions
    )
    assert all(simulation.dataset is None for simulation in ledger.simulations)
    # One metadata system plus one reform system for the family, never one
    # per batch.
    assert [system.reform for system in ledger.reform_systems] == [
        None,
        "mock_credit",
    ]

    assert compilation["target_materialization_batching"] == {
        "method": "household_position_batches",
        "maximum_microsim_batch_size": batch_size,
        "households": len(household_ids),
        "batches": expected_batches,
        "largest_batch_households": min(batch_size, len(household_ids)),
        "base_household_columns": compilation["target_materialization_batching"][
            "base_household_columns"
        ],
        "group_nesting_verified": True,
        "population_aggregate_guard_armed": True,
        "population_aggregate_variables_checked": list(
            builder.US_POPULATION_AGGREGATE_VARIABLES
        ),
        "jct_reform_families_simulated": 1,
    }
    # Every target column but the JCT row and the base-table input, plus the
    # two formula helpers the JCT subtraction and state rows read.
    base_columns = {spec.measure for spec in _TARGETS} - {
        "jct_mock_credit",
        "household_input",
    }
    base_columns |= {"income_tax", "state_income_tax"}
    assert compilation["target_materialization_batching"][
        "base_household_columns"
    ] == len(base_columns)


@pytest.mark.parametrize("batch_size", [None, 0, 5, 50])
def test_unbatched_base_simulation_runs_once_over_the_whole_frame(
    monkeypatch, batch_size
) -> None:
    builder = _load_builder_module()
    frame = _nested_frame()
    _, _, compilation, ledger = _materialize(builder, monkeypatch, frame, batch_size)

    base = [entry for entry in ledger.constructions if entry["reform"] is None]
    assert len(base) == 1
    assert len(base[0]["households"]) == frame.n("household")
    receipt = compilation["target_materialization_batching"]
    assert receipt["batches"] == 1
    assert receipt["largest_batch_households"] == frame.n("household")
    assert receipt["group_nesting_verified"] is False
    assert receipt["population_aggregate_guard_armed"] is False
    assert receipt["population_aggregate_variables_checked"] == []


@pytest.mark.parametrize(
    ("person_index", "crossing_person", "crossing_unit"),
    [
        # Person 5 (household 3, second batch) joins household 2's marital
        # unit 20000 (first batch): the group straddles the batch boundary,
        # and each batch on its own still nests. Only a pool-wide check
        # refuses it.
        pytest.param(4, (5, 3, 30, 300, 3000, 20000), 20000, id="across-batches"),
        # Person 4 (household 2) joins household 1's marital unit 10001;
        # both households sit in the first batch.
        pytest.param(3, (4, 2, 20, 200, 2000, 10001), 10001, id="within-a-batch"),
    ],
)
def test_batched_base_simulation_refuses_groups_that_cross_households(
    monkeypatch, person_index, crossing_person, crossing_unit
) -> None:
    """A group split across two batches would be simulated twice, each time
    with part of its members. Batching refuses it before building an engine."""

    builder = _load_builder_module()
    crossing = list(_PERSONS)
    crossing[person_index] = crossing_person
    frame = _nested_frame(tuple(crossing))
    ledger = _install_fake_engine(builder, monkeypatch, reform_specs=_REFORMS)

    with pytest.raises(
        ValueError,
        match=rf"marital_unit units must be nested .*\[{crossing_unit}\]",
    ):
        builder._materialize_target_frame(
            frame, _TARGETS, maximum_microsim_batch_size=2
        )
    assert ledger.constructions == []


_AGGREGATE_PROBE_TARGETS = (
    _variable("aggregate_probe_total", base_variable="aggregate_probe"),
)


@pytest.mark.parametrize("claiming_tax_unit_id", [20, 50, 999])
def test_batched_base_simulation_refuses_nonlocal_claiming_tax_units(
    monkeypatch, claiming_tax_unit_id
) -> None:
    builder = _load_builder_module()
    frame = _nested_frame()
    person = frame.table("person").copy()
    person["medicaid_claiming_tax_unit_id"] = 0
    person.loc[0, "medicaid_claiming_tax_unit_id"] = claiming_tax_unit_id
    frame = Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        US_SCHEMA,
        {"household": frame.weights_for("household")},
    )
    ledger = _install_fake_engine(builder, monkeypatch, reform_specs=())
    with pytest.raises(ValueError, match="medicaid_claiming_tax_unit_id.*household"):
        builder._materialize_target_frame(
            frame, _AGGREGATE_PROBE_TARGETS, maximum_microsim_batch_size=2
        )
    assert ledger.constructions == []


@pytest.mark.parametrize("claiming_tax_unit_id", [0, 10, 11])
def test_batched_base_simulation_allows_household_local_claiming_tax_units(
    monkeypatch, claiming_tax_unit_id
) -> None:
    builder = _load_builder_module()
    frame = _nested_frame()
    person = frame.table("person").copy()
    person["medicaid_claiming_tax_unit_id"] = 0
    person.loc[0, "medicaid_claiming_tax_unit_id"] = claiming_tax_unit_id
    frame = Frame(
        {
            entity: person if entity == "person" else frame.table(entity)
            for entity in frame.entities
        },
        US_SCHEMA,
        {"household": frame.weights_for("household")},
    )
    ledger = _install_fake_engine(builder, monkeypatch, reform_specs=())
    _, _, compilation = builder._materialize_target_frame(
        frame, _AGGREGATE_PROBE_TARGETS, maximum_microsim_batch_size=2
    )
    assert len(ledger.constructions) == 3
    assert compilation["target_materialization_population_aggregate_guard"] == {
        "armed": True,
        "population_aggregate_variables_checked": list(
            builder.US_POPULATION_AGGREGATE_VARIABLES
        ),
    }


def test_batched_base_simulation_refuses_each_population_aggregate(
    monkeypatch,
) -> None:
    """A batch engine that computed any listed population aggregate is
    refused, and released, at the first batch; one simulation over the whole
    pool computes the same formulas without refusal."""

    builder = _load_builder_module()
    assert builder.US_POPULATION_AGGREGATE_VARIABLES == (
        "household_income_decile",
        "medicaid_slcsp_state_average_cost_index",
        "medicaid_slcsp_state_denominator",
        "spm_unit_income_decile",
    )
    for aggregate in builder.US_POPULATION_AGGREGATE_VARIABLES:
        ledger = _install_fake_engine(
            builder,
            monkeypatch,
            reform_specs=(),
            aggregate_reads={"aggregate_probe": (aggregate,)},
        )
        with pytest.raises(
            ValueError,
            match=(
                "Target materialization is not batch-invariant: the "
                rf"engine for household batch 1/3 computed {aggregate}@"
                rf"{builder.PERIOD}\. .*unbatched.*one slice or chunk"
            ),
        ):
            builder._materialize_target_frame(
                _nested_frame(),
                _AGGREGATE_PROBE_TARGETS,
                maximum_microsim_batch_size=2,
            )
        assert len(ledger.constructions) == 1, aggregate
        assert all(simulation.dataset is None for simulation in ledger.simulations)

        target_frame, _, compilation = builder._materialize_target_frame(
            _nested_frame(),
            _AGGREGATE_PROBE_TARGETS,
            maximum_microsim_batch_size=None,
        )
        assert compilation["dropped_target_names"] == []
        assert "aggregate_probe_total" in target_frame.table("household")
        assert ledger.simulations[-1].known[aggregate] == [str(builder.PERIOD)]
        receipt = compilation["target_materialization_batching"]
        assert receipt["batches"] == 1
        assert receipt["population_aggregate_guard_armed"] is False
        assert receipt["population_aggregate_variables_checked"] == []


def test_batched_base_simulation_refuses_an_aggregate_first_reached_in_last_batch(
    monkeypatch,
) -> None:
    builder = _load_builder_module()
    ledger = _install_fake_engine(
        builder,
        monkeypatch,
        reform_specs=(),
        aggregate_reads={"aggregate_probe": ("medicaid_slcsp_state_denominator",)},
        aggregate_household=5,
    )
    with pytest.raises(
        ValueError,
        match=r"household batch 3/3 computed medicaid_slcsp_state_denominator@",
    ):
        builder._materialize_target_frame(
            _nested_frame(), _AGGREGATE_PROBE_TARGETS, maximum_microsim_batch_size=2
        )
    assert len(ledger.constructions) == 3
    assert all(simulation.dataset is None for simulation in ledger.simulations)


@pytest.mark.parametrize("branch_depth", [1, 2])
@pytest.mark.parametrize("clone_branches", [False, True])
def test_batched_base_simulation_reads_aggregates_held_by_live_branches(
    monkeypatch, branch_depth, clone_branches
) -> None:
    builder = _load_builder_module()
    _install_fake_engine(
        builder,
        monkeypatch,
        reform_specs=(),
        aggregate_reads={"aggregate_probe": ("spm_unit_income_decile",)},
        aggregate_branch="/".join(f"branch_{depth}" for depth in range(branch_depth)),
        clone_branches=clone_branches,
    )
    with pytest.raises(ValueError, match=r"computed spm_unit_income_decile@"):
        builder._materialize_target_frame(
            _nested_frame(), _AGGREGATE_PROBE_TARGETS, maximum_microsim_batch_size=2
        )


@pytest.mark.parametrize("clone_branches", [False, True])
def test_batched_base_simulation_reads_aggregates_held_by_deleted_branches(
    monkeypatch, clone_branches
) -> None:
    builder = _load_builder_module()
    ledger = _install_fake_engine(
        builder,
        monkeypatch,
        reform_specs=(),
        aggregate_reads={"aggregate_probe": ("spm_unit_income_decile",)},
        aggregate_branch="temporary_parent/temporary_child",
        aggregate_delete_branch=True,
        clone_branches=clone_branches,
    )
    with pytest.raises(ValueError, match=r"computed spm_unit_income_decile@"):
        builder._materialize_target_frame(
            _nested_frame(), _AGGREGATE_PROBE_TARGETS, maximum_microsim_batch_size=2
        )
    assert ledger.simulations[0].branches == {}
    assert ledger.simulations[0].dataset is None


def test_batched_base_simulation_refuses_a_stored_aggregate_input(
    monkeypatch,
) -> None:
    """Any known period is refused, including values present at construction."""

    builder = _load_builder_module()
    ledger = _install_fake_engine(
        builder,
        monkeypatch,
        reform_specs=(),
        stored_inputs={"household_income_decile": (str(builder.PERIOD),)},
    )
    with pytest.raises(ValueError, match=r"computed household_income_decile@"):
        builder._materialize_target_frame(
            _nested_frame(), _AGGREGATE_PROBE_TARGETS, maximum_microsim_batch_size=2
        )
    assert len(ledger.constructions) == 1
    assert all(simulation.dataset is None for simulation in ledger.simulations)


@pytest.mark.parametrize("aggregate_in_baseline", [False, True])
def test_batched_reform_simulation_refuses_population_aggregates(
    monkeypatch, aggregate_in_baseline
) -> None:
    builder = _load_builder_module()
    ledger = _install_fake_engine(
        builder,
        monkeypatch,
        reform_specs=_REFORMS,
        aggregate_reads={"income_tax": ("medicaid_slcsp_state_denominator",)},
        aggregate_branch="baseline/temporary",
        aggregate_delete_branch=True,
        aggregate_reform_only=True,
        aggregate_in_baseline=aggregate_in_baseline,
        clone_branches=True,
    )
    with pytest.raises(ValueError, match=r"computed medicaid_slcsp_state_denominator@"):
        builder._materialize_target_frame(
            _nested_frame(), _TARGETS, maximum_microsim_batch_size=2
        )
    assert sum(entry["reform"] is None for entry in ledger.constructions) == 3
    assert sum(entry["reform"] is not None for entry in ledger.constructions) == 1
    assert all(simulation.dataset is None for simulation in ledger.simulations)


def _second_batch_drops_a_column(real):
    calls = [0]

    def columns(*args, **kwargs):
        calls[0] += 1
        result = real(*args, **kwargs)
        if calls[0] == 2:
            result.pop("income_tax")
        return result

    return columns


def _one_value_short(real):
    def columns(*args, **kwargs):
        result = real(*args, **kwargs)
        result["income_tax"] = result["income_tax"][:-1]
        return result

    return columns


@pytest.mark.parametrize(
    ("target", "wrap", "message"),
    [
        (
            "_select_households_by_position",
            # Each batch is handed the next batch's households.
            lambda real: lambda frame, positions: real(frame, positions + 1),
            "does not carry exactly its households in pool order",
        ),
        (
            "_base_simulation_household_columns",
            _second_batch_drops_a_column,
            "different column sets or orders",
        ),
        (
            "_base_simulation_household_columns",
            _one_value_short,
            "has shape",
        ),
    ],
)
def test_batched_base_simulation_refuses_batches_it_cannot_place(
    monkeypatch, target, wrap, message
) -> None:
    """The loop places batch values by pool position, so it refuses a batch
    whose households, columns or lengths do not line up with that position."""

    builder = _load_builder_module()
    _install_fake_engine(builder, monkeypatch, reform_specs=_REFORMS)
    monkeypatch.setattr(builder, target, wrap(getattr(builder, target)))

    with pytest.raises(RuntimeError, match=message):
        builder._materialize_target_frame(
            _nested_frame(), _TARGETS, maximum_microsim_batch_size=2
        )


def test_checkpoint_round_trip_keeps_the_batching_receipt(
    monkeypatch, tmp_path
) -> None:
    builder = _load_builder_module()
    frame = _nested_frame()
    ledger = _install_fake_engine(builder, monkeypatch, reform_specs=_REFORMS)
    identity = {
        "kind": "fixture",
        "materializer_version": builder.TARGET_FRAME_CHECKPOINT_MATERIALIZER_VERSION,
    }
    path = tmp_path / "target_frame_checkpoint.h5"

    target_frame, _, compilation = builder._load_or_materialize_target_frame(
        frame,
        _TARGETS,
        target_frame_checkpoint_path=path,
        target_frame_checkpoint_identity=identity,
        target_frame_checkpoint_build_commit="a" * 40,
        maximum_microsim_batch_size=2,
    )
    assert compilation["target_frame_checkpoint"]["status"] == "miss_written"
    receipt = compilation["target_materialization_batching"]
    assert receipt["batches"] == 3
    engines_after_miss = len(ledger.constructions)

    reloaded, _, reloaded_compilation = builder._load_or_materialize_target_frame(
        frame,
        _TARGETS,
        target_frame_checkpoint_path=path,
        target_frame_checkpoint_identity=identity,
        target_frame_checkpoint_build_commit="b" * 40,
        maximum_microsim_batch_size=2,
    )
    checkpoint = reloaded_compilation["target_frame_checkpoint"]
    assert checkpoint["status"] == "hit"
    # microcosm#1018: the hit names the commit that wrote the checkpoint,
    # beside the batching receipt it restores.
    assert checkpoint["source_build_commit"] == "a" * 40
    # The hit runs no engine and says so; the writing run's receipt survives
    # in the stored compilation.
    assert len(ledger.constructions) == engines_after_miss
    assert reloaded_compilation["target_materialization_batching"] == {
        "status": "skipped_target_frame_checkpoint_hit"
    }
    assert checkpoint["stored_compilation"]["target_materialization_batching"] == (
        receipt
    )
    pd.testing.assert_frame_equal(
        reloaded.table("household")[list(target_frame.table("household"))],
        target_frame.table("household"),
        check_dtype=False,
    )


# ---------------------------------------------------------------------------
# Real engine (requires the [us] extra)
# ---------------------------------------------------------------------------


def _real_engine_frame() -> Frame:
    """Four households over three states; two share a household across tax
    units, and every group nests in its household."""

    persons = pd.DataFrame(
        {
            "person_id": np.arange(1, 8, dtype="int64"),
            "person_household_id": np.asarray([1, 1, 2, 3, 3, 3, 4], "int64"),
            "person_tax_unit_id": np.asarray([10, 11, 20, 30, 31, 32, 40], "int64"),
            "person_spm_unit_id": np.asarray(
                [100, 100, 200, 300, 300, 301, 400], "int64"
            ),
            "person_family_id": np.asarray(
                [1000, 1000, 2000, 3000, 3000, 3001, 4000], "int64"
            ),
            "person_marital_unit_id": np.asarray(
                [10000, 10001, 20000, 30000, 30001, 30002, 40000], "int64"
            ),
            "age": np.asarray([34.0, 71.0, 45.0, 29.0, 67.0, 19.0, 52.0]),
            "employment_income_before_lsr": np.asarray(
                [48_000.0, 0.0, 180_000.0, 9_000.0, 0.0, 14_000.0, 72_000.0]
            ),
        }
    )
    tables = {"person": persons}
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        ids = np.unique(persons[f"person_{entity}_id"].to_numpy())
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    tables["household"] = pd.DataFrame(
        {
            "household_id": np.asarray([1, 2, 3, 4], "int64"),
            "state_fips": np.asarray([6, 36, 24, 6], "int64"),
        }
    )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.ones(4, dtype=np.float64), WeightKind.DESIGN)},
    )


@pytest.mark.requires_us
def test_real_engine_batched_base_simulation_matches_unbatched(monkeypatch) -> None:
    """Compare these household-local targets at batch sizes 1, 2 and unbatched."""

    builder = _load_builder_module()
    monkeypatch.setattr(builder, "US_JCT_TAX_EXPENDITURE_REFORMS", ())
    frame = _real_engine_frame()
    targets = (
        _soi("agi_amount", "adjusted_gross_income"),
        _soi(
            "agi_under_50k_count",
            "count",
            agi_lower_bound="-inf",
            agi_upper_bound="50000",
        ),
        _soi("single_returns", "count", filing_status="Single"),
        _soi("wages", "employment_income"),
        _soi("filer_individuals", "tax_filer_individual_count"),
        _population_age("pop_18_to_65", 18, 65),
        _population_age("ca_pop_65_plus", 65, "inf", state_fips="06"),
        _variable("snap_amount", base_variable="snap"),
        _variable("ssi_recipients", base_variable="ssi", measure_mode="indicator_sum"),
        _spec("ca_state_income_tax", "state_income_tax", {"state_fips": "06"}),
        _spec("md_state_income_tax", "state_income_tax", {"state_fips": "24"}),
        _spec("eitc", "fixture"),
    )

    results = {}
    for batch_size in (None, 2, 1):
        target_frame, registry, compilation = builder._materialize_target_frame(
            frame,
            targets,
            maximum_microsim_batch_size=batch_size,
        )
        assert compilation["dropped_target_names"] == []
        results[batch_size] = (target_frame.table("household"), registry.version)

    unbatched, unbatched_version = results[None]
    # The fixture reaches real engine output: taxes, credits and transfers.
    for column in (
        "income_tax",
        "eitc",
        "snap_amount",
        "ssi_recipients",
        "ca_state_income_tax",
        "md_state_income_tax",
    ):
        assert unbatched[column].to_numpy().any(), column
    for batch_size in (2, 1):
        household, version = results[batch_size]
        pd.testing.assert_frame_equal(household, unbatched, check_exact=True)
        assert version == unbatched_version


@pytest.mark.requires_us
def test_real_engine_base_batches_build_no_system_and_release_engines(
    monkeypatch,
) -> None:
    """Check variable-file loads and live engines across the base batch loop."""

    import gc

    from policyengine_core.taxbenefitsystems import TaxBenefitSystem

    builder = _load_builder_module()
    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    variable_file_loads = [0]
    load_variable_file = TaxBenefitSystem.add_variables_from_file

    def counting_load(self, file_path):
        variable_file_loads[0] += 1
        return load_variable_file(self, file_path)

    monkeypatch.setattr(TaxBenefitSystem, "add_variables_from_file", counting_load)

    def alive_microsimulations() -> int:
        gc.collect()
        return sum(1 for obj in gc.get_objects() if isinstance(obj, Microsimulation))

    frame = _real_engine_frame()
    targets = (
        _soi("agi_amount", "adjusted_gross_income"),
        _variable("snap_amount", base_variable="snap"),
    )
    system = CountryTaxBenefitSystem()
    # Sanity: the counter sees a real build.
    assert variable_file_loads[0] > 1_000
    unbatched, _ = builder._materialize_base_simulation_columns(
        frame,
        targets,
        system=system,
        microsimulation_cls=Microsimulation,
        maximum_microsim_batch_size=None,
    )
    alive_before = alive_microsimulations()
    loads_before_batches = variable_file_loads[0]
    batched, receipt = builder._materialize_base_simulation_columns(
        frame,
        targets,
        system=system,
        microsimulation_cls=Microsimulation,
        maximum_microsim_batch_size=1,
    )

    assert receipt["batches"] == frame.n("household")
    assert variable_file_loads[0] == loads_before_batches, (
        f"{receipt['batches']} base batches loaded "
        f"{variable_file_loads[0] - loads_before_batches} variable files: a "
        "tax-benefit system was built per batch (microcosm#456)"
    )
    assert alive_microsimulations() <= alive_before
    assert list(batched) == list(unbatched)
    for column, values in unbatched.items():
        np.testing.assert_array_equal(batched[column], values)


def _medicaid_frame() -> Frame:
    """Four California parents with children; the first parent's earnings
    exceed the other three parents' earnings."""

    n_households = 4
    household_ids = np.arange(1, n_households + 1, dtype="int64")
    persons = pd.DataFrame(
        {
            "person_id": np.arange(1, 2 * n_households + 1, dtype="int64"),
            "person_household_id": np.repeat(household_ids, 2),
            "person_tax_unit_id": np.repeat(household_ids * 10, 2),
            "person_spm_unit_id": np.repeat(household_ids * 100, 2),
            "person_family_id": np.repeat(household_ids * 1000, 2),
            "person_marital_unit_id": np.arange(1, 2 * n_households + 1, dtype="int64")
            * 10_000,
            "age": np.tile([30.0, 5.0], n_households),
            "employment_income_before_lsr": np.asarray(
                [400_000.0, 0.0, 12_000.0, 0.0, 15_000.0, 0.0, 5_000.0, 0.0]
            ),
        }
    )
    tables = {"person": persons}
    for entity in ("tax_unit", "spm_unit", "family", "marital_unit"):
        ids = np.unique(persons[f"person_{entity}_id"].to_numpy())
        tables[entity] = pd.DataFrame({f"{entity}_id": ids})
    tables["household"] = pd.DataFrame(
        {
            "household_id": household_ids,
            "state_fips": np.full(n_households, 6, dtype="int64"),
        }
    )
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.asarray([1.0, 2.0, 3.0, 4.0]), WeightKind.DESIGN)},
    )


@pytest.mark.requires_us
def test_real_engine_refuses_a_batched_medicaid_cost_target() -> None:
    """Compare guarded materialization with unguarded one-household engines."""

    from policyengine_us import CountryTaxBenefitSystem, Microsimulation

    builder = _load_builder_module()
    frame = _medicaid_frame()
    targets = (_variable("medicaid_cost_total", base_variable="medicaid_cost"),)
    system = CountryTaxBenefitSystem()
    whole, receipt = builder._materialize_base_simulation_columns(
        frame,
        targets,
        system=system,
        microsimulation_cls=Microsimulation,
        maximum_microsim_batch_size=None,
    )
    assert receipt["batches"] == 1
    assert receipt["population_aggregate_variables_checked"] == []
    pool_cost = whole["medicaid_cost_total"]
    single_batch, receipt = builder._materialize_base_simulation_columns(
        frame,
        targets,
        system=system,
        microsimulation_cls=Microsimulation,
        maximum_microsim_batch_size=frame.n("household"),
    )
    np.testing.assert_array_equal(single_batch["medicaid_cost_total"], pool_cost)
    assert receipt["batches"] == 1

    unguarded_cost = []
    enrolled = 0
    aggregate_periods = []
    for position in range(frame.n("household")):
        batch_frame = builder._select_households_by_position(
            frame, np.asarray([position], dtype=np.int64)
        )
        simulation = Microsimulation(
            dataset=builder._dataset_from_frame(
                batch_frame, assert_no_formula_owned_columns=False
            )
        )
        try:
            batch_columns = builder._base_simulation_household_columns(
                batch_frame, targets, simulation=simulation, system=system
            )
            unguarded_cost.extend(batch_columns["medicaid_cost_total"])
            enrolled += int(
                np.asarray(
                    simulation.calculate("medicaid_enrolled", builder.PERIOD)
                ).sum()
            )
            aggregate_periods.append(
                builder._engine_known_periods(
                    simulation, builder.US_POPULATION_AGGREGATE_VARIABLES
                )
            )
        finally:
            builder.release_engine_simulation(simulation)
    unguarded_cost = np.asarray(unguarded_cost)
    weights = frame.weights_for("household").values
    assert enrolled > 0
    assert aggregate_periods[0] == set()
    assert (
        "medicaid_slcsp_state_denominator",
        str(builder.PERIOD),
    ) in aggregate_periods[1]
    assert pool_cost[0] == 0
    assert np.all(pool_cost[1:] > 0)
    assert np.any(unguarded_cost != pool_cost)
    assert np.dot(unguarded_cost, weights) > np.dot(pool_cost, weights)
    print(
        f"Medicaid control: enrollees={enrolled}, "
        f"whole_pool_weighted={np.dot(pool_cost, weights):.9g}, "
        f"unguarded_batch_size_1_weighted={np.dot(unguarded_cost, weights):.9g}, "
        f"changed_households={np.count_nonzero(unguarded_cost != pool_cost)}"
    )

    for batch_size in (1, 2):
        first_aggregate_batch = 2 if batch_size == 1 else 1
        with pytest.raises(
            ValueError,
            match=(
                rf"household batch {first_aggregate_batch}/"
                rf"{frame.n('household') // batch_size} computed .*"
                rf"medicaid_slcsp_state_denominator@{builder.PERIOD}"
            ),
        ):
            builder._materialize_base_simulation_columns(
                frame,
                targets,
                system=system,
                microsimulation_cls=Microsimulation,
                maximum_microsim_batch_size=batch_size,
            )


_POPULATION_AGGREGATE_MARKER = re.compile(
    r"\bsum_by_state\(|\bMicroSeries\(|\.decile_rank\(|\bquantile\(|\bpercentile\("
    r"|\bnp\.(?:sum|mean|median|average|nansum|nanmean|nanmedian|sort|argsort"
    r"|cumsum|isin|in1d|unique|bincount|searchsorted)\((?![^\n]*\baxis\s*=)"
    r"|\.(?:sum|mean|max|min|median)\(\s*\)"
)
_WEIGHT_READ = re.compile(r"[\"'](\w+_weight)[\"']")


def _is_variable_class(node: ast.AST) -> bool:
    return isinstance(node, ast.ClassDef) and any(
        getattr(base, "id", None) == "Variable" for base in node.bases
    )


def _engine_population_aggregate_sources(
    root: Path | None = None,
    *,
    source_digests: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Scan package Variable classes and module helpers for aggregation markers.

    Pattern matches identify formulas to inspect; they do not prove household
    locality or recognize every possible cross-record computation.
    """
    if root is None:
        spec = importlib.util.find_spec("policyengine_us")
        root = Path(next(iter(spec.submodule_search_locations)))
    modules = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        if "tests" in relative.parts or path.name.startswith("test_"):
            continue
        source = path.read_text()
        modules.append((source, ast.parse(source)))

    helpers: dict[str, list[str]] = {}
    for source, tree in modules:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                helpers.setdefault(node.name, []).append(
                    ast.get_source_segment(source, node)
                )
    reaching = {
        name
        for name, bodies in helpers.items()
        if any(_POPULATION_AGGREGATE_MARKER.search(body) for body in bodies)
    }
    grew = True
    while grew:
        grew = False
        for name, bodies in helpers.items():
            if name not in reaching and any(
                re.search(rf"\b{other}\(", body)
                for body in bodies
                for other in reaching
            ):
                reaching.add(name)
                grew = True

    found: dict[str, list[str]] = {}
    for source, tree in modules:
        for node in ast.walk(tree):
            if not _is_variable_class(node):
                continue
            body = ast.get_source_segment(source, node)
            evidence = [
                match.group(0) for match in _POPULATION_AGGREGATE_MARKER.finditer(body)
            ]
            evidence += [
                f"{helper}()"
                for helper in sorted(reaching)
                if re.search(rf"\b{helper}\(", body)
            ]
            if not node.name.endswith("_weight"):
                evidence += _WEIGHT_READ.findall(body)
            if evidence:
                found[node.name] = evidence
                if source_digests is not None:
                    sources = [body]
                    pending = [
                        helper
                        for helper in reaching
                        if re.search(rf"\b{helper}\(", body)
                    ]
                    seen = set()
                    while pending:
                        helper = pending.pop()
                        if helper in seen:
                            continue
                        seen.add(helper)
                        sources.extend(helpers[helper])
                        pending.extend(
                            other
                            for helper_body in helpers[helper]
                            for other in reaching - seen
                            if re.search(rf"\b{other}\(", helper_body)
                        )
                    source_digests[node.name] = hashlib.sha256(
                        "\n".join(sorted(sources)).encode()
                    ).hexdigest()
    return found


def test_population_aggregate_scan_follows_helpers_and_cross_record_markers(
    tmp_path,
) -> None:
    (tmp_path / "variables").mkdir()
    (tmp_path / "tools").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "variables" / "allocation.py").write_text(
        "def _state_total(values, state):\n"
        "    return np.sum(values[state == 6])\n"
        "def _axis_total(values):\n"
        "    return np.sum(values, axis=1)\n"
        "class helper_share(Variable):\n"
        "    def formula(person, period):\n"
        "        return _state_total(values, state)\n"
        "class direct_share(Variable):\n"
        "    def formula(person, period):\n"
        "        return values / np.sum(values)\n"
        "class axis_local(Variable):\n"
        "    def formula(person, period):\n"
        "        return _axis_total(values)\n"
        "class tools_rank(Variable):\n"
        "    def formula(person, period):\n"
        "        return _rank(values)\n"
    )
    (tmp_path / "tools" / "rank.py").write_text(
        "def _rank(values):\n"
        "    return _ordered(values)\n"
        "def _ordered(values):\n"
        "    return np.argsort(values)\n"
    )
    (tmp_path / "tests" / "fixture.py").write_text(
        "class ignored_test_fixture(Variable):\n"
        "    def formula(person, period):\n"
        "        return values.sum()\n"
    )
    markers = {
        "method_sum": "values.sum()",
        "method_mean": "values.mean()",
        "method_max": "values.max()",
        "method_min": "values.min()",
        "method_median": "values.median()",
        "join_isin": "np.isin(values, ids)",
        "join_in1d": "np.in1d(values, ids)",
        "join_unique": "np.unique(values)",
        "join_bincount": "np.bincount(values)",
        "join_searchsorted": "np.searchsorted(values, ids)",
    }
    (tmp_path / "variables" / "markers.py").write_text(
        "\n".join(
            f"class {name}(Variable):\n"
            "    def formula(person, period):\n"
            f"        return {expression}\n"
            for name, expression in markers.items()
        )
    )
    found = _engine_population_aggregate_sources(tmp_path)
    assert set(found) == {"direct_share", "helper_share", "tools_rank", *markers}
    assert found["helper_share"] == ["_state_total()"]
    assert found["tools_rank"] == ["_rank()"]


# These inspected 2.2.1 formulas compare a record's status or geography with
# parameter lists, literal lists or enum members, rather than another record.
_CONSTANT_MEMBERSHIP_FORMULAS = {
    "additional_senior_deduction_eligible_person",
    "ak_ccap_rate_region",
    "al_ccsp_region",
    "american_worker_tax_rebate_eligible",
    "ar_sra_zone",
    "ca_ala_general_assistance_immigration_status_eligible",
    "ca_calworks_child_care_immigration_status_eligible_person",
    "ca_cc_general_assistance_immigration_status_eligible",
    "ca_marin_general_relief_immigration_status_eligible_person",
    "ca_oc_general_relief_immigration_status_eligible",
    "ca_riv_general_relief_immigration_status_eligible",
    "ca_sbd_general_relief_immigration_status_eligible",
    "ca_scc_general_assistance_immigration_status_eligible",
    "ca_sf_caap_immigration_status_eligible",
    "ca_smc_general_assistance_immigration_status_eligible_person",
    "ca_snap_immigration_status_eligible",
    "ca_tanf_immigration_status_eligible_person",
    "ca_tanf_region1",
    "co_ccap_fpg_eligible",
    "ct_c4k_region",
    "dc_ccsp_immigration_status_eligible_person",
    "dc_ccsp_is_full_time",
    "dc_tanf_immigration_status_eligible_person",
    "ga_caps_zone",
    "id_iccp_county_cluster",
    "il_aabd_area",
    "il_aabd_immigration_status_eligible_person",
    "il_ccap_county_group",
    "il_dhs_csfp_county_eligible",
    "il_hfs_immigration_status_eligible",
    "il_tanf_county_group",
    "il_tanf_immigration_status_eligible_person",
    "in_ny_mctd_zone_2",
    "in_nyc",
    "is_aca_ptc_immigration_status_eligible",
    "is_basic_health_program_eligible",
    "is_basic_health_program_immigration_status_eligible",
    "is_ca_medicaid_immigration_status_eligible",
    "is_ccdf_immigration_eligible_child",
    "is_chip_fcep_eligible_person",
    "is_citizen_or_legal_immigrant",
    "is_in_snap_abawd_waived_area",
    "is_medicaid_immigration_status_eligible",
    "is_snap_gross_test_full_income_count_alien",
    "is_snap_immigration_status_eligible",
    "is_snap_prorated_income_member",
    "is_snap_state_discretion_ineligible_alien",
    "is_ssi_qualified_noncitizen",
    "ks_ccap_rate_group",
    "ks_dcf_csfp_county_eligible",
    "ks_tanf_county_group",
    "ky_ccap_rate_region",
    "ma_ccfa_immigration_status_eligible",
    "ma_ccfa_region",
    "ma_dese_csfp_county_eligible",
    "md_ccs_region",
    "me_ccap_region",
    "meets_ctc_child_identification_requirements",
    "meets_ctc_identification_requirements",
    "mo_ccs_region",
    "ms_ccpp_facility_location",
    "mt_tanf_immigration_status_eligible_person",
    "ne_child_care_subsidy_location",
    "nh_ccap_immigration_status_eligible_person",
    "ny_ccap_county_group",
    "oh_ccap_county_rate_category",
    "or_healthier_oregon_immigration_status_eligible",
    "overtime_income_deduction_ssn_requirement_met",
    "pa_ccw_region",
    "pa_ccw_stepparent_county_group",
    "pa_tanf_county_group",
    "sc_ccap_geography",
    "sd_cca_region",
    "state_group",
    "state_itemized_deductions",
    "state_standard_deduction",
    "taxsim_state_agi",
    "tip_income_deduction_ssn_requirement_met",
    "tn_ccap_county_tier",
    "trump_dividend_eligible",
    "tx_ccs_workforce_board_region",
    "va_ccsp_income_eligible",
    "va_ccsp_locality_group",
    "va_ccsp_ready_region",
    "va_medicaid_lifc_locality_group",
    "va_tanf",
    "va_tanf_grant_standard",
    "va_tanf_need_standard",
    "va_tanf_up_grant_standard",
    "wa_rca_immigration_status_eligible",
    "wa_tanf_immigration_status_eligible",
    "wa_wccc_center_region",
    "wa_wccc_region",
    "wic_income_limit",
}


@pytest.mark.requires_us
def test_population_aggregate_list_matches_installed_engine_sources() -> None:
    """Pin the guard list to weight reads and aggregation markers in US sources.

    The scan covers the package except tests, including module helpers beside
    Variable classes and in tools. It is not a proof of household locality.
    """
    builder = _load_builder_module()
    source_digests = {}
    found = _engine_population_aggregate_sources(source_digests=source_digests)
    triaged = {name: {"np.isin("} for name in _CONSTANT_MEMBERSHIP_FORMULAS}
    # These search sorted parameter brackets or literal earnings thresholds.
    triaged.update(
        {
            name: {"np.searchsorted("}
            for name in (
                "aca_required_contribution_percentage",
                "ca_premium_subsidy_applicable_percentage",
                "md_premium_assistance_target_contribution_percentage",
                "nm_premium_assistance_target_contribution_percentage",
                "substitution_elasticity",
            )
        }
    )
    triaged.update(
        {
            # np.unique groups the loop by indexing year; each person still
            # reads their own earnings and the year's common wage index.
            "ss_aime": {"_compute_aime()"},
            # The helper groups a packaged county schedule by effective year,
            # then each household joins its own county and bedroom count.
            "hud_utility_allowance": {"utility_allowance_schedule()"},
            # A lexical collision: ndarray.reshape matches the converter's
            # module helper named reshape. Actual sums use axis=1 on brackets.
            "ny_supplemental_tax": {"get_previous_threshold()"},
            # These cross-person joins are permitted only after the frame
            # precondition proves positive claiming IDs stay in the household.
            "medicaid_has_known_claiming_tax_unit": {"np.isin("},
            "medicaid_household_income": {
                "medicaid_claiming_tax_unit_value()",
                "medicaid_external_claimed_sum()",
            },
            "medicaid_household_size": {
                "medicaid_claiming_tax_unit_value()",
                "medicaid_external_claimed_sum()",
            },
        }
    )
    assert set(found) == set(builder.US_POPULATION_AGGREGATE_VARIABLES) | set(
        triaged
    ), found
    for name, evidence in triaged.items():
        assert set(found[name]) == evidence, (name, found[name])
    # Pin the reviewed exceptions' class and reachable helper source bodies;
    # adding another aggregate to an allowed name still requires a fresh review.
    triaged_digest = hashlib.sha256(
        "\n".join(f"{name}:{source_digests[name]}" for name in sorted(triaged)).encode()
    ).hexdigest()
    assert triaged_digest == (
        "f26eb560e474207fc1fa1d8828b62ba4791fb763089ea2ce7a85b57a259f2755"
    ), triaged_digest


def test_refusal_reads_a_value_held_only_on_baseline() -> None:
    """Target materialization and post-export scoring share one walker. It
    also reads a ``reform=`` engine's ``baseline`` simulation, which holds its
    own values and is not among the engine's branches."""
    builder = _load_builder_module()

    def engine(known, **extra):
        return SimpleNamespace(
            get_holder=lambda name: SimpleNamespace(
                get_known_periods=lambda: known.get(name, [])
            ),
            branches={},
            **extra,
        )

    root = engine({}, baseline=engine({"medicaid_slcsp_state_denominator": ["2024"]}))
    with pytest.raises(
        ValueError, match=r"computed medicaid_slcsp_state_denominator@2024"
    ):
        builder._refuse_batch_population_aggregates(
            root, batch=1, batches=3, n_households=6
        )
