"""Invented CREATE/FILTER/masked-rewrite coverage, no engine or native data."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import input_coverage_profile as profile
from microcosm.build.us_runtime import population_input_coverage as coverage
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.graph import (
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Owned,
    Ownership,
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.population import token_for_dtype

BIG = 2**53 + 101
SCHEMA = EntitySchema(group_entities=("household", "tax_unit"))


def _frame(spec):
    household_ids = np.array([10, 20, 30, 40], dtype=np.int64)
    household_members = [10, 10, 20, 30, 30, 40]
    if spec.get("cross_origin"):
        household_members[0] = 20
    people = pd.DataFrame(
        {
            "person_id": np.arange(BIG, BIG + 6, dtype=np.int64),
            "person_household_id": np.array(household_members, dtype=np.int64),
            "person_tax_unit_id": np.array([1, 1, 2, 3, 3, 4], dtype=np.int64),
            "person_support_channel": pd.Series(
                ["acs", "acs", "asec"] * 2, dtype="string"
            ),
            "person_support_clone_index": np.array([0, 0, 0, 1, 1, 1], dtype=np.int64),
            "person_spine_source_id": np.array([1, 2, 3, 1, 2, 3], dtype=np.int64),
            "is_detail": np.array([False] * 3 + [True] * 3),
            "age": pd.Series([0.0, None, 40.0] * 2, dtype="float64"),
            "taxable_interest_income": pd.Series([7.0, None, 0.0] * 2, dtype="float64"),
            "bank_account_assets": pd.Series(
                [0, pd.NA, 10, 0, pd.NA, 10], dtype="Int64"
            ),
        }
    )
    if spec.get("unknown_origin"):
        people.loc[0, "person_support_channel"] = pd.NA
    households = pd.DataFrame(
        {
            "household_id": household_ids,
            "household_support_channel": pd.Series(["acs", "asec"] * 2, dtype="string"),
            "household_support_clone_index": np.array([0, 0, 1, 1], dtype=np.int64),
            "household_spine_source_id": np.array([10, 20, 10, 20], dtype=np.int64),
            "household_weight": np.full(4, 999.0),
            "net_worth": np.array([0.0, 10.0, 0.0, 10.0]),
        }
    )
    if not spec.get("missing_block"):
        households["census_block_geoid"] = pd.Series(
            [
                "010010001001001",
                "020010001001001",
                "010010001001002",
                "020010001001002",
            ],
            dtype="string",
        )
    tax_units = pd.DataFrame(
        {
            "tax_unit_id": np.arange(1, 5, dtype=np.int64),
            "tax_unit_support_channel": pd.Series(["acs", "asec"] * 2, dtype="string"),
            "tax_unit_support_clone_index": np.array([0, 0, 1, 1], dtype=np.int64),
            "tax_unit_spine_source_id": np.array([1, 2, 1, 2], dtype=np.int64),
        }
    )
    if spec.get("float_source_ids"):
        households["household_spine_source_id"] = households[
            "household_spine_source_id"
        ].astype("float64")
    if spec.get("ambiguous_age"):
        households["age"] = np.array([99.0] * 4)
    if spec.get("invalid_numeric"):
        people.loc[0, "age"] = np.inf
    return Frame(
        {"person": people, "household": households, "tax_unit": tax_units},
        SCHEMA,
        {"household": Weights(np.array([0.0, 2.0, 0.0, 3.0]), WeightKind.DESIGN)},
        pd.Series(["invented"] * 6, name="stratum"),
        metadata={"fixture": "invented-only"},
    )


class InventedCreate(KernelBase):
    ref = "test.coverage.create.v1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        spec = json.loads(context.sources["invented"].read_bytes())
        return KernelResult(frame=_frame(spec))


class InventedKeep(KernelBase):
    ref = "test.coverage.keep.v1"
    capabilities = Capabilities(
        determinism=Determinism.DETERMINISTIC, structural=StructuralDelta.FILTER
    )

    def run(self, context):
        ids = context.tables["person"]["person_id"]
        return KernelResult(keep=pd.Series(True, index=pd.Index(ids), dtype="bool"))


class InventedMasked(KernelBase):
    ref = "test.coverage.masked.v1"
    capabilities = Capabilities(determinism=Determinism.DETERMINISTIC)

    def run(self, context):
        people = context.tables["person"]
        ids = people.loc[people.is_detail, "person_id"]
        return KernelResult(
            columns={
                ("person", "taxable_interest_income"): pd.Series(
                    ([np.nan] * 3 if context.params["absent"] else [0.0, 100.0, 200.0]),
                    index=pd.Index(ids),
                    dtype="float64",
                )
            }
        )


@pytest.fixture
def actual(tmp_path):
    def build(**spec):
        source = tmp_path / "invented.json"
        source.write_text(json.dumps(spec))
        frame = _frame(spec)
        structural = {(e, frame.schema.entity_id_column(e)) for e in frame.entities}
        structural.update(
            ("person", frame.schema.membership_column(e))
            for e in frame.schema.group_entities
        )
        create = Node(
            id="create",
            kernel=InventedCreate.ref,
            structural=StructuralDelta.CREATE,
            sources=("invented",),
            outputs=tuple(
                Owned(
                    entity, column, token_for_dtype(frame.table(entity)[column].dtype)
                )
                for entity in frame.entities
                for column in frame.table(entity)
                if (entity, column) not in structural
            ),
        )
        kept = Node(
            id="kept",
            kernel=InventedKeep.ref,
            structural=StructuralDelta.FILTER,
            base="create",
            inputs=(Slice("person", ("is_detail",)),),
        )
        masked = Node(
            id="masked",
            kernel=InventedMasked.ref,
            population="kept",
            params={"absent": bool(spec.get("absent_interest"))},
            inputs=(Slice("person", ("is_detail",)),),
            outputs=(
                Owned(
                    "person",
                    "taxable_interest_income",
                    "float64",
                    rows="is_detail",
                    rewrite=True,
                    ownership=Ownership.ABSENT
                    if spec.get("absent_interest")
                    else Ownership.PRODUCED,
                ),
            ),
        )
        compiled = compile_graph(
            Graph(
                country="us",
                sources=(SourceRef("invented", "raw-bytes-v1"),),
                nodes=(create, kept, masked),
            )
        )
        kernels = KernelRegistry()
        kernels.register(InventedCreate())
        kernels.register(InventedKeep())
        kernels.register(InventedMasked())
        store = ContentStore(tmp_path / "store")
        observed = {}
        manifest = run_graph(
            compiled,
            sources={"invented": source},
            store=store,
            kernels=kernels,
            _population_observer=lambda node, population: observed.__setitem__(
                node, population
            ),
        )
        return SimpleNamespace(
            population=observed["masked"],
            observed=observed,
            manifest=manifest,
            compiled=compiled,
            store=store,
            kernels=kernels,
            source=source,
        )

    return build


def _diagnose(run):
    return coverage.diagnose_us_input_coverage(
        run.population, compiled=run.compiled, manifest=run.manifest
    )


def _input(report, name, entity="person"):
    return next(
        row for row in report.inputs if (row.name, row.entity) == (name, entity)
    )


def test_closed_profiles_match_manifest_without_changing_historical_default():
    path = (
        Path(profile.__file__).parents[1]
        / "us/release_input_coverage_manifest.json"
    )
    manifest = json.loads(path.read_text())
    expected = tuple(
        name for name, row in manifest["columns"].items() if row["status"] == "required"
    )
    assert profile.required_us_inputs() == expected
    assert (
        len(expected) == 163
        and len(profile.required_us_inputs(profile.USInputProfile.NATIONAL_CD)) == 161
    )
    assert set(expected) - set(profile.NATIONAL_CD_REQUIRED_INPUTS) == {
        "block_geoid",
        "tract_geoid",
    }
    assert "employment_income_last_year" not in expected
    with pytest.raises(TypeError, match="PROFILE_TYPE"):
        profile.required_us_inputs("us_national_cd_161_v1")


def test_actual_masked_coverage_separates_origin_clone_grain_and_unknownness(actual):
    run = actual()
    report = _diagnose(run)
    # These are actual executor owners, not a fixture-authored expected map.
    assert run.population.version == "kept"
    assert run.population.owners["person", "age"] == "kept"
    assert run.population.owners["person", "person_id"] == "kept"
    assert run.population.owners["person", "taxable_interest_income"] == "masked"
    created = coverage.diagnose_us_input_coverage(
        run.observed["create"], compiled=run.compiled, manifest=run.manifest
    )
    assert created.version == "create"
    assert _input(created, "age").owner == "create"
    assert report.manifest_key == run.manifest.key
    assert report.profile is profile.USInputProfile.NATIONAL_CD
    assert report.profile_reference_sha256 == profile.MANIFEST_SHA256
    document = json.loads(report.to_bytes())
    assert document["protocol"] == "microcosm.us.input_coverage_diagnostic.v1"
    assert document["release_eligible"] is False
    assert {group.entity for group in report.groups} == {
        "person",
        "household",
        "tax_unit",
    }
    first = report.groups[0]
    assert first.ordered_ids == (BIG, BIG + 1) and first.id_dtype == "int64"
    interest = _input(report, "taxable_interest_income")
    assert interest.owner == "masked" and interest.writer_mask == "is_detail"
    arm = {
        (report.groups[c.group].origin, report.groups[c.group].clone): c
        for c in interest.counts
    }
    assert (arm["acs", 0].known, arm["acs", 0].unknown, arm["acs", 0].carried_rows) == (
        1,
        1,
        2,
    )
    assert (arm["acs", 1].known, arm["acs", 1].writer_rows) == (2, 2)
    assert run.population.frame.person.loc[0, "taxable_interest_income"] == 7.0
    assert pd.isna(run.population.frame.person.loc[1, "taxable_interest_income"])
    assets = _input(report, "bank_account_assets")
    assert (
        sum(c.known for c in assets.counts) == 4
        and sum(c.unknown for c in assets.counts) == 2
    )
    assert all(
        c.not_applicable == 0 and c.applicability_unresolved > 0 for c in assets.counts
    )
    assert (
        _input(report, "first_home_mortgage_balance", None).declaration
        == "missing_unresolved_grain"
    )
    assert (
        not report.release_eligible
        and not report.source_ancestry_verified
        and not report.statistical_signal_verified
    )
    producer = next(p for p in report.producers if p.node == "create")
    assert producer.declared_sources == (("invented", "raw-bytes-v1"),)
    assert producer.key == run.manifest.node("create").key


def test_required_replay_preserves_coverage_and_manifest_identity(actual):
    run = actual()
    first = _diagnose(run)
    observed = {}
    warm = run_graph(
        run.compiled,
        sources={"invented": run.source},
        store=run.store,
        kernels=run.kernels,
        resume="require",
        _population_observer=lambda node, pop: observed.__setitem__(node, pop),
    )
    assert all(row.hit for row in warm.nodes.values())
    second = coverage.diagnose_us_input_coverage(
        observed["masked"], compiled=run.compiled, manifest=warm
    )
    # Physical nullable backing may normalize at the named store boundary.
    assert replace(first, population_storage_sha256="") == replace(
        second, population_storage_sha256=""
    )


@pytest.mark.parametrize("part", ["owners", "manifest", "ledger", "compiled", "values"])
def test_inconsistent_population_manifest_or_compiler_refuses(actual, part):
    run = actual()
    if part == "owners":
        owners = dict(run.population.owners)
        owners["person", "age"] = "masked"
        run.population = replace(run.population, owners=owners)
    elif part == "manifest":
        run.manifest = replace(
            run.manifest, nodes={"create": run.manifest.node("create")}
        )
    elif part == "ledger":
        run.manifest = replace(run.manifest, mass_ledgers={})
    elif part == "compiled":
        run.compiled = replace(run.compiled, owners={})
    else:
        tables = {
            e: run.population.frame.table(e).copy(deep=True)
            for e in run.population.frame.entities
        }
        tables["person"].loc[0, "age"] = 71.0
        frame = Frame(
            tables,
            SCHEMA,
            {"household": run.population.frame.weights_for("household")},
            run.population.frame.strata,
            metadata=run.population.frame.metadata,
            mass_log=run.population.frame.mass_log,
        )
        run.population = replace(run.population, frame=frame)
    with pytest.raises((ValueError, KeyError)):
        _diagnose(run)


@pytest.mark.parametrize(
    "flag,reason",
    [
        ("unknown_origin", "ORIGIN"),
        ("cross_origin", "MEMBERSHIP_ORIGIN_CLONE"),
        ("float_source_ids", "ID_DTYPE"),
    ],
)
def test_source_role_and_membership_are_checked_on_actual_rows(actual, flag, reason):
    with pytest.raises(ValueError, match=reason):
        _diagnose(actual(**{flag: True}))


def test_zero_weight_origin_and_zero_observations_remain_known(actual):
    run = actual()
    report = _diagnose(run)
    assert run.population.frame.weights_for("household").values.tolist() == [
        0.0,
        2.0,
        0.0,
        3.0,
    ]
    weights = _input(report, "household_weight", "household")
    assert weights.declaration == "typed_weight_carrier"
    assert sum(c.known for c in weights.counts) == 4
    assert sum(c.known for c in _input(report, "net_worth", "household").counts) == 4
    assert not report.statistical_signal_verified


def test_profile_omissions_do_not_omit_assigned_block_diagnostic(actual):
    report = _diagnose(actual(missing_block=True))
    assert (
        "block_geoid" not in report.missing_inputs
        and "tract_geoid" not in report.missing_inputs
    )
    assert report.block_storage_issues == ("missing_assigned_block",)
    assert report.assigned_block.name == "census_block_geoid"


def test_duplicate_grains_are_rejected_by_the_actual_frame_contract():
    with pytest.raises(
        ValueError, match="^Column names must be globally unique across entity tables "
    ) as error:
        _frame({"ambiguous_age": True})
    assert "'age' on ['person', 'household']" in str(error.value)


def test_nonfinite_values_and_missing_grains_remain_unresolved(actual):
    report = _diagnose(actual(invalid_numeric=True))
    assert report.ambiguous_grains == ()
    age = _input(report, "age")
    assert sum(c.invalid for c in age.counts) == 1
    assert sum(c.known for c in age.counts) == 3
    assert sum(c.unknown for c in age.counts) == 2
    missing = _input(report, "first_home_mortgage_balance", None)
    assert missing.entity is None and missing.counts == ()
    assert missing.declaration == "missing_unresolved_grain"


def test_last_manifest_borrow_mutation_refuses(actual):
    run = actual()
    fired = False

    def callback(frame, event, arg):
        nonlocal fired
        if (
            event == "return"
            and frame.f_code is type(run.manifest).population.__code__
            and not fired
        ):
            fired = True
            run.population.frame.person["bank_account_assets"].array._data[0] = 123

    old = sys.getprofile()
    try:
        sys.setprofile(callback)
        with pytest.raises(ValueError):
            _diagnose(run)
    finally:
        sys.setprofile(old)
    assert fired


def test_declared_absence_is_not_an_applicability_exemption(actual):
    report = _diagnose(actual(absent_interest=True))
    interest = _input(report, "taxable_interest_income")
    assert interest.declaration == Ownership.ABSENT.value
    detail = [c for c in interest.counts if report.groups[c.group].clone == 1]
    assert sum(c.known for c in detail) == 0
    assert sum(c.unknown for c in detail) == 3
    assert sum(c.applicability_unresolved for c in detail) == 3
    assert sum(c.not_applicable for c in detail) == 0


@pytest.mark.parametrize(
    "entity,column,wrong_owner",
    [
        ("person", "age", "create"),
        ("person", "person_id", "create"),
        ("person", "person_household_id", "create"),
        ("household", "household_id", "create"),
        ("person", "taxable_interest_income", "kept"),
    ],
)
def test_complete_current_ownership_refuses_wrong_carrier_or_writer(
    actual, entity, column, wrong_owner
):
    run = actual()
    assert run.population.owners[entity, column] != wrong_owner
    owners = dict(run.population.owners)
    owners[entity, column] = wrong_owner
    run.population = replace(run.population, owners=owners)
    with pytest.raises(ValueError, match="^US_INPUT_COVERAGE_CURRENT_OWNERS$"):
        _diagnose(run)


@pytest.mark.parametrize("version", ["masked", "missing"])
def test_population_version_must_be_an_actual_structural_version(actual, version):
    run = actual()
    run.population = replace(run.population, version=version)
    with pytest.raises(ValueError, match="^US_INPUT_COVERAGE_POPULATION_VERSION$"):
        _diagnose(run)


def test_unclaimed_physical_column_is_not_accepted_as_a_structural_carrier(actual):
    run = actual()
    original = run.population.frame
    tables = {
        entity: original.table(entity).copy(deep=True) for entity in original.entities
    }
    tables["person"]["undeclared_probe"] = np.zeros(
        original.n("person"), dtype=np.int64
    )
    frame = Frame(
        tables,
        original.schema,
        {entity: original.weights_for(entity) for entity in original.weighted_entities},
        original.strata,
        metadata=original.metadata,
        mass_log=original.mass_log,
    )
    owners = dict(run.population.owners)
    owners["person", "undeclared_probe"] = run.population.version
    run.population = replace(run.population, frame=frame, owners=owners)
    with pytest.raises(ValueError, match="^US_INPUT_COVERAGE_CURRENT_OWNERS$"):
        _diagnose(run)
