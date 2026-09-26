"""Invented export controls, not qualification of an actual model or release."""

import hashlib
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.gates import TargetFitRequirement
from microcosm.build.us_runtime import snap_release_acceptance as subject
from microcosm.build.us_runtime.h5_io import write_nullable_us_h5
from microcosm.calibrate.registry import TargetRegistry, TargetSpec
from microcosm.calibrate.solve import TargetDiagnostic
from microcosm.frame import Frame, WeightKind, Weights, put_frame_table
from microcosm.frame.accounting import wsum
from microcosm.frame.rules import ExportContract
from microcosm.frame.units import US_SCHEMA


class FrameSeries:
    """Test-only weighted protocol double backed by maintained Frame accounting.

    It does not implement a policy model. Production demands MicroSeries; this
    double only verifies grain, masking and preservation of weighted operations.
    """

    def __init__(self, frame, values):
        self.frame, self.values = frame, np.asarray(values)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.values, dtype=dtype)

    def __gt__(self, value):
        return FrameSeries(self.frame, self.values > value)

    def __getitem__(self, mask):
        return FrameSeries(self.frame, np.where(mask, self.values, 0))

    def sum(self):
        tables = {
            entity: self.frame.table(entity).copy() for entity in self.frame.entities
        }
        tables["spm_unit"]["test_value"] = self.values
        weighted = Frame(
            tables, US_SCHEMA, {"household": self.frame.resolve_weights("household")}
        )
        return wsum(weighted, "test_value")


@pytest.fixture
def export(tmp_path, monkeypatch):
    states = sorted(subject.SNAP_STATE_FIPS)
    # First physical household has three persons in TWO SPM units. One of those
    # units contains two people: neither person nor household counting is right.
    hh = np.array([101, 101, 101, *range(102, 152)])
    spm = np.array([201, 201, 202, *range(203, 253)])
    person = pd.DataFrame({"person_id": np.arange(len(hh)) + 1001})
    tables = {"person": person}
    for entity in US_SCHEMA.group_entities:
        ids = (
            hh
            if entity == "household"
            else spm
            if entity == "spm_unit"
            else person.person_id.to_numpy() + 2000
        )
        person[f"person_{entity}_id"] = ids
        tables[entity] = pd.DataFrame({f"{entity}_id": pd.unique(ids)})
    tables["household"]["state_fips"] = [int(state) for state in states]
    tables["household"]["county_fips"] = [int(state + "001") for state in states]
    frame = Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([3.0, *([2.0] * 50)]), WeightKind.CALIBRATED)},
    )
    path = tmp_path / "invented-final.h5"
    write_nullable_us_h5(
        frame, path, period=2024, artifact_kind="invented-SNAP-test-only"
    )
    contract = ExportContract(
        required=("state_fips", "county_fips", "household_weight"),
        forbidden=(),
        optional=(),
        formula_owned_excluded=("snap",),
        closed=True,
    )
    specs = []
    for state in states:
        for role, mode, target in (
            ("snap_households", "indicator_sum", 6 if state == states[0] else 2),
            ("snap_total", "sum", 900 if state == states[0] else 200),
        ):
            name = f"test.snap.{state}.{role}"
            specs.append(
                TargetSpec(
                    name=name,
                    entity="household",
                    measure=name,
                    value=target,
                    period=2024,
                    source="Invented test only",
                    family="usda_snap",
                    tolerance=0,
                    metadata={
                        "state_fips": state,
                        "target_role": role,
                        "materializer": "policyengine_variable",
                        "base_variable": "snap",
                        "measure_mode": mode,
                        "source_period": "2023",
                        "target_period": "2024",
                        "source_measure_id": "average_monthly_households"
                        if role == "snap_households"
                        else "total_benefits",
                        **(
                            {"fact_aggregation": "time_mean"}
                            if role == "snap_households"
                            else {}
                        ),
                    },
                )
            )
    registry = TargetRegistry(specs, country="us")
    diagnostics = tuple(
        TargetDiagnostic(
            f"{spec.name}@{spec.period}", spec.value, spec.value, spec.value, 0, True
        )
        for spec in specs
    )
    values = np.array([100.0, 200.0, *([100.0] * 50)])

    class Simulation:
        def calculate(self, variable, *, period, map_to):
            assert period == 2024 and map_to == "spm_unit"
            if variable == "spm_unit_id":
                return frame.table("spm_unit").spm_unit_id.to_numpy()
            assert variable == "snap"
            return FrameSeries(frame, values)

    simulation = Simulation()
    monkeypatch.setitem(
        sys.modules, "microdf", SimpleNamespace(MicroSeries=FrameSeries)
    )
    monkeypatch.setattr(
        subject, "default_simulate_factory", lambda candidate: lambda reform: simulation
    )
    return SimpleNamespace(
        path=path,
        frame=frame,
        contract=contract,
        registry=registry,
        diagnostics=diagnostics,
        simulation=simulation,
        values=values,
    )


def evaluate(export, **overrides):
    arguments = dict(
        expected_sha256=hashlib.sha256(export.path.read_bytes()).hexdigest(),
        contract=export.contract,
        period=2024,
        registry=export.registry,
        diagnostics=export.diagnostics,
    )
    arguments.update(overrides)
    return subject.snap_export_acceptance(export.path, **arguments)


def rewrite(export, entity, mutate):
    with pd.HDFStore(export.path, mode="a") as store:
        table = store[entity]
        mutate(table)
        put_frame_table(store, entity, table, preferred_format="fixed")


def test_final_artifact_two_units_not_one_household_or_three_persons(export):
    report = evaluate(export)
    assert all(check["status"] == "passed" for check in report["checks"].values()), {
        key: value
        for key, value in report["checks"].items()
        if value["status"] != "passed"
    }
    rows = report["checks"]["final_export_fit"]["rows"]
    assert len(rows) == 102
    assert rows[0]["final_estimate"] == 6  # two SPM units, calibrated HH weight 3
    assert rows[1]["final_estimate"] == 900
    assert rows[0]["source_period"] == "2023" and rows[0]["model_period"] == 2024
    assert "passed" not in report and "approved" not in report
    assert report["fy2022_eligible_person_comparison"]["blocking"] is False
    assert report["checks"]["artifact_identity"]["final_io_verified"] is True


def test_positive_indicator_is_applied_before_aggregation(export):
    export.values[1] = 0
    rows = evaluate(export)["checks"]["final_export_fit"]["rows"]
    assert rows[0]["final_estimate"] == 3
    assert rows[1]["final_estimate"] == 300
    assert rows[0]["status"] == "failed"


def test_stage_fit_cannot_waive_export_miss(export):
    # The calibration diagnostics all hit their targets exactly; final export does not.
    export.values[0] = 0
    report = evaluate(export)
    assert report["checks"]["registry_and_diagnostics"]["status"] == "passed"
    assert report["checks"]["final_export_fit"]["status"] == "failed"
    assert "saturation cannot waive" in report["stage_take_up"]


def test_missing_tolerance_is_pending_and_existing_fit_decision_can_fill_it(export):
    specs = tuple(replace(spec, tolerance=None) for spec in export.registry.specs)
    registry = TargetRegistry(specs, country="us")
    diagnostics = tuple(
        replace(row, within_tolerance=None) for row in export.diagnostics
    )
    assert (
        evaluate(export, registry=registry, diagnostics=diagnostics)["checks"][
            "final_export_fit"
        ]["status"]
        == "pending"
    )
    decision = TargetFitRequirement(
        "invented-approved-decision",
        "Test only",
        0.01,
        accepted_name_prefixes=("test.snap.",),
        min_matches=102,
    )
    result = evaluate(
        export, registry=registry, diagnostics=diagnostics, fit_requirements=(decision,)
    )
    assert result["checks"]["final_export_fit"]["status"] == "passed"


def test_stricter_absolute_tolerance_is_not_overridden(export):
    export.values[0] += 1
    decision = TargetFitRequirement(
        "test", "Test only", 0.5, accepted_name_prefixes=("test.snap.",)
    )
    assert (
        evaluate(export, fit_requirements=(decision,))["checks"]["final_export_fit"][
            "status"
        ]
        == "failed"
    )


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate-state",
        "wrong-period",
        "wrong-mode",
        "wrong-universe",
        "district-subset",
        "missing-source-period",
        "wrong-family",
    ],
)
def test_incomplete_or_wrong_target_surface_fails(export, change):
    specs = list(export.registry.specs)
    if change == "missing":
        specs.pop()
    else:
        spec = specs[0]
        metadata = dict(spec.metadata)
        if change == "duplicate-state":
            metadata["state_fips"] = "02"
        elif change == "wrong-mode":
            metadata["measure_mode"] = "sum"
        elif change == "wrong-universe":
            metadata["indicator_map_to"] = "person"
        elif change == "district-subset":
            metadata["congressional_district_geoid"] = "101"
        elif change == "missing-source-period":
            metadata.pop("source_period")
        specs[0] = replace(
            spec,
            metadata=metadata,
            period=2025 if change == "wrong-period" else 2024,
            family="wrong" if change == "wrong-family" else spec.family,
        )
    assert (
        evaluate(export, registry=TargetRegistry(specs, country="us"))["checks"][
            "registry_and_diagnostics"
        ]["status"]
        == "failed"
    )


@pytest.mark.parametrize(
    "change",
    ["missing", "duplicate", "target", "error", "within", "nonfinite", "summary"],
)
def test_diagnostics_cannot_replace_artifact_evidence(export, change):
    rows = list(export.diagnostics)
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(rows[0])
    elif change == "summary":
        rows[0] = {"passed": True}
    else:
        rows[0] = replace(
            rows[0],
            **{
                "target": {"target": 100},
                "error": {"relative_error": 0.9},
                "within": {"within_tolerance": False},
                "nonfinite": {"final_estimate": float("nan")},
            }[change],
        )
    assert (
        evaluate(export, diagnostics=rows)["checks"]["registry_and_diagnostics"][
            "status"
        ]
        == "failed"
    )


@pytest.mark.parametrize(
    "column,value",
    [
        ("county_fips", 2001),
        ("county_fips", 1000),
        ("county_fips", 1001.5),
        ("county_fips", None),
        ("state_fips", 72),
        ("household_weight", -1),
        ("household_weight", 0),
    ],
)
def test_geography_and_weight_failures(export, column, value):
    def mutate(table):
        table[column] = table[column].astype(object)
        table.loc[0, column] = value

    rewrite(export, "household", mutate)
    assert evaluate(export)["checks"]["artifact_evaluation"]["status"] == "failed"


@pytest.mark.parametrize(
    "change",
    [
        "required",
        "formula",
        "closed-extra",
        "empty",
        "period",
        "duplicate-id",
        "cross-household-spm",
    ],
)
def test_export_contract_and_linkage(export, change):
    contract = export.contract
    if change == "required":
        contract = replace(contract, required=(*contract.required, "missing"))
    elif change == "empty":
        contract = ExportContract.empty()
    elif change == "formula":
        rewrite(export, "spm_unit", lambda table: table.__setitem__("snap", 100))
    elif change == "closed-extra":
        rewrite(export, "household", lambda table: table.__setitem__("extra", 100))
    elif change == "period":
        with pd.HDFStore(export.path, mode="a") as store:
            store.put("_time_period", pd.Series([2025]), format="table")
    elif change == "duplicate-id":
        rewrite(export, "person", lambda table: table.__setitem__("person_id", 1))
    else:
        rewrite(
            export,
            "person",
            lambda table: table.loc.__setitem__((0, "person_household_id"), 102),
        )
    assert (
        evaluate(export, contract=contract)["checks"]["artifact_evaluation"]["status"]
        == "failed"
    )


def test_engine_missing_stays_pending(export, monkeypatch):
    def missing(*args):
        raise ModuleNotFoundError("policyengine_us")

    monkeypatch.setattr(subject, "default_simulate_factory", missing)
    report = evaluate(export)
    assert report["checks"]["engine"]["status"] == "pending"
    assert report["checks"]["final_export_fit"]["status"] == "pending"


def test_export_missing_stays_pending(export):
    digest = hashlib.sha256(export.path.read_bytes()).hexdigest()
    export.path.unlink()
    result = subject.snap_export_acceptance(
        export.path,
        expected_sha256=digest,
        contract=export.contract,
        period=2024,
        registry=export.registry,
        diagnostics=export.diagnostics,
    )
    assert result["checks"]["artifact_identity"]["status"] == "pending"


def test_matching_hash_does_not_make_a_malformed_hdf_valid(export, monkeypatch):
    export.path.write_bytes(b"This is an invented malformed HDF container.")

    def refuse_engine(*args, **kwargs):
        raise AssertionError("Malformed export must refuse before engine creation.")

    monkeypatch.setattr(subject, "default_simulate_factory", refuse_engine)
    report = evaluate(export)
    assert report["checks"]["artifact_identity"]["status"] == "passed"
    assert report["checks"]["artifact_evaluation"] == {
        "status": "failed",
        "reason": "Export is not a readable HDF5 container.",
    }
    assert report["checks"]["final_export_fit"]["status"] == "pending"


@pytest.mark.parametrize("change", ["order", "plain-series", "nonfinite", "negative"])
def test_engine_array_alignment_and_weight_retention(export, monkeypatch, change):
    original = export.simulation.calculate

    def calculate(variable, **kwargs):
        result = original(variable, **kwargs)
        if change == "order" and variable == "spm_unit_id":
            return result[::-1]
        if variable == "snap":
            if change == "plain-series":
                return pd.Series(export.values)
            if change == "nonfinite":
                export.values[0] = np.nan
            if change == "negative":
                export.values[0] = -1
        return result

    monkeypatch.setattr(export.simulation, "calculate", calculate)
    assert evaluate(export)["checks"]["artifact_evaluation"]["status"] == "failed"


@pytest.mark.parametrize("when", ["initial", "calculate", "teardown"])
def test_hash_verified_after_final_io(export, monkeypatch, when):
    digest = hashlib.sha256(export.path.read_bytes()).hexdigest()

    def change_file(*args):
        with export.path.open("ab") as handle:
            handle.write(b"changed")

    if when == "initial":
        change_file()
    elif when == "calculate":
        original = export.simulation.calculate

        def calculate(variable, **kwargs):
            value = original(variable, **kwargs)
            if variable == "snap":
                change_file()
            return value

        monkeypatch.setattr(export.simulation, "calculate", calculate)
    else:
        monkeypatch.setattr(subject, "release_engine_simulation", change_file)
    report = evaluate(export, expected_sha256=digest)
    assert report["checks"]["artifact_identity"]["status"] == "failed"
    assert report["checks"]["final_export_fit"]["status"] == "failed"


@pytest.mark.parametrize("mutate", [False, True])
def test_teardown_exception_still_verifies_final_hash(export, monkeypatch, mutate):
    def broken_cleanup(simulation):
        if mutate:
            with export.path.open("ab") as handle:
                handle.write(b"cleanup-mutated")
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(subject, "release_engine_simulation", broken_cleanup)
    report = evaluate(export)
    assert report["checks"]["engine_cleanup"]["status"] == "failed"
    assert report["checks"]["artifact_identity"]["status"] == (
        "failed" if mutate else "passed"
    )
    assert report["checks"]["final_export_fit"]["status"] == "failed"


def test_maintained_diagnostic_generator_is_accepted(export):
    from microcosm.calibrate.matrix import build_constraint_matrix
    from microcosm.calibrate.solve import _build_diagnostics
    from microcosm.calibrate.target import TargetSet

    # Pure tiny matrix compilation and diagnostics, not a calibration solve.
    specs = tuple(
        replace(spec, tolerance=None if index % 2 else 0)
        for index, spec in enumerate(export.registry.specs)
    )
    tables = {
        entity: export.frame.table(entity).copy() for entity in export.frame.entities
    }
    household = tables["household"]
    values = {}
    for spec in specs:
        state = int(spec.metadata["state_fips"])
        count = spec.metadata["target_role"] == "snap_households"
        measure = 2 if count else 300
        if state != 1:
            measure = 1 if count else 100
        values[spec.measure] = np.where(household.state_fips == state, measure, 0)
    tables["household"] = pd.concat([household, pd.DataFrame(values)], axis=1)
    frame = Frame(
        tables, US_SCHEMA, {"household": export.frame.resolve_weights("household")}
    )
    problem = build_constraint_matrix(
        frame,
        TargetSet([spec.to_target() for spec in specs]),
        weight_entity="household",
    )
    diagnostics = _build_diagnostics(
        problem, frame, problem.initial_weights.values, problem.initial_weights.values
    )
    subject._validate_diagnostics(specs, diagnostics)
    assert len(diagnostics) == 102
    assert all(row.relative_error == 0 for row in diagnostics)
    assert {row.within_tolerance for row in diagnostics} == {True, None}
