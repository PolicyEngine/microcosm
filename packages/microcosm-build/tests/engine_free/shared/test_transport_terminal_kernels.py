"""Terminal kernels: diagnostics, export prepare/readback, package receipt.

Invariant (Hypothesis property test):

- export round trip: for any table drawn from the writer's exact dtypes, the
  readback of the written export passes, and any one change to the written
  file (a cell in any column, a column's dtype, a dropped table, the period)
  fails it with the line naming that change.

Example tests:

- diagnostics validate against the ``microcosm-diagnostics`` schema-8 model,
  carry every target's hierarchy and its compiled entity and measure, and
  refuse a population whose weights are not the solution's, a surface the
  problem was not compiled from, a result from another solve, and a problem
  on another entity;
- the export readback refuses a corrupted file: an unreadable file fails the
  GATE node (charter F4), leaves the package ``unreached`` and records no host
  path; a readable file that differs fails its outcome, and the package
  refuses it;
- export preparation refuses when any gate report does not permit it and when
  no gate report is wired, and it refuses dtypes the writer cannot
  round-trip;
- export preparation and the package read one battery's reports: the same
  manifest document, resource hash and posture, through the last phase that
  can block;
- the package never authorizes a release, binds every input artifact, and
  refuses a gate report that does not permit the artifact.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames

from microcosm.build.transport.artifact_types import (
    DIAGNOSTICS_TYPE,
    EXPORT_DESCRIPTOR_TYPE,
    EXPORT_READBACK_TYPE,
    GATE_REPORT_TYPE,
    PACKAGE_RECEIPT_TYPE,
    TARGET_SURFACE_TYPE,
)
from microcosm.build.transport.gate_kernels import GateBatteryKernel
from microcosm.build.transport.terminal_kernels import (
    DIAGNOSTICS_CALIBRATION,
    EXPORT_PREPARE,
    TRANSPORT_PACKAGE,
    describe_export,
    materialize_export,
    validate_export,
)
from microcosm.calibrate.artifacts import PROBLEM_TYPE, RESULT_TYPE, SOLUTION_TYPE
from microcosm.diagnostics import (
    CalibrationDiagnosticsV8,
    parse_calibration_diagnostics,
    write_calibration_diagnostics,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights
from microcosm.frame.adapters.axiom import AxiomEntityTableDataset
from microcosm.graph import (
    ArtifactOutput,
    ArtifactValue,
    KernelContext,
    Node,
    NodeRejectedError,
    NumericScope,
)
from microcosm.graph.canonical import canonical_json
from test_support.microcosm_build.transport_graph import (
    CALIBRATION_TARGETS,
    COUNTRY,
    DEFAULT_CONFIG,
    ENTITIES,
    PERIOD,
    TOY_BINDINGS,
    canonical_text,
    run_through,
    run_toy,
    toy_frame,
    toy_gates,
    with_config,
    write_toy_sources,
)

DIAGNOSTICS = "toy.calibration.diagnostics"
_KEY = "b" * 64


def _value(payload: bytes, kind) -> ArtifactValue:
    return ArtifactValue(
        payload=payload, type=kind, key=_KEY, producer_key=_KEY, numerics=NumericScope()
    )


@pytest.fixture(scope="module")
def toy_run(tmp_path_factory):
    """One full toy build: every terminal node runs on real upstream bytes."""

    root = tmp_path_factory.mktemp("toy")
    sources = write_toy_sources(root / "src")
    return run_toy(root, sources)


def test_full_build_passes_its_gates_readback_and_package(toy_run) -> None:
    nodes = toy_run.manifest.nodes
    assert nodes["toy.gates.terminal"].receipt["outcome"] == "pass"
    assert nodes["toy.export.readback"].receipt["outcome"] == "pass"
    assert toy_run.manifest.known_failures == ()

    receipt = json.loads(toy_run.artifact("toy.package", "receipt"))
    assert receipt["release_authorized"] is False
    assert receipt["readback"]["passed"] is True
    assert receipt["gates"] == {
        "gate_report": {
            "outcome": "pass",
            "phase": "terminal",
            "gates_sha256": DEFAULT_CONFIG.gates_sha256,
            "release_candidate": False,
            "synthetic_smoke": False,
        }
    }
    package = nodes["toy.package"]
    inputs = {"comparison", "diagnostics", "export_readback", "gate_report", "surface"}
    assert set(receipt["artifacts"]) == inputs
    for alias, entry in receipt["artifacts"].items():
        assert len(entry["sha256"]) == 64, alias
        assert entry["producer_key"] in {r.key for r in nodes.values()}, alias
    assert package.opaque_artifacts.keys() == {"receipt"}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_validate_against_schema_8(toy_run, tmp_path) -> None:
    payload = toy_run.artifact(DIAGNOSTICS, "diagnostics")
    model = parse_calibration_diagnostics(json.loads(payload))

    assert isinstance(model, CalibrationDiagnosticsV8)
    assert model.schema_version == DIAGNOSTICS_TYPE.schema_version == 8
    assert model.target_registry.country == COUNTRY
    assert model.n_records == 6
    assert [row.target_name for row in model.targets] == [
        name for name, *_ in CALIBRATION_TARGETS
    ]
    # Rows describe each target as compiled, not as a contribution vector.
    assert [
        (row.entity, row.measure.name, None if row.filter is None else row.filter.name)
        for row in model.targets
    ] == [
        (entity, measure, filter_)
        for _name, entity, measure, filter_, _x in CALIBRATION_TARGETS
    ]
    assert {row.hierarchy.category.id for row in model.targets} == {"toy.population"}
    assert set(model.build) >= {
        "problem_sha256",
        "solution_sha256",
        "result_sha256",
        "surface_sha256",
    }
    # The toy calibrates straight from its CREATE design weights, so the
    # solver's ratio (to its starting weights) is the executor's (to design).
    assert model.realized_max_weight_ratio == pytest.approx(
        toy_run.manifest.nodes["toy.calibrate"].receipt["realized_max_weight_ratio"]
    )
    outcome = write_calibration_diagnostics(model, tmp_path / "diagnostics.json")
    assert outcome.status == "available"
    assert (
        canonical_json(model.model_dump(mode="python", exclude_defaults=True))
        == payload
    )


def _diagnostics_artifacts(run, **overrides) -> dict[str, ArtifactValue]:
    artifacts = {
        alias: _value(run.artifact(producer, name), kind)
        for alias, producer, name, kind in (
            ("problem", "toy.targets.problem", "problem", PROBLEM_TYPE),
            ("solution", "toy.calibrate", "solution", SOLUTION_TYPE),
            ("result", "toy.calibrate", "result", RESULT_TYPE),
            ("surface", "toy.targets.compile", "surface", TARGET_SURFACE_TYPE),
        )
    }
    artifacts.update(overrides)
    return artifacts


def _diagnostics_context(artifacts, *, weights=None, entity="household"):
    frame = toy_frame() if weights is None else toy_frame(weights=weights)
    node = Node(
        "diagnostics",
        "diagnostics.calibration@1",
        population="terminal",
        params={"weight_entity": entity},
        artifact_outputs=(ArtifactOutput("diagnostics", DIAGNOSTICS_TYPE),),
    )
    return KernelContext(
        node=node,
        tables={"household": frame.table("household")},
        weights={"household": frame.weights_for("household")},
        strata=frame.strata,
        params=node.params,
        rng=np.random.default_rng(0),
        artifacts=artifacts,
    )


def test_diagnostics_refuse_mismatched_evidence(toy_run, tmp_path) -> None:
    from microcosm.calibrate.artifacts import decode_solution

    solved = decode_solution(toy_run.artifact("toy.calibrate", "solution")).weights
    good = _diagnostics_context(_diagnostics_artifacts(toy_run), weights=solved)
    assert DIAGNOSTICS_CALIBRATION.run(good).artifacts["diagnostics"] == (
        toy_run.artifact(DIAGNOSTICS, "diagnostics")
    )

    # The population before calibration does not carry the solution.
    with pytest.raises(ValueError, match="not this solution's"):
        DIAGNOSTICS_CALIBRATION.run(
            _diagnostics_context(_diagnostics_artifacts(toy_run))
        )

    # A surface the problem was not compiled from.
    holdout = run_through(
        tmp_path / "holdout",
        write_toy_sources(tmp_path / "holdout_src"),
        with_config(DEFAULT_CONFIG, calibration_targets=CALIBRATION_TARGETS[:2]),
        "toy.targets.compile",
    )
    foreign = holdout[1].load_bytes(
        holdout[0].nodes["toy.targets.compile"].opaque_artifacts["surface"]
    )
    with pytest.raises(ValueError, match="different surface"):
        DIAGNOSTICS_CALIBRATION.run(
            _diagnostics_context(
                _diagnostics_artifacts(
                    toy_run, surface=_value(foreign, TARGET_SURFACE_TYPE)
                ),
                weights=solved,
            )
        )

    # The same registry under other provenance is still another surface.
    from microcosm.build.transport.target_kernels import (
        decode_target_surface,
        encode_target_surface,
        parse_reference_document,
    )
    from test_support.microcosm_build.transport_graph import references_for

    surface = decode_target_surface(toy_run.artifact("toy.targets.compile", "surface"))
    relabelled = encode_target_surface(
        surface.registry,
        parse_reference_document(references_for(CALIBRATION_TARGETS), country=COUNTRY),
        references_sha256="3" * 64,
        facts=surface.facts,
    )
    assert decode_target_surface(relabelled).registry.version == (
        surface.registry.version
    )
    with pytest.raises(ValueError, match="different surface"):
        DIAGNOSTICS_CALIBRATION.run(
            _diagnostics_context(
                _diagnostics_artifacts(
                    toy_run, surface=_value(relabelled, TARGET_SURFACE_TYPE)
                ),
                weights=solved,
            )
        )

    # A result from another solve of the same problem.
    other_manifest, other_store = run_through(
        tmp_path / "other",
        write_toy_sources(tmp_path / "other_src"),
        with_config(DEFAULT_CONFIG, epochs=7),
        "toy.calibrate",
    )
    other_result = other_store.load_bytes(
        other_manifest.nodes["toy.calibrate"].opaque_artifacts["result"]
    )
    with pytest.raises(ValueError, match="result and solution weights differ"):
        DIAGNOSTICS_CALIBRATION.run(
            _diagnostics_context(
                _diagnostics_artifacts(
                    toy_run, result=_value(other_result, RESULT_TYPE)
                ),
                weights=solved,
            )
        )

    # A problem on another entity.
    with pytest.raises(ValueError, match="calibrates 'household'"):
        DIAGNOSTICS_CALIBRATION.run(
            _diagnostics_context(
                _diagnostics_artifacts(toy_run), weights=solved, entity="person"
            )
        )


# ---------------------------------------------------------------------------
# Export readback refuses a corrupted H5
# ---------------------------------------------------------------------------


def _truncate(path) -> None:
    path.write_bytes(path.read_bytes()[: len(path.read_bytes()) // 2])


def _not_hdf5(path) -> None:
    path.write_bytes(b"this is not an HDF5 file\n")


def _rewrite(path, change) -> None:
    dataset = AxiomEntityTableDataset(file_path=path)
    tables = {name: table.copy() for name, table in dataset.tables.items()}
    period = change(tables, dataset.time_period)
    AxiomEntityTableDataset(
        tables=tables,
        time_period=dataset.time_period if period is None else period,
    ).save(path)


def _alter_one_cell(path) -> None:
    def change(tables, _period):
        tables["household"].loc[tables["household"].index[0], "rent"] += 1.0

    _rewrite(path, change)


def _add_table(path) -> None:
    def change(tables, _period):
        tables["benefit_unit"] = pd.DataFrame({"benefit_unit_id": np.arange(2)})

    _rewrite(path, change)


@pytest.mark.parametrize("corrupt", [_truncate, _not_hdf5])
def test_readback_fails_an_unreadable_file_and_unreaches_the_package(
    tmp_path, corrupt
) -> None:
    sources = write_toy_sources(tmp_path / "src")
    run = run_toy(tmp_path, sources, corrupt=corrupt)
    readback = run.manifest.nodes["toy.export.readback"].receipt

    assert readback["outcome"] == "fail"
    assert readback["execution"]["state"] == "gate_exception"
    assert run.manifest.nodes["toy.package"].receipt["outcome"] == "unreached"
    assert "toy.export.readback" in run.manifest.known_failures
    # The cached receipt binds no host path.
    message = readback["evidence"]["message"]
    assert "cannot read declared source 'exported_dataset'" in message
    assert str(tmp_path) not in json.dumps(dict(readback["evidence"]))


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        (_alter_one_cell, "'household' table differs"),
        (_add_table, "undescribed table"),
    ],
)
def test_readback_fails_a_different_file_and_the_package_refuses_it(
    tmp_path, corrupt, message
) -> None:
    sources = write_toy_sources(tmp_path / "src")
    run = run_toy(tmp_path, sources, corrupt=corrupt, endpoint="toy.export.readback")
    readback = run.manifest.nodes["toy.export.readback"].receipt

    assert readback["outcome"] == "fail"
    assert any(message in line for line in readback["evidence"]["failures"])
    with pytest.raises(NodeRejectedError, match="failed readback"):
        run_toy(tmp_path, sources, store=run.store, exported=run.exported)


def _written(tmp_path):
    frame = toy_frame()
    descriptor = describe_export(
        frame,
        entities=ENTITIES,
        weight_entity="household",
        time_period=PERIOD,
        bindings={},
    )
    path = tmp_path / "written.h5"
    materialize_export(frame, descriptor, path)
    assert validate_export(path, descriptor)["passed"] is True
    return path, descriptor


def _period_plus_one(tables, period):
    return period + 1


def _drop_family(tables, _period):
    del tables["family"]


def _age_as_int32(tables, _period):
    tables["person"]["age"] = tables["person"]["age"].astype("int32")


@pytest.mark.parametrize(
    ("change", "failure"),
    [
        (_period_plus_one, "Exported time period differs from its descriptor."),
        (_drop_family, "Exported file lacks described table(s) ['family']."),
        (_age_as_int32, "Exported 'person' table differs from its descriptor."),
    ],
)
def test_readback_names_a_changed_period_table_or_dtype(
    tmp_path, change, failure
) -> None:
    path, descriptor = _written(tmp_path)
    _rewrite(path, change)
    report = validate_export(path, descriptor)

    assert report["passed"] is False
    assert failure in report["failures"]


# ---------------------------------------------------------------------------
# Export preparation and the package
# ---------------------------------------------------------------------------


def test_describe_refuses_dtypes_the_writer_cannot_round_trip() -> None:
    frame = toy_frame()
    for dtype, values in (
        ("Int64", pd.array([1, None, 3, 4, 5, 6], dtype="Int64")),
        ("boolean", pd.array([True, None, False, True, True, False], dtype="boolean")),
    ):
        household = frame.table("household").assign(extra=values)
        altered = Frame(
            {
                "person": frame.table("person"),
                "household": household,
                "family": frame.table("family"),
            },
            frame.schema,
            {"household": frame.weights_for("household")},
            frame.strata,
        )
        with pytest.raises(ValueError, match=dtype):
            describe_export(
                altered,
                entities=ENTITIES,
                weight_entity="household",
                time_period=PERIOD,
                bindings={},
            )


def _report(
    *,
    release_candidate: bool,
    gates=None,
    gates_sha256: str = "2" * 64,
    phase: str = "terminal",
) -> bytes:
    node = Node(
        "gate",
        "gates.battery@1",
        population="terminal",
        params={
            "country": COUNTRY,
            "gates": canonical_text(toy_gates() if gates is None else gates),
            "gates_sha256": gates_sha256,
            "phase": phase,
            "release_candidate": release_candidate,
        },
        artifact_outputs=(ArtifactOutput("gate_report", GATE_REPORT_TYPE),),
    )
    context = KernelContext(
        node=node,
        tables={},
        weights={},
        strata=pd.Series([], dtype=object, name="stratum"),
        params=node.params,
        rng=np.random.default_rng(0),
    )
    return GateBatteryKernel(TOY_BINDINGS).run(context).artifacts["gate_report"]


def _prepare_context(artifacts) -> KernelContext:
    frame = toy_frame()
    node = Node(
        "prepare",
        "export.prepare@1",
        population="terminal",
        params={
            "entities": ENTITIES,
            "weight_entity": "household",
            "time_period": PERIOD,
        },
        artifact_outputs=(ArtifactOutput("export_descriptor", EXPORT_DESCRIPTOR_TYPE),),
    )
    return KernelContext(
        node=node,
        tables={entity: frame.table(entity) for entity in ENTITIES},
        weights={"household": frame.weights_for("household")},
        strata=frame.strata,
        params=node.params,
        rng=np.random.default_rng(0),
        artifacts=artifacts,
    )


def test_prepare_refuses_blocking_gates_and_an_ungated_export() -> None:
    # Frameless, the toy phase lacks evidence: a release candidate blocks.
    blocking = _value(_report(release_candidate=True), GATE_REPORT_TYPE)
    with pytest.raises(ValueError, match="refused by blocking gate report"):
        EXPORT_PREPARE.run(_prepare_context({"gate_report": blocking}))
    with pytest.raises(ValueError, match="no gate has evaluated"):
        EXPORT_PREPARE.run(_prepare_context({}))
    permitted = _value(_report(release_candidate=False), GATE_REPORT_TYPE)
    result = EXPORT_PREPARE.run(_prepare_context({"gate_report": permitted}))
    descriptor = json.loads(result.artifacts["export_descriptor"])
    assert descriptor["entities"] == list(ENTITIES)
    assert descriptor["tables"]["household"]["columns"][-1] == "household_weight"
    assert descriptor["bindings"]["gates"]["gate_report"]["outcome"] == (
        "evidence_absent"
    )


def test_prepare_reads_one_batterys_reports_through_its_last_blocking_phase() -> None:
    terminal = _value(_report(release_candidate=False), GATE_REPORT_TYPE)
    preflight = _value(
        _report(release_candidate=False, phase="preflight"), GATE_REPORT_TYPE
    )
    # The toy's preflight phase cannot block, so the terminal report suffices,
    # and the preflight report alone does not.
    EXPORT_PREPARE.run(_prepare_context({"terminal": terminal, "preflight": preflight}))
    with pytest.raises(ValueError, match="last phase that can block, 'terminal'"):
        EXPORT_PREPARE.run(_prepare_context({"preflight": preflight}))
    other_manifest = _value(
        _report(release_candidate=False, gates=toy_gates(max_ratio=9.0)),
        GATE_REPORT_TYPE,
    )
    other_resource = _value(
        _report(release_candidate=False, gates_sha256="5" * 64), GATE_REPORT_TYPE
    )
    for other in (other_manifest, other_resource):
        with pytest.raises(ValueError, match="different manifests or postures"):
            EXPORT_PREPARE.run(_prepare_context({"terminal": terminal, "other": other}))


def _package_context(artifacts) -> KernelContext:
    node = Node(
        "package",
        "transport.package@1",
        population="terminal",
        artifact_outputs=(ArtifactOutput("receipt", PACKAGE_RECEIPT_TYPE),),
    )
    return KernelContext(
        node=node,
        tables={},
        weights={},
        strata=pd.Series([], dtype=object, name="stratum"),
        params=node.params,
        rng=np.random.default_rng(0),
        artifacts=artifacts,
    )


def test_package_reads_the_gate_reports_the_export_was_prepared_under(
    toy_run,
) -> None:
    readback_payload = toy_run.artifact("toy.export.readback", "export_readback")
    readback = _value(readback_payload, EXPORT_READBACK_TYPE)
    (prepared_key,) = {
        entry["artifact_key"]
        for entry in json.loads(readback_payload)["bindings"]["gates"].values()
    }
    gate_payload = toy_run.artifact("toy.gates.terminal", "gate_report")

    def report(payload: bytes, key: str) -> ArtifactValue:
        return ArtifactValue(
            payload=payload,
            type=GATE_REPORT_TYPE,
            key=key,
            producer_key=_KEY,
            numerics=NumericScope(),
        )

    receipt = json.loads(
        TRANSPORT_PACKAGE.run(
            _package_context(
                {
                    "export_readback": readback,
                    "gates": report(gate_payload, prepared_key),
                }
            )
        ).artifacts["receipt"]
    )
    assert receipt["release_authorized"] is False
    assert receipt["gates"] == {
        "gates": {
            "outcome": "pass",
            "phase": "terminal",
            "gates_sha256": DEFAULT_CONFIG.gates_sha256,
            "release_candidate": False,
            "synthetic_smoke": False,
        }
    }
    # Another report, even one that permits, is not the one export used.
    other = report(_report(release_candidate=False), "d" * 64)
    with pytest.raises(ValueError, match="not the ones the export"):
        TRANSPORT_PACKAGE.run(
            _package_context({"export_readback": readback, "gates": other})
        )
    blocking = report(_report(release_candidate=True), prepared_key)
    with pytest.raises(ValueError, match="does not permit the artifact"):
        TRANSPORT_PACKAGE.run(
            _package_context({"export_readback": readback, "gates": blocking})
        )
    with pytest.raises(ValueError, match="export_readback"):
        TRANSPORT_PACKAGE.run(
            _package_context({"gates": report(gate_payload, prepared_key)})
        )
    with pytest.raises(ValueError, match="no gate has evaluated"):
        TRANSPORT_PACKAGE.run(_package_context({"export_readback": readback}))


def test_prepare_refuses_caller_supplied_gate_bindings() -> None:
    permitted = _value(_report(release_candidate=False), GATE_REPORT_TYPE)
    context = _prepare_context({"gate_report": permitted})
    node = Node(
        "prepare",
        "export.prepare@1",
        population="terminal",
        params={**context.params, "bindings": canonical_text({"gates": {}})},
        artifact_outputs=context.node.artifact_outputs,
    )
    with pytest.raises(ValueError, match="must not carry a 'gates' key"):
        EXPORT_PREPARE.run(
            KernelContext(
                node=node,
                tables=context.tables,
                weights=context.weights,
                strata=context.strata,
                params=node.params,
                rng=context.rng,
                artifacts=context.artifacts,
            )
        )


def test_materialize_refuses_a_population_that_differs(tmp_path) -> None:
    frame = toy_frame()
    descriptor = describe_export(
        frame,
        entities=ENTITIES,
        weight_entity="household",
        time_period=PERIOD,
        bindings={},
    )
    heavier = toy_frame(weights=[w * 2 for w in frame.weights_for("household").values])
    with pytest.raises(ValueError, match="differs from its graph descriptor"):
        materialize_export(heavier, descriptor, tmp_path / "x.h5")
    tampered = {**descriptor, "time_period": PERIOD + 1}
    with pytest.raises(ValueError, match="content identity"):
        materialize_export(frame, tampered, tmp_path / "y.h5")
    with pytest.raises(ValueError, match=r"\.h5"):
        materialize_export(frame, descriptor, tmp_path / "z.parquet")


# ---------------------------------------------------------------------------
# Export round trip (property)
# ---------------------------------------------------------------------------

_finite = st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, width=64)
_finite32 = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, width=32)

#: (column, the cast that changes only its dtype)
_CASTS = {
    "income": "float32",
    "small": "float64",
    "count": "int32",
    "count32": "int64",
}


def _mutate(tables, period, mutation, row, column_name):
    household = tables["household"]
    if mutation == "period":
        return period + 1
    if mutation == "drop":
        del tables["household"]
        return None
    if mutation == "cast":
        household[column_name] = household[column_name].astype(_CASTS[column_name])
        return None
    position = household.index[row]
    value = household.loc[position, column_name]
    if column_name == "flag":
        household.loc[position, column_name] = not bool(value)
    elif column_name == "label":
        household.loc[position, column_name] = str(value) + "!"
    else:
        household.loc[position, column_name] = value + 1
    return None


_EXPECTED = {
    "period": "Exported time period differs from its descriptor.",
    "drop": "Exported file lacks described table(s) ['household'].",
    "cast": "Exported 'household' table differs from its descriptor.",
    "cell": "Exported 'household' table differs from its descriptor.",
}


@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=(HealthCheck.function_scoped_fixture, HealthCheck.too_slow),
)
@given(
    data_frames(
        [
            column("income", elements=_finite, dtype=np.float64),
            column("small", elements=_finite32, dtype=np.float32),
            column("count", elements=st.integers(-1000, 1000), dtype=np.int64),
            column("count32", elements=st.integers(-1000, 1000), dtype=np.int32),
            column("flag", elements=st.booleans(), dtype=bool),
            column(
                "label",
                elements=st.text(alphabet="abcxyz", min_size=1, max_size=5),
                dtype=object,
            ),
        ],
        index=st.just(pd.RangeIndex(4)),
    ),
    st.sampled_from(("cell", "cast", "drop", "period")),
    st.integers(0, 3),
    st.sampled_from(("income", "small", "count", "count32", "flag", "label")),
)
def test_property_readback_passes_iff_the_written_content_matches(
    tmp_path, table, mutation, row, column_name
) -> None:
    if mutation == "cast" and column_name not in _CASTS:
        column_name = "count"
    n = len(table)
    household = table.assign(
        household_id=np.arange(n, dtype=np.int64),
        label=table["label"].astype("string"),
    )[["household_id", "income", "small", "count", "count32", "flag", "label"]]
    person = pd.DataFrame(
        {
            "person_id": np.arange(n, dtype=np.int64),
            "person_household_id": np.arange(n, dtype=np.int64),
        }
    )
    frame = Frame(
        {"person": person, "household": household},
        EntitySchema(group_entities=("household",)),
        {"household": Weights(np.linspace(1.0, 2.0, n), WeightKind.CALIBRATED)},
    )
    descriptor = describe_export(
        frame,
        entities=("person", "household"),
        weight_entity="household",
        time_period=PERIOD,
        bindings={},
    )
    path = tmp_path / "round_trip.h5"
    materialize_export(frame, descriptor, path)
    assert validate_export(path, descriptor)["passed"] is True

    _rewrite(
        path, lambda tables, period: _mutate(tables, period, mutation, row, column_name)
    )
    report = validate_export(path, descriptor)
    assert report["passed"] is False
    assert report["failures"] == [_EXPECTED[mutation]]
