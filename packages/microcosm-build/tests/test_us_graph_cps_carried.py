"""CPS-carried graph uses the actual source facts, projection and ID merge."""

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import cps_carried
from microcosm.build.us_runtime import multispine_pool as pool
from microcosm.build.us_runtime.graph_context import (
    US_FRAME_CONTEXT_TYPE,
    encode_us_frame_context,
)
from microcosm.build.us_runtime.graph_cps_carried import (
    _CPS_IMPLEMENTATION_MODULES,
    CPS_CARRIED_DEPENDENCIES,
    USCpsCarriedKernel,
)
from microcosm.build.us_runtime.graph_sources import frame_column_declarations
from microcosm.frame import Frame
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ContentStore,
    Graph,
    KernelRegistry,
    Node,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.errors import NodeRejectedError


def _hours_fixtures():
    spec = importlib.util.spec_from_file_location(
        "_cps_hours_fixtures",
        Path(__file__).with_name("test_us_graph_early_preparation.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hours = _hours_fixtures()


def _frame():
    frame = hours._frame()
    cps = frame.person.PERIDNUM.notna()
    frame.person["SPM_CHILDCAREXPNS"] = np.nan
    frame.person.loc[cps, "SPM_CHILDCAREXPNS"] = [400.0, 250.0, 0.0, 75.0]
    for offset, column in enumerate(
        (
            "NOW_MRK",
            "NOW_NONM",
            "NOW_MCAID",
            "NOW_GRP",
            "NOW_CHAMPVA",
            "NOW_MIL",
            "NOW_VACARE",
            "NOW_OTHMT",
            "NOW_IHSFLG",
        )
    ):
        frame.person[column] = np.nan
        frame.person.loc[cps, column] = np.roll([1.0, 2.0, 2.0, 1.0], offset)
    frame.person.loc[cps, "OI_VAL"] = [120.0, 230.0, 340.0, 0.0]
    frame.person["OI_OFF"] = pd.to_numeric(frame.person.OI_OFF).astype("Int64")
    frame.person.loc[cps, "OI_OFF"] = [20.0, 12.0, 7.0, 0.0]
    return frame


def _direct(frame):
    return pool._run_source_operator_chain(
        frame,
        phase="pre_clone",
        operator_names=("derive_us_cps_carried_inputs",),
        operators={
            "derive_us_cps_carried_inputs": cps_carried.derive_us_cps_carried_inputs
        },
    )


def _setup(tmp_path, frame):
    from microcosm.build.us_runtime.graph_cps_carried import (
        register_us_cps_carried_kernels,
        us_cps_carried_nodes,
    )

    store = ContentStore(tmp_path / "store")
    source = ContentStore(tmp_path / "inputs").put_frame("a" * 64, frame)
    columns = frame_column_declarations(frame)
    nodes = us_cps_carried_nodes(columns, base="source", context_producer="source")
    graph = Graph(
        country="us",
        sources=(SourceRef("population", "frame-store"),),
        nodes=(
            Node(
                "source",
                hours._Create.ref,
                sources=("population",),
                structural=StructuralDelta.CREATE,
                outputs=columns,
                params={"frame_context": encode_us_frame_context(frame).decode()},
                artifact_outputs=(
                    ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),
                ),
            ),
            *nodes,
            Node(
                "after",
                hours._Consumer.ref,
                population=nodes[0].id,
                artifact_inputs=(
                    ArtifactInput(
                        "frame_context",
                        nodes[-1].id,
                        "frame_context",
                        US_FRAME_CONTEXT_TYPE,
                    ),
                ),
            ),
        ),
    )
    kernels = KernelRegistry()
    kernels.register(hours._Create())
    kernels.register(hours._Consumer())
    register_us_cps_carried_kernels(kernels)
    return compile_graph(graph), {"population": source}, store, kernels


def _assert_equal(actual, expected):
    for entity in expected.entities:
        pd.testing.assert_frame_equal(actual.table(entity), expected.table(entity))
    pd.testing.assert_series_equal(actual.strata, expected.strata)
    for entity in expected.weighted_entities:
        assert actual.weights_for(entity).kind == expected.weights_for(entity).kind
        assert (
            actual.weights_for(entity).values.tobytes()
            == expected.weights_for(entity).values.tobytes()
        )


def test_actual_graph_cold_warm_required_match_direct_values_receipt_and_context(
    tmp_path,
):
    frame = _frame()
    expected = _direct(frame)
    compiled, sources, store, kernels = _setup(tmp_path, frame)
    for resume in ("forbid", "auto", "require"):
        manifest = run_graph(
            compiled, sources=sources, store=store, kernels=kernels, resume=resume
        )
        _assert_equal(
            manifest.population("early_preparation.cps_carried.boundary"),
            expected.frame,
        )
        node = manifest.node("early_preparation.cps_carried")
        assert canonical_json(node.receipt["source_preparation"]) == canonical_json(
            expected.receipt
        )
        assert store.load_bytes(
            node.opaque_artifacts["frame_context"]
        ) == encode_us_frame_context(expected.frame)
        assert manifest.node("after").receipt["received_context"]
        if resume != "forbid":
            assert all(node.hit for node in manifest.nodes.values())
    # These raw amounts are carried nominally, without price alignment.
    cps = frame.person.PERIDNUM.notna()
    assert expected.frame.person.loc[cps, "employment_income_before_lsr"].tolist() == [
        40000.0,
        0.0,
        50000.0,
        0.0,
    ]
    units = expected.frame.table("spm_unit").set_index("spm_unit_id")
    ids = frame.person.loc[cps, "person_spm_unit_id"].drop_duplicates()
    assert units.spm_unit_pre_subsidy_childcare_expenses.reindex(ids).tolist() == [
        400.0,
        75.0,
    ]


def test_pure_table_helper_matches_direct_and_aligns_shuffled_spm_ids():
    frame = hours.fixtures._real_pre_clone_source_frame()
    frame.person["SPM_CHILDCAREXPNS"] = [400.0, 250.0, 0.0, 75.0]
    expected = cps_carried.derive_us_cps_carried_inputs(frame)
    source_person = frame.person.copy()
    spm = frame.table("spm_unit").iloc[::-1].copy()
    actual = cps_carried.derive_us_cps_carried_tables(source_person, spm)
    pd.testing.assert_frame_equal(actual.person, expected.person)
    pd.testing.assert_frame_equal(
        actual.spm_unit, expected.table("spm_unit").iloc[::-1]
    )
    pd.testing.assert_frame_equal(source_person, frame.person)
    assert actual.spm_unit.spm_unit_pre_subsidy_childcare_expenses.tolist() == [
        75.0,
        400.0,
    ]


@pytest.mark.parametrize(
    "surface", ("all_null", "partial", "incumbent", "shuffled_person", "native_schema")
)
def test_presence_native_cells_and_membership_order_match_direct(tmp_path, surface):
    frame = _frame()
    cps = frame.person.PERIDNUM.notna()
    if surface == "shuffled_person":
        tables = {entity: frame.table(entity) for entity in frame.entities}
        tables["person"] = frame.person.iloc[[3, 0, 5, 2, 1, 4]].copy()
        frame = Frame(
            tables,
            frame.schema,
            {"household": frame.weights_for("household")},
            frame.strata.reindex(tables["person"].index),
            metadata=frame.metadata,
            mass_log=frame.mass_log,
        )
    else:
        if surface == "native_schema":
            # Dtypes verified from preserved real f001 geography H5 metadata:
            # ACS already supplies this nullable boolean alongside age/money.
            frame.person["is_female"] = pd.Series(
                [pd.NA] * 6, index=frame.person.index, dtype="boolean"
            )
            frame.person.loc[~cps, "is_female"] = [True, False]
        frame.person["has_esi"] = pd.Series(
            [pd.NA] * 6, index=frame.person.index, dtype="boolean"
        )
        frame.person.loc[~cps, "has_esi"] = [True, False]
        frame.person["employment_income_before_lsr"] = 123.0
        frame.person.loc[cps, "employment_income_before_lsr"] = np.nan
        spm = frame.table("spm_unit")
        spm["receives_tanf"] = pd.Series([pd.NA] * len(spm), dtype="boolean")
        if surface != "all_null":
            frame.person.loc[frame.person.index[0], "has_esi"] = True
            frame.person.loc[frame.person.index[0], "employment_income_before_lsr"] = (
                456.0
            )
            spm.loc[0, "receives_tanf"] = False
        if surface == "incumbent":
            frame.person["age"] = frame.person.age.astype("Int64")
            frame.person.loc[cps, "age"] = [77, 78, 79, 80]
    expected = _direct(frame)
    compiled, sources, store, kernels = _setup(tmp_path, frame)
    manifest = run_graph(compiled, sources=sources, store=store, kernels=kernels)
    _assert_equal(
        manifest.population("early_preparation.cps_carried.boundary"), expected.frame
    )
    assert canonical_json(
        manifest.node("early_preparation.cps_carried").receipt["source_preparation"]
    ) == canonical_json(expected.receipt)


@pytest.mark.parametrize(
    "failure", ("missing_paw_type", "missing_other_income_pair", "formula_owned")
)
def test_original_source_refusals_remain_visible(tmp_path, failure):
    frame = _frame()
    if failure == "missing_paw_type":
        frame.person.drop(columns="PAW_TYP", inplace=True)
    elif failure == "missing_other_income_pair":
        frame.person.drop(columns="OI_OFF", inplace=True)
    else:
        frame.person["employment_income"] = np.nan
    with pytest.raises(ValueError) as direct:
        _direct(frame)
    compiled, sources, store, kernels = _setup(tmp_path, frame)
    with pytest.raises(NodeRejectedError) as graph:
        run_graph(compiled, sources=sources, store=store, kernels=kernels)
    assert str(direct.value) in str(graph.value)


def test_omitted_present_raw_or_incumbent_spm_inputs_refuse(tmp_path):
    frame = _frame()
    frame.table("spm_unit")["receives_tanf"] = pd.Series(
        [False] * frame.n("spm_unit"), dtype="boolean"
    )
    compiled, sources, store, kernels = _setup(tmp_path, frame)
    derive = compiled.graph.nodes[2]
    for entity, column in [("person", "NOW_MCAID"), ("spm_unit", "receives_tanf")]:
        slices = tuple(
            replace(
                value, columns=tuple(name for name in value.columns if name != column)
            )
            for value in derive.inputs
            if value.entity != entity or len(value.columns) > 1
        )
        changed = replace(derive, inputs=slices)
        altered = compile_graph(
            replace(
                compiled.graph,
                nodes=(*compiled.graph.nodes[:2], changed, *compiled.graph.nodes[3:]),
            )
        )
        if entity == "person":
            with pytest.raises(NodeRejectedError, match="omits available"):
                run_graph(altered, sources=sources, store=store, kernels=kernels)
        else:
            # Owned rewrites themselves declare the incumbent read, even
            # without an additional SPM Slice. It is never an omitted cell.
            result = run_graph(altered, sources=sources, store=store, kernels=kernels)
            _assert_equal(
                result.population("early_preparation.cps_carried.boundary"),
                _direct(frame).frame,
            )


def test_legacy_baseline_fingerprints(tmp_path):
    output = _direct(_frame())
    store = ContentStore(tmp_path)
    store.put_frame("b" * 64, output.frame)
    actual = {
        "payload": hashlib.sha256(
            canonical_json(store.metadata("b" * 64)["payloads"])
        ).hexdigest(),
        "receipt": hashlib.sha256(canonical_json(output.receipt)).hexdigest(),
        "context": hashlib.sha256(encode_us_frame_context(output.frame)).hexdigest(),
    }
    # Measured at unmodified 04f65483 before the table-helper extraction.
    assert actual == {
        "context": "f616242d050809c87f7f330ed68264d784ee750decca53458cbeb226689e40fc",
        "payload": "2f1149c319176883a122e0ddf8d6cae1b7de7892586a52661c771f0c1ff85480",
        "receipt": "8dd3bcedbc4f2abc2ae54de41f50dd306451fa0c05e826218b0a5d62bc374387",
    }


def test_first_spm_outputs_receive_only_actual_structural_ids(tmp_path, monkeypatch):
    original = USCpsCarriedKernel.run
    observed = []

    def observe(self, context):
        observed.append(tuple(context.tables["spm_unit"].columns))
        return original(self, context)

    monkeypatch.setattr(USCpsCarriedKernel, "run", observe)
    frame = _frame()
    compiled, sources, store, kernels = _setup(tmp_path, frame)
    assert all(value.entity != "spm_unit" for value in compiled.graph.nodes[2].inputs)
    run_graph(compiled, sources=sources, store=store, kernels=kernels)
    assert observed == [("spm_unit_id",)]


@pytest.mark.parametrize("entity", ("person", "spm_unit"))
def test_misbound_actual_entity_identity_refuses(tmp_path, entity):
    compiled, sources, store, kernels = _setup(tmp_path, _frame())
    source = compiled.graph.nodes[0]
    document = json.loads(source.params["frame_context"])
    document["entities"][entity]["ordered_ids_sha256"] = "0" * 64
    source = replace(
        source, params={"frame_context": canonical_json(document).decode()}
    )
    compiled = compile_graph(
        replace(compiled.graph, nodes=(source, *compiled.graph.nodes[1:]))
    )
    with pytest.raises(NodeRejectedError, match="identity differs"):
        run_graph(compiled, sources=sources, store=store, kernels=kernels)


def test_declares_every_actual_numeric_source_read(monkeypatch):
    original = cps_carried._source
    observed = set()

    def observe(person, column):
        observed.add(column)
        return original(person, column)

    monkeypatch.setattr(cps_carried, "_source", observe)
    cps_carried.derive_us_cps_carried_inputs(
        hours.fixtures._real_pre_clone_source_frame()
    )
    assert observed <= cps_carried.CPS_CARRIED_RAW_PERSON_COLUMNS
    assert {
        "OI_VAL",
        "OI_OFF",
        "SPM_SNAPSUB",
    } <= cps_carried.CPS_CARRIED_RAW_PERSON_COLUMNS


@pytest.mark.parametrize("dtype", ("bool", "float64"))
def test_incompatible_boolean_incumbent_representation_refuses_explicitly(
    tmp_path, dtype
):
    frame = _frame()
    values = [True, False] * 3 if dtype == "bool" else [np.nan] * 6
    frame.person["is_female"] = pd.Series(values, index=frame.person.index, dtype=dtype)
    with pytest.raises(ValueError, match="requires nullable boolean"):
        _setup(tmp_path, frame)


@pytest.mark.parametrize("module", _CPS_IMPLEMENTATION_MODULES)
def test_actual_module_bytes_bind_cps_implementation(monkeypatch, module):
    kernel = USCpsCarriedKernel()
    before = kernel.implementation_hash()
    target = Path(import_module(module).__file__).resolve()
    original = Path.read_bytes

    def changed(path):
        content = original(path)
        return (
            content + b"\n# intercepted implementation revision\n"
            if path.resolve() == target
            else content
        )

    monkeypatch.setattr(Path, "read_bytes", changed)
    assert kernel.implementation_hash() != before


def test_actual_source_resource_binds_cps_implementation(monkeypatch):
    from importlib.resources import files

    kernel = USCpsCarriedKernel()
    before = kernel.implementation_hash()
    target = Path(files("microcosm.build.us").joinpath("source_stages.json")).resolve()
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (
            original(path) + b" " if path.resolve() == target else original(path)
        ),
    )
    assert kernel.implementation_hash() != before


@pytest.mark.parametrize("dependency", CPS_CARRIED_DEPENDENCIES)
def test_dependency_versions_bind_cps_implementation(monkeypatch, dependency):
    from microcosm.graph import kernel as graph_kernel

    kernel = USCpsCarriedKernel()
    before = kernel.implementation_hash()
    original = graph_kernel.importlib_metadata.version
    monkeypatch.setattr(
        graph_kernel.importlib_metadata,
        "version",
        lambda name: (
            original(name) + ".probe" if name == dependency else original(name)
        ),
    )
    assert kernel.implementation_hash() != before


def test_live_formula_projection_changes_actual_receipt_and_cps_identity(monkeypatch):
    kernel = USCpsCarriedKernel()
    before = kernel.implementation_hash()
    old_contract = pool.multispine_cps_carried_implementation_contract()
    monkeypatch.setattr(
        pool,
        "_FORMULA_OWNED_SOURCE_OUTPUTS",
        {
            **pool._FORMULA_OWNED_SOURCE_OUTPUTS,
            "person": frozenset({*pool._FORMULA_OWNED_SOURCE_OUTPUTS["person"], "age"}),
        },
    )
    changed = pool.multispine_cps_carried_implementation_contract()
    assert changed["transient_outputs"] != old_contract["transient_outputs"]
    assert kernel.implementation_hash() != before


def test_live_output_registry_changes_cps_identity(monkeypatch):
    kernel = USCpsCarriedKernel()
    before = kernel.implementation_hash()
    monkeypatch.setitem(
        pool.PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
        "cps_carried",
        {
            **pool.PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES["cps_carried"],
            "person": frozenset({"age"}),
        },
    )
    assert kernel.implementation_hash() != before
