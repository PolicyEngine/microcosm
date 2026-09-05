"""The bounded fresh-source housing graph reproduces the direct pre-clone pass.

Every fixture here is generated: the ACS rent donor is a real archive-layout H5
written by this test, the recipient pool is an assembled mixed ASEC/ACS spine,
and the QRF fits two trees. No source microdata is read and no model is
downloaded. The graph result is compared both against the direct registered
operator chain computed in-process and against the payload oracle captured from
the unedited runtime before this module existed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from functools import partial
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import _graph_housing_source_artifacts as codec
from microcosm.build.us_runtime import graph_acs_rent as rent
from microcosm.build.us_runtime import graph_acs_rent_source as rent_source
from microcosm.build.us_runtime import graph_housing_source, housing_inputs
from microcosm.build.us_runtime import multispine_pool as pool
from microcosm.build.us_runtime.graph_context import (
    US_FRAME_CONTEXT_TYPE,
    _json_data,
    _mass_records,
    encode_us_frame_context,
)
from microcosm.build.us_runtime.graph_housing_source import (
    _HOUSING_IMPLEMENTATION_MODULES,
    HOUSING_SOURCE_ACCEPTED_CONTEXT,
    HOUSING_SOURCE_DEPENDENCIES,
    HOUSING_SOURCE_MERGE_POPULATION,
    HOUSING_SOURCE_POPULATION,
    HOUSING_SOURCE_PREFIX,
    USHousingMergeKernel,
    register_us_housing_source_kernels,
    us_housing_source_nodes,
)
from microcosm.build.us_runtime.graph_sources import frame_column_declarations
from microcosm.build.us_runtime.spine_assembly import assemble_spines
from microcosm.fit import graph_legacy_train
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    SourceRef,
    StructuralDelta,
    compile_graph,
    load_source,
    run_graph,
)
from microcosm.graph.availability import execution_state
from microcosm.graph.canonical import canonical_json
from microcosm.graph.codecs import SourceCodecRegistry, load_frame_store
from microcosm.graph.decl import GraphError
from microcosm.graph.errors import NodeRejectedError
from microcosm.graph.population import dtype_for_token

GOLDEN = Path(__file__).with_name("golden") / "us_housing_source_pre_graph.json"
SEED = pool.POOL_RANDOM_SEED
N_ESTIMATORS = 2
NUMERIC_RAW = (
    "H_TENURE",
    "SPM_CAPHOUSESUB",
    "SPM_TENMORTSTATUS",
    "HRSWK",
    "A_HRS1",
    "WKSWORK",
)
PREFIX_OPERATORS = (
    "derive_us_cps_carried_inputs",
    "with_us_hours_worked_inputs",
    "with_us_prior_year_income_inputs",
    "with_us_relationship_inputs",
)


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_housing_source_{name}", Path(__file__).with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pool_fixtures = _sibling("test_us_multispine_pool")
rent_fixtures = _sibling("test_us_graph_acs_rent")


# --------------------------------------------------------------------------
# Fixtures: the same mixed ASEC/ACS pool the pre-edit oracle captured.
# --------------------------------------------------------------------------


def _asec_spine() -> Frame:
    """Three CPS households, mixed tenure codes, one subsidized SPM unit."""
    base = pool_fixtures._real_pre_clone_source_frame()
    person = base.table("person").copy()
    third = person.iloc[[0, 1]].copy()
    third["person_id"] = [5, 6]
    third["person_household_id"] = [3, 3]
    third["person_tax_unit_id"] = [103, 103]
    third["person_spm_unit_id"] = [203, 203]
    third["person_family_id"] = [303, 303]
    third["person_marital_unit_id"] = [405, 406]
    third["PERIDNUM"] = ["third-parent", "third-child"]
    third["PH_SEQ"] = [30, 30]
    third["WSAL_VAL"] = [30_000.0, 0.0]
    third["A_AGE"] = [55, 14]
    person = pd.concat([person, third], ignore_index=True)
    person["SPM_CAPHOUSESUB"] = [0.0, 0.0, 4_800.0, 4_800.0, 0.0, 0.0]
    person["SPM_TENMORTSTATUS"] = [1, 1, 3, 3, 2, 2]
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "state_fips": [6.0, 36.0, 48.0],
                "H_TENURE": [1, 2, 3],
            }
        ),
        "tax_unit": pd.DataFrame(
            {"tax_unit_id": [101, 102, 103], "filing_status_input": ["SINGLE"] * 3}
        ),
        "spm_unit": pd.DataFrame({"spm_unit_id": [201, 202, 203]}),
        "family": pd.DataFrame({"family_id": [301, 302, 303]}),
        "marital_unit": pd.DataFrame(
            {"marital_unit_id": [401, 402, 403, 404, 405, 406]}
        ),
    }
    return Frame(
        tables,
        US_SCHEMA,
        {"household": Weights(np.array([1.0, 3.0, 2.0]), WeightKind.DESIGN)},
    )


def _acs_spine() -> Frame:
    """Two native non-CPS households already carrying observed rent and tenure."""
    frame = pool_fixtures._source_frame(offset=100.0)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"]["pre_subsidy_rent"] = np.array([777.0, 0.0])
    tables["household"]["state_fips"] = np.array([12.0, 24.0])
    tables["household"]["tenure_type"] = np.array(
        ["RENTED", "OWNED_OUTRIGHT"], dtype=object
    )
    tables["spm_unit"]["spm_unit_tenure_type"] = np.array(
        ["RENTER", "OWNER_WITHOUT_MORTGAGE"], dtype=object
    )
    return Frame(
        tables,
        US_SCHEMA,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
    )


def _pool_frame() -> Frame:
    """The assembled mixed pre-clone pool, matching the captured oracle."""
    frame = assemble_spines(
        {"asec": _asec_spine(), "acs": _acs_spine()},
        household_mass_shares={"asec": 0.6, "acs": 0.4},
    )
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for table in tables.values():
        for column in table:
            if pd.api.types.is_object_dtype(table[column]) or isinstance(
                table[column].dtype, pd.StringDtype
            ):
                # The graph's canonical "string" token is python storage with
                # pd.NA; the content store refuses any other physical dtype.
                table[column] = table[column].astype(dtype_for_token("string"))
        for column in NUMERIC_RAW:
            if column in table:
                table[column] = pd.to_numeric(table[column]).astype("float64")
    person = tables["person"]
    person.index = pd.Index(np.arange(len(person)) * 7 + 101, name="original_row")
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata.set_axis(person.index),
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _prefix_operators():
    return {
        "derive_us_cps_carried_inputs": pool.derive_us_cps_carried_inputs,
        "with_us_hours_worked_inputs": pool._with_gated_us_hours_worked_inputs,
        "with_us_prior_year_income_inputs": lambda frame: (
            pool.with_us_prior_year_income_inputs(
                frame, seed=pool.POOL_RANDOM_SEED, time_period=pool.POOL_TIME_PERIOD
            )
        ),
        "with_us_relationship_inputs": lambda frame: pool.with_us_relationship_inputs(
            frame, seed=pool.POOL_RANDOM_SEED, time_period=pool.POOL_TIME_PERIOD
        ),
    }


def _relationship_base(frame: Frame) -> Frame:
    """The true direct relationship-produced predecessor of the housing stage.

    The additive relationship graph is still under review, so this integration
    supplies the real direct wrapper's Frame as the graph's base version. Full
    graph composition is therefore not claimed here; only the housing stage is
    executed through the graph.
    """
    return pool._run_source_operator_chain(
        frame,
        phase="pre_clone",
        operator_names=PREFIX_OPERATORS,
        operators=_prefix_operators(),
    ).frame


def _direct_housing(base: Frame, donor: pd.DataFrame):
    return pool._run_source_operator_chain(
        base,
        phase="pre_clone",
        operator_names=("with_us_housing_inputs",),
        operators={
            "with_us_housing_inputs": lambda frame: pool.with_us_housing_inputs(
                frame,
                seed=SEED,
                time_period=pool.POOL_TIME_PERIOD,
                acs_rent_donor=donor,
                n_estimators=N_ESTIMATORS,
            )
        },
    )


def _direct_eligibility(frame: Frame):
    return pool._run_source_operator_chain(
        frame,
        phase="pre_clone",
        operator_names=("with_us_eligibility_inputs",),
        operators={
            "with_us_eligibility_inputs": lambda current: (
                pool.with_us_eligibility_inputs(
                    current,
                    seed=pool.POOL_RANDOM_SEED,
                    time_period=pool.POOL_TIME_PERIOD,
                )
            )
        },
    )


# --------------------------------------------------------------------------
# Graph harness
# --------------------------------------------------------------------------


class _Create(KernelBase):
    ref = "test.housing_source.base@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.CREATE,
        numeric=Numeric.PLATFORM_BITWISE,
    )

    def run(self, context):
        return KernelResult(
            frame=load_source("frame-store", context.sources["population"]),
            artifacts={"frame_context": context.params["frame_context"].encode()},
        )


class _Consumer(KernelBase):
    """Stands in for the eligibility stage: it consumes the accepted context."""

    ref = "test.housing_source.consumer@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, numeric=Numeric.PLATFORM_BITWISE
    )

    def run(self, context):
        return KernelResult(
            receipt={
                "accepted_context_bytes": len(
                    context.artifacts["frame_context"].payload
                )
            }
        )


def _donor(root: Path, *, mutation=None) -> Path:
    return Path(rent_fixtures._donor_file(root / "donor.h5", mutation=mutation))


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _nodes(frame: Frame, descriptor: bytes, *, consumer=True, **kwargs):
    columns = frame_column_declarations(frame)
    produced = us_housing_source_nodes(
        columns,
        base="source",
        context_producer="source",
        donor_descriptor=descriptor,
        seed=SEED,
        n_estimators=N_ESTIMATORS,
        **kwargs,
    )
    nodes = [
        Node(
            "source",
            _Create.ref,
            sources=("population",),
            structural=StructuralDelta.CREATE,
            outputs=columns,
            params={"frame_context": encode_us_frame_context(frame).decode()},
            artifact_outputs=(ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),),
        ),
        *produced,
    ]
    if consumer:
        nodes.append(
            Node(
                "after",
                _Consumer.ref,
                population=HOUSING_SOURCE_MERGE_POPULATION,
                artifact_inputs=(
                    ArtifactInput(
                        "frame_context",
                        HOUSING_SOURCE_ACCEPTED_CONTEXT,
                        "frame_context",
                        US_FRAME_CONTEXT_TYPE,
                    ),
                ),
            )
        )
    return tuple(nodes), columns


def _setup(tmp_path, frame, *, donor_path=None, **kwargs):
    donor_path = _donor(tmp_path) if donor_path is None else Path(donor_path)
    donor_sha = _sha(donor_path)
    descriptor = rent_source.prepare_acs_rent_source_descriptor(
        donor_path, expected_sha256=donor_sha
    )
    codecs = SourceCodecRegistry()
    codecs.register("frame-store", load_frame_store)
    codecs.register(
        rent_source.RENT_SOURCE_CODEC,
        partial(rent_source.load_graph_acs_rent_source, expected_sha256=donor_sha),
    )
    nodes, columns = _nodes(frame, descriptor, **kwargs)
    graph = Graph(
        country="us",
        sources=(
            SourceRef("population", "frame-store"),
            rent_source.RENT_SOURCE,
        ),
        nodes=nodes,
    )
    kernels = KernelRegistry()
    kernels.register(_Create())
    kernels.register(_Consumer())
    register_us_housing_source_kernels(kernels, source_codecs=codecs)
    store = ContentStore(tmp_path / "store", codecs=codecs)
    population = ContentStore(tmp_path / "inputs").put_frame("a" * 64, frame)
    sources = {
        "population": population,
        rent_source.RENT_SOURCE_NAME: donor_path,
    }
    return compile_graph(graph), sources, store, kernels, nodes, donor_path


def _run(tmp_path, frame, *, resume="forbid", **kwargs):
    compiled, sources, store, kernels, nodes, donor_path = _setup(
        tmp_path, frame, **kwargs
    )
    manifest = run_graph(
        compiled, sources=sources, store=store, kernels=kernels, resume=resume
    )
    return manifest, store, nodes, donor_path


def _assert_unreached(manifest, node_ids) -> None:
    for node_id in node_ids:
        receipt = manifest.nodes[node_id]
        assert execution_state(receipt.receipt) == "unreached", node_id
        assert not receipt.artifacts and not receipt.opaque_artifacts, node_id
        assert receipt.frame_key is None and receipt.weight_key is None, node_id


def _payload(manifest, store, node, name):
    return store.load_bytes(manifest.nodes[node].opaque_artifacts[name])


def _restore(population: Frame, context: bytes) -> Frame:
    """Rebuild the operator-visible Frame from executor cells and typed context.

    Population cells, weights and strata are executor-owned; normative metadata
    and the legacy mass history travel in the accepted typed context, exactly as
    the landed rent and hours boundaries already assume.
    """
    document = json.loads(context)
    return Frame(
        {entity: population.table(entity) for entity in population.entities},
        US_SCHEMA,
        {
            entity: population.weights_for(entity)
            for entity in population.weighted_entities
        },
        population.strata,
        mass_log=_mass_records(document["mass_log"]),
        metadata=document["metadata"],
    )


def _assert_frames_equal(actual: Frame, expected: Frame) -> None:
    assert set(actual.entities) == set(expected.entities)
    for entity in expected.entities:
        pd.testing.assert_frame_equal(actual.table(entity), expected.table(entity))
    pd.testing.assert_series_equal(actual.strata, expected.strata)
    assert set(actual.weighted_entities) == set(expected.weighted_entities)
    for entity in expected.weighted_entities:
        assert actual.weights_for(entity).kind == expected.weights_for(entity).kind
        assert (
            actual.weights_for(entity).values.tobytes()
            == expected.weights_for(entity).values.tobytes()
        )
    assert _json_data(actual.metadata) == _json_data(expected.metadata)
    assert actual.mass_log == expected.mass_log


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text())


# --------------------------------------------------------------------------
# Oracle parity
# --------------------------------------------------------------------------


def _snapshot_table(table: pd.DataFrame) -> dict[str, object]:
    return {
        "columns": list(table.columns),
        "dtypes": [str(dtype) for dtype in table.dtypes],
        "index": {
            "class": type(table.index).__name__,
            "dtype": str(table.index.dtype),
            "name": table.index.name,
            "values": table.index.tolist(),
        },
        "values": [
            [None if value is pd.NA or value is None else value for value in row]
            for row in table.astype(object)
            .where(table.notna(), None)
            .to_numpy()
            .tolist()
        ],
    }


def _snapshot_frame(frame: Frame) -> dict[str, object]:
    return {
        "entities": {
            entity: _snapshot_table(frame.table(entity)) for entity in frame.entities
        },
        "weights": {
            entity: {
                "kind": frame.weights_for(entity).kind.value,
                "values": frame.weights_for(entity).values.tolist(),
            }
            for entity in frame.weighted_entities
        },
        "strata": frame.strata.tolist(),
        "metadata": _json_data(frame.metadata),
        "mass_log": [
            {
                "entity": record.entity,
                "old_total": record.old_total,
                "new_total": record.new_total,
                "declared_factor": record.declared_factor,
                "reason": record.reason,
            }
            for record in frame.mass_log
        ],
    }


def test_direct_pre_edit_oracle_is_unchanged(tmp_path, golden):
    """The unedited direct path this port must reproduce has not moved."""
    donor_path = _donor(tmp_path)
    donor = housing_inputs.load_acs_2022_rent_donor(
        donor_path, expected_sha256=_sha(donor_path)
    )
    start = _pool_frame()
    assert _snapshot_frame(start) == golden["pool"]
    base = _relationship_base(start)
    assert _snapshot_frame(base) == golden["relationship_base"]
    housing = _direct_housing(base, donor)
    assert _snapshot_frame(housing.frame) == golden["housing"]["frame"]
    assert (
        json.loads(json.dumps(housing.receipt, default=str))
        == (golden["housing"]["receipt"])
    )
    eligibility = _direct_eligibility(housing.frame)
    assert _snapshot_frame(eligibility.frame) == golden["eligibility"]["frame"]


# --------------------------------------------------------------------------
# Cold / auto / require parity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("resume", ["forbid", "auto", "require"])
def test_graph_matches_direct_values_receipt_context_and_payloads(
    tmp_path, golden, resume
):
    frame = _relationship_base(_pool_frame())
    if resume != "forbid":
        _run(tmp_path, frame)
    manifest, store, nodes, donor_path = _run(tmp_path, frame, resume=resume)
    donor = housing_inputs.load_acs_2022_rent_donor(
        donor_path, expected_sha256=_sha(donor_path)
    )
    expected = _direct_housing(frame, donor)
    accepted = _payload(
        manifest, store, HOUSING_SOURCE_ACCEPTED_CONTEXT, "frame_context"
    )
    actual = _restore(manifest.population(HOUSING_SOURCE_MERGE_POPULATION), accepted)
    _assert_frames_equal(actual, expected.frame)
    assert _snapshot_frame(actual) == golden["housing"]["frame"]

    merge = manifest.nodes[f"{HOUSING_SOURCE_PREFIX}.merge"]
    assert canonical_json(merge.receipt["source_preparation"]) == canonical_json(
        expected.receipt
    )
    gate = manifest.nodes[HOUSING_SOURCE_ACCEPTED_CONTEXT]
    assert gate.receipt["outcome"] == "pass"
    assert gate.receipt["scope"] == codec.HOUSING_GATE_SCOPE
    assert gate.receipt["mode"] == codec.FRESH_SOURCE_MODE

    document = json.loads(
        _payload(manifest, store, HOUSING_SOURCE_ACCEPTED_CONTEXT, "frame_context")
    )
    original = json.loads(_payload(manifest, store, "source", "frame_context"))
    assert document["metadata"] == original["metadata"]
    assert document["mass_log"] == original["mass_log"]
    assert document["weight_sources"] == original["weight_sources"]
    for entity in US_SCHEMA.entities:
        assert document["entities"][entity]["columns"] == list(
            actual.table(entity).columns
        )
    assert manifest.nodes["after"].receipt["accepted_context_bytes"] > 0


def test_required_replay_reuses_every_node(tmp_path):
    frame = _relationship_base(_pool_frame())
    cold, _, _, _ = _run(tmp_path, frame)
    warm, _, _, _ = _run(tmp_path, frame, resume="require")
    assert warm.key == cold.key
    assert all(record.hit for record in warm.nodes.values())


# --------------------------------------------------------------------------
# Selection, projection and measured alignment
# --------------------------------------------------------------------------


def test_source_projection_selects_actual_cps_rows_and_group_ids(tmp_path):
    frame = _relationship_base(_pool_frame())
    manifest, store, _, _ = _run(tmp_path, frame)
    projected = manifest.population(HOUSING_SOURCE_POPULATION)
    person = frame.table("person")
    mask = person["PERIDNUM"].notna().to_numpy()
    expected = frame.select(mask)
    owned = {column for _entity, column, _dtype in codec.merge_columns()}
    for entity in expected.entities:
        # The version's final state also carries this stage's measured leaves;
        # every column the FILTER carried is byte-identical to Frame.select.
        carried = [
            column for column in expected.table(entity).columns if column not in owned
        ]
        pd.testing.assert_frame_equal(
            projected.table(entity).loc[:, carried],
            expected.table(entity).loc[:, carried],
        )
    pd.testing.assert_series_equal(projected.strata, expected.strata)
    for entity in expected.weighted_entities:
        np.testing.assert_array_equal(
            projected.weights_for(entity).values, expected.weights_for(entity).values
        )
    projection = codec.decode_projection(
        _payload(manifest, store, HOUSING_SOURCE_POPULATION, "projection")
    )
    assert projection["mode"] == codec.FRESH_SOURCE_MODE
    assert projection["selection"] == {
        "column": "PERIDNUM",
        "phase": "pre_clone",
        "clone_index_column": "person_support_clone_index",
    }
    for entity in US_SCHEMA.entities:
        selected = projection["entities"][entity]["selected"]
        assert selected["rows"] == len(expected.table(entity))
        full = projection["entities"][entity]["full"]
        assert full["rows"] == len(frame.table(entity))

    context = json.loads(
        _payload(manifest, store, HOUSING_SOURCE_POPULATION, "frame_context")
    )
    assert context["metadata"] == {}
    assert context["mass_log"] == []
    for entity in US_SCHEMA.entities:
        assert context["entities"][entity]["rows"] == len(expected.table(entity))


def test_measured_leaves_align_by_spm_and_household_id(tmp_path):
    frame = _relationship_base(_pool_frame())
    manifest, _, _, _ = _run(tmp_path, frame)
    measured = manifest.population(HOUSING_SOURCE_POPULATION)
    source = frame.select(frame.table("person")["PERIDNUM"].notna().to_numpy())
    household, spm_unit = housing_inputs.derive_us_housing_tables(
        source.table("person"), source.table("household"), source.table("spm_unit")
    )
    produced = manifest.population(f"{rent.RENT_PREFIX}.boundary")
    np.testing.assert_array_equal(
        produced.table("household")["tenure_type"].to_numpy(dtype=object),
        household["tenure_type"].to_numpy(dtype=object),
    )
    for column in (
        "receives_housing_assistance",
        "takes_up_housing_assistance_if_eligible",
    ):
        np.testing.assert_array_equal(
            produced.table("spm_unit")[column].to_numpy(dtype=bool),
            spm_unit[column].to_numpy(dtype=bool),
        )
        assert produced.table("spm_unit")[column].dtype == np.dtype(bool)
    np.testing.assert_array_equal(
        produced.table("spm_unit")["spm_unit_tenure_type"].to_numpy(dtype=object),
        spm_unit["spm_unit_tenure_type"].to_numpy(dtype=object),
    )
    del measured


def test_native_non_cps_cells_survive_the_merge(tmp_path):
    frame = _relationship_base(_pool_frame())
    manifest, _, _, _ = _run(tmp_path, frame)
    merged = manifest.population(HOUSING_SOURCE_MERGE_POPULATION)
    person = merged.table("person")
    native = person["PERIDNUM"].isna().to_numpy()
    np.testing.assert_array_equal(
        person.loc[native, "pre_subsidy_rent"].to_numpy(),
        frame.table("person").loc[native, "pre_subsidy_rent"].to_numpy(),
    )
    household = merged.table("household")
    native_household = frame.table("household")["tenure_type"].notna().to_numpy()
    pd.testing.assert_series_equal(
        household.loc[native_household, "tenure_type"],
        frame.table("household").loc[native_household, "tenure_type"],
    )
    spm_unit = merged.table("spm_unit")
    native_spm = frame.table("spm_unit")["spm_unit_tenure_type"].notna().to_numpy()
    pd.testing.assert_series_equal(
        spm_unit.loc[native_spm, "spm_unit_tenure_type"],
        frame.table("spm_unit").loc[native_spm, "spm_unit_tenure_type"],
    )
    assert spm_unit.loc[native_spm, "receives_housing_assistance"].isna().all()
    assert str(spm_unit["receives_housing_assistance"].dtype) == "boolean"


# --------------------------------------------------------------------------
# Fresh-source refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("observed", ["partial", "complete"])
def test_observed_cps_rent_refuses_before_any_recipient_work(
    tmp_path, monkeypatch, observed
):
    frame = _relationship_base(_pool_frame())
    person = frame.table("person").copy()
    cps = person["PERIDNUM"].notna().to_numpy()
    values = person["pre_subsidy_rent"].to_numpy(dtype=float).copy()
    positions = np.flatnonzero(cps)
    values[positions if observed == "complete" else positions[:1]] = 1234.0
    person["pre_subsidy_rent"] = values
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    refused = Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )

    fits = []
    original = graph_legacy_train.fit_target
    recipients = []
    original_recipient = rent.USACSRentRecipientKernel.run

    def counted(*args, **kwargs):
        fits.append(1)
        return original(*args, **kwargs)

    def counted_recipient(self, context):
        recipients.append(1)
        return original_recipient(self, context)

    monkeypatch.setattr(graph_legacy_train, "fit_target", counted)
    monkeypatch.setattr(rent.USACSRentRecipientKernel, "run", counted_recipient)
    with pytest.raises(NodeRejectedError, match="fresh_source only"):
        _run(tmp_path, refused)
    # The donor CREATE is recipient-independent and may load first; nothing that
    # depends on the recipient projection runs once the boundary refuses.
    assert fits == [] and recipients == []


def test_unsupported_mode_is_refused_at_declaration(tmp_path):
    frame = _relationship_base(_pool_frame())
    with pytest.raises(ValueError, match="mode"):
        _nodes(frame, b"{}", mode="complete_incumbent")


def test_absent_string_tenure_incumbent_is_refused(tmp_path):
    frame = _relationship_base(_pool_frame())
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["household"] = tables["household"].drop(columns=["tenure_type"])
    stripped = Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    donor_path = _donor(tmp_path)
    descriptor = rent_source.prepare_acs_rent_source_descriptor(
        donor_path, expected_sha256=_sha(donor_path)
    )
    with pytest.raises(ValueError, match="canonical string"):
        _nodes(stripped, descriptor)


def test_numeric_tenure_incumbent_is_refused(tmp_path):
    frame = _relationship_base(_pool_frame())
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["household"]["tenure_type"] = np.arange(
        len(tables["household"]), dtype=np.float64
    )
    donor_path = _donor(tmp_path)
    descriptor = rent_source.prepare_acs_rent_source_descriptor(
        donor_path, expected_sha256=_sha(donor_path)
    )
    numeric = Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    with pytest.raises(ValueError, match="canonical string"):
        _nodes(numeric, descriptor)


def test_repeated_person_labels_with_numeric_incumbent_are_refused(tmp_path):
    """The inherited shared-merge label seam is refused, not worked around."""
    frame = _relationship_base(_pool_frame())
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    person = tables["person"]
    person.index = pd.Index([7] * len(person), name=person.index.name)
    duplicated = Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata.set_axis(person.index),
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    with pytest.raises(NodeRejectedError, match="repeated"):
        _run(tmp_path, duplicated)


# --------------------------------------------------------------------------
# Gate reachability and artifact integrity
# --------------------------------------------------------------------------


def test_failed_rent_validation_leaves_acceptance_and_downstream_unreached(
    tmp_path, monkeypatch
):
    frame = _relationship_base(_pool_frame())

    def broken(prepared, raw_rent):
        raise ValueError("injected rent finalization failure")

    monkeypatch.setattr(housing_inputs, "finalize_acs_rent", broken)
    manifest, _store, _nodes_, _donor_path = _run(tmp_path, frame)
    validation = manifest.nodes[f"{rent.RENT_PREFIX}.validation"]
    assert validation.receipt["outcome"] == "fail"
    assert (
        "injected rent finalization failure"
        in (validation.receipt["evidence"]["message"])
    )
    _assert_unreached(
        manifest,
        (
            rent.RENT_OUTPUT_CONTEXT,
            f"{HOUSING_SOURCE_PREFIX}.source_outputs",
            HOUSING_SOURCE_MERGE_POPULATION,
            f"{HOUSING_SOURCE_PREFIX}.merge",
            HOUSING_SOURCE_ACCEPTED_CONTEXT,
            "after",
        ),
    )
    assert HOUSING_SOURCE_MERGE_POPULATION not in manifest.populations


def test_failed_completion_gate_leaves_downstream_unreached(tmp_path, monkeypatch):
    frame = _relationship_base(_pool_frame())
    original = USHousingMergeKernel.run

    def tampered(self, context):
        result = original(self, context)
        payload = json.loads(result.artifacts["merge_evidence"])
        payload["merged_rows"]["person"] += 1
        return replace(
            result,
            artifacts={
                **result.artifacts,
                "merge_evidence": canonical_json(payload),
            },
        )

    monkeypatch.setattr(USHousingMergeKernel, "run", tampered)
    manifest, _store, _nodes_, _donor_path = _run(tmp_path, frame)
    gate = manifest.nodes[HOUSING_SOURCE_ACCEPTED_CONTEXT]
    assert gate.receipt["outcome"] == "fail"
    assert not gate.opaque_artifacts
    _assert_unreached(manifest, ("after",))


def test_pending_context_cannot_satisfy_the_accepted_context_edge(tmp_path):
    frame = _relationship_base(_pool_frame())
    donor_path = _donor(tmp_path)
    descriptor = rent_source.prepare_acs_rent_source_descriptor(
        donor_path, expected_sha256=_sha(donor_path)
    )
    nodes, _ = _nodes(frame, descriptor, consumer=False)
    bad = Node(
        "after",
        _Consumer.ref,
        population=HOUSING_SOURCE_MERGE_POPULATION,
        artifact_inputs=(
            ArtifactInput(
                "frame_context",
                f"{HOUSING_SOURCE_PREFIX}.merge",
                "frame_context",
                US_FRAME_CONTEXT_TYPE,
            ),
        ),
    )
    with pytest.raises(GraphError):
        compile_graph(
            Graph(
                country="us",
                sources=(
                    SourceRef("population", "frame-store"),
                    rent_source.RENT_SOURCE,
                ),
                nodes=(*nodes, bad),
            )
        )


# --------------------------------------------------------------------------
# Implementation identity, resources and rosters
# --------------------------------------------------------------------------


def test_implementation_hash_binds_modules_resources_and_string_options(monkeypatch):
    from microcosm.build.us_runtime.graph_housing_source import (
        USHousingSourceBoundaryKernel,
    )

    kernel = USHousingSourceBoundaryKernel()
    first = kernel.implementation_hash()
    assert first == kernel.implementation_hash()
    with pd.option_context("mode.string_storage", "pyarrow"):
        assert kernel.implementation_hash() != first
    monkeypatch.setattr(
        graph_housing_source,
        "multispine_housing_implementation_contract",
        lambda: {"outputs": {}},
    )
    assert kernel.implementation_hash() != first


def test_implementation_modules_exist_and_include_every_bound_seam():
    for name in _HOUSING_IMPLEMENTATION_MODULES:
        assert import_module(name) is not None
    required = {
        "microcosm.build.us_runtime.graph_housing_source",
        "microcosm.build.us_runtime._graph_housing_source_artifacts",
        "microcosm.build.us_runtime.graph_acs_rent",
        "microcosm.build.us_runtime.graph_acs_rent_source",
        "microcosm.build.us_runtime._graph_acs_rent_artifacts",
        "microcosm.build.us_runtime.housing_inputs",
        "microcosm.build.us_runtime.multispine_pool",
        "microcosm.build.us_runtime.operator_boundary",
        "microcosm.build.us_runtime.support_provenance",
        "microcosm.build.us_runtime.graph_context",
    }
    assert required <= set(_HOUSING_IMPLEMENTATION_MODULES)
    assert HOUSING_SOURCE_DEPENDENCIES == ("numpy", "pandas", "pyarrow")


def test_live_housing_output_roster_is_bound_not_hardcoded(monkeypatch):
    assert codec.housing_output_family() == {
        "person": ("pre_subsidy_rent",),
        "household": ("tenure_type",),
        "spm_unit": (
            "receives_housing_assistance",
            "spm_unit_tenure_type",
            "takes_up_housing_assistance_if_eligible",
        ),
    }


# --------------------------------------------------------------------------
# Donor and recipient cache behavior
# --------------------------------------------------------------------------


def test_recipient_changes_reuse_the_actual_fitted_forest(tmp_path, monkeypatch):
    frame = _relationship_base(_pool_frame())
    calls = []
    original = graph_legacy_train.fit_target

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(graph_legacy_train, "fit_target", counted)
    first, _, _, _ = _run(tmp_path, frame)
    changed_tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    changed_tables["person"]["age"] = (
        changed_tables["person"]["age"].to_numpy(dtype=float) + 3.0
    )
    changed = Frame(
        changed_tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    second, _, _, _ = _run(tmp_path, changed, resume="auto")
    fit = f"{rent.RENT_PREFIX}.fit.000"
    assert calls == [1]
    assert first.nodes[fit].key == second.nodes[fit].key
    assert second.nodes[fit].hit


def test_donor_change_invalidates_the_fit(tmp_path):
    frame = _relationship_base(_pool_frame())
    first, _, _, _ = _run(tmp_path, frame)
    changed_root = tmp_path / "changed"
    changed_root.mkdir()
    changed = _donor(changed_root, mutation="values")
    second, _, _, _ = _run(tmp_path, frame, resume="auto", donor_path=changed)
    fit = f"{rent.RENT_PREFIX}.fit.000"
    assert first.nodes[fit].key != second.nodes[fit].key
    assert not second.nodes[fit].hit


# --------------------------------------------------------------------------
# Declaration, producer and payload refusals inside a real run
# --------------------------------------------------------------------------


def _rewire(nodes, node_id, **changes):
    return tuple(
        replace(node, **changes) if node.id == node_id else node for node in nodes
    )


def _run_nodes(tmp_path, frame, nodes, donor_path, *, resume="forbid"):
    donor_sha = _sha(donor_path)
    codecs = SourceCodecRegistry()
    codecs.register("frame-store", load_frame_store)
    codecs.register(
        rent_source.RENT_SOURCE_CODEC,
        partial(rent_source.load_graph_acs_rent_source, expected_sha256=donor_sha),
    )
    kernels = KernelRegistry()
    kernels.register(_Create())
    kernels.register(_Consumer())
    register_us_housing_source_kernels(kernels, source_codecs=codecs)
    compiled = compile_graph(
        Graph(
            country="us",
            sources=(
                SourceRef("population", "frame-store"),
                rent_source.RENT_SOURCE,
            ),
            nodes=nodes,
        )
    )
    return run_graph(
        compiled,
        sources={
            "population": ContentStore(tmp_path / "inputs").put_frame("a" * 64, frame),
            rent_source.RENT_SOURCE_NAME: donor_path,
        },
        store=ContentStore(tmp_path / "store", codecs=codecs),
        kernels=kernels,
        resume=resume,
    )


def _declared(tmp_path, frame):
    donor_path = _donor(tmp_path)
    descriptor = rent_source.prepare_acs_rent_source_descriptor(
        donor_path, expected_sha256=_sha(donor_path)
    )
    nodes, _columns = _nodes(frame, descriptor)
    return nodes, donor_path


def test_swapped_context_producer_on_the_merge_is_refused(tmp_path):
    frame = _relationship_base(_pool_frame())
    nodes, donor_path = _declared(tmp_path, frame)
    merge = next(node for node in nodes if node.id == f"{HOUSING_SOURCE_PREFIX}.merge")
    swapped = tuple(
        replace(edge, producer="source") if edge.name == "frame_context" else edge
        for edge in merge.artifact_inputs
    )
    nodes = _rewire(nodes, f"{HOUSING_SOURCE_PREFIX}.merge", artifact_inputs=swapped)
    with pytest.raises(NodeRejectedError, match="authored artifact producers"):
        _run_nodes(tmp_path, frame, nodes, donor_path)


@pytest.mark.parametrize(
    "node_id",
    [
        HOUSING_SOURCE_POPULATION,
        f"{HOUSING_SOURCE_PREFIX}.measured",
        f"{HOUSING_SOURCE_PREFIX}.source_outputs",
        HOUSING_SOURCE_MERGE_POPULATION,
        f"{HOUSING_SOURCE_PREFIX}.merge",
    ],
)
def test_wrong_outer_phase_is_refused_at_every_node(tmp_path, node_id):
    frame = _relationship_base(_pool_frame())
    nodes, donor_path = _declared(tmp_path, frame)
    node = next(item for item in nodes if item.id == node_id)
    nodes = _rewire(nodes, node_id, params={**dict(node.params), "phase": "post_clone"})
    with pytest.raises(NodeRejectedError):
        _run_nodes(tmp_path, frame, nodes, donor_path)


def test_wrong_mode_param_is_refused_at_the_boundary(tmp_path):
    frame = _relationship_base(_pool_frame())
    nodes, donor_path = _declared(tmp_path, frame)
    node = next(item for item in nodes if item.id == HOUSING_SOURCE_POPULATION)
    nodes = _rewire(
        nodes,
        HOUSING_SOURCE_POPULATION,
        params={**dict(node.params), "mode": "complete_incumbent"},
    )
    with pytest.raises(NodeRejectedError, match="mode"):
        _run_nodes(tmp_path, frame, nodes, donor_path)


def test_conserving_mass_on_the_row_dropping_source_filter_is_refused(tmp_path):
    frame = _relationship_base(_pool_frame())
    nodes, donor_path = _declared(tmp_path, frame)
    nodes = _rewire(nodes, HOUSING_SOURCE_POPULATION, mass="conserve")
    with pytest.raises(NodeRejectedError):
        _run_nodes(tmp_path, frame, nodes, donor_path)


def test_tampered_source_output_payload_is_refused_downstream(tmp_path, monkeypatch):
    frame = _relationship_base(_pool_frame())
    original = graph_housing_source.USHousingSourceOutputsKernel.run

    def tampered(self, context):
        result = original(self, context)
        payload = bytearray(result.artifacts["source_outputs"])
        payload[-1] ^= 0xFF
        return replace(
            result, artifacts={**result.artifacts, "source_outputs": bytes(payload)}
        )

    monkeypatch.setattr(
        graph_housing_source.USHousingSourceOutputsKernel, "run", tampered
    )
    with pytest.raises(NodeRejectedError, match="digest"):
        _run(tmp_path, frame)


def test_source_output_bound_to_a_foreign_projection_is_refused(tmp_path, monkeypatch):
    frame = _relationship_base(_pool_frame())
    original = graph_housing_source.USHousingSourceOutputsKernel.run

    def swapped(self, context):
        result = original(self, context)
        header, body = _split_outputs(result.artifacts["source_outputs"])
        header["projection_sha256"] = "0" * 64
        return replace(
            result,
            artifacts={
                **result.artifacts,
                "source_outputs": _reframe_outputs(header, body),
            },
        )

    monkeypatch.setattr(
        graph_housing_source.USHousingSourceOutputsKernel, "run", swapped
    )
    with pytest.raises(NodeRejectedError, match="stale"):
        _run(tmp_path, frame)


def _split_outputs(payload):
    start = len(codec.OUTPUTS_MAGIC)
    length = int.from_bytes(payload[start : start + 4], "big")
    return (
        json.loads(payload[start + 4 : start + 4 + length]),
        payload[start + 4 + length :],
    )


def _reframe_outputs(header, body):
    encoded = codec.encode_json(header)
    return codec.OUTPUTS_MAGIC + len(encoded).to_bytes(4, "big") + encoded + body


# --------------------------------------------------------------------------
# Unsupported source scope and raw inputs
# --------------------------------------------------------------------------


def _with_person(frame: Frame, person: pd.DataFrame) -> Frame:
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["person"] = person
    return Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def test_post_clone_support_population_is_refused(tmp_path):
    frame = _relationship_base(_pool_frame())
    person = frame.table("person").copy()
    clone = person["person_support_clone_index"].to_numpy().copy()
    clone[-1] = 1
    person["person_support_clone_index"] = clone
    with pytest.raises(NodeRejectedError, match="clone provenance"):
        _run(tmp_path, _with_person(frame, person))


def test_missing_assembly_manifest_is_refused(tmp_path):
    frame = _relationship_base(_pool_frame())
    stripped = Frame(
        {entity: frame.table(entity).copy() for entity in frame.entities},
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
    )
    with pytest.raises(NodeRejectedError, match="assembly manifest"):
        _run(tmp_path, stripped)


def test_absent_raw_household_tenure_is_refused_at_declaration(tmp_path):
    frame = _relationship_base(_pool_frame())
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    tables["household"] = tables["household"].drop(columns=["H_TENURE"])
    stripped = Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    with pytest.raises(ValueError, match="H_TENURE"):
        _nodes(stripped, b"{}")


@pytest.mark.parametrize(
    ("column", "values", "match"),
    [
        ("H_TENURE", [9.0, 2.0, 3.0, np.nan, np.nan], "unknown code"),
        ("SPM_CAPHOUSESUB", None, "negative"),
        ("SPM_TENMORTSTATUS", None, "constant"),
    ],
)
def test_negative_raw_code_and_constancy_errors_are_refused(
    tmp_path, column, values, match
):
    frame = _relationship_base(_pool_frame())
    if column == "H_TENURE":
        tables = {entity: frame.table(entity).copy() for entity in frame.entities}
        tables["household"]["H_TENURE"] = np.asarray(values, dtype=np.float64)
        broken = Frame(
            tables,
            frame.schema,
            {e: frame.weights_for(e) for e in frame.weighted_entities},
            frame.strata,
            mass_log=frame.mass_log,
            metadata=frame.metadata,
        )
    else:
        person = frame.table("person").copy()
        raw = person[column].to_numpy(dtype=float).copy()
        if column == "SPM_CAPHOUSESUB":
            # Both members of the first SPM unit, so the constancy check passes
            # and the negative-value check is the one that fires.
            raw[:2] = -1.0
        else:
            raw[0] = raw[0] + 1.0
        person[column] = raw
        broken = _with_person(frame, person)
    with pytest.raises(NodeRejectedError, match=match):
        _run(tmp_path, broken)


# --------------------------------------------------------------------------
# Group-view reality and ID-only SPM rows
# --------------------------------------------------------------------------


def test_spm_group_views_are_real_id_only_rows_not_placeholders(tmp_path):
    frame = _relationship_base(_pool_frame())
    manifest, _store, _nodes_, _donor_path = _run(tmp_path, frame)
    merged = manifest.population(HOUSING_SOURCE_MERGE_POPULATION)
    source = frame.select(frame.table("person")["PERIDNUM"].notna().to_numpy())
    selected_ids = np.unique(source.table("person")["person_spm_unit_id"].to_numpy())
    spm_unit = merged.table("spm_unit")
    np.testing.assert_array_equal(
        np.sort(spm_unit["spm_unit_id"].to_numpy()),
        np.sort(frame.table("spm_unit")["spm_unit_id"].to_numpy()),
    )
    treated = spm_unit["spm_unit_id"].isin(selected_ids).to_numpy()
    assert treated.sum() == len(selected_ids)
    assert spm_unit.loc[treated, "receives_housing_assistance"].notna().all()
    assert spm_unit.loc[~treated, "receives_housing_assistance"].isna().all()


# --------------------------------------------------------------------------
# Cache identity
# --------------------------------------------------------------------------


def test_string_option_change_rekeys_the_stage_without_touching_the_donor_fit(
    tmp_path,
):
    frame = _relationship_base(_pool_frame())
    first, _store, _nodes_, _donor_path = _run(tmp_path, frame)
    with pd.option_context("mode.string_storage", "python"):
        second, _s, _n, _d = _run(tmp_path, frame, resume="auto")
    stage = f"{HOUSING_SOURCE_PREFIX}.measured"
    assert first.nodes[stage].key != second.nodes[stage].key
    assert not second.nodes[stage].hit


def test_recipient_row_order_change_reuses_the_donor_fit(tmp_path, monkeypatch):
    frame = _relationship_base(_pool_frame())
    calls = []
    original = graph_legacy_train.fit_target

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(graph_legacy_train, "fit_target", counted)
    first, _store, _nodes_, _donor_path = _run(tmp_path, frame)
    person = frame.table("person").copy()
    person.index = pd.Index(np.arange(len(person)) * 11 + 3, name=person.index.name)
    reordered = Frame(
        {
            **{entity: frame.table(entity).copy() for entity in frame.entities},
            "person": person,
        },
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata.set_axis(person.index),
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    second, _s, _n, _d = _run(tmp_path, reordered, resume="auto")
    fit = f"{rent.RENT_PREFIX}.fit.000"
    assert calls == [1]
    assert first.nodes[fit].key == second.nodes[fit].key and second.nodes[fit].hit


# --------------------------------------------------------------------------
# Flat inventory and CI grouping
# --------------------------------------------------------------------------


def test_test_files_are_flat_and_land_in_the_authored_ci_groups():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_housing_ci_groups",
        Path(__file__).resolve().parents[3] / "tools" / "ci_test_groups.py",
    )
    groups = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = groups
    spec.loader.exec_module(groups)
    for name in (
        "test_us_graph_housing_source.py",
        "test_us_graph_housing_source_artifacts.py",
    ):
        path = f"packages/microcosm-build/tests/{name}"
        assert (Path(__file__).with_name(name)).is_file()
        assert groups.engine_group(path) == "us-am"
        assert groups.fast_group(path) == "rest"
        assert groups.process_for("us-am", path) == "build"
    assert not groups.stray_nested_test_files(groups.tracked_test_files())


def test_new_runtime_modules_are_classified_and_source_blind():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_housing_blindness", Path(__file__).with_name("test_us_spine_blindness.py")
    )
    blindness = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = blindness
    spec.loader.exec_module(blindness)
    runtime = Path(graph_housing_source.__file__).parent
    for name in ("graph_housing_source.py", "_graph_housing_source_artifacts.py"):
        assert name in blindness._SPINE_BLIND_OPERATOR_MODULES
        assert blindness._source_spine_accesses((runtime / name).read_text()) == ()


# --------------------------------------------------------------------------
# Durable failure evidence on replay, restored bytes, and remaining scope
# --------------------------------------------------------------------------


@pytest.mark.parametrize("resume", ["forbid", "auto", "require"])
def test_failed_rent_validation_stays_unreached_on_every_replay(
    tmp_path, monkeypatch, resume
):
    frame = _relationship_base(_pool_frame())

    def broken(prepared, raw_rent):
        raise ValueError("injected rent finalization failure")

    monkeypatch.setattr(housing_inputs, "finalize_acs_rent", broken)
    cold, _store, _nodes_, _donor_path = _run(tmp_path, frame)
    manifest = cold if resume == "forbid" else _run(tmp_path, frame, resume=resume)[0]
    assert manifest.key == cold.key
    assert manifest.nodes[f"{rent.RENT_PREFIX}.validation"].receipt["outcome"] == "fail"
    _assert_unreached(
        manifest,
        (
            rent.RENT_OUTPUT_CONTEXT,
            f"{HOUSING_SOURCE_PREFIX}.source_outputs",
            HOUSING_SOURCE_MERGE_POPULATION,
            f"{HOUSING_SOURCE_PREFIX}.merge",
            HOUSING_SOURCE_ACCEPTED_CONTEXT,
            "after",
        ),
    )


@pytest.mark.parametrize("resume", ["forbid", "auto", "require"])
def test_failed_completion_gate_stays_unreached_on_every_replay(
    tmp_path, monkeypatch, resume
):
    frame = _relationship_base(_pool_frame())
    original = USHousingMergeKernel.run

    def tampered(self, context):
        result = original(self, context)
        payload = json.loads(result.artifacts["merge_evidence"])
        payload["merged_rows"]["person"] += 1
        return replace(
            result,
            artifacts={**result.artifacts, "merge_evidence": canonical_json(payload)},
        )

    monkeypatch.setattr(USHousingMergeKernel, "run", tampered)
    cold, _store, _nodes_, _donor_path = _run(tmp_path, frame)
    manifest = cold if resume == "forbid" else _run(tmp_path, frame, resume=resume)[0]
    assert manifest.key == cold.key
    gate = manifest.nodes[HOUSING_SOURCE_ACCEPTED_CONTEXT]
    assert gate.receipt["outcome"] == "fail"
    assert gate.receipt["execution"]["unavailable_artifacts"] == ("frame_context",)
    assert "row counts differ" in gate.receipt["evidence"]["message"]
    _assert_unreached(manifest, ("after",))


def test_merged_pool_round_trips_through_a_real_frame_checkpoint(tmp_path):
    from microcosm.build.frame_checkpoint import (
        load_frame_checkpoint,
        write_frame_checkpoint,
    )

    frame = _relationship_base(_pool_frame())
    manifest, store, _nodes_, _donor_path = _run(tmp_path, frame)
    accepted = _payload(
        manifest, store, HOUSING_SOURCE_ACCEPTED_CONTEXT, "frame_context"
    )
    merged = _restore(manifest.population(HOUSING_SOURCE_MERGE_POPULATION), accepted)
    write_frame_checkpoint(tmp_path / "merged.h5", merged)
    restored = load_frame_checkpoint(tmp_path / "merged.h5").frame
    for entity in merged.entities:
        pd.testing.assert_frame_equal(restored.table(entity), merged.table(entity))
    for entity in merged.weighted_entities:
        np.testing.assert_array_equal(
            restored.weights_for(entity).values, merged.weights_for(entity).values
        )


def test_partially_populated_measured_incumbents_are_overwritten_on_cps_rows(
    tmp_path,
):
    frame = _relationship_base(_pool_frame())
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    household = tables["household"]
    cps_household = household["H_TENURE"].notna().to_numpy()
    tenure = household["tenure_type"].to_numpy(dtype=object).copy()
    tenure[np.flatnonzero(cps_household)[0]] = "NONE"
    household["tenure_type"] = pd.Series(
        tenure, index=household.index, dtype=dtype_for_token("string")
    )
    partial = Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )
    manifest, _store, _nodes_, donor_path = _run(tmp_path, partial)
    donor = housing_inputs.load_acs_2022_rent_donor(
        donor_path, expected_sha256=_sha(donor_path)
    )
    expected = _direct_housing(partial, donor)
    merged = manifest.population(HOUSING_SOURCE_MERGE_POPULATION)
    pd.testing.assert_series_equal(
        merged.table("household")["tenure_type"],
        expected.frame.table("household")["tenure_type"],
    )
    # The derived value replaces the stale CPS incumbent; natives are untouched.
    assert merged.table("household").loc[cps_household, "tenure_type"].tolist() == [
        "OWNED_WITH_MORTGAGE",
        "RENTED",
        "NONE",
    ]


def test_duplicate_household_heads_are_refused_by_the_existing_rent_stage(tmp_path):
    frame = _relationship_base(_pool_frame())
    person = frame.table("person").copy()
    heads = person["is_household_head"].to_numpy(dtype=object).copy()
    heads[1] = True
    person["is_household_head"] = pd.Series(heads, index=person.index, dtype="boolean")
    with pytest.raises(NodeRejectedError, match="head"):
        _run(tmp_path, _with_person(frame, person))


@pytest.mark.parametrize("stale", ["row_identity", "column_inventory"])
def test_stale_parent_context_is_refused_before_any_output_merges(tmp_path, stale):
    """A context that does not describe this population never reaches the merge."""
    frame = _relationship_base(_pool_frame())
    donor_path = _donor(tmp_path)
    descriptor = rent_source.prepare_acs_rent_source_descriptor(
        donor_path, expected_sha256=_sha(donor_path)
    )
    nodes, _columns = _nodes(frame, descriptor)
    if stale == "row_identity":
        # Describes a strict subset of the actual rows.
        described = frame.select(frame.table("person")["PERIDNUM"].notna().to_numpy())
        expected_node = HOUSING_SOURCE_POPULATION
    else:
        # Describes the un-operated pool: same rows, narrower column inventory,
        # so the stale context is only caught where the wider slice is read.
        described = _pool_frame()
        expected_node = f"{rent.RENT_PREFIX}.recipient"
    nodes = _rewire(
        nodes,
        "source",
        params={"frame_context": encode_us_frame_context(described).decode()},
    )
    with pytest.raises(NodeRejectedError, match=expected_node.replace(".", r"\.")):
        _run_nodes(tmp_path, frame, nodes, donor_path)


def test_measured_views_declare_no_support_role_or_undeclared_predictor_columns(
    tmp_path,
):
    frame = _relationship_base(_pool_frame())
    donor_path = _donor(tmp_path)
    descriptor = rent_source.prepare_acs_rent_source_descriptor(
        donor_path, expected_sha256=_sha(donor_path)
    )
    nodes, _columns = _nodes(frame, descriptor)
    measured = next(
        node for node in nodes if node.id == f"{HOUSING_SOURCE_PREFIX}.measured"
    )
    declared = {
        (view.entity, column) for view in measured.inputs for column in view.columns
    }
    assert declared == {
        ("person", "SPM_CAPHOUSESUB"),
        ("person", "SPM_TENMORTSTATUS"),
        ("person", "PERIDNUM"),
        ("person", "person_support_clone_index"),
        ("household", "H_TENURE"),
    }
    assert not any("support_channel" in column for _entity, column in declared)
    # No pension or social-security component is read by the measured node; the
    # rent recipient owns those, on its own declared slices.
    assert not any(
        column.startswith(("social_security", "taxable_", "tax_exempt_"))
        for _entity, column in declared
    )


def test_typed_artifact_payloads_stay_small_and_bounded(tmp_path):
    frame = _relationship_base(_pool_frame())
    manifest, store, _nodes_, _donor_path = _run(tmp_path, frame)
    sizes = {
        "projection": len(
            _payload(manifest, store, HOUSING_SOURCE_POPULATION, "projection")
        ),
        "source_outputs": len(
            _payload(
                manifest,
                store,
                f"{HOUSING_SOURCE_PREFIX}.source_outputs",
                "source_outputs",
            )
        ),
        "merge_evidence": len(
            _payload(
                manifest, store, f"{HOUSING_SOURCE_PREFIX}.merge", "merge_evidence"
            )
        ),
    }
    assert all(0 < size <= codec.HEADER_MAX_BYTES for size in sizes.values()), sizes
    # The five-leaf export is a bounded header plus exact column buffers, not a
    # Frame copy: it stays far under the projection's own carried columns.
    assert sizes["source_outputs"] < 8 * 1024


def test_wrong_outer_phase_on_the_gate_fails_the_gate_and_blocks_downstream(tmp_path):
    frame = _relationship_base(_pool_frame())
    nodes, donor_path = _declared(tmp_path, frame)
    gate = next(item for item in nodes if item.id == HOUSING_SOURCE_ACCEPTED_CONTEXT)
    nodes = _rewire(
        nodes,
        HOUSING_SOURCE_ACCEPTED_CONTEXT,
        params={**dict(gate.params), "phase": "post_clone"},
    )
    manifest = _run_nodes(tmp_path, frame, nodes, donor_path)
    receipt = manifest.nodes[HOUSING_SOURCE_ACCEPTED_CONTEXT]
    assert receipt.receipt["outcome"] == "fail"
    assert "authored outer phase" in receipt.receipt["evidence"]["message"]
    _assert_unreached(manifest, ("after",))


@pytest.mark.parametrize(
    ("node_id", "coordinate"),
    [
        (f"{HOUSING_SOURCE_PREFIX}.merge", ("person", "pre_subsidy_rent")),
        (f"{HOUSING_SOURCE_PREFIX}.merge", ("household", "tenure_type")),
        (f"{HOUSING_SOURCE_PREFIX}.measured", ("household", "tenure_type")),
    ],
)
def test_a_declaration_that_hides_a_carried_incumbent_is_refused(
    tmp_path, node_id, coordinate
):
    """Owning a carried leaf without a rewrite would erase its native cells."""
    frame = _relationship_base(_pool_frame())
    nodes, donor_path = _declared(tmp_path, frame)
    node = next(item for item in nodes if item.id == node_id)
    outputs = tuple(
        replace(owned, rewrite=False)
        if (owned.entity, owned.column) == coordinate
        else owned
        for owned in node.outputs
    )
    assert outputs != node.outputs
    nodes = _rewire(nodes, node_id, outputs=outputs)
    with pytest.raises(NodeRejectedError, match="must declare rewrite"):
        _run_nodes(tmp_path, frame, nodes, donor_path)


def test_a_declaration_that_hides_an_available_rent_incumbent_is_refused(tmp_path):
    """The fresh-source scope cannot be evaded by narrowing the declared slice."""
    frame = _relationship_base(_pool_frame())
    nodes, donor_path = _declared(tmp_path, frame)
    node = next(item for item in nodes if item.id == HOUSING_SOURCE_POPULATION)
    narrowed = tuple(
        replace(
            view,
            columns=tuple(
                column for column in view.columns if column != "pre_subsidy_rent"
            ),
        )
        for view in node.inputs
    )
    assert narrowed != node.inputs
    nodes = _rewire(nodes, HOUSING_SOURCE_POPULATION, inputs=narrowed)
    with pytest.raises(NodeRejectedError, match="must declare the available"):
        _run_nodes(tmp_path, frame, nodes, donor_path)
