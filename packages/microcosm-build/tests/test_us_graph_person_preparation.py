"""Relationship and eligibility person preparation on the real pre-clone graph.

The pre-clone source order is CPS-carried, accepted hours, prior year,
relationship, housing and eligibility. The housing adapter is authored
separately, so the composed graph here stitches a declared context/population
stand-in in housing's place and the eligibility node is additionally exercised
on the actual state the direct housing operator produces. This is a bounded
development fixture, not a full pre-clone graph.
"""

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from importlib import import_module
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_runtime import SourceRuntimeError
from microcosm.build.us_runtime import eligibility_inputs as eligibility
from microcosm.build.us_runtime import housing_inputs as housing
from microcosm.build.us_runtime import multispine_pool as pool
from microcosm.build.us_runtime import prior_year_income as prior
from microcosm.build.us_runtime import relationship_inputs as relationship
from microcosm.build.us_runtime.graph_context import (
    US_FRAME_CONTEXT_TYPE,
    _mass_records,
    encode_us_frame_context,
)
from microcosm.build.us_runtime.graph_cps_carried import (
    register_us_cps_carried_kernels,
    us_cps_carried_nodes,
)
from microcosm.build.us_runtime.graph_early_preparation import (
    HOURS_PHASE,
    US_HOURS_PENDING_CONTEXT_TYPE,
    register_us_hours_kernels,
    us_hours_nodes,
)
from microcosm.build.us_runtime.graph_person_preparation import (
    _PERSON_PREPARATION_IMPLEMENTATION_MODULES,
    ELIGIBILITY_NODE,
    PERSON_PREPARATION_DEPENDENCIES,
    RELATIONSHIP_NODE,
    USEligibilitySourceKernel,
    USRelationshipSourceKernel,
    register_us_person_preparation_kernels,
    us_eligibility_source_nodes,
    us_relationship_source_nodes,
)
from microcosm.build.us_runtime.graph_prior_year import (
    register_us_prior_year_source_kernels,
    us_prior_year_source_nodes,
)
from microcosm.build.us_runtime.graph_sources import frame_column_declarations
from microcosm.frame import Frame, Weights
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
    Slice,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.errors import NodeRejectedError


def _fixtures():
    spec = importlib.util.spec_from_file_location(
        "_person_preparation_fixtures",
        Path(__file__).with_name("test_us_graph_cps_carried.py"),
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cps = _fixtures()
_assert_equal = cps._assert_equal

OPERATORS = (
    "derive_us_cps_carried_inputs",
    "with_us_hours_worked_inputs",
    "with_us_prior_year_income_inputs",
    "with_us_relationship_inputs",
    "with_us_housing_inputs",
    "with_us_eligibility_inputs",
)
#: The housing node is authored in its own scope; the composed graph declares a
#: stand-in in its place, so graph/direct parity uses this subsequence.
GRAPH_OPERATORS = tuple(name for name in OPERATORS if name != "with_us_housing_inputs")
RELATIONSHIP_OUTPUTS = relationship.US_RELATIONSHIP_INPUTS_OUTPUT_COLUMNS
ELIGIBILITY_OUTPUTS = eligibility.US_ELIGIBILITY_INPUTS_OUTPUT_COLUMNS


def _frame():
    """The CPS-carried fixture with real ASEC relationship/eligibility variation."""
    frame = cps._frame()
    person = frame.person
    measured = person.PERIDNUM.notna()
    for column, values in (
        ("A_MARITL", ["4", "7", "6", "7"]),
        ("PEDISDRS", ["1", "2", "2", "2"]),
        ("PEDISEYE", ["2", "2", "1", "2"]),
        ("A_HSCOL", ["0", "2", "0", "0"]),
        ("A_FTPT", ["0", "1", "0", "0"]),
        ("A_AGE", ["40", "19", "41", "11"]),
    ):
        person.loc[measured, column] = values
    person.loc[measured, "VET_VAL"] = [0.0, 0.0, 7200.0, 0.0]
    person.loc[measured, "SSI_VAL"] = [0.0, 0.0, 0.0, 4800.0]
    # Native ACS head cells the CPS-only merge must leave untouched.
    person["is_household_head"] = pd.Series(pd.NA, index=person.index, dtype="boolean")
    person.loc[~measured, "is_household_head"] = [True, False]
    return frame


def _string_backed(frame):
    """Re-declare pandas ``str`` columns as ``string``, as the hours fixture does.

    The housing operator emits a ``str``-dtype tenure column that the graph
    store's declared ``string`` token does not accept. That is a housing-side
    storage seam, out of scope for this bounded person-preparation fixture.
    """
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for table in tables.values():
        for column in table:
            if table[column].dtype == object or str(table[column].dtype) == "str":
                table[column] = table[column].astype("string")
    return Frame(
        tables,
        frame.schema,
        {entity: frame.weights_for(entity) for entity in frame.weighted_entities},
        frame.strata,
        mass_log=frame.mass_log,
        metadata=frame.metadata,
    )


def _operators():
    donor = cps.hours.fixtures._rent_donor()
    return {
        "derive_us_cps_carried_inputs": pool.derive_us_cps_carried_inputs,
        "with_us_hours_worked_inputs": pool._with_gated_us_hours_worked_inputs,
        "with_us_prior_year_income_inputs": lambda frame: (
            prior.with_us_prior_year_income_inputs(
                frame, seed=pool.POOL_RANDOM_SEED, time_period=pool.POOL_TIME_PERIOD
            )
        ),
        "with_us_relationship_inputs": lambda frame: (
            relationship.with_us_relationship_inputs(
                frame, seed=pool.POOL_RANDOM_SEED, time_period=pool.POOL_TIME_PERIOD
            )
        ),
        "with_us_housing_inputs": lambda frame: housing.with_us_housing_inputs(
            frame,
            seed=pool.POOL_RANDOM_SEED,
            time_period=pool.POOL_TIME_PERIOD,
            acs_rent_donor=donor,
        ),
        "with_us_eligibility_inputs": lambda frame: (
            eligibility.with_us_eligibility_inputs(
                frame, seed=pool.POOL_RANDOM_SEED, time_period=pool.POOL_TIME_PERIOD
            )
        ),
    }


def _direct(frame, operators=GRAPH_OPERATORS):
    return pool._run_source_operator_chain(
        frame,
        phase="pre_clone",
        operator_names=tuple(operators),
        operators={name: _operators()[name] for name in operators},
    )


def _upstream(family):
    """The real pre-clone state each family runs on.

    Relationship follows the prior-year join; eligibility follows housing, so
    its fixture runs the direct housing operator and re-declares the tenure
    column's storage. No housing graph node exists here.
    """
    if family == "relationship":
        return _direct(_frame(), OPERATORS[:3]).frame
    return _string_backed(_direct(_frame(), OPERATORS[:5]).frame)


def _family_operator(family):
    return (
        ("with_us_relationship_inputs",)
        if family == "relationship"
        else ("with_us_eligibility_inputs",)
    )


def _family_node(family):
    return RELATIONSHIP_NODE if family == "relationship" else ELIGIBILITY_NODE


def _family_outputs(family):
    return RELATIONSHIP_OUTPUTS if family == "relationship" else ELIGIBILITY_OUTPUTS


def _family_helper(family):
    return (
        "prepare_us_relationship_person"
        if family == "relationship"
        else "prepare_us_eligibility_person"
    )


def _family_observer(family):
    return (
        pool.observe_multispine_relationship_person
        if family == "relationship"
        else pool.observe_multispine_eligibility_person
    )


class _HousingSeam(KernelBase):
    """Declared stand-in for the separately authored housing adapter.

    It occupies housing's population and context edges so the eligibility
    factory's ``base``/``context_producer`` seam is exercised. It derives
    nothing and is not a housing implementation.
    """

    ref = "test.person_preparation.housing_seam@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.FILTER,
        numeric=Numeric.PLATFORM_BITWISE,
    )

    def run(self, context):
        return KernelResult(
            keep=pd.Series(
                True,
                index=pd.Index(context.tables["person"].person_id, name="person_id"),
                dtype=bool,
            ),
            artifacts={"frame_context": context.artifacts["frame_context"].payload},
        )


def _seam_nodes(columns, *, base, context_producer):
    return (
        Node(
            "early_preparation.housing_seam",
            _HousingSeam.ref,
            base=base,
            structural=StructuralDelta.FILTER,
            inputs=(Slice("person", ("PERIDNUM",)),),
            artifact_inputs=(
                ArtifactInput(
                    "frame_context",
                    context_producer,
                    "frame_context",
                    US_FRAME_CONTEXT_TYPE,
                ),
            ),
            artifact_outputs=(ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),),
        ),
    )


_FACTORIES = {
    "cps_carried": us_cps_carried_nodes,
    "hours": us_hours_nodes,
    "prior_year": us_prior_year_source_nodes,
    "relationship": us_relationship_source_nodes,
    "housing_seam": _seam_nodes,
    "eligibility": us_eligibility_source_nodes,
}
FULL_COMPOSITION = (
    "cps_carried",
    "hours",
    "prior_year",
    "relationship",
    "housing_seam",
    "eligibility",
)


def _setup(tmp_path, frame, *, families=FULL_COMPOSITION):
    store = ContentStore(tmp_path / "store")
    source = ContentStore(tmp_path / "inputs").put_frame("a" * 64, frame)
    columns = frame_column_declarations(frame)
    nodes = [
        Node(
            "source",
            cps.hours._Create.ref,
            sources=("population",),
            structural=StructuralDelta.CREATE,
            outputs=columns,
            params={"frame_context": encode_us_frame_context(frame).decode()},
            artifact_outputs=(ArtifactOutput("frame_context", US_FRAME_CONTEXT_TYPE),),
        )
    ]
    base = context_producer = "source"
    for family in families:
        produced = _FACTORIES[family](
            columns, base=base, context_producer=context_producer
        )
        nodes.extend(produced)
        inventory = {(value.entity, value.column): value for value in columns}
        for node in produced:
            for value in node.outputs:
                inventory[value.entity, value.column] = value
        columns = tuple(inventory.values())
        base, context_producer = produced[0].id, produced[-1].id
    nodes.append(
        Node(
            "after",
            cps.hours._Consumer.ref,
            population=base,
            artifact_inputs=(
                ArtifactInput(
                    "frame_context",
                    context_producer,
                    "frame_context",
                    US_FRAME_CONTEXT_TYPE,
                ),
            ),
        )
    )
    registry = KernelRegistry()
    registry.register(cps.hours._Create())
    registry.register(cps.hours._Consumer())
    registry.register(_HousingSeam())
    register_us_cps_carried_kernels(registry)
    register_us_hours_kernels(registry)
    register_us_prior_year_source_kernels(registry)
    register_us_person_preparation_kernels(registry)
    return (
        compile_graph(
            Graph(
                country="us",
                sources=(SourceRef("population", "frame-store"),),
                nodes=tuple(nodes),
            )
        ),
        {"population": source},
        store,
        registry,
    )


def _forbid_qrf(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Person preparation must never construct a QRF")

    monkeypatch.setattr(prior, "QRF", forbidden)


def _restored(actual, document):
    return Frame(
        {entity: actual.table(entity) for entity in actual.entities},
        actual.schema,
        {entity: actual.weights_for(entity) for entity in actual.weighted_entities},
        actual.strata,
        metadata=document["metadata"],
        mass_log=_mass_records(document["mass_log"]),
    )


def test_declared_pre_clone_order_keeps_relationship_before_housing_and_eligibility():
    assert pool.POOL_PRE_CLONE_SOURCE_OPERATOR_ORDER == OPERATORS
    assert OPERATORS.index("with_us_relationship_inputs") == 3
    assert OPERATORS.index("with_us_housing_inputs") == 4
    assert OPERATORS.index("with_us_eligibility_inputs") == 5


def test_actual_composition_cold_auto_require_matches_direct_values_and_receipts(
    tmp_path, monkeypatch
):
    _forbid_qrf(monkeypatch)
    frame = _frame()
    expected = _direct(frame)
    individual = []
    current = frame
    for name in GRAPH_OPERATORS:
        step = _direct(current, (name,))
        individual.append(step.receipt)
        current = step.frame
    _assert_equal(current, expected.frame)
    compiled, sources, store, kernels = _setup(tmp_path, frame)
    for resume in ("forbid", "auto", "require"):
        manifest = run_graph(
            compiled, sources=sources, store=store, kernels=kernels, resume=resume
        )
        actual = manifest.population(f"{ELIGIBILITY_NODE}.boundary")
        _assert_equal(actual, expected.frame)
        for node_id, receipt in zip(
            (
                "early_preparation.cps_carried",
                f"{HOURS_PHASE}.gate",
                "early_preparation.prior_year",
                RELATIONSHIP_NODE,
                ELIGIBILITY_NODE,
            ),
            individual,
            strict=True,
        ):
            assert canonical_json(
                manifest.node(node_id).receipt["source_preparation"]
            ) == canonical_json(receipt)
        node = manifest.node(ELIGIBILITY_NODE)
        context_payload = store.load_bytes(node.opaque_artifacts["frame_context"])
        assert context_payload == encode_us_frame_context(expected.frame)
        output_store = ContentStore(tmp_path / f"restored-{resume}")
        output_store.put_frame("b" * 64, _restored(actual, json.loads(context_payload)))
        output_store.put_frame("c" * 64, expected.frame)
        assert (
            output_store.metadata("b" * 64)["payloads"]
            == output_store.metadata("c" * 64)["payloads"]
        )
        assert manifest.node("after").receipt["received_context"]
        if resume != "forbid":
            assert all(node.hit for node in manifest.nodes.values())
    measured = actual.person.PERIDNUM.notna()
    assert actual.person.loc[measured, "is_household_head"].tolist() == [
        True,
        False,
        True,
        False,
    ]
    assert actual.person.loc[measured, "is_surviving_spouse"].tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert actual.person.loc[measured, "is_separated"].tolist() == [
        False,
        False,
        True,
        False,
    ]
    assert actual.person.loc[measured, "is_disabled"].tolist() == [
        True,
        False,
        True,
        True,
    ]
    assert actual.person.loc[measured, "is_blind"].tolist() == [
        False,
        False,
        True,
        False,
    ]
    assert actual.person.loc[measured, "is_full_time_college_student"].tolist() == [
        False,
        True,
        False,
        False,
    ]
    assert actual.person.loc[measured, "own_children_in_household"].tolist() == [
        1.0,
        0.0,
        1.0,
        0.0,
    ]
    assert actual.person.loc[measured, "veterans_benefits"].tolist() == [
        0.0,
        0.0,
        7200.0,
        0.0,
    ]
    # Native non-CPS rows keep their own head cells and stay null elsewhere.
    assert actual.person.loc[~measured, "is_household_head"].tolist() == [True, False]
    for column in (*RELATIONSHIP_OUTPUTS, *ELIGIBILITY_OUTPUTS):
        if column == "is_household_head":
            continue
        assert actual.person.loc[~measured, column].isna().all()
    for receipt in individual[-2:]:
        assert receipt["transient_outputs_carried_through_clone"] == {}
        assert receipt["suboperators"][0]["formula_owned_outputs_removed"] == {}
        assert receipt["suboperators"][0]["kernel_receipt"] == {}


def test_actual_eligibility_after_direct_housing_state_matches_direct(tmp_path):
    """The eligibility node runs on the state the real housing operator leaves.

    The housing graph adapter is authored separately, so this fixture reaches
    the post-housing pre-clone state through the direct operator. It is not a
    full pre-clone graph and certifies nothing about housing.
    """
    frame = _string_backed(_direct(_frame(), OPERATORS[:5]).frame)
    housing_outputs = pool.PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES["housing_inputs"]
    for entity, columns in housing_outputs.items():
        present = set(columns) & set(frame.table(entity).columns)
        assert present, f"housing state absent for {entity}"
    expected = _direct(frame, OPERATORS[5:])
    args, sources, store, kernels = _setup(tmp_path, frame, families=("eligibility",))
    result = run_graph(args, sources=sources, store=store, kernels=kernels)
    _assert_equal(result.population(f"{ELIGIBILITY_NODE}.boundary"), expected.frame)
    assert canonical_json(
        result.node(ELIGIBILITY_NODE).receipt["source_preparation"]
    ) == canonical_json(expected.receipt)
    assert store.load_bytes(
        result.node(ELIGIBILITY_NODE).opaque_artifacts["frame_context"]
    ) == encode_us_frame_context(expected.frame)


def test_full_legacy_pre_clone_composition_retains_pre_edit_byte_fingerprints(
    tmp_path,
):
    output = _direct(_frame(), OPERATORS)
    store = ContentStore(tmp_path)
    store.put_frame("b" * 64, output.frame)
    actual = {
        "payload": hashlib.sha256(
            canonical_json(store.metadata("b" * 64)["payloads"])
        ).hexdigest(),
        "receipt": hashlib.sha256(canonical_json(output.receipt)).hexdigest(),
        "context": hashlib.sha256(encode_us_frame_context(output.frame)).hexdigest(),
    }
    # Measured on unmodified base ebb47671, before the pool/graph edits here.
    assert actual == {
        "payload": "92bc9d0c9186b92bc7682409e1d2ce07da2e53ddba2b8076e31de4fd9bb32a22",
        "receipt": "b2ef05db284e05840744e500590ecf7673bb2ed39039fde3f3c662e8165031fd",
        "context": "6960d7e11b43345303bb1e2ecab83bf5c7718618373cbe1f678f755e896e7137",
    }


@pytest.mark.parametrize(
    ("family", "surface"),
    [
        ("relationship", surface)
        for surface in ("all_null", "partial", "pass_through", "shuffled_person")
    ]
    + [
        ("eligibility", surface)
        for surface in (
            "all_null",
            "partial",
            "pass_through",
            "constant_non_disability",
            "shuffled_person",
        )
    ],
)
def test_actual_presence_null_authority_and_row_order(
    tmp_path, monkeypatch, family, surface
):
    _forbid_qrf(monkeypatch)
    outputs = _family_outputs(family)
    operator = _family_operator(family)
    node = _family_node(family)
    frame = _upstream(family)
    measured = frame.person.PERIDNUM.notna()
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
        booleans = {
            "is_household_head",
            "is_separated",
            "is_surviving_spouse",
            "is_disabled",
            "is_blind",
            "is_full_time_college_student",
        }
        for name in outputs:
            frame.person[name] = (
                pd.Series(pd.NA, index=frame.person.index, dtype="boolean")
                if name in booleans
                else np.nan
            )
        for name in outputs:
            frame.person.loc[~measured, name] = (
                [True, False] if name in booleans else [11.0, 22.0]
            )
        if surface != "all_null":
            head = frame.person.index[0]
            for name in outputs:
                frame.person.loc[head, name] = True if name in booleans else 5.0
        if surface in {"pass_through", "constant_non_disability"}:
            varied = [True, False, True, False]
            for name in outputs:
                frame.person.loc[measured, name] = (
                    varied if name in booleans else [1.0, 2.0, 3.0, 4.0]
                )
            if surface == "constant_non_disability":
                for name in outputs:
                    if name in {"is_disabled", "is_household_head"}:
                        continue
                    frame.person.loc[measured, name] = True if name in booleans else 7.0
            # A signal-carrying surface passes through without reading raw
            # inputs, so the raw column may be missing entirely.
            frame.person.drop(
                columns="A_MARITL" if family == "relationship" else "PEPAR2",
                inplace=True,
            )
    expected = _direct(frame, operator)
    args, sources, store, kernels = _setup(tmp_path, frame, families=(family,))
    result = run_graph(args, sources=sources, store=store, kernels=kernels)
    _assert_equal(result.population(f"{node}.boundary"), expected.frame)
    assert canonical_json(
        result.node(node).receipt["source_preparation"]
    ) == canonical_json(expected.receipt)
    assert store.load_bytes(
        result.node(node).opaque_artifacts["frame_context"]
    ) == encode_us_frame_context(expected.frame)


@pytest.mark.parametrize("family", ("relationship", "eligibility"))
def test_complete_pass_through_never_recomputes(tmp_path, monkeypatch, family):
    """A signal-carrying surface passes through, exactly like the wrapper."""
    outputs = _family_outputs(family)
    helper = _family_helper(family)
    frame = _direct(_frame(), OPERATORS[:3]).frame
    booleans = {
        "is_household_head",
        "is_separated",
        "is_surviving_spouse",
        "is_disabled",
        "is_blind",
        "is_full_time_college_student",
    }
    for name in outputs:
        frame.person[name] = pd.Series(
            [True, False, True, False, True, False]
            if name in booleans
            else [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            index=frame.person.index,
            dtype="boolean" if name in booleans else "float64",
        )

    def forbidden(*args, **kwargs):
        raise AssertionError(f"{helper} must not run on a pass-through surface")

    monkeypatch.setattr(pool, helper, forbidden)
    operator = _family_operator(family)
    node = _family_node(family)
    expected = _direct(frame, operator)
    args, sources, store, kernels = _setup(tmp_path, frame, families=(family,))
    result = run_graph(args, sources=sources, store=store, kernels=kernels)
    _assert_equal(result.population(f"{node}.boundary"), expected.frame)
    assert canonical_json(
        result.node(node).receipt["source_preparation"]
    ) == canonical_json(expected.receipt)
    for name in outputs:
        assert result.population(f"{node}.boundary").person[name].tolist() == (
            frame.person[name].tolist()
        )


@pytest.mark.parametrize("family", ("relationship", "eligibility"))
def test_registered_observer_uses_actual_selected_weights_and_ids(monkeypatch, family):
    frame = _upstream(family)
    operator = _family_operator(family)
    observe = _family_observer(family)
    helper = _family_helper(family)
    expected = _direct(frame, operator)
    observed = []
    original = getattr(pool, helper)

    def capture(person, weights, **kwargs):
        observed.append((person.copy(), np.asarray(weights, dtype=np.float64)))
        return original(person, weights, **kwargs)

    monkeypatch.setattr(pool, helper, capture)
    result, receipt = observe(
        frame.person,
        weights=frame.resolve_weights("person"),
        metadata=frame.metadata,
        pool_rows={entity: frame.n(entity) for entity in frame.entities},
    )
    pd.testing.assert_frame_equal(result, expected.frame.person)
    assert canonical_json(receipt) == canonical_json(expected.receipt)
    selected, weights = observed[0]
    measured = frame.person.PERIDNUM.notna()
    assert selected.person_id.tolist() == (
        frame.person.loc[measured, "person_id"].tolist()
    )
    assert selected.index.equals(frame.person.index[measured])
    assert not {"person_support_channel", "person_support_clone_index"} & set(
        selected.columns
    )
    # The helper receives exactly the direct chain's available projection: role
    # metadata removed and union-created all-null outputs omitted.
    evidence = pool._cps_source_evidence_mask(frame, phase="pre_clone")
    available = pool._without_unavailable_output_columns(
        pool._source_available_projection(frame, evidence, phase="pre_clone"),
        pool.PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES[
            "relationship_inputs" if family == "relationship" else "eligibility_inputs"
        ],
    )
    assert list(selected.columns) == list(available.table("person").columns)
    canonical = frame.resolve_weights("person")
    assert weights.tobytes() == canonical.values[measured.to_numpy()].tobytes()
    assert len(set(weights.tolist())) > 1


@pytest.mark.parametrize(
    ("family", "failure"),
    [
        ("relationship", failure)
        for failure in (
            "missing_raw",
            "invalid_code",
            "two_heads",
            "support_context",
        )
    ]
    + [("eligibility", failure) for failure in ("missing_raw", "support_context")],
)
def test_original_source_refusal_or_preclone_misuse_refuses(
    tmp_path, monkeypatch, family, failure
):
    _forbid_qrf(monkeypatch)
    operator = _family_operator(family)
    frame = _upstream(family)
    measured = frame.person.PERIDNUM.notna()
    if failure == "missing_raw":
        frame.person.drop(
            columns="A_MARITL" if family == "relationship" else "PEPAR1",
            inplace=True,
        )
    elif failure == "invalid_code":
        frame.person.loc[frame.person.index[measured][0], "A_MARITL"] = "9"
    elif failure == "two_heads":
        frame.person.loc[frame.person.index[measured][1], "P_SEQ"] = "1"
    else:
        from microcosm.build.us_runtime.support_provenance import (
            support_clone_index_column,
        )

        frame.person.loc[
            frame.person.index[0], support_clone_index_column("person")
        ] = 1
    with pytest.raises((ValueError, SourceRuntimeError)) as direct:
        _direct(frame, operator)
    args, sources, store, kernels = _setup(tmp_path, frame, families=(family,))
    with pytest.raises(NodeRejectedError) as graph:
        run_graph(args, sources=sources, store=store, kernels=kernels)
    assert str(direct.value) in str(graph.value)


@pytest.mark.parametrize("family", ("relationship", "eligibility"))
def test_present_raw_read_and_output_ownership_cannot_be_omitted(tmp_path, family):
    node_id = _family_node(family)
    raw = "A_MARITL" if family == "relationship" else "PEPAR1"
    frame = _upstream(family)
    args, sources, store, kernels = _setup(tmp_path, frame, families=(family,))
    derive = next(node for node in args.graph.nodes if node.id == node_id)
    for changed in (
        replace(
            derive,
            inputs=tuple(
                replace(
                    value,
                    columns=tuple(name for name in value.columns if name != raw),
                )
                for value in derive.inputs
            ),
        ),
        replace(derive, outputs=derive.outputs[:-1]),
        replace(derive, params={"phase": "post_clone"}),
    ):
        compiled = compile_graph(
            replace(
                args.graph,
                nodes=tuple(
                    changed if node.id == derive.id else node
                    for node in args.graph.nodes
                ),
            )
        )
        with pytest.raises(
            NodeRejectedError,
            match="omits available|must own exactly|authored outer phase",
        ):
            run_graph(compiled, sources=sources, store=store, kernels=kernels)


@pytest.mark.parametrize("family", ("relationship", "eligibility"))
@pytest.mark.parametrize("field", ("identity", "counts", "weights"))
def test_malformed_full_context_refuses(tmp_path, family, field):
    frame = _upstream(family)
    args, sources, store, kernels = _setup(tmp_path, frame, families=(family,))
    source = args.graph.nodes[0]
    document = json.loads(source.params["frame_context"])
    if field == "identity":
        document["entities"]["person"]["ordered_ids_sha256"] = "0" * 64
    elif field == "counts":
        document["entities"]["household"]["rows"] += 1
    else:
        document["weight_sources"]["person"] = "design"
    source = replace(
        source, params={"frame_context": canonical_json(document).decode()}
    )
    changed = compile_graph(replace(args.graph, nodes=(source, *args.graph.nodes[1:])))
    with pytest.raises(
        NodeRejectedError,
        match="identity differs|counts differ|one explicit weight source",
    ):
        run_graph(changed, sources=sources, store=store, kernels=kernels)


@pytest.mark.parametrize("family", ("relationship", "eligibility"))
def test_multiple_explicit_weight_sources_refuse(tmp_path, family):
    frame = _upstream(family)
    frame = Frame(
        {entity: frame.table(entity) for entity in frame.entities},
        frame.schema,
        {
            "household": frame.weights_for("household"),
            "person": Weights(
                frame.resolve_weights("person").values,
                frame.resolve_weights("person").kind,
            ),
        },
        frame.strata,
        metadata=frame.metadata,
        mass_log=frame.mass_log,
    )
    args, sources, store, kernels = _setup(tmp_path, frame, families=(family,))
    with pytest.raises(NodeRejectedError, match="one explicit weight source"):
        run_graph(args, sources=sources, store=store, kernels=kernels)


@pytest.mark.parametrize(
    ("family", "column"),
    (
        ("relationship", "is_household_head"),
        ("eligibility", "is_disabled"),
    ),
)
@pytest.mark.parametrize("dtype", ("bool", "float64"))
def test_incompatible_boolean_incumbent_refuses_before_execution(
    tmp_path, family, column, dtype
):
    frame = _upstream(family)
    frame.person[column] = pd.Series(
        False if dtype == "bool" else np.nan, index=frame.person.index, dtype=dtype
    )
    with pytest.raises(ValueError, match="requires nullable boolean"):
        _setup(tmp_path, frame, families=(family,))


def test_person_preparation_requires_the_accepted_hours_context_not_pending(tmp_path):
    args, _, _, _ = _setup(tmp_path, _frame())
    boundary = next(
        node for node in args.graph.nodes if node.id == f"{RELATIONSHIP_NODE}.boundary"
    )
    assert boundary.artifact_inputs[0].producer == "early_preparation.prior_year"
    changed = replace(
        boundary,
        artifact_inputs=(replace(boundary.artifact_inputs[0], producer=HOURS_PHASE),),
    )
    with pytest.raises(ValueError, match="type"):
        compile_graph(
            replace(
                args.graph,
                nodes=tuple(
                    changed if node.id == boundary.id else node
                    for node in args.graph.nodes
                ),
            )
        )


def test_hours_pending_context_type_is_never_accepted():
    assert US_HOURS_PENDING_CONTEXT_TYPE != US_FRAME_CONTEXT_TYPE


def test_failed_hours_gate_leaves_person_preparation_unreached_on_every_resume(
    tmp_path, monkeypatch
):
    _forbid_qrf(monkeypatch)
    frame = _frame()
    measured = frame.person.PERIDNUM.notna()
    frame.person.loc[measured, ["HRSWK", "A_HRS1", "WKSWORK"]] = 0.0
    args, sources, store, kernels = _setup(tmp_path, frame)
    for resume in ("forbid", "auto", "require"):
        result = run_graph(
            args, sources=sources, store=store, kernels=kernels, resume=resume
        )
        assert result.node(f"{HOURS_PHASE}.gate").receipt["outcome"] == "fail"
        for name in (
            f"{RELATIONSHIP_NODE}.boundary",
            RELATIONSHIP_NODE,
            f"{ELIGIBILITY_NODE}.boundary",
            ELIGIBILITY_NODE,
            "after",
        ):
            assert result.node(name).receipt["execution"]["state"] == "unreached"
            assert result.node(name).opaque_artifacts == {}


@pytest.mark.parametrize("family", ("relationship", "eligibility"))
@pytest.mark.parametrize(
    "misuse", ("not_weights", "short_weights", "row_mismatch", "membership_mismatch")
)
def test_observer_refuses_malformed_weights_or_counts(family, misuse):
    frame = _upstream(family)
    observe = _family_observer(family)
    weights = frame.resolve_weights("person")
    rows = {entity: frame.n(entity) for entity in frame.entities}
    if misuse == "not_weights":
        weights = weights.values
    elif misuse == "short_weights":
        weights = Weights(weights.values[:-1], weights.kind)
    elif misuse == "row_mismatch":
        rows["person"] += 1
    else:
        rows["household"] += 1
    with pytest.raises(ValueError, match="misaligned|differ from memberships"):
        observe(
            frame.person,
            weights=weights,
            metadata=frame.metadata,
            pool_rows=rows,
        )


@pytest.mark.parametrize("family", ("relationship", "eligibility"))
def test_registered_projection_rejects_helper_identity_change(monkeypatch, family):
    frame = _upstream(family)
    helper = _family_helper(family)
    observe = _family_observer(family)
    original = getattr(pool, helper)

    def invalid(person, weights, **kwargs):
        output = original(person, weights, **kwargs).copy()
        output.loc[output.index[0], "person_id"] = -111
        return output

    monkeypatch.setattr(pool, helper, invalid)
    with pytest.raises(ValueError, match="identity|IDs|id"):
        observe(
            frame.person,
            weights=frame.resolve_weights("person"),
            metadata=frame.metadata,
            pool_rows={entity: frame.n(entity) for entity in frame.entities},
        )


def test_person_signal_validator_is_bound_to_the_implementation():
    assert (
        "microcosm.build.us_runtime._person_signal_summary"
        in _PERSON_PREPARATION_IMPLEMENTATION_MODULES
    )
    for name in (
        "microcosm.build.us_runtime.relationship_inputs",
        "microcosm.build.us_runtime.eligibility_inputs",
        "microcosm.build.us_runtime.multispine_pool",
        "microcosm.build.us_runtime.operator_boundary",
        "microcosm.build.us_runtime.graph_person_preparation",
    ):
        assert name in _PERSON_PREPARATION_IMPLEMENTATION_MODULES


def test_relationship_and_eligibility_identities_are_distinct():
    assert (
        USRelationshipSourceKernel().implementation_hash()
        != USEligibilitySourceKernel().implementation_hash()
    )


@pytest.mark.parametrize("module", _PERSON_PREPARATION_IMPLEMENTATION_MODULES)
@pytest.mark.parametrize(
    "kernel", (USRelationshipSourceKernel, USEligibilitySourceKernel)
)
def test_actual_module_bytes_bind_person_preparation(monkeypatch, module, kernel):
    instance = kernel()
    before = instance.implementation_hash()
    target = Path(import_module(module).__file__).resolve()
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (
            original(path) + b"\n# independent revision\n"
            if path.resolve() == target
            else original(path)
        ),
    )
    assert instance.implementation_hash() != before


@pytest.mark.parametrize("dependency", PERSON_PREPARATION_DEPENDENCIES)
@pytest.mark.parametrize(
    "kernel", (USRelationshipSourceKernel, USEligibilitySourceKernel)
)
def test_dependency_versions_bind_person_preparation(monkeypatch, dependency, kernel):
    from microcosm.graph import kernel as graph_kernel

    instance = kernel()
    before = instance.implementation_hash()
    original = graph_kernel.importlib_metadata.version
    monkeypatch.setattr(
        graph_kernel.importlib_metadata,
        "version",
        lambda name: (
            original(name) + ".probe" if name == dependency else original(name)
        ),
    )
    assert instance.implementation_hash() != before


@pytest.mark.parametrize(
    "kernel", (USRelationshipSourceKernel, USEligibilitySourceKernel)
)
def test_source_resource_binds_person_preparation(monkeypatch, kernel):
    instance = kernel()
    before = instance.implementation_hash()
    target = Path(files("microcosm.build.us").joinpath("source_stages.json")).resolve()
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (
            original(path) + b" " if path.resolve() == target else original(path)
        ),
    )
    assert instance.implementation_hash() != before


@pytest.mark.parametrize(
    ("kernel", "family"),
    (
        (USRelationshipSourceKernel, "relationship_inputs"),
        (USEligibilitySourceKernel, "eligibility_inputs"),
    ),
)
def test_live_output_registry_binds_person_preparation(monkeypatch, kernel, family):
    instance = kernel()
    before = instance.implementation_hash()
    monkeypatch.setitem(
        pool.PRE_ASSEMBLY_OPERATOR_OUTPUT_FAMILIES,
        family,
        {"person": frozenset({"is_household_head"})},
    )
    assert instance.implementation_hash() != before


@pytest.mark.parametrize("option", ("POOL_RANDOM_SEED", "POOL_TIME_PERIOD"))
@pytest.mark.parametrize(
    "kernel", (USRelationshipSourceKernel, USEligibilitySourceKernel)
)
def test_executed_options_bind_person_preparation(monkeypatch, option, kernel):
    instance = kernel()
    before = instance.implementation_hash()
    monkeypatch.setattr(pool, option, getattr(pool, option) + 1)
    assert instance.implementation_hash() != before


def test_adapter_never_names_source_role_or_selection_columns():
    """Role stripping and CPS selection stay inside the registered pool owner."""
    text = Path(
        import_module("microcosm.build.us_runtime.graph_person_preparation").__file__
    ).read_text()
    for token in (
        "PERIDNUM",
        "person_support_channel",
        "person_support_clone_index",
        "without_support_role_metadata",
    ):
        assert token not in text


@pytest.mark.parametrize("family", ("relationship", "eligibility"))
def test_absent_cps_evidence_refuses_at_the_boundary(tmp_path, family):
    """An empty source-evidence column refuses exactly as the direct chain does."""
    frame = _upstream(family)
    frame.person["PERIDNUM"] = pd.Series(
        pd.NA, index=frame.person.index, dtype="string"
    )
    with pytest.raises(ValueError) as direct:
        _direct(frame, _family_operator(family))
    args, sources, store, kernels = _setup(tmp_path, frame, families=(family,))
    with pytest.raises(NodeRejectedError) as graph:
        run_graph(args, sources=sources, store=store, kernels=kernels)
    assert str(direct.value) in str(graph.value)


def test_eligibility_context_producer_seam_is_caller_controlled(tmp_path):
    """Housing's slot is a caller-bound seam, not a neighbour this module names."""
    frame = _frame()
    with_seam, sources, store, kernels = _setup(tmp_path, frame)
    without_seam, other_sources, other_store, other_kernels = _setup(
        tmp_path / "direct",
        frame,
        families=("cps_carried", "hours", "prior_year", "relationship", "eligibility"),
    )
    seamed = run_graph(with_seam, sources=sources, store=store, kernels=kernels)
    plain = run_graph(
        without_seam,
        sources=other_sources,
        store=other_store,
        kernels=other_kernels,
    )
    _assert_equal(
        seamed.population(f"{ELIGIBILITY_NODE}.boundary"),
        plain.population(f"{ELIGIBILITY_NODE}.boundary"),
    )
    assert canonical_json(
        seamed.node(ELIGIBILITY_NODE).receipt["source_preparation"]
    ) == canonical_json(plain.node(ELIGIBILITY_NODE).receipt["source_preparation"])
    boundary = next(
        node
        for node in with_seam.graph.nodes
        if node.id == f"{ELIGIBILITY_NODE}.boundary"
    )
    assert boundary.artifact_inputs[0].producer == "early_preparation.housing_seam"
    pending = replace(
        boundary,
        artifact_inputs=(replace(boundary.artifact_inputs[0], producer=HOURS_PHASE),),
    )
    with pytest.raises(ValueError, match="type"):
        compile_graph(
            replace(
                with_seam.graph,
                nodes=tuple(
                    pending if node.id == boundary.id else node
                    for node in with_seam.graph.nodes
                ),
            )
        )


def test_person_preparation_never_constructs_a_qrf():
    """Neither deterministic family carries a quantile-forest path."""
    for module in (relationship, eligibility):
        assert not hasattr(module, "QRF")
    text = Path(
        import_module("microcosm.build.us_runtime.graph_person_preparation").__file__
    ).read_text()
    assert "QRF" not in text


def test_changed_implementation_identity_invalidates_the_cached_node(
    tmp_path, monkeypatch
):
    frame = _upstream("relationship")
    args, sources, store, kernels = _setup(tmp_path, frame, families=("relationship",))
    first = run_graph(
        args, sources=sources, store=store, kernels=kernels, resume="auto"
    )
    replayed = run_graph(
        args, sources=sources, store=store, kernels=kernels, resume="auto"
    )
    assert replayed.node(RELATIONSHIP_NODE).hit
    target = Path(
        import_module("microcosm.build.us_runtime.graph_person_preparation").__file__
    ).resolve()
    original = Path.read_bytes
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: (
            original(path) + b"\n# independent revision\n"
            if path.resolve() == target
            else original(path)
        ),
    )
    revised = run_graph(
        args, sources=sources, store=store, kernels=kernels, resume="auto"
    )
    assert not revised.node(RELATIONSHIP_NODE).hit
    _assert_equal(
        revised.population(f"{RELATIONSHIP_NODE}.boundary"),
        first.population(f"{RELATIONSHIP_NODE}.boundary"),
    )


def test_adapter_carries_no_source_fence_exemption():
    spec = importlib.util.spec_from_file_location(
        "_person_preparation_blindness",
        Path(__file__).with_name("test_us_spine_blindness.py"),
    )
    blindness = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = blindness
    spec.loader.exec_module(blindness)
    assert (
        "graph_person_preparation.py" not in blindness._SOURCE_SPINE_PROVENANCE_OWNERS
    )
    assert "graph_person_preparation.py" in blindness._OTHER_US_RUNTIME_MODULES
    assert "graph_person_preparation.py" in blindness._CLASSIFIED_US_RUNTIME_MODULES
