"""Canonical state binding over genuine invented post-clone geography.

Fixture legacy writers deliberately supply unqualified carried state values.
They exercise replacement only; the actual survey issuers and atomic geography
operators establish the original observed/derived inputs. No release is tested.
"""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_current_asec_demographics import _demographic_arguments
from test_us_graph_atomic_survey_population import _support_payload

from microcosm.build.us_runtime import current_survey_geography as observed
from microcosm.build.us_runtime import graph_atomic_survey_population as parent
from microcosm.build.us_runtime import graph_current_survey_state as binding
from microcosm.calibrate.geography_constants import US_STATE_FIPS_TO_POSTAL
from microcosm.frame import Frame
from microcosm.graph import (
    ArtifactInput,
    ArtifactType,
    ArtifactValue,
    Capabilities,
    Determinism,
    KernelBase,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    NumericScope,
    Owned,
    Slice,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.canonical import canonical_json
from microcosm.graph.population import dtype_for_token


def _table(code="06", *, existing=None, dtype="int64"):
    table = pd.DataFrame(
        {
            "household_id": np.array([10, 20], dtype="int64"),
            "assigned_state_fips": pd.array(
                [code, code], dtype=dtype_for_token("string")
            ),
            "survey_observed_state": pd.array(
                [code, code], dtype=dtype_for_token("string")
            ),
        }
    )
    if existing is not None:
        table["state_fips"] = pd.array(existing, dtype=dtype)
    return table


def _view(table):
    return SimpleNamespace(table=lambda entity: table)


def _artifacts(node):
    gate = canonical_json(
        {"outcome": "pass", "scope": "atomic_geography_mapping_integrity"}
    )
    return {
        edge.name: ArtifactValue(
            type=edge.type,
            payload=gate if edge.name == "geography_validation" else b"host",
            key="a" * 64,
            producer_key="b" * 64,
            numerics=NumericScope(Numeric.BITWISE),
        )
        for edge in node.artifact_inputs
    }


@pytest.mark.parametrize("code", tuple(US_STATE_FIPS_TO_POSTAL))
def test_exact_closed_conversion_preserves_axis_and_inputs(code):
    table = _table(code)
    before = table.copy(deep=True)
    result = binding.bind_state_fips(table)
    assert result.tolist() == [int(code), int(code)]
    assert result.dtype == np.dtype("int64")
    assert result.index.tolist() == [10, 20]
    assert result.index.name == "household_id"
    pd.testing.assert_frame_equal(table, before, check_exact=True)


@pytest.mark.parametrize(
    "token", [None, "", "6", "006", "6.0", " 06", "06 ", "00", "03", "72", "99", "０６"]
)
def test_invalid_or_unknown_states_never_default(token):
    with pytest.raises(ValueError, match="STATE_DOMAIN"):
        binding.bind_state_fips(_table(token))


@pytest.mark.parametrize(
    "dtype,value", [("int64", 6), ("float64", 6.0), ("bool", True), ("object", "06")]
)
def test_noncanonical_assigned_storage_refuses(dtype, value):
    table = _table()
    table["assigned_state_fips"] = pd.Series([value, value], dtype=dtype)
    with pytest.raises(ValueError, match="STATE_DOMAIN"):
        binding.bind_state_fips(table)


@pytest.mark.parametrize(
    "dtype,existing", [("int64", [6, 6]), ("int64", [99, 99]), ("Int64", [None, 99])]
)
def test_carried_canonical_values_are_replaced_not_qualified(dtype, existing):
    table = _table(existing=existing, dtype=dtype)
    before = table.copy(deep=True)
    result = binding.bind_state_fips(table)
    assert result.tolist() == [6, 6]
    assert str(result.dtype) == dtype
    node = binding.state_binding_node(_view(table), population="existing-clones")
    assert node.outputs == (Owned("household", "state_fips", dtype, rewrite=True),)
    pd.testing.assert_frame_equal(table, before, check_exact=True)


@pytest.mark.parametrize("dtype", ["float64", "string", "bool"])
def test_unsupported_incumbent_dtype_refuses(dtype):
    table = _table(existing=[6, 6], dtype=dtype)
    with pytest.raises(ValueError, match="INCUMBENT_DTYPE"):
        binding.bind_state_fips(table)


def test_observed_assigned_mismatch_and_axis_mutation_refuse():
    table = _table()
    table.loc[1, "survey_observed_state"] = "36"
    with pytest.raises(ValueError, match="OBSERVED_STATE_MISMATCH"):
        binding.bind_state_fips(table)
    table = _table()
    table.loc[1, "household_id"] = 10
    with pytest.raises(ValueError, match="HOUSEHOLD_AXIS"):
        binding.bind_state_fips(table)


@pytest.mark.parametrize(
    "change",
    [
        "gate_fail",
        "gate_type",
        "gate_scope",
        "extra_artifact",
        "output",
        "missing_edge",
    ],
)
def test_declared_output_and_typed_gate_mutations_refuse(change):
    table = _table()
    node = binding.state_binding_node(_view(table), population="existing-clones")
    artifacts = _artifacts(node)
    if change in ("gate_fail", "gate_scope"):
        document = json.loads(artifacts["geography_validation"].payload)
        document["outcome" if change == "gate_fail" else "scope"] = "wrong"
        artifacts["geography_validation"] = replace(
            artifacts["geography_validation"], payload=canonical_json(document)
        )
    elif change == "gate_type":
        artifacts["geography_validation"] = replace(
            artifacts["geography_validation"], type=ArtifactType("wrong", 1)
        )
    elif change == "extra_artifact":
        artifacts["extra"] = artifacts["geography_validation"]
    elif change == "output":
        node = replace(node, outputs=(Owned("household", "county_fips", "int64"),))
    else:
        node = replace(node, artifact_inputs=())
    with pytest.raises(ValueError):
        binding.state_binding_result(node, table, artifacts)


def test_optional_host_edge_is_visible_and_no_authority_is_issued():
    table = _table()
    edge = ArtifactInput(
        "previous", "host.attach", "attachment", ArtifactType("host", 1)
    )
    node = binding.state_binding_node(
        _view(table), population="existing-clones", after=edge
    )
    result = binding.state_binding_result(node, table, _artifacts(node))
    assert node.structural is StructuralDelta.NONE and node.weights is None
    assert node.inputs == (Slice("household", binding.INPUTS),)
    assert node.artifact_inputs[1] == edge
    assert result.frame is result.keep is result.weights is None
    assert set(result.columns) == {("household", "state_fips")}
    assert result.receipt["new_geography_assignment"] is False
    assert result.receipt["source_admission_issued"] is False
    assert result.receipt["release_eligible"] is False


@pytest.fixture(scope="module")
def genuine_atomic(tmp_path_factory):
    root = tmp_path_factory.mktemp("canonical-state-invented")
    with pytest.MonkeyPatch.context() as patch:
        arguments = _demographic_arguments(root, patch, unknown=False)
        payload, identities = _support_payload()
        support = root / "invented-support.npz"
        support.write_bytes(payload)
        config = parent.reconstruction.AtomicSurveyReconstruction(
            support_path=str(support),
            support_sha256=hashlib.sha256(payload).hexdigest(),
            source_ids=tuple(sorted(identities.items())),
            seed=17,
        )
        run = parent.run_atomic_survey_population(
            **arguments,
            store_root=root / "store",
            geography_config=config,
            return_values=True,
        )
        yield SimpleNamespace(run=run, arguments=arguments)
        parent.survey._source_owner().verify_survey_population_preparation(
            run.preparation
        )


class _FixtureLegacyState(KernelBase):
    ref = "fixture.legacy_state@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, numeric=Numeric.PLATFORM_BITWISE
    )

    def run(self, context):
        table = context.tables["household"]
        values = binding.bind_state_fips(table)
        if context.params["kind"] != "same":
            values[:] = 99
        values = values.astype(context.node.outputs[0].dtype)
        if context.params["kind"] == "nullable":
            values.iloc[0] = pd.NA
        return KernelResult(columns={("household", "state_fips"): values})


class _FixtureKeepAll(KernelBase):
    ref = "fixture.state_keep_all@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.FILTER
    )

    def run(self, context):
        return KernelResult(
            keep=pd.Series(True, index=context.tables["person"].person_id, dtype="bool")
        )


def _with_households(frame, table):
    return Frame(
        {
            entity: table if entity == "household" else frame.table(entity)
            for entity in frame.entities
        },
        frame.schema,
        dict(frame._weights),
        frame.strata,
        metadata=frame.metadata,
        mass_log=frame.mass_log,
    )


@pytest.mark.parametrize("incumbent", [None, "same", "different"])
def test_genuine_atomic_clone_binding_and_required_replay(genuine_atomic, incumbent):
    run = genuine_atomic.run
    before = run.geography_population
    original_stamp = parent.reconstruction._population_stamp(before)
    observed_before = observed.qualify_current_survey_geography(run.preparation)
    registry = KernelRegistry()
    for kernel in run.kernels.as_mapping().values():
        registry.register(kernel)
    kernel = binding.CurrentSurveyStateKernel()
    registry.register(kernel)
    receiving = before.frame
    population = before.version
    added = ()
    if incumbent is not None:
        table = receiving.table("household").copy(deep=True)
        dtype = "Int64" if incumbent == "nullable" else "int64"
        table["state_fips"] = pd.array([99] * len(table), dtype=dtype)
        receiving = _with_households(receiving, table)
        legacy = Node(
            "fixture.legacy_state",
            _FixtureLegacyState.ref,
            population=population,
            inputs=(Slice("household", binding.INPUTS),),
            outputs=(
                Owned(
                    "household",
                    "state_fips",
                    dtype,
                    rewrite="state_fips" in before.frame.table("household"),
                ),
            ),
            params={"kind": incumbent},
        )
        carrier = Node(
            "fixture.state_carrier",
            _FixtureKeepAll.ref,
            structural=StructuralDelta.FILTER,
            base=population,
            inputs=(Slice("person", ("person_support_clone_index",)),),
        )
        added = (legacy, carrier)
        population = carrier.id
        registry.register(_FixtureLegacyState())
        registry.register(_FixtureKeepAll())
    node = binding.state_binding_node(receiving, population=population)
    compiled = compile_graph(
        replace(run.compiled.graph, nodes=(*run.compiled.graph.nodes, *added, node))
    )
    snapshots = {}
    manifest = run_graph(
        compiled,
        sources=run.sources,
        store=run.store,
        kernels=registry,
        _population_observer=lambda name, value: snapshots.setdefault(name, value),
    )
    base = snapshots["geography.gate" if incumbent is None else population]
    completed = manifest.population(population)
    expected = binding.bind_state_fips(base.frame.table("household"))
    actual = completed.table("household").set_index("household_id").state_fips
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    for entity in completed.entities:
        columns = [
            name
            for name in base.frame.table(entity)
            if not (entity == "household" and name == "state_fips")
        ]
        pd.testing.assert_frame_equal(
            completed.table(entity)[columns],
            base.frame.table(entity)[columns],
            check_exact=True,
        )
    np.testing.assert_array_equal(
        completed.weights_for("household").values,
        base.frame.weights_for("household").values,
    )
    assert (
        completed.weights_for("household").kind
        == base.frame.weights_for("household").kind
    )
    assert manifest.mass_ledger(population) == base.mass_ledger
    assert len(completed.table("household")) == 2 * len(
        run.allocated_population.frame.table("household")
    )
    assert "geography.gate" in compiled.predecessors[node.id]
    assert manifest.node(node.id).receipt["release_eligible"] is False
    warm = run_graph(
        compiled,
        sources=run.sources,
        store=run.store,
        kernels=registry,
        resume="require",
    )
    assert all(record.hit for record in warm.nodes.values())
    assert warm.key == manifest.key
    parent.survey._same_frame(completed, warm.population(population))
    fresh = observed.qualify_current_survey_geography(run.preparation)
    pd.testing.assert_frame_equal(
        fresh.household, observed_before.household, check_exact=True
    )
    assert fresh.receipt == observed_before.receipt
    assert parent.reconstruction._population_stamp(before) == original_stamp


def test_source_member_mutation_remains_a_host_refusal(tmp_path, monkeypatch):
    """Pure binding never replaces the host's live source requalification."""
    arguments = _demographic_arguments(tmp_path, monkeypatch, unknown=False)
    preparation = parent.survey._source_owner().prepare_authenticated_survey_population(
        **arguments
    )
    path = arguments["source_dir"] / "asec" / "hhpub25.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        observed.qualify_current_survey_geography(preparation)
