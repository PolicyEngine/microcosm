"""Actual archived-layout rent loading, fitted-model reuse and checked placement."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.us_runtime import _graph_acs_rent_artifacts as codec
from microcosm.build.us_runtime import graph_acs_rent as rent
from microcosm.build.us_runtime import graph_acs_rent_source as source
from microcosm.build.us_runtime import graph_context, housing_inputs
from microcosm.build.us_runtime.graph_sources import frame_column_declarations
from microcosm.fit import graph_legacy_train
from microcosm.frame import US_SCHEMA, Frame, WeightKind, Weights
from microcosm.graph import (
    ArtifactOutput,
    ArtifactValue,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelContext,
    KernelRegistry,
    KernelResult,
    Node,
    Numeric,
    NumericScope,
    SourceRef,
    StructuralDelta,
    compile_graph,
    run_graph,
)
from microcosm.graph.codecs import SourceCodecRegistry


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _donor_file(path, *, mutation=None):
    n = 60
    ids = np.arange(1, n + 1, dtype=np.int64)
    tenure = np.resize(
        np.array(["NONE", "OWNED_WITH_MORTGAGE", "RENTED"], dtype="S24"), n
    )
    arrays = {
        "person_id": ids + 1000,
        "person_household_id": ids,
        "is_household_head": np.ones(n, dtype=bool),
        "age": np.linspace(25.0, 75.0, n),
        "is_male": ids % 2 == 0,
        "employment_income": ids * 1500.0,
        "self_employment_income": ids * 50.0,
        "social_security": ids * 80.0,
        "taxable_private_pension_income": ids * 40.0,
        "rent": np.where(tenure == b"RENTED", ids * 900.0, 0.0),
        "rent_is_allocated": ids % 13 == 0,
        "real_estate_taxes": np.where(tenure == b"RENTED", 0.0, ids * 80.0),
        "real_estate_taxes_is_allocated": ids % 17 == 0,
        "household_id": ids,
        "household_weight": np.linspace(0.0, 2.0, n),
        "state_fips": np.resize([6, 36, 48], n),
        "tenure_type": tenure,
    }
    if mutation == "values":
        arrays["employment_income"][3] += 123.0
    if mutation == "weights":
        arrays["household_weight"][3] += 4.0
    if mutation == "allocation":
        arrays["rent_is_allocated"][3] = True
    if mutation == "categories":
        arrays["state_fips"][3] = 12
    with h5py.File(path, "w") as h5:
        for name, values in arrays.items():
            h5[name] = values
    return path


def _frame(*, n=20, weights=1.0, age_shift=0.0, incumbent=False):
    ids = np.arange(11, 11 + n, dtype=np.int64)
    # Person ordering deliberately differs from household order, with a child
    # between heads. This catches the historical wrong-household scatter.
    membership = np.r_[ids[::-1], ids[-1]]
    person_ids = np.arange(101, 102 + n, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": person_ids,
            "person_household_id": membership,
            "is_household_head": np.r_[np.ones(n, dtype=bool), False],
            "age": np.r_[np.linspace(25.0, 70.0, n) + age_shift, 9.0],
            "is_female": person_ids % 2 == 0,
            "employment_income_before_lsr": person_ids * 250.0,
            "self_employment_income_before_lsr": person_ids * 10.0,
            "social_security_retirement": person_ids * 20.0,
            "taxable_private_pension_income": person_ids * 10.0,
            "SPM_CAPHOUSESUB": np.zeros(n + 1),
            "SPM_TENMORTSTATUS": np.ones(n + 1, dtype=np.int64),
        },
        index=pd.Index(
            np.arange(500, 500 + 3 * (n + 1), 3), name="original_person_row"
        ),
    )
    person["P_SEQ"] = pd.array(
        [1 if i % 2 else None for i in range(n + 1)], dtype="Int64"
    )
    if incumbent:
        person.insert(2, "pre_subsidy_rent", np.nan)
    tables = {
        "person": person,
        "household": pd.DataFrame(
            {
                "household_id": ids,
                "state_fips": np.resize([6, 36], n),
                "H_TENURE": np.resize([1, 2, 3], n),
            }
        ),
    }
    for i, group in enumerate(US_SCHEMA.group_entities):
        if group == "household":
            continue
        person[f"person_{group}_id"] = membership + (i + 1) * 1000
        tables[group] = pd.DataFrame({f"{group}_id": ids + (i + 1) * 1000})
    frame = Frame(
        tables,
        US_SCHEMA,
        {
            "household": Weights(
                np.linspace(1.0, 2.0, n) * weights, WeightKind.IMPORTANCE
            )
        },
    )
    # Actual predecessor helper; the rent adapter does not invent these values.
    frame = housing_inputs.derive_us_housing_inputs(frame)
    tables = {entity: frame.table(entity).copy() for entity in frame.entities}
    for table in tables.values():
        for name in table:
            if pd.api.types.is_object_dtype(
                table[name]
            ) or pd.api.types.is_string_dtype(table[name].dtype):
                table[name] = table[name].astype(pd.StringDtype(storage="python"))
    return Frame(
        tables,
        frame.schema,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        mass_log=frame.mass_log,
    )


def _load_recipient(path, *, store=None):
    del store
    return load_frame_checkpoint(path).frame


class _Checkpoint(KernelBase):
    ref = "test.actual_rent_prior_stage_checkpoint@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.CREATE,
        numeric=Numeric.PLATFORM_BITWISE,
    )

    def run(self, context):
        frame = _load_recipient(context.sources["recipient"])
        return KernelResult(
            frame=frame,
            artifacts={"frame_context": graph_context.encode_us_frame_context(frame)},
        )


def _run(root, *, resume="forbid", recipient=None, donor=None):
    root = Path(root)
    recipient = root / "recipient.h5" if recipient is None else Path(recipient)
    donor = root / "donor.h5" if donor is None else Path(donor)
    frame = _load_recipient(recipient)
    descriptor = source.prepare_acs_rent_source_descriptor(
        donor, expected_sha256=_sha(donor)
    )
    codecs = SourceCodecRegistry()
    codecs.register("test-prior-stage-h5", _load_recipient)
    codecs.register(
        source.RENT_SOURCE_CODEC,
        partial(source.load_graph_acs_rent_source, expected_sha256=_sha(donor)),
    )
    columns = frame_column_declarations(frame)
    nodes = (
        Node(
            "recipient",
            _Checkpoint.ref,
            sources=("recipient",),
            structural=StructuralDelta.CREATE,
            outputs=columns,
            artifact_outputs=(
                ArtifactOutput("frame_context", graph_context.US_FRAME_CONTEXT_TYPE),
            ),
        ),
        *rent.us_acs_rent_nodes(
            columns,
            base="recipient",
            context_producer="recipient",
            donor_descriptor=descriptor,
            seed=31,
            n_estimators=2,
        ),
    )
    graph = compile_graph(
        Graph(
            country="us",
            sources=(SourceRef("recipient", "test-prior-stage-h5"), source.RENT_SOURCE),
            nodes=nodes,
        )
    )
    kernels = KernelRegistry()
    kernels.register(_Checkpoint())
    rent.register_us_acs_rent_kernels(kernels, source_codecs=codecs)
    store = ContentStore(root / "store", codecs=codecs)
    manifest = run_graph(
        graph,
        sources={"recipient": recipient, source.RENT_SOURCE_NAME: donor},
        store=store,
        kernels=kernels,
        resume=resume,
    )
    return manifest, store, frame, nodes


def _inputs(tmp_path, **kwargs):
    _donor_file(tmp_path / "donor.h5")
    write_frame_checkpoint(tmp_path / "recipient.h5", _frame(**kwargs))


def _payload(manifest, store, node, name):
    return store.load_bytes(manifest.nodes[node].opaque_artifacts[name])


@pytest.mark.parametrize("incumbent", [False, True])
def test_real_helper_graph_parity_nullable_h5_and_cold_require(tmp_path, incumbent):
    _inputs(tmp_path, incumbent=incumbent)
    cold, store, frame, nodes = _run(tmp_path)
    actual = cold.population(rent.RENT_OUTPUT_POPULATION)
    donor = housing_inputs.load_acs_2022_rent_donor(
        tmp_path / "donor.h5", expected_sha256=_sha(tmp_path / "donor.h5")
    )
    expected = housing_inputs.impute_us_pre_subsidy_rent(
        frame, donor, seed=31, n_estimators=2
    )
    np.testing.assert_array_equal(actual.person.pre_subsidy_rent, expected)
    assert len(cold.nodes) == 8
    assert all(node.params.get("phase") == codec.RENT_PHASE for node in nodes[1:])
    document = json.loads(
        _payload(cold, store, rent.RENT_OUTPUT_CONTEXT, "frame_context")
    )
    for entity in frame.entities:
        expected_table = frame.table(entity).copy()
        if entity == "person":
            expected_table["pre_subsidy_rent"] = expected
        pd.testing.assert_frame_equal(actual.table(entity), expected_table)
        np.testing.assert_array_equal(
            actual.resolve_weights(entity).values, frame.resolve_weights(entity).values
        )
        assert document["entities"][entity]["columns"] == list(expected_table.columns)
    assert cold.nodes[f"{rent.RENT_PREFIX}.validation"].receipt["outcome"] == "pass"
    assert (
        actual.person.P_SEQ.dtype == pd.Int64Dtype()
        and actual.person.P_SEQ.isna().any()
    )
    # Real nullable Frame checkpoint, preserving original table indices and dtypes.
    output = Frame(
        {e: actual.table(e) for e in actual.entities},
        US_SCHEMA,
        {e: frame.weights_for(e) for e in frame.weighted_entities},
        mass_log=graph_context._mass_records(document["mass_log"]),
    )
    write_frame_checkpoint(tmp_path / "output.h5", output)
    pd.testing.assert_frame_equal(
        load_frame_checkpoint(tmp_path / "output.h5").frame.person, output.person
    )
    warm, _, _, _ = _run(tmp_path, resume="require")
    assert warm.key == cold.key and all(record.hit for record in warm.nodes.values())


def test_require_replays_in_fresh_interpreter_without_fit_or_source_execution(tmp_path):
    _inputs(tmp_path)
    cold, _, _, _ = _run(tmp_path)
    code = """
import importlib.util,sys
spec=importlib.util.spec_from_file_location(sys.argv[3],sys.argv[1])
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def forbidden(*args,**kwargs):raise AssertionError('replay executed kernel')
m._Checkpoint.run=forbidden
m.graph_legacy_train.LegacyQRFTrainKernel.run=forbidden
m.rent.USACSRentRecipientKernel.run=forbidden
result,*_=m._run(sys.argv[2],resume='require')
assert all(node.hit for node in result.nodes.values())
print(result.key)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, __file__, str(tmp_path), __name__],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == cold.key


@pytest.mark.parametrize(
    "change", ["membership", "weights", "predictors", "source_pin"]
)
def test_recipient_changes_reuse_actual_fitted_forest(tmp_path, monkeypatch, change):
    _inputs(tmp_path)
    calls = []
    original = graph_legacy_train.fit_target

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(graph_legacy_train, "fit_target", counted)
    first, store, _, _ = _run(tmp_path)
    changed = _frame(
        n=18 if change == "membership" else 20,
        weights=4.0 if change == "weights" else 1.0,
        age_shift=3.0 if change == "predictors" else 0.0,
    )
    write_frame_checkpoint(
        tmp_path / "changed.h5", changed, metadata={"fixture_pin": change}
    )
    second, _, _, _ = _run(tmp_path, recipient=tmp_path / "changed.h5", resume="auto")
    fit = f"{rent.RENT_PREFIX}.fit.000"
    assert calls == [1]
    assert first.nodes[fit].key == second.nodes[fit].key and second.nodes[fit].hit
    assert (
        first.nodes[fit].opaque_artifacts["model"]
        == second.nodes[fit].opaque_artifacts["model"]
    )
    assert not second.nodes[f"{rent.RENT_PREFIX}.apply.000"].hit
    assert (
        first.nodes[f"{rent.RENT_PREFIX}.recipient"].key
        != second.nodes[f"{rent.RENT_PREFIX}.recipient"].key
    )


@pytest.mark.parametrize("change", ["values", "weights", "allocation", "categories"])
def test_actual_donor_changes_invalidate_fit(tmp_path, change):
    _inputs(tmp_path)
    first, _, _, _ = _run(tmp_path)
    donor = _donor_file(tmp_path / "changed-donor.h5", mutation=change)
    second, _, _, _ = _run(tmp_path, donor=donor, resume="auto")
    fit = f"{rent.RENT_PREFIX}.fit.000"
    assert first.nodes[fit].key != second.nodes[fit].key and not second.nodes[fit].hit


def test_registered_production_source_rejects_synthetic_unregistered_archive(tmp_path):
    path = _donor_file(tmp_path / "donor.h5")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        source.load_graph_acs_rent_source(path)


def _validation_context(manifest, store, frame, nodes, *, final=False):
    node = nodes[-1] if final else nodes[-2]
    values = {}
    from microcosm.graph import platform_fingerprint

    for edge in node.artifact_inputs:
        producer = manifest.nodes[edge.producer]
        key = producer.opaque_artifacts[edge.artifact]
        values[edge.name] = ArtifactValue(
            store.load_bytes(key),
            edge.type,
            key,
            producer.key,
            NumericScope(Numeric.PLATFORM_BITWISE, platform=platform_fingerprint()),
        )
    tables = {}
    for view in node.inputs:
        structural = [f"{view.entity}_id"]
        if view.entity == "person":
            structural.extend(
                US_SCHEMA.membership_column(group) for group in US_SCHEMA.group_entities
            )
        tables[view.entity] = (
            frame.table(view.entity).loc[:, [*structural, *view.columns]].copy()
        )
    return KernelContext(
        node=node,
        tables=tables,
        weights={e: frame.resolve_weights(e) for e in tables},
        strata=frame.strata,
        params=node.params,
        rng=np.random.default_rng(0),
        artifacts=values,
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "matrix_digest",
        "matrix_producer",
        "raw_digest",
        "raw_sibling",
        "encoding_producer",
        "person_index",
        "head_membership",
        "head_flag",
        "training_config",
        "draw_seed",
    ],
)
def test_gate_reports_precise_stale_binding_or_association_failure(tmp_path, mutation):
    _inputs(tmp_path)
    manifest, store, frame, nodes = _run(tmp_path)
    context = _validation_context(manifest, store, frame, nodes)
    values = dict(context.artifacts)
    if mutation in {
        "matrix_digest",
        "matrix_producer",
        "raw_digest",
        "training_config",
        "draw_seed",
    }:
        state = json.loads(values["apply_state"].payload)
        if mutation == "matrix_digest":
            state["matrix_sha256"] = "f" * 64
        elif mutation == "matrix_producer":
            state["matrix_producer_key"] = "f" * 64
        elif mutation == "raw_digest":
            state["application"]["raw_targets"][0]["sha256"] = "f" * 64
        elif mutation == "draw_seed":
            state["application"]["seed"] += 1
        else:
            state["application"]["state"]["model_config"]["n_estimators"] += 1
        values["apply_state"] = replace(
            values["apply_state"], payload=codec.encode_json(state)
        )
    elif mutation in {"raw_sibling", "encoding_producer"}:
        name = "raw_draw" if mutation == "raw_sibling" else "encoding"
        values[name] = replace(values[name], producer_key="f" * 64)
    elif mutation == "person_index":
        context.tables["person"].index = context.tables["person"].index + 1
    elif mutation == "head_membership":
        index = context.tables["person"].index
        first = context.tables["person"].loc[index[0], "person_household_id"]
        second = context.tables["person"].loc[index[1], "person_household_id"]
        context.tables["person"].loc[index[0], "person_household_id"] = second
        context.tables["person"].loc[index[1], "person_household_id"] = first
    else:
        context.tables["person"].iloc[
            0, context.tables["person"].columns.get_loc("is_household_head")
        ] = False
    with pytest.raises(ValueError, match="us_acs_rent_assignment_integrity"):
        rent.USACSRentValidationKernel().run(replace(context, artifacts=values))
    # A failed gate packet cannot be used to produce a rent column.
    final = _validation_context(manifest, store, frame, nodes, final=True)
    final_values = dict(final.artifacts)
    final_values["validation"] = replace(
        final_values["validation"],
        payload=codec.encode_json(
            {
                "schema_version": 1,
                "passed": False,
                "failure": "test refusal",
                "bindings": {},
            }
        ),
    )
    with pytest.raises(ValueError, match="passed validation"):
        rent.USACSRentFinalizeKernel().run(replace(final, artifacts=final_values))


def test_stale_preflight_descriptor_refused_after_actual_source_reconstruction(
    tmp_path,
):
    path = _donor_file(tmp_path / "donor.h5")
    descriptor = source.prepare_acs_rent_source_descriptor(
        path, expected_sha256=_sha(path)
    )
    changed = _donor_file(tmp_path / "changed.h5", mutation="values")
    codecs = SourceCodecRegistry()
    codecs.register(
        source.RENT_SOURCE_CODEC,
        partial(source.load_graph_acs_rent_source, expected_sha256=_sha(changed)),
    )
    node = rent.us_acs_rent_nodes(
        frame_column_declarations(_frame()),
        base="source",
        context_producer="source",
        donor_descriptor=descriptor,
        seed=31,
        n_estimators=2,
    )[0]
    context = KernelContext(
        node=node,
        params=node.params,
        tables={},
        weights={},
        strata=pd.Series([], dtype=str),
        rng=np.random.default_rng(0),
        sources={source.RENT_SOURCE_NAME: changed},
    )
    with pytest.raises(ValueError, match="authenticated preflight"):
        source.USACSRentDonorKernel(codecs).run(context)


@pytest.mark.parametrize("group", ["household", "tax_unit"])
def test_all_person_boundary_refuses_stale_group_count_before_filter(tmp_path, group):
    from microcosm.graph import platform_fingerprint

    _inputs(tmp_path)
    frame = _frame()
    descriptor = source.prepare_acs_rent_source_descriptor(
        tmp_path / "donor.h5", expected_sha256=_sha(tmp_path / "donor.h5")
    )
    node = rent.us_acs_rent_nodes(
        frame_column_declarations(frame),
        base="recipient",
        context_producer="recipient",
        donor_descriptor=descriptor,
        seed=31,
        n_estimators=2,
    )[2]
    document = json.loads(graph_context.encode_us_frame_context(frame))
    document["entities"][group]["rows"] += 1
    value = ArtifactValue(
        codec.encode_json(document),
        graph_context.US_FRAME_CONTEXT_TYPE,
        "a" * 64,
        "b" * 64,
        NumericScope(Numeric.PLATFORM_BITWISE, platform=platform_fingerprint()),
    )
    context = KernelContext(
        node=node,
        params=node.params,
        tables={"person": frame.person},
        weights={"person": frame.resolve_weights("person")},
        strata=frame.strata,
        rng=np.random.default_rng(0),
        artifacts={"frame_context": value},
    )
    with pytest.raises(ValueError, match=f"unreferenced {group}"):
        rent.USACSRentBoundaryKernel().run(context)


def test_named_gate_failure_preserves_manifest_and_blocks_finalize_on_replay(
    tmp_path, monkeypatch
):
    _inputs(tmp_path)
    original = rent.USACSRentRecipientKernel.run

    def damaged(self, context):
        result = original(self, context)
        payload = result.artifacts["placement"]
        offset = len(codec.PLACEMENT_MAGIC) + 4
        length = int.from_bytes(payload[offset - 4 : offset], "big")
        header = json.loads(payload[offset : offset + length])
        header["matrix_sha256"] = "f" * 64
        encoded = codec.encode_json(header)
        artifacts = dict(result.artifacts)
        artifacts["placement"] = (
            codec.PLACEMENT_MAGIC
            + len(encoded).to_bytes(4, "big")
            + encoded
            + payload[offset + length :]
        )
        return replace(result, artifacts=artifacts)

    def forbidden(*args, **kwargs):
        raise AssertionError("failed validation reached finalizer")

    monkeypatch.setattr(rent.USACSRentRecipientKernel, "run", damaged)
    monkeypatch.setattr(rent.USACSRentFinalizeKernel, "run", forbidden)
    cold, _, _, _ = _run(tmp_path)
    gate = cold.nodes[f"{rent.RENT_PREFIX}.validation"]
    assert gate.receipt["outcome"] == "fail"
    assert "us_acs_rent_assignment_integrity" in gate.receipt["evidence"]["message"]
    assert not gate.opaque_artifacts
    final = cold.nodes[rent.RENT_OUTPUT_CONTEXT]
    assert (
        final.receipt["outcome"] == "unreached"
        and not final.artifacts
        and not final.opaque_artifacts
    )
    assert "pre_subsidy_rent" not in cold.population(rent.RENT_OUTPUT_POPULATION).person
    warm, _, _, _ = _run(tmp_path, resume="require")
    assert warm.key == cold.key and all(record.hit for record in warm.nodes.values())


@pytest.mark.parametrize(
    "target",
    [
        "recipient",
        "donor_helper",
        "training_contract",
        "gate_contract",
        "unrelated_assembly",
    ],
)
def test_explicit_source_scopes_preserve_donor_recipient_separation(
    tmp_path, monkeypatch, target
):
    from microcosm.build.us_runtime import graph_sources

    donor_kernel = source.USACSRentDonorKernel()
    recipient_kernel = rent.USACSRentRecipientKernel()
    before = (
        donor_kernel.implementation_hash(),
        recipient_kernel.implementation_hash(),
    )
    module = {
        "recipient": rent,
        "donor_helper": housing_inputs,
        "training_contract": rent.qrf_target,
        "gate_contract": sys.modules[rent.GateReport.__module__],
        "unrelated_assembly": graph_sources,
    }[target]
    path = tmp_path / "mutated.py"
    path.write_bytes(
        Path(module.__file__).read_bytes() + b"\n# source scope mutation\n"
    )
    monkeypatch.setattr(module, "__file__", str(path))
    after = (donor_kernel.implementation_hash(), recipient_kernel.implementation_hash())
    assert (before[0] != after[0]) == (target == "donor_helper")
    assert (before[1] != after[1]) == (target != "unrelated_assembly")


_RUNTIME_KERNELS = {
    "donor": source.USACSRentDonorKernel,
    "boundary": rent.USACSRentBoundaryKernel,
    "recipient": rent.USACSRentRecipientKernel,
    "validation": rent.USACSRentValidationKernel,
    "finalizer": rent.USACSRentFinalizeKernel,
}


@pytest.mark.parametrize("name", _RUNTIME_KERNELS)
def test_rent_arrow_version_is_declared_and_rekeys_implementation(monkeypatch, name):
    from microcosm.graph import kernel as graph_kernel

    kernel = _RUNTIME_KERNELS[name]()
    before = kernel.implementation_hash()
    original = graph_kernel.importlib_metadata.version
    monkeypatch.setattr(
        graph_kernel.importlib_metadata,
        "version",
        lambda package: (
            "test-arrow-version-change" if package == "pyarrow" else original(package)
        ),
    )
    assert "pyarrow" in kernel.capabilities.dependencies
    assert kernel.implementation_hash() != before


@pytest.mark.parametrize("name", _RUNTIME_KERNELS)
@pytest.mark.parametrize(
    "option,first,second",
    [
        ("mode.string_storage", "python", "pyarrow"),
        ("future.infer_string", False, True),
    ],
)
def test_rent_string_option_rekeys_implementation(name, option, first, second):
    kernel = _RUNTIME_KERNELS[name]()
    with pd.option_context(option, first):
        before = kernel.implementation_hash()
    with pd.option_context(option, second):
        after = kernel.implementation_hash()
    assert before != after


def test_rent_runtime_identity_records_actual_resolved_backend():
    for storage in ("python", "pyarrow", "auto"):
        with pd.option_context(
            "mode.string_storage", storage, "future.infer_string", True
        ):
            runtime = codec.string_runtime_identity()
            sample = pd.Series(np.array(["rent"], dtype=object)).astype(str)
            assert runtime["options"] == {
                "mode.string_storage": storage,
                "future.infer_string": True,
            }
            assert runtime["resolved"]["astype_str"]["storage"] == sample.dtype.storage
            assert (
                runtime["resolved"]["astype_str"]["array_class"]
                == f"{type(sample.array).__module__}.{type(sample.array).__qualname__}"
            )


@pytest.mark.parametrize("change", ["pyarrow", "string_storage", "infer_string"])
def test_changed_string_runtime_refuses_old_required_cache_and_rebuilds(
    tmp_path, monkeypatch, change
):
    from microcosm.graph import StoreMiss
    from microcosm.graph import kernel as graph_kernel

    with pd.option_context(
        "mode.string_storage", "python", "future.infer_string", True
    ):
        _inputs(tmp_path)
        first, _, _, _ = _run(tmp_path)
        for node in (f"{rent.RENT_PREFIX}.donor", f"{rent.RENT_PREFIX}.recipient"):
            assert (
                first.nodes[node].receipt["string_runtime"]
                == codec.string_runtime_identity()
            )
        original = graph_kernel.importlib_metadata.version
        with monkeypatch.context() as patch:
            if change == "pyarrow":
                patch.setattr(
                    graph_kernel.importlib_metadata,
                    "version",
                    lambda name: (
                        "test-arrow-version-change"
                        if name == "pyarrow"
                        else original(name)
                    ),
                )
                option, value = "mode.string_storage", "python"
            elif change == "string_storage":
                option, value = "mode.string_storage", "pyarrow"
            else:
                option, value = "future.infer_string", False
            with pd.option_context(option, value):
                with pytest.raises(StoreMiss):
                    _run(tmp_path, resume="require")
                rebuilt, _, _, _ = _run(tmp_path, resume="auto")
                for node in (
                    f"{rent.RENT_PREFIX}.donor",
                    f"{rent.RENT_PREFIX}.recipient",
                ):
                    assert (
                        rebuilt.nodes[node].receipt["string_runtime"]
                        == codec.string_runtime_identity()
                    )
                for node in (
                    f"{rent.RENT_PREFIX}.donor",
                    f"{rent.RENT_PREFIX}.fit.000",
                    f"{rent.RENT_PREFIX}.recipient",
                    rent.RENT_OUTPUT_CONTEXT,
                ):
                    assert first.nodes[node].key != rebuilt.nodes[node].key
                    assert not rebuilt.nodes[node].hit
                warm, _, _, _ = _run(tmp_path, resume="require")
                assert warm.key == rebuilt.key and all(
                    record.hit for record in warm.nodes.values()
                )


@pytest.mark.parametrize("target", ["donor", "recipient"])
def test_rent_refuses_string_option_change_during_actual_encoding(
    tmp_path, monkeypatch, target
):
    from microcosm.graph import NodeRejectedError

    helper_name = f"prepare_acs_rent_{target}"
    original = getattr(housing_inputs, helper_name)

    def changed(*args, **kwargs):
        prepared = original(*args, **kwargs)
        pd.set_option("mode.string_storage", "pyarrow")
        return prepared

    with pd.option_context(
        "mode.string_storage", "python", "future.infer_string", True
    ):
        _inputs(tmp_path)
        monkeypatch.setattr(housing_inputs, helper_name, changed)
        error = ValueError if target == "donor" else NodeRejectedError
        with pytest.raises(error, match="string runtime changed during encoding"):
            _run(tmp_path)
