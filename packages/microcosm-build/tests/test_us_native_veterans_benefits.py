"""Invented veterans receipt/provenance and genuine source/graph boundaries."""

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_asec_veterans_source as veterans
from microcosm.build.us_runtime import current_survey_amounts as amounts


def test_published_receipt_domain_and_six_digit_amount():
    basis = veterans.reporting_basis(
        np.array([999999, 0, 0, 0, 12, 0, 12, np.nan], dtype="float64"),
        np.array([15, 99, 35, 35, 35, 14, 14, 35], dtype="float64"),
        ["1", "2", "1", "0", "2", "0", "1", "1"],
    )
    np.testing.assert_array_equal(
        basis.canonical_amount,
        [999999, 0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
    )
    assert basis.reporting_status.tolist() == [
        "known_receipt",
        "known_nonreceipt",
        "ambiguous_recipient_zero",
        "niu",
        "contradictory_no_positive",
        "outside_reporting_universe",
        "contradictory_outside_reporting_universe",
        "missing_amount",
    ]


@pytest.mark.parametrize("value", [-1, 1000000, np.inf, 0.5])
def test_bad_veterans_amount_refuses(value):
    with pytest.raises(ValueError):
        veterans.reporting_basis(
            np.array([value], dtype="float64"), np.array([35.0]), ["1"]
        )


def test_veterans_requires_full_original_and_is_not_default():
    with pytest.raises(ValueError, match="VETERANS_FULL_ORIGINAL_REQUIRED"):
        amounts.qualify_current_survey_amounts(object(), groups=("veterans_benefits",))
    assert amounts.selected_groups(("veterans_benefits",))[0].fields == (
        ("VET_VAL", "veterans_benefits"),
    )


def test_flag_zero_only_means_unallocated_within_its_conditional_universe():
    basis = veterans.reporting_basis(
        np.array([120000.0, 0.0, 0.0, 50.0, np.nan]),
        np.array([40.0] * 5),
        ["1", "2", "0", "1", ""],
    )
    raw = pd.DataFrame(
        {
            "VET_VAL": ["120000", "0", "0", "50", ""],
            "I_VETYN": ["0", "9", "0", "bad", ""],
            "I_VETVAL": ["11", "0", "0", "", "0"],
        }
    )
    out = veterans.allocation_basis(basis, raw)
    assert out.I_VETYN_allocation_status.tolist()[:4] == [
        "not_allocated_in_flag_universe",
        "publisher_allocated",
        "outside_flag_universe",
        "unresolved_allocation_literal",
    ]
    assert out.I_VETVAL_allocation_status.tolist()[:4] == [
        "publisher_allocated",
        "outside_flag_universe",
        "outside_flag_universe",
        "allocation_flag_not_populated",
    ]
    assert pd.isna(out.I_VETYN_flag_universe.iloc[4])
    assert pd.isna(out.I_VETVAL_flag_universe.iloc[4])
    assert out.I_VETVAL_literal.tolist() == raw.I_VETVAL.tolist()
    assert out.canonical_amount.iloc[3] == 50  # Flags describe allocation, not receipt.


def _veterans_arguments(root, patch):
    import hashlib
    import shutil

    from test_us_asec_coverage_authentication import _changed_parent
    from test_us_native_workers_compensation import _wc_source_arguments

    from microcosm.build.frame_checkpoint import load_frame_checkpoint
    from microcosm.build.us_runtime import asec_person_income_source as restoration
    from microcosm.build.us_runtime import current_asec_demographics as demographics

    arguments = _wc_source_arguments(root, patch)
    source = arguments["source_dir"] / "asec"
    parent, attachment = source / "parent.h5", source / "household-attachment.h5"
    person = load_frame_checkpoint(parent).frame.person
    values = person.VET_VAL.to_numpy(dtype="float64", copy=True)
    for pid, value in zip(
        (105, 106, 107, 108), (120000.0, 0.0, 0.0, 500.0), strict=True
    ):
        values[person.person_id.eq(pid)] = value
    _changed_parent(parent, attachment, patch, {"VET_VAL": values})
    updated = load_frame_checkpoint(parent).frame.person.set_index("PERIDNUM")
    paths, pins = {}, []
    for year, member, archive, *_ in veterans.receipt.coverage._MEMBER_PINS:
        path = source / member
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        raw["VET_VAL"] = [str(int(updated.loc[key, "VET_VAL"])) for key in raw.PERIDNUM]
        raw["VET_YN"] = [
            "0" if int(age) < 15 else "1" if int(value) > 0 else "2"
            for age, value in zip(raw.A_AGE, raw.VET_VAL, strict=True)
        ]
        raw["I_VETYN"], raw["I_VETVAL"] = "0", "0"
        if year == 2024:
            raw.loc[raw.PERIDNUM.eq(str(5).zfill(22)), "I_VETVAL"] = "11"
            raw.loc[raw.PERIDNUM.eq(str(7).zfill(22)), "VET_YN"] = "1"
        raw.iloc[::-1].to_csv(path, index=False)
        payload = path.read_bytes()
        pins.append(
            (
                year,
                member,
                archive,
                hashlib.sha256(payload).hexdigest(),
                len(raw),
                len(payload),
            )
        )
        paths[year] = path
    for module in (veterans.receipt.coverage, restoration, demographics.demographic):
        patch.setattr(module, "_MEMBER_PINS", tuple(pins))
    output = root / "veterans-restored-money"
    restoration.restore_asec_person_income_source(
        parent, attachment, member_paths=paths, output_dir=output
    )
    shutil.copyfile(
        output / restoration.CHECKPOINT_FILENAME, source / "person-income-attachment.h5"
    )
    return arguments


def test_genuine_source_full_donors_and_flag_byte_mutation(tmp_path, monkeypatch):
    import json
    import shutil
    from fractions import Fraction

    arguments = _veterans_arguments(tmp_path, monkeypatch)
    partial = tmp_path / "partial"
    shutil.copytree(arguments["source_dir"], partial)
    request = json.loads((partial / "selection-request.json").read_bytes())
    request["fraction"], request["seed"] = [2, 3], 41
    (partial / "selection-request.json").write_text(
        json.dumps(request, sort_keys=True, separators=(",", ":"))
    )
    snapshots = tmp_path / "partial-captures"
    snapshots.mkdir()
    prepared = veterans.receipt.source.prepare_authenticated_survey_population(
        source_dir=partial, snapshot_root=snapshots, fraction=Fraction(2, 3), seed=41
    )
    observed = veterans.qualify_current_asec_veterans(prepared)
    assert 105 not in observed.person.native_person_id.values
    ids = observed.person.index.append(pd.Index([-999], name="person_id"))
    selected = np.array([True] * len(observed.person) + [False])
    reports = amounts._receipt_reports(observed, ids, selected, "survey_veterans_")
    assert reports.loc[-999].isna().all()
    assert str(reports.survey_veterans_I_VETYN_flag_universe.dtype) == "boolean"
    assert str(reports.survey_veterans_I_VETVAL_code.dtype) == "Int64"
    assert str(reports.survey_veterans_I_VETVAL_literal.dtype) == "string"
    assert (
        reports.loc[ids[:-1], "survey_veterans_I_VETVAL_literal"].tolist()
        == observed.person.I_VETVAL_literal.tolist()
    )
    full, evidence = veterans.receipt._qualify_receipt_amount(
        prepared, family="veterans_benefits", full_original=True
    )
    np.testing.assert_array_equal(full.canonical_amount, [120000, 0, np.nan, np.nan])
    assert full.loc[105, "I_VETVAL_literal"] == "11"
    assert full.loc[105, "I_VETVAL_allocation_status"] == "publisher_allocated"
    assert full.loc[106, "I_VETVAL_allocation_status"] == "outside_flag_universe"
    assert full.loc[107, "reporting_status"] == "ambiguous_recipient_zero"
    assert evidence["read_columns"] == list(veterans.READ_COLUMNS)
    donor = amounts.full_donor.qualify_full_original_amount_donor(
        prepared, amounts.selected_groups(("veterans_benefits",))
    )
    np.testing.assert_array_equal(donor.amounts.VET_VAL, [120000, 0, np.nan, np.nan])
    assert donor.frame.weights_for("household").values.tolist() == [2552.12, 100]
    _run_veterans_graph(
        tmp_path / "graph", monkeypatch, donor, carried=-999.0, combined=True
    )
    source = partial / "asec" / "pppub25.csv"
    payload = source.read_bytes()
    # Same-size flag-only byte change still revokes genuine source qualification.
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    raw.loc[raw.PERIDNUM.eq(str(5).zfill(22)), "I_VETVAL"] = "12"
    raw.to_csv(source, index=False)
    assert len(source.read_bytes()) == len(payload)
    with pytest.raises(ValueError):
        veterans.qualify_current_asec_veterans(prepared)


def _veterans_family(combined=False):
    from dataclasses import replace

    from test_us_native_child_support import _child_family

    from microcosm.build.us_runtime import graph_us_survey_enrichment as graph

    qualified, receiving = _child_family()
    child = qualified.groups[0]
    spec = amounts.selected_groups(("veterans_benefits",))[0]
    group = replace(
        child,
        spec=spec,
        donor_columns=child.donor_columns.rename(
            columns={child.spec.targets[0]: spec.targets[0]}
        ),
    )
    native = qualified.native.copy()
    native["veterans_benefits"] = [np.nan, np.nan, 120000.0, 0.0]
    reports = qualified.reports.copy()
    reports["survey_current_VET_VAL_origin"] = pd.Series(
        [
            "unresolved",
            "modeled_from_current_asec",
            "known_receipt",
            "known_nonreceipt",
        ],
        index=native.index,
        dtype="string",
    )
    if not combined:
        native = native[["veterans_benefits"]]
        reports = reports[["survey_current_VET_VAL_origin"]]
    qualified = replace(
        qualified,
        native=native,
        reports=reports,
        groups=(child, group) if combined else (group,),
    )
    assert graph._canonical_ids(qualified) == (
        graph.CANONICAL_VERSION_NODE,
        graph.CANONICAL_ATTACH_NODE,
    )
    return qualified, receiving


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("carried", [120000.0, -999.0])
def test_only_declared_veterans_rewrite_is_allowed(dtype, carried):
    from microcosm.fit import model_input

    qualified, receiving = _veterans_family()
    receiving.person["veterans_benefits"] = np.full(
        len(receiving.person), carried, dtype=dtype
    )
    before = receiving.person.copy(deep=True)
    group = qualified.groups[0]
    draws = {
        group.spec.key: pd.DataFrame(
            {group.spec.targets[0]: [0.0]},
            index=model_input.decode_recipient_matrix(group.matrix).features.index,
        )
    }
    columns = amounts.attach_columns(qualified, receiving, draws)
    np.testing.assert_array_equal(
        columns["person", "veterans_benefits"],
        [np.nan, 0, 120000, 0, np.nan, 0, 120000, 0],
    )
    assert str(columns["person", "veterans_benefits"].dtype) == dtype
    pd.testing.assert_frame_equal(receiving.person, before, check_exact=True)
    receiving.person["survey_current_VET_VAL_origin"] = "wrong"
    with pytest.raises(ValueError, match="ATTACH_OWNERSHIP_COLLISION"):
        amounts.attach_columns(qualified, receiving, draws)


@pytest.mark.parametrize("dtype", ["int64", "Int64", "Float64", "string", "bool"])
def test_unsupported_veterans_incumbent_storage_refuses(dtype):
    qualified, receiving = _veterans_family()
    receiving.person["veterans_benefits"] = pd.Series(
        "1" if dtype == "string" else 1, index=receiving.person.index, dtype=dtype
    )
    with pytest.raises(ValueError, match="REWRITE_DTYPE"):
        amounts.attachment_dtype(qualified, receiving, "veterans_benefits", "float64")


def test_uc_wc_range_unchanged_and_named_fields_not_trailing_flags(tmp_path):
    from microcosm.build.us_runtime import (
        current_asec_workers_compensation_source as wc,
    )

    for module in (wc, veterans.receipt):
        with pytest.raises(ValueError, match="AMOUNT_OR_TOKEN_DOMAIN"):
            module.reporting_basis(np.array([100000.0]), np.array([40.0]), ["1"])
    path = tmp_path / "invented.csv"
    path.write_text(
        "PERIDNUM,PH_SEQ,A_LINENO,A_AGE,VET_VAL,VET_YN,I_VETYN,I_VETVAL\n0000000000000000000001,1,1,40,120000,1,9,15\n"
    )
    raw = veterans.read_capture(path, rows=1)
    assert raw.VET_VAL.iloc[0] == "120000" and raw.I_VETVAL.iloc[0] == "15"
    with pytest.raises(ValueError, match="RECEIPT_FIELD_PAIR"):
        veterans.receipt._read_capture(
            path, rows=1, amount_field="UC_VAL", receipt_field="VET_YN"
        )


def _run_veterans_graph(tmp_path, monkeypatch, projection, carried, *, combined=False):
    from dataclasses import replace
    from types import SimpleNamespace

    from microcosm.build.us_runtime import graph_us_survey_enrichment as graph
    from microcosm.fit import model_input
    from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
    from microcosm.fit.graph_legacy_train import LegacyQRFTrainKernel
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
        Owned,
        SourceRef,
        StructuralDelta,
        compile_graph,
        run_graph,
    )
    from microcosm.graph import population as population_ops

    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    tmp_path.mkdir()
    canonical_state = combined
    qualified, receiving = _veterans_family(combined)
    spec = qualified.groups[-1].spec
    if canonical_state:
        # Invented receiving geography only. Genuine geography qualification is
        # separately covered by test_us_current_survey_state_host; this fixture
        # proves terminal composition, never a native PUF/enrichment issuer.
        household = receiving.table("household")
        for name in graph.state_graph.INPUTS:
            household[name] = pd.array(
                ["06"] * len(household), dtype=population_ops.dtype_for_token("string")
            )
        household["state_fips"] = np.full(len(household), 99, dtype="int64")
    full = projection.frame
    keep = np.isfinite(projection.features.to_numpy()).all(axis=1) & np.isfinite(
        projection.amounts.VET_VAL.to_numpy()
    )
    columns = projection.features.loc[keep].copy()
    columns[spec.targets[0]] = projection.amounts.loc[keep, "VET_VAL"]
    recipient = columns.loc[:, list(projection.features)].iloc[:1].copy()
    recipient.index = pd.Index([1], name="person_id")
    matrix = model_input.encode_recipient_matrix(
        recipient, entity="person", entity_ids=np.array([1], dtype="<i8")
    )
    qualified = replace(qualified, features=tuple(projection.features))
    if carried is not None:
        for name in amounts.canonical_outputs(qualified):
            receiving.person[name] = np.full(
                len(receiving.person), carried, dtype="float64"
            )
    source_before = qualified.source_frame.person.copy(deep=True)
    omitted = 105
    group = amounts.GroupValues(spec, full.select(keep), columns, matrix, keep)
    groups = [group]
    if combined:
        child_spec = qualified.groups[0].spec
        child_columns = columns.rename(columns={spec.targets[0]: child_spec.targets[0]})
        groups.insert(
            0,
            amounts.GroupValues(
                child_spec, full.select(keep), child_columns, matrix, keep
            ),
        )
    qualified = replace(qualified, full_donor_source_frame=full, groups=tuple(groups))
    assert omitted not in qualified.source_frame.person.person_id.values
    assert group.donor_columns.loc[omitted, spec.targets[0]] > 0
    before = {e: receiving.table(e).copy(deep=True) for e in receiving.entities}
    weight = receiving.weights_for("household")
    weight_bytes, weight_kind = weight.values.tobytes(), weight.kind
    nodes = graph.amount_nodes(
        qualified, receiving, parent_digest="a" * 64, n_estimators=2
    )
    assert sum(n.id == graph.FULL_DONOR_SOURCE_NODE for n in nodes) == 1
    donor_id = graph._ids(group)[0]
    assert (
        next(n for n in nodes if n.id == donor_id).base == graph.FULL_DONOR_SOURCE_NODE
    )

    class Receiving(KernelBase):
        ref = "test.full-amount.receiving@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
        )

        def implementation_hash(self):
            return "b" * 64

        def run(self, context):
            return KernelResult(frame=receiving)

    class Finalization(KernelBase):
        ref = "test.full-amount.finalization@1"
        capabilities = Capabilities(Determinism.DETERMINISTIC)

        def implementation_hash(self):
            return "c" * 64

        def run(self, context):
            return KernelResult(
                artifacts={"finalization": b"invented; no native parent authority"}
            )

    class ReceivingFilter(KernelBase):
        ref = "test.child.receiving_filter@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, structural=StructuralDelta.FILTER
        )

        def implementation_hash(self):
            return "d" * 64

        def run(self, context):
            return KernelResult(
                keep=pd.Series(
                    True,
                    index=pd.Index(receiving.person.person_id, name="person_id"),
                    dtype=bool,
                )
            )

    create = Node(
        "test.child.source",
        Receiving.ref,
        sources=(graph.health_graph.SOURCE_NAME,),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned(
                selection.entity,
                c,
                population_ops.token_for_dtype(
                    receiving.table(selection.entity)[c].dtype
                ),
            )
            for selection in graph.predictor_graph._inputs(receiving)
            for c in selection.columns
        ),
    )
    receiving_node = Node(
        graph.parent.attach.FILTER_NODE,
        ReceivingFilter.ref,
        base=create.id,
        structural=StructuralDelta.FILTER,
        mass="free",
        inputs=graph.predictor_graph._inputs(receiving),
    )
    finalization = Node(
        graph.parent.attach.ATTACH_NODE,
        Finalization.ref,
        population=receiving_node.id,
        artifact_outputs=(
            ArtifactOutput("finalization", graph.parent.attach.FINALIZATION_TYPE),
        ),
    )

    class Later(KernelBase):
        ref = "test.child.later@1"
        capabilities = Capabilities(
            Determinism.DETERMINISTIC, numeric=Numeric.PLATFORM_BITWISE
        )

        def implementation_hash(self):
            return "e" * 64

        def run(self, context):
            return KernelResult(
                columns={
                    ("person", "test_late"): pd.Series(
                        receiving.person.person_id.to_numpy() * 2,
                        index=pd.Index(receiving.person.person_id, name="person_id"),
                        dtype="int64",
                    )
                }
            )

    later = Node(
        "test.child.later",
        Later.ref,
        population=receiving_node.id,
        outputs=(Owned("person", "test_late", "int64"),),
        artifact_inputs=(
            ArtifactInput(
                "amounts", graph.ATTACH_NODE, "attachment", graph.ATTACHMENT_TYPE
            ),
        ),
    )
    extra = ()
    if canonical_state:

        class GeographyFixture(KernelBase):
            ref = "test.child.state_geography@1"
            capabilities = Capabilities(Determinism.DETERMINISTIC)

            def run(self, context):
                return KernelResult(
                    artifacts={
                        "validation": graph.state_graph.canonical_json(
                            {
                                "outcome": "pass",
                                "scope": "atomic_geography_mapping_integrity",
                            }
                        )
                    }
                )

        gate = Node(
            "geography.gate",
            GeographyFixture.ref,
            population=receiving_node.id,
            artifact_outputs=(
                ArtifactOutput(
                    "validation", graph.state_graph.ATOMIC_GEOGRAPHY_VALIDATION_TYPE
                ),
            ),
        )
        state_nodes = graph.Boundary._state_nodes(
            SimpleNamespace(
                canonical_state_input=True,
                qualified=qualified,
                run=SimpleNamespace(population=SimpleNamespace(frame=receiving)),
            )
        )
        assert state_nodes[0].base == graph.CANONICAL_VERSION_NODE
        assert state_nodes[0].artifact_inputs[0].producer == graph.CANONICAL_ATTACH_NODE
        nodes = (*nodes, *state_nodes)
        extra = (gate,)
    compiled = compile_graph(
        Graph(
            "us",
            (SourceRef(graph.health_graph.SOURCE_NAME, "raw-bytes-v1"),),
            (create, receiving_node, finalization, *nodes, later, *extra),
        )
    )
    original_seal = amounts.seal(qualified)

    def pure():
        assert amounts.seal(qualified) == original_seal

    def context(value):
        pure()
        assert value.node in nodes
        return qualified

    boundary = SimpleNamespace(
        qualified=qualified,
        context=context,
        pure=pure,
        run=SimpleNamespace(population=SimpleNamespace(frame=receiving)),
        parent_view=SimpleNamespace(digest="a" * 64),
        n_estimators=2,
        compiled=compiled,
    )
    registry = KernelRegistry()
    for kernel in (
        Receiving(),
        ReceivingFilter(),
        Finalization(),
        Later(),
        graph.CurrentSurveyAmountProjectionKernel(boundary),
        graph.CurrentSurveyFullAmountSourceKernel(boundary),
        graph.CurrentSurveyAmountDonorKernel(boundary),
        graph.CurrentSurveyAmountColumnsKernel(boundary),
        graph.CurrentSurveyAmountAttachKernel(boundary),
        graph.CurrentSurveyChildVersionKernel(boundary),
        graph.CurrentSurveyChildAttachKernel(boundary),
        LegacyQRFTrainKernel(),
        LegacyQRFApplyMatrixKernel(),
    ):
        registry.register(kernel)
    if canonical_state:
        registry.register(GeographyFixture())
        registry.register(graph.CurrentSurveyStateVersionKernel(boundary))
        registry.register(graph.CurrentSurveyCanonicalStateKernel(boundary))
    path = tmp_path / "invented.txt"
    path.write_bytes(b"invented full-original amount graph")
    store = ContentStore(tmp_path / "store")
    kwargs = dict(
        sources={graph.health_graph.SOURCE_NAME: path}, store=store, kernels=registry
    )
    cold = run_graph(compiled, **kwargs)
    observed = {}
    warm = run_graph(
        compiled,
        **kwargs,
        resume="require",
        _population_observer=lambda node, population: observed.__setitem__(
            node, population
        ),
    )
    assert cold.key == warm.key and all(n.hit for n in warm.nodes.values())
    fitted = warm.population(donor_id)
    assert fitted.person.person_id.tolist() == columns.index.tolist()
    assert fitted.person[spec.targets[0]].tolist() == columns[spec.targets[0]].tolist()
    assert omitted in fitted.person.person_id.values
    assert fitted.person.person_id.tolist() == [105, 106]
    assert fitted.person[spec.targets[0]].tolist() == [120000, 0]
    assert fitted.weights_for("household").values.tolist() == [2552.12]
    assert not any("CHSP" in node.id for node in compiled.graph.nodes)
    assert sum(
        node.kernel == LegacyQRFTrainKernel.ref for node in compiled.graph.nodes
    ) == len(groups)
    output = warm.population(graph.CANONICAL_VERSION_NODE)
    np.testing.assert_array_equal(output.person.test_late, output.person.person_id * 2)
    assert graph.CANONICAL_ATTACH_NODE in cold.nodes
    assert compiled.versions[graph.PROJECTION_NODE] == receiving_node.id
    assert (
        compiled.versions[graph.CANONICAL_ATTACH_NODE] == graph.CANONICAL_VERSION_NODE
    )
    base_population = population_ops.Population.from_frame(
        warm.population(receiving_node.id),
        receiving_node.id,
        mass_ledger=warm.mass_ledger(receiving_node.id),
    )
    reconstructed = graph._child_version_population(
        base_population, compiled.graph.node(graph.CANONICAL_VERSION_NODE)
    )
    for entity in output.entities:
        pd.testing.assert_frame_equal(
            reconstructed.frame.table(entity),
            warm.population(receiving_node.id).table(entity),
            check_exact=True,
        )
    assert reconstructed.mass_ledger[:-1] == base_population.mass_ledger
    assert reconstructed.mass_ledger[-1].policy == "conserve"
    for e in receiving.entities:
        unrelated = [
            c
            for c in before[e]
            if e != "person" or c not in amounts.canonical_outputs(qualified)
        ]
        pd.testing.assert_frame_equal(
            output.table(e)[unrelated], before[e][unrelated], check_exact=True
        )
    assert output.weights_for("household").kind is weight_kind
    assert output.weights_for("household").values.tobytes() == weight_bytes
    # Native observed cells survive; only the eligible ACS original gets a draw.
    expected = output.person.set_index("person_id").veterans_benefits
    assert expected.loc[2] == expected.loc[12] == 120000
    assert expected.loc[3] == expected.loc[13] == 0
    assert np.isnan(expected.loc[0]) and np.isnan(expected.loc[10])
    assert expected.loc[1] == expected.loc[11]
    assert expected.loc[1] in (0.0, 120000.0)
    if combined:
        paid = output.person.set_index("person_id").child_support_expense
        assert paid.loc[2] == paid.loc[12] == 2000
        assert paid.drop([2, 12]).isna().all()
        child_received = output.person.set_index("person_id").child_support_received
        assert child_received.loc[2] == child_received.loc[12] == 120
        assert child_received.loc[3] == child_received.loc[13] == 0
        assert child_received.loc[1] == child_received.loc[11]
        assert child_received.loc[1] in (0, 120000)
        assert child_received.loc[[0, 10]].isna().all()
    pd.testing.assert_frame_equal(
        qualified.source_frame.person, source_before, check_exact=True
    )
    for e in receiving.entities:
        pd.testing.assert_frame_equal(receiving.table(e), before[e], check_exact=True)
    loaded = {
        (node.id, name): store.load_bytes(key)
        for node in compiled.graph.nodes
        for name, key in warm.node(node.id).opaque_artifacts.items()
    }
    version_node = compiled.graph.node(graph.CANONICAL_VERSION_NODE)
    replayed_version = graph._child_version_population(observed[later.id], version_node)
    graph.physical.replay.same_replayed_population(
        replayed_version, observed[graph.CANONICAL_VERSION_NODE]
    )
    child_node = compiled.graph.node(graph.CANONICAL_ATTACH_NODE)
    child_result = graph._attachment_result(
        boundary,
        graph.parent._loaded_values(boundary, warm, loaded, child_node),
        child_only=True,
    )
    assert (
        loaded[graph.CANONICAL_ATTACH_NODE, "attachment"]
        == child_result.artifacts["attachment"]
    )
    replayed_child = population_ops.patch(replayed_version, child_node, child_result)
    graph.physical.replay.same_replayed_population(
        replayed_child, observed[graph.CANONICAL_ATTACH_NODE]
    )
    if canonical_state:
        prior = observed[graph.CANONICAL_ATTACH_NODE]
        for node in state_nodes:
            reconstructed = graph._state_expected_population(
                prior,
                node,
                graph.parent._loaded_values(boundary, warm, loaded, node),
                {a.name: loaded[node.id, a.name] for a in node.artifact_outputs},
            )
            graph.physical.replay.same_replayed_population(
                reconstructed, observed[node.id]
            )
            prior = reconstructed
        final = warm.population(graph.STATE_VERSION_NODE)
        for entity in output.entities:
            other = [
                c
                for c in output.table(entity)
                if (entity, c) != ("household", "state_fips")
            ]
            pd.testing.assert_frame_equal(
                final.table(entity)[other],
                output.table(entity)[other],
                check_exact=True,
            )
        assert final.table("household").state_fips.eq(6).all()
        assert output.table("household").state_fips.eq(99).all()
        assert final.person.test_late.equals(output.person.test_late)
        assert warm.mass_ledger(graph.STATE_VERSION_NODE)[:-1] == warm.mass_ledger(
            graph.CANONICAL_VERSION_NODE
        )
        assert final.weights_for("household").values.tobytes() == weight_bytes
    graph._verify_models(
        boundary,
        loaded,
        {
            graph._ids(g)[0]: population_ops.Population.from_frame(
                warm.population(graph._ids(g)[0]), graph._ids(g)[0]
            )
            for g in groups
        },
    )
    assert model_input.decode_recipient_matrix(
        group.matrix
    ).features.index.tolist() == [1]
    pure()


@pytest.mark.parametrize(
    "name",
    [
        "current_asec_veterans_source.py",
        "current_asec_unemployment_source.py",
        "current_asec_amount_donor.py",
        "current_survey_amounts.py",
        "graph_us_survey_enrichment.py",
    ],
)
def test_changed_modules_keep_source_spine_contract(name):
    from test_us_spine_blindness import _US_RUNTIME, _non_owner_source_spine_accesses

    assert not _non_owner_source_spine_accesses(name, (_US_RUNTIME / name).read_text())


def test_live_flag_map_and_callable_changes_are_detected(monkeypatch):
    from microcosm.build.us_runtime import graph_us_survey_enrichment as graph

    before = graph._live()

    class Changed(dict):
        def __getitem__(self, key):
            return (999,)

    for name in ("ALLOCATION_CODES", "DICTIONARY"):
        with monkeypatch.context() as patch:
            patch.setattr(veterans, name, Changed(getattr(veterans, name)))
            assert graph._live() != before
    with monkeypatch.context() as patch:
        patch.setattr(veterans.reporting_basis, "__code__", (lambda: None).__code__)
        assert graph._live() != before


def test_lossy_rewrite_and_original_clone_mutation_refuse():
    from microcosm.fit import model_input

    qualified, receiving = _veterans_family()
    group = qualified.groups[0]
    draws = {
        group.spec.key: pd.DataFrame(
            {group.spec.targets[0]: [0.1]},
            index=model_input.decode_recipient_matrix(group.matrix).features.index,
        )
    }
    receiving.person["veterans_benefits"] = np.zeros(
        len(receiving.person), dtype="float32"
    )
    with pytest.raises(ValueError, match="REWRITE_LOSS"):
        amounts.attach_columns(qualified, receiving, draws)
    draws[group.spec.key].iloc[0, 0] = 120000.0
    clone = amounts.provenance.support_clone_index_column("person")
    receiving.person.loc[0, clone] = 7
    with pytest.raises(ValueError, match="CLONE_PAIR"):
        amounts.attach_columns(qualified, receiving, draws)


@pytest.mark.parametrize("enabled", [False, True])
def test_state_after_veterans_with_other_independent_options(enabled):
    from types import SimpleNamespace

    from microcosm.build.us_runtime import graph_us_survey_enrichment as graph

    qualified, receiving = _veterans_family(combined=enabled)
    for name in graph.state_graph.INPUTS:
        receiving.table("household")[name] = pd.array(
            ["06"] * receiving.n("household"),
            dtype=graph.population_ops.dtype_for_token("string"),
        )
    boundary = SimpleNamespace(
        canonical_state_input=True,
        qualified=qualified,
        run=SimpleNamespace(population=SimpleNamespace(frame=receiving)),
        spm=object() if enabled else None,
        immigration_transfer=object() if enabled else None,
        sex=object() if enabled else None,
        race=object() if enabled else None,
        health_completion=object() if enabled else None,
        full_original_amount_donors=True,
    )
    state = graph.Boundary._state_nodes(boundary)
    assert state[0].base == graph.CANONICAL_VERSION_NODE
    assert state[0].artifact_inputs[0].producer == graph.CANONICAL_ATTACH_NODE
    nodes = graph.amount_nodes(
        qualified, receiving, parent_digest="a" * 64, n_estimators=2
    )
    assert graph.CHILD_ATTACH_NODE not in {n.id for n in nodes}
    assert tuple(o.column for o in nodes[-1].outputs) == amounts.canonical_outputs(
        qualified
    )
    # Earlier source/model reads remain on the original receiving version.
    assert nodes[0].population == graph.parent.attach.FILTER_NODE


def test_live_routing_callable_and_active_module_binding_are_sealed(monkeypatch):
    from types import ModuleType

    from microcosm.build.us_runtime import graph_us_survey_enrichment as graph

    before = graph._live()
    with monkeypatch.context() as patch:
        patch.setattr(
            veterans.routing._codes_frame, "__code__", (lambda: None).__code__
        )
        assert graph._live() != before
    replacement = ModuleType("invented_veterans_routing")
    replacement._codes_frame = lambda *args: None
    with monkeypatch.context() as patch:
        patch.setattr(veterans, "routing", replacement)
        assert graph._live() != before
