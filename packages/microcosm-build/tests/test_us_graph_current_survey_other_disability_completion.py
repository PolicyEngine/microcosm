"""Genuine invented source ownership plus a deliberately invented receiver graph.

This proves the bounded fragment, not integration with an issued enriched host.
All source files are created under the test directory before fresh issuance.
No country engine, actual microdata, calibration or scientific acceptance.
"""

import hashlib
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from ss_report_source_fixture import ss_report_source_arguments
from test_us_asec_coverage_authentication import _changed_parent
from test_us_current_survey_other_disability_completion import (
    invented_case,
    two_clone_receiver,
)

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.us_runtime import (
    graph_current_survey_other_disability_completion as graph,
)
from microcosm.graph import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactType,
    Capabilities,
    ContentStore,
    Determinism,
    Graph,
    KernelBase,
    KernelResult,
    Node,
    Numeric,
    Owned,
    SourceRef,
    StructuralDelta,
    compile_graph,
    load_source,
    run_graph,
)

values = graph.values
AFTER_TYPE = ArtifactType("invented.other_disability.receiver", 1)
AFTER_PAYLOAD = b'{"scope":"invented receiver; not an issued host"}'


class RealDiskFixturePatch:
    """Reuse fixture source construction without its unrelated fake disk probe."""

    def __init__(self, patch):
        self.patch = patch

    def setattr(self, obj, name, value, *args, **kwargs):
        if obj is shutil and name == "disk_usage":
            return
        return self.patch.setattr(obj, name, value, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.patch, name)


def disability_arguments(root, monkeypatch):
    patch = RealDiskFixturePatch(monkeypatch)
    arguments = ss_report_source_arguments(root, patch)
    folder = arguments["source_dir"] / "asec"
    parent, attachment = folder / "parent.h5", folder / "household-attachment.h5"
    people = load_frame_checkpoint(parent).frame.person
    literal = {
        105: (1, 6, 200),
        106: (0, 0, 0),
        109: (1, 1, 500),
        110: (2, 0, 0),
        107: (1, 2, 300),
        108: (0, 0, 0),
        111: (1, 0, 0),
        112: (0, 0, 0),
    }
    changed = {}
    for name in values.observed.AMOUNT_FIELDS:
        data = people[name].to_numpy(copy=True)
        for pid, (_, _, amount) in literal.items():
            data[people.person_id.eq(pid)] = amount if name == "DIS_VAL1" else 0
        changed[name] = data
    _changed_parent(parent, attachment, patch, changed)
    people = load_frame_checkpoint(parent).frame.person.set_index("PERIDNUM")
    coverage = values.observed.routing.coverage
    restoration = values.source.asec_native.restoration
    paths, pins = {}, []
    for year, member, archive, *_ in coverage._MEMBER_PINS:
        path = folder / member
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        if year == 2024:
            for name in values.observed.detail.READ_COLUMNS:
                if name not in raw:
                    raw[name] = "0"
            for i, record in raw.iterrows():
                pid = int(people.loc[record.PERIDNUM, "person_id"])
                receipt, code, amount = literal[pid]
                # The source owner normalizes ANN's printed -1 to a retained
                # zero with DECLARED_NIU. Preserve the original parent literal
                # here; inventing a raw zero would contradict that owner.
                for name in values.observed.detail.RETAINED_MONEY_FIELDS:
                    value = people.loc[record.PERIDNUM, name]
                    assert np.isfinite(value) and value == np.floor(value)
                    raw.loc[i, name] = str(int(value))
                for name, value in (
                    ("DIS_YN", receipt),
                    ("DIS_SC1", code),
                    ("DIS_VAL1", amount),
                    ("DIS_SC2", 0),
                    ("DIS_VAL2", 0),
                ):
                    raw.loc[i, name] = str(value)
                for name in (
                    *values.observed.ALLOCATION_FIELDS,
                    *values.observed.TOPCODE_FIELDS,
                ):
                    raw.loc[i, name] = "0"
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
    for owner in (coverage, restoration):
        patch.setattr(owner, "_MEMBER_PINS", tuple(pins))
    restored = root / "other-disability-restored-money"
    restoration.restore_asec_person_income_source(
        parent, attachment, member_paths=paths, output_dir=restored
    )
    shutil.copyfile(
        restored / restoration.CHECKPOINT_FILENAME,
        folder / "person-income-attachment.h5",
    )
    return arguments


class ReceiverKernel(KernelBase):
    ref = "invented.other_disability.receiver@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC,
        structural=StructuralDelta.CREATE,
        numeric=Numeric.PLATFORM_BITWISE,
    )

    def run(self, context):
        return KernelResult(
            frame=load_source("frame-store", context.sources["receiver"]),
            artifacts={"after": AFTER_PAYLOAD},
        )


def after_edge():
    return ArtifactInput("after", "receiver", "after", AFTER_TYPE)


def compiled(boundary):
    frame = boundary.receiving
    implicit = {
        "person_id",
        *(frame.schema.membership_column(e) for e in frame.schema.group_entities),
    }
    receiver = Node(
        "receiver",
        ReceiverKernel.ref,
        sources=("receiver",),
        structural=StructuralDelta.CREATE,
        outputs=tuple(
            Owned(
                "person", c, graph.population_ops.token_for_dtype(frame.person[c].dtype)
            )
            for c in frame.person
            if c not in implicit
        ),
        artifact_outputs=(ArtifactOutput("after", AFTER_TYPE),),
    )
    return compile_graph(
        Graph(
            "us",
            sources=(
                SourceRef("receiver", "frame-store"),
                SourceRef(graph.SOURCE_NAME, "frame-store"),
            ),
            nodes=(receiver, *boundary.nodes),
        )
    )


@pytest.fixture(scope="module")
def genuine_graph(tmp_path_factory):
    root = tmp_path_factory.mktemp("other-disability-fragment")
    with pytest.MonkeyPatch.context() as patch:
        arguments = disability_arguments(root, patch)
        preparation = values.source.prepare_authenticated_survey_population(**arguments)
        qualified = values.qualify_current_survey_other_disability_completion(
            preparation
        )
        receiver = two_clone_receiver(qualified.originals)
        # Deliberately exercise canonical replacement across the version
        # barrier, not only adding an absent column.
        receiver.person[values.observed.OUTPUT] = np.full(receiver.n("person"), 1234.0)
        boundary = graph._CompletionBoundary(
            preparation,
            receiver,
            receiving_version="receiver",
            after=after_edge(),
            after_payload=AFTER_PAYLOAD,
            seed=29,
            n_estimators=2,
        )
        yield root, boundary


def test_full_original_scope_retains_omitted_positive_and_zero_donors(genuine_graph):
    _, boundary = genuine_graph
    q = boundary.qualified
    assert set(q.full_source.person.index) == {105, 106, 107, 108, 109, 110, 111, 112}
    assert 105 not in set(q.selected_source.person.native_person_id)
    assert 109 not in set(q.selected_source.person.native_person_id)
    donors = q.donor_columns.loc[q.donor_columns[values.ELIGIBLE]]
    assert set(donors.index) == {105, 107, 109, 110}
    assert donors.loc[105, values.TARGET] == 200
    assert donors.loc[109, values.TARGET] == donors.loc[110, values.TARGET] == 0
    assert q.source_frame.resolve_weights("person").kind is values.WeightKind.DESIGN
    assert q.evidence["excluded_affirmed_receipt_unknown_amount_rows"] == 1
    assert q.evidence["excluded_affirmed_receipt_unknown_amount_design_mass"] == 100
    assert q.evidence["known_positive_design_mass"] == 2652.12
    selected = values.observed.qualify_current_asec_other_disability(
        boundary.preparation
    )
    pd.testing.assert_frame_equal(
        selected.person, q.selected_source.person, check_exact=True
    )
    assert "projection_scope" not in selected.evidence["retirement_detail_evidence"]
    assert (
        q.full_source.evidence["retirement_detail_evidence"]["projection_scope"]
        == "full_original_current_asec"
    )
    boundary.validate()


def test_actual_tiny_qrf_cold_required_and_immutable_report_artifact(genuine_graph):
    root, boundary = genuine_graph
    boundary.validate()
    q = boundary.qualified
    seal = values.qualified_seal(q)
    before = values.source._frame_identity(boundary.receiving)
    store = ContentStore(root / "fragment-store")
    refs = {
        graph.SOURCE_NAME: store.put_frame(
            graph.codec.sha(q.donor_projection), q.source_frame
        ),
        "receiver": store.put_frame(before, boundary.receiving),
    }
    registry = graph.other_disability_kernel_registry(boundary)
    registry.register(ReceiverKernel())
    populations = {}
    cold = run_graph(
        compiled(boundary),
        sources=refs,
        store=store,
        kernels=registry,
        _population_observer=lambda name, p: populations.setdefault(name, p),
    )
    assert len(cold.nodes) == 9 and not any(n.hit for n in cold.nodes.values())
    final = populations[graph.ATTACH_NODE].frame
    assert final.n("person") == boundary.receiving.n("person")
    for entity in final.entities:
        old = boundary.receiving.table(entity)
        preserved = (
            old.columns.difference([values.observed.OUTPUT])
            if entity == "person"
            else old.columns
        )
        pd.testing.assert_frame_equal(
            final.table(entity).loc[:, preserved],
            old.loc[:, preserved],
            check_exact=True,
        )
    for entity in final.weighted_entities:
        old_weights = boundary.receiving.weights_for(entity)
        assert final.weights_for(entity).kind is old_weights.kind
        np.testing.assert_array_equal(
            final.weights_for(entity).values, old_weights.values
        )
    pd.testing.assert_series_equal(final.strata, boundary.receiving.strata)
    assert final.mass_log == boundary.receiving.mass_log
    assert final.person[values.observed.OUTPUT].dtype == np.dtype("float64")
    acs = final.person[values.provenance.support_channel_column("person")].eq("acs")
    assert final.person.loc[acs, values.CANONICAL_KNOWN].all()
    assert not final.person.loc[
        acs, values.observed.attached_name(values.observed.KNOWN_COLUMN)
    ].any()
    assert final.person.loc[acs, values.VALUE_ORIGIN].eq("modeled").all()
    original_ids = final.person[values.provenance.support_source_id_column("person")]
    source_rows = q.selected_source.person.reindex(original_ids[~acs].to_numpy())
    np.testing.assert_array_equal(
        final.person.loc[~acs, values.observed.OUTPUT].to_numpy(),
        source_rows[values.observed.AMOUNT_COLUMN].to_numpy(
            dtype="float64", na_value=np.nan
        ),
    )
    np.testing.assert_array_equal(
        final.person.loc[~acs, values.CANONICAL_KNOWN].to_numpy(dtype=bool),
        source_rows[values.observed.KNOWN_COLUMN].to_numpy(dtype=bool),
    )
    rewritten = [o.column for n in boundary.nodes for o in n.outputs if o.rewrite]
    assert rewritten == [values.observed.OUTPUT]
    payload = store.load_bytes(
        cold.node(graph.ATTACH_NODE).opaque_artifacts["attachment"]
    )
    report = graph.codec.decode_json(payload)
    assert report["source_reports"] == graph._source_report_storage(
        q.selected_source.person
    )
    assert report["source_reports"]["physical_sha256"] == values._table(
        q.selected_source.person
    )
    assert report["receipt"]["modeled_originals"] == len(q.recipient_features)
    assert report["receipt"]["source_knownness_preserved"] is True
    assert graph.values.observed.OUTPUT in report["frame_projection"]
    assert "DIS_VAL1_published_amount" in report["artifact_only_source_fields"]
    donor = populations[graph.DONOR_NODE].frame
    assert set(donor.person.person_id) == {105, 107, 109, 110}
    assert donor.resolve_weights("person").kind is values.WeightKind.DESIGN
    fit = graph.codec.decode_json(
        store.load_bytes(
            cold.node(graph.FIT_PREFIX + ".000").opaque_artifacts["training_state"]
        )
    )
    assert fit["models"][0]["target"] == values.TARGET
    boundary.validate()
    previous, entered = sys.getprofile(), []

    def no_fit_or_draw(frame, event, arg):
        if previous is not None:
            previous(frame, event, arg)
        if event == "call" and frame.f_code in (
            graph.qrf_target.fit_target.__code__,
            graph.qrf_target.apply_target.__code__,
        ):
            entered.append(True)
            raise AssertionError("required replay entered model fitting or drawing")

    try:
        sys.setprofile(no_fit_or_draw)
        warm = run_graph(
            compiled(boundary),
            sources=refs,
            store=store,
            kernels=registry,
            resume="require",
        )
    finally:
        sys.setprofile(previous)
    assert (
        not entered and all(n.hit for n in warm.nodes.values()) and warm.key == cold.key
    )
    assert (
        store.load_bytes(warm.node(graph.ATTACH_NODE).opaque_artifacts["attachment"])
        == payload
    )
    assert (
        values.qualified_seal(q) == seal
        and values.source._frame_identity(boundary.receiving) == before
    )
    boundary.validate()
    boundary.pure()


def test_artifact_keeps_masked_source_bytes_separate_from_canonical_projection():
    q, receiver = invented_case()
    draws = pd.Series(0.0, index=q.recipient_features.index)
    _, payload = graph.attachment_payload(q, receiver, draws, {})
    before = graph.codec.decode_json(payload)["source_reports"]
    raw = q.selected_source.person[values.observed.AMOUNT_COLUMN].array
    unknown = np.flatnonzero(raw._mask)[0]
    raw._data[unknown] = 987.0
    columns, changed = graph.attachment_payload(q, receiver, draws, {})
    after = graph.codec.decode_json(changed)["source_reports"]
    assert before != after
    assert columns["person", values.observed.OUTPUT].isna().sum() == 6


def test_no_recipient_declaration_has_no_fit_or_application():
    from dataclasses import replace

    q, receiver = invented_case()
    q = replace(q, matrix=None, recipient_features=q.recipient_features.iloc[:0].copy())
    nodes = graph.other_disability_completion_nodes(
        q,
        receiver,
        receiving_version="receiver",
        after=after_edge(),
        after_payload=AFTER_PAYLOAD,
        seed=29,
        n_estimators=2,
    )
    assert len(nodes) == 4
    assert not any("fit" in n.id or "apply" in n.id for n in nodes)
    assert next(n for n in nodes if n.id == graph.VERSION_NODE).base == "receiver"


@pytest.mark.parametrize("field", ["full_original", "non_bool"])
def test_full_original_options_refuse_non_boolean_before_source_access(field):
    with pytest.raises(ValueError, match="FULL_ORIGINAL_OPTION"):
        if field == "full_original":
            values.observed.qualify_current_asec_other_disability(None, full_original=1)
        else:
            values.observed.detail.qualify_current_asec_retirement_detail(
                None, full_original="yes"
            )
