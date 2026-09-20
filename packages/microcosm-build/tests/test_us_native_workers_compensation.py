"""Invented WC receipt universes and the maintained amount graph; no engine."""

import hashlib
import shutil
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.us_runtime import current_asec_workers_compensation_source as wc
from microcosm.build.us_runtime import current_survey_amounts as amounts
from microcosm.build.us_runtime import graph_us_survey_enrichment as enrichment
from microcosm.fit import _graph_legacy_qrf as codec
from microcosm.fit import model_input
from microcosm.fit.graph_legacy_apply_matrix import LegacyQRFApplyMatrixKernel
from microcosm.fit.graph_legacy_qrf import LegacyQRFTrainKernel
from microcosm.graph import (
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
    load_source,
    run_graph,
)


class InventedAmountSource(KernelBase):
    ref = "test.wc.source@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, structural=StructuralDelta.CREATE
    )

    def run(self, context):
        return KernelResult(
            frame=load_source("frame-store", context.sources[context.node.sources[0]])
        )


class InventedAmountMatrix(KernelBase):
    ref = "test.wc.matrix@1"
    capabilities = Capabilities(
        Determinism.DETERMINISTIC, numeric=Numeric.PLATFORM_BITWISE
    )

    def run(self, context):
        return KernelResult(
            artifacts={
                "workers_compensation_matrix": bytes.fromhex(
                    context.params["matrix_hex"]
                )
            }
        )


def _invented_wc_family():
    from test_us_current_survey_amounts import _invented_family

    qualified, receiving, _ = _invented_family()
    # These are detached invented component values, never an issued native run.
    qualified.source_frame.person.loc[2, "age"] = 40
    receiving.person.loc[receiving.person.person_id.isin((2, 12)), "age"] = 40
    spec = amounts.selected_groups(("workers_compensation",))[0]
    native = qualified.native.rename(
        columns={"unemployment_compensation": "workers_compensation"}
    )
    native.loc[2, "workers_compensation"] = 120.0
    reports = qualified.reports.rename(
        columns={"survey_current_UC_VAL_origin": "survey_current_WC_VAL_origin"}
    )
    reports.loc[2, "survey_current_WC_VAL_origin"] = "known_receipt"
    keep = np.array([False, False, True, True])
    donor = qualified.source_frame.select(keep)
    features = pd.DataFrame(
        {qualified.features[0]: [40.0, 40.0], spec.targets[0]: [120.0, 0.0]},
        index=pd.Index([2, 3], name="person_id"),
    )
    group = amounts.GroupValues(spec, donor, features, qualified.groups[0].matrix, keep)
    return replace(
        qualified, native=native, reports=reports, groups=(group,)
    ), receiving


def test_wc_declared_real_fit_draw_and_clone_attachment(tmp_path, monkeypatch):
    monkeypatch.setenv("POPULACE_FIT_N_JOBS", "1")
    monkeypatch.setenv("POPULACE_FIT_PREDICT_WORKERS", "1")
    qualified, receiving = _invented_wc_family()
    group = qualified.groups[0]
    before = {
        entity: receiving.table(entity).copy(deep=True) for entity in receiving.entities
    }
    before_weights = receiving.resolve_weights("household")
    weight_bytes, weight_kind = before_weights.values.tobytes(), before_weights.kind
    nodes = enrichment.amount_nodes(
        qualified, receiving, parent_digest="a" * 64, n_estimators=2
    )
    fits = tuple(node for node in nodes if node.kernel == LegacyQRFTrainKernel.ref)
    applies = tuple(
        node for node in nodes if node.kernel == LegacyQRFApplyMatrixKernel.ref
    )
    assert len(fits) == len(applies) == 1
    assert any(
        edge.producer == enrichment.PROJECTION_NODE
        for edge in applies[0].artifact_inputs
    )
    assert any(edge.producer == fits[0].id for edge in applies[0].artifact_inputs)
    assert any(edge.producer == applies[0].id for edge in nodes[-1].artifact_inputs)

    # Exercise the exact numerical nodes declared above, with explicit invented
    # input/matrix producers. This is not a substitute for native owner issuance.
    donor = group.donor_frame
    for name in group.donor_columns:
        donor.person[name] = group.donor_columns[name].to_numpy()
    store = ContentStore(tmp_path / "store")
    source_nodes, paths = [], {}
    for node_id, frame, name in (
        (fits[0].population, donor, "donor"),
        (applies[0].population, receiving, "recipient"),
    ):
        digest = hashlib.sha256(
            name.encode() + frame.person.to_json(orient="table").encode()
        ).hexdigest()
        paths[name] = store.put_frame(digest, frame)
        source_nodes.append(
            Node(
                node_id,
                InventedAmountSource.ref,
                sources=(name,),
                structural=StructuralDelta.CREATE,
                outputs=tuple(
                    Owned(entity, column, str(frame.table(entity)[column].dtype))
                    for entity in frame.entities
                    for column in frame.table(entity)
                    if column != entity + "_id"
                    and not (
                        entity == "person"
                        and column
                        in {
                            "person_" + group + "_id"
                            for group in frame.schema.group_entities
                        }
                    )
                ),
            )
        )
    matrix = Node(
        enrichment.PROJECTION_NODE,
        InventedAmountMatrix.ref,
        population=applies[0].population,
        params={"matrix_hex": group.matrix.hex()},
        artifact_outputs=(
            ArtifactOutput(
                "workers_compensation_matrix", model_input.RECIPIENT_MATRIX_TYPE
            ),
        ),
    )
    declaration = Graph(
        "us",
        (SourceRef("donor", "frame-store"), SourceRef("recipient", "frame-store")),
        (*source_nodes, matrix, *fits, *applies),
    )
    kernels = KernelRegistry()
    for kernel in (
        InventedAmountSource(),
        InventedAmountMatrix(),
        LegacyQRFTrainKernel(),
        LegacyQRFApplyMatrixKernel(),
    ):
        kernels.register(kernel)
    compiled = compile_graph(declaration)
    run = run_graph(compiled, sources=paths, store=store, kernels=kernels)
    decoded = model_input.decode_recipient_matrix(group.matrix)
    draw = codec.read_raw_target(
        store.load_bytes(run.node(applies[0].id).opaque_artifacts["raw_draw"]),
        target=group.spec.targets[0],
        index=decoded.features.index,
    )
    assert len(draw) == 1 and draw[0] in (0.0, 120.0)
    columns = amounts.attach_columns(
        qualified,
        receiving,
        {
            group.spec.key: pd.DataFrame(
                {group.spec.targets[0]: draw}, index=decoded.features.index
            )
        },
    )
    np.testing.assert_array_equal(
        columns["person", "workers_compensation"],
        [np.nan, draw[0], 120, 0, np.nan, draw[0], 120, 0],
    )
    for entity in receiving.entities:
        pd.testing.assert_frame_equal(receiving.table(entity), before[entity])
    assert receiving.resolve_weights("household").kind is weight_kind
    assert receiving.resolve_weights("household").values.tobytes() == weight_bytes
    assert qualified.native.workers_compensation.iloc[:2].isna().all()
    warm = run_graph(
        compiled, sources=paths, store=store, kernels=kernels, resume="require"
    )
    assert warm.key == run.key and all(node.hit for node in warm.nodes.values())
    assert store.load_bytes(
        warm.node(applies[0].id).opaque_artifacts["raw_draw"]
    ) == store.load_bytes(run.node(applies[0].id).opaque_artifacts["raw_draw"])


def _wc_source_arguments(root, patch):
    from test_us_asec_coverage_authentication import _changed_parent
    from test_us_graph_us_survey_enrichment import enrichment_source_arguments

    from microcosm.build.frame_checkpoint import load_frame_checkpoint
    from microcosm.build.us_runtime import asec_person_income_source as restoration
    from microcosm.build.us_runtime import current_asec_demographics as demographics

    arguments = enrichment_source_arguments(root, patch)
    source = arguments["source_dir"] / "asec"
    parent, attachment = source / "parent.h5", source / "household-attachment.h5"
    person = load_frame_checkpoint(parent).frame.person
    wc_values = person.WC_VAL.to_numpy(dtype="float64", copy=True)
    for pid, value in zip((105, 106, 107, 108), (120.0, 0.0, 0.0, 500.0), strict=True):
        wc_values[person.person_id.eq(pid)] = value
    _changed_parent(parent, attachment, patch, {"WC_VAL": wc_values})
    updated = load_frame_checkpoint(parent).frame.person.set_index("PERIDNUM")
    paths, pins = {}, []
    for year, member, archive, *_ in wc.receipt.coverage._MEMBER_PINS:
        path = source / member
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        raw["WC_VAL"] = [str(int(updated.loc[key, "WC_VAL"])) for key in raw.PERIDNUM]
        raw["WC_YN"] = [
            "0" if int(age) < 15 else "1" if int(value) > 0 else "2"
            for age, value in zip(raw.A_AGE, raw.WC_VAL, strict=True)
        ]
        if year == 2024:
            raw.loc[raw.PERIDNUM.eq(str(7).zfill(22)), "WC_YN"] = "1"
            raw.loc[raw.PERIDNUM.eq(str(8).zfill(22)), "WC_YN"] = "0"
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
    for module in (wc.receipt.coverage, restoration, demographics.demographic):
        patch.setattr(module, "_MEMBER_PINS", tuple(pins))
    output = root / "wc-restored-money"
    restoration.restore_asec_person_income_source(
        parent, attachment, member_paths=paths, output_dir=output
    )
    shutil.copyfile(
        output / restoration.CHECKPOINT_FILENAME, source / "person-income-attachment.h5"
    )
    return arguments


def test_authenticated_wc_capture_preserves_knownness_and_refuses_changed_source(
    tmp_path, monkeypatch
):
    # Only invented bytes/pins are changed, before the genuine preparation issues.
    # This does not construct the expensive financial/PUF/enrichment host.
    arguments = _wc_source_arguments(tmp_path, monkeypatch)
    prepared = wc.receipt.source.prepare_authenticated_survey_population(**arguments)
    qualified = wc.qualify_current_asec_workers_compensation(prepared)
    person = qualified.person.set_index("native_person_id")
    np.testing.assert_array_equal(
        person.loc[[105, 106, 107, 108], "canonical_amount"],
        [120.0, 0.0, np.nan, np.nan],
    )
    assert person.loc[107, "reporting_status"] == "ambiguous_recipient_zero"
    assert (
        person.loc[108, "reporting_status"]
        == "contradictory_outside_reporting_universe"
    )
    assert person.loc[108, "source_amount"] == 500.0
    assert qualified.evidence["dictionary"] == wc.DICTIONARY
    assert qualified.evidence["read_columns"] == list(wc.READ_COLUMNS)
    assert (
        qualified.evidence["projection_sha256"]
        == hashlib.sha256(qualified.person.to_json(orient="table").encode()).hexdigest()
    )
    assert qualified.evidence["unallocated_observation_claim"] is False
    # The extracted UC branch retains its own field identity and old protocol.
    uc = wc.receipt.qualify_current_asec_unemployment(prepared)
    assert uc.evidence["protocol"] == wc.receipt.PROTOCOL
    assert uc.evidence["read_columns"] == list(wc.receipt.READ_COLUMNS)
    changed = arguments["source_dir"] / "asec" / "pppub25.csv"
    changed.write_bytes(changed.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        wc.qualify_current_asec_workers_compensation(prepared)


def test_workers_compensation_keeps_reported_zero_distinct_from_niu():
    # ASEC2025 dictionary page51: WC_YN asks age15+, WC_VAL0 is none or NIU.
    source = np.array([120.0, 0.0, 0.0, 0.0, 500.0, 0.0, np.nan])
    age = np.array([40.0, 15.0, 40.0, 14.0, 40.0, 40.0, 40.0])
    result = wc.reporting_basis(source, age, ("1", "2", "0", "0", "2", "1", "1"))
    np.testing.assert_array_equal(result.source_amount, source)
    np.testing.assert_array_equal(
        result.canonical_amount, [120.0, 0.0, np.nan, np.nan, np.nan, np.nan, np.nan]
    )
    assert result.reporting_status.tolist() == [
        "known_receipt",
        "known_nonreceipt",
        "niu",
        "outside_reporting_universe",
        "contradictory_no_positive",
        "ambiguous_recipient_zero",
        "missing_amount",
    ]
    assert (
        result.source_reporting_universe.tolist() == [True] * 3 + [False] + [True] * 3
    )


@pytest.mark.parametrize("token", ["", "01", "yes", "9"])
def test_workers_compensation_never_treats_unknown_receipt_as_no(token):
    result = wc.reporting_basis(np.array([0.0]), np.array([40.0]), (token,))
    assert np.isnan(result.canonical_amount.iloc[0])
    assert result.receipt_literal.iloc[0] == token


def test_workers_compensation_declares_opt_in_closed_group():
    spec = amounts.selected_groups(("workers_compensation",))[0]
    assert spec.fields == (("WC_VAL", "workers_compensation"),)
    assert spec.targets == ("survey_amount_target_WC_VAL",)
    assert [
        g.key for g in amounts.selected_groups(("unemployment", "health_costs"))
    ] == [
        "unemployment",
        "health_costs",
    ]
    with pytest.raises(ValueError, match="GROUP_ROSTER_OR_ORDER"):
        amounts.selected_groups(("invented_receipt",))


def test_workers_compensation_qualifier_refuses_unissued_preparation():
    with pytest.raises(ValueError, match="PREPARATION_TYPE"):
        wc.qualify_current_asec_workers_compensation(object())


def test_workers_compensation_literal_reader_uses_wc_not_uc_fields(tmp_path):
    path = tmp_path / "invented.csv"
    path.write_text(
        ",".join(wc.READ_COLUMNS) + "\n0000000000000000000001,1,1,40,123,1\n"
    )
    raw = wc.read_capture(path, rows=1)
    assert raw.iloc[0].WC_VAL == "123"
    assert raw.iloc[0].WC_YN == "1"
    assert not {"UC_VAL", "UC_YN"} & set(raw)
    path.write_text(path.read_text().replace("WC_YN", "UC_YN"))
    with pytest.raises(ValueError, match="HEADER"):
        wc.read_capture(path, rows=1)


@pytest.mark.parametrize(
    "name",
    (
        "current_asec_workers_compensation_source.py",
        "current_asec_unemployment_source.py",
        "current_survey_amounts.py",
        "graph_us_survey_enrichment.py",
    ),
)
def test_wc_changed_modules_keep_the_source_spine_contract(name):
    from test_us_spine_blindness import _US_RUNTIME, _non_owner_source_spine_accesses

    assert not _non_owner_source_spine_accesses(name, (_US_RUNTIME / name).read_text())
