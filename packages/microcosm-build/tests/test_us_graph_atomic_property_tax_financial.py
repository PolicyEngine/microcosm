"""Optional 38-node financial host over actual invented source owners."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_us_graph_atomic_property_financial import property_host_source  # noqa: F401

from microcosm.build.us_runtime import graph_atomic_survey_financial as runner
from microcosm.build.us_runtime import graph_current_survey_property as property_graph
from microcosm.build.us_runtime import graph_property_tax_leaves as tax
from microcosm.build.us_runtime import graph_puf55_survey_recipients as recipients
from microcosm.graph import describe, explain_html
from microcosm.graph.store import StoreCorrupt


@pytest.fixture(scope="module")
def tax_host_cold(request):
    case = request.getfixturevalue("property_host_source")
    return runner.run_atomic_survey_financial(
        **case.call,
        property_income=replace(case.options, completion_routing=True),
        rebase_property_taxes=True,
        resume="auto",
    )


@pytest.fixture(scope="module")
def tax_host_warm(request, tax_host_cold):
    case = request.getfixturevalue("property_host_source")
    return runner.run_atomic_survey_financial(
        **case.call,
        property_income=replace(case.options, completion_routing=True),
        rebase_property_taxes=True,
        resume="require",
    )


def test_opt_in_cold_required_issued_output(tax_host_cold, tax_host_warm):
    cold, warm = tax_host_cold, tax_host_warm
    assert len(cold.compiled.order) == len(warm.compiled.order) == 38
    assert cold.manifest.key == warm.manifest.key
    assert all(record.hit for record in warm.manifest.nodes.values())
    runner.atomic.same_replayed_population(
        cold.financial_population, warm.financial_population
    )
    checked = cold.checked_view()
    assert checked.payload == warm.checked_view().payload
    document = runner.codec.decode_json(checked.payload)
    assert document["property_income"]["completion_routing"] is True
    assert document["tax_split_rebased"] is True
    assert document["tax_rebase_node_count"] == 3
    assert document["tax_leaf_complete"] is False
    assert document["release_eligible"] is False
    assert runner.financial_output_node(cold) == tax.GATE_NODE
    assert cold.financial_population.version == tax.RECEIVING_NODE


def test_opt_in_preserves_full_prior_populations_and_unknowns(tax_host_cold):
    run = tax_host_cold
    state = runner._run_entry(run)[2]
    legacy, before, after = (
        state.legacy_financial_population,
        state.property_population,
        run.financial_population,
    )
    assert before.version == legacy.version != after.version
    for entity in before.frame.entities:
        kept = [
            name
            for name in before.frame.table(entity)
            if name not in tax.TAX_LEAF_COLUMNS
        ]
        pd.testing.assert_frame_equal(
            before.frame.table(entity)[kept],
            after.frame.table(entity)[kept],
            check_exact=True,
        )
    assert before.frame.schema == after.frame.schema
    assert before.frame.metadata == after.frame.metadata
    assert before.frame.mass_log == after.frame.mass_log
    assert after.mass_ledger[:-1] == before.mass_ledger
    assert after.mass_ledger[-1].node_id == tax.RECEIVING_NODE
    pd.testing.assert_series_equal(before.frame.strata, after.frame.strata)
    for entity in before.design_weights:
        np.testing.assert_array_equal(
            before.design_weights[entity], after.design_weights[entity]
        )
    expected = tax.split_property_tax_leaves(before.frame.person)
    for name in tax.TAX_LEAF_COLUMNS:
        np.testing.assert_array_equal(
            after.frame.person[name].to_numpy().view("uint64"),
            expected[name].to_numpy().view("uint64"),
        )
    assert after.frame.person[list(tax.TAX_LEAF_COLUMNS)].isna().any().any()
    assert legacy.frame.person[list(tax.TAX_LEAF_COLUMNS)].notna().all().all()


def test_incomplete_development_run_refuses_puf_before_source_qualification(
    tax_host_cold,
):
    run = tax_host_cold
    assert (
        runner.codec.decode_json(runner._run_entry(run)[2].tax_verification)["complete"]
        is False
    )
    with pytest.raises(ValueError, match="PROPERTY_TAX_INPUTS_INCOMPLETE"):
        recipients.values.qualify_puf55_survey_recipients(run)
    edges = recipients._edges(run)
    edge = next(e for e in edges if e.name == "property_tax_verification")
    assert edge.producer == tax.GATE_NODE and edge.type == tax.VERIFICATION_TYPE
    assert recipients._pins(run)[edge.name]["payload_sha256"] == runner.codec.sha(
        runner._run_entry(run)[2].tax_verification
    )


@pytest.mark.parametrize("target", ["property", "leaf", "unknown"])
def test_retained_tax_lifetime_refuses_changes(tax_host_cold, target):
    run = tax_host_cold
    state = runner._run_entry(run)[2]
    population = (
        state.property_population if target == "property" else run.financial_population
    )
    table = population.frame.person
    column = (
        tax.PROPERTY_COMPONENTS[0] if target == "property" else tax.TAX_LEAF_COLUMNS[0]
    )
    row = table.index[
        table[column].isna() if target == "unknown" else table[column].notna()
    ][0]
    old = table.loc[row, column]
    table.loc[row, column] = 17.0
    try:
        with pytest.raises(ValueError, match="POPULATION_CHANGED"):
            run.checked_view()
    finally:
        table.loc[row, column] = old
    runner._pure_run(run, runner._run_entry(run))


def test_rebased_dataclass_copy_remains_unissued(tax_host_cold):
    with pytest.raises(ValueError, match="UNISSUED_FINANCIAL_RUN"):
        replace(tax_host_cold).checked_view()


@pytest.mark.parametrize("flag", [None, 1, "yes", True])
def test_invalid_or_unconfigured_rebase_refuses_before_source_io(flag):
    with pytest.raises(
        ValueError, match="PROPERTY_TAX_FLAG|PROPERTY_TAX_REQUIRES_PROPERTY"
    ):
        runner.run_atomic_survey_financial(
            None,
            snapshot_root=None,
            store_root=None,
            fraction=1.0,
            seed=17,
            geography_config=None,
            rebase_property_taxes=flag,
        )


def test_completion_private_artifact_and_aggregate_receipt_replay(
    tax_host_cold, tax_host_warm
):
    """Actual enabled host/required path, with no additional source qualification."""
    private_payloads, receipts = [], []
    for run in (tax_host_cold, tax_host_warm):
        state = runner._run_entry(run)
        record = run.manifest.node(property_graph.PROJECTION_NODE)
        assert set(record.opaque_artifacts) == {
            "projection",
            "donor_matrix",
            "recipient_matrix",
            "completion_routing",
        }
        payload = run.store.load_bytes(record.opaque_artifacts["completion_routing"])
        runner.survey._final_artifact(
            run.manifest,
            run.store,
            node_id=property_graph.PROJECTION_NODE,
            name="completion_routing",
            type_=property_graph.completion.PROPERTY_COMPLETION_TYPE,
            payload=payload,
            capabilities=run.kernels.get(
                property_graph.CurrentSurveyPropertyProjectionKernel.ref
            ).capabilities,
        )
        assert (
            property_graph.PROJECTION_NODE,
            "completion_routing",
            runner.codec.sha(payload),
        ) in state[2].artifact_hashes
        private = json.loads(payload)
        assert private["stage"] == "before_property_models"
        assert set(private["tables"]) == {
            "person",
            "components",
            "reasons",
            "clones",
            "summary",
        }
        assert len(
            private["tables"]["clones"]["rows"]
        ) == run.financial_population.frame.n("person")
        receipt = record.receipt["completion_routing"]
        assert set(receipt) == {
            "summary",
            "private_artifact",
            "private_artifact_sha256",
        }
        assert receipt["private_artifact"] == "completion_routing"
        assert receipt["private_artifact_sha256"] == runner.codec.sha(payload)
        summary = receipt["summary"]
        assert (
            summary["source_authority"] is False
            and summary["amounts_assigned"] is False
        )
        assert summary["model_executed"] is False
        assert summary["routes_mutually_exclusive"] is True
        assert summary["reasons_overlap"] is True
        assert all(
            set(row) == {"selection", *property_graph.completion._PUBLIC_MEASURES}
            for row in summary["summary"]
        )
        public = explain_html(run.compiled, run.manifest) + describe(
            run.compiled, property_graph.PROJECTION_NODE, run.manifest
        )
        assert "completion_routing" in public and "carry_known_components" in public
        assert payload.decode() not in public
        private_payloads.append(payload)
        receipts.append(receipt)
        runner._pure_run(run, state)
    assert private_payloads[0] == private_payloads[1]
    assert receipts[0] == receipts[1]


def test_completion_artifact_retained_lifetime_refuses_store_tampering(tax_host_cold):
    """Keep this destructive artifact control last; restore only test-owned bytes."""
    run = tax_host_cold
    record = run.manifest.node(property_graph.PROJECTION_NODE)
    path = (
        run.store.object_path(record.opaque_artifacts["completion_routing"])
        / "payload.bin"
    )
    original = path.read_bytes()
    try:
        path.write_bytes(original + b" ")
        with pytest.raises(StoreCorrupt):
            run.checked_view()
    finally:
        path.write_bytes(original)
    runner._pure_run(run, runner._run_entry(run))
