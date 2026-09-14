"""Issued 19/35-node financial graphs over invented source files and blocks."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_current_asec_demographics import _demographic_arguments
from test_us_current_property_income_sources import source_arguments
from test_us_graph_atomic_survey_population import _support_payload
from test_us_survey_social_security import _social_security_arguments

from microcosm.build.us_runtime import graph_atomic_survey_financial as runner
from microcosm.build.us_runtime import graph_current_survey_property as property_graph
from microcosm.build.us_runtime import graph_puf55_survey_recipients as puf_recipient
from microcosm.graph import compile_graph


@pytest.fixture(scope="module")
def property_host_source(tmp_path_factory):
    import test_us_current_asec_demographics as demographic_fixture_module
    import test_us_current_asec_income_routing as routing_fixture
    import test_us_survey_population_preparation as survey_fixture_module

    from microcosm.build.us_runtime import current_asec_demographics as demographics

    root = tmp_path_factory.mktemp("atomic-property-financial")
    with pytest.MonkeyPatch.context() as patch:
        # Compose complete invented member bytes before the real source issuer
        # runs. Only fixture constructors and their registry pins are replaced.
        # The nesting order is original source -> Social Security -> demographics
        # -> routing -> property. Each extension preserves prior columns and
        # rebuilds the retained attachment; preparation is issued only afterward.
        def social_security_fixture(path, monkeypatch, *, zero):
            assert zero is False
            return _social_security_arguments(path, monkeypatch, ambiguous=False)

        def demographic_fixture(path, monkeypatch):
            return _demographic_arguments(path, monkeypatch, unknown=False, zero=False)

        patch.setattr(demographic_fixture_module, "fixture", social_security_fixture)
        patch.setattr(routing_fixture, "fixture", demographic_fixture)
        original_person = survey_fixture_module._person

        def person(*args, **kwargs):
            row = original_person(*args, **kwargs)
            # The property fixture makes this person age 14. Its other income
            # questions must also be blank before the source archive is built.
            if row["SERIALNO"] == "2024GQ0000001":
                row.update(SSP="", SSIP="")
            return row

        patch.setattr(survey_fixture_module, "_person", person)
        arguments = source_arguments(root, patch)
        patch.setattr(
            demographics.demographic,
            "_MEMBER_PINS",
            property_graph.sources.routing.coverage._MEMBER_PINS,
        )
        payload, source_ids = _support_payload()
        support = root / "invented-block-support.npz"
        support.write_bytes(payload)
        config = runner.reconstruction.AtomicSurveyReconstruction(
            support_path=str(support),
            support_sha256=hashlib.sha256(payload).hexdigest(),
            source_ids=tuple(sorted(source_ids.items())),
            seed=17,
        )
        options = property_graph.PropertyIncomeOptions(
            scales=(1.0, 1.0, 1.0, 1.0), atol=1e-10, rtol=1e-12, n_estimators=2
        )
        call = {
            **arguments,
            "geography_config": config,
            "demographic_conditioning": True,
            "n_estimators": 2,
            "return_values": True,
        }
        yield SimpleNamespace(root=root, call=call, options=options)


@pytest.fixture(scope="module")
def property_host_base(property_host_source):
    case = property_host_source
    run = runner.run_atomic_survey_financial(
        **case.call, property_income=None, resume="auto"
    )
    yield run
    runner.values.source.verify_survey_population_preparation(run.prefix.preparation)


@pytest.fixture(scope="module")
def property_host_cold(property_host_source):
    case = property_host_source
    run = runner.run_atomic_survey_financial(
        **case.call, property_income=case.options, resume="auto"
    )
    yield run
    runner.values.source.verify_survey_population_preparation(run.prefix.preparation)


@pytest.fixture(scope="module")
def property_host_warm(property_host_source, property_host_cold):
    case = property_host_source
    run = runner.run_atomic_survey_financial(
        **case.call, property_income=case.options, resume="require"
    )
    yield run
    runner.values.source.verify_survey_population_preparation(run.prefix.preparation)


@pytest.fixture(scope="module")
def actual_property_host(request, property_host_source):
    # Explicit ordering demonstrates reuse of the unchanged 19-node prefix.
    # A targeted PUF test requests only the cold host and need not rerun these
    # separately accepted baseline/replay checks.
    base = request.getfixturevalue("property_host_base")
    cold = request.getfixturevalue("property_host_cold")
    warm = request.getfixturevalue("property_host_warm")
    (property_host_source.root / "host-acceptance.json").write_text(
        json.dumps(
            {
                "scope": "invented source-issued financial host; no native or engine",
                "base_nodes": len(base.compiled.order),
                "extended_nodes": len(cold.compiled.order),
                "cold_hits": sum(n.hit for n in cold.manifest.nodes.values()),
                "required_hits": sum(n.hit for n in warm.manifest.nodes.values()),
                "persons": cold.financial_population.frame.n("person"),
                "households": cold.financial_population.frame.n("household"),
                "owned_property_columns": [
                    o.column for o in property_graph.owned_columns()
                ],
                "tax_split_rebased": False,
            },
            indent=2,
        )
        + "\n"
    )
    return SimpleNamespace(
        base=base, cold=cold, warm=warm, options=property_host_source.options
    )


def test_default_graph_and_payload_remain_nineteen_nodes(actual_property_host):
    run = actual_property_host.base
    document = runner.codec.decode_json(run.checked_view().payload)
    assert len(run.compiled.order) == 19
    assert runner.financial_output_node(run) == runner.financial.ATTACH_NODE
    assert document["owned_columns"] == list(runner.values.OUTPUTS)
    assert (
        not {"property_income", "property_node_count", "tax_split_rebased"}
        & document.keys()
    )
    assert not any(
        n.startswith(property_graph.PREFIX + ".") for n in run.compiled.order
    )


def test_actual_extended_host_cold_required_and_checked_issuance(actual_property_host):
    case = actual_property_host
    assert len(case.cold.compiled.order) == len(case.warm.compiled.order) == 35
    assert case.cold.manifest.key == case.warm.manifest.key
    assert sum(n.hit for n in case.cold.manifest.nodes.values()) == 19
    assert all(n.hit for n in case.warm.manifest.nodes.values())
    runner.atomic.same_replayed_population(
        case.cold.financial_population, case.warm.financial_population
    )
    checked = [run.checked_view() for run in (case.cold, case.warm)]
    assert checked[0].payload == checked[1].payload
    document = runner.codec.decode_json(checked[0].payload)
    assert document["property_income"] == runner.codec.decode_json(
        case.options.to_bytes()
    )
    assert document["property_node_count"] == 16
    assert document["tax_split_rebased"] is False
    assert document["capital_gains_conditioning"] == property_graph.CAP_LIMITATION
    assert document["release_eligible"] is False
    assert runner.financial_output_node(case.cold) == property_graph.ATTACH_NODE


def test_extended_host_retains_complete_legacy_population(actual_property_host):
    base, extended = actual_property_host.base, actual_property_host.cold
    state = runner._run_entry(extended)[2]
    runner.atomic.same_replayed_population(
        base.financial_population, state.legacy_financial_population
    )
    before, after = base.financial_population, extended.financial_population
    for entity in before.frame.entities:
        pd.testing.assert_frame_equal(
            after.frame.table(entity)[before.frame.table(entity).columns],
            before.frame.table(entity),
            check_exact=True,
        )
    assert after.version == before.version
    assert after.frame.schema == before.frame.schema
    assert after.frame.links == before.frame.links
    assert after.frame.metadata == before.frame.metadata
    assert after.mass_ledger == before.mass_ledger
    assert after.frame.mass_log == before.frame.mass_log
    pd.testing.assert_series_equal(after.frame.strata, before.frame.strata)
    for entity in before.design_weights:
        np.testing.assert_array_equal(
            after.design_weights[entity], before.design_weights[entity]
        )
        np.testing.assert_array_equal(
            after.frame.weights_for(entity).values,
            before.frame.weights_for(entity).values,
        )
    assert dict(after.owners) == {
        **before.owners,
        **{
            ("person", o.column): property_graph.ATTACH_NODE
            for o in property_graph.owned_columns()
        },
    }


def test_property_values_and_unknowns_follow_original_clone_identity(
    actual_property_host,
):
    run = actual_property_host.cold
    people = run.financial_population.frame.person
    source_id = runner.values.provenance.support_source_id_column("person")
    columns = [o.column for o in property_graph.owned_columns()]
    for _, pair in people.groupby(source_id, sort=False):
        assert len(pair) == 2
        pd.testing.assert_series_equal(
            pair.iloc[0][columns], pair.iloc[1][columns], check_names=False
        )
    assert (~people.property_anchor_known).any()
    assert (
        people.loc[
            ~people.property_anchor_known, property_graph.PROPERTY_REPORTED_TOTAL
        ]
        .isna()
        .all()
    )
    known = people.property_components_known
    assert known.any() and (~known).any()
    channel = runner.values.provenance.support_channel_column("person")
    modeled = known & people[channel].astype(str).eq("acs")
    assert modeled.any()
    values = people.loc[modeled, list(property_graph.PROPERTY_COMPONENTS)]
    np.testing.assert_allclose(
        values.sum(axis=1),
        people.loc[modeled, property_graph.PROPERTY_REPORTED_TOTAL],
        atol=1e-10,
        rtol=1e-12,
    )
    assert (values.iloc[:, :3] >= 0).all().all()
    assert (people.property_reported_total < 0).any()


@pytest.mark.parametrize("defect", ("options", "legacy", "final"))
def test_retained_host_lifetime_rejects_mutation(actual_property_host, defect):
    run = actual_property_host.cold
    state = runner._run_entry(run)[2]
    if defect == "options":
        original = state.property_income.atol
        object.__setattr__(state.property_income, "atol", original + 1.0)
        try:
            with pytest.raises(
                ValueError, match="PROPERTY_OPTIONS_OR_LEGACY_POPULATION_CHANGED"
            ):
                run.checked_view()
        finally:
            object.__setattr__(state.property_income, "atol", original)
    else:
        population = (
            state.legacy_financial_population
            if defect == "legacy"
            else run.financial_population
        )
        table = population.frame.person
        column = (
            runner.values.OUTPUTS[0]
            if defect == "legacy"
            else property_graph.PROPERTY_REPORTED_TOTAL
        )
        index = table.index[table[column].notna()][0]
        original = table.loc[index, column]
        table.loc[index, column] = original + 1.0
        try:
            with pytest.raises(
                ValueError,
                match=(
                    "PROPERTY_OPTIONS_OR_LEGACY_POPULATION_CHANGED"
                    "|FINANCIAL_RUN_POPULATION_CHANGED"
                    "|FINANCIAL_RUN_ATTACHED_POPULATION_CHANGED"
                ),
            ):
                run.checked_view()
        finally:
            table.loc[index, column] = original
    runner._pure_run(run, runner._run_entry(run))


def test_detached_run_copy_has_no_issuer_authority(actual_property_host):
    with pytest.raises(ValueError, match="UNISSUED_FINANCIAL_RUN"):
        replace(actual_property_host.cold).checked_view()


def test_puf_recipient_reads_complete_final_property_writer(property_host_cold):
    run = property_host_cold
    qualified = puf_recipient.values.qualify_puf55_survey_recipients(run)
    nodes = puf_recipient.puf55_survey_recipient_nodes(qualified)
    compiled = compile_graph(
        replace(run.compiled.graph, nodes=(*run.compiled.graph.nodes, *nodes))
    )
    assert len(compiled.order) == 37
    for node in nodes:
        assert property_graph.ATTACH_NODE in compiled.predecessors[node.id]
        inputs = next(s.columns for s in node.inputs if s.entity == "person")
        assert {o.column for o in property_graph.owned_columns()}.issubset(inputs)
    assert qualified.financial_run is run
