"""Actual invented originals through the fixed US enrichment host.

Fixture source bytes and private test pins are built before any owner issues.
No production issuer, qualification method, donor model or engine is replaced.
"""

import hashlib
import os
import shutil
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_asec_coverage_authentication import _changed_parent
from test_us_current_survey_health_coverage import (
    _health_acs_person,
    _health_asec_table,
)
from test_us_graph_atomic_survey_population import _support_payload
from test_us_graph_puf55_canonical_donor import _original_sources
from test_us_puf55_survey_recipients import _recipient_source_arguments

from microcosm.build.frame_checkpoint import load_frame_checkpoint
from microcosm.build.us_runtime import asec_person_income_source as restoration
from microcosm.build.us_runtime import current_asec_demographics as demographics
from microcosm.build.us_runtime import current_asec_unemployment_source as uc
from microcosm.build.us_runtime import graph_us_survey_enrichment as graph
from microcosm.graph.keys import opaque_artifact_key


def enrichment_source_arguments(root, patch):
    import test_us_puf55_survey_recipients as recipient_fixture
    import test_us_survey_population_preparation as source_fixture

    patch.setattr(source_fixture, "_person", _health_acs_person(source_fixture._person))
    patch.setattr(
        recipient_fixture, "_person", _health_acs_person(recipient_fixture._person)
    )
    arguments = _recipient_source_arguments(root, patch)
    source = arguments["source_dir"] / "asec"
    parent_path, attachment = source / "parent.h5", source / "household-attachment.h5"
    person = load_frame_checkpoint(parent_path).frame.person
    changes = {}
    for field, amounts in (
        ("UC_VAL", [120.0, 0.0, 0.0, 500.0]),
        ("PHIP_VAL", [0.0, 100.0, 200.0, 400.0]),
        ("PMED_VAL", [0.0, 10.0, 20.0, 40.0]),
        ("POTC_VAL", [0.0, 1.0, 2.0, 4.0]),
    ):
        data = person[field].to_numpy(dtype="float64", copy=True)
        for pid, amount in zip((105, 106, 107, 108), amounts, strict=True):
            positions = np.flatnonzero(person.person_id.to_numpy() == pid)
            assert len(positions) == 1
            data[positions[0]] = amount
        changes[field] = data
    _changed_parent(parent_path, attachment, patch, changes)
    updated = load_frame_checkpoint(parent_path).frame.person.set_index("PERIDNUM")
    paths, pins = {}, []
    for year, member, archive, *_ in uc.coverage._MEMBER_PINS:
        path = source / f"pppub{year - 1999}.csv"
        raw = _health_asec_table(pd.read_csv(path, dtype=str, keep_default_na=False))
        for field in changes:
            raw[field] = [str(int(updated.loc[key, field])) for key in raw.PERIDNUM]
        raw["UC_YN"] = [
            "0" if int(age) < 15 else "1" if int(amount) > 0 else "2"
            for age, amount in zip(raw.A_AGE, raw.UC_VAL, strict=True)
        ]
        if year == 2024:
            # These remain source observations, not known model donors.
            raw.loc[raw.PERIDNUM.eq(str(7).zfill(22)), "UC_YN"] = "1"
            raw.loc[raw.PERIDNUM.eq(str(8).zfill(22)), "UC_YN"] = "0"
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
    for owner in (uc.coverage, restoration, demographics.demographic):
        patch.setattr(owner, "_MEMBER_PINS", tuple(pins))
    output = root / "amount-restored-money"
    restoration.restore_asec_person_income_source(
        parent_path, attachment, member_paths=paths, output_dir=output
    )
    shutil.copyfile(
        output / restoration.CHECKPOINT_FILENAME, source / "person-income-attachment.h5"
    )
    return arguments


def test_actual_current_uc_projection_preserves_ambiguous_and_contradictory_sources(
    tmp_path, monkeypatch
):
    arguments = enrichment_source_arguments(tmp_path, monkeypatch)
    prepared = uc.source.prepare_authenticated_survey_population(**arguments)
    qualified = uc.qualify_current_asec_unemployment(prepared)
    person = qualified.person.set_index("native_person_id")
    assert person.loc[105, "canonical_amount"] == 120.0
    assert np.isnan(person.loc[106, "canonical_amount"])
    assert person.loc[106, "reporting_status"] == "outside_reporting_universe"
    assert person.loc[107, "reporting_status"] == "ambiguous_recipient_zero"
    assert (
        person.loc[108, "reporting_status"]
        == "contradictory_outside_reporting_universe"
    )
    assert person.loc[[107, 108], "canonical_amount"].isna().all()
    assert person.loc[108, "source_amount"] == 500.0
    assert person.amount_validity.eq(1).all()
    assert (
        hashlib.sha256(qualified.person.to_json(orient="table").encode()).hexdigest()
        == qualified.evidence["projection_sha256"]
    )
    health = graph.health_graph.qualify_health_coverage(prepared)
    assert set(health.raw.source) == {"asec", "acs"}
    graph.health_graph.health_coverage_seal(health)
    prepared.checked_view()


@pytest.fixture(scope="module")
def enriched(tmp_path_factory, request):
    root = tmp_path_factory.mktemp("survey-enrichment")
    with pytest.MonkeyPatch.context() as patch:
        arguments = enrichment_source_arguments(root, patch)
        # Development retries copy only an explicitly supplied invented cache.
        # Every hit still passes the real graph source/key/artifact validators;
        # no previous process's source handle or receipt grants authority.
        cache = os.environ.get("MICROCOSM_TEST_INVENTED_ENRICHMENT_CACHE")
        if cache:
            shutil.copytree(cache, root / "store")
        payload, source_ids = _support_payload()
        support = root / "invented-block-support.npz"
        support.write_bytes(payload)
        config = graph.parent.financial.reconstruction.AtomicSurveyReconstruction(
            support_path=str(support),
            support_sha256=hashlib.sha256(payload).hexdigest(),
            source_ids=tuple(sorted(source_ids.items())),
            seed=17,
        )
        print("INVENTED financial cold start", flush=True)
        financial = graph.parent.financial.run_atomic_survey_financial(
            **arguments,
            geography_config=config,
            store_root=root / "store",
            demographic_conditioning=True,
            n_estimators=2,
            return_values=True,
        )
        print("INVENTED financial closed; PUF source start", flush=True)
        definition, paths, _ = _original_sources(
            root / "original-puf", full_finalization_support=True
        )
        print("INVENTED PUF cold start", flush=True)
        parent = graph.parent.run_survey_puf55(
            financial,
            donor_sources=paths,
            fixture_definition=definition,
            seed=578,
            n_estimators=2,
            zero_atol=0,
        )
        print(
            "INVENTED PUF closed; cached nodes",
            sum(n.hit for n in parent.manifest.nodes.values()),
            "enrichment cold start",
            flush=True,
        )
        cold = graph.run_us_survey_enrichment(parent, n_estimators=2)
        final_control_only = all(
            item.name == "test_final_parent_revocation_invalidates_the_retained_child"
            for item in request.session.items
        )
        warm = None
        if not final_control_only:
            print("INVENTED enrichment cold closed; required start", flush=True)
            warm = graph.run_us_survey_enrichment(
                parent, n_estimators=2, resume="require"
            )
            print("INVENTED required closed; owner and readback controls", flush=True)
        else:
            print(
                "INVENTED child ready; focused final parent revocation control",
                flush=True,
            )
        case = SimpleNamespace(
            parent=parent,
            financial=financial,
            cold=cold,
            warm=warm,
            root=root,
            parent_revoked=False,
        )
        yield case
        if not case.parent_revoked:
            graph.parent.check_survey_puf55_run(parent)


def test_actual_enrichment_cold_required_and_complete_parent_preservation(enriched):
    cold, warm = enriched.cold, enriched.warm
    assert cold.manifest.key == warm.manifest.key
    assert all(r.hit for r in warm.manifest.nodes.values())
    assert len(cold.compiled.order) == 271
    graph.physical.replay.same_replayed_population(cold.population, warm.population)
    parent = enriched.parent.population
    for entity in parent.frame.entities:
        pd.testing.assert_frame_equal(
            parent.frame.table(entity),
            cold.population.frame.table(entity).loc[
                :, parent.frame.table(entity).columns
            ],
            check_exact=True,
        )
    for owner, value in parent.owners.items():
        assert cold.population.owners[owner] == value
    assert cold.population.mass_ledger == parent.mass_ledger
    people = cold.population.frame.person
    for _, rows in people.groupby(
        graph.values.provenance.support_source_id_column("person")
    ):
        assert len(rows) == 2
        outputs = [out for g in graph.values.GROUPS for _, out in g.fields]
        np.testing.assert_array_equal(
            rows[outputs].iloc[0].to_numpy(), rows[outputs].iloc[1].to_numpy()
        )
    assert people.unemployment_compensation.isna().any()
    assert people.survey_uc_reporting_status.eq("ambiguous_recipient_zero").any()
    assert people.survey_uc_reporting_status.eq(
        "contradictory_outside_reporting_universe"
    ).any()
    assert set(f.output for f in graph.health_graph.health.FIELDS) <= set(people)
    assert people.has_medicaid_health_coverage_at_interview.isna().any()
    assert people.has_esi.notna().all()
    for run in (cold, warm):
        assert run.checked_view().population is run.population
    table = warm.population.frame.person
    column = "unemployment_compensation"
    original = float(table.loc[table.index[0], column])
    table.loc[table.index[0], column] = 777.0
    with pytest.raises(ValueError, match="RUN_CHANGED"):
        warm.checked_view()
    table.loc[table.index[0], column] = original
    with pytest.raises(ValueError, match="UNISSUED_RUN"):
        warm.checked_view()


def test_checked_enrichment_refuses_copied_handle(enriched):
    with pytest.raises(ValueError, match="UNISSUED_RUN"):
        graph.check_survey_enrichment_run(replace(enriched.cold))


def test_retained_projection_reports_and_artifact_lineage_controls(enriched):
    run = enriched.cold
    boundary = graph._ISSUED[id(run)][1]
    for table, column in (
        (boundary.qualified.native, "unemployment_compensation"),
        (boundary.qualified.reports, "survey_current_UC_VAL_origin"),
    ):
        original = table[column].iloc[0]
        table.loc[table.index[0], column] = (
            987.0 if column == "unemployment_compensation" else "invented-mutation"
        )
        with pytest.raises(ValueError, match="BOUNDARY_CHANGED"):
            boundary.pure()
        table.loc[table.index[0], column] = original
        boundary.pure()
    original = boundary.health.raw.iloc[0]["HINS1"]
    boundary.health.raw.loc[boundary.health.raw.index[0], "HINS1"] = "invented-mutation"
    with pytest.raises(ValueError, match="PROJECTION_BINDING"):
        boundary.pure()
    boundary.health.raw.loc[boundary.health.raw.index[0], "HINS1"] = original
    boundary.pure()
    loaded = graph.parent.financial._artifacts(
        run.manifest,
        run.compiled,
        run.store,
        run.kernels,
        dict(boundary.keys),
        dict(boundary.implementations),
    )
    node = run.compiled.graph.node(graph.ATTACH_NODE)
    artifacts = graph.parent._loaded_values(boundary, run.manifest, loaded, node)
    edge = artifacts["projection"]
    changed = {
        **artifacts,
        "projection": replace(
            edge, producer_key="f" * 64, key=opaque_artifact_key("f" * 64, "projection")
        ),
    }
    with pytest.raises(ValueError, match="ARTIFACT_PRODUCER_KEY"):
        boundary.context(SimpleNamespace(node=node, sources={}, artifacts=changed))
    boundary.context(SimpleNamespace(node=node, sources={}, artifacts=artifacts))
    run.checked_view()


def test_complete_frame_store_readback_and_input_coverage(enriched):
    from microcosm.build.us_runtime.population_input_coverage import (
        diagnose_us_input_coverage,
    )
    from microcosm.build.us_runtime.survey_population_replay import same_replayed_frame

    run = enriched.cold
    run.checked_view()
    before = diagnose_us_input_coverage(
        enriched.parent.population,
        compiled=enriched.parent.compiled,
        manifest=enriched.parent.manifest,
    )
    after = diagnose_us_input_coverage(
        run.population, compiled=run.compiled, manifest=run.manifest
    )
    outputs = {out for g in graph.values.GROUPS for _, out in g.fields} | {
        f.output for f in graph.health_graph.health.FIELDS
    }
    assert outputs <= set(before.missing_inputs)
    assert not outputs & set(after.missing_inputs)
    assert len(after.inputs) == 161
    assert not after.ambiguous_grains and not after.block_storage_issues
    key = hashlib.sha256(b"invented-survey-enrichment-export" + run.receipt).hexdigest()
    run.store.put_frame(key, run.population.frame)
    restored = run.store.load_frame(key, node_key=key)
    same_replayed_frame(run.population.frame, restored)
    (enriched.root / "coverage.json").write_bytes(after.to_bytes())
    (enriched.root / "acceptance.json").write_bytes(
        graph.codec.encode_json(
            {
                "protocol": "microcosm.us.invented-enrichment-test.v1",
                "nodes": len(run.compiled.order),
                "input_names": len(after.inputs),
                "missing_inputs": len(after.missing_inputs),
                "frame_store_key": key,
                "full_frame_readback": True,
                "financial_setup_hits": sum(
                    n.hit for n in enriched.financial.manifest.nodes.values()
                ),
                "puf_setup_hits": sum(
                    n.hit for n in enriched.parent.manifest.nodes.values()
                ),
                "cold_hits": sum(n.hit for n in run.manifest.nodes.values()),
                "required_hits": sum(
                    n.hit for n in enriched.warm.manifest.nodes.values()
                ),
                "native_data_used": False,
                "release_eligible": False,
            }
        )
    )
    run.checked_view()


def test_final_parent_revocation_invalidates_the_retained_child(enriched):
    # Last destructive control after readback: never reconstruct or restore an issuer.
    run = enriched.cold
    people = enriched.parent.population.frame.person
    column = "employment_income_before_lsr"
    position = np.flatnonzero(np.isfinite(people[column].to_numpy()))[0]
    row = people.index[position]
    original = float(people.loc[row, column])
    people.loc[row, column] = original + 321.0
    with pytest.raises(ValueError):
        run.checked_view()
    enriched.parent_revoked = True
    people.loc[row, column] = original
    with pytest.raises(ValueError, match="UNISSUED_PUF55_RUN"):
        graph.parent.check_survey_puf55_run(enriched.parent)
    with pytest.raises(ValueError, match="UNISSUED_RUN"):
        run.checked_view()
