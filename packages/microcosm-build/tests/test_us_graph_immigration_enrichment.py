"""Genuine invented original immigration through the existing enrichment host.

No issuer, source validator, fit implementation or country engine is replaced.
Source literal decorators run before issuance; this module contains one shared
fixture so its expensive genuine ancestry is constructed only once.
"""

from __future__ import annotations

import cProfile
import hashlib
import json
import os
import pstats
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from microcosm.build.us_runtime import graph_current_survey_immigration as fragment
from microcosm.build.us_runtime import graph_us_survey_enrichment as graph


def _phase(name):
    if os.environ.get("MICROCOSM_INVENTED_PHASES") == "1":
        print(
            "IMMIGRATION_CLONE_PHASE "
            + json.dumps(
                {
                    "phase": name,
                    "cpu_s": time.process_time(),
                    "wall_s": time.monotonic(),
                }
            ),
            flush=True,
        )


def _profile_rows(stats, root):
    """Only aggregate implementation statistics; never arguments or locals."""
    rows, omitted_calls = [], 0
    project = str(root / "packages") + "/"
    runtime_prefixes = tuple(
        dict.fromkeys(
            [
                str(Path(sys.base_prefix) / "lib") + "/",
                str(Path(sys.prefix) / "lib") + "/",
                *(str(Path(p)) + "/" for p in sys.path if p.endswith("site-packages")),
            ]
        )
    )
    for (filename, line, function), (
        primitive,
        calls,
        total,
        cumulative,
        _,
    ) in stats.items():
        if filename.startswith(project):
            name = "packages/" + filename[len(project) :]
        elif filename == "~":
            name = "builtin"
        elif any(filename.startswith(prefix) for prefix in runtime_prefixes):
            prefix = next(
                prefix for prefix in runtime_prefixes if filename.startswith(prefix)
            )
            name = "runtime/" + filename[len(prefix) :]
        else:
            omitted_calls += calls
            continue
        rows.append(
            {
                "file": name,
                "line": line,
                "function": function,
                "primitive_calls": primitive,
                "calls": calls,
                "tottime": total,
                "cumtime": cumulative,
            }
        )
    return {
        "functions": sorted(rows, key=lambda r: (r["file"], r["line"], r["function"])),
        "omitted_calls": omitted_calls,
    }


def _profiled(label, call, *, forbidden=()):
    """Observe an already necessary phase without rebinding sealed callables."""
    destination = os.environ.get("MICROCOSM_INVENTED_PROFILE_DIR")
    if destination is None and not forbidden:
        return call()
    assert sys.getprofile() is None
    profiler = cProfile.Profile()
    succeeded = False
    start_cpu, start_wall = time.process_time(), time.monotonic()
    try:
        profiler.enable()
        result = call()
        succeeded = True
    finally:
        profiler.disable()
        assert sys.getprofile() is None
        stats = pstats.Stats(profiler).stats
        code_keys = {
            (
                function.__code__.co_filename,
                function.__code__.co_firstlineno,
                function.__code__.co_name,
            )
            for function in forbidden
        }
        forbidden_count = sum(
            value[1] for key, value in stats.items() if key in code_keys
        )
        if destination is not None:
            root = Path(__file__).resolve().parents[3]
            report = {
                "label": label,
                "succeeded": succeeded,
                "cpu_seconds": time.process_time() - start_cpu,
                "wall_seconds": time.monotonic() - start_wall,
                "forbidden_calls": forbidden_count,
                **_profile_rows(stats, root),
            }
            folder = Path(destination)
            assert folder.is_dir()
            path = folder / (label + ".json")
            with path.open("x") as out:
                json.dump(report, out, sort_keys=True)
                out.write("\n")
    assert forbidden_count == 0
    return result


def test_profile_export_keeps_only_aggregate_implementation_values():
    root = Path("/invented/repo")
    raw = {
        ("/invented/repo/packages/example/src/module.py", 10, "check"): (
            2,
            3,
            0.1,
            0.2,
            {"private": "dropped"},
        ),
        ("/private/source/person-identifier.csv", 1, "secret"): (
            1,
            1,
            1.0,
            1.0,
            {"raw": "dropped"},
        ),
        ("~", 0, "<method 'update' of '_hashlib.HASH' objects>"): (
            1,
            1,
            0.01,
            0.01,
            {},
        ),
    }
    report = _profile_rows(raw, root)
    text = json.dumps(report)
    assert (
        "identifier" not in text
        and "secret" not in text
        and "private" not in text
        and "dropped" not in text
    )
    assert report["omitted_calls"] == 1
    assert report == _profile_rows(dict(reversed(list(raw.items()))), root)
    assert all(
        set(row)
        == {
            "file",
            "line",
            "function",
            "primitive_calls",
            "calls",
            "tottime",
            "cumtime",
        }
        for row in report["functions"]
    )


def test_profile_helper_restores_hook_and_never_exports_values(tmp_path, monkeypatch):
    monkeypatch.setenv("MICROCOSM_INVENTED_PROFILE_DIR", str(tmp_path))

    def ordinary():
        private_value = "never-export-this-local-value"
        return len(private_value)

    assert _profiled("tiny_success", ordinary) == len("never-export-this-local-value")
    assert sys.getprofile() is None
    text = (tmp_path / "tiny_success.json").read_text()
    assert "never-export-this-local-value" not in text
    assert json.loads(text)["succeeded"]
    with pytest.raises(AssertionError):
        _profiled("tiny_forbidden", ordinary, forbidden=(ordinary,))
    assert (
        json.loads((tmp_path / "tiny_forbidden.json").read_text())["forbidden_calls"]
        == 1
    )
    assert sys.getprofile() is None


def source_arguments(root, patch):
    import test_us_current_survey_immigration_transfer as original_fixture
    import test_us_survey_population_preparation as preparation_fixture
    from test_us_current_survey_health_coverage import (
        _health_acs_person,
        _health_asec_table,
    )
    from test_us_current_survey_hours_source import (
        add_hours_source_fields,
        hours_acs_person,
    )
    from test_us_current_survey_housing import add_housing_source_fields

    from microcosm.build.us_runtime import asec_person_income_source as restoration
    from microcosm.build.us_runtime import current_asec_demographics as demographics
    from microcosm.build.us_runtime import current_survey_person_status_source as status

    patch.setattr(
        preparation_fixture,
        "_person",
        hours_acs_person(_health_acs_person(preparation_fixture._person)),
    )
    call = original_fixture.source_arguments(root, patch)
    folder = call["source_dir"] / "asec"
    paths, pins = {}, []
    for (
        year,
        member,
        archive,
        *_,
    ) in (
        fragment.owner.assignment_owner.donor_owner.literals.original.asec._MEMBER_PINS
    ):
        path = folder / member
        raw = _health_asec_table(pd.read_csv(path, dtype=str, keep_default_na=False))
        for field in ("UC_VAL", "PHIP_VAL", "PMED_VAL", "POTC_VAL"):
            if field not in raw:
                raw[field] = "0"
        raw["UC_YN"] = [
            "0" if int(age) < 15 else "1" if float(amount) > 0 else "2"
            for age, amount in zip(raw.A_AGE, raw.UC_VAL, strict=True)
        ]
        raw.to_csv(path, index=False)
        data = path.read_bytes()
        pins.append(
            (
                year,
                member,
                archive,
                hashlib.sha256(data).hexdigest(),
                len(raw),
                len(data),
            )
        )
        paths[year] = path
    for module in (
        fragment.owner.assignment_owner.donor_owner.literals.original.asec,
        restoration,
        demographics.demographic,
        status.student,
    ):
        patch.setattr(module, "_MEMBER_PINS", tuple(pins))
    restored = root / "clone-health-restored"
    restoration.restore_asec_person_income_source(
        folder / "parent.h5",
        folder / "household-attachment.h5",
        member_paths=paths,
        output_dir=restored,
    )
    shutil.copyfile(
        restored / restoration.CHECKPOINT_FILENAME,
        folder / "person-income-attachment.h5",
    )
    add_hours_source_fields(call, patch)
    # All consumers retain the final invented raw roster, after the hours helper.
    final_pins = (
        fragment.owner.assignment_owner.donor_owner.literals.original.asec._MEMBER_PINS
    )
    for module in (demographics.demographic, status.student):
        patch.setattr(module, "_MEMBER_PINS", final_pins)
    return add_housing_source_fields(call, patch)


def test_source_fixture_composes_literals_before_issuance(tmp_path, monkeypatch):
    call = source_arguments(tmp_path, monkeypatch)
    assert call["fraction"] == 1 and call["household_roles"] and call["person_status"]
    paths = call["source_dir"] / "asec"
    pins = (
        fragment.owner.assignment_owner.donor_owner.literals.original.asec._MEMBER_PINS
    )
    for year, member, _, digest, count, size in pins:
        raw = (paths / member).read_bytes()
        assert len(raw) == size and hashlib.sha256(raw).hexdigest() == digest
        table = pd.read_csv(paths / member, dtype=str, keep_default_na=False)
        assert len(table) == count
        assert set(
            fragment.owner.assignment_owner.donor_owner.literals.ASEC_VALUE_COLUMNS
        ) <= set(table)
        assert set(graph.health_graph.health.ASEC_VALUE_COLUMNS) <= set(table)
        assert {"UC_YN", "HRSWK", "MARSUPWT"} <= set(table)
        if year == 2024:
            assert int(table.A_AGE.eq("15").sum()) == 1


@pytest.fixture(scope="module")
def originals(tmp_path_factory):
    import test_us_current_survey_immigration_transfer as original_fixture

    with pytest.MonkeyPatch.context() as patch:
        root = tmp_path_factory.mktemp("immigration-clone-host")
        _phase("sources.before")
        call = source_arguments(root, patch)
        _phase("sources.after")
        _phase("full51.before")
        financial = fragment.owner.host.run_atomic_survey_financial(**call)
        _phase("full51.after")
        original_fixture.zero_fixture_controls(patch)
        _phase("original_sources.before")
        donor = fragment.owner.assignment_owner.donor_owner.borrow_full_asec_immigration_donor(
            financial.prefix.preparation
        )
        assigned = fragment.owner.assignment_owner.assign_full_asec_immigration(
            donor, seed=42
        )
        acs = fragment.owner.acs_owner.borrow_current_acs_immigration_projection(
            financial.prefix.preparation
        )
        _phase("original_sources.after")
        _phase("original_transfer.before")
        transfer = _profiled(
            "original_constructor",
            lambda: fragment.owner.transfer_current_survey_immigration(
                financial,
                assigned,
                acs,
                seed=42,
                n_estimators=2,
                bank_root=root / "original-bank",
            ),
        )
        _phase("original_transfer.after")
        original_stamp = fragment.owner.allocation._population_stamp(
            financial.prefix.allocated_population
        )
        transfer_stamp = fragment.owner.source._frame_identity(transfer.frame)
        pair_stamp = fragment.pair_bytes(transfer.pairs)
        yield SimpleNamespace(
            root=root,
            call=call,
            financial=financial,
            transfer=transfer,
            original_stamp=original_stamp,
            transfer_stamp=transfer_stamp,
            pair_stamp=pair_stamp,
        )
        # Pure teardown cannot restore or reissue an invalidated handle.
        assert (
            fragment.owner.allocation._population_stamp(
                financial.prefix.allocated_population
            )
            == original_stamp
        )
        assert fragment.owner.source._frame_identity(transfer.frame) == transfer_stamp
        assert fragment.pair_bytes(transfer.pairs) == pair_stamp


def test_genuine_full51_clone_mapping_prefix(originals):
    """Declaration/artifact/mapping prefix only; no PUF or enrichment issuer."""
    from microcosm.graph import (
        ArtifactInput,
        KernelResult,
        artifact_edges,
        compile_graph,
    )
    from microcosm.graph import population as population_ops

    run, transfer = originals.financial, originals.transfer
    entry = fragment.retained_entry(transfer)
    assert entry[2].run is run
    receiving = run.financial_population
    before = fragment.owner.allocation._population_stamp(receiving)
    terminal = run.compiled.graph.node(fragment.owner.host._tax_module().GATE_NODE)
    assert run.manifest.population(receiving.version).person.person_id.equals(
        receiving.frame.person.person_id
    )
    output = terminal.artifact_outputs[0]
    assert output.name == "verification"
    assert output.type == fragment.owner.host._tax_module().VERIFICATION_TYPE
    after = ArtifactInput(
        "financial_verification", terminal.id, output.name, output.type
    )
    descriptor = run.manifest.node(terminal.id).typed_artifacts["outputs"][output.name]
    parent_artifact = artifact_edges.value_from_descriptor(
        run.store.load_bytes(descriptor["key"]), descriptor
    )
    _phase("clone_mapping.before")
    nodes = fragment.immigration_nodes(
        transfer, receiving.frame, receiving_version=receiving.version, after=after
    )
    compiled = compile_graph(
        replace(run.compiled.graph, nodes=(*run.compiled.graph.nodes, *nodes))
    )
    assert compiled.order[-2:] == (fragment.SOURCE_NODE, fragment.ATTACH_NODE)
    assert all(
        compiled.graph.node(n) == run.compiled.graph.node(n) for n in run.compiled.order
    )
    result = fragment.immigration_result(
        transfer, nodes[0], {after.name: parent_artifact}
    )
    assert not result.columns
    columns = fragment.read_pairs(result.artifacts["pairs"])
    pd.testing.assert_frame_equal(columns, fragment.canonical_pairs(transfer.pairs))
    # This is the exact production clone mapper on the genuine executor-observed
    # full51 receiving Population; no new allocation/clone/issuer is constructed.
    mapped = fragment.attachment.attach_columns(
        entry[2].acs_entry[2].qualified.origins, receiving.frame, columns
    )
    patched = population_ops.patch(receiving, nodes[1], KernelResult(columns=mapped))
    for entity in receiving.frame.entities:
        actual = patched.frame.table(entity)
        if entity == "person":
            actual = actual.drop(columns=list(fragment.owner.OUTPUTS))
        pd.testing.assert_frame_equal(
            actual, receiving.frame.table(entity), check_exact=True
        )
    for entity in receiving.frame.weighted_entities:
        assert patched.frame.weights_for(entity) is receiving.frame.weights_for(entity)
    assert patched.mass_ledger == receiving.mass_ledger
    assert patched.frame.mass_log == receiving.frame.mass_log
    assert patched.frame.metadata == receiving.frame.metadata
    for key, value in receiving.owners.items():
        assert patched.owners[key] == value
    support = fragment.attachment.provenance.support_source_id_column("person")
    for name in fragment.owner.OUTPUTS:
        values = patched.frame.person[name]
        assert values.dtype == fragment.STRING
        assert (
            values.tolist()
            == transfer.pairs[name].reindex(receiving.frame.person[support]).tolist()
        )
    params = nodes[0].params
    assert params["scope"] == "realized_original_pairs"
    assert not params["graph_fit_artifact_qualified"]
    assert (
        params["original_frame_sha256"]
        == json.loads(transfer.receipt)["original_frame_sha256"]
    )
    assert "original_population_sha256" not in params
    stamp = fragment.owner.allocation._population_stamp(patched)
    _phase("clone_mapping.after")
    _phase("original_validate.before")
    _profiled("original_explicit_validator", transfer.validate)
    _phase("original_validate.after")
    fragment.owner._pure(transfer, entry)
    assert fragment.owner.allocation._population_stamp(receiving) == before
    assert fragment.owner.allocation._population_stamp(patched) == stamp
    # A descriptive patch and compiled continuation are not retained host owners.
    with pytest.raises(ValueError, match="UNISSUED_RUN"):
        graph.check_survey_enrichment_run(patched)


@pytest.fixture(scope="module")
def genuine(originals):
    """Authored complete host acceptance; excluded from bounded prefix admission."""
    from test_us_graph_puf55_canonical_donor import _original_sources

    root, call, financial, transfer = (
        originals.root,
        originals.call,
        originals.financial,
        originals.transfer,
    )
    _phase("puf.before")
    definition, paths, _ = _original_sources(
        root / "invented-puf", full_finalization_support=True
    )
    parent = graph.parent.run_survey_puf55(
        financial,
        donor_sources=paths,
        fixture_definition=definition,
        seed=578,
        n_estimators=2,
        zero_atol=0,
    )
    _phase("puf.after")
    _phase("enrichment_disabled.before")
    disabled = graph.run_us_survey_enrichment(parent, n_estimators=2)
    _phase("enrichment_disabled.after")
    _phase("enrichment_enabled.before")
    enabled = graph.run_us_survey_enrichment(
        parent, n_estimators=2, immigration_transfer=transfer
    )
    _phase("enrichment_enabled.after")
    _phase("enrichment_required.before")
    required = graph.run_us_survey_enrichment(
        parent, n_estimators=2, immigration_transfer=transfer, resume="require"
    )
    _phase("enrichment_required.after")
    return SimpleNamespace(
        root=root,
        call=call,
        financial=financial,
        parent=parent,
        transfer=transfer,
        disabled=disabled,
        enabled=enabled,
        required=required,
        original_stamp=originals.original_stamp,
        transfer_stamp=originals.transfer_stamp,
        pair_stamp=originals.pair_stamp,
    )


def test_genuine_clone_attachment_required_replay_and_nonowned_parity(genuine):
    state = fragment.retained_entry(genuine.transfer)[2]
    disabled, enabled, required = genuine.disabled, genuine.enabled, genuine.required
    assert not set((fragment.SOURCE_NODE, fragment.ATTACH_NODE)) & set(
        disabled.compiled.order
    )
    assert enabled.manifest.key == required.manifest.key
    assert all(record.hit for record in required.manifest.nodes.values())
    assert enabled.compiled.order[-2:] == (fragment.SOURCE_NODE, fragment.ATTACH_NODE)
    assert (
        tuple(
            n
            for n in enabled.compiled.order
            if n not in (fragment.SOURCE_NODE, fragment.ATTACH_NODE)
        )
        == disabled.compiled.order
    )
    for node_id in disabled.compiled.order:
        assert enabled.compiled.graph.node(node_id) == disabled.compiled.graph.node(
            node_id
        )
        assert enabled.manifest.node(node_id).key == disabled.manifest.node(node_id).key
    for entity in disabled.population.frame.entities:
        expected = disabled.population.frame.table(entity)
        actual = enabled.population.frame.table(entity)
        if entity == "person":
            actual = actual.drop(columns=list(fragment.owner.OUTPUTS))
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    graph.physical.replay.same_replayed_population(
        enabled.population, required.population
    )
    assert enabled.population.mass_ledger == disabled.population.mass_ledger
    assert enabled.population.frame.mass_log == disabled.population.frame.mass_log
    assert enabled.population.frame.metadata == disabled.population.frame.metadata
    for entity in disabled.population.frame.weighted_entities:
        left, right = (
            enabled.population.frame.weights_for(entity),
            disabled.population.frame.weights_for(entity),
        )
        assert left.kind is right.kind
        pd.testing.assert_series_equal(pd.Series(left.values), pd.Series(right.values))
    expected = fragment.attachment.attach_columns(
        state.acs_entry[2].qualified.origins,
        disabled.population.frame,
        fragment.canonical_pairs(genuine.transfer.pairs),
    )
    for (_entity, name), series in expected.items():
        pd.testing.assert_series_equal(
            enabled.population.frame.person.set_index("person_id")[name], series
        )
    params = enabled.compiled.graph.node(fragment.SOURCE_NODE).params
    assert set(params) == {
        "protocol",
        "transfer_receipt_sha256",
        "pairs_sha256",
        "original_frame_sha256",
        "original_population_version",
        "original_rows",
        "receiving_rows",
        "scope",
        "graph_fit_artifact_qualified",
    }
    assert (
        params["original_frame_sha256"]
        == json.loads(genuine.transfer.receipt)["original_frame_sha256"]
    )
    assert "original_population_sha256" not in params
    receipt = json.loads(enabled.receipt)
    assert receipt["immigration"]["enabled"]
    assert not receipt["immigration"]["graph_fit_artifact_qualified"]
    assert not receipt["immigration"]["national_stock_alignment_qualified"]
    assert not receipt["release_eligible"]
    _phase("original_validate.before")
    genuine.transfer.validate()
    _phase("original_validate.after")
    _phase("enriched_validate.before")
    assert enabled.checked_view().population is enabled.population
    _phase("enriched_validate.after")


def test_genuine_pair_artifact_and_producer_controls(genuine):
    from microcosm.graph import artifact_edges

    run = genuine.enabled
    node = run.compiled.graph.node(fragment.ATTACH_NODE)
    descriptors = run.manifest.node(node.id).typed_artifacts["inputs"]
    artifacts = {
        edge.name: artifact_edges.value_from_descriptor(
            run.store.load_bytes(descriptors[edge.name]["key"]), descriptors[edge.name]
        )
        for edge in node.artifact_inputs
    }
    detached = {
        **artifacts,
        "immigration_pairs": replace(artifacts["immigration_pairs"], payload=b"{}"),
    }
    with pytest.raises(ValueError, match="PAIR_ARTIFACT_CHANGED"):
        fragment.immigration_result(
            genuine.transfer, node, detached, genuine.disabled.population.frame.person
        )
    detached = {
        **artifacts,
        "immigration_pairs": replace(
            artifacts["immigration_pairs"], producer_key="f" * 64
        ),
    }
    boundary = graph._ISSUED[id(run)][1]
    with pytest.raises(ValueError, match="ARTIFACT_PRODUCER_KEY"):
        boundary.context(SimpleNamespace(node=node, sources={}, artifacts=detached))


def test_genuine_host_refuses_copied_transfer_before_attachment(genuine):
    with pytest.raises(ValueError, match="ISSUED_TRANSFER_REQUIRED"):
        graph.run_us_survey_enrichment(
            genuine.parent,
            n_estimators=2,
            immigration_transfer=replace(genuine.transfer),
        )


def test_genuine_development_storage_retains_typed_pair_without_release_claim(genuine):
    from microcosm.build.us_runtime import native_survey_handoff as handoff

    written = handoff.write_native_survey_development_checkpoint(
        genuine.enabled, genuine.root / "handoff"
    )
    assert written.owner_live_verified
    assert not written.report["release_eligible"]
    restored = handoff.load_native_survey_development_checkpoint(
        genuine.root / "handoff"
    )
    assert not restored.owner_live_verified
    for name in fragment.owner.OUTPUTS:
        pd.testing.assert_series_equal(
            restored.frame.person[name], genuine.enabled.population.frame.person[name]
        )


def test_final_foreign_io_mutation_revokes_enriched_owner_last(genuine):
    """Observe an actual return boundary; no validator/function is replaced."""
    run = genuine.required
    entry = graph._ISSUED[id(run)]
    boundary = entry[1]
    frame = run.population.frame.person
    name = fragment.owner.OUTPUTS[0]
    value = frame.at[frame.index[0], name]
    previous = sys.getprofile()
    assert previous is None
    fired = False
    validations = 0

    def observe(current, event, argument):
        nonlocal fired, validations
        if (
            not fired
            and event == "return"
            and current.f_code
            is fragment.owner.CurrentSurveyImmigrationTransfer.validate.__code__
            and current.f_back is not None
            and current.f_back.f_code is graph.Boundary.borrow.__code__
            and current.f_back.f_locals.get("self") is boundary
        ):
            validations += 1
            # check_survey_enrichment_run borrows before and after artifact I/O.
            # Mutate only on the second, final foreign-owner return boundary.
            if validations == 2:
                fired = True
                frame.at[frame.index[0], name] = (
                    "NONE" if value != "NONE" else "CITIZEN"
                )

    try:
        sys.setprofile(observe)
        with pytest.raises(ValueError, match="RUN_CHANGED"):
            run.checked_view()
    finally:
        sys.setprofile(previous)
        frame.at[frame.index[0], name] = value
    assert fired and validations == 2
    with pytest.raises(ValueError, match="UNISSUED_RUN"):
        run.checked_view()
