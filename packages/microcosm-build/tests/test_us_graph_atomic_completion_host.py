"""Source-authored optional host acceptance. All cases are UNRUN in this task.

One shared invented source fixture supplies actual ACS/ASEC owners and the real
compiler, store, executor and numeric kernels. No owner or runtime is replaced.
"""

import ast
import csv
import hashlib
import io
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_us_child_property_income_graph import _options
from test_us_graph_atomic_survey_population import _support_payload

from microcosm.build.us_runtime import graph_atomic_survey_financial as host
from microcosm.build.us_runtime import graph_child_property_income as child
from microcosm.build.us_runtime import graph_property_tax_leaves as tax
from microcosm.build.us_runtime import graph_survey_completion as receiving
from microcosm.build.us_runtime import graph_survey_completion_host as extension


def _source_arguments(
    root, patch, *, incomplete_adult=False, household_roles=True, person_status=True
):
    """Compose all literals and restore attachments before any host is issued."""
    import test_us_child_property_income_source_owner as children
    import test_us_current_asec_demographics as demographics
    import test_us_current_asec_income_routing as routing
    import test_us_survey_population_preparation as preparation
    from test_us_current_survey_person_status_source import (
        _source_arguments as statuses,
    )
    from test_us_survey_social_security import _social_security_arguments

    from microcosm.build.us_runtime import asec_person_income_source as restoration
    from microcosm.build.us_runtime import (
        current_survey_person_status_source as status_source,
    )

    original_build = preparation.build_fixture

    def supported_housing(*args, **kwargs):
        # GQ reference roles are deliberately unsupported. This positive fixture
        # contains only genuine housing reference-person observations; no GQ
        # incumbent is erased or given a fabricated role after issuance.
        for kind in ("household_members", "person_members"):
            members = []
            for member in kwargs[kind]:
                rows = list(csv.DictReader(io.StringIO(member.data.decode())))
                rows = [row for row in rows if "GQ" not in row["SERIALNO"]]
                if kind == "person_members":
                    for row in rows:
                        if int(row["AGEP"]) >= 15:
                            row["INTP"] = (
                                "bad"
                                if incomplete_adult
                                and row["SERIALNO"] == "2024HU0000002"
                                else "100"
                            )
                members.append(replace(member, data=preparation._csv(rows)))
            kwargs[kind] = tuple(members)
        return original_build(*args, **kwargs)

    patch.setattr(preparation, "build_fixture", supported_housing)

    def social(path, monkeypatch, *, zero):
        assert zero is False
        return _social_security_arguments(path, monkeypatch, ambiguous=False)

    def status(path, monkeypatch, *, zero):
        assert zero is False
        return statuses(path, monkeypatch)

    patch.setattr(demographics, "fixture", social)
    patch.setattr(preparation, "fixture", status)
    # Either selected ASEC household remains a valid property donor. The child
    # donor itself remains the one full-source teenager outside the selection.
    patch.setattr(routing, "AMOUNTS", {**routing.AMOUNTS, 107: routing.AMOUNTS[105]})
    patch.setattr(
        routing, "LITERALS", {**routing.LITERALS, 107: dict(routing.LITERALS[105])}
    )
    original_change = children._change_unissued_literals

    def final_literals(arguments, monkeypatch, donor, changes):
        original_change(arguments, monkeypatch, donor, changes)
        folder = arguments["source_dir"] / "asec"
        pins, paths = [], {}
        for year, member, archive, *_ in child.child.routing.coverage._MEMBER_PINS:
            path = folder / member
            table = pd.read_csv(path, dtype=str, keep_default_na=False)
            if year == 2024:
                adults = table.A_AGE.astype(int).ge(15)
                table.loc[adults, "SUR_YN"] = "2"
                table.loc[adults, "SUR_SC2"] = "0"
                table.to_csv(path, index=False)
            data = path.read_bytes()
            pins.append(
                (
                    year,
                    member,
                    archive,
                    hashlib.sha256(data).hexdigest(),
                    len(table),
                    len(data),
                )
            )
            paths[year] = path
        for module in (
            child.child.routing.coverage,
            restoration,
            demographics.owner.demographic,
            status_source.student,
        ):
            monkeypatch.setattr(module, "_MEMBER_PINS", tuple(pins))
        restored = root / "completion-restored-money"
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

    patch.setattr(children, "_change_unissued_literals", final_literals)
    full, partial, donor = children.source_arguments(root, patch)
    # These adapters only compose unissued sources. Release them before issuing
    # a run so a later independently selected fixture cannot wrap this closure
    # again and overwrite its own literals or reuse this fixture's output path.
    patch.setattr(preparation, "build_fixture", original_build)
    patch.setattr(children, "_change_unissued_literals", original_change)
    arguments = full if incomplete_adult else partial
    payload, source_ids = _support_payload()
    support = root / "invented-block-support.npz"
    support.write_bytes(payload)
    config = host.reconstruction.AtomicSurveyReconstruction(
        support_path=str(support),
        support_sha256=hashlib.sha256(payload).hexdigest(),
        source_ids=tuple(sorted(source_ids.items())),
        seed=17,
    )
    options = host._property_module().PropertyIncomeOptions(
        scales=(1.0, 1.0, 1.0, 1.0),
        atol=1e-10,
        rtol=1e-12,
        n_estimators=2,
        completion_routing=True,
    )
    return dict(
        **arguments,
        store_root=root / "store",
        geography_config=config,
        demographic_conditioning=True,
        n_estimators=2,
        property_income=options,
        rebase_property_taxes=True,
        person_status=person_status,
        household_roles=household_roles,
        child_property=_options(only_positive=True),
        return_values=True,
    ), donor


@pytest.fixture(scope="module")
def completion_host(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        call, donor = _source_arguments(
            tmp_path_factory.mktemp("completion-host"), patch
        )
        cold = host.run_atomic_survey_financial(**call)
        warm = host.run_atomic_survey_financial(**call, resume="require")
        yield SimpleNamespace(cold=cold, warm=warm, call=call, donor=donor)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"household_roles": 1}, "HOUSEHOLD_ROLES_FLAG"),
        ({"household_roles": None}, "HOUSEHOLD_ROLES_FLAG"),
        ({"household_roles": True}, "ROLES_REQUIRE_CHILD"),
        ({"child_property": object()}, "CHILD_REQUIRES_PROPERTY"),
        (
            {"child_property": object(), "property_income": object()},
            "CHILD_REQUIRES_TAX_REBASE",
        ),
        (
            {
                "child_property": object(),
                "property_income": object(),
                "rebase_property_taxes": True,
            },
            "CHILD_OPTIONS_TYPE",
        ),
    ],
)
def test_refused_options_precede_source_io(changes, reason):
    with pytest.raises(ValueError, match=reason):
        host.run_atomic_survey_financial(
            None,
            snapshot_root=None,
            store_root=None,
            fraction=1,
            seed=17,
            geography_config=None,
            **changes,
        )


def _assert_all_row_lineage(run, *, household_roles, person_status):
    boundary = host._run_entry(run)[2].completion_boundary
    order = run.compiled.order
    writers = [
        *boundary.base.prefix.compiled.order,
        host.financial.ATTACH_NODE,
        host._property_module().ATTACH_NODE,
    ]
    if person_status:
        writers.append(host._person_status_module().BIND_NODE)
    assert all(order.index(n) < order.index(receiving.NODE) for n in writers)
    base = boundary.base.financial_population
    received = boundary.observed[receiving.NODE]
    completed = boundary.observed[child.VERIFY]
    tax_received = boundary.observed[tax.RECEIVING_NODE]
    for before, after, node_id in (
        (base, received, receiving.NODE),
        (completed, tax_received, tax.RECEIVING_NODE),
    ):
        node = run.compiled.graph.node(node_id)
        assert node.base == before.version
        assert after.version == node_id
        assert before.version != after.version
        assert before.frame.entities == after.frame.entities
        assert before.frame.schema == after.frame.schema
        for entity in before.frame.entities:
            assert before.frame.table(entity).shape == after.frame.table(entity).shape
            pd.testing.assert_frame_equal(
                before.frame.table(entity), after.frame.table(entity), check_exact=True
            )
        pd.testing.assert_series_equal(before.frame.strata, after.frame.strata)
        for entity in before.design_weights:
            np.testing.assert_array_equal(
                before.design_weights[entity], after.design_weights[entity]
            )
        for entity in before.frame.weighted_entities:
            np.testing.assert_array_equal(
                before.frame.weights_for(entity).values,
                after.frame.weights_for(entity).values,
            )
        assert set(after.owners) == set(before.owners)
        assert set(after.owners.values()) == {node_id}
        assert after.mass_ledger[:-1] == before.mass_ledger
        record = after.mass_ledger[-1]
        assert (record.node_id, record.operation, record.policy) == (
            node_id,
            "filter",
            "conserve",
        )
        assert record.before_total == record.after_total
        assert record.before_by_stratum == record.after_by_stratum
        assert record.before_by_partition_stratum == record.after_by_partition_stratum
    assert len(run.financial_population.mass_ledger) == len(base.mass_ledger) + 2
    assert boundary.parent.version == completed.version == receiving.NODE
    assert received.owners["person", child.child.TARGETS[0]] == receiving.NODE
    for name in child.child.TARGETS:
        assert base.owners["person", name] == host._property_module().ATTACH_NODE
        assert completed.owners["person", name] == child.ATTACH
    if household_roles:
        roles = extension._roles()
        assert (
            boundary.parent.owners["person", roles.roles.CANONICAL_COLUMN]
            == roles.BIND_NODE
        )
    if person_status:
        status = host._person_status_module()
        bind = run.compiled.graph.node(status.BIND_NODE)
        for output in bind.outputs:
            cell = output.entity, output.column
            assert base.owners[cell] == status.BIND_NODE
            assert received.owners[cell] == receiving.NODE
            assert completed.owners[cell] == receiving.NODE
            assert tax_received.owners[cell] == tax.RECEIVING_NODE
    assert run.financial_population.version == tax.RECEIVING_NODE
    for name in tax.TAX_LEAF_COLUMNS:
        assert run.financial_population.owners["person", name] == tax.TAX_LEAVES_NODE


def _roles_disabled_pair(root, patch, *, person_status, count):
    # Explicit partial ACS+ASEC selection from the same pre-issuance source
    # composition: the full-source teenage donor is outside the selected sample.
    call, _ = _source_arguments(
        root, patch, household_roles=False, person_status=person_status
    )
    cold = host.run_atomic_survey_financial(**call)
    warm = host.run_atomic_survey_financial(**call, resume="require")
    assert cold.compiled.graph == warm.compiled.graph
    assert cold.manifest.key == warm.manifest.key
    assert cold.checked_view().payload == warm.checked_view().payload
    assert (
        host._run_entry(cold)[2].artifact_hashes
        == host._run_entry(warm)[2].artifact_hashes
    )
    roles = extension._roles()
    for run in (cold, warm):
        state = host._run_entry(run)[2]
        boundary = state.completion_boundary
        assert len(run.compiled.order) == count
        assert len(boundary.base.compiled.order) == count - 10
        assert (state.person_status_boundary is not None) is person_status
        assert boundary.household_roles is False
        assert boundary.role_nodes == () and boundary.role_qualified is None
        assert boundary.parent is boundary.receiving
        assert roles.roles.CANONICAL_COLUMN not in boundary.owned_columns()
        assert {roles.SOURCE_NODE, roles.BIND_NODE}.isdisjoint(run.compiled.order)
        assert all(
            output.column != roles.roles.CANONICAL_COLUMN
            for node in run.compiled.graph.nodes
            if node.id not in boundary.base.compiled.order
            for output in node.outputs
        )
        before = boundary.base.financial_population.frame.person
        after = run.financial_population.frame.person
        assert (roles.roles.CANONICAL_COLUMN in before) == (
            roles.roles.CANONICAL_COLUMN in after
        )
        if roles.roles.CANONICAL_COLUMN in before:
            pd.testing.assert_series_equal(
                before[roles.roles.CANONICAL_COLUMN],
                after[roles.roles.CANONICAL_COLUMN],
                check_exact=True,
            )
        document = json.loads(run.checked_view().payload)
        assert document["household_roles"] is False
        assert document["household_role_node_count"] == 0
        assert document["child_property_node_count"] == 6
        assert document["child_source_knownness_changed"] is False
        assert json.loads(state.tax_verification)["complete"] is True
        _assert_all_row_lineage(run, household_roles=False, person_status=person_status)
    base_order = host._run_entry(cold)[2].completion_boundary.base.compiled.order
    assert all(cold.manifest.node(n).hit for n in base_order)
    assert all(
        not cold.manifest.node(n).hit
        for n in cold.compiled.order
        if n not in base_order
    )
    assert all(record.hit for record in warm.manifest.nodes.values())
    return cold, warm


def test_roles_disabled_status_enabled_49_cold_required_and_output_identity(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        cold, warm = _roles_disabled_pair(tmp_path, patch, person_status=True, count=49)
        boundary = host._run_entry(warm)[2].completion_boundary
        original = boundary.completed
        boundary.completed = replace(original)
        assert boundary.completed is not original
        assert host.reconstruction._population_stamp(
            boundary.completed
        ) == host.reconstruction._population_stamp(original)
        with pytest.raises(ValueError, match="COMPLETION_OUTPUT_IDENTITY$"):
            boundary.pure()
        boundary.completed = original
        assert boundary.revoked and boundary.child.revoked
        with pytest.raises(ValueError, match="COMPLETION_HOST_REVOKED$"):
            boundary.pure()
        boundary.base.checked_view()
        cold.checked_view()


def _assert_private_support_custody(run):
    """Actual executor observations and manifest views keep distinct full seals."""
    state = host._run_entry(run)[2]
    boundary = state.completion_boundary
    retained = dict(zip(run.compiled.order, state.node_populations, strict=True))
    for node_id in (child.DONOR, child.RECIPIENT, child.FIT, child.DRAW):
        actual = boundary.observed[node_id]
        expected_version = (
            child.DONOR if node_id in (child.DONOR, child.FIT) else child.RECIPIENT
        )
        assert actual.version == run.compiled.versions[node_id] == expected_version
        stamp = extension._population_stamp(boundary, run.compiled, node_id, actual)
        assert stamp == dict(boundary.observed_stamps)[node_id]
        assert retained[node_id][0] is actual and retained[node_id][1] == stamp
        assert stamp[1] == child.child.physical._population_stamp(actual)
        with pytest.raises(ValueError, match="^FRAME_TYPE$"):
            host.reconstruction._population_stamp(actual)
    for version in (child.DONOR, child.RECIPIENT):
        attached = run.manifest.population(version)
        assert type(attached) is extension.PopulationView
        assert attached is not boundary.observed[version].frame
        actual = host.Population.from_frame(
            attached, version, mass_ledger=run.manifest.mass_ledger(version)
        )
        assert (
            extension._population_stamp(
                boundary, run.compiled, version, actual, manifest=True
            )
            == dict(state.manifest_populations)[version]
        )


def test_roles_disabled_status_disabled_45_cold_required(tmp_path):
    with pytest.MonkeyPatch.context() as patch:
        cold, warm = _roles_disabled_pair(
            tmp_path, patch, person_status=False, count=45
        )
        for run in (cold, warm):
            _assert_private_support_custody(run)
        # All original positive cold/required checks finish before either issued
        # handle is deliberately poisoned. The observer is detached from the
        # manifest; both actual custody paths need their own refusal.
        for run, surface, reason in (
            (cold, "observed_fit", "COMPLETION_OBSERVED_CHANGED"),
            (warm, "manifest_recipient", "FINANCIAL_RUN_ATTACHED_POPULATION_CHANGED"),
        ):
            boundary = host._run_entry(run)[2].completion_boundary
            frame = (
                boundary.observed[child.FIT].frame
                if surface == "observed_fit"
                else run.manifest.population(child.RECIPIENT)
            )
            original = frame.person["source_year"].copy(deep=True)
            frame.person.loc[0, "source_year"] += 1
            try:
                with pytest.raises(
                    ValueError, match="^CURRENT_SURVEY_PREDICTOR_" + reason + "$"
                ):
                    run.checked_view()
            finally:
                frame.person["source_year"] = original
            assert boundary.revoked and boundary.child.revoked
            with pytest.raises(
                ValueError, match="^CURRENT_SURVEY_PREDICTOR_COMPLETION_HOST_REVOKED$"
            ):
                run.checked_view()
            boundary.base.checked_view()


def test_disabled_signature_and_serialized_option_parity(completion_host):
    tree = ast.parse(Path(host.__file__).read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_atomic_survey_financial"
    )
    defaults = dict(
        zip(
            (a.arg for a in function.args.kwonlyargs),
            function.args.kw_defaults,
            strict=True,
        )
    )
    assert ast.literal_eval(defaults["household_roles"]) is False
    assert ast.literal_eval(defaults["child_property"]) is None
    bases = [
        host._run_entry(run)[2].completion_boundary.base
        for run in (completion_host.cold, completion_host.warm)
    ]
    assert bases[0].compiled.graph == bases[1].compiled.graph
    assert bases[0].checked_view().payload == bases[1].checked_view().payload
    for base in bases:
        document = json.loads(base.checked_view().payload)
        assert len(base.compiled.order) == 39
        assert (
            not {"child_property", "household_roles", "child_property_node_count"}
            & document.keys()
        )
        assert host._run_entry(base)[2].completion_boundary is None


def test_exact_51_roster_order_and_actual_required_hits(completion_host):
    cold, warm = completion_host.cold, completion_host.warm
    boundary = host._run_entry(cold)[2].completion_boundary
    expected = (
        *boundary.base.compiled.graph.nodes,
        boundary.receiving_node,
        *boundary.role_nodes,
        *boundary.child.nodes,
        *host._tax_nodes(
            boundary.completed.frame,
            boundary.completed.version,
            host._run_entry(cold)[2].property_income,
            completion=boundary.tax_edge(),
        ),
    )
    assert tuple(cold.compiled.graph.nodes) == expected
    assert len(expected) == len(cold.compiled.order) == len(warm.compiled.order) == 51
    assert all(cold.manifest.node(n).hit for n in boundary.base.compiled.order)
    assert all(not cold.manifest.node(n.id).hit for n in expected[39:])
    assert all(record.hit for record in warm.manifest.nodes.values())
    assert cold.manifest.key == warm.manifest.key
    assert cold.checked_view().payload == warm.checked_view().payload
    _assert_all_row_lineage(cold, household_roles=True, person_status=True)
    _assert_all_row_lineage(warm, household_roles=True, person_status=True)
    order = cold.compiled.order
    assert (
        order.index(receiving.NODE)
        < order.index(extension._roles().BIND_NODE)
        < order.index(child.ATTACH)
    )
    assert (
        order.index(child.VERIFY)
        < order.index(tax.RECEIVING_NODE)
        < order.index(tax.GATE_NODE)
    )
    assert cold.compiled.graph.node(child.ATTACH).population == receiving.NODE
    assert cold.compiled.graph.node(tax.RECEIVING_NODE).base == receiving.NODE
    for name in (tax.RECEIVING_NODE, tax.TAX_LEAVES_NODE, tax.GATE_NODE):
        assert boundary.tax_edge() in cold.compiled.graph.node(name).artifact_inputs
    assert set(boundary.base.kernels.refs()).isdisjoint(
        {node.kernel for node in expected[39:]}
    )
    boundary.base.checked_view()


def test_actual_donor_outside_selection_and_paired_child_draws(completion_host):
    run = completion_host.cold
    boundary = host._run_entry(run)[2].completion_boundary
    donor = boundary.observed[child.DONOR].frame.person
    assert len(donor) == 1
    selected = boundary.child.qualified.recipients
    selected_people = boundary.base.prefix.preparation.checked_view().frame.person
    asec = selected_people[child.provenance.support_channel_column("person")].eq("asec")
    assert not selected_people.loc[asec, "A_AGE"].between(15, 17).any()
    assert donor[child.WEIGHT].iloc[0] > 0
    output = boundary.completed.frame.person
    original_id = child.provenance.support_source_id_column("person")
    eligible = selected.index[selected.eligible]
    assert len(eligible) > 0
    for identity in eligible:
        pair = output.loc[output[original_id].eq(identity)]
        assert len(pair) == 2 and pair[child.IMPUTED].all()
        np.testing.assert_array_equal(
            pair[list(child.child.TARGETS)].iloc[0],
            pair[list(child.child.TARGETS)].iloc[1],
        )
        assert pair[child.STATUS].eq("imputed_positive").all()
    # Full nonrecipient and source-known O/D preserve storage and null masks.
    before = boundary.parent.frame.person
    unchanged = ~output[child.IMPUTED]
    for name in child.child.TARGETS:
        pd.testing.assert_series_equal(
            before.loc[unchanged, name], output.loc[unchanged, name], check_exact=True
        )
    for name in before:
        if name not in child.child.TARGETS:
            pd.testing.assert_series_equal(before[name], output[name], check_exact=True)


def test_full_weights_geography_status_roles_and_complete_gate(completion_host):
    run = completion_host.warm
    state = host._run_entry(run)[2]
    boundary = state.completion_boundary
    before, after = boundary.parent, run.financial_population
    for entity in before.frame.entities:
        unchanged = [
            c
            for c in before.frame.table(entity)
            if c not in (*child.child.TARGETS, *tax.TAX_LEAF_COLUMNS)
        ]
        pd.testing.assert_frame_equal(
            before.frame.table(entity)[unchanged],
            after.frame.table(entity)[unchanged],
            check_exact=True,
        )
    pd.testing.assert_series_equal(before.frame.strata, after.frame.strata)
    assert before.frame.schema == after.frame.schema
    for entity in before.design_weights:
        np.testing.assert_array_equal(
            before.design_weights[entity], after.design_weights[entity]
        )
        np.testing.assert_array_equal(
            before.frame.weights_for(entity).values,
            after.frame.weights_for(entity).values,
        )
    assert json.loads(state.tax_verification)["complete"] is True
    host.require_complete_property_taxes(run)
    assert host.financial_output_node(run) == tax.GATE_NODE
    assert (
        json.loads(run.checked_view().payload)["property_completion_routing_scope"]
        == "pre_child_source_gaps"
    )
    document = json.loads(run.checked_view().payload)
    verification = boundary.payloads[child.VERIFY, "verification"]
    assert (
        document["child_verification_sha256"]
        == hashlib.sha256(verification).hexdigest()
    )
    assert (
        document["child_source_knownness_changed"]
        is json.loads(verification)["source_knownness_changed"]
    )
    assert (
        child.VERIFY,
        "verification",
        document["child_verification_sha256"],
    ) in state.artifact_hashes


def test_detached_parent_refuses_without_owner_substitution(completion_host):
    run = completion_host.cold
    base = host._run_entry(run)[2].completion_boundary.base
    with pytest.raises(ValueError, match="UNISSUED_FINANCIAL_RUN"):
        extension.extend(
            replace(base),
            options=completion_host.call["child_property"],
            household_roles=True,
            resume="require",
        )
    with pytest.raises(ValueError, match="UNISSUED_FINANCIAL_RUN"):
        replace(run).checked_view()
    base.checked_view()


def test_actual_tax_rebased_parent_refuses(completion_host):
    run = completion_host.cold
    with pytest.raises(ValueError, match="COMPLETION_ACTUAL_PROPERTY_PARENT$"):
        extension.extend(
            run,
            options=completion_host.call["child_property"],
            household_roles=True,
            resume="require",
        )
    run.checked_view()


@pytest.mark.parametrize("defect", ["compensating_child_tax", "receiving", "roles"])
def test_wrong_fragment_roster_refuses_before_compile(completion_host, defect):
    run = completion_host.cold
    boundary = host._run_entry(run)[2].completion_boundary
    tax_nodes = host._tax_nodes(
        boundary.completed.frame,
        boundary.completed.version,
        host._run_entry(run)[2].property_income,
        completion=boundary.tax_edge(),
    )
    original = boundary.receiving_node, boundary.role_nodes, boundary.child.nodes
    assert (
        extension._completion_nodes(boundary, tax_nodes)
        == tuple(run.compiled.graph.nodes)[39:]
    )
    if defect == "compensating_child_tax":
        # Preserve the union count while moving a real tax node into child7.
        boundary.child.nodes = (*boundary.child.nodes, tax_nodes[0])
        tax_nodes = tax_nodes[1:]
    elif defect == "receiving":
        boundary.receiving_node = replace(boundary.receiving_node, id="wrong.receiving")
    else:
        boundary.role_nodes = tuple(reversed(boundary.role_nodes))
    try:
        with pytest.raises(ValueError, match="COMPLETION_FRAGMENT_ROSTER$"):
            extension._completion_nodes(boundary, tax_nodes)
    finally:
        boundary.receiving_node, boundary.role_nodes, boundary.child.nodes = original
    boundary.pure()


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("base_miss", "COMPLETION_BASE_MUST_HIT"),
        ("missing_observation", "COMPLETION_OBSERVER_ROSTER"),
    ],
)
def test_actual_union_observation_and_base_cache_refusals(
    completion_host, defect, reason
):
    run = completion_host.warm
    boundary = host._run_entry(run)[2].completion_boundary
    extension._check_observed_union(
        boundary.base, run.compiled, run.manifest, boundary.observed
    )
    manifest, observed = run.manifest, dict(boundary.observed)
    if defect == "base_miss":
        node_id = boundary.base.compiled.order[-1]
        manifest = replace(
            manifest,
            nodes={
                **manifest.nodes,
                node_id: replace(manifest.node(node_id), hit=False),
            },
        )
        # Hits are deliberately outside manifest content identity; the host
        # must enforce this separate staged-execution invariant itself.
        assert manifest.key == run.manifest.key
    else:
        observed.pop(child.VERIFY)
    with pytest.raises(ValueError, match=reason + "$"):
        extension._check_observed_union(boundary.base, run.compiled, manifest, observed)
    boundary.pure()


@pytest.mark.parametrize(
    "defect", ["source_key", "node_key", "implementation", "artifact"]
)
def test_actual_union_final_io_identity_refusals_without_execution(
    completion_host, defect
):
    run = completion_host.warm
    state = host._run_entry(run)[2]
    source_keys, keys, implementations = (
        dict(state.source_keys),
        dict(state.keys),
        dict(state.implementations),
    )
    loaded = host._artifacts(
        run.manifest, run.compiled, run.store, run.kernels, keys, implementations
    )
    expected = dict(
        source_keys=source_keys,
        keys=keys,
        implementations=implementations,
        loaded=loaded,
    )
    assert extension._checked_final_io(run, **expected) == (source_keys, loaded)
    # Change a retained pre-I/O identity; the maintained boundary re-reads the
    # authentic source/store/implementations. No runner or issuer is replaced.
    selected = expected[
        {
            "source_key": "source_keys",
            "node_key": "keys",
            "implementation": "implementations",
            "artifact": "loaded",
        }[defect]
    ]
    key = next(iter(selected))
    selected[key] = (
        b"changed retained payload"
        if defect == "artifact"
        else "changed retained identity"
    )
    with pytest.raises(ValueError, match="COMPLETION_FINAL_IO_CHANGED$"):
        extension._checked_final_io(run, **expected)
    state.completion_boundary.pure()


@pytest.mark.parametrize("field", ["scenario_id", "stream", "transport"])
def test_explicit_scientific_identity_changes_declarations(completion_host, field):
    boundary = host._run_entry(completion_host.cold)[2].completion_boundary
    options = boundary.options
    replacement = {
        "scenario_id": "another-explicit-scenario",
        "stream": (*options.stream[:-1], options.stream[-1] + 1),
        "transport": replace(options.transport, receipt_factor=0.0),
    }[field]
    changed = replace(options, **{field: replacement})
    nodes = child.child_property_nodes(
        boundary.child.qualified,
        boundary.child.origins,
        boundary.parent,
        options=changed,
        host_edges=boundary.host_edges,
        host_pins=boundary.host_pins,
    )
    assert tuple(n.normative() for n in nodes) != boundary.child.declaration


def test_incomplete_adult_refuses_actual_puf_qualification_before_fitting(tmp_path):
    from microcosm.build.us_runtime import graph_puf55_survey_recipients as recipients

    with pytest.MonkeyPatch.context() as patch:
        call, _ = _source_arguments(tmp_path, patch, incomplete_adult=True)
        run = host.run_atomic_survey_financial(**call)
        state = host._run_entry(run)[2]
        completed = state.completion_boundary.completed.frame.person
        assert (
            completed.loc[~completed[child.IMPUTED], list(child.child.TARGETS)]
            .isna()
            .any()
            .any()
        )
        assert json.loads(state.tax_verification)["complete"] is False
        with pytest.raises(ValueError, match="PROPERTY_TAX_INPUTS_INCOMPLETE"):
            recipients.values.qualify_puf55_survey_recipients(run)
        assert len(run.compiled.order) == 51  # No PUF fit graph was constructed.


@pytest.mark.parametrize(
    "defect,reason",
    [
        ("parent", "RETAINED_VALUES_CHANGED"),
        ("artifact", "RECONSTRUCTED_ARTIFACT_CHANGED"),
        ("identity", "MATERIALIZED_VERIFICATION"),
    ],
)
def test_real_child_verifier_refuses_changed_custody(completion_host, defect, reason):
    run = completion_host.cold
    original = host._run_entry(run)[2].completion_boundary
    # Reuse the actual executed support/output and the same issued source/base;
    # a new private boundary isolates permanent revocation. Construction does
    # not fit child income, but verification reconstructs and can refit it.
    boundary = extension._CompletionHost(original.base, original.options, True)
    keys = dict(host._run_entry(run)[2].keys)
    loaded = host._artifacts(
        run.manifest,
        run.compiled,
        run.store,
        run.kernels,
        keys,
        dict(host._run_entry(run)[2].implementations),
    )
    artifacts = extension._inputs(
        boundary.child.nodes[-1], run.compiled, run.kernels, keys, loaded
    )
    verification = extension._value(
        boundary.tax_edge(), run.compiled, run.kernels, keys, loaded
    )
    if defect == "parent":
        # FILTER -> Frame.select -> Frame.__init__ copies each table; role
        # patching also uses _copied_tables(deep=True). Check isolation before
        # asking the verifier to refuse, without substituting any owner.
        base_stamp = host.reconstruction._population_stamp(
            original.base.financial_population
        )
        boundary.parent.frame.person.loc[0, child.child.TARGETS[0]] = 314.0
        assert (
            host.reconstruction._population_stamp(original.base.financial_population)
            == base_stamp
        )
    elif defect == "artifact":
        artifacts["draws"] = replace(artifacts["draws"], payload=b"{}")
    else:
        verification = replace(verification, producer_key="f" * 64)
    with pytest.raises(ValueError, match="^CHILD_PROPERTY_" + reason + "$"):
        boundary.child.verify_materialized(
            original.completed,
            artifacts,
            support_populations={
                n: original.observed[n] for n in (child.DONOR, child.RECIPIENT)
            },
            verification=verification,
        )
    original.base.checked_view()
    assert boundary.child.revoked
    with pytest.raises(ValueError, match="REVOKED"):
        boundary.child._pure()


@pytest.mark.parametrize("when", ["first", "final"])
def test_final_source_io_mutation_permanently_revokes_host(
    completion_host, monkeypatch, when
):
    # Last controls: each poisons one independently issued handle permanently.
    run = completion_host.cold if when == "first" else completion_host.warm
    boundary = host._run_entry(run)[2].completion_boundary
    read = Path.read_bytes
    count = []
    target = 1 if when == "first" else 2
    before = run.financial_population.frame.person[child.STATUS].copy()

    def mutate(path):
        data = read(path)
        if path == Path(extension.__file__):
            count.append(True)
            if len(count) == target:
                run.financial_population.frame.person.loc[0, child.STATUS] = "changed"
        return data

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", mutate)
        with pytest.raises(ValueError):
            run.checked_view()
    assert len(count) >= target
    run.financial_population.frame.person[child.STATUS] = before
    assert boundary.revoked and boundary.child.revoked
    with pytest.raises(ValueError, match="REVOKED"):
        run.checked_view()
