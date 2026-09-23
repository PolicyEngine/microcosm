"""Real source owners over invented bytes; no donor model or receiving issuer."""

import copy
import hashlib
import json
import shutil
import sys
import threading
from contextlib import contextmanager
from dataclasses import replace
from fractions import Fraction
from types import ModuleType

import pytest
from test_us_survey_population_preparation import fixture as source_fixture

from microcosm.build.us_runtime import current_survey_household_domains as domains
from microcosm.build.us_runtime import graph_survey_population as graph
from microcosm.build.us_runtime import survey_population_preparation as source
from microcosm.frame import Frame, Weights


@contextmanager
def normal_return_observer(target, *, ordinal, callback):
    """Observe exact normal returns, leaving exceptional unwinds uncounted.

    The housing tests target successful source validation / producer returns.
    They assert the mutation fired, so an earlier exception cannot silently
    substitute for the intended closing boundary. No global event is enabled.
    """
    assert type(ordinal) is int and ordinal > 0
    monitor, tool, name = sys.monitoring, 5, "housing-test-normal-return"
    code = target.__code__
    owner_thread = threading.get_ident()
    assert monitor.get_tool(tool) is None, "RETURN_OBSERVER_TOOL_OCCUPIED"
    assert monitor.get_events(tool) == monitor.get_local_events(tool, code) == 0
    profiles = (sys.getprofile(), threading.getprofile())
    others = tuple(
        (
            i,
            monitor.get_tool(i),
            monitor.get_events(i),
            monitor.get_local_events(i, code),
        )
        for i in range(6)
        if i != tool
    )
    state = {"normal_returns": 0, "fired": False}

    def returned(actual_code, _offset, _value):
        # Local monitoring events reach all threads; match the previous
        # sys.setprofile observer's installing-thread scope explicitly.
        if threading.get_ident() != owner_thread:
            return
        assert actual_code is code
        state["normal_returns"] += 1
        if state["normal_returns"] == ordinal:
            state["fired"] = True
            callback()

    monitor.use_tool_id(tool, name)
    try:
        assert (
            monitor.register_callback(tool, monitor.events.PY_RETURN, returned) is None
        )
        monitor.set_local_events(tool, code, monitor.events.PY_RETURN)
        yield state
    finally:
        assert threading.get_ident() == owner_thread, "RETURN_OBSERVER_THREAD_CHANGED"
        assert monitor.get_tool(tool) == name, "RETURN_OBSERVER_OWNERSHIP_CHANGED"
        monitor.set_local_events(tool, code, 0)
        previous = monitor.register_callback(tool, monitor.events.PY_RETURN, None)
        monitor.clear_tool_id(tool)
        monitor.free_tool_id(tool)
        assert previous is returned
        assert monitor.get_tool(tool) is None
        assert (sys.getprofile(), threading.getprofile()) == profiles
        assert (
            tuple(
                (
                    i,
                    monitor.get_tool(i),
                    monitor.get_events(i),
                    monitor.get_local_events(i, code),
                )
                for i in range(6)
                if i != tool
            )
            == others
        )


class RealDiskFixturePatch:
    """Ignore only maintained fixture disk stubs; preserve actual OS probes."""

    def __init__(self, patch):
        self.patch = patch
        self.original_disk_usage = shutil.disk_usage
        self.suppressed = 0

    def setattr(self, target, name, value, *args, **kwargs):
        if target is shutil and name == "disk_usage":
            assert shutil.disk_usage is self.original_disk_usage
            self.suppressed += 1
            return None
        return self.patch.setattr(target, name, value, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.patch, name)


def _arguments(root, patch):
    genuine = RealDiskFixturePatch(patch)
    arguments = source_fixture(root, genuine)
    assert genuine.suppressed > 0
    assert shutil.disk_usage is genuine.original_disk_usage
    return {**arguments, "store_root": root / "graph-store"}


@pytest.fixture(scope="module")
def raw_prefix(tmp_path_factory):
    # Reuse one real, immutable raw prefix for bounded source-only checks. Each
    # checked_rows/qualification still runs its own actual verification epoch.
    with pytest.MonkeyPatch.context() as patch:
        arguments = _arguments(tmp_path_factory.mktemp("domains"), patch)
        live = graph.run_authenticated_survey_population(
            **arguments, clones=True, return_values=True
        )
        yield live, arguments


def _qualify(live):
    return domains.qualify_current_survey_household_domains(
        live.preparation, live.allocated_population, live.clone_population
    )


def _copy_frame(frame):
    return Frame(
        {e: frame.table(e).copy(deep=True) for e in frame.entities},
        frame.schema,
        {
            e: Weights(frame.weights_for(e).values.copy(), frame.weights_for(e).kind)
            for e in frame.weighted_entities
        },
        frame.strata.copy(deep=True),
        metadata=frame.metadata,
        mass_log=frame.mass_log,
    )


def test_live_implementation_seal_is_repeatable():
    # Fresh immutable declaration records carry equal scientific content. Their
    # transient identities must not make an unchanged implementation fail.
    first, second = domains.domains.declaration(), domains.domains.declaration()
    assert first == second and all(
        a is not b for a, b in zip(first, second, strict=True)
    )
    assert domains._live() == domains._live() == domains._LIVE


def test_real_raw_prefix_and_required_replay_project_complete_source_roster(raw_prefix):
    cold, arguments = raw_prefix
    qualified = _qualify(cold)
    rows = qualified.checked_rows()
    view = cold.preparation.checked_view()
    selected = {
        (r.key.source, r.key.source_year, r.key.survey_year, r.key.native_id): r
        for r in view.selection_plan.selected
    }
    assert len(rows) == 2 * len(selected) == 12
    assert {r.source for r in rows} == {
        domains.domains.Source.ACS,
        domains.domains.Source.ASEC,
    }
    assert len({r.household_id for r in rows}) == len(rows)
    for key, original in selected.items():
        copies = [
            r
            for r in rows
            if (r.source, r.source_year, r.survey_year, r.raw_native_id) == key
        ]
        assert {r.clone_index for r in copies} == {0, 1}
        assert len({(r.support_source_id, r.spine_source_id) for r in copies}) == 1
        assert all(r.domain is original.domain for r in copies)
        assert all(
            r.original_design_anchor == original.original_design_weight for r in copies
        )
        assert all(type(r.original_design_anchor) is Fraction for r in copies)
    gq = [r for r in rows if r.statistical_unit == "person"]
    assert len(gq) == 4 and not any(r.occupied_housing_tail_eligible for r in gq)
    assert {r.domain for r in gq} == {
        domains.domains.Domain.INSTITUTIONAL_GQ,
        domains.domains.Domain.NONINSTITUTIONAL_GQ,
    }
    zero = [r for r in rows if r.original_design_anchor == 0]
    assert zero and all(r.occupied_housing_tail_eligible for r in zero)
    assert {(r.source.value, r.source_year, r.survey_year) for r in rows} == {
        ("acs", 2024, 2024),
        ("asec", 2024, 2025),
    }
    for entity in cold.clone_population.frame.entities:
        assert "block_geoid" not in cold.clone_population.frame.table(entity)
    receipt = json.loads(qualified.payload)
    assert receipt["roster_rows"] == len(rows) == 12
    assert receipt["original_households"] == len(selected) == 6
    assert receipt["occupied_housing_rows"] == 8
    assert receipt["roster_sha256"] == domains._roster_digest(rows)
    assert receipt["preparation_sha256"] == hashlib.sha256(view.payload).hexdigest()
    assert receipt["raw_prefix_only"] and not qualified.release_eligible
    assert not any(
        receipt[key]
        for key in (
            "geography_assigned",
            "matching_qualified",
            "donor_role_qualified",
            "source_window_equivalence_claim",
            "release_eligible",
        )
    )
    warm = graph.run_authenticated_survey_population(
        **arguments, clones=True, return_values=True, resume="require"
    )
    assert all(row.store_hit for row in warm.manifest.nodes.values())
    # Same-process/store semantic parity; not independent-process actual-data proof.
    assert _qualify(warm).checked_rows() == rows


def test_public_and_unissued_constructors_do_not_issue_authority():
    with pytest.raises(ValueError, match="NO_PUBLIC_CONSTRUCTOR"):
        domains.QualifiedSurveyHouseholdDomains(b"{}")
    unissued = object.__new__(domains.QualifiedSurveyHouseholdDomains)
    object.__setattr__(unissued, "payload", b"{}")
    with pytest.raises(ValueError, match="UNISSUED_OR_CHANGED"):
        unissued.checked_rows()


@pytest.mark.parametrize("kind", ("copy", "deepcopy", "preparation"))
def test_copies_do_not_inherit_issuance(raw_prefix, kind):
    live, _ = raw_prefix
    if kind == "preparation":
        with pytest.raises(ValueError, match="UNISSUED"):
            domains.qualify_current_survey_household_domains(
                copy.copy(live.preparation),
                live.allocated_population,
                live.clone_population,
            )
    else:
        qualified = _qualify(live)
        copied = getattr(copy, kind)(qualified)
        assert copied is not qualified
        with pytest.raises(ValueError, match="UNISSUED_OR_CHANGED"):
            copied.checked_rows()
        assert qualified.checked_rows()


@pytest.mark.parametrize(
    "change",
    (
        "clone_value",
        "allocation_value",
        "design_anchor",
        "ownership",
        "prepared_frame_replaced",
        "clone_frame_replaced",
        "plan_changed",
        "clone2",
    ),
)
def test_live_retained_evidence_mutation_revokes_capsule(
    raw_prefix, monkeypatch, change
):
    live, _ = raw_prefix
    qualified = _qualify(live)
    prep = source._ISSUED[id(live.preparation)][2]
    frame = live.clone_population.frame
    if change in ("clone_value", "allocation_value", "clone2"):
        table = (
            live.allocated_population.frame if change == "allocation_value" else frame
        ).person
        column = (
            domains.provenance.support_clone_index_column("person")
            if change == "clone2"
            else "age"
        )
        monkeypatch.setitem(table, column, table[column] + 1)
    elif change == "design_anchor":
        target, attribute = live.clone_population, "design_weights"
        old = target.design_weights
        changed = dict(old)
        changed["household"] = changed["household"] + 1
        object.__setattr__(target, attribute, changed)
    elif change == "ownership":
        target, attribute = live.clone_population, "owners"
        old = target.owners
        changed = dict(old)
        changed[next(iter(changed))] = "unissued-replacement"
        object.__setattr__(target, attribute, changed)
    elif change == "plan_changed":
        old = prep.plan
        object.__setattr__(prep, "plan", replace(old, selected=old.selected[::-1]))
    else:
        target = prep if change == "prepared_frame_replaced" else live.clone_population
        old = target.frame
        object.__setattr__(target, "frame", _copy_frame(old))
    try:
        with pytest.raises(ValueError):
            qualified.checked_rows()
        assert id(qualified) not in domains._ISSUED
    finally:
        if change == "plan_changed":
            object.__setattr__(prep, "plan", old)
        elif change in ("prepared_frame_replaced", "clone_frame_replaced"):
            object.__setattr__(target, "frame", old)
        elif change in ("design_anchor", "ownership"):
            object.__setattr__(target, attribute, old)


@pytest.mark.parametrize("change", ("version", "geography_column"))
def test_cold_qualification_requires_exact_raw_prefix(raw_prefix, change):
    live, _ = raw_prefix
    if change == "version":
        foreign = replace(live.clone_population, version="foreign-descendant")
    else:
        frame = _copy_frame(live.clone_population.frame)
        frame.table("household")["block_geoid"] = "invented-assignment"
        foreign = replace(
            live.clone_population,
            frame=frame,
            owners={
                **live.clone_population.owners,
                ("household", "block_geoid"): "invented-owner",
            },
        )
    with pytest.raises(ValueError):
        domains.qualify_current_survey_household_domains(
            live.preparation, live.allocated_population, foreign
        )


@pytest.mark.parametrize(
    "change,reason",
    (
        ("missing", "ORIGIN_COVERAGE"),
        ("duplicate", "RAW_IDENTITY_COVERAGE"),
        ("year", "RAW_IDENTITY_COVERAGE"),
        ("anchor", "ORIGINAL_ANCHOR"),
        ("unit", "DOMAIN_UNIT"),
        ("clone2", "CLONE_ORIGIN"),
        ("duplicate_clone", "DUPLICATE_CLONE_ORIGIN"),
        ("missing_clone", "ORDINARY_CLONE_ROSTER"),
    ),
)
def test_pure_join_refuses_incomplete_or_inconsistent_descriptive_views(
    raw_prefix, change, reason
):
    live, _ = raw_prefix
    original = live.preparation.checked_view()
    receipt = copy.deepcopy(original.receipt)
    view = replace(original, receipt=receipt)
    clone = replace(
        live.clone_population, frame=_copy_frame(live.clone_population.frame)
    )
    origins = receipt["origins"]["households"]
    if change == "missing":
        origins.pop()
    elif change == "duplicate":
        origins[1] = origins[0].copy()
    elif change == "year":
        origins[0]["survey_year"] += 1
    elif change == "anchor":
        origins[0]["original_anchor"][0] += 1
    elif change == "unit":
        plan = original.selection_plan
        view = replace(
            view,
            selection_plan=replace(
                plan,
                selected=(
                    replace(plan.selected[0], statistical_unit="wrong"),
                    *plan.selected[1:],
                ),
            ),
        )
    elif change == "clone2":
        clone.frame.table("household").loc[
            0, domains.provenance.support_clone_index_column("household")
        ] = 2
    elif change == "duplicate_clone":
        households = clone.frame.table("household")
        support_column = domains.provenance.support_source_id_column("household")
        clone_column = domains.provenance.support_clone_index_column("household")
        pair = households.loc[
            households[support_column] == households[support_column].iloc[0]
        ]
        assert len(pair) == 2 and set(pair[clone_column]) == {0, 1}
        households.loc[pair.index[1], clone_column] = pair[clone_column].iloc[0]
    else:
        # Mutate the copied descriptive Frame after its genuine construction;
        # the pure helper must refuse the now-incomplete roster, never issue it.
        clone.frame._tables["household"] = (
            clone.frame.table("household").iloc[:-1].copy()
        )
    # This private pure helper grants no authority; no fake source owner is issued.
    with pytest.raises(ValueError, match=reason):
        domains._project(view, clone)


@pytest.mark.parametrize("operation", ("qualify", "borrow"))
@pytest.mark.parametrize("boundary", ("source_close", "final_code_read"))
def test_final_callbacks_cannot_return_stale_success(raw_prefix, operation, boundary):
    live, _ = raw_prefix
    qualified = _qualify(live) if operation == "borrow" else None
    target = source._validate if boundary == "source_close" else domains._producer
    table = live.clone_population.frame.person
    before = table["age"].copy(deep=True)

    def mutate():
        table["age"] = before + 1

    try:
        # These are successful validations/producer reads. Exceptional unwinds
        # are deliberately not counted; assert fired rules out an earlier error
        # masquerading as the intended second normal-return mutation boundary.
        with normal_return_observer(target, ordinal=2, callback=mutate) as observed:
            with pytest.raises(ValueError, match="FINAL_POPULATION_SEAL"):
                qualified.checked_rows() if qualified is not None else _qualify(live)
        assert observed == {"normal_returns": 2, "fired": True}
    finally:
        table["age"] = before
    if qualified is not None:
        assert id(qualified) not in domains._ISSUED


@pytest.mark.parametrize("operation", ("qualify", "borrow"))
def test_final_code_callback_cannot_change_live_contract(raw_prefix, operation):
    live, _ = raw_prefix
    qualified = _qualify(live) if operation == "borrow" else None
    original_limit = domains.MAX_ORIGINS

    def mutate():
        # The tiny roster still fits this lower limit; only the final live-code
        # contract fence can catch a change after the last producer check.
        domains.MAX_ORIGINS = original_limit - 1

    try:
        with normal_return_observer(
            domains._producer, ordinal=2, callback=mutate
        ) as observed:
            with pytest.raises(ValueError, match="FINAL_IMPLEMENTATION_CHANGED"):
                qualified.checked_rows() if qualified is not None else _qualify(live)
        assert observed == {"normal_returns": 2, "fired": True}
    finally:
        domains.MAX_ORIGINS = original_limit
    if qualified is not None:
        assert id(qualified) not in domains._ISSUED


@pytest.mark.parametrize("change", ("limit", "declaration", "source", "budget"))
def test_implementation_contract_change_refuses(raw_prefix, monkeypatch, change):
    live, _ = raw_prefix
    qualified = _qualify(live)
    if change == "limit":
        monkeypatch.setattr(domains, "MAX_ORIGINS", domains.MAX_ORIGINS - 1)
    elif change == "declaration":
        declared = domains.domains.declaration()
        monkeypatch.setattr(domains.domains, "declaration", lambda: declared[::-1])
    else:
        original = getattr(domains, change)
        replacement = ModuleType(original.__name__)
        replacement.__dict__.update(vars(original))
        # Every function is still the genuine function; a different module
        # object cannot stand in for the retained implementation binding.
        monkeypatch.setattr(domains, change, replacement)
    with pytest.raises(ValueError, match="IMPLEMENTATION_(BINDING_)?CHANGED"):
        qualified.checked_rows()


def test_source_byte_change_refuses_with_original_owner(tmp_path, monkeypatch):
    arguments = _arguments(tmp_path, monkeypatch)
    live = graph.run_authenticated_survey_population(
        **arguments, clones=True, return_values=True
    )
    qualified = _qualify(live)
    path = arguments["source_dir"] / "selection-request.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        qualified.checked_rows()
    assert id(qualified) not in domains._ISSUED
