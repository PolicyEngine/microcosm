"""Real survey issuers and captures over invented archives, without an engine."""

import hashlib
import io
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from microcosm.build.us_runtime import current_survey_hours_source as owner


def hours_acs_person(original_person, *, gq_age_changes_afterward=False):
    def person(*args, **kwargs):
        row = original_person(*args, **kwargs)
        under16 = int(row["AGEP"]) < 16 or (
            gq_age_changes_afterward and row["SERIALNO"] == "2024GQ0000001"
        )
        row.update(WKHP="" if under16 else "40", WKL="" if under16 else "1", FWKHP="0")
        return row

    return person


def add_hours_source_fields(arguments, monkeypatch):
    """Set wholly invented hours literals before any source owner is issued."""
    from microcosm.build.us_runtime import asec_person_income_source as restoration

    folder = arguments["source_dir"] / "asec"
    pins, paths = [], {}
    for year, member, archive, *_ in owner.original.asec._MEMBER_PINS:
        path = folder / member
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        raw["MARSUPWT"] = "10000"
        for name in owner.asec_hours.EARNINGS_FIELDS:
            if name not in raw:
                raw[name] = "0"
            raw.loc[raw.A_AGE.map(int).lt(15), name] = "0"
        for name in (
            "HRSWK",
            "WKSWORK",
            "WORKYN",
            "WTEMP",
            "WRK_CK",
            *owner.hours.ALLOCATION_FLAGS,
        ):
            raw[name] = "0"
        for position, age in enumerate(raw.A_AGE.map(int)):
            if age >= 15:
                raw.loc[position, ["HRSWK", "WKSWORK", "WORKYN", "WRK_CK"]] = [
                    "12",
                    "30",
                    "1",
                    "1",
                ]
        # Preserve a valid complete-supplement-nonresponse flag as a distinct fact.
        raw["FL_665"] = "0"
        raw.to_csv(path, index=False)
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
    for module in (owner.original.asec, restoration):
        monkeypatch.setattr(module, "_MEMBER_PINS", tuple(pins))
    restored = folder.parent.parent / "hours-restored-money"
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
    # A private fixture cardinality seam, before any owner is issued. Production
    # remains bound to 2174 keys from the complete income-2024 ASEC coverage.
    monkeypatch.setattr(owner, "_EXPECTED_DONORS", 1)
    monkeypatch.setattr(owner, "_LIVE", owner._live())
    return arguments


def source_arguments(tmp_path, monkeypatch):
    import test_us_child_property_income_source_owner as fixture
    import test_us_survey_population_preparation as preparation_fixture

    monkeypatch.setattr(
        preparation_fixture,
        "_person",
        hours_acs_person(preparation_fixture._person, gq_age_changes_afterward=True),
    )
    full, partial, donor = fixture.source_arguments(tmp_path, monkeypatch)
    add_hours_source_fields(full, monkeypatch)
    for member in (
        *(pin[1] for pin in owner.original.asec._MEMBER_PINS),
        "person-income-attachment.h5",
    ):
        shutil.copyfile(
            full["source_dir"] / "asec" / member,
            partial["source_dir"] / "asec" / member,
        )
    return full, partial, donor


def qualify(preparation):
    return owner.qualify_current_survey_hours(
        preparation,
        age15_policy=owner.hours.AGE15_POLICY,
        under15_policy=owner.hours.UNDER15_POLICY,
    )


@pytest.fixture(scope="module")
def actual(tmp_path_factory):
    root = tmp_path_factory.mktemp("hours-source-owner")
    with pytest.MonkeyPatch.context() as patch:
        full_args, partial_args, donor = source_arguments(root, patch)
        full = owner.source.prepare_authenticated_survey_population(**full_args)
        partial = owner.source.prepare_authenticated_survey_population(**partial_args)
        full_seal = owner.source._frame_identity(full._checked()[2].frame)
        partial_seal = owner.source._frame_identity(partial._checked()[2].frame)
        values, smaller = qualify(full), qualify(partial)
        yield SimpleNamespace(
            full=full, partial=partial, values=values, smaller=smaller, donor=donor
        )
        assert owner.source._frame_identity(full._checked()[2].frame) == full_seal
        assert owner.source._frame_identity(partial._checked()[2].frame) == partial_seal


def test_full_donor_roster_is_preserved_outside_selected_support(actual):
    left, right = actual.values, actual.smaller
    left.validate()
    right.validate()
    assert not right.asec_selected_raw.A_AGE.eq("15").any()
    pd.testing.assert_frame_equal(left.donor_raw, right.donor_raw)
    assert left.donor_raw.PERIDNUM.tolist() == [str(actual.donor - 100).zfill(22)]
    assert left.donor_raw.FL_665.tolist() == ["0"]
    assert left.proposals.supplied_donors == right.proposals.supplied_donors == 1
    all_proposals = {p.key: p for p in left.proposals.proposals}
    assert all(p == all_proposals[p.key] for p in right.proposals.proposals)
    teenager = next(p for p in left.proposals.proposals if dict(p.raw)["AGEP"] == "15")
    assert teenager.hours == 12.0
    assert teenager.policy == owner.hours.AGE15_POLICY
    assert teenager.donor_key.person == str(actual.donor - 100).zfill(22)
    assert dict(teenager.donor_allocation_flags)["FL_665"] == 0
    assert dict(teenager.raw)["WKHP"] == ""
    assert not left.proposals.source_authenticated
    evidence = json.loads(left.receipt)
    assert evidence["projection_scope"] == "selected_original_survey_people"
    assert evidence["full_age15_donor_rows"] == 1
    assert evidence["selected_acs_rows"] == len(left.proposals.proposals)
    assert not evidence["all_person_engine_input_qualified"]
    assert evidence["asec_own_arm_hours_qualified"]
    assert evidence["selected_original_person_hours_qualified"]
    assert not evidence["source_admission_issued"] and not evidence["release_eligible"]


def test_both_original_arms_have_complete_ordered_hours_without_mutating_frame(actual):
    value = actual.values
    assert value.person_hours.index.equals(value.origins.index)
    assert not value.person_hours[owner.hours.TARGET].isna().any()
    assert owner.hours.TARGET not in value.source_frame.person
    asec_ids = value.origins.index[value.origins.source.eq("asec")]
    for pid, proposal in zip(asec_ids, value.asec_proposals.proposals, strict=True):
        row = value.asec_selected_raw.loc[pid]
        assert proposal.key == owner.hours._asec_key(row)
        assert value.person_hours.at[pid, owner.hours.TARGET] == proposal.hours
        assert value.person_hours.at[pid, "hours_provenance"] == proposal.provenance
        if int(row.A_AGE) < 15:
            assert proposal.hours == 0.0
            assert proposal.policy == owner.hours.UNDER15_POLICY
            assert proposal.provenance == "under15_explicit_modeled_zero"
            assert dict(proposal.raw)["HRSWK"] == "0"
        else:
            assert proposal.hours == 12.0
            assert proposal.policy is None
    expected = value.person_hours.loc[actual.smaller.person_hours.index]
    pd.testing.assert_frame_equal(actual.smaller.person_hours, expected)
    evidence = json.loads(value.receipt)
    assert evidence["selected_person_rows"] == len(value.person_hours)
    assert evidence["asec_completion_protocol"] == owner.asec_hours.COMPLETION_PROTOCOL


def test_nested_asec_proposal_mutation_refuses(actual):
    proposal = actual.values.asec_proposals.proposals[0]
    previous = proposal.hours
    try:
        object.__setattr__(proposal, "hours", 99.0)
        with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
            actual.values.validate()
    finally:
        object.__setattr__(proposal, "hours", previous)
    actual.values.validate()


def test_asec_completion_policy_change_cannot_reuse_source_owner(actual, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(owner.asec_hours, "COMPLETION_PROTOCOL", "different")
        with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
            actual.values.validate()
    actual.values.validate()


def test_numeric_person_hours_mutation_refuses(actual):
    table = actual.values.person_hours
    pid = table.index[0]
    previous = table.at[pid, owner.hours.TARGET]
    try:
        table.at[pid, owner.hours.TARGET] = previous + 1
        with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
            actual.values.validate()
    finally:
        table.at[pid, owner.hours.TARGET] = previous
    actual.values.validate()


def receiving_people(qualified):
    import numpy as np

    provenance = owner.attachment.provenance
    origins = qualified.origins
    original = np.tile(origins.index.to_numpy(), 2)
    selected = origins.reindex(original)
    return (
        pd.DataFrame(
            {
                "person_id": np.arange(len(original), dtype="int64") + 100000,
                provenance.support_source_id_column("person"): original,
                provenance.support_clone_index_column("person"): np.repeat(
                    [0, 1], len(origins)
                ).astype("int64"),
                provenance.spine_source_id_column(
                    "person"
                ): selected.native_person_id.to_numpy(),
                provenance.support_channel_column("person"): pd.array(
                    selected.source.to_numpy(), dtype=owner.STRING
                ),
            }
        )
        .iloc[::-1]
        .reset_index(drop=True)
    )


def test_clone_transport_preserves_original_draw_and_provenance_under_reordering(
    actual,
):
    value = actual.values
    people = receiving_people(value)
    before = people.copy(deep=True)
    result = owner.borrow_cloned_hours_columns(value, people)
    source_id = owner.attachment.provenance.support_source_id_column("person")
    for column in value.person_hours:
        expected = value.person_hours[column].reindex(people[source_id])
        expected.index = pd.Index(people.person_id.to_numpy(), name="person_id")
        pd.testing.assert_series_equal(result["person", column], expected)
    pd.testing.assert_frame_equal(people, before)
    result["person", owner.hours.TARGET].iloc[0] = 97.0
    value.validate()


@pytest.mark.parametrize(
    "change", ["source", "native", "channel", "clone", "collision"]
)
def test_clone_transport_refuses_wrong_identity_or_owned_output(actual, change):
    people = receiving_people(actual.values)
    p = owner.attachment.provenance
    if change == "collision":
        people[owner.hours.TARGET] = 40.0
    elif change == "channel":
        people.loc[0, p.support_channel_column("person")] = "foreign"
    else:
        column = {
            "source": p.support_source_id_column("person"),
            "native": p.spine_source_id_column("person"),
            "clone": p.support_clone_index_column("person"),
        }[change]
        people.loc[0, column] = 99999
    with pytest.raises(ValueError):
        owner.borrow_cloned_hours_columns(actual.values, people)


def test_clone_transport_checks_receiving_mutation_after_final_source_io(
    actual, monkeypatch
):
    people = receiving_people(actual.values)
    original = owner.QualifiedSurveyHoursProposals.validate
    calls = SimpleNamespace(count=0)

    def validation(value):
        original(value)
        calls.count += 1
        if calls.count == 2:
            people.loc[0, "person_id"] += 1

    # Change the method only before this test's accepted implementation profile;
    # the resulting callback simulates mutation during final foreign-owner I/O.
    with monkeypatch.context() as patch:
        patch.setattr(owner.QualifiedSurveyHoursProposals, "validate", validation)
        patch.setattr(owner, "_LIVE", owner._live())
        with pytest.raises(ValueError, match="RECEIVING_CHANGED"):
            owner.borrow_cloned_hours_columns(actual.values, people)
    actual.values.validate()


@pytest.mark.parametrize("which", ["owner", "callback", "preparation", "receipt"])
def test_copies_callbacks_and_receipts_cannot_replace_the_live_owner(actual, which):
    value = actual.values
    if which == "owner":
        with pytest.raises(ValueError, match="PROJECTION_OBJECT_CHANGED"):
            replace(value).validate()
    elif which == "preparation":
        fake = object.__new__(type(actual.full))
        object.__setattr__(fake, "payload", actual.full.payload)
        with pytest.raises(ValueError):
            qualify(fake)
    else:
        field = "_revalidate" if which == "callback" else "receipt"
        old = getattr(value, field)
        try:
            object.__setattr__(
                value, field, (lambda _: None) if which == "callback" else b"{}"
            )
            with pytest.raises(ValueError):
                value.validate()
        finally:
            object.__setattr__(value, field, old)
    value.validate()


@pytest.mark.parametrize(
    "field,column",
    [
        ("acs_raw", "WKHP"),
        ("asec_selected_raw", "HRSWK"),
        ("donor_raw", "HRSWK"),
        ("origins", "source"),
        ("person_hours", "hours_provenance"),
        ("asec_selected_raw", "WSAL_VAL"),
    ],
)
def test_mutated_source_projection_refuses(actual, field, column):
    table = getattr(actual.values, field)
    position = table.index[0]
    previous = table.at[position, column]
    try:
        table.at[position, column] = "changed"
        with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
            actual.values.validate()
    finally:
        table.at[position, column] = previous
    actual.values.validate()


@pytest.mark.parametrize("field", ["hours", "donor_key", "donor_allocation_flags"])
def test_nested_proposal_mutation_refuses(actual, field):
    proposal = next(
        p for p in actual.values.proposals.proposals if p.donor_key is not None
    )
    previous = getattr(proposal, field)
    try:
        object.__setattr__(
            proposal,
            field,
            {"hours": 99.0, "donor_key": None, "donor_allocation_flags": ()}[field],
        )
        with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
            actual.values.validate()
    finally:
        object.__setattr__(proposal, field, previous)
    actual.values.validate()


def test_final_io_cannot_mutate_then_hide_a_projection(actual, monkeypatch):
    value = actual.values
    pid = value.donor_raw.index[0]
    previous = value.donor_raw.at[pid, "HRSWK"]
    read_bytes = Path.read_bytes
    triggered = []

    def mutate_after_read(path):
        data = read_bytes(path)
        if not triggered:
            triggered.append(True)
            value.donor_raw.at[pid, "HRSWK"] = "99"
        return data

    try:
        with monkeypatch.context() as patch:
            patch.setattr(Path, "read_bytes", mutate_after_read)
            with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
                value.validate()
        assert triggered == [True]
    finally:
        value.donor_raw.at[pid, "HRSWK"] = previous
    value.validate()


@pytest.mark.parametrize(
    "field", ["acs_raw", "donor_raw", "proposals", "asec_proposals", "person_hours"]
)
def test_equal_detached_projection_cannot_replace_owned_identity(actual, field):
    value = actual.values
    previous = getattr(value, field)
    detached = (
        replace(previous)
        if field in ("proposals", "asec_proposals")
        else previous.copy(deep=True)
    )
    try:
        object.__setattr__(value, field, detached)
        with pytest.raises(ValueError, match="PROJECTION_CHANGED"):
            value.validate()
    finally:
        object.__setattr__(value, field, previous)
    value.validate()


def test_retained_source_coverage_mutation_refuses(actual):
    callback = actual.values._revalidate
    captured = dict(
        zip(
            callback.__code__.co_freevars,
            (c.cell_contents for c in callback.__closure__),
            strict=True,
        )
    )
    coverage = captured["coverage"]
    previous = coverage._body
    try:
        object.__setattr__(coverage, "_body", previous + b"changed")
        with pytest.raises(ValueError):
            actual.values.validate()
    finally:
        object.__setattr__(coverage, "_body", previous)
    actual.values.validate()


def test_changed_draw_policy_cannot_reuse_an_issued_projection(actual, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(owner.hours, "SEED", owner.hours.SEED + 1)
        with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
            actual.values.validate()
    actual.values.validate()


@pytest.mark.parametrize("policy", ["age15_policy", "under15_policy"])
def test_model_policies_are_explicit_before_any_source_access(policy):
    options = dict(
        age15_policy=owner.hours.AGE15_POLICY, under15_policy=owner.hours.UNDER15_POLICY
    )
    options[policy] = None
    with pytest.raises(ValueError, match="POLICY"):
        owner.qualify_current_survey_hours(object(), **options)


@pytest.mark.parametrize(
    "change,code",
    [
        ("header", "SOURCE_HEADER"),
        ("shape", "SOURCE_ROW_SHAPE"),
        ("duplicate", "DUPLICATE_SELECTED_SOURCE_KEY"),
        ("bound", "SOURCE_ROW_SHAPE"),
    ],
)
def test_scanner_exhausts_and_rejects_malformed_or_duplicate_records(change, code):
    columns = owner.hours.ACS_FIELDS
    row = ["2024HU0000001", "1", "30", "40", "1", "0", "100", "0"]
    header = list(columns)
    rows = [row]
    maximum = 2
    if change == "header":
        header[-1] = header[-2]
    elif change == "shape":
        rows.append(row[:-1])
    elif change == "duplicate":
        rows.append(row)
    else:
        maximum = 0
    raw = (",".join(header) + "\n" + "".join(",".join(r) + "\n" for r in rows)).encode()
    with pytest.raises(ValueError, match=code):
        owner._scan(
            io.BytesIO(raw),
            survey="acs",
            wanted={(row[0], "1", 1)},
            selected={},
            donors={},
            maximum=maximum,
        )
