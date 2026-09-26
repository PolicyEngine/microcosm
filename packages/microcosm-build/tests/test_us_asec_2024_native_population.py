"""Real closed loaders over invented source bytes; no genuine source admission."""

import csv
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.frame_checkpoint import (
    load_frame_checkpoint,
    write_frame_checkpoint,
)
from microcosm.build.outer_stage_runtime import frame_identity
from microcosm.build.us_runtime import asec_2024_native_population as native
from microcosm.build.us_runtime import asec_coverage_authentication as coverage
from microcosm.build.us_runtime import asec_current_money_source as money
from microcosm.build.us_runtime import asec_household_observations as household_owner
from microcosm.build.us_runtime import asec_person_income_source as restoration
from microcosm.build.us_runtime import native_household_origin as origin
from microcosm.frame import Frame, WeightKind, Weights


def _helpers():
    spec = importlib.util.spec_from_file_location(
        "native_2024_invented_fixture",
        Path(__file__).with_name("test_us_asec_coverage_authentication.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rebuild_observations(
    tmp_path,
    monkeypatch,
    persons,
    *,
    extra_household,
    legacy_weights,
    missing_money=None,
    current_predictor_money=None,
):
    """Build larger invented real checkpoints and the actual observation attachment."""
    parent_path = tmp_path / "invented-original-v4.h5"
    household_path = tmp_path / "invented-household-attachment.h5"
    loaded = load_frame_checkpoint(parent_path)
    tables = {e: loaded.frame.table(e).copy(deep=True) for e in loaded.frame.entities}
    strata = loaded.frame.strata.copy(deep=True)
    if extra_household:
        extra = tables["person"].iloc[-2:].copy(deep=True)
        extra["person_id"] = np.array([107, 108], dtype=np.int64)
        extra["PERIDNUM"] = pd.array([str(i).zfill(22) for i in (7, 8)], dtype="string")
        extra["source_household_id"] = 8
        extra["PH_SEQ"] = 4
        for entity in loaded.frame.schema.group_entities:
            extra[f"person_{entity}_id"] = 4
            group = tables[entity].iloc[-1:].copy(deep=True)
            group[f"{entity}_id"] = 4
            tables[entity] = pd.concat([tables[entity], group], ignore_index=True)
        tables["person"] = pd.concat([tables["person"], extra])
        strata = pd.concat([strata, strata.iloc[-2:]])
        cohort_path = tmp_path / "invented-cohort-2024.h5"
        cohort = pd.read_hdf(cohort_path, key="household")
        extra_cohort = cohort.iloc[-1:].copy(deep=True)
        extra_cohort["H_SEQ"] = 8
        pd.concat([cohort, extra_cohort], ignore_index=True).to_hdf(
            cohort_path, key="household", format="fixed", mode="w"
        )

        def add_people(rows):
            extra_rows = [r.copy() for r in rows[1:]]
            for row in extra_rows:
                row[0] = str(int(row[0]) + 2).zfill(22)
                row[1] = "8"
            rows.extend(extra_rows)

        _helpers()._rewrite(persons[2024], add_people)
    if missing_money is not None:
        year, field = missing_money
        assert (year, field) in ((2024, "ANN_VAL"), (2022, "WSAL_VAL"))
        person = tables["person"]
        position = np.flatnonzero(person.source_year.to_numpy() == year)[0]
        # Alter only invented source construction, before identities, attachments
        # and normal issuance. Never mutate an already authenticated parent.
        person.iloc[position, person.columns.get_loc(field)] = np.nan
    if current_predictor_money is not None:
        # Alter only invented original source construction before checkpoint
        # identities, observation attachments, private pins and fresh issuance.
        person = tables["person"]
        positions = np.flatnonzero(person.source_year.to_numpy() == 2024)
        assert set(current_predictor_money) == {
            "WSAL_VAL",
            "SEMP_VAL",
            "INT_VAL",
            "DIV_VAL",
            "CAP_VAL",
        }
        for field, observations in current_predictor_money.items():
            observations = np.asarray(observations)
            assert observations.dtype == np.dtype("float64")
            assert (
                observations.shape == (len(positions),)
                and np.isfinite(observations).all()
            )
            person.iloc[positions, person.columns.get_loc(field)] = observations
    values = (
        legacy_weights
        if legacy_weights is not None
        else ([1, 2, 3, 9] if extra_household else [1, 2, 3])
    )
    frame = Frame(
        tables,
        loaded.frame.schema,
        {"household": Weights(np.array(values), WeightKind.DESIGN)},
        strata,
    )
    metadata = loaded.metadata
    for key in ("identity", "source_construction_identity"):
        metadata[key] = frame_identity(frame).to_payload()
    sources = []
    for year in (2022, 2023, 2024):
        path = tmp_path / f"invented-cohort-{year}.h5"
        sources.append(
            household_owner.AsecHouseholdObservationSource(
                year, path, hashlib.sha256(path.read_bytes()).hexdigest()
            )
        )
    metadata["source_receipt"]["sources"] = [
        {"year": source.year, "path": str(source.path), "sha256": source.sha256}
        for source in sources
    ]
    write_frame_checkpoint(parent_path, frame, metadata=metadata)
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    attachment = household_owner.with_asec_household_observations(
        frame,
        checkpoint_metadata=metadata,
        checkpoint_sha256=parent_sha,
        sources=sources,
    )
    write_frame_checkpoint(
        household_path,
        attachment.frame,
        metadata={
            "schema_version": 1,
            "artifact_kind": "microcosm.asec_household_observations_source",
            "parent_checkpoint_sha256": parent_sha,
            "household_observations": attachment.receipt,
        },
    )
    monkeypatch.setattr(
        money,
        "_SOURCE_PINS",
        (
            parent_sha,
            hashlib.sha256(household_path.read_bytes()).hexdigest(),
            tuple((s.year, s.sha256) for s in sources),
        ),
    )
    _helpers()._pin(persons, monkeypatch)


def _fixture(
    tmp_path,
    monkeypatch,
    *,
    household_rows=None,
    extra_household=False,
    legacy_weights=None,
    missing_money=None,
    current_predictor_money=None,
    **changes,
):
    helpers = _helpers()
    _, persons = helpers._fixtures(tmp_path, monkeypatch, **changes)
    if (
        extra_household
        or legacy_weights is not None
        or missing_money is not None
        or current_predictor_money is not None
    ):
        _rebuild_observations(
            tmp_path,
            monkeypatch,
            persons,
            extra_household=extra_household,
            legacy_weights=legacy_weights,
            missing_money=missing_money,
            current_predictor_money=current_predictor_money,
        )
    # Both real owners must pin the same final invented original member bytes.
    monkeypatch.setattr(restoration, "_MEMBER_PINS", coverage._MEMBER_PINS)
    parent = tmp_path / "invented-original-v4.h5"
    household = tmp_path / "invented-household-attachment.h5"
    output = tmp_path / "restored"
    restoration.restore_asec_person_income_source(
        parent, household, member_paths=persons, output_dir=output
    )
    member = tmp_path / "hhpub25.csv"
    rows = (
        household_rows
        if household_rows is not None
        else [
            ["00007", "1", "000255212", "6", "1", "2"],
            ["00008", "2", "", "0", "0", "0"],
        ]
    )
    if extra_household and household_rows is None:
        rows[1] = ["00008", "1", "00000000", "6", "1", "2"]
    with member.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["H_SEQ", "H_HHTYPE", "HSUP_WGT", "HRHTYPE", "H_LIVQRT", "H_NUMPER"]
        )
        writer.writerows(rows)
    data = member.read_bytes()
    monkeypatch.setattr(
        origin,
        "_ASEC_MEMBER_PINS",
        (
            origin.AsecNativeMemberPin(
                income_year=2024,
                survey_year=2025,
                canonical_member_id="invented/2025/household/hhpub25.csv",
                member_name="hhpub25.csv",
                archive_sha256=origin.ASEC_EDUCATION_ASSISTANCE_ARCHIVES[
                    2024
                ].zip_sha256,
                member_sha256=hashlib.sha256(data).hexdigest(),
                size_bytes=len(data),
                rows=len(rows),
            ),
        ),
    )
    return dict(
        parent_path=parent,
        household_attachment_path=household,
        person_income_attachment_path=output / restoration.CHECKPOINT_FILENAME,
        person_member_paths=persons,
        household_member_path=member,
    )


def test_real_three_cohort_loader_selects_2024_original_anchors(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    result = native.load_authenticated_asec_2024_native_population(**paths)
    frame, receipt = result.frame, result.receipt
    assert frame.person.source_year.tolist() == [2024, 2024]
    assert frame.person.person_id.tolist() == [105, 106]
    assert frame.person.A_AGE.tolist() == [55, 14]
    assert frame.person.asec_PTOTVAL.tolist() == [-9999, 0]
    assert frame.weights_for("household").kind is WeightKind.DESIGN
    assert frame.weights_for("household").values.tolist() == [2552.12]
    assert dict(frame.metadata) == {} and frame.mass_log == ()
    assert json.loads(result.context)["entities"]["person"]["rows"] == 2
    assert receipt["parent_custody"]["cohorts"] == [2022, 2023, 2024]
    assert receipt["parent_custody"]["population_cohorts"] == [2024]
    assert receipt["households"][0]["HSUP_WGT"] == "000255212"
    assert receipt["households"][0]["fraction"] == [63803, 25]
    assert receipt["households"][0]["native_key"] == [2024, 7]
    assert not receipt["release_eligible"]
    assert not receipt["selection"]["inclusion_probabilities_known"]
    assert {r["A_LINENO"] for r in receipt["persons"]} == {1, 2}
    for entity in frame.schema.group_entities:
        assert len(frame.table(entity)) == 1
    # Real parent remains separate, complete and sealed after all child borrows.
    state = native._ISSUED[id(result)][2]
    assert state.parent.frame.person.source_year.tolist() == [
        2022,
        2022,
        2023,
        2023,
        2024,
        2024,
    ]
    assert state.parent.frame.weights_for("household").values.tolist() == [1, 2, 3]
    state.parent.validate()


def test_exact_selection_candidate_reconstruction_and_value_equal_copy_refusal(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path, monkeypatch)
    result = native.load_authenticated_asec_2024_native_population(
        **paths, selected_households=((2024, 7),)
    )
    candidate = result.to_bytes()
    reconstructed = native.load_authenticated_asec_2024_native_population(
        **paths, selected_households=((2024, 7),), candidate=candidate
    )
    assert reconstructed == result and reconstructed is not result
    reconstructed.validate()
    for copy in (
        replace(result),
        native.AuthenticatedAsec2024NativePopulation(candidate),
    ):
        assert copy == result
        with pytest.raises(native.AsecNativePopulationError, match="UNISSUED"):
            copy.validate()
    with pytest.raises(native.AsecNativePopulationError, match="CANDIDATE_MISMATCH"):
        native.load_authenticated_asec_2024_native_population(**paths, candidate=b"{}")


@pytest.mark.parametrize(
    "selection",
    [
        (),
        ((2023, 7),),
        ((2024, 7), (2024, 7)),
        ((2024, True),),
        [(2024, 7)],
        ((2024, 8),),
    ],
)
def test_non_exact_or_absent_selection_refuses(tmp_path, monkeypatch, selection):
    paths = _fixture(tmp_path, monkeypatch)
    with pytest.raises(native.AsecNativePopulationError, match="SELECTION"):
        native.load_authenticated_asec_2024_native_population(
            **paths, selected_households=selection
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("H_NUMPER", "1"),
        ("H_NUMPER", ""),
        ("H_SEQ", "9"),
        ("HSUP_WGT", "2552.12"),
        ("HSUP_WGT", ""),
        ("HSUP_WGT", "-1"),
        ("HSUP_WGT", "0"),
        ("H_HHTYPE", "2"),
    ],
)
def test_source_household_roster_or_unusable_anchor_refuses(
    tmp_path, monkeypatch, field, value
):
    names = ["H_SEQ", "H_HHTYPE", "HSUP_WGT", "HRHTYPE", "H_LIVQRT", "H_NUMPER"]
    row = ["00007", "1", "255212", "6", "1", "2"]
    row[names.index(field)] = value
    paths = _fixture(tmp_path, monkeypatch, household_rows=[row])
    with pytest.raises(native.AsecNativePopulationError):
        native.load_authenticated_asec_2024_native_population(**paths)


def test_absent_interview_household_not_silently_lost(tmp_path, monkeypatch):
    paths = _fixture(
        tmp_path,
        monkeypatch,
        household_rows=[
            ["7", "1", "100", "6", "1", "2"],
            ["8", "1", "200", "6", "1", "1"],
        ],
    )
    with pytest.raises(
        native.AsecNativePopulationError, match="HOUSEHOLD_MEMBER_COUNT"
    ):
        native.load_authenticated_asec_2024_native_population(
            **paths, selected_households=((2024, 7),)
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "age",
        "unrelated",
        "group",
        "weight",
        "strata",
        "strata_flags",
        "table_flags",
        "attrs",
        "columns",
        "metadata",
        "membership",
    ],
)
def test_full_receiving_frame_mutation_refuses(tmp_path, monkeypatch, mutation):
    result = native.load_authenticated_asec_2024_native_population(
        **_fixture(tmp_path, monkeypatch)
    )
    frame = result.frame
    if mutation == "age":
        frame.person.loc[:, "A_AGE"] += 1
    elif mutation == "unrelated":
        frame.person.loc[:, "unrelated_raw_observation"] += 1
    elif mutation == "group":
        frame.table("household").loc[:, "state_fips"] += 1
    elif mutation == "weight":
        vector = frame.weights_for("household").values
        vector.setflags(write=True)
        vector[0] += 1
        vector.setflags(write=False)
    elif mutation == "strata":
        frame.strata.iloc[0] = "other"
    elif mutation == "strata_flags":
        frame.strata.flags.allows_duplicate_labels = False
    elif mutation == "table_flags":
        frame.table("household").flags.allows_duplicate_labels = False
    elif mutation == "attrs":
        frame.person.attrs["changed"] = True
    elif mutation == "columns":
        frame.person.columns.name = "changed"
    elif mutation == "metadata":
        frame._metadata = {"changed": True}
    elif mutation == "membership":
        frame.person.iloc[0, frame.person.columns.get_loc("person_tax_unit_id")] = 999
    with pytest.raises(native.AsecNativePopulationError):
        result.validate()


@pytest.mark.parametrize(
    "which",
    [
        "parent_path",
        "household_attachment_path",
        "person_income_attachment_path",
        "household_member_path",
        "older_person",
    ],
)
def test_changed_original_source_refuses_existing_borrow(tmp_path, monkeypatch, which):
    paths = _fixture(tmp_path, monkeypatch)
    result = native.load_authenticated_asec_2024_native_population(**paths)
    path = (
        paths["person_member_paths"][2022] if which == "older_person" else paths[which]
    )
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(native.AsecNativePopulationError, match="SOURCE_FILE"):
        result.validate()


def test_capsule_payload_rehash_and_live_owner_changes_refuse(tmp_path, monkeypatch):
    paths = _fixture(tmp_path, monkeypatch)
    result = native.load_authenticated_asec_2024_native_population(**paths)
    state = native._ISSUED[id(result)][2]
    payload = state.anchors.payload + b" "
    object.__setattr__(state.anchors, "payload", payload)
    object.__setattr__(
        state.anchors, "_issued_sha256", hashlib.sha256(payload).hexdigest()
    )
    with pytest.raises(native.AsecNativePopulationError):
        result.validate()
    original = restoration._implementation
    monkeypatch.setattr(restoration, "_implementation", lambda: original())
    with pytest.raises(native.AsecNativePopulationError, match="PRODUCER_CODE_CHANGED"):
        native.load_authenticated_asec_2024_native_population(**paths)


@pytest.mark.parametrize("legacy", [[100, 200, 17], [0.001, 10_000, 0.0001]])
def test_original_anchor_ignores_legacy_pooled_weights(tmp_path, monkeypatch, legacy):
    paths = _fixture(tmp_path, monkeypatch, legacy_weights=legacy)
    result = native.load_authenticated_asec_2024_native_population(**paths)
    assert result.frame.weights_for("household").values.tolist() == [2552.12]
    assert result.receipt["households"][0]["fraction"] == [63803, 25]
    assert (
        native._ISSUED[id(result)][2]
        .parent.frame.weights_for("household")
        .values.tolist()
        == legacy
    )


def test_mixed_zero_complete_households_and_selected_zero_refusal(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path, monkeypatch, extra_household=True)
    result = native.load_authenticated_asec_2024_native_population(**paths)
    assert result.frame.person.person_id.tolist() == [105, 106, 107, 108]
    assert result.frame.person.A_AGE.tolist() == [55, 14, 55, 14]
    assert result.frame.weights_for("household").values.tolist() == [2552.12, 0]
    assert [r["fraction"] for r in result.receipt["households"]] == [
        [63803, 25],
        [0, 1],
    ]
    selected = native.load_authenticated_asec_2024_native_population(
        **paths, selected_households=((2024, 7),)
    )
    assert selected.frame.person.person_id.tolist() == [105, 106]
    for entity in selected.frame.schema.group_entities:
        assert selected.frame.table(entity)[f"{entity}_id"].tolist() == [3]
    with pytest.raises(native.AsecNativePopulationError):
        native.load_authenticated_asec_2024_native_population(
            **paths, selected_households=((2024, 8),)
        )


def test_late_mutation_during_final_file_verification_refuses(tmp_path, monkeypatch):
    result = native.load_authenticated_asec_2024_native_population(
        **_fixture(tmp_path, monkeypatch)
    )
    frame = result.frame
    fired = []

    def trace(stack, event, argument):
        if (
            event == "return"
            and stack.f_code is native._file_identity.__code__
            and not fired
        ):
            fired.append(True)
            frame.person.loc[:, "A_AGE"] += 1
        return trace

    sys.settrace(trace)
    try:
        with pytest.raises(native.AsecNativePopulationError, match="FRAME_CHANGED"):
            result.validate()
    finally:
        sys.settrace(None)
    assert fired


def test_real_loaders_run_without_prepared_population_or_money_execution(
    tmp_path, monkeypatch
):
    paths = _fixture(tmp_path, monkeypatch)
    called = set()

    def trace(stack, event, argument):
        if event == "call":
            module = stack.f_globals.get("__name__", "")
            name = stack.f_code.co_name
            assert module not in (
                "microcosm.build.us_runtime.asec_prepared_source",
                "microcosm.build.us_runtime.engine",
            )
            assert name not in (
                "classify_asec_money",
                "prepare_asec_current_money_population",
            )
            if module.startswith("microcosm.build.us_runtime."):
                called.add(name)

    sys.setprofile(trace)
    try:
        native.load_authenticated_asec_2024_native_population(**paths)
    finally:
        sys.setprofile(None)
    assert {
        "load_authenticated_restored_current_money_source",
        "authenticate_asec_coverage",
        "load_authenticated_asec_household_weights",
        "load_authenticated_asec_household_coverage_fields",
    } <= called


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "substitute", "split", "age"]
)
def test_original_person_mutation_refuses_reconstruction(
    tmp_path, monkeypatch, mutation
):
    paths = _fixture(tmp_path, monkeypatch)

    def change(rows):
        if mutation == "missing":
            rows.pop()
        elif mutation == "duplicate":
            rows[2] = rows[1].copy()
        elif mutation == "substitute":
            rows[1][0] = "9" * 22
        elif mutation == "split":
            rows[1][1] = "8"
        elif mutation == "age":
            rows[1][3] = "54"

    _helpers()._rewrite(paths["person_member_paths"][2024], change)
    with pytest.raises(native.AsecNativePopulationError):
        native.load_authenticated_asec_2024_native_population(**paths)


def test_external_state_and_rehashed_population_cannot_be_rewritten(
    tmp_path, monkeypatch
):
    result = native.load_authenticated_asec_2024_native_population(
        **_fixture(tmp_path, monkeypatch)
    )
    state = native._ISSUED[id(result)][2]
    with pytest.raises(AttributeError):
        object.__setattr__(state, "frame_identity", "a" * 64)
    changed = json.loads(result.payload)
    changed["frame_sha256"] = "a" * 64
    object.__setattr__(result, "payload", native._encode(changed))
    with pytest.raises(native.AsecNativePopulationError, match="UNISSUED_OR_CHANGED"):
        result.validate()


@pytest.mark.parametrize("owner", ["anchors", "fields", "coverage"])
def test_equal_unissued_source_evidence_refuses(tmp_path, monkeypatch, owner):
    result = native.load_authenticated_asec_2024_native_population(
        **_fixture(tmp_path, monkeypatch)
    )
    state = native._ISSUED[id(result)][2]
    evidence = getattr(state, owner)
    copy = object.__new__(type(evidence))
    # Copy/deserialization cannot inherit the reader's external issuance seal.
    for name in evidence.__slots__:
        if name != "__weakref__":
            object.__setattr__(copy, name, getattr(evidence, name))
    if owner != "coverage":
        assert copy == evidence
    verifier = {
        "anchors": native.anchor_owner.verify_asec_household_weights_source,
        "fields": native.field_owner.verify_asec_household_coverage_fields,
        "coverage": lambda value: native.coverage_owner.verify_asec_coverage_parent(
            value, state.parent
        ),
    }[owner]
    with pytest.raises(ValueError):
        verifier(copy)
    result.validate()


@pytest.mark.parametrize(
    "interface", ["validate", "frame", "context", "receipt", "to_bytes"]
)
def test_late_population_payload_mutation_refuses_every_interface(
    tmp_path, monkeypatch, interface
):
    result = native.load_authenticated_asec_2024_native_population(
        **_fixture(tmp_path, monkeypatch)
    )
    fired = []

    def trace(stack, event, argument):
        if (
            event == "return"
            and stack.f_code is native._file_identity.__code__
            and not fired
        ):
            fired.append(True)
            object.__setattr__(result, "payload", b'{"release_eligible":true}')
        return trace

    sys.settrace(trace)
    try:
        with pytest.raises(native.AsecNativePopulationError):
            value = getattr(result, interface)
            if callable(value):
                value()
    finally:
        sys.settrace(None)
    assert fired


@pytest.mark.parametrize("owner", ["anchors", "fields", "coverage", "parent"])
def test_late_attached_evidence_mutation_refuses(tmp_path, monkeypatch, owner):
    result = native.load_authenticated_asec_2024_native_population(
        **_fixture(tmp_path, monkeypatch)
    )
    state = native._ISSUED[id(result)][2]
    fired = []

    def trace(stack, event, argument):
        if (
            event == "return"
            and stack.f_code is native._file_identity.__code__
            and not fired
        ):
            fired.append(True)
            if owner == "parent":
                object.__setattr__(
                    state.parent.source, "evidence", state.parent.source.evidence + b" "
                )
            elif owner == "coverage":
                object.__setattr__(state.coverage, "_body", state.coverage._body + b"x")
            else:
                source = getattr(state, owner)
                object.__setattr__(source, "payload", source.payload + b" ")

    return_trace = trace

    def tracing(stack, event, argument):
        return_trace(stack, event, argument)
        return tracing

    sys.settrace(tracing)
    try:
        with pytest.raises(native.AsecNativePopulationError):
            result.to_bytes()
    finally:
        sys.settrace(None)
    assert fired
