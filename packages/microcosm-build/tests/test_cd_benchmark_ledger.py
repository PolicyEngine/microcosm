"""Invented ledger records: chain, terminality, identity and exposure precedence.

No file is opened anywhere in this module. Every ledger is bytes built in
memory, which is what the contract requires of an implementation slice: the
real local ledger contract is future work and no production ledger exists.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

import pytest

from microcosm.build.cd_benchmark.ledger import (
    UUID4_PATTERN,
    EventType,
    ExposureHistory,
    LedgerError,
    LedgerRecord,
    build_records,
    exposed_families,
    exposure_status,
    new_evaluation_id,
    registration_claim,
    verify_ledger,
)
from microcosm.build.cd_benchmark.protocol import canonical_protocol
from microcosm.build.cd_benchmark.reasons import ExposureStatus, Reason

FIRST = "9f1c2a3b-4d5e-4f60-8a91-b2c3d4e5f607"
SECOND = "0a1b2c3d-4e5f-4061-9a2b-3c4d5e6f7081"


@pytest.fixture(scope="module")
def protocol():
    return canonical_protocol()


def record(sequence, event, payload, previous, *, ledger_id="ledger:invented"):
    return LedgerRecord(
        ledger_id=ledger_id,
        schema_version=1,
        sequence=sequence,
        recorded_at="2026-09-05T12:00:00Z",
        actor_id="actor:invented-fixture",
        event_type=event,
        payload=payload,
        previous_record_sha256=previous,
    )


def registration_payload(protocol):
    return {
        "protocol_sha256": protocol.sha256,
        "protocol_parent_sha256": protocol.parent_sha256,
        "legacy_parent_artifact_hashes_if_first_version": [],
        "parameter_changes_old_new_reason_evidence": [],
        "root_approval_artifact_sha256": "c" * 64,
        "peer_approval_artifact_sha256": "d" * 64,
        "reference_inventory_sha256": protocol.reference_inventory_sha256,
        "block_definition_hashes": dict(protocol.block_definition_hashes),
    }


def attempt_payload(protocol, evaluation_id, families=("B19001", "B25003")):
    payload = {
        name: [] if name.endswith("ids") else "invented"
        for name, _ in protocol.attempt_payload_fields
    }
    payload["evaluation_id"] = evaluation_id
    payload["attempt_state"] = "started"
    payload["all_derivative_reference_family_ids"] = list(families)
    return payload


def completion_payload(protocol, evaluation_id, attempt_sha, state="completed"):
    payload = {name: "invented" for name, _ in protocol.completion_payload_fields}
    payload["evaluation_id"] = evaluation_id
    payload["attempt_record_sha256"] = attempt_sha
    payload["terminal_state"] = state
    return payload


def interruption_payload(evaluation_id, attempt_sha, *, exposure_possible=True):
    return {
        "evaluation_id": evaluation_id,
        "attempt_record_sha256": attempt_sha,
        "interruption_reason": "writer process exited",
        "known_output_hashes": None,
        "exposure_possible": exposure_possible,
    }


def chain(protocol, *, terminal="completion", evaluation_id=FIRST):
    zero = record(
        0, EventType.PROTOCOL_REGISTERED, registration_payload(protocol), None
    )
    one = record(
        1,
        EventType.EVALUATION_ATTEMPT,
        attempt_payload(protocol, evaluation_id),
        zero.sha256,
    )
    records = [zero, one]
    if terminal == "completion":
        records.append(
            record(
                2,
                EventType.EVALUATION_COMPLETION,
                completion_payload(protocol, evaluation_id, one.sha256),
                one.sha256,
            )
        )
    elif terminal == "interrupted":
        records.append(
            record(
                2,
                EventType.EVALUATION_INTERRUPTED,
                interruption_payload(evaluation_id, one.sha256),
                one.sha256,
            )
        )
    return records


def test_a_well_formed_chain_verifies_and_snapshots(protocol):
    records = chain(protocol)
    data = build_records(records)
    assert data.endswith(b"\n")
    assert data.count(b"\n") == 3
    verified = verify_ledger(data, protocol=protocol)
    assert verified.ok
    assert verified.reasons == ()
    snapshot = verified.ledger.snapshot()
    assert snapshot.record_count == 3
    assert snapshot.tip_sequence == 2
    assert snapshot.tip_record_sha256 == records[-1].sha256
    assert snapshot.prefix_bytes == len(data)
    assert set(protocol.ledger_snapshot_fields) == {
        "ledger_id",
        "record_count",
        "tip_sequence",
        "tip_record_sha256",
        "exact_NDJSON_prefix_sha256",
        "prefix_bytes",
    }


def test_a_successor_roster_cannot_omit_a_consumed_identifier_array(protocol):
    protocol = replace(
        protocol,
        attempt_payload_fields=tuple(
            item
            for item in protocol.attempt_payload_fields
            if item[0] != "all_derivative_reference_family_ids"
        ),
    )
    records = chain(protocol)
    payload = dict(records[1].payload)
    del payload["all_derivative_reference_family_ids"]
    records[1] = replace(records[1], payload=payload)
    # The malformed attempt is reached before any downstream chain check.
    result = verify_ledger(build_records(records[:2]), protocol=protocol)
    assert result.reasons == (Reason.LEDGER_INTEGRITY_UNKNOWN,)


def test_a_record_hash_covers_the_complete_previous_record(protocol):
    records = chain(protocol)
    assert records[1].previous_record_sha256 == records[0].sha256
    assert records[0].previous_record_sha256 is None
    assert "sha256" not in records[0].as_object()
    broken = [
        records[0],
        records[1],
        record(2, EventType.EVALUATION_COMPLETION, records[2].payload, "0" * 64),
    ]
    result = verify_ledger(build_records(broken), protocol=protocol)
    assert result.reasons == (Reason.LEDGER_INTEGRITY_UNKNOWN,)
    assert "chain is broken or forked" in result.evidence["refusal"]


def test_sequences_start_at_zero_and_stay_consecutive(protocol):
    records = chain(protocol)
    skipped = [
        records[0],
        record(2, records[1].event_type, records[1].payload, records[0].sha256),
    ]
    assert verify_ledger(build_records(skipped), protocol=protocol).reasons == (
        Reason.LEDGER_INTEGRITY_UNKNOWN,
    )
    nonnull_root = [
        record(
            0, EventType.PROTOCOL_REGISTERED, registration_payload(protocol), "a" * 64
        )
    ]
    assert not verify_ledger(build_records(nonnull_root), protocol=protocol).ok


def test_an_attempt_takes_exactly_one_terminal_record(protocol):
    records = chain(protocol)
    duplicate = [
        *records,
        record(
            3, EventType.EVALUATION_COMPLETION, records[2].payload, records[2].sha256
        ),
    ]
    result = verify_ledger(build_records(duplicate), protocol=protocol)
    assert result.reasons == (Reason.LEDGER_INTEGRITY_UNKNOWN,)
    assert "exactly one terminal record" in result.evidence["refusal"]


def test_an_interrupted_attempt_cannot_be_retroactively_completed(protocol):
    interrupted = chain(protocol, terminal="interrupted")
    retro = [
        *interrupted,
        record(
            3,
            EventType.EVALUATION_COMPLETION,
            completion_payload(protocol, FIRST, interrupted[1].sha256),
            interrupted[2].sha256,
        ),
    ]
    result = verify_ledger(build_records(retro), protocol=protocol)
    assert not result.ok
    assert "exactly one terminal record" in result.evidence["refusal"]
    just_interrupted = verify_ledger(build_records(interrupted), protocol=protocol)
    assert just_interrupted.ok
    state = just_interrupted.ledger.attempts[FIRST]
    assert state.interrupted is True
    assert state.unfinished is False
    assert state.terminal_state == "interrupted"


def test_a_second_interruption_is_refused(protocol):
    interrupted = chain(protocol, terminal="interrupted")
    twice = [
        *interrupted,
        record(
            3,
            EventType.EVALUATION_INTERRUPTED,
            interruption_payload(FIRST, interrupted[1].sha256),
            interrupted[2].sha256,
        ),
    ]
    assert not verify_ledger(build_records(twice), protocol=protocol).ok


def test_an_interruption_must_record_that_exposure_is_possible(protocol):
    records = chain(protocol)[:2]
    denied = [
        *records,
        record(
            2,
            EventType.EVALUATION_INTERRUPTED,
            interruption_payload(FIRST, records[1].sha256, exposure_possible=False),
            records[1].sha256,
        ),
    ]
    assert not verify_ledger(build_records(denied), protocol=protocol).ok


def test_a_terminal_record_must_bind_its_attempt_hash(protocol):
    records = chain(protocol)[:2]
    mismatched = [
        *records,
        record(
            2,
            EventType.EVALUATION_COMPLETION,
            completion_payload(protocol, FIRST, "b" * 64),
            records[1].sha256,
        ),
    ]
    result = verify_ledger(build_records(mismatched), protocol=protocol)
    assert "attempt hash does not bind" in result.evidence["refusal"]


def test_a_terminal_record_without_an_attempt_is_refused(protocol):
    zero = record(
        0, EventType.PROTOCOL_REGISTERED, registration_payload(protocol), None
    )
    orphan = [
        zero,
        record(
            1,
            EventType.EVALUATION_COMPLETION,
            completion_payload(protocol, FIRST, "b" * 64),
            zero.sha256,
        ),
    ]
    assert not verify_ledger(build_records(orphan), protocol=protocol).ok


def test_duplicate_evaluation_ids_are_refused_within_and_across_histories(protocol):
    records = chain(protocol)
    repeated = [
        *records,
        record(
            3,
            EventType.EVALUATION_ATTEMPT,
            attempt_payload(protocol, FIRST),
            records[2].sha256,
        ),
    ]
    assert not verify_ledger(build_records(repeated), protocol=protocol).ok
    linked = verify_ledger(
        build_records(records),
        protocol=protocol,
        linked_evaluation_ids=frozenset({FIRST}),
    )
    assert not linked.ok
    assert "Duplicate evaluation_id" in linked.evidence["refusal"]


@pytest.mark.parametrize(
    "evaluation_id",
    [
        FIRST.upper(),
        "9f1c2a3b-4d5e-1f60-8a91-b2c3d4e5f607",
        "9f1c2a3b-4d5e-4f60-ca91-b2c3d4e5f607",
        "9f1c2a3b4d5e4f608a91b2c3d4e5f607",
    ],
)
def test_only_a_lowercase_canonical_uuid4_identifies_an_attempt(
    protocol, evaluation_id
):
    zero = record(
        0, EventType.PROTOCOL_REGISTERED, registration_payload(protocol), None
    )
    bad = [
        zero,
        record(
            1,
            EventType.EVALUATION_ATTEMPT,
            attempt_payload(protocol, evaluation_id),
            zero.sha256,
        ),
    ]
    assert not verify_ledger(build_records(bad), protocol=protocol).ok
    assert UUID4_PATTERN.fullmatch(evaluation_id) is None


def test_generated_identifiers_are_checked_for_shape_and_collision():
    generated = new_evaluation_id([], generator=lambda: uuid.UUID(FIRST))
    assert generated == FIRST
    with pytest.raises(LedgerError):
        new_evaluation_id([FIRST], generator=lambda: uuid.UUID(FIRST))
    with pytest.raises(LedgerError):
        new_evaluation_id([], generator=lambda: uuid.UUID(int=0, version=1))
    assert UUID4_PATTERN.fullmatch(str(uuid.uuid4())) is not None


def test_payload_keys_are_closed_per_event(protocol):
    payload = registration_payload(protocol)
    for mutate in (
        lambda item: {**item, "extra": 1},
        lambda item: {k: v for k, v in item.items() if k != "protocol_sha256"},
    ):
        bad = [record(0, EventType.PROTOCOL_REGISTERED, mutate(payload), None)]
        assert not verify_ledger(build_records(bad), protocol=protocol).ok


def test_pinned_payload_values_are_enforced(protocol):
    zero = record(
        0, EventType.PROTOCOL_REGISTERED, registration_payload(protocol), None
    )
    started = attempt_payload(protocol, FIRST)
    started["attempt_state"] = "running"
    assert not verify_ledger(
        build_records(
            [zero, record(1, EventType.EVALUATION_ATTEMPT, started, zero.sha256)]
        ),
        protocol=protocol,
    ).ok
    records = chain(protocol)[:2]
    wrong = completion_payload(protocol, FIRST, records[1].sha256, state="finished")
    assert not verify_ledger(
        build_records(
            [
                *records,
                record(2, EventType.EVALUATION_COMPLETION, wrong, records[1].sha256),
            ]
        ),
        protocol=protocol,
    ).ok


def test_unknown_record_fields_and_event_types_are_refused(protocol):
    valid = build_records(chain(protocol))
    lines = valid.split(b"\n")
    tampered = lines[0].replace(b'"actor_id"', b'"actor"') + b"\n"
    assert not verify_ledger(tampered, protocol=protocol).ok
    unknown_event = (
        lines[0].replace(
            b'"event_type":"protocol_registered"', b'"event_type":"protocol_blessed"'
        )
        + b"\n"
    )
    assert not verify_ledger(unknown_event, protocol=protocol).ok


def test_a_line_must_be_its_own_canonical_serialization(protocol):
    valid = build_records(chain(protocol))
    spaced = valid.replace(b'{"actor_id"', b'{ "actor_id"', 1)
    assert not verify_ledger(spaced, protocol=protocol).ok


def test_a_partial_tail_or_byte_order_mark_is_never_truncated_away(protocol):
    valid = build_records(chain(protocol))
    partial = verify_ledger(valid[:-1], protocol=protocol)
    assert partial.reasons == (Reason.LEDGER_INTEGRITY_UNKNOWN,)
    assert partial.evidence["partial_or_corrupt_tail"] is True
    marked = verify_ledger(b"\xef\xbb\xbf" + valid, protocol=protocol)
    assert marked.evidence["byte_order_mark"] is True
    truncated = verify_ledger(
        b"".join(line + b"\n" for line in valid.split(b"\n")[:1]), protocol=protocol
    )
    assert truncated.ok, "a shorter valid prefix is a different file, not a repair"


def test_a_non_utc_timestamp_is_refused(protocol):
    payload = registration_payload(protocol)
    shifted = LedgerRecord(
        ledger_id="ledger:invented",
        schema_version=1,
        sequence=0,
        recorded_at="2026-09-05T12:00:00+01:00",
        actor_id="actor",
        event_type=EventType.PROTOCOL_REGISTERED,
        payload=payload,
        previous_record_sha256=None,
    )
    assert not verify_ledger(build_records([shifted]), protocol=protocol).ok


def test_one_ledger_has_one_stable_identifier(protocol):
    records = chain(protocol)
    renamed = [
        records[0],
        record(
            1,
            EventType.EVALUATION_ATTEMPT,
            records[1].payload,
            records[0].sha256,
            ledger_id="ledger:other",
        ),
    ]
    assert not verify_ledger(build_records(renamed), protocol=protocol).ok


def test_prior_exposure_does_not_need_a_completed_score(protocol):
    started_only = verify_ledger(build_records(chain(protocol)[:2]), protocol=protocol)
    assert started_only.ok
    assert exposed_families(started_only.ledger) == {"B19001", "B25003"}
    interrupted = verify_ledger(
        build_records(chain(protocol, terminal="interrupted")), protocol=protocol
    )
    assert exposed_families(interrupted.ledger) == {"B19001", "B25003"}


def test_external_exposure_records_also_mark_families(protocol):
    zero = record(
        0, EventType.PROTOCOL_REGISTERED, registration_payload(protocol), None
    )
    external = record(
        1,
        EventType.EXTERNAL_EXPOSURE_RECORDED,
        {
            "prior_evaluation_ids": [SECOND],
            "related_ancestry": "invented tuning family",
            "block_families": ["B25003"],
            "occurred_at": "2026-08-01T00:00:00Z",
            "artifacts": [],
        },
        zero.sha256,
    )
    verified = verify_ledger(build_records([zero, external]), protocol=protocol)
    assert verified.ok
    assert exposed_families(verified.ledger) == {"B25003"}


def test_exposure_precedence_puts_unknown_history_first(protocol):
    incomplete = ExposureHistory(True, False, True, frozenset({"B19001"}))
    for block in protocol.blocks:
        status, reasons = exposure_status(block, incomplete)
        assert status is ExposureStatus.EXPOSURE_HISTORY_UNKNOWN
        assert reasons == (Reason.EXPOSURE_HISTORY_UNKNOWN,)


def test_unreserved_s0101_blocks_never_claim_a_holdout(protocol):
    complete = ExposureHistory(True, True, True, frozenset())
    for block_id in ("population_total", "age_distribution"):
        status, reasons = exposure_status(protocol.block(block_id), complete)
        assert status is ExposureStatus.NOT_RESERVED_CONFORMITY
        assert reasons == ()


def test_reserved_blocks_are_fresh_only_on_a_clean_first_attempt(protocol):
    clean = ExposureHistory(True, True, True, frozenset())
    exposed = ExposureHistory(True, True, True, frozenset({"B19001"}))
    for block_id in ("income_distribution", "occupied_household_total"):
        block = protocol.block(block_id)
        assert (
            exposure_status(block, clean)[0]
            is ExposureStatus.FRESH_SAME_SOURCE_HOLDOUT_ELIGIBLE
        )
        assert (
            exposure_status(block, exposed)[0]
            is ExposureStatus.POST_EXPOSURE_CONFORMITY
        )
    tenure = protocol.block("tenure_distribution")
    assert (
        exposure_status(tenure, exposed)[0]
        is ExposureStatus.FRESH_SAME_SOURCE_HOLDOUT_ELIGIBLE
    )
    both = ExposureHistory(True, True, True, frozenset({"B19001", "B25003"}))
    assert exposure_status(tenure, both)[0] is ExposureStatus.POST_EXPOSURE_CONFORMITY


def test_the_occupied_household_block_inherits_both_reserved_families(protocol):
    block = protocol.block("occupied_household_total")
    for family in ("B19001", "B25003"):
        history = ExposureHistory(True, True, True, frozenset({family}))
        assert (
            exposure_status(block, history)[0]
            is ExposureStatus.POST_EXPOSURE_CONFORMITY
        )


def test_a_registration_record_becomes_a_claim_but_never_an_authority(protocol):
    from microcosm.build.cd_benchmark.approval import (
        ApprovalAuthority,
        ApprovalRecord,
        ApprovalScope,
        ApprovalVerdict,
        ReviewerRole,
        resolve_action,
    )

    records = chain(protocol)
    verified = verify_ledger(build_records(records), protocol=protocol)
    claim = registration_claim(verified.ledger.records[0])
    assert claim.protocol_sha256 == protocol.sha256
    assert claim.reference_inventory_sha256 == protocol.reference_inventory_sha256
    # The payload names digests "c"*64 and "d"*64. No authority holds them, so
    # the ledger line cannot authorize itself.
    empty = resolve_action(
        protocol,
        authority=ApprovalAuthority.empty(),
        action=ApprovalScope.INVENTED_FIXTURE_IMPLEMENTATION_ONLY,
        registration=claim,
    )
    assert empty.authorized is False
    assert Reason.PROTOCOL_UNAPPROVED in empty.reasons
    # Only an authority actually holding matching records resolves it, and the
    # payload's digests are not those records' digests.
    root = ApprovalRecord(
        protocol.sha256,
        ApprovalVerdict.APPROVE,
        ReviewerRole.ROOT,
        "root",
        frozenset({ApprovalScope.INVENTED_FIXTURE_IMPLEMENTATION_ONLY}),
        "c" * 64,
    )
    peer = ApprovalRecord(
        protocol.sha256,
        ApprovalVerdict.APPROVE,
        ReviewerRole.PEER,
        "peer",
        frozenset({ApprovalScope.INVENTED_FIXTURE_IMPLEMENTATION_ONLY}),
        "d" * 64,
    )
    mismatched = resolve_action(
        protocol,
        authority=ApprovalAuthority(records=(root, peer)),
        action=ApprovalScope.INVENTED_FIXTURE_IMPLEMENTATION_ONLY,
        registration=claim,
    )
    assert mismatched.authorized is False
    assert mismatched.evidence["root_artifact_unresolved"] is True


def test_only_a_registration_record_yields_a_claim(protocol):
    records = chain(protocol)
    verified = verify_ledger(build_records(records), protocol=protocol)
    with pytest.raises(LedgerError):
        registration_claim(verified.ledger.records[1])


def test_a_previous_hash_must_be_a_lowercase_digest(protocol):
    records = chain(protocol)
    uppercased = build_records(records).replace(
        records[0].sha256.encode(), records[0].sha256.upper().encode()
    )
    assert not verify_ledger(uppercased, protocol=protocol).ok


def external_exposure_payload(**overrides):
    payload = {
        "prior_evaluation_ids": [SECOND],
        "related_ancestry": "invented tuning family",
        "block_families": ["B25003"],
        "occurred_at": "2026-08-01T00:00:00Z",
        "artifacts": [],
    }
    payload.update(overrides)
    return payload


def test_a_family_list_is_typed_and_never_a_string_iterated_per_character(protocol):
    """A bare string cannot pass as a family array and manufacture freshness."""
    zero = record(
        0, EventType.PROTOCOL_REGISTERED, registration_payload(protocol), None
    )
    listed = record(
        1,
        EventType.EVALUATION_ATTEMPT,
        attempt_payload(protocol, FIRST, families=("B19001",)),
        zero.sha256,
    )
    baseline = verify_ledger(build_records([zero, listed]), protocol=protocol)
    assert baseline.ok
    assert exposed_families(baseline.ledger) == {"B19001"}
    bare = attempt_payload(protocol, FIRST)
    bare["all_derivative_reference_family_ids"] = "B19001"
    result = verify_ledger(
        build_records(
            [zero, record(1, EventType.EVALUATION_ATTEMPT, bare, zero.sha256)]
        ),
        protocol=protocol,
    )
    assert not result.ok
    assert result.reasons == (Reason.LEDGER_INTEGRITY_UNKNOWN,)
    assert "list of nonempty strings" in result.evidence["refusal"]


@pytest.mark.parametrize(
    "field", ["all_derivative_reference_family_ids", "tuning_family_root_ids"]
)
@pytest.mark.parametrize("value", ["B19001", ["B19001", 4], ["B19001", ""], {"a": 1}])
def test_attempt_identifier_lists_admit_only_nonempty_strings(protocol, field, value):
    zero = record(
        0, EventType.PROTOCOL_REGISTERED, registration_payload(protocol), None
    )
    payload = attempt_payload(protocol, FIRST)
    payload[field] = value
    assert not verify_ledger(
        build_records(
            [zero, record(1, EventType.EVALUATION_ATTEMPT, payload, zero.sha256)]
        ),
        protocol=protocol,
    ).ok


@pytest.mark.parametrize("field", ["block_families", "prior_evaluation_ids"])
def test_external_exposure_identifier_lists_are_typed(protocol, field):
    zero = record(
        0, EventType.PROTOCOL_REGISTERED, registration_payload(protocol), None
    )
    baseline = verify_ledger(
        build_records(
            [
                zero,
                record(
                    1,
                    EventType.EXTERNAL_EXPOSURE_RECORDED,
                    external_exposure_payload(),
                    zero.sha256,
                ),
            ]
        ),
        protocol=protocol,
    )
    assert baseline.ok
    assert exposed_families(baseline.ledger) == {"B25003"}
    result = verify_ledger(
        build_records(
            [
                zero,
                record(
                    1,
                    EventType.EXTERNAL_EXPOSURE_RECORDED,
                    external_exposure_payload(**{field: "B25003"}),
                    zero.sha256,
                ),
            ]
        ),
        protocol=protocol,
    )
    assert not result.ok
    assert "list of nonempty strings" in result.evidence["refusal"]


def test_a_new_attempt_waits_for_the_prior_attempt_to_be_terminal(protocol):
    """An interruption or a completion is appended before the next attempt starts."""
    for terminal in ("completion", "interrupted"):
        prior = chain(protocol, terminal=terminal)
        following = [
            *prior,
            record(
                3,
                EventType.EVALUATION_ATTEMPT,
                attempt_payload(protocol, SECOND),
                prior[2].sha256,
            ),
        ]
        verified = verify_ledger(build_records(following), protocol=protocol)
        assert verified.ok, terminal
        assert set(verified.ledger.attempts) == {FIRST, SECOND}
        assert verified.ledger.attempts[SECOND].unfinished is True
    started = chain(protocol)[:2]
    overlapping = [
        *started,
        record(
            2,
            EventType.EVALUATION_ATTEMPT,
            attempt_payload(protocol, SECOND),
            started[1].sha256,
        ),
    ]
    result = verify_ledger(build_records(overlapping), protocol=protocol)
    assert not result.ok
    assert result.reasons == (Reason.LEDGER_INTEGRITY_UNKNOWN,)
    assert "unfinished" in result.evidence["refusal"]


def test_a_verified_ledger_surfaces_its_unfinished_attempt(protocol):
    started = verify_ledger(build_records(chain(protocol)[:2]), protocol=protocol)
    assert started.ok
    assert started.evidence["unfinished_attempt_ids"] == (FIRST,)
    finished = verify_ledger(build_records(chain(protocol)), protocol=protocol)
    assert finished.evidence["unfinished_attempt_ids"] == ()
