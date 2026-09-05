"""Source universe, candidate bindings, and the one permitted scope reduction.

The ACS and ASEC rows here are invented and classified by the adapters the
contract pins, so the membership rules under test are the shipped ones. No
candidate, population or Census artifact is read.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from microcosm.build.cd_benchmark.approval import (
    ApprovalAuthority,
    ApprovalRecord,
    ApprovalScope,
    ApprovalVerdict,
    RegistrationClaim,
    ReviewerRole,
    resolve_action,
)
from microcosm.build.cd_benchmark.binding import (
    ArtifactRef,
    BindingError,
    CandidateBinding,
    ProducerAuthenticator,
    resolve_binding,
)
from microcosm.build.cd_benchmark.evaluation import (
    CLAIMS_NOT_ESTABLISHED,
    BenchmarkEvaluation,
    district_scope_reasons,
    evaluate_block,
    evaluate_scope,
)
from microcosm.build.cd_benchmark.metrics import point_check
from microcosm.build.cd_benchmark.protocol import canonical_protocol
from microcosm.build.cd_benchmark.reasons import (
    AXES,
    BenchmarkVerdict,
    ExposureStatus,
    MappingStatus,
    Reason,
)
from microcosm.build.cd_benchmark.universe import (
    GqWeightBridge,
    UniverseError,
    VacancyBridge,
    acs_housing_universe,
    asec_housing_universe,
    gq_weight_bridge_reasons,
)

FIXTURE_SCOPE = ApprovalScope.INVENTED_FIXTURE_IMPLEMENTATION_ONLY


@pytest.fixture(scope="module")
def protocol():
    return canonical_protocol()


class FixtureAuthenticator:
    """A test-only authenticator. The package deliberately ships none."""

    def __init__(self, resolved: set[str]):
        self._resolved = resolved

    def authenticates(self, slot: str, reference: ArtifactRef) -> bool:
        return slot in self._resolved


def acs_frame(type_hugq, persons, tenure):
    return pd.DataFrame(
        {
            "TYPEHUGQ": np.array(type_hugq, dtype=np.int64),
            "NP": np.array(persons, dtype=np.int64),
            "TEN": np.array(tenure, dtype=np.int64),
        }
    )


def asec_frame(hhtype, livqrt, hrhtype, tenure):
    size = len(hhtype)
    return pd.DataFrame(
        {
            "household_id": np.arange(1, size + 1, dtype=np.int64),
            "income_year": np.full(size, 2024, dtype=np.int64),
            "H_SEQ": np.arange(1, size + 1, dtype=np.int64),
            "H_HHTYPE": np.array(hhtype, dtype=np.int64),
            "H_LIVQRT": np.array(livqrt, dtype=np.int64),
            "HRHTYPE": np.array(hrhtype, dtype=np.int64),
            "H_TENURE": np.array(tenure, dtype=np.int64),
        }
    )


def bridge():
    return VacancyBridge(
        definition_id="vacancy-definition/1",
        definition_sha256="a" * 64,
        classifier_implementation_sha256="b" * 64,
        vacancy_artifact_id="vacancy-artifact/1",
        vacancy_artifact_sha256="c" * 64,
        source_custody_sha256="d" * 64,
    )


def test_acs_membership_and_tenure_come_from_the_pinned_adapter():
    frame = acs_frame(
        [1, 1, 1, 1, 2, 3, 1], [2, 1, 3, 0, 1, 1, 4], [1, 3, 2, 0, 0, 0, 9]
    )
    valid = np.array([True, True, True, False, False, False, False])
    result = acs_housing_universe(
        frame, tenure_valid=valid, vacant_household_rows_dropped=0
    )
    assert (result.occupied, result.outside, result.occupancy_unresolved) == (4, 2, 1)
    assert (result.owner, result.renter, result.tenure_unclassified) == (2, 1, 1)
    assert result.evidence["adapter_source_authentication"] == (
        "unbound_semantic_classifier"
    )


def test_a_retained_np_zero_row_is_denominator_unknown_not_vacant():
    frame = acs_frame([1, 1], [2, 0], [1, 0])
    valid = np.array([True, False])
    retained = acs_housing_universe(
        frame, tenure_valid=valid, vacant_household_rows_dropped=0
    )
    assert retained.occupancy_unresolved == 1
    assert retained.reasons == (Reason.DENOMINATOR_UNKNOWN,)
    assert Reason.VACANCY_BRIDGE_MISSING not in retained.reasons
    assert retained.evidence["np_zero_asserts_no_vacancy"] is True
    bridged = acs_housing_universe(
        frame,
        tenure_valid=valid,
        vacant_household_rows_dropped=0,
        vacancy_bridge=bridge(),
    )
    assert bridged.reasons == ()


def test_dropping_np_zero_rows_without_a_bridge_is_a_different_refusal():
    frame = acs_frame([1, 1], [2, 3], [1, 3])
    valid = np.array([True, True])
    dropped = acs_housing_universe(
        frame, tenure_valid=valid, vacant_household_rows_dropped=7
    )
    assert dropped.reasons == (Reason.VACANCY_BRIDGE_MISSING,)
    unknown = acs_housing_universe(
        frame, tenure_valid=valid, vacant_household_rows_dropped=None
    )
    assert unknown.reasons == (Reason.UNRESOLVED_SOURCE_LINEAGE,)
    known_zero = acs_housing_universe(
        frame, tenure_valid=valid, vacant_household_rows_dropped=0
    )
    assert known_zero.reasons == ()
    assert known_zero.evidence["vacant_household_rows_dropped"] == 0


def test_an_out_of_domain_source_code_refuses_the_whole_batch():
    frame = acs_frame([1, 4], [2, 1], [1, 1])
    valid = np.array([True, True])
    refused = acs_housing_universe(
        frame, tenure_valid=valid, vacant_household_rows_dropped=0
    )
    assert refused.reasons == (Reason.SOURCE_DOMAIN_REFUSED,)
    assert refused.occupied == 0
    assert refused.tenure_unclassified == 0
    assert refused.evidence["no_fabricated_classification"] is True


def test_asec_membership_follows_the_three_native_axes():
    frame = asec_frame(
        [1, 1, 1, 2, 1], [1, 2, 9, 1, 1], [1, 2, 9, 0, 3], [1, 2, 0, 0, 3]
    )
    result = asec_housing_universe(frame)
    assert (result.occupied, result.outside, result.occupancy_unresolved) == (3, 1, 1)
    assert (result.owner, result.renter, result.tenure_unclassified) == (1, 2, 0)
    assert result.reasons == (Reason.DENOMINATOR_UNKNOWN,)
    assert result.evidence["release_eligible"] is False


def test_an_out_of_domain_asec_tenure_is_refused_not_made_unclassified():
    frame = asec_frame([1, 1], [1, 1], [1, 3], [1, 4])
    result = asec_housing_universe(frame)
    assert result.reasons == (Reason.SOURCE_DOMAIN_REFUSED,)
    assert result.evidence["adapter_refusal"] == "CONTEXT_DOMAIN"
    assert result.tenure_unclassified == 0
    assert result.occupied == 0


def test_a_domain_valid_asec_tenure_zero_is_unclassified_within_a_known_hu():
    frame = asec_frame([1, 1], [1, 1], [1, 3], [1, 0])
    result = asec_housing_universe(frame)
    assert result.occupied == 2
    assert result.tenure_unclassified == 1
    assert result.unresolved_reason_bits[64] == 1
    assert Reason.SOURCE_DOMAIN_REFUSED not in result.reasons


def test_hu_quarters_with_a_gq_kind_is_never_known_outside():
    frame = asec_frame([1], [1], [9], [0])
    result = asec_housing_universe(frame)
    assert result.occupied == 0
    assert result.outside == 0
    assert result.occupancy_unresolved == 1
    assert result.unresolved_reason_bits[16] == 1
    assert result.reasons == (Reason.DENOMINATOR_UNKNOWN,)


def test_the_gq_weight_bridge_is_an_external_predicate_with_no_default():
    assert gq_weight_bridge_reasons(None) == (Reason.GQ_WEIGHT_BRIDGE_MISSING,)
    supplied = GqWeightBridge(
        weight_definition_id="gq-weights/1",
        weight_definition_sha256="a" * 64,
        person_to_placeholder_mapping_sha256="b" * 64,
        custody_manifest_sha256="c" * 64,
    )
    assert gq_weight_bridge_reasons(supplied) == ()
    with pytest.raises(UniverseError):
        GqWeightBridge("gq-weights/1", "short", "b" * 64, "c" * 64)


def test_filled_slots_are_caller_evidence_not_authentication(protocol):
    full = CandidateBinding(
        slots={
            name: ArtifactRef(f"artifact:{name}", "a" * 64)
            for name in protocol.required_binding_slots
        }
    )
    unauthenticated = resolve_binding(protocol, full)
    assert unauthenticated.status is MappingStatus.UNAPPROVED
    assert unauthenticated.missing_slots == ()
    assert len(unauthenticated.unauthenticated_slots) == 21
    assert Reason.MAPPING_UNAPPROVED in unauthenticated.reasons
    assert unauthenticated.evidence["authenticator_injected"] is False
    authenticated = resolve_binding(
        protocol,
        full,
        authenticator=FixtureAuthenticator(set(protocol.required_binding_slots)),
    )
    assert authenticated.status is MappingStatus.APPROVED
    assert authenticated.reasons == ()
    assert isinstance(FixtureAuthenticator(set()), ProducerAuthenticator)


def test_missing_slots_report_the_axis_they_belong_to(protocol):
    empty = resolve_binding(protocol, CandidateBinding.empty())
    assert empty.status is MappingStatus.UNAPPROVED
    assert len(empty.missing_slots) == 21
    assert set(empty.reasons) == {
        Reason.CANDIDATE_IDENTITY_UNAVAILABLE,
        Reason.UNRESOLVED_SOURCE_LINEAGE,
        Reason.MAPPING_UNAPPROVED,
        Reason.EXPOSURE_HISTORY_UNKNOWN,
    }


def test_the_income_slot_is_required_only_when_income_is_claimed(protocol):
    slots = {
        name: ArtifactRef(f"artifact:{name}", "a" * 64)
        for name in protocol.required_binding_slots
        if name != "income_eight_component_definition_id_and_sha256_when_income_claimed"
    }
    authenticator = FixtureAuthenticator(set(slots))
    claimed = resolve_binding(
        protocol, CandidateBinding(slots=slots), authenticator=authenticator
    )
    assert claimed.missing_slots == (
        "income_eight_component_definition_id_and_sha256_when_income_claimed",
    )
    not_claimed = resolve_binding(
        protocol,
        CandidateBinding(slots=slots, income_claimed=False),
        authenticator=authenticator,
    )
    assert not_claimed.missing_slots == ()
    assert not_claimed.status is MappingStatus.APPROVED


def test_an_unknown_slot_is_not_quietly_accepted(protocol):
    binding = CandidateBinding(slots={"looks_official": ArtifactRef("x", "a" * 64)})
    result = resolve_binding(protocol, binding)
    assert result.unknown_slots == ("looks_official",)
    assert Reason.MAPPING_UNAPPROVED in result.reasons


def test_a_slot_reference_must_be_an_artifact_not_a_label(protocol):
    with pytest.raises(BindingError):
        ArtifactRef("candidate", "not-a-digest")
    with pytest.raises(BindingError):
        CandidateBinding(slots={"candidate_id": "candidate-2024-a"})


def test_the_slot_partition_must_match_the_document(protocol):
    drifted = dataclasses.replace(
        protocol, required_binding_slots=(*protocol.required_binding_slots, "extra")
    )
    with pytest.raises(BindingError):
        resolve_binding(drifted, CandidateBinding.empty())


def authority(protocol, scopes):
    root = ApprovalRecord(
        protocol.sha256,
        ApprovalVerdict.APPROVE,
        ReviewerRole.ROOT,
        "root:invented-fixture",
        scopes,
        "c" * 64,
    )
    peer = ApprovalRecord(
        protocol.sha256,
        ApprovalVerdict.APPROVE,
        ReviewerRole.PEER,
        "peer:invented-fixture",
        scopes,
        "d" * 64,
    )
    registration = RegistrationClaim(
        protocol_sha256=protocol.sha256,
        protocol_parent_sha256=protocol.parent_sha256,
        legacy_parent_artifact_hashes=(),
        parameter_changes=(),
        root_approval_artifact_sha256=root.sha256,
        peer_approval_artifact_sha256=peer.sha256,
        reference_inventory_sha256=protocol.reference_inventory_sha256,
        block_definition_hashes=dict(protocol.block_definition_hashes),
    )
    return ApprovalAuthority(records=(root, peer)), registration


def conforming_block(protocol):
    return evaluate_block(
        block_id="population_total",
        geography_id="5001900US0601",
        metric=point_check(
            protocol.block("population_total"), candidate=100, reference=100
        ),
        exposure=ExposureStatus.NOT_RESERVED_CONFORMITY,
    )


def authenticated_binding(protocol):
    slots = {
        name: ArtifactRef(f"artifact:{name}", "a" * 64)
        for name in protocol.required_binding_slots
    }
    return resolve_binding(
        protocol,
        CandidateBinding(slots=slots),
        authenticator=FixtureAuthenticator(set(slots)),
    )


def test_implementation_approval_never_permits_scoring(protocol):
    injected, registration = authority(protocol, frozenset({FIXTURE_SCOPE}))
    approval = resolve_action(
        protocol,
        authority=injected,
        action=ApprovalScope.CANDIDATE_SCORING,
        registration=registration,
    )
    result = evaluate_scope(
        protocol,
        approval=approval,
        binding=authenticated_binding(protocol),
        blocks=[conforming_block(protocol)],
    )
    assert result.scoring_permitted is False
    assert result.verdict is BenchmarkVerdict.SCOPE_INCOMPLETE
    assert Reason.PROTOCOL_UNAPPROVED in result.reasons
    # The measured block result is still there to inspect.
    assert result.blocks[0].metric.within_bound is True


def test_the_six_axes_are_reported_separately(protocol):
    injected, registration = authority(
        protocol, frozenset({ApprovalScope.CANDIDATE_SCORING})
    )
    approval = resolve_action(
        protocol,
        authority=injected,
        action=ApprovalScope.CANDIDATE_SCORING,
        registration=registration,
    )
    result = evaluate_scope(
        protocol,
        approval=approval,
        binding=authenticated_binding(protocol),
        blocks=[conforming_block(protocol)],
    )
    assert result.scoring_permitted is True
    assert result.verdict is BenchmarkVerdict.BENCHMARK_CONFORMING
    assert set(result.axes) == set(AXES) == set(protocol.independent_axes)
    assert result.axes["mapping_status"] == str(MappingStatus.APPROVED)


def test_an_unapproved_mapping_alone_makes_the_scope_incomplete(protocol):
    injected, registration = authority(
        protocol, frozenset({ApprovalScope.CANDIDATE_SCORING})
    )
    approval = resolve_action(
        protocol,
        authority=injected,
        action=ApprovalScope.CANDIDATE_SCORING,
        registration=registration,
    )
    result = evaluate_scope(
        protocol,
        approval=approval,
        binding=resolve_binding(protocol, CandidateBinding.empty()),
        blocks=[conforming_block(protocol)],
    )
    assert result.verdict is BenchmarkVerdict.SCOPE_INCOMPLETE
    assert result.axes["mapping_status"] == str(MappingStatus.UNAPPROVED)


def test_nonconformity_is_retained_under_an_incomplete_scope(protocol):
    injected, registration = authority(
        protocol, frozenset({ApprovalScope.CANDIDATE_SCORING})
    )
    approval = resolve_action(
        protocol,
        authority=injected,
        action=ApprovalScope.CANDIDATE_SCORING,
        registration=registration,
    )
    failing = evaluate_block(
        block_id="population_total",
        geography_id="5001900US0601",
        metric=point_check(
            protocol.block("population_total"), candidate=110, reference=100
        ),
        exposure=ExposureStatus.NOT_RESERVED_CONFORMITY,
    )
    scoped = evaluate_scope(
        protocol,
        approval=approval,
        binding=authenticated_binding(protocol),
        blocks=[failing],
        scope_reasons=(Reason.REQUIRED_CELL_OR_DISTRICT_MISSING,),
    )
    assert scoped.verdict is BenchmarkVerdict.SCOPE_INCOMPLETE
    assert Reason.PUBLISHED_BENCHMARK_NONCONFORMITY in scoped.reasons
    assert scoped.evidence["nonconformities_retained"] == ("population_total",)
    without_scope_gap = evaluate_scope(
        protocol,
        approval=approval,
        binding=authenticated_binding(protocol),
        blocks=[failing],
    )
    assert without_scope_gap.verdict is BenchmarkVerdict.BENCHMARK_NONCONFORMING


def test_a_busy_ledger_writer_stops_scoring_before_it_starts(protocol):
    injected, registration = authority(
        protocol, frozenset({ApprovalScope.CANDIDATE_SCORING})
    )
    approval = resolve_action(
        protocol,
        authority=injected,
        action=ApprovalScope.CANDIDATE_SCORING,
        registration=registration,
    )
    result = evaluate_scope(
        protocol,
        approval=approval,
        binding=authenticated_binding(protocol),
        blocks=[conforming_block(protocol)],
        ledger_writer_available=False,
    )
    assert Reason.LEDGER_BUSY in result.reasons
    assert result.verdict is BenchmarkVerdict.SCOPE_INCOMPLETE
    assert result.evidence["no_ledger_created_or_appended"] is True


def test_no_field_of_an_evaluation_reads_as_a_release_verdict():
    names = {item.name for item in dataclasses.fields(BenchmarkEvaluation)}
    forbidden = ("release", "eligib", "certif", "passed", "publish", "launch")
    assert not [name for name in names if any(bad in name for bad in forbidden)]
    assert "launch or release approval" in CLAIMS_NOT_ESTABLISHED
    assert "population truth" in CLAIMS_NOT_ESTABLISHED


def test_one_missing_district_is_enough_to_break_the_scope(protocol):
    required = {"5001900US0601", "5001900US0602", protocol.scope.dc_canonical_id}
    published = {protocol.scope.dc_canonical_id: protocol.scope.dc_published_id}
    complete, notes = district_scope_reasons(
        dataclasses.replace(protocol.scope, district_count=3),
        required_geography_ids=required,
        present_geography_ids=required,
        published_geography_ids=published,
    )
    assert complete == ()
    assert notes["missing_districts"] == ()
    missing, notes = district_scope_reasons(
        dataclasses.replace(protocol.scope, district_count=3),
        required_geography_ids=required,
        present_geography_ids=required - {"5001900US0602"},
        published_geography_ids=published,
    )
    assert missing == (Reason.REQUIRED_CELL_OR_DISTRICT_MISSING,)
    assert notes["missing_districts"] == ("5001900US0602",)
    assert notes["allowed_missing_fraction"] == "0"


def test_the_dc_alias_must_keep_both_identities(protocol):
    required = {protocol.scope.dc_canonical_id}
    scope = dataclasses.replace(protocol.scope, district_count=1)
    wrong, notes = district_scope_reasons(
        scope,
        required_geography_ids=required,
        present_geography_ids=required,
        published_geography_ids={protocol.scope.dc_canonical_id: "5001900US1100"},
    )
    assert wrong == (Reason.IDENTITY_MISMATCH,)
    assert notes["dc_alias_mismatch"] is True
    dropped, notes = district_scope_reasons(
        scope,
        required_geography_ids=required,
        present_geography_ids=required,
        published_geography_ids={},
    )
    assert dropped == (Reason.IDENTITY_MISMATCH,)
    assert notes["dc_published_identity_missing"] is True
