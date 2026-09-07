"""Source universe, candidate bindings, and the one permitted scope reduction.

The ACS and ASEC rows here are invented and classified by the adapters the
contract pins, so the membership rules under test are the shipped ones. No
candidate, population or Census artifact is read.
"""

from __future__ import annotations

import dataclasses
from decimal import Decimal

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
from microcosm.build.cd_benchmark.metrics import categorical_check, point_check
from microcosm.build.cd_benchmark.protocol import canonical_protocol
from microcosm.build.cd_benchmark.reasons import (
    AXES,
    BenchmarkVerdict,
    ExposureStatus,
    MappingStatus,
    Reason,
)
from microcosm.build.cd_benchmark.reference import (
    published_percentage_precision,
    reference_cell,
    relative_moe_precision,
    subset_share_precision,
)
from microcosm.build.cd_benchmark.support import (
    SourceRecord,
    SupportMeasure,
    bin_support,
    denominator_support,
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


def population_cell():
    return reference_cell(
        {
            "E": "100",
            "M": "5",
            "EA": None,
            "MA": None,
            "published_GEO_ID": "5001900US0601",
            "canonical_geography_id": "5001900US0601",
            "dataset": "acs/acs1/subject",
            "table": "S0101",
            "variable": "S0101_C01_001",
            "year": 2024,
            "congress": 119,
            "units": "persons",
            "universe": "population",
        }
    )


def percentage_cell():
    """A published percentage cell whose printed margin is 0.5 points."""
    return reference_cell(
        {
            "E": "6.4",
            "M": "0.5",
            "EA": None,
            "MA": None,
            "published_GEO_ID": "5001900US0601",
            "canonical_geography_id": "5001900US0601",
            "dataset": "acs/acs1/subject",
            "table": "S0101",
            "variable": "S0101_C02_002",
            "year": 2024,
            "congress": 119,
            "units": "percent",
            "universe": "population",
        }
    )


def fixture_cell(variable, dataset, estimate, *, units="persons", moe="0.5"):
    return reference_cell(
        {
            "E": str(estimate),
            "M": moe,
            "EA": None,
            "MA": None,
            "published_GEO_ID": "5001900US0601",
            "canonical_geography_id": "5001900US0601",
            "dataset": dataset,
            "table": variable.split("_")[0],
            "variable": variable,
            "year": 2024,
            "congress": 119,
            "units": units,
            "universe": "population"
            if units in ("persons", "percent")
            else "occupied_housing_units",
        }
    )


def support_reading(protocol, block_id, total=100, *, bin_id=None):
    count = 1000 if bin_id is None else 50
    mass = Decimal(total) / count
    records = [
        SourceRecord(origin_key=f"origin-{i}", weight=mass, contribution=mass)
        for i in range(count)
    ]
    kwargs = dict(
        block_id=block_id,
        geography_id="5001900US0601",
        measure_kind=SupportMeasure.PERSONS
        if block_id in ("population_total", "age_distribution")
        else SupportMeasure.OCCUPIED_HOUSING_UNITS,
    )
    if bin_id is None:
        return denominator_support(protocol.support, records, **kwargs)
    return bin_support(
        protocol.support,
        records,
        bin_id=bin_id,
        reference_share_positive=True,
        **kwargs,
    )


def complete_axes(protocol, candidate=100):
    block = protocol.block("population_total")
    return {
        "precision": (
            relative_moe_precision(
                population_cell(), ceiling=block.reference_relative_moe_ceiling
            ),
        ),
        "support": (support_reading(protocol, block.id, candidate),),
    }


def conforming_block(protocol):
    block = protocol.block("population_total")
    return evaluate_block(
        block,
        geography_id="5001900US0601",
        metric=point_check(
            block, candidate=100, reference=100, geography_id="5001900US0601"
        ),
        exposure=ExposureStatus.NOT_RESERVED_CONFORMITY,
        **complete_axes(protocol),
    )


def exact_distribution(protocol, block_id):
    block = protocol.block(block_id)
    counts = dict.fromkeys((item.id for item in block.known_bins), 100)
    total = 100 * len(counts)
    return categorical_check(
        block,
        partition=protocol.partition,
        candidate_bin_masses=counts,
        unclassified_mass=0,
        candidate_denominator=total,
        reference_counts=counts,
        reference_total=total,
        geography_id="5001900US0601",
    )


def every_mandatory_block(protocol):
    """One fully identified evaluation per block in one invented district."""
    results = [conforming_block(protocol)]
    households = protocol.block("occupied_household_total")
    results.append(
        evaluate_block(
            households,
            geography_id="5001900US0601",
            metric=point_check(
                households, candidate=1000, reference=1000, geography_id="5001900US0601"
            ),
            precision=(
                relative_moe_precision(
                    fixture_cell(
                        households.variables[0],
                        households.dataset,
                        1000,
                        units="occupied_housing_units",
                    ),
                    ceiling=households.reference_relative_moe_ceiling,
                ),
            ),
            support=(support_reading(protocol, households.id, 1000),),
            exposure=ExposureStatus.FRESH_SAME_SOURCE_HOLDOUT_ELIGIBLE,
        )
    )
    for block in protocol.distribution_blocks:
        total = 100 * len(block.known_bins)
        if block.precision_variables:
            precision = tuple(
                published_percentage_precision(
                    fixture_cell(var, block.dataset, "5.6", units="percent"),
                    ceiling=block.reference_moe_ceiling_pp,
                )
                for var in block.precision_variables
            )
        else:
            denominator = fixture_cell(
                block.denominator_variable,
                block.dataset,
                total,
                units="occupied_housing_units",
            )
            precision = tuple(
                subset_share_precision(
                    fixture_cell(
                        bin.publisher_variable,
                        block.dataset,
                        100,
                        units="occupied_housing_units",
                    ),
                    denominator,
                    ceiling=block.reference_moe_ceiling_pp,
                )
                for bin in block.known_bins
            )
        results.append(
            evaluate_block(
                block,
                geography_id="5001900US0601",
                metric=exact_distribution(protocol, block.id),
                precision=precision,
                support=(
                    support_reading(protocol, block.id, total),
                    *(
                        support_reading(protocol, block.id, 100, bin_id=bin.id)
                        for bin in block.known_bins
                    ),
                ),
                exposure=ExposureStatus.FRESH_SAME_SOURCE_HOLDOUT_ELIGIBLE
                if block.reserved
                else ExposureStatus.NOT_RESERVED_CONFORMITY,
            )
        )
    return results


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
        blocks=every_mandatory_block(protocol),
    )
    assert result.scoring_permitted is False
    assert result.verdict is BenchmarkVerdict.SCOPE_INCOMPLETE
    assert Reason.PROTOCOL_UNAPPROVED in result.reasons
    # The measured block result is still there to inspect.
    assert result.blocks[0].metric.within_bound is True


def test_the_six_axes_are_reported_separately(protocol):
    protocol = dataclasses.replace(
        protocol, scope=dataclasses.replace(protocol.scope, district_count=1)
    )
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
        blocks=every_mandatory_block(protocol),
    )
    assert result.scoring_permitted is True
    assert result.verdict is BenchmarkVerdict.BENCHMARK_CONFORMING
    assert set(result.axes) == set(AXES) == set(protocol.independent_axes)
    assert result.axes["mapping_status"] == str(MappingStatus.APPROVED)
    assert result.axes["scope_completeness"] is True
    assert result.evidence["missing_mandatory_blocks"] == ()
    assert dict(result.axes["exposure_status"])["income_distribution"] == (
        "fresh_same_source_holdout_eligible"
    )


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
        blocks=every_mandatory_block(protocol),
    )
    assert result.verdict is BenchmarkVerdict.SCOPE_INCOMPLETE
    assert result.axes["mapping_status"] == str(MappingStatus.UNAPPROVED)


def test_nonconformity_is_retained_under_an_incomplete_scope(protocol):
    protocol = dataclasses.replace(
        protocol, scope=dataclasses.replace(protocol.scope, district_count=1)
    )
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
        protocol.block("population_total"),
        geography_id="5001900US0601",
        metric=point_check(
            protocol.block("population_total"),
            candidate=110,
            reference=100,
            geography_id="5001900US0601",
        ),
        exposure=ExposureStatus.NOT_RESERVED_CONFORMITY,
        **complete_axes(protocol, candidate=110),
    )
    blocks = [failing, *every_mandatory_block(protocol)[1:]]
    scoped = evaluate_scope(
        protocol,
        approval=approval,
        binding=authenticated_binding(protocol),
        blocks=blocks,
        scope_reasons=(Reason.REQUIRED_CELL_OR_DISTRICT_MISSING,),
    )
    assert scoped.verdict is BenchmarkVerdict.SCOPE_INCOMPLETE
    assert Reason.PUBLISHED_BENCHMARK_NONCONFORMITY in scoped.reasons
    assert scoped.evidence["nonconformities_retained"] == ("population_total",)
    without_scope_gap = evaluate_scope(
        protocol,
        approval=approval,
        binding=authenticated_binding(protocol),
        blocks=blocks,
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
        blocks=every_mandatory_block(protocol),
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


def test_a_mandatory_block_without_axis_evidence_is_incomplete(protocol):
    block = protocol.block("population_total")
    assert block.mandatory is True
    bare = evaluate_block(
        block,
        geography_id="5001900US0601",
        metric=point_check(
            block, candidate=100, reference=100, geography_id="5001900US0601"
        ),
        exposure=ExposureStatus.NOT_RESERVED_CONFORMITY,
    )
    assert bare.metric_conforming is True
    assert bare.scope_complete is False
    assert set(bare.reasons) == {
        Reason.REFERENCE_PRECISION_INADEQUATE,
        Reason.UNRESOLVED_SOURCE_LINEAGE,
    }
    assert bare.evidence["absent_axis_evidence"] == ("reference_precision", "support")
    unmeasured = evaluate_block(
        block,
        geography_id="5001900US0601",
        metric=None,
        exposure=ExposureStatus.NOT_RESERVED_CONFORMITY,
        **complete_axes(protocol),
    )
    assert unmeasured.metric_conforming is None
    assert unmeasured.reasons == (Reason.REQUIRED_CELL_OR_DISTRICT_MISSING,)
    complete = conforming_block(protocol)
    assert complete.scope_complete is True
    assert complete.reasons == ()
    assert complete.evidence["absent_axis_evidence"] == ()


def test_no_axis_is_allowed_to_rescue_another(protocol):
    block = protocol.block("population_total")
    result = evaluate_block(
        block,
        geography_id="5001900US0601",
        metric=point_check(
            block, candidate=110, reference=100, geography_id="5001900US0601"
        ),
        exposure=ExposureStatus.NOT_RESERVED_CONFORMITY,
        **complete_axes(protocol, candidate=110),
    )
    assert result.evidence["precision_does_not_widen_point_bound"] is True
    assert result.evidence["support_does_not_rescue_a_metric"] is True
    assert result.evidence["support_verdicts"] == ("within_bounds",)
    assert result.reasons == (Reason.PUBLISHED_BENCHMARK_NONCONFORMITY,)


def test_reviewer_verdicts_alone_do_not_activate_the_protocol(protocol):
    """A resolution that skipped registration cannot be passed off as one."""
    injected, registration = authority(
        protocol, frozenset({ApprovalScope.CANDIDATE_SCORING})
    )
    bare = injected.resolve(protocol.sha256, ApprovalScope.CANDIDATE_SCORING)
    assert bare.authorized is True
    assert bare.registration_resolved is False
    result = evaluate_scope(
        protocol,
        approval=bare,
        binding=authenticated_binding(protocol),
        blocks=every_mandatory_block(protocol),
    )
    assert result.scoring_permitted is False
    assert Reason.PROTOCOL_UNAPPROVED in result.reasons
    activated = resolve_action(
        protocol,
        authority=injected,
        action=ApprovalScope.CANDIDATE_SCORING,
        registration=registration,
    )
    assert activated.registration_resolved is True
    assert (
        evaluate_scope(
            protocol,
            approval=activated,
            binding=authenticated_binding(protocol),
            blocks=every_mandatory_block(protocol),
        ).scoring_permitted
        is True
    )


def test_every_mandatory_block_must_be_present(protocol):
    injected, registration = authority(
        protocol, frozenset({ApprovalScope.CANDIDATE_SCORING})
    )
    approval = resolve_action(
        protocol,
        authority=injected,
        action=ApprovalScope.CANDIDATE_SCORING,
        registration=registration,
    )
    assert protocol.scope.allowed_missing_mandatory_block_fraction == 0
    partial = evaluate_scope(
        protocol,
        approval=approval,
        binding=authenticated_binding(protocol),
        blocks=[conforming_block(protocol)],
    )
    assert partial.verdict is BenchmarkVerdict.SCOPE_INCOMPLETE
    assert Reason.REQUIRED_CELL_OR_DISTRICT_MISSING in partial.reasons
    assert partial.evidence["missing_mandatory_blocks"] == (
        "age_distribution",
        "income_distribution",
        "occupied_household_total",
        "tenure_distribution",
    )


def test_districts_cannot_fill_each_others_mandatory_block_gaps(protocol):
    """All five IDs globally do not establish five blocks in either district."""
    injected, registration = authority(
        protocol, frozenset({ApprovalScope.CANDIDATE_SCORING})
    )
    approval = resolve_action(
        protocol,
        authority=injected,
        action=ApprovalScope.CANDIDATE_SCORING,
        registration=registration,
    )
    blocks = every_mandatory_block(protocol)
    # Every axis moves with its block, so only the per-district roster is wrong.
    first = blocks[0]
    geography = "5001900US0602"
    blocks[0] = evaluate_block(
        protocol.block(first.block_id),
        geography_id=geography,
        metric=dataclasses.replace(first.metric, geography_id=geography),
        precision=tuple(
            dataclasses.replace(
                reading,
                cells=tuple(
                    dataclasses.replace(
                        cell,
                        canonical_geography_id=geography,
                        published_geo_id=geography,
                    )
                    for cell in reading.cells
                ),
            )
            for reading in first.precision
        ),
        support=tuple(
            dataclasses.replace(reading, geography_id=geography)
            for reading in first.support
        ),
        exposure=first.exposure,
    )
    protocol = dataclasses.replace(
        protocol, scope=dataclasses.replace(protocol.scope, district_count=2)
    )
    assert all(not block.reasons for block in blocks)
    result = evaluate_scope(
        protocol,
        approval=approval,
        binding=authenticated_binding(protocol),
        blocks=blocks,
    )
    assert result.verdict is BenchmarkVerdict.SCOPE_INCOMPLETE
    assert Reason.REQUIRED_CELL_OR_DISTRICT_MISSING in result.reasons
    assert result.evidence["missing_mandatory_blocks"] == ()
    assert result.evidence["missing_mandatory_block_geographies"] == (
        ("5001900US0601", "population_total"),
        ("5001900US0602", "age_distribution"),
        ("5001900US0602", "income_distribution"),
        ("5001900US0602", "occupied_household_total"),
        ("5001900US0602", "tenure_distribution"),
    )
    assert result.blocks == tuple(blocks)


@pytest.mark.parametrize(
    "omitted",
    [
        "population_total",
        "occupied_household_total",
        "age_distribution",
        "income_distribution",
        "tenure_distribution",
        "all",
    ],
)
def test_each_mandatory_block_and_empty_scope_refuse(protocol, omitted):
    injected, registration = authority(
        protocol, frozenset({ApprovalScope.CANDIDATE_SCORING})
    )
    approval = resolve_action(
        protocol,
        authority=injected,
        action=ApprovalScope.CANDIDATE_SCORING,
        registration=registration,
    )
    complete = every_mandatory_block(protocol)
    blocks = (
        []
        if omitted == "all"
        else [block for block in complete if block.block_id != omitted]
    )
    result = evaluate_scope(
        protocol,
        approval=approval,
        binding=authenticated_binding(protocol),
        blocks=blocks,
    )
    assert result.verdict is BenchmarkVerdict.SCOPE_INCOMPLETE
    expected = (
        tuple(sorted(block.block_id for block in complete))
        if omitted == "all"
        else (omitted,)
    )
    assert result.evidence["missing_mandatory_blocks"] == expected
    assert result.axes["scope_completeness"] is False
    assert result.blocks == tuple(blocks)
