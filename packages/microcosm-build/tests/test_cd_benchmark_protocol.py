"""The compiled contract is the approved bytes, and approval is never local.

Every constant checked here is read back out of the packaged canonical document,
so a drift between the code's vocabulary and the contract fails a test rather
than passing quietly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from decimal import Decimal

import pytest

# The agreement test below pins this package's canonical encoder against the
# shared aggregate-JSON serializer; that serializer is module-private.
from microcosm.build.candidate_quality import _bytes as candidate_quality_bytes
from microcosm.build.cd_benchmark.approval import (
    ApprovalAuthority,
    ApprovalError,
    ApprovalRecord,
    ApprovalScope,
    ApprovalVerdict,
    RegistrationClaim,
    ReviewerRole,
    resolve_action,
)
from microcosm.build.cd_benchmark.canonical import (
    CanonicalJsonError,
    canonical_bytes,
    digest,
    strict_load,
)
from microcosm.build.cd_benchmark.ledger import EventType
from microcosm.build.cd_benchmark.protocol import (
    APPROVED_PROTOCOL_BYTES,
    APPROVED_PROTOCOL_ID,
    APPROVED_PROTOCOL_SHA256,
    RESERVED_FAMILIES,
    PrecisionKind,
    ProtocolError,
    assign_bin,
    canonical_protocol,
    compile_protocol,
    load_canonical_bytes,
)
from microcosm.build.cd_benchmark.reasons import ExposureStatus, Reason
from microcosm.build.cd_benchmark.universe import verify_pinned_adapters

FIXTURE_SCOPE = ApprovalScope.INVENTED_FIXTURE_IMPLEMENTATION_ONLY


@pytest.fixture(scope="module")
def protocol():
    return canonical_protocol()


def _authority(protocol, *, scopes=frozenset({FIXTURE_SCOPE})):
    root = ApprovalRecord(
        protocol_sha256=protocol.sha256,
        verdict=ApprovalVerdict.APPROVE,
        reviewer_role=ReviewerRole.ROOT,
        reviewer_id="root:invented-fixture",
        scopes=scopes,
        review_artifact_sha256="c" * 64,
    )
    peer = ApprovalRecord(
        protocol_sha256=protocol.sha256,
        verdict=ApprovalVerdict.APPROVE,
        reviewer_role=ReviewerRole.PEER,
        reviewer_id="peer:invented-fixture",
        scopes=scopes,
        review_artifact_sha256="d" * 64,
    )
    return ApprovalAuthority(records=(root, peer)), root, peer


def _registration(protocol, root, peer, **overrides):
    fields = {
        "protocol_sha256": protocol.sha256,
        "protocol_parent_sha256": protocol.parent_sha256,
        "legacy_parent_artifact_hashes": (),
        "parameter_changes": (),
        "root_approval_artifact_sha256": root.sha256,
        "peer_approval_artifact_sha256": peer.sha256,
        "reference_inventory_sha256": protocol.reference_inventory_sha256,
        "block_definition_hashes": dict(protocol.block_definition_hashes),
    }
    fields.update(overrides)
    return RegistrationClaim(**fields)


def test_packaged_resource_is_the_approved_document(protocol):
    raw = load_canonical_bytes()
    assert len(raw) == APPROVED_PROTOCOL_BYTES == 86626
    assert digest(raw) == APPROVED_PROTOCOL_SHA256
    assert protocol.sha256 == APPROVED_PROTOCOL_SHA256
    assert protocol.protocol_id == APPROVED_PROTOCOL_ID
    assert protocol.canonical is True
    assert protocol.hash_scope.startswith("entire proposal JSON bytes")


def test_a_one_byte_change_is_not_the_approved_protocol():
    raw = load_canonical_bytes()
    mutated = raw.replace(b'"status":"PROPOSAL"', b'"status":"APPROVED"', 1)
    assert mutated != raw
    compiled = compile_protocol(mutated)
    assert compiled.canonical is False
    assert compiled.sha256 != APPROVED_PROTOCOL_SHA256


def test_the_authoring_snapshot_is_never_treated_as_installed_approval(protocol):
    assert protocol.status == "PROPOSAL"
    assert protocol.approval_installed_in_document is False
    assert protocol.scoring_permitted_in_document is False


def test_closed_vocabularies_equal_the_document(protocol):
    assert {item.value for item in Reason} == set(protocol.reason_codes)
    assert len(protocol.reason_codes) == 21
    assert {item.value for item in ExposureStatus} == set(
        protocol.exposure_status_values
    )
    assert {item.value for item in EventType} == set(protocol.ledger_event_types)


def test_every_declared_threshold_is_an_exact_decimal(protocol):
    population = protocol.block("population_total")
    households = protocol.block("occupied_household_total")
    assert population.point_relative_bound == Decimal("0.02")
    assert population.reference_relative_moe_ceiling == Decimal("0.05")
    assert population.controlled_reference == "separate_controlled_estimate_class"
    assert households.point_relative_bound == Decimal("0.03")
    assert households.reference_relative_moe_ceiling == Decimal("0.05")
    assert households.count_once is True
    assert households.duplicate_reference_aliases == ("B19001_001",)
    expected = {
        "age_distribution": (Decimal("1"), Decimal("0.03"), Decimal("1")),
        "income_distribution": (Decimal("2"), Decimal("0.05"), Decimal("2")),
        "tenure_distribution": (Decimal("3"), Decimal("0.03"), Decimal("3")),
    }
    for block_id, (per_bin, variation, moe) in expected.items():
        block = protocol.block(block_id)
        assert block.max_each_bin_discrepancy_pp == per_bin
        assert block.max_total_variation == variation
        assert block.reference_moe_ceiling_pp == moe
    assert protocol.partition.maximum_unclassified_fraction == Decimal("0.005")
    assert protocol.partition.comparison == "<="
    assert protocol.partition.renormalize_known_bins is False
    support = protocol.support
    assert support.minimum_positive_source_household_origins == 1000
    assert support.minimum_positive_origins_per_positive_reference_bin == 50
    assert support.minimum_origin_cluster_concentration_ess == Decimal("500")
    assert support.maximum_origin_denominator_share == Decimal("0.005")
    assert support.maximum_origin_positive_bin_share == Decimal("0.05")


def test_precision_rules_differ_by_block(protocol):
    age = protocol.block("age_distribution")
    assert age.precision_kind is PrecisionKind.PUBLISHED_PERCENTAGE_MOE
    assert age.precision_variables[0] == "S0101_C02_002"
    assert len(age.precision_variables) == 18
    for block_id in ("income_distribution", "tenure_distribution"):
        assert (
            protocol.block(block_id).precision_kind
            is PrecisionKind.SUBSET_SHARE_APPROXIMATION
        )


def test_reservation_covers_only_the_two_holdout_families(protocol):
    assert RESERVED_FAMILIES == {"B19001", "B25003"}
    reserved = {block.id for block in protocol.blocks if block.reserved}
    assert reserved == {
        "occupied_household_total",
        "income_distribution",
        "tenure_distribution",
    }
    assert protocol.block("occupied_household_total").exposure_families == {
        "B19001",
        "B25003",
    }


def _invented_acs_table(table: str) -> dict:
    """Run the shipped inventory adapter over one invented published cell."""
    from microcosm.build.us_runtime.cd_reference import acs_table_inventory

    stem = f"{table}_001"
    metadata = {
        "variables": {
            stem + suffix: {"label": f"{stem}{suffix}"}
            for suffix in ("E", "M", "EA", "MA")
        }
    }
    header = ["NAME", "state", "congressional district", *metadata["variables"]]
    row = ["District 1", "06", "01", "1000", "50", None, None]
    return acs_table_inventory(table, metadata, [header, row])


def test_reservation_matches_the_shipped_reference_inventory_adapter():
    """The contract's reserved set is the adapter's, not a second opinion."""
    from microcosm.build.us_runtime.cd_reference import TABLES

    assert set(TABLES) == {"S0101", "B01001", "B19001", "B25003"}
    reserved = set()
    for table in TABLES:
        inventory = _invented_acs_table(table)
        reservation = inventory["reservation"]
        assert reservation["targets_allowed"] is False
        assert reservation["source_independent"] is False
        if reservation["status"] == "reserved_same_source_holdout":
            assert reservation["covers_derived_versions"] is True
            reserved.add(table)
        else:
            assert reservation["status"] == "inventory_only"
    assert reserved == set(RESERVED_FAMILIES) == {"B19001", "B25003"}


def test_canonical_district_identity_comes_from_the_shipped_adapter():
    inventory = _invented_acs_table("B25003")
    assert inventory["geography_ids"] == ["5001900US0601"]
    assert inventory["congress"] == 119
    assert inventory["year"] == 2024
    assert inventory["cells"][0]["estimate"]["numeric_value"] == 1000.0
    assert inventory["cells"][0]["moe"]["class"] == "numeric"


def test_every_distribution_carries_one_protocol_constructed_residual(protocol):
    for block_id, known in (
        ("age_distribution", 18),
        ("income_distribution", 16),
        ("tenure_distribution", 2),
    ):
        block = protocol.block(block_id)
        assert len(block.known_bins) == known
        residual = block.unclassified_bin
        assert residual.id == "unclassified"
        assert residual.publisher_variable is None
        assert residual.reference_share == Decimal("0")
        assert residual.authority == (
            "protocol_constructed_residual_category_not_publisher_cell"
        )


@pytest.mark.parametrize(
    ("block_id", "value", "expected"),
    [
        ("age_distribution", "0", "S0101_C01_002"),
        ("age_distribution", "4.999", "S0101_C01_002"),
        ("age_distribution", "5", "S0101_C01_003"),
        ("age_distribution", "84.999", "S0101_C01_018"),
        ("age_distribution", "85", "S0101_C01_019"),
        ("age_distribution", "120", "S0101_C01_019"),
        ("age_distribution", "-1", None),
        ("income_distribution", "-4000", "B19001_002"),
        ("income_distribution", "9999.99", "B19001_002"),
        ("income_distribution", "10000", "B19001_003"),
        ("income_distribution", "199999.99", "B19001_016"),
        ("income_distribution", "200000", "B19001_017"),
    ],
)
def test_bin_boundaries_are_lower_inclusive_and_upper_exclusive(
    protocol, block_id, value, expected
):
    assert assign_bin(protocol.block(block_id), Decimal(value)) == expected


def test_tenure_bins_have_no_numeric_boundary_to_assign(protocol):
    with pytest.raises(ProtocolError):
        assign_bin(protocol.block("tenure_distribution"), Decimal("1"))


def test_open_ends_are_infinite_not_clamped(protocol):
    income = protocol.block("income_distribution")
    assert income.known_bins[0].lower_inclusive is None
    assert income.known_bins[-1].upper_exclusive is None
    age = protocol.block("age_distribution")
    assert age.open_upper is True
    assert age.known_bins[-1].upper_exclusive is None


def test_tenure_bins_are_not_independent_votes_and_others_say_nothing(protocol):
    assert protocol.block("tenure_distribution").independent_votes is False
    assert protocol.block("tenure_distribution").complement_only_when == (
        "unclassified candidate mass is exactly zero"
    )
    assert protocol.block("age_distribution").independent_votes is None
    assert protocol.block("income_distribution").independent_votes is None


def test_dc_alias_matches_the_shipped_district_identity_algorithm(protocol):
    """The document's DC alias is checked against the adapter that mints ids."""
    from microcosm.build.us_runtime.cd_reference import _district_id

    assert protocol.scope.dc_canonical_id == _district_id("11", "98", 119)
    assert protocol.scope.dc_published_id == "5001900US1198"
    assert protocol.scope.dc_canonical_id != protocol.scope.dc_published_id
    assert protocol.scope.published_and_canonical_geography_ids_retained is True
    assert protocol.scope.district_count == 436
    assert protocol.scope.congress == 119
    assert protocol.scope.acs_year == 2024
    assert protocol.scope.puerto_rico == "excluded_by_predeclared_scope"


def test_pinned_adapters_match_the_contract(protocol):
    """A changed adapter is a scope review, not a pin to refresh in passing."""
    assert all(verify_pinned_adapters(protocol).values())


def test_block_definition_hashes_cover_every_block(protocol):
    assert set(protocol.block_definition_hashes) == {
        block.id for block in protocol.blocks
    }
    assert len(set(protocol.block_definition_hashes.values())) == 5


def test_canonical_encoder_agrees_with_the_shared_serializer():
    value = {"b": [1, 2, {"d": None, "c": True}], "a": "x"}
    assert canonical_bytes(value) + b"\n" == candidate_quality_bytes(value)


def test_canonical_loader_refuses_duplicate_keys_bom_and_nonfinite():
    with pytest.raises(CanonicalJsonError):
        strict_load(b'{"a":1,"a":2}')
    with pytest.raises(CanonicalJsonError):
        strict_load("﻿{}".encode())
    with pytest.raises(CanonicalJsonError):
        strict_load(b'{"a":NaN}')
    assert strict_load(b'{"a":1}') == {"a": 1}


def test_compile_refuses_a_document_that_is_not_this_contract():
    with pytest.raises(ProtocolError):
        compile_protocol(b'{"schema":"other"}')
    with pytest.raises(ProtocolError):
        compile_protocol(canonical_bytes({"identity": {"algorithm": "md5"}}))


def test_a_float_threshold_is_refused_as_a_bound():
    raw = load_canonical_bytes()
    document = json.loads(raw)
    document["blocks"][0]["point_relative_bound"] = 0.02
    with pytest.raises(ProtocolError):
        compile_protocol(canonical_bytes(document))


def test_injected_authority_authorizes_only_the_approved_scope(protocol):
    authority, root, peer = _authority(protocol)
    registration = _registration(protocol, root, peer)
    fixture = resolve_action(
        protocol,
        authority=authority,
        action=FIXTURE_SCOPE,
        registration=registration,
    )
    assert fixture.authorized is True
    assert fixture.reasons == ()
    assert fixture.root.reviewer_id == "root:invented-fixture"
    scoring = resolve_action(
        protocol,
        authority=authority,
        action=ApprovalScope.CANDIDATE_SCORING,
        registration=registration,
    )
    assert scoring.authorized is False
    assert Reason.PROTOCOL_UNAPPROVED in scoring.reasons


def test_there_is_no_default_or_production_authority(protocol):
    authority, root, peer = _authority(protocol)
    empty = resolve_action(
        protocol,
        authority=ApprovalAuthority.empty(),
        action=FIXTURE_SCOPE,
        registration=_registration(protocol, root, peer),
    )
    assert empty.authorized is False
    assert Reason.PROTOCOL_UNAPPROVED in empty.reasons
    with pytest.raises(ApprovalError):
        resolve_action(
            protocol, authority=None, action=FIXTURE_SCOPE, registration=None
        )
    assert authority.records


def test_registration_is_required_and_must_resolve_both_review_artifacts(protocol):
    authority, root, peer = _authority(protocol)
    without = resolve_action(
        protocol, authority=authority, action=FIXTURE_SCOPE, registration=None
    )
    assert without.authorized is False
    forged = resolve_action(
        protocol,
        authority=authority,
        action=FIXTURE_SCOPE,
        registration=_registration(
            protocol, root, peer, root_approval_artifact_sha256="e" * 64
        ),
    )
    assert forged.authorized is False
    assert forged.evidence["root_artifact_unresolved"] is True
    swapped = resolve_action(
        protocol,
        authority=authority,
        action=FIXTURE_SCOPE,
        registration=_registration(
            protocol,
            root,
            peer,
            root_approval_artifact_sha256=peer.sha256,
            peer_approval_artifact_sha256=root.sha256,
        ),
    )
    assert swapped.authorized is False


def test_registration_must_name_the_contract_it_registers(protocol):
    authority, root, peer = _authority(protocol)
    for override in (
        {"reference_inventory_sha256": "f" * 64},
        {"protocol_parent_sha256": "a" * 64},
        {"block_definition_hashes": {"population_total": "b" * 64}},
    ):
        result = resolve_action(
            protocol,
            authority=authority,
            action=FIXTURE_SCOPE,
            registration=_registration(protocol, root, peer, **override),
        )
        assert result.authorized is False
        assert Reason.IDENTITY_MISMATCH in result.reasons


def test_a_reviewer_approving_the_wrong_scope_does_not_count(protocol):
    authority, root, peer = _authority(
        protocol, scopes=frozenset({ApprovalScope.PRODUCTION_LEDGER_REGISTRATION})
    )
    result = resolve_action(
        protocol,
        authority=authority,
        action=FIXTURE_SCOPE,
        registration=_registration(protocol, root, peer),
    )
    assert result.authorized is False
    assert result.evidence["root_approvals_for_action"] == 0


def test_one_reviewer_cannot_approve_alone(protocol):
    _, root, peer = _authority(protocol)
    only_root = ApprovalAuthority(records=(root,))
    result = resolve_action(
        protocol,
        authority=only_root,
        action=FIXTURE_SCOPE,
        registration=_registration(protocol, root, peer),
    )
    assert result.authorized is False


def test_a_rejection_blocks_even_beside_an_approval(protocol):
    _, root, peer = _authority(protocol)
    rejection = ApprovalRecord(
        protocol_sha256=protocol.sha256,
        verdict=ApprovalVerdict.REJECT,
        reviewer_role=ReviewerRole.PEER,
        reviewer_id="peer:second-look",
        scopes=frozenset({FIXTURE_SCOPE}),
        review_artifact_sha256="9" * 64,
    )
    authority = ApprovalAuthority(records=(root, peer, rejection))
    result = resolve_action(
        protocol,
        authority=authority,
        action=FIXTURE_SCOPE,
        registration=_registration(protocol, root, peer),
    )
    assert result.authorized is False
    assert result.evidence["peer_rejection_present"] is True


def test_non_canonical_bytes_cannot_be_approved_by_any_authority(protocol):
    raw = load_canonical_bytes()
    other = compile_protocol(raw.replace(b'"congress":119', b'"congress":118', 1))
    root = ApprovalRecord(
        protocol_sha256=other.sha256,
        verdict=ApprovalVerdict.APPROVE,
        reviewer_role=ReviewerRole.ROOT,
        reviewer_id="root",
        scopes=frozenset({FIXTURE_SCOPE}),
        review_artifact_sha256="c" * 64,
    )
    peer = ApprovalRecord(
        protocol_sha256=other.sha256,
        verdict=ApprovalVerdict.APPROVE,
        reviewer_role=ReviewerRole.PEER,
        reviewer_id="peer",
        scopes=frozenset({FIXTURE_SCOPE}),
        review_artifact_sha256="d" * 64,
    )
    result = resolve_action(
        other,
        authority=ApprovalAuthority(records=(root, peer)),
        action=FIXTURE_SCOPE,
        registration=RegistrationClaim(
            protocol_sha256=other.sha256,
            protocol_parent_sha256=other.parent_sha256,
            legacy_parent_artifact_hashes=(),
            parameter_changes=(),
            root_approval_artifact_sha256=root.sha256,
            peer_approval_artifact_sha256=peer.sha256,
            reference_inventory_sha256=other.reference_inventory_sha256,
            block_definition_hashes=dict(other.block_definition_hashes),
        ),
    )
    assert result.authorized is False
    assert result.evidence["non_canonical_protocol_bytes"] is True


def test_importing_the_pure_core_loads_no_engine_or_us_runtime():
    """The pure logic sits above the eager us_runtime package, by measurement."""
    program = (
        "import sys;"
        "import microcosm.build.cd_benchmark as p;"
        "import microcosm.build.cd_benchmark.approval;"
        "import microcosm.build.cd_benchmark.binding;"
        "import microcosm.build.cd_benchmark.evaluation;"
        "import microcosm.build.cd_benchmark.ledger;"
        "import microcosm.build.cd_benchmark.metrics;"
        "import microcosm.build.cd_benchmark.origin;"
        "import microcosm.build.cd_benchmark.reference;"
        "import microcosm.build.cd_benchmark.support;"
        "import microcosm.build.cd_benchmark.universe;"
        "print(sorted(name for name in sys.modules "
        "if name.split('.')[0] in {'policyengine_us', 'policyengine_uk',"
        " 'policyengine_core'} or name.startswith('microcosm.build.us_runtime')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"
