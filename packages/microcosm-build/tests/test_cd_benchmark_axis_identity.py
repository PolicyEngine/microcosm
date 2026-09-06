"""Regressions for mismatched and incomplete evidence; all inputs invented."""

import runpy
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from microcosm.build.cd_benchmark.approval import ApprovalScope, resolve_action
from microcosm.build.cd_benchmark.binding import (
    ArtifactRef,
    BindingError,
    CandidateBinding,
    resolve_binding,
)
from microcosm.build.cd_benchmark.evaluation import evaluate_block, evaluate_scope
from microcosm.build.cd_benchmark.metrics import categorical_check, point_check
from microcosm.build.cd_benchmark.protocol import canonical_protocol
from microcosm.build.cd_benchmark.reasons import (
    BenchmarkVerdict,
    ExposureStatus,
    MappingStatus,
    Reason,
)
from microcosm.build.cd_benchmark.reference import relative_moe_precision

_fixtures = runpy.run_path(
    str(Path(__file__).with_name("test_cd_benchmark_universe.py"))
)
authenticated_binding = _fixtures["authenticated_binding"]
authority = _fixtures["authority"]
every_mandatory_block = _fixtures["every_mandatory_block"]
fixture_cell = _fixtures["fixture_cell"]


def evaluated(protocol, original, **changes):
    args = dict(
        geography_id=original.geography_id,
        metric=original.metric,
        precision=original.precision,
        support=original.support,
        exposure=original.exposure,
    )
    args.update(changes)
    return evaluate_block(protocol.block(original.block_id), **args)


@pytest.fixture
def baseline():
    protocol = canonical_protocol()
    # This invented scope has one district. It cannot stand for all 436.
    protocol = replace(protocol, scope=replace(protocol.scope, district_count=1))
    blocks = every_mandatory_block(protocol)
    assert all(block.scope_complete and not block.reasons for block in blocks)
    return protocol, blocks


def score_scope(protocol, blocks, binding):
    injected, registration = authority(
        protocol, frozenset({ApprovalScope.CANDIDATE_SCORING})
    )
    approval = resolve_action(
        protocol,
        authority=injected,
        action=ApprovalScope.CANDIDATE_SCORING,
        registration=registration,
    )
    return evaluate_scope(protocol, approval=approval, binding=binding, blocks=blocks)


@pytest.mark.parametrize("include_unused_slot", [False, True])
def test_income_block_cannot_use_an_income_waiver(baseline, include_unused_slot):
    protocol, blocks = baseline
    slot = "income_eight_component_definition_id_and_sha256_when_income_claimed"
    slots = {
        name: ArtifactRef(f"invented:{name}", "a" * 64)
        for name in protocol.required_binding_slots
        if include_unused_slot or name != slot
    }
    binding = resolve_binding(
        protocol,
        CandidateBinding(slots, income_claimed=False),
        authenticator=_fixtures["FixtureAuthenticator"](set(slots)),
    )
    assert binding.status is MappingStatus.APPROVED  # Valid for its declared claim.
    result = score_scope(protocol, blocks, binding)
    assert result.verdict is BenchmarkVerdict.SCOPE_INCOMPLETE
    assert result.reasons == (Reason.MAPPING_UNAPPROVED,)
    assert result.axes["mapping_status"] == str(MappingStatus.UNAPPROVED)
    assert result.evidence["income_claim_binding_mismatch"] is True


def test_authenticated_income_claim_covers_the_income_block(baseline):
    protocol, blocks = baseline
    result = score_scope(protocol, blocks, authenticated_binding(protocol))
    assert result.verdict is BenchmarkVerdict.BENCHMARK_CONFORMING
    assert not result.reasons
    assert result.evidence["income_claim_binding_mismatch"] is False


@pytest.mark.parametrize("claim", [None, 0, 1, "", "false", [], {}])
def test_income_claim_is_a_boolean(claim):
    with pytest.raises(BindingError, match="income_claimed must be a boolean"):
        CandidateBinding({}, income_claimed=claim)


def test_one_district_cannot_establish_the_canonical_436_district_scope():
    protocol = canonical_protocol()
    assert protocol.scope.district_count == 436
    result = score_scope(
        protocol, every_mandatory_block(protocol), authenticated_binding(protocol)
    )
    assert result.verdict is BenchmarkVerdict.SCOPE_INCOMPLETE
    assert result.reasons == (Reason.REQUIRED_CELL_OR_DISTRICT_MISSING,)
    assert result.evidence["present_district_count"] == 1
    assert result.evidence["declared_district_count"] == 436


@pytest.mark.parametrize("part", ["block", "geography", "bound"])
def test_metric_cannot_be_borrowed(baseline, part):
    protocol, blocks = baseline
    pop = blocks[0]
    if part == "block":
        metric = blocks[1].metric
    elif part == "geography":
        metric = replace(pop.metric, geography_id="5001900US0602")
    else:
        loose = replace(
            protocol.block(pop.block_id), point_relative_bound=Decimal("0.2")
        )
        metric = point_check(
            loose, candidate=100, reference=100, geography_id=pop.geography_id
        )
    result = evaluated(protocol, pop, metric=metric)
    assert not result.scope_complete
    assert Reason.IDENTITY_MISMATCH in result.reasons


@pytest.mark.parametrize("part", ["block", "geography", "measure", "mass", "scope"])
def test_support_cannot_be_borrowed(baseline, part):
    protocol, blocks = baseline
    pop, household, age = blocks[:3]
    support = pop.support[0]
    substitutions = {
        "block": replace(support, block_id=household.block_id),
        "geography": replace(support, geography_id="5001900US0602"),
        "measure": replace(support, measure=household.support[0].measure),
        "mass": replace(support, contribution_total=Decimal(999)),
        "scope": age.support[1],
    }
    assert not evaluated(protocol, pop, support=(substitutions[part],)).scope_complete


@pytest.mark.parametrize(
    "part",
    [
        "missing_bin",
        "duplicate_bin",
        "missing_precision",
        "duplicate_precision",
        "wrong_kind",
    ],
)
def test_distribution_requires_complete_distinct_axes(baseline, part):
    protocol, blocks = baseline
    age, income = blocks[2:4]
    changes = {
        "missing_bin": dict(support=age.support[:1]),
        "duplicate_bin": dict(support=age.support[:-1] + (age.support[1],)),
        "missing_precision": dict(precision=age.precision[:1]),
        "duplicate_precision": dict(precision=age.precision[:-1] + (age.precision[0],)),
        "wrong_kind": dict(precision=income.precision),
    }
    assert not evaluated(protocol, age, **changes[part]).scope_complete


@pytest.mark.parametrize("index", range(5))
def test_reserved_status_must_match_the_block(baseline, index):
    protocol, blocks = baseline
    block = blocks[index]
    wrong = (
        ExposureStatus.NOT_RESERVED_CONFORMITY
        if protocol.block(block.block_id).reserved
        else ExposureStatus.FRESH_SAME_SOURCE_HOLDOUT_ELIGIBLE
    )
    assert not evaluated(protocol, block, exposure=wrong).scope_complete


def test_scope_cannot_use_approval_for_another_protocol(baseline):
    protocol, blocks = baseline
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
        replace(protocol, sha256="0" * 64),
        approval=approval,
        binding=authenticated_binding(protocol),
        blocks=blocks,
    )
    assert not result.scoring_permitted
    assert Reason.IDENTITY_MISMATCH in result.reasons


def test_printed_decimal_moe_boundary_is_exact():
    cell = fixture_cell("S0101_C01_001", "acs/acs1/subject", "0.7", moe="0.035")
    result = relative_moe_precision(cell, ceiling=Decimal("0.05"))
    assert result.measured == Decimal("0.05")
    assert not result.reasons


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset", "acs/acs5/subject"),
        ("table", "B19001"),
        ("canonical_geography_id", "5001900US0602"),
        ("units", "dollars"),
        ("universe", "families"),
    ],
)
def test_precision_source_identity_must_match(baseline, field, value):
    protocol, blocks = baseline
    pop = blocks[0]
    reading = pop.precision[0]
    bad = replace(reading, cells=(replace(reading.cells[0], **{field: value}),))
    assert not evaluated(protocol, pop, precision=(bad,)).scope_complete


@pytest.mark.parametrize(
    "part",
    [
        "year",
        "congress",
        "support_bounds",
        "duplicate_block",
        "geography_inventory",
        "metric_definition",
    ],
)
def test_scope_rechecks_supplied_evidence_against_its_protocol(baseline, part):
    protocol, blocks = baseline
    injected, registration = authority(
        protocol, frozenset({ApprovalScope.CANDIDATE_SCORING})
    )
    approval = resolve_action(
        protocol,
        authority=injected,
        action=ApprovalScope.CANDIDATE_SCORING,
        registration=registration,
    )
    original = blocks[0]
    kwargs = {}
    if part in ("year", "congress"):
        reading = original.precision[0]
        cell = replace(reading.cells[0], **{part: 2023 if part == "year" else 118})
        blocks[0] = replace(original, precision=(replace(reading, cells=(cell,)),))
    elif part == "support_bounds":
        reading = original.support[0]
        blocks[0] = replace(
            original,
            support=(
                replace(
                    reading,
                    bounds=replace(
                        reading.bounds, minimum_positive_source_household_origins=1
                    ),
                ),
            ),
        )
    elif part == "duplicate_block":
        blocks.append(original)
    elif part == "geography_inventory":
        kwargs["required_geography_ids"] = ["5001900US0602"]
    else:
        # Preserving an old block's empty reasons does not bypass recomposition.
        loose = replace(
            protocol.block(original.block_id), point_relative_bound=Decimal("0.20")
        )
        metric = point_check(
            loose, candidate=100, reference=100, geography_id=original.geography_id
        )
        blocks[0] = replace(original, metric=metric)
    result = evaluate_scope(
        protocol,
        approval=approval,
        binding=authenticated_binding(protocol),
        blocks=blocks,
        **kwargs,
    )
    assert not result.axes["scope_completeness"]


def test_stricter_declared_renter_bound_is_enforced():
    protocol = canonical_protocol()
    block = replace(
        protocol.block("tenure_distribution"),
        additional_bounds={"max_renter_discrepancy_pp": Decimal("1")},
    )
    result = categorical_check(
        block,
        partition=protocol.partition,
        candidate_bin_masses={"owner": 52, "renter": 48},
        unclassified_mass=0,
        candidate_denominator=100,
        reference_counts={"owner": 50, "renter": 50},
        reference_total=100,
    )
    assert not result.within_bound
    assert Reason.PUBLISHED_BENCHMARK_NONCONFORMITY in result.reasons
