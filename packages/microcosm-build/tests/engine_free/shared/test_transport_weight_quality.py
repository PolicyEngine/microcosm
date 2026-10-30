"""Original-source weight quality and cell screens over synthetic donors."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from microcosm.build.country_spec import GatesManifest, load_country_spec
from microcosm.build.gate_battery import EvidenceContext, GateStatus, evaluate_phase
from microcosm.build.transport.gate_bindings import (
    TRANSPORT_GATE_REGISTRY,
    weight_ess_gate,
    weight_ratio_gate,
    weight_summary,
)
from microcosm.build.transport.takeup_kernels import (
    apply_cell_support,
    source_cell_diagnostics,
)
from microcosm.frame import EntitySchema, Frame, WeightKind, Weights

PROPERTY = settings(max_examples=60, deadline=None, database=None)
_WEIGHTS = st.lists(st.integers(min_value=1, max_value=1000), min_size=2, max_size=24)
_SCALE = st.one_of(
    st.floats(min_value=0.01, max_value=100, allow_nan=False, allow_infinity=False),
    st.integers(min_value=-20, max_value=20).map(
        lambda exponent: math.ldexp(1, exponent)
    ),
)
_THRESHOLDS = {
    "minimum_cell_ess": 100,
    "maximum_contribution_cv": 2,
    "maximum_single_donor_dollar_share": 0.1,
    "target_minimum_cell_ess": 400,
    "target_maximum_single_donor_dollar_share": 0.05,
}


def _frame(weights, ids=None, *, kind=WeightKind.CALIBRATED, entity="household"):
    size = len(weights)
    tables = {
        "person": pd.DataFrame(
            {"person_id": range(size), "person_household_id": range(size)}
        ),
        "household": pd.DataFrame({"household_id": range(size)}),
    }
    if ids is not None:
        tables[entity]["source_household_id"] = ids
    return Frame(
        tables,
        EntitySchema(person_entity="person", group_entities=("household",)),
        {entity: Weights(np.asarray(weights, dtype=np.float64), kind)},
    )


def _report(gate, parameters, *, frame=None, artifacts=None):
    manifest = GatesManifest.from_mapping(
        {
            "country": "xx",
            "version": 1,
            "policy": "Synthetic weight-quality gate fixtures.",
            "phases": ["terminal"],
            "gates": [
                {
                    "id": "quality",
                    "gate": gate,
                    "phase": "terminal",
                    "criticality": "release_blocking",
                    "parameters": parameters,
                    "evidence_absent_blocks": True,
                }
            ],
        }
    )
    return evaluate_phase(
        manifest,
        "terminal",
        EvidenceContext(frame=frame, artifacts=artifacts or {}),
        registry=TRANSPORT_GATE_REGISTRY,
    )


def _source_parameters(**extra):
    return {
        "aggregation_column": "source_household_id",
        "required_weight_entity": "household",
        "required_weight_kind": "calibrated",
        **extra,
    }


def _cell_parameters():
    return _source_parameters(
        required_scenarios=["S0", "S1"],
        required_cells=["decile_1_single"],
        **_THRESHOLDS,
    )


def _cell_artifact(*, scenarios=("S0", "S1"), suppressed=True):
    diagnostics = source_cell_diagnostics([1, 1], [10, 1], [10, 20])
    return {
        "schema_version": 1,
        "kind": "transport_cell_support_diagnostics",
        "weight_entity": "household",
        "weight_kind": "calibrated",
        "source_household_column": "source_household_id",
        "scenarios": [
            {
                "scenario": scenario,
                "cells": [
                    {
                        "name": "decile_1_single",
                        "period": 2026,
                        **diagnostics,
                        "suppressed": suppressed,
                        "meets_tracked_target": False,
                    }
                ],
            }
            for scenario in scenarios
        ],
    }


@PROPERTY
@given(weights=_WEIGHTS)
def test_property_unique_sources_equal_row_ess_and_ratio_exactly(weights):
    row = weight_summary(weights)
    source = weight_summary(weights, source_household_ids=list(range(len(weights))))
    assert source["effective_sample_size"] == row["effective_sample_size"]
    assert (
        source["max_to_median_positive_weight"] == row["max_to_median_positive_weight"]
    )


@PROPERTY
@given(weights=_WEIGHTS)
def test_property_merging_clones_never_increases_ess(weights):
    row = weight_summary(weights)
    source = weight_summary(
        weights, source_household_ids=[index // 2 for index in range(len(weights))]
    )
    assert source["effective_sample_size"] <= row["effective_sample_size"] + 1e-12
    # Every weight is positive and at least one pair merges, so independence
    # drops strictly; using row weights cannot satisfy this stronger check.
    assert source["effective_sample_size"] < row["effective_sample_size"]


@PROPERTY
@given(weights=_WEIGHTS, scale=_SCALE)
def test_property_source_weight_ess_and_ratio_are_scale_invariant(weights, scale):
    ids = [index // 2 for index in range(len(weights))]
    original = weight_summary(weights, source_household_ids=ids)
    scaled = weight_summary(
        [weight * scale for weight in weights],
        source_household_ids=ids,
    )
    assert scaled["effective_sample_size"] == pytest.approx(
        original["effective_sample_size"], rel=1e-12
    )
    assert scaled["max_to_median_positive_weight"] == pytest.approx(
        original["max_to_median_positive_weight"], rel=1e-12
    )


def test_clones_cannot_pass_an_absolute_source_household_floor():
    weights = [1, 1, 1, 1, 1, 1]
    ids = ["original-c", "original-a", "original-b"] * 2
    assert weight_ess_gate(weights, minimum_effective_sample_size=4).passed
    report = _report(
        "weight_ess",
        _source_parameters(minimum_effective_sample_size=4),
        frame=_frame(weights, ids),
    )
    outcome = report.outcomes[0]
    assert outcome.status is GateStatus.FAILED
    assert outcome.result.details["effective_sample_size"] == 3
    assert report.blocking_outcomes(release_candidate=False)


def test_absolute_floor_does_not_change_when_row_count_changes():
    for count in (2, 3, 4, 7):
        result = weight_ess_gate(
            [1] * count,
            minimum_ess_fraction=0.5,
            minimum_effective_sample_size=4,
        )
        assert result.passed is (count >= 4)
        assert result.details["minimum_effective_sample_size"] == 4


def test_ratio_uses_aggregated_source_weights():
    weights = [1, 1, 1, 1, 1, 4]
    assert not weight_ratio_gate(weights, maximum_max_to_median_ratio=3).passed
    report = _report(
        "weight_ratio",
        _source_parameters(maximum_max_to_median_ratio=3),
        frame=_frame(weights, [10, 10, 20, 20, 30, 30]),
    )
    assert report.outcomes[0].status is GateStatus.PASSED
    assert report.outcomes[0].result.details["max_to_median_positive_weight"] == 2.5


@pytest.mark.parametrize(
    ("gate", "threshold"),
    [
        ("weight_ess", {"minimum_effective_sample_size": 1}),
        ("weight_ratio", {"maximum_max_to_median_ratio": 5}),
    ],
)
def test_source_weight_gate_missing_lineage_blocks(gate, threshold):
    report = _report(gate, _source_parameters(**threshold), frame=_frame([1, 1]))
    outcome = report.outcomes[0]
    assert outcome.status is GateStatus.FAILED
    assert "source_household_id" in " ".join(outcome.result.failures)
    assert report.blocking_outcomes(release_candidate=False)


@pytest.mark.parametrize(
    ("gate", "threshold"),
    [
        ("weight_ess", {"minimum_effective_sample_size": 1}),
        ("weight_ratio", {"maximum_max_to_median_ratio": 5}),
    ],
)
@pytest.mark.parametrize(
    ("kind", "entity", "reason"),
    [
        (WeightKind.DESIGN, "household", "calibrated"),
        (WeightKind.CALIBRATED, "person", "household"),
    ],
)
def test_source_weight_gates_refuse_wrong_weight_provenance(
    gate, threshold, kind, entity, reason
):
    report = _report(
        gate,
        _source_parameters(**threshold),
        frame=_frame([1, 1], [10, 20], kind=kind, entity=entity),
    )
    outcome = report.outcomes[0]
    assert outcome.status is GateStatus.FAILED
    assert reason in " ".join(outcome.result.failures)
    assert report.blocking_outcomes(release_candidate=False)


@PROPERTY
@given(
    weights=_WEIGHTS,
    scale=_SCALE,
    data=st.data(),
)
def test_property_every_cell_diagnostic_is_scale_invariant(weights, scale, data):
    amounts = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=1000),
            min_size=len(weights),
            max_size=len(weights),
        )
    )
    ids = [index // 2 for index in range(len(weights))]
    original = source_cell_diagnostics(weights, amounts, ids)
    scaled = source_cell_diagnostics(
        [weight * scale for weight in weights], amounts, ids
    )
    assert scaled.keys() == original.keys()
    for field in original:
        if isinstance(original[field], float):
            assert scaled[field] == pytest.approx(original[field], rel=1e-12, abs=1e-12)
        else:
            assert scaled[field] == original[field]


@PROPERTY
@given(
    weights=st.lists(st.integers(min_value=1, max_value=100), min_size=4, max_size=20),
    increase=st.integers(min_value=2, max_value=100),
    data=st.data(),
)
def test_property_increasing_a_donor_weight_never_decreases_its_dollar_share(
    weights, increase, data
):
    ids = np.arange(len(weights)) // 2
    source_count = int(ids.max()) + 1
    donor = data.draw(st.integers(min_value=0, max_value=source_count - 1))
    amounts = np.asarray(
        data.draw(
            st.lists(
                st.integers(min_value=1, max_value=1000),
                min_size=len(weights),
                max_size=len(weights),
            )
        )
    )
    weights = np.asarray(weights)
    before = source_cell_diagnostics(weights, amounts, ids)
    enlarged = weights.copy()
    enlarged[ids == donor] *= increase
    after = source_cell_diagnostics(enlarged, amounts, ids)
    contributions = np.zeros(source_count)
    larger_contributions = np.zeros(source_count)
    np.add.at(contributions, ids, weights * amounts)
    np.add.at(larger_contributions, ids, enlarged * amounts)
    shares = contributions / contributions.sum()
    larger_shares = larger_contributions / larger_contributions.sum()
    assert larger_shares[donor] >= shares[donor]
    assert before["maximum_single_donor_dollar_share"] == pytest.approx(
        float(shares.max()), rel=1e-12
    )
    assert after["maximum_single_donor_dollar_share"] == pytest.approx(
        float(larger_shares.max()), rel=1e-12
    )


def test_cell_diagnostics_merge_clone_dollars_and_distinct_contributors():
    diagnostics = source_cell_diagnostics([1, 1, 1, 1], [2, 3, 0, 5], [10, 10, 20, 30])
    assert diagnostics["source_household_ess"] == pytest.approx(8 / 3)
    assert diagnostics["positive_contributors"] == 2
    assert diagnostics["maximum_single_donor_dollar_share"] == 0.5
    assert diagnostics["dollar_contribution_ess"] == 2
    assert not diagnostics["structural_zero"]


def test_structural_zero_cell_is_identified_without_positive_contributors():
    diagnostics = source_cell_diagnostics([1, 1], [0, 0], [10, 20])
    assert diagnostics["positive_contributors"] == 0
    assert diagnostics["structural_zero"]
    assert diagnostics["dollar_contribution_ess"] == 0


def test_empty_declared_cell_is_a_structural_zero():
    diagnostics = source_cell_diagnostics(
        [1, 1], [10, 20], [10, 20], support=[False, False]
    )
    assert diagnostics == {
        "source_household_ess": 0,
        "positive_contributors": 0,
        "contribution_cv": 0,
        "maximum_single_donor_dollar_share": 0,
        "dollar_contribution_ess": 0,
        "structural_zero": True,
    }


def test_failing_cell_is_suppressed_before_reporting():
    diagnostics = source_cell_diagnostics([1, 1], [10, 1], [10, 20])
    original = {"E": 11, "A": 4, "gap": 7}
    screened = apply_cell_support(original, diagnostics, _THRESHOLDS)
    assert screened["suppressed"]
    assert screened["suppression_reasons"]
    assert screened["E"] is None
    assert screened["A"] is None
    assert screened["gap"] is None
    assert original == {"E": 11, "A": 4, "gap": 7}


def test_supported_cell_retains_its_entitlement_and_tracks_tighter_target():
    diagnostics = source_cell_diagnostics([1] * 500, [10] * 500, list(range(500)))
    original = {"E": 5000, "A": 4000, "gap": 1000}
    screened = apply_cell_support(original, diagnostics, _THRESHOLDS)
    assert not screened["suppressed"]
    assert screened["E"] == 5000
    assert screened["gap"] == 1000


def test_missing_cell_diagnostics_artifact_blocks_every_build_posture():
    report = _report("cell_support", _cell_parameters(), artifacts={})
    outcome = report.outcomes[0]
    assert outcome.status is GateStatus.EVIDENCE_ABSENT
    assert "cell_diagnostics" in outcome.reason
    assert report.blocking_outcomes(release_candidate=False)
    assert report.blocking_outcomes(release_candidate=True)


def test_missing_declared_scenario_in_cell_diagnostics_blocks():
    report = _report(
        "cell_support",
        _cell_parameters(),
        artifacts={"cell_diagnostics": _cell_artifact(scenarios=("S0",))},
    )
    outcome = report.outcomes[0]
    assert outcome.status is GateStatus.FAILED
    assert "S1" in " ".join(outcome.result.failures)
    assert report.blocking_outcomes(release_candidate=False)


def test_suppressed_failing_cells_are_diagnostic_and_do_not_block():
    report = _report(
        "cell_support",
        _cell_parameters(),
        artifacts={"cell_diagnostics": _cell_artifact()},
    )
    assert report.outcomes[0].status is GateStatus.PASSED
    assert report.outcomes[0].result.details["suppressed_cells"] == [
        "S0/decile_1_single",
        "S1/decile_1_single",
    ]
    assert not report.blocking_outcomes(release_candidate=True)


def test_unsuppressed_failing_cell_blocks_release():
    report = _report(
        "cell_support",
        _cell_parameters(),
        artifacts={"cell_diagnostics": _cell_artifact(suppressed=False)},
    )
    outcome = report.outcomes[0]
    assert outcome.status is GateStatus.FAILED
    assert "not suppressed" in " ".join(outcome.result.failures)
    assert report.blocking_outcomes(release_candidate=True)


def test_missing_declared_cell_in_diagnostics_blocks():
    parameters = _cell_parameters()
    parameters["required_cells"] = ["decile_1_single", "decile_2_single"]
    report = _report(
        "cell_support",
        parameters,
        artifacts={"cell_diagnostics": _cell_artifact()},
    )
    outcome = report.outcomes[0]
    assert outcome.status is GateStatus.FAILED
    assert "decile_2_single" in " ".join(outcome.result.failures)
    assert report.blocking_outcomes(release_candidate=False)


@pytest.mark.parametrize(
    ("key", "incorrect", "reason"),
    [
        ("weight_entity", "person", "weight_entity"),
        ("weight_kind", "design", "weight_kind"),
        ("source_household_column", "row_id", "source_household_column"),
    ],
)
def test_cell_diagnostics_wrong_provenance_blocks(key, incorrect, reason):
    artifact = _cell_artifact()
    artifact[key] = incorrect
    report = _report(
        "cell_support", _cell_parameters(), artifacts={"cell_diagnostics": artifact}
    )
    outcome = report.outcomes[0]
    assert outcome.status is GateStatus.FAILED
    assert reason in " ".join(outcome.result.failures)
    assert report.blocking_outcomes(release_candidate=False)


def test_adopted_global_target_is_recorded_and_never_blocks():
    gates = load_country_spec("nz").gates
    entry = next(
        row for row in gates.gates if row.id == "weight_effective_sample_size_target_a"
    )
    report = evaluate_phase(
        replace(gates, gates=(entry,)),
        "terminal",
        EvidenceContext(frame=_frame([1] * 5, list(range(5)))),
        registry=TRANSPORT_GATE_REGISTRY,
    )
    outcome = report.outcomes[0]
    assert outcome.status is GateStatus.FAILED
    assert outcome.result.details["effective_sample_size"] == 5
    assert outcome.result.details["minimum_effective_sample_size"] == 16000
    assert not report.blocking_outcomes(release_candidate=True)


def test_adopted_cell_targets_are_flagged_and_never_block():
    gates = load_country_spec("nz").gates
    entry = next(
        row for row in gates.gates if row.id == "reported_cell_support_target_a"
    )
    artifact = _cell_artifact(scenarios=entry.parameters["required_scenarios"])
    diagnostics = source_cell_diagnostics([1] * 5, [10] * 5, list(range(5)))
    for scenario in artifact["scenarios"]:
        scenario["cells"] = [
            {
                "name": name,
                "period": 2026,
                **diagnostics,
                "suppressed": True,
                "meets_tracked_target": False,
            }
            for name in entry.parameters["required_cells"]
        ]
    report = evaluate_phase(
        replace(gates, gates=(entry,)),
        "terminal",
        EvidenceContext(artifacts={"cell_diagnostics": artifact}),
        registry=TRANSPORT_GATE_REGISTRY,
    )
    outcome = report.outcomes[0]
    assert outcome.status is GateStatus.FAILED
    flags = outcome.result.details["meets_tracked_target"]
    assert len(flags) == 43 * 10
    assert not any(flags.values())
    assert not report.blocking_outcomes(release_candidate=True)


def test_adopted_four_thousand_floor_rejects_cloned_row_support():
    gates = load_country_spec("nz").gates
    entry = next(row for row in gates.gates if row.id == "weight_effective_sample_size")
    weights = np.ones(6000)
    assert weight_ess_gate(weights, minimum_effective_sample_size=4000).passed
    report = evaluate_phase(
        replace(gates, gates=(entry,)),
        "terminal",
        EvidenceContext(frame=_frame(weights, np.tile(np.arange(3000), 2))),
        registry=TRANSPORT_GATE_REGISTRY,
    )
    outcome = report.outcomes[0]
    assert outcome.status is GateStatus.FAILED
    assert outcome.result.details["effective_sample_size"] == 3000
    assert outcome.result.details["minimum_effective_sample_size"] == 4000
    assert report.blocking_outcomes(release_candidate=True)
