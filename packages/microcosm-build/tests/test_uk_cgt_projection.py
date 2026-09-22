"""The #970 projection fence: the engine path, the evaluator, the binding."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.gate_battery import EvidenceContext, GateStatus, evaluate_phase
from microcosm.build.uk_runtime.battery_bindings import UK_GATE_REGISTRY
from microcosm.build.uk_runtime.calibration_run import UK_CALIBRATION_GATE_SCOPE
from microcosm.build.uk_runtime.cgt_projection import (
    UK_CGT_EXEMPT_AMOUNT_PARAMETER,
    UK_CGT_GAINS_GROWTH_PARAMETER,
    UK_CGT_PROJECTION_ARTIFACT_KEY,
    UKCGTProjection,
    uk_cgt_projection,
)
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_CONDITIONING_RESOURCE,
    load_hmrc_cgt_conditioning_facts,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.terminal_gates import uk_cgt_projection_entrants_gate

GATE_ID = "uk_cgt_projection_entrants"


class _Reader:
    """A parameter reader that records every (path, year) it is asked for."""

    def __init__(self, *, growth: float = 0.03, exempt: float = 3_000.0) -> None:
        self.growth = growth
        self.exempt = exempt
        self.calls: list[tuple[str, int]] = []

    def __call__(self, path: str, year: int) -> float:
        self.calls.append((path, year))
        if path == UK_CGT_GAINS_GROWTH_PARAMETER:
            return self.growth
        if path == UK_CGT_EXEMPT_AMOUNT_PARAMETER:
            return self.exempt
        raise AssertionError(path)


def _projection(base_year: int = 2024, horizon_year: int = 2030, **reader_kwargs):
    return uk_cgt_projection(
        base_year, horizon_year, parameter_reader=_Reader(**reader_kwargs)
    )


def _frame(gains: list[float], weights: list[float], *, time_period: str = "2024"):
    ids = np.arange(1, len(gains) + 1, dtype="int64")
    return uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": ids,
                "person_benunit_id": ids,
                "person_household_id": ids,
                "capital_gains": np.asarray(gains, dtype=float),
            }
        ),
        benunit=pd.DataFrame({"benunit_id": ids}),
        household=pd.DataFrame(
            {"household_id": ids, "household_weight": np.asarray(weights, dtype=float)}
        ),
        time_period=time_period,
    )


def test_projection_reads_exactly_the_declared_paths_and_compounds() -> None:
    reader = _Reader()
    projection = uk_cgt_projection(2024, 2027, parameter_reader=reader)

    assert reader.calls == [
        (UK_CGT_EXEMPT_AMOUNT_PARAMETER, 2024),
        (UK_CGT_GAINS_GROWTH_PARAMETER, 2025),
        (UK_CGT_EXEMPT_AMOUNT_PARAMETER, 2025),
        (UK_CGT_GAINS_GROWTH_PARAMETER, 2026),
        (UK_CGT_EXEMPT_AMOUNT_PARAMETER, 2026),
        (UK_CGT_GAINS_GROWTH_PARAMETER, 2027),
        (UK_CGT_EXEMPT_AMOUNT_PARAMETER, 2027),
    ]
    assert projection.projected_years == (2025, 2026, 2027)
    assert list(projection.cumulative_gains_factor_by_year.values()) == pytest.approx(
        [1.03, 1.03**2, 1.03**3]
    )
    assert set(projection.exempt_amount_by_year) == {"2024", "2025", "2026", "2027"}
    assert set(projection.exempt_amount_by_year.values()) == {3_000.0}
    assert projection.engine == "supplied_parameter_reader"
    payload = projection.payload()
    assert set(payload) == {
        "base_year",
        "horizon_year",
        "growth_parameter",
        "exempt_amount_parameter",
        "instant_rule",
        "engine",
        "yoy_growth_by_year",
        "cumulative_gains_factor_by_year",
        "exempt_amount_by_year",
    }
    assert payload["instant_rule"] == "january_first"


def test_projection_refuses_unusable_inputs() -> None:
    with pytest.raises(ValueError, match="must follow the base year"):
        _projection(2024, 2024)
    with pytest.raises(ValueError, match="growth"):
        _projection(growth=-1.5)
    with pytest.raises(ValueError, match="growth"):
        _projection(growth=math.nan)
    with pytest.raises(ValueError, match="exempt amount"):
        _projection(exempt=0.0)
    with pytest.raises(ValueError, match="every projected year"):
        UKCGTProjection(
            base_year=2024,
            horizon_year=2026,
            growth_parameter=UK_CGT_GAINS_GROWTH_PARAMETER,
            exempt_amount_parameter=UK_CGT_EXEMPT_AMOUNT_PARAMETER,
            yoy_growth_by_year={"2025": 0.03},
            cumulative_gains_factor_by_year={"2025": 1.03},
            exempt_amount_by_year={"2024": 3_000.0, "2025": 3_000.0},
            engine="test",
        )


def test_gate_counts_entrants_year_by_year_against_the_bound() -> None:
    projection = _projection(growth=0.05)
    # At the exempt amount: crosses in the first year. 2,800 needs a factor
    # above 1.0714, so the second year. 2,000 needs 1.5, beyond the horizon.
    # Losses, zeros and liable gains are never entrants.
    person = pd.DataFrame(
        {"capital_gains": [3_000.0, 3_000.0, 2_000.0, 2_800.0, -50.0, 0.0, 10_000.0]}
    )
    weights = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])

    passed = uk_cgt_projection_entrants_gate(
        person, weights, projection, bound=73_000.0, bound_source="test"
    )
    assert passed.passed
    assert passed.name == "cgt_projection_entrants"
    entrants = passed.details["entrants_by_year"]
    assert entrants["2025"] == 30.0
    assert all(entrants[str(year)] == 70.0 for year in range(2026, 2031))
    assert passed.details["worst_year"] == 2026
    assert passed.details["max_entrants"] == 70.0
    sub_exempt = passed.details["sub_exempt"]
    assert sub_exempt["rows"] == 4
    assert sub_exempt["weighted_persons"] == 100.0
    assert sub_exempt["weighted_at_exempt_amount"] == 30.0
    assert sub_exempt["p10"] == 2_000.0
    assert sub_exempt["p50"] == 2_800.0
    assert sub_exempt["p90"] == 3_000.0
    assert "min" not in sub_exempt and "max" not in sub_exempt

    # Equality to the bound passes; one person more fails.
    assert uk_cgt_projection_entrants_gate(
        person, weights, projection, bound=70.0, bound_source="test"
    ).passed
    failed = uk_cgt_projection_entrants_gate(
        person, weights, projection, bound=69.0, bound_source="the thin band"
    )
    assert not failed.passed
    assert "by 2026" in failed.failures[0]
    assert "the thin band" in failed.failures[0]


def test_gate_fails_closed_on_missing_or_malformed_evidence() -> None:
    projection = _projection()
    with pytest.raises(ValueError, match="capital_gains"):
        uk_cgt_projection_entrants_gate(
            pd.DataFrame({"age": [1.0]}),
            np.ones(1),
            projection,
            bound=1.0,
            bound_source="test",
        )
    with pytest.raises(ValueError, match="one weight per person"):
        uk_cgt_projection_entrants_gate(
            pd.DataFrame({"capital_gains": [1.0, 2.0]}),
            np.ones(1),
            projection,
            bound=1.0,
            bound_source="test",
        )
    with pytest.raises(ValueError, match="positive finite bound"):
        uk_cgt_projection_entrants_gate(
            pd.DataFrame({"capital_gains": [1.0]}),
            np.ones(1),
            projection,
            bound=0.0,
            bound_source="test",
        )


def _outcome(frame, artifacts):
    battery = evaluate_phase(
        load_country_spec("uk").gates,
        "terminal",
        EvidenceContext(frame=frame, artifacts=artifacts),
        registry=UK_GATE_REGISTRY,
    )
    return battery, next(o for o in battery.outcomes if o.entry.id == GATE_ID)


def test_binding_reads_the_vendored_bound_and_cross_checks_the_artifact() -> None:
    projection = _projection(growth=0.05)
    heavy = _frame([3_000.0, 2_000.0], [80_000.0, 10.0])
    battery, outcome = _outcome(heavy, {UK_CGT_PROJECTION_ARTIFACT_KEY: projection})
    assert outcome.status is GateStatus.FAILED
    assert outcome.result.details["bound"] == 73_000.0
    assert outcome.result.details["max_entrants"] == 80_000.0
    assert "tax year 2024" in outcome.result.details["bound_source"]
    assert HMRC_CGT_CONDITIONING_RESOURCE in outcome.result.details["bound_source"]

    light = _frame([3_000.0, 2_000.0], [72_999.0, 10.0])
    _, outcome = _outcome(light, {UK_CGT_PROJECTION_ARTIFACT_KEY: projection})
    assert outcome.status is GateStatus.PASSED

    mismatched = _projection(horizon_year=2029, growth=0.05)
    _, outcome = _outcome(light, {UK_CGT_PROJECTION_ARTIFACT_KEY: mismatched})
    assert outcome.status is GateStatus.FAILED
    assert "disagrees with the declared projection" in "".join(outcome.result.failures)

    wrong_base = _frame([3_000.0], [1.0], time_period="2023")
    _, outcome = _outcome(wrong_base, {UK_CGT_PROJECTION_ARTIFACT_KEY: projection})
    assert outcome.status is GateStatus.FAILED

    battery, outcome = _outcome(light, {})
    assert outcome.status is GateStatus.EVIDENCE_ABSENT
    assert GATE_ID in {
        o.entry.id for o in battery.blocking_outcomes(release_candidate=False)
    }


def test_manifest_entry_and_bound_are_the_reviewed_ones() -> None:
    entry = next(g for g in load_country_spec("uk").gates.gates if g.id == GATE_ID)
    assert entry.gate == "cgt_projection_entrants"
    assert entry.phase == "terminal"
    assert entry.criticality == "release_blocking"
    assert entry.evidence_absent_blocks is True
    assert dict(entry.parameters) == {
        "horizon_year": 2030,
        "gains_growth_parameter": UK_CGT_GAINS_GROWTH_PARAMETER,
        "exempt_amount_parameter": UK_CGT_EXEMPT_AMOUNT_PARAMETER,
        "bound_resource": HMRC_CGT_CONDITIONING_RESOURCE,
        "bound_size_band_lower_bound": 3_000,
    }
    assert GATE_ID in UK_CALIBRATION_GATE_SCOPE
    binding = UK_GATE_REGISTRY["cgt_projection_entrants"]
    assert binding.artifact_keys == frozenset({UK_CGT_PROJECTION_ARTIFACT_KEY})
    assert binding.parameter_keys == frozenset(entry.parameters)
    facts = load_hmrc_cgt_conditioning_facts(HMRC_CGT_CONDITIONING_RESOURCE)
    band = facts.size_band(3_000)
    assert facts.tax_year == 2024
    assert (band.lower_bound, band.upper_bound, band.taxpayers) == (
        3_000,
        6_000,
        73_000.0,
    )


@pytest.mark.requires_uk
def test_engine_projection_matches_the_published_growth_path() -> None:
    projection = uk_cgt_projection(2024, 2030)

    assert projection.engine.startswith("policyengine-uk==")
    assert set(projection.exempt_amount_by_year.values()) == {3_000.0}
    assert list(projection.yoy_growth_by_year.values()) == pytest.approx(
        [0.0438, 0.0292, 0.0323, 0.0310, 0.0296, 0.0315], abs=5e-4
    )
    assert projection.cumulative_gains_factor_by_year["2030"] == pytest.approx(
        1.2143, abs=5e-4
    )
