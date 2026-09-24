"""The #970 projection fence: the engine path, the evaluator, the binding."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from microcosm.build.country_spec import load_country_spec
from microcosm.build.gate_battery import EvidenceContext, GateStatus, evaluate_phase
from microcosm.build.uk_runtime import calibration_run
from microcosm.build.uk_runtime.battery_bindings import UK_GATE_REGISTRY
from microcosm.build.uk_runtime.calibration_run import UK_CALIBRATION_GATE_SCOPE
from microcosm.build.uk_runtime.cgt_projection import (
    UK_CGT_EXEMPT_AMOUNT_PARAMETER,
    UK_CGT_GAINS_GROWTH_PARAMETER,
    UK_CGT_PROJECTION_ARTIFACT_KEY,
    UK_CGT_PROJECTION_PINS_ENGINE,
    UKCGTProjection,
    uk_cgt_projection,
    uk_cgt_projection_from_pins,
    uk_cgt_projection_read_from_engine,
    uk_engine_installed,
)
from microcosm.build.uk_runtime.hmrc_capital_gains import (
    HMRC_CGT_CONDITIONING_RESOURCE,
    load_hmrc_cgt_conditioning_facts,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.build.uk_runtime.terminal_gates import uk_cgt_projection_entrants_gate
from test_support.microcosm_build.uk_cgt_projection import GATE_ID, manifest_entry


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


class _PinnedReader:
    """A reader that returns the manifest's pinned growth path and exempt amount."""

    def __init__(self, *, growth_offset: float = 0.0) -> None:
        parameters = manifest_entry().parameters
        self.growth = {
            int(year): float(rate) + growth_offset
            for year, rate in parameters["expected_yoy_growth_by_year"].items()
        }
        self.exempt = {
            int(year): float(amount)
            for year, amount in parameters["expected_exempt_amount_by_year"].items()
        }

    def __call__(self, path: str, year: int) -> float:
        if path == UK_CGT_GAINS_GROWTH_PARAMETER:
            return self.growth[year]
        if path == UK_CGT_EXEMPT_AMOUNT_PARAMETER:
            return self.exempt[year]
        raise AssertionError(path)


def _pinned_projection(base_year: int = 2024, horizon_year: int = 2030, **kwargs):
    return uk_cgt_projection(
        base_year, horizon_year, parameter_reader=_PinnedReader(**kwargs)
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
    projection = _pinned_projection()
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

    mismatched = _pinned_projection(horizon_year=2029)
    _, outcome = _outcome(light, {UK_CGT_PROJECTION_ARTIFACT_KEY: mismatched})
    assert outcome.status is GateStatus.FAILED
    assert "disagrees with the declared projection" in "".join(outcome.result.failures)

    wrong_base = _frame([3_000.0], [1.0], time_period="2023")
    _, outcome = _outcome(wrong_base, {UK_CGT_PROJECTION_ARTIFACT_KEY: projection})
    assert outcome.status is GateStatus.FAILED

    # An engine whose growth path moved off the pinned one fails visibly.
    drifted = _pinned_projection(growth_offset=0.001)
    _, outcome = _outcome(light, {UK_CGT_PROJECTION_ARTIFACT_KEY: drifted})
    assert outcome.status is GateStatus.FAILED
    assert "drifted from the pinned path" in "".join(outcome.result.failures)
    # A projection past the declared horizon is refused as a disagreement
    # with the manifest before any drift check runs; the unpinned-year branch
    # of the drift check has its own test below.
    beyond_horizon = _projection(growth=0.05, horizon_year=2031)
    _, outcome = _outcome(light, {UK_CGT_PROJECTION_ARTIFACT_KEY: beyond_horizon})
    assert outcome.status is GateStatus.FAILED
    assert "disagrees with the declared projection" in "".join(outcome.result.failures)

    battery, outcome = _outcome(light, {})
    assert outcome.status is GateStatus.EVIDENCE_ABSENT
    assert GATE_ID in {
        o.entry.id for o in battery.blocking_outcomes(release_candidate=False)
    }


def test_binding_names_the_year_the_pins_do_not_cover() -> None:
    """A projection that reaches a year the manifest does not pin fails the
    drift check, and the failure names the year for both the growth rate and
    the exempt amount, so the re-pin a horizon change needs is spelled out."""

    parameters = dict(manifest_entry().parameters)
    parameters["expected_yoy_growth_by_year"] = {
        year: rate
        for year, rate in parameters["expected_yoy_growth_by_year"].items()
        if year != "2030"
    }
    parameters["expected_exempt_amount_by_year"] = {
        year: amount
        for year, amount in parameters["expected_exempt_amount_by_year"].items()
        if year != "2030"
    }
    binding = UK_GATE_REGISTRY["cgt_projection_entrants"]
    context = EvidenceContext(
        frame=_frame([2_000.0], [1.0]),
        artifacts={UK_CGT_PROJECTION_ARTIFACT_KEY: _pinned_projection()},
    )
    with pytest.raises(ValueError, match="drifted from the pinned path") as excinfo:
        binding.evaluator(context, parameters)
    assert "no pinned growth rate for 2030" in str(excinfo.value)
    assert "no pinned exempt amount for 2030" in str(excinfo.value)


def test_manifest_entry_and_bound_are_the_reviewed_ones() -> None:
    entry = manifest_entry()
    assert entry.gate == "cgt_projection_entrants"
    assert entry.phase == "terminal"
    assert entry.criticality == "release_blocking"
    assert entry.evidence_absent_blocks is True
    assert dict(entry.parameters) == {
        "horizon_year": 2030,
        "gains_growth_parameter": UK_CGT_GAINS_GROWTH_PARAMETER,
        "exempt_amount_parameter": UK_CGT_EXEMPT_AMOUNT_PARAMETER,
        # The OBR per-capita path as the engine carries it; 2030 is its last
        # published year and later years repeat the 2030 rate.
        "expected_yoy_growth_by_year": {
            "2023": 0.0532,
            "2024": 0.0372,
            "2025": 0.0438,
            "2026": 0.0292,
            "2027": 0.0323,
            "2028": 0.031,
            "2029": 0.0296,
            "2030": 0.0315,
        },
        # 6,000 in 2023-24, 3,000 from 2024-25: a 2023-period frame's base
        # year reads the older amount.
        "expected_exempt_amount_by_year": {
            "2023": 6_000.0,
            **{str(year): 3_000.0 for year in range(2024, 2031)},
        },
        "maximum_growth_drift": 0.0005,
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


def test_projection_from_pins_is_the_engine_free_statement_of_the_path() -> None:
    parameters = manifest_entry().parameters
    pinned = uk_cgt_projection_from_pins(
        2023,
        2030,
        growth_by_year=parameters["expected_yoy_growth_by_year"],
        exempt_amount_by_year=parameters["expected_exempt_amount_by_year"],
    )
    assert pinned.engine == UK_CGT_PROJECTION_PINS_ENGINE
    assert pinned.exempt_amount_by_year["2023"] == 6_000.0
    assert pinned.exempt_amount_by_year["2024"] == 3_000.0
    assert pinned.yoy_growth_by_year["2024"] == pytest.approx(0.0372)
    with pytest.raises(ValueError, match="no .* value for 2031"):
        uk_cgt_projection_from_pins(
            2024,
            2031,
            growth_by_year=parameters["expected_yoy_growth_by_year"],
            exempt_amount_by_year=parameters["expected_exempt_amount_by_year"],
        )


def test_seam_artifact_states_the_pins_only_when_no_engine_is_installed(
    monkeypatch,
) -> None:
    """The secrets-free fast lane and data-only builds have no engine: the
    seam still produces the projection, from the pins, without touching the
    engine path, and the receipt says so. The release certifier refuses that
    receipt (``test_uk_release_certification``)."""

    monkeypatch.setattr(calibration_run, "uk_engine_installed", lambda: False)

    def never(*args, **kwargs):
        raise AssertionError("an absent engine must not be read")

    monkeypatch.setattr(calibration_run, "uk_cgt_projection", never)
    frame = _frame([3_000.0, 2_000.0], [1.0, 1.0], time_period="2023")
    projection = calibration_run.uk_cgt_projection_artifact(
        frame, load_country_spec("uk").gates
    )
    assert projection.engine == UK_CGT_PROJECTION_PINS_ENGINE
    assert not uk_cgt_projection_read_from_engine(projection.engine)
    assert projection.base_year == 2023 and projection.horizon_year == 2030
    assert projection.exempt_amount_by_year["2023"] == 6_000.0
    # The binding accepts a pinned projection (it drift-checks against
    # itself); only the certifier refuses it.
    battery, outcome = _outcome(frame, {UK_CGT_PROJECTION_ARTIFACT_KEY: projection})
    assert outcome.status is GateStatus.PASSED
    assert outcome.result.details["projection_engine"] == UK_CGT_PROJECTION_PINS_ENGINE


def test_seam_artifact_lets_an_installed_engine_that_will_not_import_raise(
    monkeypatch,
) -> None:
    """An installed but broken engine is not "unavailable": the import error
    surfaces instead of the pins quietly standing in for the engine."""

    monkeypatch.setattr(calibration_run, "uk_engine_installed", lambda: True)

    def broken(*args, **kwargs):
        raise ImportError("libomp.dylib not found")

    monkeypatch.setattr(calibration_run, "uk_cgt_projection", broken)
    frame = _frame([3_000.0], [1.0], time_period="2023")
    with pytest.raises(ImportError, match="libomp"):
        calibration_run.uk_cgt_projection_artifact(frame, load_country_spec("uk").gates)


def test_engine_is_reported_unavailable_without_the_uk_extra() -> None:
    assert not uk_engine_installed()


def test_only_a_versioned_engine_label_reads_as_the_engine() -> None:
    assert uk_cgt_projection_read_from_engine("policyengine-uk==2.98.0")
    assert not uk_cgt_projection_read_from_engine(UK_CGT_PROJECTION_PINS_ENGINE)
    assert not uk_cgt_projection_read_from_engine(
        "policyengine-uk (version unavailable)"
    )
    assert not uk_cgt_projection_read_from_engine("supplied_parameter_reader")
    assert not uk_cgt_projection_read_from_engine("policyengine-uk==")
    assert not uk_cgt_projection_read_from_engine(None)
