from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import measure_simulation
from microcosm.build.uk_runtime.income_tax_rate_classes import UK_RATE_CLASS_ALLOCATION
from microcosm.build.uk_runtime.measure_simulation import (
    UK_SAVINGS_INTEREST_OUTSIDE_ISAS,
    UKMeasureResolver,
    apply_uk_calibration_measure_exclusions,
    compute_uk_measure_input,
    load_uk_calibration_measure_exclusions,
)
from microcosm.calibrate import TargetRegistry, TargetSpec
from microcosm.frame.adapters.policyengine_uk import (
    UK_ISA_INCOME_CONTRACT_ATTR,
    UK_LEGACY_ISA_INCOME_CONTRACT,
)


class FrameStub:
    def __init__(self):
        self._tables = {
            "person": pd.DataFrame(
                {
                    "person_id": [1, 2, 3],
                    "person_benunit_id": [10, 10, 20],
                    "person_household_id": [100, 100, 200],
                }
            ),
            "benunit": pd.DataFrame({"benunit_id": [10, 20]}),
            "household": pd.DataFrame({"household_id": [100, 200]}),
        }

    def table(self, entity):
        return self._tables[entity]


class ReleaseIdentifierFrameStub(FrameStub):
    """The stub on the identifiers a release carries: one plus the rank per
    entity (microcosm#1194), which a scratch engine file leaves as they are."""

    def __init__(self):
        self._tables = {
            "person": pd.DataFrame(
                {
                    "person_id": [1, 2, 3],
                    "person_benunit_id": [1, 1, 2],
                    "person_household_id": [1, 1, 2],
                }
            ),
            "benunit": pd.DataFrame({"benunit_id": [1, 2]}),
            "household": pd.DataFrame({"household_id": [1, 2]}),
        }


def _stub_value_type(values):
    """The policyengine ``value_type`` a real variable definition would carry."""

    return {"f": float, "i": int, "b": bool}.get(np.asarray(values).dtype.kind, str)


class SimulationStub:
    def __init__(self, values):
        self.values = values
        variables = {
            name: SimpleNamespace(
                entity=SimpleNamespace(key=entity),
                value_type=_stub_value_type(stub_values),
            )
            for name, (entity, stub_values) in values.items()
        }
        self.tax_benefit_system = SimpleNamespace(variables=variables)
        self.calls = []

    def calculate(self, variable, year, map_to=None):
        self.calls.append((variable, year, map_to))
        key = (variable, map_to)
        if key in self.values:
            return self.values[key][1]
        return self.values[variable][1]


@pytest.mark.parametrize("missing_variable", [True, False])
def test_direct_measure_resolver_refuses_dropped_uc_claimant_input(
    monkeypatch, tmp_path, missing_variable
):
    frame = FrameStub()
    frame.table("person")["is_uc_claimant"] = [True, False, True]
    simulation = SimulationStub(
        {"is_uc_claimant": ("person", np.array([True, False, True]))}
    )
    simulation.input_variables = []
    if missing_variable:
        simulation.tax_benefit_system.variables.clear()
    else:
        simulation.tax_benefit_system.variables[
            "is_uc_claimant"
        ].definition_period = "year"
    monkeypatch.setattr(
        measure_simulation,
        "_policyengine_uk_module",
        lambda: SimpleNamespace(__version__="2.94.0"),
    )
    with pytest.raises(ValueError, match="is_uc_claimant"):
        UKMeasureResolver(
            simulation_source=tmp_path / "synthetic.h5",
            scratch_dir=tmp_path,
            year=2025,
            frame=frame,
            microsimulation_factory=lambda **kwargs: simulation,
        )


def test_compute_uk_measure_input_native_route():
    sim = SimulationStub({"income_tax": ("person", np.array([1.0, 2.0, 3.0]))})

    values, route = compute_uk_measure_input(
        FrameStub(), sim, "person", "income_tax", 2025
    )

    assert route == "native"
    assert values.tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("year,factor", [(2024, 1.0), (2025, 1.1)])
def test_cgt_period_measures_bypass_stored_inputs_without_mutation(
    monkeypatch, tmp_path, year, factor
):
    from microcosm.build.target_materialization import (
        materialize_target_bindings,
        resolve_target_measures,
    )

    frame = FrameStub()
    frame.metadata = {"time_period": "2024"}
    frame.table("person")["capital_gains"] = [2900.0, 3000.0, 20000.0]
    frame.table("person")["capital_gains_tax"] = [99.0, 99.0, 99.0]
    original = frame.table("person").copy(deep=True)
    gains = original.capital_gains.to_numpy() * factor
    tax = np.maximum(gains - 3000, 0) * 0.18
    sim = SimulationStub(
        {"capital_gains": ("person", gains), "capital_gains_tax": ("person", tax)}
    )
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(__version__="2.98.0", Microsimulation=lambda **kw: sim),
    )
    resolver = UKMeasureResolver(
        simulation_source=tmp_path / "input.h5",
        scratch_dir=tmp_path,
        year=year,
        frame=frame,
    )
    names = (
        "hmrc.cgt.gains_total",
        "hmrc.cgt.taxpayers_total",
        "hmrc.cgt.liability_total",
    )
    resolver.contract_targets = {
        name: {"bindings": {"policyengine": binding}}
        for name, binding in zip(
            names,
            [
                {
                    "kind": "parameter_gated_threshold",
                    "gate_parameter": "gov.hmrc.cgt.annual_exempt_amount",
                    "gated_variable": "cgt_calibration_gains",
                    "value_variable": "cgt_calibration_gains",
                    "from_entity": "person",
                    "measurement_period": year,
                },
                {
                    "kind": "parameter_gated_threshold",
                    "gate_parameter": "gov.hmrc.cgt.annual_exempt_amount",
                    "gated_variable": "cgt_calibration_gains",
                    "value_variable": "person_count",
                    "from_entity": "person",
                    "measurement_period": year,
                },
                {
                    "value_variable": "cgt_calibration_tax",
                    "from_entity": "person",
                    "measurement_period": year,
                },
            ],
            strict=True,
        )
    }
    registry = TargetRegistry(
        [
            TargetSpec(
                name=name,
                entity="person",
                measure=f"test_{i}",
                value=1,
                source="test",
                metadata={"contract_target_id": name},
            )
            for i, name in enumerate(names)
        ],
        country="uk",
    )

    class Adapter:
        def __init__(self):
            self.person = original.copy(deep=True)
            self.tables = {"person": self.person}

        def column(self, entity, variable):
            if variable not in self.person:
                raise KeyError(f"{entity}.{variable}")
            return self.person[variable].to_numpy()

        def set_column(self, entity, variable, values):
            self.person[variable] = values

        def parameter(self, path, period):
            assert period == year
            return 3000.0

    result = resolve_target_measures(Adapter, registry, resolver, period=year)
    adapter = Adapter()
    for (entity, variable), values in result.measure_inputs.items():
        adapter.set_column(entity, variable, values)
    materialized = materialize_target_bindings(
        adapter, registry, resolver.contract_targets, period=year
    )
    assert not materialized.skipped
    np.testing.assert_allclose(adapter.person.test_0, np.where(gains > 3000, gains, 0))
    np.testing.assert_array_equal(adapter.person.test_1, gains > 3000)
    np.testing.assert_allclose(adapter.person.test_2, tax)
    assert {call[:2] for call in sim.calls} == {
        ("capital_gains", year),
        ("capital_gains_tax", year),
    }
    pd.testing.assert_frame_equal(frame.table("person"), original)
    pd.testing.assert_series_equal(adapter.person.capital_gains, original.capital_gains)
    receipt = result.receipt["provider"]["cgt_period_contract"]
    assert receipt["input_period"] == "2024"
    assert receipt["default_engine_period"] == year
    assert receipt["gains_basis"] == "after_losses_before_annual_exempt_amount"
    with pytest.raises(ValueError, match="measurement period"):
        resolve_target_measures(Adapter, registry, resolver, period=year + 1)


def test_resolver_refuses_persisted_cgt_measure_aliases(monkeypatch, tmp_path):
    frame = FrameStub()
    frame.table("person")["cgt_calibration_gains"] = [1, 2, 3]
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(
            __version__="2.98.0", Microsimulation=lambda **kw: SimulationStub({})
        ),
    )
    with pytest.raises(ValueError, match="must not be persisted"):
        UKMeasureResolver(
            simulation_source=tmp_path / "input.h5",
            scratch_dir=tmp_path,
            year=2025,
            frame=frame,
        )


@pytest.mark.parametrize("measurement_period", [2023, 2025])
def test_resolver_rejects_amount_threshold_year_mismatch(
    monkeypatch, tmp_path, measurement_period
):
    monkeypatch.setattr(
        measure_simulation,
        "_policyengine_uk_module",
        lambda: SimpleNamespace(__version__="test"),
    )
    monkeypatch.setattr(
        measure_simulation,
        "_uk_contract_targets",
        lambda: {
            "cgt": {
                "bindings": {
                    "policyengine": {
                        "value_variable": "cgt_2024_gains",
                        "measurement_period": measurement_period,
                    }
                }
            }
        },
    )
    with pytest.raises(ValueError, match="amount/threshold period mismatch"):
        UKMeasureResolver(
            simulation_source=tmp_path / "input.h5",
            scratch_dir=tmp_path,
            frame=FrameStub(),
            year=2025,
            microsimulation_factory=lambda **_: SimulationStub({}),
        )


def test_dated_cgt_measure_uses_disposal_year_with_later_calibration_year():
    sim = SimulationStub(
        {"capital_gains": ("person", np.array([3000.0, 5000.0, 20000.0]))}
    )
    values, route = compute_uk_measure_input(
        FrameStub(), sim, "person", "cgt_2024_gains", 2025
    )
    assert sim.calls == [("capital_gains", 2024, None)]
    assert route == "engine_period:2024:capital_gains:native"
    assert (values > 3000).tolist() == [False, True, True]


@pytest.mark.parametrize("precomputed", [False, True])
def test_dated_cgt_binding_gates_at_observation_year_and_refuses_stale_fact(
    precomputed,
):
    from microcosm.build.target_materialization import materialize_target_bindings

    class Adapter:
        values = np.array([999.0, 999.0, 999.0])

        def has_column(self, entity, name):
            return precomputed

        def parameter(self, name, period):
            assert period == 2024
            return 3000.0

        def column(self, entity, variable):
            assert variable == "cgt_2024_gains"
            return np.array([3000.0, 5000.0, 20000.0])

        def set_column(self, entity, variable, values):
            self.values = values

    binding = {
        "bindings": {
            "policyengine": {
                "kind": "parameter_gated_threshold",
                "gate_parameter": "gov.hmrc.cgt.annual_exempt_amount",
                "gated_variable": "cgt_2024_gains",
                "value_variable": "person_count",
                "from_entity": "person",
                "measurement_period": 2024,
                "require_matching_fact_period": True,
            }
        }
    }
    for fact_year in (2024, 2023):
        registry = TargetRegistry(
            [
                TargetSpec(
                    name="cgt",
                    entity="person",
                    measure="cgt",
                    value=1,
                    source="test",
                    metadata={
                        "contract_target_id": "cgt",
                        "ledger_fact_period": str(fact_year),
                    },
                )
            ],
            country="uk",
        )
        adapter = Adapter()
        result = materialize_target_bindings(
            adapter, registry, {"cgt": binding}, period=2025
        )
        if fact_year == 2024:
            assert not result.skipped
            assert adapter.values.tolist() == [0.0, 1.0, 1.0]
        else:
            assert len(result.skipped) == 1
            assert "observation period" in result.skipped[0].reason


def test_compute_uk_measure_input_categorical_broadcast_group_to_person():
    sim = SimulationStub({"family_type": ("benunit", np.array(["couple", "single"]))})

    values, route = compute_uk_measure_input(
        FrameStub(), sim, "person", "family_type", 2025
    )

    assert route == "categorical_broadcast_benunit_to_person"
    assert values.tolist() == ["couple", "couple", "single"]


def test_compute_uk_measure_input_bool_any_collapse_person_to_group():
    sim = SimulationStub({"is_disabled": ("person", np.array([False, True, False]))})

    values, route = compute_uk_measure_input(
        FrameStub(), sim, "household", "is_disabled", 2025
    )

    assert route == "bool_any_collapse_person_to_household"
    assert values.tolist() == [1.0, 0.0]


def test_compute_uk_measure_input_numeric_map_to_entity():
    sim = SimulationStub(
        {
            "benefit": ("person", np.array([1.0, 2.0, 3.0])),
            ("benefit", "household"): ("household", np.array([3.0, 3.0])),
        }
    )

    values, route = compute_uk_measure_input(
        FrameStub(), sim, "household", "benefit", 2025
    )

    assert route == "map_to_household"
    assert values.tolist() == [3.0, 3.0]
    assert sim.calls[-1] == ("benefit", 2025, "household")


def test_compute_uk_measure_input_refuses_unknown_and_length_mismatch():
    with pytest.raises(KeyError, match="no variable"):
        compute_uk_measure_input(
            FrameStub(), SimulationStub({}), "person", "missing", 2025
        )

    sim = SimulationStub({"income_tax": ("person", np.array([1.0]))})
    with pytest.raises(ValueError, match="produced 1 values"):
        compute_uk_measure_input(FrameStub(), sim, "person", "income_tax", 2025)


def _write_exclusions(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _entry(**overrides) -> dict:
    entry = {
        "name": "a",
        "reason": "reviewed reason",
        "tracking": "microcosm#623",
        "approved_by": "juaristi22",
        "adjudication": "microcosm#757",
        "approved_on": "2026-08-25",
        "expires_on": "2026-11-25",
    }
    entry.update(overrides)
    return entry


def test_exclusion_loader_refusals(tmp_path: Path):
    base = {"schema_version": 2, "exclusions": [_entry()]}

    with pytest.raises(ValueError, match="schema_version"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "bad-version.json", {**base, "schema_version": 1}
            )
        )
    with pytest.raises(ValueError, match="unknown top-level"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(tmp_path / "unknown.json", {**base, "extra": True})
        )
    with pytest.raises(ValueError, match="empty reason"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "empty.json",
                {**base, "exclusions": [_entry(reason="")]},
            )
        )
    with pytest.raises(ValueError, match="empty approved_by"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "unapproved.json",
                {**base, "exclusions": [_entry(approved_by="")]},
            )
        )
    with pytest.raises(ValueError, match="canonical ISO"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "sloppy-date.json",
                {**base, "exclusions": [_entry(approved_on="2026-8-25")]},
            )
        )
    with pytest.raises(ValueError, match="after approved_on"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "inverted.json",
                {**base, "exclusions": [_entry(expires_on="2026-08-25")]},
            )
        )
    with pytest.raises(
        ValueError, match="unknown UK calibration measure exclusion field"
    ):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "extra-field.json",
                {**base, "exclusions": [_entry(sneaky="value")]},
            )
        )
    with pytest.raises(ValueError, match="duplicate"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "duplicate.json",
                {**base, "exclusions": [_entry(), _entry(reason="two")]},
            )
        )


def test_exclusion_applier_returns_pruned_registry_and_receipt():
    registry = TargetRegistry(
        [
            TargetSpec(
                name="drop", entity="person", measure="drop", value=1.0, source="test"
            ),
            TargetSpec(
                name="keep", entity="person", measure="keep", value=1.0, source="test"
            ),
        ],
        country="uk",
    )

    from datetime import date

    window = _entry(name="drop", reason="reviewed")
    pruned, receipt = apply_uk_calibration_measure_exclusions(
        registry, (window,), now=date(2026, 9, 1)
    )

    assert [spec.name for spec in pruned.specs] == ["keep"]
    assert receipt == {
        "drop": {
            "reason": "reviewed",
            "tracking": "microcosm#623",
            "approved_by": "juaristi22",
            "adjudication": "microcosm#757",
            "approved_on": "2026-08-25",
            "expires_on": "2026-11-25",
            "evaluated_on": "2026-09-01",
        }
    }
    with pytest.raises(ValueError, match="matched zero"):
        apply_uk_calibration_measure_exclusions(
            registry, (_entry(name="stale"),), now=date(2026, 9, 1)
        )
    # Outside the reviewed window the run refuses with correct-or-renew —
    # the narrowing neither lapses silently nor lives forever.
    with pytest.raises(ValueError, match="correct the underlying gap"):
        apply_uk_calibration_measure_exclusions(
            registry, (window,), now=date(2026, 11, 26)
        )
    with pytest.raises(ValueError, match="correct the underlying gap"):
        apply_uk_calibration_measure_exclusions(
            registry, (window,), now=date(2026, 8, 24)
        )


def test_exclusion_applier_warns_within_week_of_expiry():
    registry = TargetRegistry(
        [
            TargetSpec(
                name="drop", entity="person", measure="drop", value=1.0, source="test"
            ),
        ],
        country="uk",
    )

    from datetime import date

    window = _entry(name="drop", reason="reviewed")
    # expires_on is 2026-11-25: five days out warns, mid-window is silent.
    with pytest.warns(UserWarning, match="within one week"):
        apply_uk_calibration_measure_exclusions(
            registry, (window,), now=date(2026, 11, 20)
        )

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("error")
        apply_uk_calibration_measure_exclusions(
            registry, (window,), now=date(2026, 9, 1)
        )


#: The adjudicated register census: one entry per class of the #757
#: ``uk_target_fit`` dispositions (issue comment 5427936411) plus the
#: standing salary-sacrifice adjudication, as revised by the #807
#: excluded-cell support re-measurement (spine-i and spine-j receipts):
#: the three mid-band "sparse cell" entries whose measured support was in
#: the hundreds are lifted. The counts are the record of what was signed;
#: a drifting count is a register change that must be re-adjudicated,
#: never absorbed.
_PACKAGED_EXCLUSION_CENSUS = {
    # microcosm#1069 c11 held the three salary-sacrifice rate bands out until
    # a band test that reads the Scottish bands; microcosm#1095 binds them by
    # rate class, so no salary-sacrifice entry remains.
    "hmrc.salary_sacrifice.": 0,
    # The next policyengine-uk bump: every HMRC £1m-and-over cell has carriers
    # on #1333's 2.145.0 spine, so arm 1 bound the ten; bound, they distorted
    # their SPI-channel neighbours, so they stay held on that measurement
    # (María's rule of 2026-10-21), re-signed on 2026-10-23. Of the five SLC
    # rows, the postgraduate repayments row retired from the targets (the
    # stage assigns no postgraduate plan) and Plan 5's liable stock binds; the
    # Plan 5 repayments and the two Plan 2 stocks stay.
    "_1_000_000_to_inf": 10,
    "slc.": 3,
    # microcosm#1095 (2026-10-07): two bands whose few supporting awards
    # policyengine-uk 2.122.2 moves out join the uk-data#452 class.
    # microcosm#1095 U3 (2026-10-16): the sparse-band support floor drops
    # every band below ten source households, so the nine entries it covers
    # on the head spine retire (seven uk-data#452, the two microcosm#736
    # zero-support bands); twelve well-supported bands remain held out per
    # row, each on its own renewal. Arm 2 of the next policyengine-uk bump
    # (2026-10-24) binds four of them, which it fits within 10 percent
    # unbound since the award-band take-up of microcosm#1344.
    "dwp/uc_payment_dist/": 8,
    # The OBR welfare-cap split retired from the targets with the next bump:
    # obr.universal_credit binds the sum of the two cells.
    "obr.universal_credit_": 0,
    # microcosm#890 E1 (2026-09-11): the all-road-users obr.fuel_duties row
    # is signed out of the reference surface (target_reference_signed_
    # exclusions.json) and obr.fuel_duties_cars binds the cars receipts, so
    # the #757 measure exclusion for it retires here rather than lapsing.
    # The three ons.household_composition entries retired with microcosm#791.
    # microcosm#762 A16 (2026-09-03): the rows the spine cannot reach by
    # reweighting — savings interest, housing benefit, the two plan-2
    # borrower stocks and JSA claimants — windowed to one month. The two
    # ONS land rows first listed were withdrawn the same day (receipt R15):
    # their apparent 8.7x / 2.5x misses were an artefact of per-block engine
    # resolution, and single-block resolution puts them within 3 % and 9 %.
    "ons.savings_interest_income": 1,
    "obr.housing_benefit": 1,
    "dwp.jsa_claimants": 1,
    # microcosm#882 (2026-09-15): three UC element rows were held out of the
    # objective, signed by María in review of microcosm#921 with a one-month
    # window. The carer and childcare rows retired on 2026-09-16 with their
    # repairs; the any-tenure housing row stays held out as the signed
    # structural bias: the v20 measurement (-2.8 % unbound, at weights that
    # never solved toward the row) reproduces the 2.79 % other/unknown-tenure
    # gap the entry records, so it is a diagnostic, not a fit target
    # (docs/evidence/uk-a16-v20/). microcosm#1095: the five-or-more children
    # row, held out on María's ruling of microcosm#1271 (2026-10-14), left the
    # targets for a diagnostic on 2026-10-16, so its entry retired.
    "dwp.uc.households_": 1,
    # microcosm#882 repairs (2026-09-16): the three Housing Benefit caseload
    # rows and the thirteen benefit-cap amount bands outside the 25 percent
    # bound are measured on every evaluation but held out of the objective;
    # the A16 rows obr.housing_benefit and dwp.jsa_claimants were
    # re-adjudicated the same day on the mechanism receipts.
    "dwp.hb.": 3,
    # Two amount bands (200.01-300, 400.01-500) retired 2026-09-21 on the
    # v20 measurement (-10.1 % and -3.8 % unbound); the eleven tail bands
    # retired 2026-10-21 under María's rule that thin support alone leaves no
    # row out: the zero-support floor drops any band no household carries,
    # and the rest are bound and measured.
    "dwp.benefit_cap.capped_households_": 0,
    # microcosm#1113 and #1123 item 3 (2026-10-08): rows whose level the
    # engine, a published-source residual or a scope the frame doesn't hold
    # sets, held out and measured with tools/diagnose_uk_consumption_taxes.py.
    # obr.vat bound again with the next policyengine-uk bump (ruling D3), and
    # was held again as a diagnostic (María's ruling of 2026-10-24, Vahid's
    # review of #1383, item 10): the engine's coverage factor was derived on
    # a certified Microcosm release, so binding the row is circular. The ONS
    # electricity row is bound again since microcosm#1418 moved the stage's
    # level onto it (María's rulings of 2026-10-28). The gas row is held
    # again: ONS 04.5.2 counts the bottled gas the frame holds as liquid fuel
    # (María's ruling of 2026-10-29).
    "obr.vat": 1,
    "obr.fuel_duties_cars": 1,
    "ons.household_gas_expenditure": 1,
    # microcosm#1301 (2026-10-16): employer NICs, gross of the Employment
    # Allowance in the engine and priced on FY2025-26 rules over a window that
    # takes a quarter of FY2024-25, held out as a measured diagnostic.
    "obr.ni_employer": 1,
    # microcosm#1278 (2026-10-20): ORR's broad rail-support total, far above
    # the frame's household rail subsidy at design weights while the 7270
    # versus 7271 basis ruling is open (microcosm#890 G), held out after the
    # national hold-out arm.
    "orr.": 1,
    # microcosm#1359 (2026-10-23): OBR's FY2024-25 excise receipts, whose
    # level the stage sets from published clearances and bases at the
    # engine's rates, measured with tools/diagnose_uk_consumption_taxes.py.
    # María ruled on 2026-10-28 to bind the betting and gaming row, but the
    # national solve cannot measure the engine's 2024 duty, so it is held
    # again pending her decision (microcosm#1420).
    "obr.beer_and_cider_duties": 1,
    "obr.wine_duties": 1,
    "obr.spirits_duties": 1,
    "obr.tobacco_duties": 1,
    "obr.betting_and_gaming_duties": 1,
}

# The carer and childcare rows were retired on 2026-09-16 with the repairs
# (care hours and the childcare take-up draw on policyengine-uk 2.98.0);
# only the any-tenure housing row remains held out (structural bias).
_UC_ELEMENT_REGISTER_ROWS = ("dwp.uc.households_housing_element",)

_A16_UNREACHABLE_ROWS = (
    "slc.borrowers.plan_2_liable",
    "slc.borrowers.plan_2_above_threshold",
    "obr.housing_benefit",
    "dwp.jsa_claimants",
    "ons.savings_interest_income",
)
_A16_READJUDICATED_ROWS = ("obr.housing_benefit", "dwp.jsa_claimants")
# PolicyEngine/chronicle#280 lane (María's ruling of 2026-09-22, R3): the SPI 2023-24
# interest rows by band are the fitting anchor for interest, uprated to the
# calibration year; the ONS D.41 row stays on the register as a measured
# diagnostic with the concept reason and the UC-tranche clock.
_A16_CONCEPT_ROWS = ("ons.savings_interest_income",)


def test_packaged_exclusions_load():
    exclusions = load_uk_calibration_measure_exclusions()
    names = [entry["name"] for entry in exclusions]
    # 65 less the nine payment bands the support floor covers (microcosm#1095 U3),
    # plus the employer NICs diagnostic (microcosm#1301) and the rail-support
    # hold-out (microcosm#1278), less the thirteen rows bound again under
    # María's rule of 2026-10-21 (the eleven benefit-cap amount bands and the
    # HMRC £500k to £1m dividend and property-income-count rows), less the
    # seven the next policyengine-uk bump retires: the three band-H region
    # cells and Plan 5's liable stock bind, and the OBR welfare-cap split and
    # the postgraduate repayments leave the targets; and obr.vat binds again
    # with it (ruling D3). The ten HMRC £1m-and-over cells it bound on arm 1
    # are held again on that arm's measured distortion. Arm 2 binds four of
    # the twelve UC payment bands, which it fits unbound (microcosm#1344), and
    # obr.vat is held again as a diagnostic (Vahid's review of #1383, 10).
    # microcosm#1359 adds the five OBR excise rows, measured at FY2024-25.
    # The next policyengine-uk bump (2.162.3) holds obr.domestic_rates once it
    # compiles (María's ruling of 2026-10-25). microcosm#1418 binds the ONS
    # electricity row again (María's ruling of 2026-10-28), and holds the ONS
    # gas row again for bottled gas (María's ruling of 2026-10-29). The
    # betting and gaming row stays held: the solve cannot yet measure its
    # 2024 duty (microcosm#1420).
    assert len(names) == len(set(names)) == 39
    (gas,) = [
        entry
        for entry in exclusions
        if entry["name"] == "ons.household_gas_expenditure"
    ]
    assert (gas["approved_on"], gas["tracking"]) == ("2026-10-29", "microcosm#1095")
    assert "bottled gas" in gas["reason"] and "liquid_fuel_consumption" in gas["reason"]
    rehold = [entry for entry in exclusions if "_1_000_000_to_inf" in entry["name"]]
    assert {(e["approved_on"], e["expires_on"], e["tracking"]) for e in rehold} == {
        ("2026-10-23", "2027-01-23", "microcosm#1095")
    }
    for entry in rehold:
        assert entry["reason"].startswith("Measured distortion"), entry["name"]
        assert "arm 1" in entry["adjudication"], entry["name"]
        # The cells rest on the SPI Public Use Tape: no figure of theirs or
        # their neighbours' enters the register.
        assert "%" not in entry["reason"], entry["name"]
    assert not [name for name in names if name.startswith("dwp.benefit_cap.")]
    # The band-H region cells bind under the floor's band-H scope.
    assert not [name for name in names if name.startswith("mhclg.council_tax_stock.")]

    for marker, expected in _PACKAGED_EXCLUSION_CENSUS.items():
        matched = [name for name in names if marker in name]
        assert len(matched) == expected, (marker, matched)
    # The sparse HMRC band cells are whatever remains: hmrc/ band cells
    # that are not the 1m+ channel cells. After the #807 revision three
    # genuinely thin 500k-1m cells stayed excluded; the self-employment one
    # retired 2026-09-21 on the v20 measurement (+0.0 % unbound), and the
    # dividend and property-income-count ones on 2026-10-21, bound again
    # under María's rule that thin support alone leaves no row out.
    sparse = [
        name
        for name in names
        if name.startswith("hmrc/") and "_1_000_000_to_inf" not in name
    ]
    assert sparse == [], sparse

    # The 2026-08-26 tranche carries the uk_target_fit disposition
    # adjudication and a uniform three-month window. Its three ONS
    # household-composition cells were retired by microcosm#791 (the
    # relationship-to-head successor): the ten cells now bind on the
    # frs_relationships stage's household type column, so no composition
    # entry may remain on the register. 35 since microcosm#890 also retired
    # the obr.fuel_duties entry (2026-09-11): the all-road-users row is
    # signed out of the reference surface and the cars receipts bind instead.
    # Three 2026-08-26 rows retired 2026-09-21 on the v20 measurement, three
    # payment bands retired 2026-10-16 under the support floor, the two thin
    # HMRC 500k-1m cells retired 2026-10-21, bound again, and fourteen more
    # with the next bump: the ten HMRC £1m-and-over cells (held again on arm
    # 1's distortion, on their own entries) and Plan 5's liable stock were
    # bound, and the OBR welfare-cap split and the postgraduate repayments
    # left the targets. The twelve payment bands and Plan 5's repayments
    # remained, renewed on 2026-10-23 on arm 1's measurement, and arm 2
    # bound four of the bands (2026-10-24), so eight bands and Plan 5's
    # repayments remain.
    assert [e for e in exclusions if e["approved_on"] == "2026-08-26"] == []
    tranche = [e for e in exclusions if "5427936411" in e["adjudication"]]
    assert len(tranche) == 9
    for entry in tranche:
        assert (entry["approved_on"], entry["expires_on"]) == (
            "2026-10-23",
            "2027-01-23",
        ), entry["name"]
        assert "renewed on 2026-10-23 to 2027-01-23" in entry["adjudication"]
    assert not [n for n in names if n.startswith("ons.household_composition.")]

    # The 2026-09-03 tranche was #762's A16: five unreachable national rows,
    # a one-month window, each row tracked on its spine-defect issue. Two of
    # them (housing benefit, JSA) were re-adjudicated on 2026-09-16 with the
    # mechanism receipts (#882) and moved to the 2026-12-08 clock; the
    # savings-interest row moved to the 2026-09-22 concept clock; the two
    # plan-2 borrower rows were renewed one month on 2026-10-02 (microcosm#1063
    # stack) and again on 2026-10-20, to the November PR's renewals (María's
    # ruling on the pre-Budget PR, microcosm#1278), so nothing remains on the
    # 2026-09-03 clock. The plan-2 rows tracked microcosm#1278 after
    # microcosm#868 closed (María's ruling of 2026-10-17), and microcosm#1095
    # since #1278 closed into it (María's ruling of 2026-10-28).
    a16_issues = {
        "ons.savings_interest_income": "microcosm#866",
        "obr.housing_benefit": "microcosm#867",
        "slc.borrowers.plan_2_liable": "microcosm#1095",
        "slc.borrowers.plan_2_above_threshold": "microcosm#1095",
        "dwp.jsa_claimants": "microcosm#869",
    }
    assert [e for e in exclusions if e["approved_on"] == "2026-09-03"] == []
    # The two thin UC payment cells that joined the uk-data#452 class on the
    # #1063 stack arm retired 2026-10-16 (the support floor drops them), and
    # the plan-2 rows left the 2026-10-02 clock on their 2026-10-20 renewal.
    assert [e for e in exclusions if e["approved_on"] == "2026-10-02"] == []
    # The plan-2 rows were renewed again on 2026-10-23, on arm 1 of the next
    # bump; the 2026-10-20 entry left is the rail-support hold-out (below).
    assert sorted(
        e["name"] for e in exclusions if e["approved_on"] == "2026-10-20"
    ) == ["orr.rail_government_support"]
    renewed = [
        e
        for e in exclusions
        if e["approved_on"] == "2026-10-23" and e["name"] in _A16_UNREACHABLE_ROWS
    ]
    assert sorted(e["name"] for e in renewed) == sorted(_A16_UNREACHABLE_ROWS[:2])
    # María's ruling of 2026-10-20 (microcosm#1278): the ORR row is held out
    # only on a national arm whose gates pass and whose share of rows within
    # 10 percent holds; the 7270 versus 7271 basis ruling stays open.
    (rail,) = [e for e in exclusions if e["name"] == "orr.rail_government_support"]
    assert rail["approved_by"] == "juaristi22"
    assert (rail["approved_on"], rail["expires_on"]) == ("2026-10-20", "2026-12-11")
    assert rail["tracking"] == "microcosm#1095"
    assert "7270" in rail["reason"] and "7271" in rail["reason"]
    assert "microcosm#890 G" in rail["reason"]
    assert "never fitted" in rail["reason"]
    assert "within 10 percent" in rail["adjudication"]
    # Bound on table 7271, the plan's default of 2026-10-22, the row failed
    # the weight-ratio gate on arm 2c of the next policyengine-uk bump, so
    # the hold stands and the basis ruling stays open.
    assert "uk_weight_ratio failed" in rail["reason"]
    assert "arm 2c" in rail["adjudication"]
    # So did the two bands the engine bump of microcosm#1095 (policyengine-uk
    # 2.122.2) emptied, the 2026-10-07 entries.
    assert [e for e in exclusions if e["approved_on"] == "2026-10-07"] == []
    # The 2026-10-08 consumption-tax tranche (microcosm#1113, #1123 item 3):
    # VAT's level is an engine factor (policyengine-uk#1996) and the two ONS
    # energy rows are a published-source residual against the stage's DESNZ
    # volume at QEP prices. The OBR cars receipts include business-paid car
    # fuel, so household road fuel is bound on ONS 07.2.2 in their place. A
    # re-solve with the four masked together left the other rows fitting no
    # worse, so they are held out of the objective and measured by the
    # diagnose tool. The PR that added them closes microcosm#1113, so the
    # three that tracked it tracked its follow-up issue, microcosm#1207, then
    # microcosm#1278 when that closed into it, and microcosm#1095 since #1278
    # closed into that (María's ruling of 2026-10-28). obr.vat tracked
    # policyengine-uk#1996 until policyengine-uk#2187 (released in 2.135.7)
    # closed it, bound again with the next policyengine-uk bump, and is held
    # again on its own entry: the factor #2187 uses was derived on a certified
    # Microcosm release (María's ruling of 2026-10-24).
    consumption_taxes = [e for e in exclusions if e["approved_on"] == "2026-10-08"]
    # The ONS electricity and gas rows left this tranche with microcosm#1418,
    # which moved the stage's electricity level onto ONS; the gas row is held
    # again on its own entry for bottled gas (María's ruling of 2026-10-29).
    assert sorted(e["name"] for e in consumption_taxes) == ["obr.fuel_duties_cars"]
    for entry in consumption_taxes:
        assert entry["expires_on"] == "2027-01-08", entry["name"]
        assert "tools/diagnose_uk_consumption_taxes.py" in entry["reason"], entry[
            "name"
        ]
        assert (
            "experiments/1113-consumption-tax-levels-receipts.md"
            in entry["adjudication"]
        ), entry["name"]
    trackings = {e["name"]: e["tracking"] for e in consumption_taxes}
    assert trackings == {"obr.fuel_duties_cars": "microcosm#1095"}
    (vat,) = [e for e in exclusions if e["name"] == "obr.vat"]
    assert (vat["approved_on"], vat["expires_on"]) == ("2026-10-24", "2027-01-24")
    assert vat["tracking"] == "microcosm#1095"
    assert "circularity" in vat["reason"]
    assert "policyengine-uk#2319" in vat["reason"]
    assert "tools/diagnose_uk_consumption_taxes.py" in vat["reason"]
    # microcosm#1095: the five-or-more children row held out on 2026-10-14
    # (María's ruling on microcosm#1271) is no target since 2026-10-16, but a
    # diagnostic beside the two-child-limit rows, so no exclusion names it and
    # only the housing-element renewal carries that date. The 2026-10-16
    # entries are the employer NICs diagnostic and the savings-interest renewal.
    assert sorted(
        e["name"] for e in exclusions if e["approved_on"] == "2026-10-16"
    ) == ["obr.ni_employer", "ons.savings_interest_income"]
    # The housing-element row's 2026-10-14 renewal was renewed again on
    # 2026-10-23, on arm 1 of the next bump.
    assert [e for e in exclusions if e["approved_on"] == "2026-10-14"] == []
    assert "dwp.uc.households_children_5_or_more" not in names
    # María's ruling of 2026-10-20 (microcosm#1278): renewed rather than made
    # diagnostics, and restated on the academic year 2024/25 stock the rows
    # bind since microcosm#868, not the 2025/26 one the reasons quoted.
    plan_2_stocks = {
        "slc.borrowers.plan_2_liable": "8.94M",
        "slc.borrowers.plan_2_above_threshold": "3.985M",
    }
    for entry in renewed:
        assert entry["approved_by"] == "juaristi22", entry["name"]
        assert entry["expires_on"] == "2027-01-23", entry["name"]
        assert "renewed on 2026-10-23 to 2027-01-23" in entry["adjudication"]
        assert entry["tracking"] == a16_issues[entry["name"]], entry["name"]
        assert "A16" in entry["adjudication"], entry["name"]
        assert "renewed one month on 2026-10-02" in entry["adjudication"], entry["name"]
        assert "microcosm#1063" in entry["adjudication"], entry["name"]
        assert "renewed on 2026-10-20 to 2026-11-26" in entry["adjudication"], entry[
            "name"
        ]
        assert "AY2024/25" in entry["reason"], entry["name"]
        assert "Table 6a" in entry["reason"], entry["name"]
        assert plan_2_stocks[entry["name"]] in entry["reason"], entry["name"]
        assert "frame-side gap" in entry["reason"], entry["name"]
    # The two re-adjudicated rows left the 2026-12-08 clock on 2026-10-24,
    # renewed on arm 2 of the next policyengine-uk bump pending the Housing
    # Benefit and JSA PR.
    for name in _A16_READJUDICATED_ROWS:
        entry = next(e for e in exclusions if e["name"] == name)
        assert entry["approved_on"] == "2026-10-24", name
        assert entry["expires_on"] == "2027-01-24", name
        assert entry["tracking"] == a16_issues[name], name
        assert "Renewed on 2026-10-24 to 2027-01-24" in entry["adjudication"], name
        assert "arm 2" in entry["reason"], name
    for name in _A16_CONCEPT_ROWS:
        entry = next(e for e in exclusions if e["name"] == name)
        # Ruling R3 of 2026-09-22, renewed on 2026-10-16 to 2027-01-08.
        assert entry["approved_on"] == "2026-10-16", name
        assert entry["expires_on"] == "2027-01-08", name
        assert entry["tracking"] == a16_issues[name], name
        assert "microcosm#1006" in entry["adjudication"], name
        assert "renewed on 2026-10-16" in entry["adjudication"], name
        assert "D.41" in entry["reason"] and "diagnostic" in entry["reason"], name
        # The fitting anchor the row hands over to: the SPI Table 3.7 interest
        # rows, uprated (PolicyEngine/chronicle#280 lane).
        assert "hmrc.spi.savings_interest_income" in entry["reason"], name
        assert "never fitted" in entry["reason"], name

    # The 2026-09-16 tranche also carried the Housing Benefit caseload rows
    # (tracked on #867 like the spend row) and the benefit-cap amount bands
    # outside the bound (tracked on #882); every entry names its tool. The
    # bands retired 2026-10-21, bound again (María's rule of that day), so
    # the caseload rows remain, renewed on 2026-10-24 on arm 2 of the next
    # policyengine-uk bump with the spend and JSA rows.
    assert [e for e in exclusions if e["approved_on"] == "2026-09-16"] == []
    repairs = [
        e
        for e in exclusions
        if e["approved_on"] == "2026-10-24"
        and e["name"] not in _A16_READJUDICATED_ROWS
        and e["name"] != "obr.vat"
    ]
    assert len(repairs) == 3
    for entry in repairs:
        assert entry["expires_on"] == "2027-01-24", entry["name"]
        # Every entry of the tranche points at the ruling that exists (the
        # 2026-09-16 status comment) and explains the shared expiry.
        assert "issuecomment-5694598278" in entry["adjudication"], entry["name"]
        assert "zero-band clock" in entry["adjudication"], entry["name"]
        if entry["name"].startswith("dwp.hb."):
            assert entry["tracking"] == "microcosm#867", entry["name"]
            assert "tools/diagnose_uk_legacy_benefits.py" in entry["adjudication"]
        else:
            assert entry["name"].startswith("dwp.benefit_cap.capped_households_")
            assert entry["tracking"] == "microcosm#882", entry["name"]
            assert "tools/diagnose_uk_benefit_cap.py" in entry["adjudication"]
            assert "tools/diagnose_uk_benefit_cap.py" in entry["reason"], entry["name"]
    assert "dwp.benefit_cap.capped_households_up_to_100" not in names
    assert "dwp.benefit_cap.capped_households" not in names

    # The 2026-09-15 tranche was #882's element rows. Carer and childcare
    # retired on 2026-09-16 with their repairs; the any-tenure housing row (a
    # structural bias: DWP's 'Yes' includes an other/unknown tenure the model
    # cannot carry) remains, adjudicated to microcosm#882 and the committed
    # baseline doc, signed in review of microcosm#921, windowed to one month,
    # and renewed one month on 2026-10-14 (microcosm#1124 local release), so
    # nothing remains on the 2026-09-15 clock.
    assert [e for e in exclusions if e["approved_on"] == "2026-09-15"] == []
    elements = [
        e
        for e in exclusions
        if e["approved_on"] == "2026-10-23" and e["tracking"] == "microcosm#882"
    ]
    assert sorted(e["name"] for e in elements) == sorted(_UC_ELEMENT_REGISTER_ROWS)
    for entry in elements:
        assert entry["expires_on"] == "2027-01-23", entry["name"]
        assert "renewed one month on 2026-10-14" in entry["adjudication"]
        assert "renewed on 2026-10-23 to 2027-01-23" in entry["adjudication"]
        assert entry["tracking"] == "microcosm#882", entry["name"]
        assert entry["approved_by"] == "juaristi22", entry["name"]
        assert "microcosm#882" in entry["adjudication"], entry["name"]
        assert "microcosm#921" in entry["adjudication"], entry["name"]
        assert "docs/uk-uc-baseline-2026-09-10.md" in entry["adjudication"], entry[
            "name"
        ]
    housing = next(
        e for e in elements if e["name"] == "dwp.uc.households_housing_element"
    )
    assert "112,518 of 4,037,650" in housing["reason"]
    # The other UC element rows ride in the objective and are not on the
    # register.
    for riding in (
        "dwp.uc.households_carer_element",
        "dwp.uc.households_childcare_element",
        "dwp.uc.households_lcwra_element",
        "dwp.uc.households_housing_element_social_rented",
        "dwp.uc.households_housing_element_private_rented",
        "dwp.uc.households_with_deduction",
        "obr.universal_credit",
    ):
        assert riding not in names, riding

    # The lever targets are deliberately NOT excluded: the six UC
    # caseload / two-child-limit cells ride the would_claim_uc lever run.
    # Exclusions cannot move surviving cells because band edges are pinned
    # to the compiled register (#792); the non-excluded cells are expected
    # to pass at published band widths on the next seam run. The three
    # cells the #807 revision lifted (measured support in the hundreds)
    # are back on the calibrated surface with them.
    excluded = set(names)
    for riding in (
        "dwp.uc.households",
        "dwp.uc.households_single_no_children",
        "dwp.uc.two_child_limit.children_disabled_child_element",
        "ons.household_composition.couple_no_children_households",
        "ons.household_composition.multi_family_households",
        "ons.household_composition.unrelated_adult_households",
        "ons.household_composition.lone_parent_non_dependent_children_households",
        "hmrc/state_pension_income_band_40_000_to_50_000",
        "hmrc/state_pension_income_band_50_000_to_70_000",
        "hmrc/self_employment_income_income_band_50_000_to_70_000",
        "hmrc/private_pension_income_count_income_band_100_000_to_150_000",
    ):
        assert riding not in excluded, riding


def test_employer_nics_are_a_measured_diagnostic():
    # microcosm#1301 (María's ruling of 2026-10-16): the engine's ni_employer
    # is gross of the Employment Allowance, which OBR's receipts are net of,
    # and the row's calendar-2025 window takes a quarter of FY2024-25 that the
    # engine prices on FY2025-26 rules. Modelling the allowance alone would
    # close only part of the gap, so the row leaves the objective as a
    # diagnostic until a locked engine nets the allowance out. The other two
    # class rows stay bound beside it.
    from datetime import date

    exclusions = {e["name"]: e for e in load_uk_calibration_measure_exclusions()}
    entry = exclusions["obr.ni_employer"]
    assert entry["tracking"] == "policyengine-uk#2459"
    assert entry["approved_by"] == "juaristi22"
    assert entry["approved_on"] == "2026-10-16"
    # Three months, the window the 2026-10-08 engine-factor diagnostic had.
    assert entry["expires_on"] == "2027-01-16"
    assert (
        date.fromisoformat(entry["expires_on"])
        - date.fromisoformat(entry["approved_on"])
    ).days == (date(2027, 1, 8) - date(2026, 10, 8)).days
    for phrase in (
        "Employment Allowance",
        "calendar-2025 window",
        "FY2025-26 rules",
        "diagnostic",
        "never fitted",
    ):
        assert phrase in entry["reason"], phrase
    assert "microcosm#1301" in entry["adjudication"]
    assert "policyengine-uk#2459" in entry["adjudication"]
    assert "obr.ni_employee" not in exclusions
    assert "obr.ni_self_employed" not in exclusions


def test_measure_resolver_direct_and_scratch_receipts(monkeypatch, tmp_path: Path):
    created = []

    class FakeMicrosimulation:
        def __init__(self, *, dataset):
            created.append(dataset)
            self.tax_benefit_system = SimpleNamespace(variables={})

    fake_policyengine_uk = SimpleNamespace(
        __version__="9.9.9", Microsimulation=FakeMicrosimulation
    )
    monkeypatch.setitem(sys.modules, "policyengine_uk", fake_policyengine_uk)
    writes = []
    monkeypatch.setattr(
        measure_simulation,
        "write_uk_national_frame",
        lambda frame, path: writes.append((frame, path)) or path,
    )

    frame = ReleaseIdentifierFrameStub()
    direct = UKMeasureResolver(
        simulation_source=tmp_path / "input.h5",
        scratch_dir=tmp_path,
        year=2025,
        frame=frame,
    )
    scratch = UKMeasureResolver(
        simulation_source=None,
        scratch_dir=tmp_path,
        year=2025,
        frame=frame,
    )

    assert direct.receipt()["mode"] == "direct_h5"
    assert direct.receipt()["policyengine_uk_version"] == "9.9.9"
    # An engine without bus pricing parameters records why it has no parity.
    assert direct.receipt()["bus_engine_parity"]["available"] is False
    assert scratch.receipt()["mode"] == "scratch_frame_export"
    # A frame carrying no export-dropped column, already on the release's
    # identifiers, is written as it is.
    assert "engine_scratch_dropped_columns" not in scratch.receipt()
    assert writes == [(frame, tmp_path / "simulation-input.h5")]
    assert created == [
        str(tmp_path / "input.h5"),
        str(tmp_path / "simulation-input.h5"),
    ]


def test_scratch_engine_input_leaves_property_wealth_to_the_engine(
    monkeypatch, tmp_path: Path
):
    """microcosm#1106: a persisted ``property_wealth`` overrides the engine's
    sum of the three property components and is never uprated, so the scratch
    H5 the calibration engine loads drops it, as the release export does. The
    resolver's own frame keeps it, and the receipt names what was left out."""

    from microcosm.build.uk_runtime.national_frame import uk_national_frame

    class FakeMicrosimulation:
        def __init__(self, *, dataset):
            self.tax_benefit_system = SimpleNamespace(variables={})

    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(__version__="9.9.9", Microsimulation=FakeMicrosimulation),
    )
    writes = []
    monkeypatch.setattr(
        measure_simulation,
        "write_uk_national_frame",
        lambda frame, path: writes.append(frame) or path,
    )
    components = (
        "main_residence_value",
        "other_residential_property_value",
        "non_residential_property_value",
    )
    frame = uk_national_frame(
        person=pd.DataFrame(
            {
                "person_id": [1, 2],
                "person_benunit_id": [1, 2],
                "person_household_id": [1, 2],
                "age": [40, 50],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [1, 2]}),
        household=pd.DataFrame(
            {
                "household_id": [1, 2],
                "main_residence_value": [300_000.0, 0.0],
                "other_residential_property_value": [150_000.0, 0.0],
                "non_residential_property_value": [0.0, 20_000.0],
                # The WAS total also counts owned land and the remainder.
                "property_wealth": [480_000.0, 20_000.0],
            }
        ),
        time_period="2024",
        household_weights=np.asarray([1.0, 1.0]),
    )

    resolver = UKMeasureResolver(
        simulation_source=None,
        scratch_dir=tmp_path,
        year=2025,
        frame=frame,
    )

    (written,) = writes
    household = written.table("household")
    assert "property_wealth" not in household.columns
    for component in components:
        assert household[component].tolist() == (
            frame.table("household")[component].tolist()
        )
    assert written.weights_for("household").values.tolist() == [1.0, 1.0]
    assert resolver.frame is frame
    assert "property_wealth" in resolver.frame.table("household").columns
    assert resolver.receipt()["engine_scratch_dropped_columns"] == ["property_wealth"]


def _resolver_over(sim, monkeypatch, tmp_path: Path) -> UKMeasureResolver:
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(__version__="9.9.9", Microsimulation=lambda *, dataset: sim),
    )
    return UKMeasureResolver(
        simulation_source=tmp_path / "input.h5",
        scratch_dir=tmp_path,
        year=2025,
        frame=FrameStub(),
    )


def test_knows_answers_only_for_routes_compute_can_take(monkeypatch, tmp_path: Path):
    sim = SimulationStub(
        {
            "income_tax": ("person", np.array([1.0, 2.0, 3.0])),
            "is_disabled": ("person", np.array([False, True, False])),
            "family_type": ("benunit", np.array(["couple", "single"])),
            "benunit_income": ("benunit", np.array([1.0, 2.0])),
        }
    )
    resolver = _resolver_over(sim, monkeypatch, tmp_path)

    assert resolver.knows("person", "income_tax")
    assert resolver.knows("household", "income_tax")
    assert resolver.knows("household", "is_disabled")
    assert resolver.knows("person", "family_type")
    # Numeric group-to-group rides the map_to route…
    assert resolver.knows("household", "benunit_income")
    # …but a categorical one has no benunit-to-household mapping at all, and
    # the fence must say so rather than let compute raise mid-resolution.
    assert not resolver.knows("household", "family_type")
    assert not resolver.knows("person", "no_such_variable")
    assert not resolver.knows("firm", "income_tax")


def test_unknown_categorical_mapping_refuses_through_the_fence(
    monkeypatch, tmp_path: Path
):
    sim = SimulationStub({"family_type": ("benunit", np.array(["couple", "single"]))})
    resolver = _resolver_over(sim, monkeypatch, tmp_path)

    assert not resolver.knows("household", "family_type")
    with pytest.raises(KeyError, match="no categorical mapping"):
        resolver.compute("household", "family_type")


def test_exclusion_loader_requires_tracking(tmp_path: Path):
    # An exclusion narrows the calibrated target surface; the register has to
    # say where each narrowing is being resolved, not only why.
    with pytest.raises(ValueError, match="empty tracking"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "untracked.json",
                {"schema_version": 2, "exclusions": [_entry(tracking="")]},
            )
        )


def test_packaged_exclusion_names_resolve_to_committed_references():
    """Every register name must be a committed reference name, not a metric.

    The applier matches ``spec.name``, which is the reference name
    (``dwp.uc.households_carer_element``), so an entry keyed by the binding's
    metric name (``dwp/uc/elements/carer``) matches nothing and the
    calibration run refuses with "matched zero registry specs". The three
    #882 element rows shipped that way in microcosm#921; this pins the
    identity so a register edit cannot drift off the reference surface again.
    """

    from microcosm.build.country_spec import load_country_spec

    reference_names = {
        reference.name for reference in load_country_spec("uk").target_references
    }
    exclusions = load_uk_calibration_measure_exclusions()
    unresolved = sorted(
        entry["name"] for entry in exclusions if entry["name"] not in reference_names
    )
    assert unresolved == [], unresolved


def test_cgt_taxable_income_nets_the_allowances_the_engine_formula_nets():
    """Hand-computed rows: the band extension and the floors (microcosm#1014)."""
    parts = {
        "net_income": [0.0, 30_000.0, 60_000.0, 20_000.0, 80_000.0, 5_000.0],
        "allowances": [12_570.0, 12_570.0, 12_570.0, 32_570.0, 27_570.0, 12_570.0],
        "personal_pension_contributions": [0.0, 0.0, 0.0, 30_000.0, 5_000.0, 0.0],
        "pension_contributions_relief": [0.0, 0.0, 0.0, 20_000.0, 15_000.0, 0.0],
    }
    np.testing.assert_allclose(
        measure_simulation.cgt_taxable_income_from_components(parts),
        # 0 floor; 30,000 - 12,570; a Gift Aid donor's income is net income
        # less the allowances, the gift extending the band instead (2.128.0,
        # pe-uk#2271); relief up to the personal contributions leaves the
        # allowances; an allowance above income floors at 0.
        [0.0, 17_430.0, 47_430.0, 7_430.0, 57_430.0, 0.0],
    )
    assert "gift_aid" not in measure_simulation._CGT_TAXABLE_INCOME_COMPONENTS


# microcosm#1069 c11: the live counterfactual route (salary sacrifice returned to pay).
class _TaxSimulation:
    """A stub engine whose tax and NI follow the pay and salary-sacrifice inputs."""

    created = 0

    def __init__(self, *, dataset):
        type(self).created += 1
        self.inputs = {
            "employment_income": np.array([30_000.0, 50_000.0, 0.0]),
            "pension_contributions_via_salary_sacrifice": np.array([3_000.0, 0.0, 0.0]),
        }
        names = (*self.inputs, "income_tax", "ni_employee")
        self.tax_benefit_system = SimpleNamespace(
            variables={
                name: SimpleNamespace(
                    entity=SimpleNamespace(key="person"), value_type=float
                )
                for name in names
            }
        )

    def set_input(self, variable, year, values):
        self.inputs[variable] = np.asarray(values, dtype=float)

    def calculate(self, variable, year, map_to=None):
        if variable in self.inputs:
            return self.inputs[variable]
        taxable = np.maximum(self.inputs["employment_income"] - 12_570.0, 0.0)
        return {"income_tax": 0.2 * taxable, "ni_employee": 0.08 * taxable}[variable]


def _relief_binding(output, **extra):
    return {
        "metric_name": f"hmrc/salary_sacrifice_{output}_relief",
        "kind": "input_substitution_counterfactual",
        "zeroed_input": "pension_contributions_via_salary_sacrifice",
        "folded_into": "employment_income",
        "output_delta": "counterfactual_minus_baseline",
        "output_variable": output,
        "from_entity": "person",
        **extra,
    }


def test_counterfactual_delta_returns_the_sacrificed_pay_relief(monkeypatch, tmp_path):
    _TaxSimulation.created = 0
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(__version__="9.9.9", Microsimulation=_TaxSimulation),
    )
    resolver = UKMeasureResolver(
        simulation_source=tmp_path / "input.h5",
        scratch_dir=tmp_path,
        year=2025,
        frame=FrameStub(),
    )

    tax, route = resolver.counterfactual_delta(_relief_binding("income_tax"), 2025)
    ni, _ = resolver.counterfactual_delta(_relief_binding("ni_employee"), 2025)

    # £3,000 of sacrificed pay returned to the first person: 20% income tax and
    # 8% NI on it; nobody else moves.
    np.testing.assert_allclose(tax, [600.0, 0.0, 0.0])
    np.testing.assert_allclose(ni, [240.0, 0.0, 0.0])
    assert "folded into employment_income" in route
    # One baseline and one counterfactual simulation serve both bindings.
    assert _TaxSimulation.created == 2
    measures = resolver.receipt()["counterfactual_measures"]
    assert measures["hmrc/salary_sacrifice_income_tax_relief"]["rows_nonzero"] == 1


def test_counterfactual_delta_refuses_bands_and_other_periods(monkeypatch, tmp_path):
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(__version__="9.9.9", Microsimulation=_TaxSimulation),
    )
    resolver = UKMeasureResolver(
        simulation_source=tmp_path / "input.h5",
        scratch_dir=tmp_path,
        year=2025,
        frame=FrameStub(),
    )
    with pytest.raises(ValueError, match="banded counterfactual measures are refused"):
        resolver.counterfactual_delta(
            _relief_binding("income_tax", band={"variable": "adjusted_net_income"}),
            2025,
        )
    # A rate-class allocation must be exactly the declared one (microcosm#1095).
    allocation = {**UK_RATE_CLASS_ALLOCATION, "rate_class": "basic"}
    for drifted in (
        {**allocation, "basic_below_rate": 0.25},
        {**allocation, "rate_class": "starter"},
        {**allocation, "taxable_income": "adjusted_net_income"},
        # The bands a Gift Aid donor's gift extends are part of the allocation.
        {k: v for k, v in allocation.items() if k != "gift_aid_extends_bands"},
    ):
        with pytest.raises(ValueError, match="rate-class allocation must declare"):
            resolver.counterfactual_delta(
                _relief_binding("earned_taxable_income", allocation=drifted),
                2025,
            )
    with pytest.raises(ValueError, match="rate-class allocation must declare"):
        resolver.counterfactual_delta(
            _relief_binding("income_tax", allocation=allocation), 2025
        )
    with pytest.raises(ValueError, match="does not match target period"):
        resolver.counterfactual_delta(_relief_binding("income_tax"), 2024)


def _toy_uk_frame_with_native_aliases():
    """A validated UK national frame as the atomic geography path leaves it:
    the nation-native alias codes are NA outside their own nation."""
    from microcosm.build.uk_runtime import uk_national_frame
    from microcosm.frame import MassChangeRecord, WeightKind

    household = pd.DataFrame(
        {
            "household_id": np.asarray([100, 200, 300], dtype=np.int64),
            "household_weight": [1.0, 1.0, 1.0],
            "region": pd.array(
                ["LONDON", "SCOTLAND", "NORTHERN_IRELAND"], dtype="string"
            ),
            "oa_code": pd.array(
                ["E00000001", "S00090001", "N00000101"], dtype="string"
            ),
            "output_area_code": pd.array(
                ["E00000001", "S00090001", None], dtype="string"
            ),
            "data_zone_code": pd.array(
                [None, "S01006506", "N00000101"], dtype="string"
            ),
            "intermediate_zone_code": pd.array(
                [None, "S02001236", None], dtype="string"
            ),
            "super_data_zone_code": pd.array([None, None, "N21000001"], dtype="string"),
            "district_electoral_area_code": pd.array(
                [None, None, "N10000101"], dtype="string"
            ),
        }
    )
    person = pd.DataFrame(
        {
            "person_id": np.asarray([1, 2, 3], dtype=np.int64),
            "person_benunit_id": np.asarray([10, 20, 30], dtype=np.int64),
            "person_household_id": np.asarray([100, 200, 300], dtype=np.int64),
            "age": [40, 41, 42],
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.asarray([10, 20, 30], dtype=np.int64)})
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2025",
        weight_kind=WeightKind.IMPORTANCE,
        mass_log=(MassChangeRecord("household", 3.0, 3.0, 1.0, "toy"),),
    )


def test_scratch_engine_input_leaves_the_native_alias_codes_behind(
    monkeypatch, tmp_path: Path
):
    """microcosm#931 derives nation-native alias codes that are NA outside
    their own nation; policyengine-uk refuses a dataset with any NaN column,
    so the scratch export the engine loads drops them like the release export
    does. The first K=25 dense build from main (2026-10-04) failed on exactly
    this at ``uk.full.measures``."""
    loaded = []

    class FakeMicrosimulation:
        def __init__(self, *, dataset):
            loaded.append(dataset)
            self.tax_benefit_system = SimpleNamespace(variables={})

    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(__version__="9.9.9", Microsimulation=FakeMicrosimulation),
    )
    frame = _toy_uk_frame_with_native_aliases()
    before = frame.table("household").copy(deep=True)
    resolver = UKMeasureResolver(
        simulation_source=None, scratch_dir=tmp_path, year=2025, frame=frame
    )
    written = pd.read_hdf(tmp_path / "simulation-input.h5", "household")
    aliases = [
        "output_area_code",
        "data_zone_code",
        "intermediate_zone_code",
        "super_data_zone_code",
        "district_electoral_area_code",
    ]
    assert not set(aliases) & set(written.columns)
    assert not written.isna().any().any()
    assert written["oa_code"].tolist() == ["E00000001", "S00090001", "N00000101"]
    # The engine file carries the release's identifiers (microcosm#1194); the
    # resolver's own frame keeps the build's.
    assert written["household_id"].tolist() == [1, 2, 3]
    # The resolver's own frame is untouched; only the engine's copy changed.
    pd.testing.assert_frame_equal(frame.table("household"), before)
    assert resolver.frame is frame
    assert loaded == [str(tmp_path / "simulation-input.h5")]
    receipt = resolver.receipt()
    assert receipt["mode"] == "scratch_frame_export"
    assert receipt["engine_scratch_dropped_columns"] == aliases


def test_scratch_engine_input_without_aliases_is_the_frame_itself(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(
            __version__="9.9.9",
            Microsimulation=lambda *, dataset: SimpleNamespace(
                tax_benefit_system=SimpleNamespace(variables={})
            ),
        ),
    )
    writes = []
    monkeypatch.setattr(
        measure_simulation,
        "write_uk_national_frame",
        lambda frame, path: writes.append(frame) or path,
    )
    frame = ReleaseIdentifierFrameStub()
    resolver = UKMeasureResolver(
        simulation_source=None, scratch_dir=tmp_path, year=2025, frame=frame
    )
    assert writes == [frame]
    assert "engine_scratch_dropped_columns" not in resolver.receipt()


def _tiny_national_frame(weights):
    from microcosm.build.uk_runtime.national_frame import uk_national_frame
    from microcosm.frame import WeightKind

    person = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_benunit_id": [10, 10, 20, 20],
            "person_household_id": [100, 100, 200, 200],
            "age": [35, 8, 40, 38],
            "is_benunit_head": [True, False, True, False],
            "is_parent": [True, False, False, False],
            "is_uc_claimant": [True, False, True, True],
        }
    )
    benunit = pd.DataFrame(
        {
            "benunit_id": [10, 20],
            "benunit_source_id": [10, 20],
            "dependent_children": [1, 0],
            "is_married": [False, False],
        }
    )
    household = pd.DataFrame(
        {
            "household_id": [100, 200],
            "household_source_id": [100, 200],
            "region": ["WALES", "NORTHERN_IRELAND"],
            "council_tax": [1200.0, 0.0],
            "rent": [6000.0, 6000.0],
            "tenure_type": ["RENT_PRIVATELY", "RENT_PRIVATELY"],
        }
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period=2024,
        household_weights=np.asarray(weights, dtype=float),
        weight_kind=WeightKind.IMPORTANCE,
    )


def _tiny_release_identifier_frame(weights):
    from microcosm.build.uk_runtime.national_frame import (
        uk_national_frame,
        uk_release_export_frame,
    )

    released = uk_release_export_frame(_tiny_national_frame(weights))
    # The engine file carries no build-identifier column: a frame whose own
    # identifiers are already the release's.
    tables = {
        entity: released.table(entity).drop(columns=[f"{entity}_build_id"])
        for entity in ("person", "benunit", "household")
    }
    return uk_national_frame(
        **tables,
        time_period="2024",
        weight_kind=released.weights_for("household").kind,
        household_weights=released.weights_for("household").values,
        mass_log=released.mass_log,
    )


def test_engine_scratch_frame_carries_the_release_identifiers():
    """microcosm#1194: policyengine-uk keys a person's birth month on
    ``person_id``, so the solve's engine file carries the identifiers the
    release will: one plus the rank of the build identifier."""
    frame = _tiny_national_frame([250.0, 125.0])

    engine_frame, dropped = measure_simulation._engine_scratch_frame(frame)

    assert dropped == ()
    assert engine_frame.table("person")["person_id"].tolist() == [1, 2, 3, 4]
    assert engine_frame.table("person")["person_benunit_id"].tolist() == [1, 1, 2, 2]
    assert engine_frame.table("person")["person_household_id"].tolist() == [1, 1, 2, 2]
    assert engine_frame.table("benunit")["benunit_id"].tolist() == [1, 2]
    assert engine_frame.table("household")["household_id"].tolist() == [1, 2]
    # No build-identifier column goes to the engine, and nothing else moves.
    for entity in ("person", "benunit", "household"):
        assert f"{entity}_build_id" not in engine_frame.table(entity).columns
        assert list(engine_frame.table(entity).columns) == list(
            frame.table(entity).columns
        )
    assert engine_frame.table("person")["age"].tolist() == [35, 8, 40, 38]
    np.testing.assert_array_equal(
        engine_frame.weights_for("household").values,
        frame.weights_for("household").values,
    )
    assert engine_frame.mass_log == frame.mass_log
    # The resolver's own frame keeps the build identifiers.
    assert frame.table("household")["household_id"].tolist() == [100, 200]


def test_engine_scratch_block_takes_its_identifiers_from_the_whole_frame():
    """A per-clone block's rows carry the identifiers they have in the release,
    which numbers the whole frame, not the block."""
    whole = _tiny_national_frame([250.0, 125.0])
    block = whole.select(whole.table("person")["person_household_id"].eq(200))

    alone, _ = measure_simulation._engine_scratch_frame(block)
    in_whole, _ = measure_simulation._engine_scratch_frame(
        block, identifier_universe=whole
    )

    assert alone.table("person")["person_id"].tolist() == [1, 2]
    assert in_whole.table("person")["person_id"].tolist() == [3, 4]
    assert in_whole.table("person")["person_household_id"].tolist() == [2, 2]
    assert in_whole.table("benunit")["benunit_id"].tolist() == [2]
    assert in_whole.table("household")["household_id"].tolist() == [2]


def test_an_identifier_universe_needs_scratch_mode(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(__version__="9.9.9", Microsimulation=lambda *, dataset: None),
    )
    with pytest.raises(ValueError, match="identifier universe needs scratch-mode"):
        UKMeasureResolver(
            simulation_source=tmp_path / "input.h5",
            scratch_dir=tmp_path,
            year=2025,
            frame=FrameStub(),
            engine_identifier_universe=FrameStub(),
        )


def test_engine_scratch_frame_scales_household_weights_for_the_engine_only():
    frame = _tiny_national_frame([250.0, 125.0])
    before = frame.weights_for("household").values.copy()

    engine_frame, dropped = measure_simulation._engine_scratch_frame(
        frame, engine_weight_scale=4.0
    )

    assert dropped == ()
    np.testing.assert_allclose(
        engine_frame.weights_for("household").values, before * 4.0
    )
    assert (
        engine_frame.weights_for("household").kind
        is frame.weights_for("household").kind
    )
    assert len(engine_frame.mass_log) == len(frame.mass_log) + 1
    record = engine_frame.mass_log[-1]
    assert record.entity == "household"
    assert record.declared_factor == 4.0
    assert record.new_total == pytest.approx(record.old_total * 4.0)
    assert record.reason == measure_simulation.ENGINE_WEIGHT_SCALE_REASON
    # the resolver's own frame keeps the block's true mass
    np.testing.assert_array_equal(frame.weights_for("household").values, before)
    # no scale (or a unit scale) on an alias-free frame already on the
    # release's identifiers: the frame itself
    released = _tiny_release_identifier_frame([250.0, 125.0])
    assert measure_simulation._engine_scratch_frame(released)[0] is released
    assert (
        measure_simulation._engine_scratch_frame(released, engine_weight_scale=1.0)[0]
        is released
    )
    with pytest.raises(ValueError, match="positive finite"):
        measure_simulation._engine_scratch_frame(frame, engine_weight_scale=0.0)


def test_resolver_writes_the_scaled_engine_frame_and_records_the_scale(
    monkeypatch, tmp_path: Path
):
    created = []

    class FakeMicrosimulation:
        def __init__(self, *, dataset):
            created.append(dataset)
            self.tax_benefit_system = SimpleNamespace(variables={})

    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(__version__="9.9.9", Microsimulation=FakeMicrosimulation),
    )
    writes = []
    monkeypatch.setattr(
        measure_simulation,
        "write_uk_national_frame",
        lambda frame, path: writes.append((frame, path)) or path,
    )
    monkeypatch.setattr(
        measure_simulation, "validate_uc_claimant_input", lambda *args, **kwargs: None
    )
    frame = _tiny_national_frame([250.0, 125.0])

    resolver = UKMeasureResolver(
        simulation_source=None,
        scratch_dir=tmp_path,
        year=2025,
        frame=frame,
        engine_weight_scale=4.0,
    )

    assert resolver.frame is frame
    written, _path = writes[0]
    np.testing.assert_allclose(
        written.weights_for("household").values,
        frame.weights_for("household").values * 4.0,
    )
    assert resolver.receipt()["engine_weight_scale"] == 4.0
    assert created == [str(tmp_path / "simulation-input.h5")]
    with pytest.raises(ValueError, match="scratch-mode"):
        UKMeasureResolver(
            simulation_source=tmp_path / "input.h5",
            scratch_dir=tmp_path,
            year=2025,
            frame=frame,
            engine_weight_scale=4.0,
        )


def _bus_frame() -> FrameStub:
    """Three households: North East, London and Wales, with NTS trips."""

    frame = FrameStub()
    frame._tables["person"] = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4],
            "person_benunit_id": [10, 10, 20, 30],
            "person_household_id": [100, 100, 200, 300],
            "bus_in_london_trips": [10.0, 0.0, 4.0, 1.0],
            "other_local_bus_trips": [2.0, 3.0, 6.0, 5.0],
        }
    )
    frame._tables["household"] = pd.DataFrame(
        {"household_id": [100, 200, 300], "region": ["NORTH_EAST", "LONDON", "WALES"]}
    )
    return frame


def test_bus_boardings_price_each_series_at_the_stage_s_factor():
    """Household boardings: NTS trips times the lcfs stage's factor (#1207)."""

    from microcosm.build.country_spec import load_country_spec
    from microcosm.build.uk_runtime.bus_fare_pricing import bus_fare_prices
    from microcosm.build.uk_runtime.lcfs_consumption import (
        UK_LCFS_VENDORED_RESOURCES,
        bus_pricing_operation,
    )

    prices = bus_fare_prices(
        bus_pricing_operation(
            load_country_spec("uk").sources.stage_map()["lcfs_consumption"]
        ),
        allowed_resources=UK_LCFS_VENDORED_RESOURCES,
    )
    london_factor = prices.london.boardings_per_trip
    other_factor = prices.other_by_region["NORTH_EAST"].boardings_per_trip
    frame = _bus_frame()

    london, route = compute_uk_measure_input(
        frame, None, "household", "bus_in_london_boardings", 2025
    )
    other, _ = compute_uk_measure_input(
        frame, None, "household", "other_local_bus_boardings", 2025
    )

    # Every resident's bus-in-London trips, wherever they live.
    assert london.tolist() == pytest.approx(
        [10.0 * london_factor, 4.0 * london_factor, 1.0 * london_factor]
    )
    assert route == "stage_boardings_per_trip:bus_in_london_trips"
    # A London resident's other local trips take England-outside-London's
    # factor; Wales is unpriced and no row binds it.
    assert prices.other_by_region["LONDON"].boardings_per_trip == other_factor
    assert other.tolist() == pytest.approx([5.0 * other_factor, 6.0 * other_factor, 0])
    with pytest.raises(KeyError, match="household-only"):
        compute_uk_measure_input(frame, None, "person", "bus_in_london_boardings", 2025)


def test_a_persisted_bus_boarding_column_is_refused(monkeypatch, tmp_path: Path):
    frame = _bus_frame()
    frame._tables["household"]["other_local_bus_boardings"] = [1.0, 2.0, 3.0]
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(__version__="9.9.9", Microsimulation=lambda *, dataset: None),
    )
    with pytest.raises(ValueError, match="must not be persisted"):
        UKMeasureResolver(
            simulation_source=tmp_path / "input.h5",
            scratch_dir=tmp_path,
            year=2025,
            frame=frame,
        )


def test_the_resolver_routes_bus_boardings_to_households(monkeypatch, tmp_path: Path):
    resolver = _resolver_over(SimulationStub({}), monkeypatch, tmp_path)
    for name in ("bus_in_london_boardings", "other_local_bus_boardings"):
        assert resolver.entity_for(name) == "household"
        assert resolver.knows("household", name)
        assert not resolver.knows("person", name)


#: microcosm#1301: the three high-value homes region measures and the
#: threshold each reads.
_HIGH_VALUE_HOME_MEASURES = {
    "owner_occupied_home_over_gbp_1m": "over_gbp_1m",
    "owner_occupied_home_over_gbp_1_5m": "over_gbp_1_5m",
    "owner_occupied_home_over_gbp_2m": "over_gbp_2m",
}


def _high_value_home_frame() -> FrameStub:
    """Synthetic households straddling the frame-basis thresholds."""

    from microcosm.build.uk_runtime.high_value_homes import (
        uk_high_value_home_thresholds,
    )

    basis = {
        threshold.threshold_id: threshold.basis_gbp
        for threshold in uk_high_value_home_thresholds()
    }
    frame = FrameStub()
    frame._tables["person"] = pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "person_benunit_id": [10, 20, 30, 40, 50],
            "person_household_id": [100, 200, 300, 400, 500],
        }
    )
    frame._tables["household"] = pd.DataFrame(
        {
            "household_id": [100, 200, 300, 400, 500],
            "main_residence_value": [
                basis["over_gbp_2m"] + 1.0,  # owner over every threshold
                basis["over_gbp_1m"] + 1.0,  # owner over the lowest only
                basis["over_gbp_1m"],  # owner exactly at the threshold
                basis["over_gbp_2m"] + 1.0,  # private renter: never counted
                np.nan,  # owner without a stored value
            ],
            "tenure_type": [
                "OWNED_OUTRIGHT",
                "OWNED_WITH_MORTGAGE",
                "OWNED_OUTRIGHT",
                "RENT_PRIVATELY",
                "OWNED_OUTRIGHT",
            ],
        }
    )
    return frame


def test_high_value_home_measures_mask_stored_owner_values_by_threshold():
    """Each measure is the shared mask on the stored value and tenure (#1301)."""

    frame = _high_value_home_frame()
    expected = {
        "owner_occupied_home_over_gbp_1m": [1.0, 1.0, 0.0, 0.0, 0.0],
        "owner_occupied_home_over_gbp_1_5m": [1.0, 0.0, 0.0, 0.0, 0.0],
        "owner_occupied_home_over_gbp_2m": [1.0, 0.0, 0.0, 0.0, 0.0],
    }
    for variable, threshold_id in _HIGH_VALUE_HOME_MEASURES.items():
        # No engine: the stored columns are read off the frame, never uprated.
        values, route = compute_uk_measure_input(
            frame, None, "household", variable, 2025
        )
        assert values.tolist() == expected[variable], variable
        assert route == f"frame_columns:high_value_home_mask:{threshold_id}"
    assert measure_simulation._HIGH_VALUE_HOME_MEASURES == _HIGH_VALUE_HOME_MEASURES
    with pytest.raises(KeyError, match="household-only"):
        compute_uk_measure_input(
            frame, None, "person", "owner_occupied_home_over_gbp_1m", 2025
        )


def test_high_value_home_measures_need_the_stored_columns():
    frame = _high_value_home_frame()
    frame._tables["household"] = frame._tables["household"].drop(
        columns=["tenure_type"]
    )
    with pytest.raises(ValueError, match=r"\['tenure_type'\]"):
        compute_uk_measure_input(
            frame, None, "household", "owner_occupied_home_over_gbp_2m", 2025
        )


def test_the_contract_binds_each_region_row_on_its_threshold_measure():
    """The bound value variables are the measures, threshold for threshold."""

    targets = measure_simulation._uk_contract_targets()
    bound = {
        target_id: target["bindings"]["policyengine"]["value_variable"]
        for target_id, target in targets.items()
        if target_id.startswith("microcosm.high_value_homes.")
    }
    assert set(bound.values()) == set(_HIGH_VALUE_HOME_MEASURES)
    for target_id, variable in bound.items():
        threshold_id = _HIGH_VALUE_HOME_MEASURES[variable]
        selector = targets[target_id]["ledger_selector"]
        assert selector["record_set_spec_id"].endswith(f".{threshold_id}.v1")


def test_a_persisted_high_value_home_column_is_refused(monkeypatch, tmp_path: Path):
    frame = _high_value_home_frame()
    frame._tables["household"]["owner_occupied_home_over_gbp_1m"] = [1.0] * 5
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(__version__="9.9.9", Microsimulation=lambda *, dataset: None),
    )
    with pytest.raises(ValueError, match="must not be persisted"):
        UKMeasureResolver(
            simulation_source=tmp_path / "input.h5",
            scratch_dir=tmp_path,
            year=2025,
            frame=frame,
        )


def test_the_resolver_routes_high_value_home_measures_to_households(
    monkeypatch, tmp_path: Path
):
    resolver = _resolver_over(SimulationStub({}), monkeypatch, tmp_path)
    for name in _HIGH_VALUE_HOME_MEASURES:
        assert resolver.entity_for(name) == "household"
        assert resolver.knows("household", name)
        assert not resolver.knows("person", name)


def _savings_frame() -> FrameStub:
    """The producer's additive layout: a saver, an ISA-only saver, a non-saver."""

    frame = FrameStub()
    frame._tables["person"] = frame._tables["person"].assign(
        savings_interest_income=[700.0, 0.0, 0.0],
        individual_savings_account_interest_income=[300.0, 300.0, 0.0],
    )
    return frame


def _legacy_savings_frame() -> FrameStub:
    """The same people in the legacy gross layout (the enhanced FRS reference)."""

    frame = FrameStub()
    frame.metadata = {
        UK_ISA_INCOME_CONTRACT_ATTR: json.dumps(dict(UK_LEGACY_ISA_INCOME_CONTRACT))
    }
    frame._tables["person"] = frame._tables["person"].assign(
        savings_interest_income=[1_000.0, 300.0, 0.0],
        tax_free_savings_income=[300.0, 300.0, 0.0],
    )
    return frame


@pytest.mark.parametrize(
    ("frame", "route"),
    [
        (_savings_frame, "frame_columns:savings_interest_income"),
        (
            _legacy_savings_frame,
            "frame_columns:savings_interest_income_less_tax_free_savings_income",
        ),
    ],
)
def test_savings_interest_outside_isas_counts_isa_interest_once(frame, route):
    """SPI Table 3.7 counts recipients of taxable interest (microcosm#1095).

    Each layout reads the frame's declared contract (microcosm#1360).
    """

    # No engine: the stored columns are read off the frame, never uprated.
    values, measured = compute_uk_measure_input(
        frame(), None, "person", UK_SAVINGS_INTEREST_OUTSIDE_ISAS, 2025
    )
    assert values.tolist() == [700.0, 0.0, 0.0]
    assert measured == route
    with pytest.raises(KeyError, match="person-only"):
        compute_uk_measure_input(
            frame(), None, "household", UK_SAVINGS_INTEREST_OUTSIDE_ISAS, 2025
        )


def test_the_spi_interest_count_counts_recipients_of_interest_outside_isas():
    targets = measure_simulation._uk_contract_targets()
    filters = targets["hmrc.spi.savings_interest_income.count_by_total_income_band"][
        "bindings"
    ]["policyengine"]["filters"]
    assert {
        "variable": UK_SAVINGS_INTEREST_OUTSIDE_ISAS,
        "operator": ">",
        "value": 0,
    } in (filters)
    assert "savings_interest_income" not in {f["variable"] for f in filters}


def test_the_resolver_routes_interest_outside_isas_to_persons(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(
            __version__="9.9.9", Microsimulation=lambda *, dataset: SimulationStub({})
        ),
    )
    resolver = UKMeasureResolver(
        simulation_source=tmp_path / "input.h5",
        scratch_dir=tmp_path,
        year=2025,
        frame=_savings_frame(),
    )
    assert resolver.entity_for(UK_SAVINGS_INTEREST_OUTSIDE_ISAS) == "person"
    assert resolver.knows("person", UK_SAVINGS_INTEREST_OUTSIDE_ISAS)
    assert not resolver.knows("household", UK_SAVINGS_INTEREST_OUTSIDE_ISAS)

    frame = _savings_frame()
    frame._tables["person"][UK_SAVINGS_INTEREST_OUTSIDE_ISAS] = [700.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="must not be persisted"):
        UKMeasureResolver(
            simulation_source=tmp_path / "input.h5",
            scratch_dir=tmp_path,
            year=2025,
            frame=frame,
        )


def _stub_engine_term_calendar(monkeypatch) -> None:
    """policyengine-uk's year of terms, without the engine installed."""

    dates = SimpleNamespace(
        term_starts=lambda year: (
            year * 10_000 + 401,
            year * 10_000 + 901,
            (year + 1) * 10_000 + 101,
        )
    )
    utils = SimpleNamespace(dates=dates)
    monkeypatch.setitem(sys.modules, "policyengine_uk", SimpleNamespace(utils=utils))
    monkeypatch.setitem(sys.modules, "policyengine_uk.utils", utils)
    monkeypatch.setitem(sys.modules, "policyengine_uk.utils.dates", dates)


def _spring_term_simulation() -> SimulationStub:
    # Person 1 is an adult, person 2 a three-year-old on working-parent hours
    # in a family that does not claim the universal hours on their own, and
    # person 3 a four-year-old already in reception (a third of the year).
    return SimulationStub(
        {
            "birth_year": ("person", np.array([1990, 2022, 2021])),
            "early_years_term_share_before_school": (
                "person",
                np.array([1.0, 1.0, 1 / 3], dtype=np.float32),
            ),
            "extended_childcare_entitlement_total_per_child": (
                "person",
                np.array([0.0, 6_000.0, 2_000.0]),
            ),
            # The engine's universal eligibility in the year: the child in
            # reception is eligible in the summer term, before school starts.
            "universal_childcare_entitlement_eligible": (
                "person",
                np.array([False, True, True]),
            ),
            "max_free_entitlement_hours_used": (
                "person",
                np.array([30.0, 30.0, 30.0]),
            ),
            "maximum_extended_childcare_hours_usage": (
                "benunit",
                np.array([30.0, 30.0]),
            ),
            # A numeric group value reaches people through map_to.
            ("maximum_extended_childcare_hours_usage", "person"): (
                "person",
                np.array([30.0, 30.0, 30.0]),
            ),
            "targeted_childcare_entitlement_eligible": (
                "benunit",
                np.array([False, False]),
            ),
            "would_claim_targeted_childcare": ("benunit", np.array([True, True])),
            "would_claim_universal_childcare": ("benunit", np.array([False, True])),
        }
    )


def test_dfe_spring_term_measures_classify_people_at_the_spring_term(monkeypatch):
    _stub_engine_term_calendar(monkeypatch)
    frame, simulation = FrameStub(), _spring_term_simulation()
    expected = {
        "dfe_age_at_31_december": [35.0, 3.0, 4.0],
        # The working-parent child keeps its universal hours; the child in
        # reception is not funded in the spring term.
        "dfe_spring_working_parent": [0.0, 1.0, 0.0],
        "dfe_spring_universal": [0.0, 1.0, 0.0],
        "dfe_spring_early_learning_2": [0.0, 0.0, 0.0],
    }
    for variable, values in expected.items():
        computed, route = compute_uk_measure_input(
            frame, simulation, "person", variable, 2025
        )
        assert computed.tolist() == values, variable
        assert route == f"dfe_spring_term:{variable}"
    with pytest.raises(KeyError, match="person-only"):
        compute_uk_measure_input(
            frame, simulation, "household", "dfe_spring_universal", 2025
        )


def test_dfe_spring_term_measures_refuse_a_moved_term_calendar(monkeypatch):
    dates = SimpleNamespace(term_starts=lambda year: (0, 0, year * 10_000 + 901))
    utils = SimpleNamespace(dates=dates)
    monkeypatch.setitem(sys.modules, "policyengine_uk", SimpleNamespace(utils=utils))
    monkeypatch.setitem(sys.modules, "policyengine_uk.utils", utils)
    monkeypatch.setitem(sys.modules, "policyengine_uk.utils.dates", dates)
    with pytest.raises(ValueError, match="April-to-March year"):
        compute_uk_measure_input(
            FrameStub(),
            _spring_term_simulation(),
            "person",
            "dfe_spring_universal",
            2025,
        )


def test_the_resolver_routes_dfe_spring_term_measures_to_people(
    monkeypatch, tmp_path: Path
):
    from microcosm.build.uk_runtime.dfe_spring_term import DFE_SPRING_TERM_MEASURES

    resolver = _resolver_over(_spring_term_simulation(), monkeypatch, tmp_path)
    for name in DFE_SPRING_TERM_MEASURES:
        assert resolver.entity_for(name) == "person"
        assert resolver.knows("person", name)
        assert not resolver.knows("household", name)
    # Without one of the engine inputs the route cannot compute, so it is
    # not known.
    partial = _spring_term_simulation()
    del partial.tax_benefit_system.variables["birth_year"]
    resolver = _resolver_over(partial, monkeypatch, tmp_path)
    assert not resolver.knows("person", "dfe_spring_universal")


def test_a_persisted_dfe_spring_term_column_is_refused(monkeypatch, tmp_path: Path):
    frame = FrameStub()
    frame._tables["person"]["dfe_spring_universal"] = [0.0, 1.0, 0.0]
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(__version__="9.9.9", Microsimulation=lambda *, dataset: None),
    )
    with pytest.raises(ValueError, match="must not be persisted"):
        UKMeasureResolver(
            simulation_source=tmp_path / "input.h5",
            scratch_dir=tmp_path,
            year=2025,
            frame=frame,
        )


# The annual income measures: each current measure on a simulation of each
# person's year, built from the frs_annual_income_spells descriptors.
_ANNUAL_DESCRIPTORS = {
    # Person 1 is paid all year; person 2 left a job and retired in the year,
    # on a State Pension that started two months before its middle; person 3
    # started a job half way through and ran a business for half of it.
    "employment_income_annual_share": [1.0, 1.0, 0.5],
    "employment_income_other_spells": [0.0, 9_000.0, 0.0],
    "self_employment_income_annual_share": [1.0, 1.0, 0.5],
    "self_employment_income_other_spells": [0.0, 0.0, 1_000.0],
    "private_pension_income_annual_share": [1.0, 0.25, 1.0],
    "contributory_benefits_annual_share": [1.0, 0.25, 1.0],
}


class _AnnualFrameStub(FrameStub):
    metadata = {"time_period": "2024"}

    def __init__(self, descriptors=_ANNUAL_DESCRIPTORS):
        super().__init__()
        for column, values in descriptors.items():
            self._tables["person"][column] = values


class _AnnualSimulation:
    """A stub engine that uprates stored inputs and taxes total income."""

    created = 0
    uprating = {
        "employment_income_before_lsr": 1.05,
        "self_employment_income": 1.03,
        "private_pension_income": 1.02,
        "jsa_contrib_reported": 1.02,
        "esa_contrib_reported": 1.02,
        "incapacity_benefit_reported": 1.02,
    }
    extra_inputs: dict = {}

    def __init__(self, *, dataset):
        type(self).created += 1
        self.stored = {
            ("employment_income_before_lsr", 2024): np.array([30_000.0, 0.0, 20_000.0]),
            ("self_employment_income", 2024): np.array([0.0, 0.0, 6_000.0]),
            ("private_pension_income", 2024): np.array([0.0, 8_000.0, 0.0]),
            ("jsa_contrib_reported", 2024): np.zeros(3),
            ("esa_contrib_reported", 2024): np.array([0.0, 2_000.0, 0.0]),
            ("incapacity_benefit_reported", 2024): np.zeros(3),
            ("loss_relief", 2024): np.zeros(3),
            **type(self).extra_inputs,
        }
        names = (
            *self.uprating,
            "employment_income",
            "loss_relief",
            "state_pension",
            "months_since_state_pension_age",
            "carers_allowance",
            "carer_support_payment",
            "scottish_carer_supplement",
            "total_income",
            "income_tax",
            "income_tax_before_winter_fuel_payment_charge",
            "basic_rate_savings_income",
            "higher_rate_savings_income",
            "add_rate_savings_income",
            "total_pension_income",
        )
        variables = {
            name: SimpleNamespace(
                entity=SimpleNamespace(key="person"), value_type=float
            )
            for name in names
        }
        variables["tax_band"] = SimpleNamespace(
            entity=SimpleNamespace(key="person"), value_type=str
        )
        self.tax_benefit_system = SimpleNamespace(variables=variables)

    def set_input(self, variable, year, values):
        self.stored[(variable, int(year))] = np.asarray(values, dtype=float)

    def calculate(self, variable, year, map_to=None):
        year = int(year)
        if (variable, year) in self.stored:
            return self.stored[(variable, year)]
        if variable in self.uprating or variable == "loss_relief":
            source = max(y for v, y in self.stored if v == variable and y <= year)
            factor = self.uprating.get(variable, 1.0) ** (year - source)
            return self.stored[(variable, source)] * factor
        pay = self.calculate("employment_income_before_lsr", year)
        if variable == "employment_income":
            return pay
        if variable == "months_since_state_pension_age":
            return np.array([-200.0, 2.0, -100.0])
        if variable == "state_pension":
            return np.array([0.0, 11_000.0, 0.0])
        if variable == "carers_allowance":
            # Earnings-tested: the year's pay must not take it away.
            return np.where(pay < 5_000.0, 4_000.0, 0.0)
        if variable in ("carer_support_payment", "scottish_carer_supplement"):
            return np.zeros(3)
        if variable == "total_pension_income":
            # The engine's sum of the two pensions (microcosm#1418).
            return self.calculate("private_pension_income", year) + self.calculate(
                "state_pension", year
            )
        total = sum(
            self.calculate(name, year)
            for name in (
                "employment_income",
                "self_employment_income",
                "private_pension_income",
                "state_pension",
                "jsa_contrib_reported",
                "esa_contrib_reported",
                "incapacity_benefit_reported",
                "carers_allowance",
            )
        )
        if variable == "total_income":
            return total
        # No stub person passes the Winter Fuel Payment charge's income test.
        if variable in ("income_tax", "income_tax_before_winter_fuel_payment_charge"):
            return 0.2 * np.maximum(total - 12_570.0, 0.0)
        if variable == "tax_band":
            return np.where(
                total > 50_270.0, "HIGHER", np.where(total > 12_570.0, "BASIC", "NONE")
            )
        # No stub person has savings income (microcosm#1418).
        if variable in (
            "basic_rate_savings_income",
            "higher_rate_savings_income",
            "add_rate_savings_income",
        ):
            return np.zeros(3)
        raise KeyError(variable)


def _annual_resolver(monkeypatch, tmp_path, frame=None):
    _AnnualSimulation.created = 0
    monkeypatch.setitem(
        sys.modules,
        "policyengine_uk",
        SimpleNamespace(__version__="9.9.9", Microsimulation=_AnnualSimulation),
    )
    return UKMeasureResolver(
        simulation_source=tmp_path / "input.h5",
        scratch_dir=tmp_path,
        year=2025,
        frame=_AnnualFrameStub() if frame is None else frame,
    )


def test_state_pension_annual_share_counts_the_months_paid_in_the_year():
    share = measure_simulation.state_pension_annual_share(
        [-12.0, -6.0, -5.5, 0.0, 2.0, 6.0, 30.0, np.nan]
    )
    np.testing.assert_allclose(share, [1.0, 1.0, 0.5 / 12, 0.5, 8 / 12, 1.0, 1.0, 1.0])


def test_annual_measures_read_each_persons_year(monkeypatch, tmp_path):
    resolver = _annual_resolver(monkeypatch, tmp_path)

    pay, route = resolver.compute("person", "employment_income_annual")
    # The engine carries the year's pay from 2024 at its own index.
    np.testing.assert_allclose(pay, [31_500.0, 9_450.0, 10_500.0])
    assert route == "annual_income_simulation:native"
    profit, route = resolver.compute(
        "person", "self_employment_income_after_loss_relief_annual"
    )
    np.testing.assert_allclose(profit, [0.0, 0.0, 4_120.0])
    assert route.startswith("annual_income_simulation:engine_components")
    pension, _ = resolver.compute("person", "private_pension_income_annual")
    np.testing.assert_allclose(pension, [0.0, 2_040.0, 0.0])
    state_pension, _ = resolver.compute("person", "state_pension_annual")
    np.testing.assert_allclose(state_pension, [0.0, 11_000.0 * 8 / 12, 0.0])
    # Both pensions over the year: the SPI area tables' pension income.
    both, _ = resolver.compute("person", "total_pension_income_annual")
    np.testing.assert_allclose(both, [0.0, 2_040.0 + 11_000.0 * 8 / 12, 0.0])
    total, _ = resolver.compute("person", "total_income_after_loss_relief_annual")
    # Person 2 keeps the Carer's Allowance paid at the earnings of the time.
    annual_total = np.array(
        [31_500.0, 9_450.0 + 2_040.0 + 11_000.0 * 8 / 12 + 510.0 + 4_000.0, 14_620.0]
    )
    np.testing.assert_allclose(total, annual_total)
    tax, _ = resolver.compute("person", "income_tax_annual")
    np.testing.assert_allclose(tax, 0.2 * np.maximum(annual_total - 12_570.0, 0.0))
    band, _ = resolver.compute("person", "tax_band_annual")
    assert list(band) == ["BASIC", "BASIC", "BASIC"]
    # The current measures are untouched, and one annual simulation serves all.
    current_tax, _ = resolver.compute("person", "income_tax")
    assert current_tax[1] == pytest.approx(
        0.2 * (8_160.0 + 11_000.0 + 2_040.0 + 4_000.0 - 12_570.0)
    )
    assert _AnnualSimulation.created == 2
    receipt = resolver.receipt()["annual_income_measures"]
    assert receipt["route"] == "annual_income_simulation"
    assert receipt["dataset_year"] == 2024
    assert receipt["computed_measures"]["income_tax_annual"] == "income_tax"
    assert receipt["held_at_baseline"] == [
        "carers_allowance",
        "carer_support_payment",
        "scottish_carer_supplement",
    ]
    for name in measure_simulation.UK_ANNUAL_INCOME_MEASURES:
        assert resolver.knows("person", name)
        assert resolver.entity_for(name) == "person"


def test_the_annual_loss_relief_terms_are_the_terms_on_the_annual_simulation(
    monkeypatch, tmp_path
):
    # Each _annual form is its named term evaluated on the engine's leaves on
    # the annual simulation: the floats the resolver gave by name before the
    # terms were declared (microcosm#1095).
    resolver = _annual_resolver(monkeypatch, tmp_path)
    after, after_route = resolver.compute(
        "person", "self_employment_income_after_loss_relief_annual"
    )
    banded, banded_route = resolver.compute(
        "person", "total_income_after_loss_relief_annual"
    )

    annual = resolver._annual_income_simulation()
    leaf = {
        name: measure_simulation.compute_uk_measure_input(
            resolver.frame, annual, "person", name, 2025
        )[0]
        for name in ("self_employment_income", "loss_relief", "total_income")
    }
    former = np.maximum(leaf["self_employment_income"] - leaf["loss_relief"], 0.0)
    assert np.array_equal(after, former)
    assert np.array_equal(
        banded, leaf["total_income"] - leaf["self_employment_income"] + former
    )
    assert after_route == (
        "annual_income_simulation:engine_components:"
        "self_employment_income_after_loss_relief"
    )
    assert banded_route == (
        "annual_income_simulation:engine_components:total_income_after_loss_relief"
    )
    for name in measure_simulation.UK_NAMED_TERMS:
        assert resolver.knows("person", name)
        assert not resolver.knows("household", name)
        assert resolver.entity_for(name) == "person"


def test_annual_measures_refuse_a_frame_without_descriptors(monkeypatch, tmp_path):
    frame = _AnnualFrameStub(descriptors={})
    resolver = _annual_resolver(monkeypatch, tmp_path, frame=frame)
    assert not resolver.knows("person", "income_tax_annual")
    with pytest.raises(KeyError, match="annual income descriptors"):
        resolver.compute("person", "income_tax_annual")
    # The incumbent alone reads its current values.
    resolver.use_current_income_for_annual_measures(reason="incumbent")
    assert resolver.knows("person", "income_tax_annual")
    tax, route = resolver.compute("person", "income_tax_annual")
    current, _ = resolver.compute("person", "income_tax")
    np.testing.assert_allclose(tax, current)
    assert route.startswith("current_values_without_annual_income_descriptors:")
    assert _AnnualSimulation.created == 1
    receipt = resolver.receipt()["annual_income_measures"]
    assert receipt["route"] == "current_values_without_annual_income_descriptors"
    assert receipt["reason"] == "incumbent"


def test_the_incumbent_fallback_never_hides_descriptors(monkeypatch, tmp_path):
    resolver = _annual_resolver(monkeypatch, tmp_path)
    resolver.use_current_income_for_annual_measures(reason="incumbent")
    pay, route = resolver.compute("person", "employment_income_annual")
    np.testing.assert_allclose(pay, [31_500.0, 9_450.0, 10_500.0])
    assert route.startswith("annual_income_simulation:")
    partial = dict(_ANNUAL_DESCRIPTORS)
    partial.pop("contributory_benefits_annual_share")
    resolver = _annual_resolver(
        monkeypatch, tmp_path, frame=_AnnualFrameStub(descriptors=partial)
    )
    with pytest.raises(ValueError, match="only some annual income descriptors"):
        resolver.knows("person", "income_tax_annual")


def test_other_spells_follow_one_engine_factor(monkeypatch, tmp_path):
    # A projection that is not one factor for every row cannot carry the
    # other spells as it carries pay.
    _AnnualSimulation.extra_inputs = {
        ("employment_income_before_lsr", 2025): np.array([31_500.0, 0.0, 22_000.0])
    }
    try:
        resolver = _annual_resolver(monkeypatch, tmp_path)
        with pytest.raises(ValueError, match="by one factor"):
            resolver.compute("person", "income_tax_annual")
    finally:
        _AnnualSimulation.extra_inputs = {}
    factors = _annual_resolver(monkeypatch, tmp_path)
    factors.compute("person", "employment_income_annual")
    receipt = factors.receipt()["annual_income_measures"]
    assert receipt["other_spells_engine_factors"] == {
        "employment_income_before_lsr": pytest.approx(1.05),
        "self_employment_income": pytest.approx(1.03),
    }


def test_no_committed_exclusion_is_about_to_lapse():
    # The fifth re-review of microcosm#1420, 1: the loader keeps an expired
    # entry, and only the national targets' window check refuses it, after
    # the spine. A clock on purpose, as for the uprating holds: this fails
    # UK_HOLD_EXPIRY_NOTICE_DAYS days before the earliest expiry. Clear it by
    # a new review and signature, or by binding the row. The register's
    # support floor is refused at the same step, so its window is on the clock
    # too (the sixth re-review, 1).
    from datetime import date

    from microcosm.build.uk_runtime.support_floor import (
        load_uk_support_floor_declaration,
        uk_support_floor_from_payload,
    )
    from microcosm.build.uk_runtime.uprating_holds import UK_HOLD_EXPIRY_NOTICE_DAYS

    expiries = {
        str(entry["name"]): date.fromisoformat(str(entry["expires_on"]))
        for entry in load_uk_calibration_measure_exclusions()
        if entry.get("expires_on")
    }
    floor = load_uk_support_floor_declaration()
    if floor is not None:
        expiries["support_floor"] = uk_support_floor_from_payload(floor).expires_on
    if not expiries:
        return
    earliest = min(expiries.values())
    lapsing = sorted(name for name, expiry in expiries.items() if expiry == earliest)
    days_left = (earliest - date.today()).days
    assert days_left > UK_HOLD_EXPIRY_NOTICE_DAYS, (
        f"UK calibration measure exclusions {lapsing} expire on "
        f"{earliest.isoformat()}, in {days_left} day(s); every UK national "
        "refuses them from the next day."
    )
