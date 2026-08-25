from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import measure_simulation
from microcosm.build.uk_runtime.measure_simulation import (
    UKMeasureResolver,
    apply_uk_calibration_measure_exclusions,
    compute_uk_measure_input,
    load_uk_calibration_measure_exclusions,
)
from microcosm.calibrate import TargetRegistry, TargetSpec


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


def _stub_value_type(values):
    """The policyengine ``value_type`` a real variable definition would carry."""

    return {"f": float, "i": int, "b": bool}.get(
        np.asarray(values).dtype.kind, str
    )


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


def test_compute_uk_measure_input_native_route():
    sim = SimulationStub({"income_tax": ("person", np.array([1.0, 2.0, 3.0]))})

    values, route = compute_uk_measure_input(
        FrameStub(), sim, "person", "income_tax", 2025
    )

    assert route == "native"
    assert values.tolist() == [1.0, 2.0, 3.0]


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
        compute_uk_measure_input(FrameStub(), SimulationStub({}), "person", "missing", 2025)

    sim = SimulationStub({"income_tax": ("person", np.array([1.0]))})
    with pytest.raises(ValueError, match="produced 1 values"):
        compute_uk_measure_input(FrameStub(), sim, "person", "income_tax", 2025)


def _write_exclusions(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_exclusion_loader_refusals(tmp_path: Path):
    base = {
        "schema_version": 1,
        "exclusions": [
            {"name": "a", "reason": "reviewed reason", "tracking": "microcosm#623"}
        ],
    }

    with pytest.raises(ValueError, match="schema_version"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(tmp_path / "bad-version.json", {**base, "schema_version": 2})
        )
    with pytest.raises(ValueError, match="unknown top-level"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(tmp_path / "unknown.json", {**base, "extra": True})
        )
    with pytest.raises(ValueError, match="empty reason"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "empty.json",
                {**base, "exclusions": [{"name": "a", "reason": "", "tracking": "x"}]},
            )
        )
    with pytest.raises(ValueError, match="duplicate"):
        load_uk_calibration_measure_exclusions(
            _write_exclusions(
                tmp_path / "duplicate.json",
                {
                    **base,
                    "exclusions": [
                        {"name": "a", "reason": "one", "tracking": "x"},
                        {"name": "a", "reason": "two", "tracking": "x"},
                    ],
                },
            )
        )


def test_exclusion_applier_returns_pruned_registry_and_receipt():
    registry = TargetRegistry(
        [
            TargetSpec(name="drop", entity="person", measure="drop", value=1.0, source="test"),
            TargetSpec(name="keep", entity="person", measure="keep", value=1.0, source="test"),
        ],
        country="uk",
    )

    pruned, receipt = apply_uk_calibration_measure_exclusions(
        registry,
        ({"name": "drop", "reason": "reviewed", "tracking": "microcosm#623"},),
    )

    assert [spec.name for spec in pruned.specs] == ["keep"]
    assert receipt == {
        "drop": {"reason": "reviewed", "tracking": "microcosm#623"}
    }
    with pytest.raises(ValueError, match="matched zero"):
        apply_uk_calibration_measure_exclusions(
            registry,
            ({"name": "stale", "reason": "reviewed", "tracking": "microcosm#623"},),
        )


def test_packaged_exclusions_load():
    exclusions = load_uk_calibration_measure_exclusions()
    assert {entry["name"] for entry in exclusions} == {
        "hmrc.salary_sacrifice.it_relief_basic_rate",
        "hmrc.salary_sacrifice.it_relief_higher_rate",
        "hmrc.salary_sacrifice.it_relief_additional_rate",
        "hmrc.salary_sacrifice.nics_relief_employee",
        "hmrc.salary_sacrifice.nics_relief_employer",
    }


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

    frame = FrameStub()
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
    assert scratch.receipt()["mode"] == "scratch_frame_export"
    assert writes == [(frame, tmp_path / "simulation-input.h5")]
    assert created == [str(tmp_path / "input.h5"), str(tmp_path / "simulation-input.h5")]


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
                {
                    "schema_version": 1,
                    "exclusions": [
                        {"name": "a", "reason": "reviewed reason", "tracking": ""}
                    ],
                },
            )
        )
