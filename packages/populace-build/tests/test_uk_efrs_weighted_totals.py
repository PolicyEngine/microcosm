"""Weighted-totals extraction from the pinned enhanced-FRS incumbent (#609).

This is the reference side of the UK input_mass_parity gate: the frozen totals
a candidate is measured against. The licensed artifact is never in CI, so the
sha pin is stubbed and the extraction runs against a synthetic H5 with the same
shape — enough to prove the plumbing (table reading, engine-variable
classification, weight broadcast, key format) before a licensed run pays for it.

Requires policyengine-uk, which the extraction imports to classify loader
inputs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]

pytest.importorskip(
    "policyengine_uk",
    reason="the eFRS weighted-totals extraction classifies loader inputs "
    "against the live UK engine",
)


def _tool_module():
    path = ROOT / "tools" / "build_uk_efrs_parity_reference.py"
    spec = importlib.util.spec_from_file_location(
        "build_uk_efrs_parity_reference",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = _tool_module()


@pytest.fixture
def synthetic_efrs(tmp_path, monkeypatch) -> Path:
    """A three-table H5 shaped like the incumbent, with the sha pin stubbed."""

    n_person, n_household = 6, 3
    household_ids = np.arange(1, n_household + 1, dtype=np.int64)
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, n_person + 1, dtype=np.int64),
            "person_household_id": np.repeat(household_ids, 2),
            "person_benunit_id": np.repeat(household_ids, 2),
            # Engine-known person inputs.
            "age": np.array([40, 38, 35, 33, 70, 68], dtype=np.int64),
            "employment_income": np.array(
                [30_000.0, 0.0, 25_000.0, 10_000.0, 0.0, 0.0]
            ),
            # Not an engine variable: must be excluded from the totals.
            "populace_scratch_column": np.arange(6, dtype=np.float64),
        }
    )
    benunit = pd.DataFrame({"benunit_id": household_ids})
    household = pd.DataFrame(
        {
            "household_id": household_ids,
            # Distinct weights so a broadcast error cannot pass unnoticed.
            "household_weight": np.array([10.0, 100.0, 1_000.0]),
            "council_tax": np.array([1_000.0, 2_000.0, 3_000.0]),
        }
    )
    path = tmp_path / "enhanced_frs_2023_24.h5"
    with pd.HDFStore(path) as store:
        store.put("person", person, format="table", data_columns=True)
        store.put("benunit", benunit, format="table", data_columns=True)
        store.put("household", household, format="table", data_columns=True)
        store.put(
            "time_period",
            pd.Series([TOOL.SOURCE_PERIOD]),
            format="table",
            data_columns=True,
        )
    monkeypatch.setattr(TOOL, "_verify_source", lambda _path: None)
    return path


def test_totals_are_weighted_and_keyed_by_owning_entity(synthetic_efrs) -> None:
    payload = TOOL.build_weighted_totals(synthetic_efrs)
    totals = payload["totals"]

    # Person values ride their household's weight: 30k*10 + 25k*1000 (person 3
    # sits in household 2 at weight 100) — assert against the explicit sum.
    weights = {1: 10.0, 2: 100.0, 3: 1_000.0}
    expected_employment = (
        30_000.0 * weights[1] + 25_000.0 * weights[2] + 10_000.0 * weights[2]
    )
    assert totals["person.employment_income"] == pytest.approx(expected_employment)
    assert totals["household.council_tax"] == pytest.approx(
        1_000.0 * weights[1] + 2_000.0 * weights[2] + 3_000.0 * weights[3]
    )
    assert totals["person.age"] == pytest.approx(
        (40 + 38) * weights[1] + (35 + 33) * weights[2] + (70 + 68) * weights[3]
    )


def test_structural_and_unknown_columns_are_not_input_mass(synthetic_efrs) -> None:
    totals = TOOL.build_weighted_totals(synthetic_efrs)["totals"]

    # The weight vector is plumbing, not mass, even though the engine knows it.
    assert "household.household_weight" not in totals
    for structural in (
        "person.person_id",
        "person.person_household_id",
        "person.person_benunit_id",
        "benunit.benunit_id",
        "household.household_id",
    ):
        assert structural not in totals
    # A pipeline scratch column the engine does not know is out of scope.
    assert "person.populace_scratch_column" not in totals


def test_payload_pins_the_reference_identity_the_gate_records(
    synthetic_efrs,
) -> None:
    payload = TOOL.build_weighted_totals(synthetic_efrs)

    assert payload["schema_version"] == 1
    assert payload["identity"] == {
        "filename": TOOL.SOURCE_FILENAME,
        "revision": TOOL.SOURCE_REVISION,
        "sha256": TOOL.SOURCE_SHA256,
        "vintage": TOOL.SOURCE_VINTAGE,
    }


def test_payload_loads_as_a_gate_reference(
    synthetic_efrs,
    tmp_path,
    monkeypatch,
) -> None:
    """The emitted file is exactly what the gate's loader accepts."""

    from populace.build.uk_runtime import weighted_integrity
    from populace.build.uk_runtime.weighted_integrity import (
        UKInputMassParityPolicy,
        load_uk_input_mass_reference,
        uk_input_mass_parity_gate,
    )

    payload = TOOL.build_weighted_totals(synthetic_efrs)
    path = tmp_path / "efrs_weighted_totals.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    # This synthetic source deliberately stubs the licensed artifact sha.
    # The reviewed 131-column digest is covered by the runtime regressions.
    monkeypatch.setattr(
        weighted_integrity,
        "_validate_input_mass_reference",
        lambda _reference: None,
    )
    reference = load_uk_input_mass_reference(path)
    identical = uk_input_mass_parity_gate(
        dict(reference.totals),
        reference,
        policy=UKInputMassParityPolicy(
            relative_tolerance=0.0,
            minimum_reference_total=0.0,
        ),
    )
    zeroed = uk_input_mass_parity_gate(
        {name: 0.0 for name in reference.totals},
        reference,
        policy=UKInputMassParityPolicy(
            relative_tolerance=0.0,
            minimum_reference_total=0.0,
        ),
    )

    assert reference.filename == TOOL.SOURCE_FILENAME
    assert identical.passed
    assert not zeroed.passed


def test_emitting_totals_never_rewrites_the_committed_reference(
    synthetic_efrs,
    tmp_path,
    monkeypatch,
) -> None:
    """A measurement run must not republish the coverage contract (#609)."""

    committed = Path(TOOL.REFERENCE_PATH)
    before = committed.read_bytes()
    destination = tmp_path / "outside" / "efrs_weighted_totals.json"
    monkeypatch.setattr(TOOL, "resolve_source_h5", lambda _explicit: synthetic_efrs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_efrs_parity_reference.py",
            "--input-h5",
            str(synthetic_efrs),
            "--emit-weighted-totals",
            str(destination),
        ],
    )

    assert TOOL.main() == 0

    assert committed.read_bytes() == before
    assert json.loads(destination.read_text())["identity"]["sha256"] == (
        TOOL.SOURCE_SHA256
    )


def test_totals_destination_inside_the_repository_is_refused(
    synthetic_efrs,
    monkeypatch,
) -> None:
    monkeypatch.setattr(TOOL, "resolve_source_h5", lambda _explicit: synthetic_efrs)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_uk_efrs_parity_reference.py",
            "--emit-weighted-totals",
            str(Path(TOOL.REPO_ROOT) / "efrs_weighted_totals.json"),
        ],
    )

    with pytest.raises(SystemExit, match="inside the repository"):
        TOOL.main()
