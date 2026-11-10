"""The column surface of a UK cut's reference build (microcosm#1095)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.uk_runtime import national_frame, rowwise_dataset, weight_year
from microcosm.build.uk_runtime.cut_reference import (
    UK_CUT_REFERENCE_INPUT_MASS_NAME,
    UK_CUT_REFERENCE_SURFACE_NAME,
    uk_cut_reference_evidence_done,
)
from microcosm.build.uk_runtime.reference_surface import (
    load_uk_reference_surface,
    uk_reference_surface_payload,
)
from test_support.microcosm_build.uk_release_input_coverage import _StubEngine

LABEL = "microcosm-uk-2024-25-national@uk-frs-calibration-attempt-test"
SHA = "a" * 64


def _payload(**overrides) -> dict:
    payload = {
        "schema_version": 1,
        "kind": "uk_reference_surface",
        "label": LABEL,
        "sha256": SHA,
        "minimum_nondefault_mass_share": 1e-6,
        "export_columns": ["person.employment_income", "person.dividend_income"],
        "populated_inputs": ["employment_income"],
        "thin_inputs": ["dividend_income"],
    }
    payload.update(overrides)
    return payload


def test_a_surface_round_trips_names_and_flags(tmp_path: Path) -> None:
    path = tmp_path / UK_CUT_REFERENCE_SURFACE_NAME
    path.write_text(json.dumps(_payload()))
    surface = load_uk_reference_surface(path)
    assert surface.label == LABEL
    assert surface.sha256 == SHA
    assert surface.minimum_nondefault_mass_share == 1e-6
    assert surface.export_columns == {
        "person.employment_income",
        "person.dividend_income",
    }
    assert surface.carried_inputs == {"employment_income", "dividend_income"}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 2}, "schema-1"),
        ({"kind": "uk_input_mass_reference"}, "schema-1"),
        ({"label": "no-build"}, "<release>@<build>"),
        ({"sha256": "short"}, "sha256"),
        ({"minimum_nondefault_mass_share": 0.0}, "effective-mass floor"),
        ({"minimum_nondefault_mass_share": True}, "effective-mass floor"),
        ({"populated_inputs": "employment_income"}, "list of column names"),
        ({"thin_inputs": ["employment_income"]}, "both populated and thin"),
        ({"export_columns": []}, "exports no column"),
    ],
)
def test_a_malformed_surface_is_refused(tmp_path: Path, overrides, message) -> None:
    path = tmp_path / UK_CUT_REFERENCE_SURFACE_NAME
    path.write_text(json.dumps(_payload(**overrides)))
    with pytest.raises(ValueError, match=message):
        load_uk_reference_surface(path)


def _reference_frame():
    # Five households: two clones of one source household (101), two more
    # sources, and a zero-weight row. Employment income sits on three
    # sources, dividends on one (both clones of 101), savings interest only
    # on the zero-weight row.
    household = pd.DataFrame(
        {
            "household_id": np.arange(1, 6, dtype="int64"),
            "household_source_id": np.asarray([101, 101, 102, 103, 104]),
        }
    )
    person = pd.DataFrame(
        {
            "person_id": np.arange(1, 6, dtype="int64"),
            "person_household_id": np.arange(1, 6, dtype="int64"),
            "person_benunit_id": np.arange(1, 6, dtype="int64"),
            "employment_income": np.asarray([1.0, 1.0, 1.0, 1.0, 0.0]),
            "dividend_income": np.asarray([5.0, 5.0, 0.0, 0.0, 0.0]),
            "savings_interest_income": np.asarray([0.0, 0.0, 0.0, 0.0, 9.0]),
        }
    )
    benunit = pd.DataFrame({"benunit_id": np.arange(1, 6, dtype="int64")})
    return national_frame.uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
        household_weights=np.asarray([10.0, 10.0, 10.0, 10.0, 0.0]),
    )


def test_the_payload_flags_inputs_by_how_many_source_households_carry_them(
    monkeypatch,
) -> None:
    frame = _reference_frame()
    monkeypatch.setattr(
        rowwise_dataset, "load_uk_rowwise_dataset", lambda path: (frame, None)
    )
    # The test frame records no solve; the real one is put back at its own
    # solved weights by the factor its mass log records.
    monkeypatch.setattr(weight_year, "uk_recorded_solve_year_frame", lambda f: f)
    engine = _StubEngine(
        {
            "employment_income": 0.0,
            "dividend_income": 0.0,
            "savings_interest_income": 0.0,
        }
    )

    payload = uk_reference_surface_payload(
        "reference.h5", label=LABEL, sha256=SHA, engine=engine
    )

    assert payload["label"] == LABEL
    assert payload["sha256"] == SHA
    assert payload["minimum_nondefault_mass_share"] == 1e-6
    assert payload["populated_inputs"] == ["employment_income"]
    # One source household carries dividends: listed by name only, "<3".
    assert payload["thin_inputs"] == ["dividend_income"]
    assert "person.savings_interest_income" in payload["export_columns"]
    assert "person.employment_income" in payload["export_columns"]
    # Names and flags only: no figure besides the policy floor.
    assert set(payload) == set(_payload())


def test_the_evidence_is_done_only_when_both_files_name_the_build(
    tmp_path: Path,
) -> None:
    reference = type("Reference", (), {"label": LABEL, "sha256": SHA})()
    mass = tmp_path / UK_CUT_REFERENCE_INPUT_MASS_NAME
    mass.write_text(json.dumps({"label": LABEL, "identity": {"sha256": SHA}}))
    done, reason = uk_cut_reference_evidence_done(reference, tmp_path)
    assert not done and UK_CUT_REFERENCE_SURFACE_NAME in reason

    surface = tmp_path / UK_CUT_REFERENCE_SURFACE_NAME
    surface.write_text(json.dumps(_payload(sha256="b" * 64)))
    done, reason = uk_cut_reference_evidence_done(reference, tmp_path)
    assert not done and "another build" in reason

    surface.write_text(json.dumps(_payload()))
    assert uk_cut_reference_evidence_done(reference, tmp_path)[0]


def test_the_evidence_reads_a_dense_line_file(tmp_path: Path) -> None:
    """A dense build's file carries one ``clone_index`` name on every table,
    which the national loader refuses; the reference evidence opens it."""

    pytest.importorskip("tables")
    from microcosm.build.uk_runtime import (
        clone_uk_dataset_tables_with_rowwise_geography,
        write_uk_rowwise_dataset,
    )
    from microcosm.build.uk_runtime.weighted_integrity import (
        uk_input_mass_reference_payload,
    )

    result = clone_uk_dataset_tables_with_rowwise_geography(
        person=pd.DataFrame(
            {
                "person_id": [1001, 2001, 2002, 3001],
                "person_household_id": [1, 2, 2, 3],
                "person_benunit_id": [101, 201, 201, 301],
                "employment_income": [5.0, 9.0, 11.0, 7.0],
            }
        ),
        benunit=pd.DataFrame({"benunit_id": [101, 201, 301]}),
        household=pd.DataFrame(
            {
                "household_id": [1, 2, 3],
                "household_source_id": [11, 12, 13],
                "household_weight": [10.0, 20.0, 30.0],
                "region": ["LONDON", "WALES", "LONDON"],
            }
        ),
        crosswalk=pd.DataFrame(
            [
                {
                    "oa_code": "E0001",
                    "lsoa_code": "E0101",
                    "msoa_code": "E0201",
                    "la_code": "E06000063",
                    "constituency_code": "E14000001",
                    "region_code": "E12000007",
                    "country": "England",
                    "population": 100,
                },
                {
                    "oa_code": "W0001",
                    "lsoa_code": "W0101",
                    "msoa_code": "W0201",
                    "la_code": "W06000001",
                    "constituency_code": "W07000041",
                    "region_code": "W99999999",
                    "country": "Wales",
                    "population": 80,
                },
            ]
        ),
        source_year=2024,
    )
    path = write_uk_rowwise_dataset(result, tmp_path / "dense.h5")
    with pytest.raises(ValueError):
        national_frame.load_uk_national_frame(path)

    surface = uk_reference_surface_payload(
        path,
        label="microcosm-uk-2024-25-dense@uk-local-candidate-test",
        sha256=SHA,
        engine=_StubEngine({"employment_income": 0.0}),
    )
    assert surface["populated_inputs"] == ["employment_income"]
    assert "household.household_clone_index" in surface["export_columns"]
    mass = uk_input_mass_reference_payload(
        path,
        label="microcosm-uk-2024-25-dense@uk-local-candidate-test",
        filename="dense.h5",
        revision="uk-local-candidate-test",
        sha256=SHA,
    )
    assert "employment_income" in mass["totals"]
