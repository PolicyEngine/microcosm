"""The SPI Public Use Tape marker (SN 9422's Special Conditions; microcosm#1095)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from microcosm.build.uk_runtime.calibration_run import spine_provenance_from_sidecar
from microcosm.build.uk_runtime.spi_dependency import (
    UK_SPI_PUBLIC_USE_TAPE_CONDITIONS,
    UK_SPI_PUBLIC_USE_TAPE_KEY,
    uk_check_spi_public_use_tape_marker,
    uk_spi_public_use_tape_marker,
    uk_spi_public_use_tape_marker_for_stage_names,
    uk_ukds_study_number,
)


def _stage(name: str, *studies: object) -> SimpleNamespace:
    return SimpleNamespace(
        stage=name,
        artifacts=tuple(
            {"role": "tab", "kind": "private_microdata", "ukds_study_number": study}
            for study in studies
        )
        + ({"role": "anchors", "kind": "public_parameter_reference"},),
    )


def test_study_numbers_read_in_either_spelling() -> None:
    assert uk_ukds_study_number(9563) == "SN 9563"
    assert uk_ukds_study_number("SN 9422") == "SN 9422"
    assert uk_ukds_study_number(" 9422 ") == "SN 9422"
    with pytest.raises(ValueError, match="study number"):
        uk_ukds_study_number("SN")


def test_the_marker_names_the_stages_that_read_the_tape() -> None:
    marker = uk_spi_public_use_tape_marker(
        [_stage("frs_spine", 9563), _stage("spi_b", "SN 9422"), _stage("spi_a", 9422)]
    )
    assert marker == {
        "study": "SN 9422",
        "dependent": True,
        "stages": ["spi_a", "spi_b"],
        "conditions": list(UK_SPI_PUBLIC_USE_TAPE_CONDITIONS),
    }
    # A run that selects no SPI stage rests on no tape, so no condition applies.
    assert uk_spi_public_use_tape_marker([_stage("frs_spine", 9563)]) == {
        "study": "SN 9422",
        "dependent": False,
        "stages": [],
        "conditions": [],
    }


def test_the_packaged_roster_reads_the_tape_in_three_stages() -> None:
    from microcosm.build.country_spec import load_country_spec

    names = [stage.stage for stage in load_country_spec("uk").sources.stages]
    marker = uk_spi_public_use_tape_marker_for_stage_names(names)
    assert marker["dependent"] is True
    assert marker["stages"] == [
        "hmrc_spi_income_spine",
        "property_components",
        "spi_income_band_donors",
    ]
    with pytest.raises(ValueError, match="declares no stage"):
        uk_spi_public_use_tape_marker_for_stage_names([*names, "unknown_stage"])


def test_a_recorded_marker_must_agree_with_its_stages() -> None:
    derived = uk_spi_public_use_tape_marker([_stage("spi_a", "SN 9422")])
    uk_check_spi_public_use_tape_marker(None, derived)
    uk_check_spi_public_use_tape_marker({}, derived)
    uk_check_spi_public_use_tape_marker(dict(derived), derived)
    with pytest.raises(ValueError, match="disagrees"):
        uk_check_spi_public_use_tape_marker({**derived, "dependent": False}, derived)


def test_the_spine_provenance_carries_the_marker(tmp_path: Path) -> None:
    marker = uk_spi_public_use_tape_marker([_stage("spi_a", "SN 9422")])
    path = tmp_path / "spine.build.json"
    sidecar = {
        "schema_version": 2,
        "stages": ["spi_a"],
        UK_SPI_PUBLIC_USE_TAPE_KEY: marker,
    }
    path.write_text(json.dumps(sidecar))
    assert spine_provenance_from_sidecar(path, sidecar)[UK_SPI_PUBLIC_USE_TAPE_KEY] == (
        marker
    )
    # A spine built before the marker records none.
    older = {"schema_version": 2, "stages": ["spi_a"]}
    path.write_text(json.dumps(older))
    assert spine_provenance_from_sidecar(path, older)[UK_SPI_PUBLIC_USE_TAPE_KEY] == {}
