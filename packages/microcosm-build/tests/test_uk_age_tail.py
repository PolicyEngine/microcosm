"""The FRS age top-code disaggregation stage (#686, ported from the #623 campaign).

The stage exists because the licensed FRS records no age above 80, so the
85+ population targets are structurally unbindable. These tests pin the
properties that make its draw safe to run inside the spine: it is keyed on
the source identity (so clone twins agree), it is deterministic under a
declared seed, it moves nobody out of the 80+ population, and it refuses
rather than guesses when its preconditions do not hold.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microcosm.build.source_manifest import SourceManifest
from microcosm.build.uk_runtime.age_tail import (
    UK_AGE_TAIL_BANDS,
    UK_AGE_TOP_CODE,
    UKAgeTailStageTransform,
    disaggregate_uk_age_top_code,
    load_uk_age_tail_band_populations,
)
from microcosm.build.uk_runtime.national_frame import uk_national_frame
from microcosm.frame import WeightKind

REPO_ROOT = Path(__file__).resolve().parents[3]

_BANDS = {
    ("MALE", "80_84"): 815910.0,
    ("MALE", "85_89"): 459511.0,
    ("MALE", "90_plus"): 210520.0,
    ("FEMALE", "80_84"): 1024480.0,
    ("FEMALE", "85_89"): 665267.0,
    ("FEMALE", "90_plus"): 414716.0,
}


def _frame(*, n_piled: int = 40, clone_twins: bool = False):
    """A minimal UK national frame with a top-code pile."""

    ages = [30.0, 55.0] + [float(UK_AGE_TOP_CODE)] * n_piled
    genders = ["MALE", "FEMALE"] + [
        "MALE" if index % 2 == 0 else "FEMALE" for index in range(n_piled)
    ]
    count = len(ages)
    person_ids = list(range(1, count + 1))
    source_ids = [f"s{index}" for index in person_ids]
    if clone_twins:
        # A clone twin shares its original's source id *and* its sex — the
        # band CDF is sex-specific, so a twin with a different gender would
        # legitimately draw a different band.
        source_ids[-1] = source_ids[-2]
        genders[-1] = genders[-2]
    person = pd.DataFrame(
        {
            "person_id": person_ids,
            "person_benunit_id": person_ids,
            "person_household_id": person_ids,
            "person_source_id": source_ids,
            "age": ages,
            "gender": genders,
        }
    )
    benunit = pd.DataFrame({"benunit_id": person_ids})
    household = pd.DataFrame(
        {"household_id": person_ids, "household_weight": [2.0] * count}
    )
    return uk_national_frame(
        person=person,
        benunit=benunit,
        household=household,
        time_period="2024",
        weight_kind=WeightKind.DESIGN,
    )


class TestDraw:
    def test_the_pile_disperses_across_every_band(self) -> None:
        frame = _frame(n_piled=400)
        receipt = disaggregate_uk_age_top_code(frame, band_populations=_BANDS)

        ages = frame.table("person")["age"].to_numpy()
        assert receipt["piled_persons"] == 400
        # Nobody leaves the 80+ population and nobody exceeds the widest band.
        assert ages[2:].min() >= UK_AGE_TOP_CODE
        assert ages.max() <= 97
        # Every band receives someone, so the 85+ targets can bind at all.
        assert {age for age in ages[2:] if age >= 85}
        assert {age for age in ages[2:] if age >= 90}
        # Ages below the top code are untouched.
        assert list(ages[:2]) == [30.0, 55.0]

    def test_the_draw_is_deterministic_under_the_seed(self) -> None:
        first = _frame(n_piled=120)
        second = _frame(n_piled=120)
        disaggregate_uk_age_top_code(first, band_populations=_BANDS, seed=0)
        disaggregate_uk_age_top_code(second, band_populations=_BANDS, seed=0)
        assert np.array_equal(
            first.table("person")["age"].to_numpy(),
            second.table("person")["age"].to_numpy(),
        )

    def test_a_different_seed_moves_the_draw(self) -> None:
        first = _frame(n_piled=120)
        second = _frame(n_piled=120)
        disaggregate_uk_age_top_code(first, band_populations=_BANDS, seed=0)
        disaggregate_uk_age_top_code(second, band_populations=_BANDS, seed=1)
        assert not np.array_equal(
            first.table("person")["age"].to_numpy(),
            second.table("person")["age"].to_numpy(),
        )

    def test_clone_twins_receive_the_same_age(self) -> None:
        # The capital-gains clone shares its original's source id, so the two
        # rows must agree or the payload-identity discipline breaks.
        frame = _frame(n_piled=40, clone_twins=True)
        disaggregate_uk_age_top_code(frame, band_populations=_BANDS)
        person = frame.table("person")
        by_source = person.groupby("person_source_id")["age"].nunique()
        assert int(by_source.max()) == 1

    def test_the_draw_follows_the_band_populations(self) -> None:
        # A degenerate CDF that puts all mass on 90+ must send the whole pile
        # there — the shares drive the draw, not a hardcoded split.
        bands = dict.fromkeys(_BANDS, 1.0)
        bands[("MALE", "90_plus")] = 1e9
        bands[("FEMALE", "90_plus")] = 1e9
        frame = _frame(n_piled=200)
        disaggregate_uk_age_top_code(frame, band_populations=bands)
        assert frame.table("person")["age"].to_numpy()[2:].min() >= 90

    def test_the_receipt_records_what_it_did(self) -> None:
        frame = _frame(n_piled=60)
        receipt = disaggregate_uk_age_top_code(frame, band_populations=_BANDS)
        assert receipt["stage"] == "uk_age_tail_disaggregation"
        assert receipt["top_code"] == UK_AGE_TOP_CODE
        assert sum(receipt["assigned_unweighted"].values()) == 60
        assert set(receipt["achieved_weighted"]) == {"MALE", "FEMALE"}
        assert receipt["draw_key"].startswith("person_source_id")


class TestRefusals:
    def test_an_already_disaggregated_surface_is_refused(self) -> None:
        frame = _frame(n_piled=10)
        disaggregate_uk_age_top_code(frame, band_populations=_BANDS)
        with pytest.raises(ValueError, match="already has ages above"):
            disaggregate_uk_age_top_code(frame, band_populations=_BANDS)

    def test_no_pile_is_refused(self) -> None:
        frame = _frame(n_piled=0)
        with pytest.raises(ValueError, match="no persons at the top-code"):
            disaggregate_uk_age_top_code(frame, band_populations=_BANDS)

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_a_non_positive_band_population_is_refused(self, bad: float) -> None:
        bands = dict(_BANDS)
        bands[("MALE", "85_89")] = bad
        with pytest.raises(ValueError, match="missing or"):
            disaggregate_uk_age_top_code(_frame(), band_populations=bands)

    def test_a_missing_band_population_is_refused(self) -> None:
        bands = {key: value for key, value in _BANDS.items() if key[1] != "90_plus"}
        with pytest.raises(ValueError, match="missing or"):
            disaggregate_uk_age_top_code(_frame(), band_populations=bands)

    def test_an_unexpected_gender_label_is_refused(self) -> None:
        frame = _frame(n_piled=4)
        person = frame.table("person")
        person.loc[person.index[-1], "gender"] = "OTHER"
        with pytest.raises(ValueError, match="unexpected gender labels"):
            disaggregate_uk_age_top_code(frame, band_populations=_BANDS)


class TestCommittedResource:
    def test_the_committed_resource_loads_all_six_cells(self) -> None:
        populations = load_uk_age_tail_band_populations()
        assert set(populations) == set(_BANDS)
        assert all(value > 0 for value in populations.values())

    def test_every_cell_names_its_register_target(self) -> None:
        # The target id is the drift check: it is how a reader confirms the
        # imputation source and the calibration denominator are one fact.
        payload = json.loads(
            (
                REPO_ROOT
                / "packages/microcosm-build/src/microcosm/build/uk"
                / "ons_age_tail_band_populations.json"
            ).read_text(encoding="utf-8")
        )
        for gender, cells in payload["bands"].items():
            for band, cell in cells.items():
                assert cell["target_id"] == f"ons.population.{gender.lower()}_{band}"

    def test_a_cell_naming_the_wrong_target_is_refused(self, tmp_path: Path) -> None:
        payload = {
            "schema_version": 1,
            "bands": {
                gender: {
                    band: {
                        "population": _BANDS[(gender, band)],
                        "target_id": f"ons.population.{gender.lower()}_{band}",
                    }
                    for band, _, _ in UK_AGE_TAIL_BANDS
                }
                for gender in ("MALE", "FEMALE")
            },
        }
        payload["bands"]["MALE"]["85_89"]["target_id"] = "ons.population.male_80_84"
        path = tmp_path / "bands.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="drift check"):
            load_uk_age_tail_band_populations(str(path))

    def test_a_missing_band_in_the_resource_is_refused(self, tmp_path: Path) -> None:
        payload = {
            "schema_version": 1,
            "bands": {
                "MALE": {"80_84": {"population": 1.0, "target_id": "x"}},
                "FEMALE": {},
            },
        }
        path = tmp_path / "bands.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="must carry exactly"):
            load_uk_age_tail_band_populations(str(path))


class TestStageTransform:
    def _stage(self):
        manifest = SourceManifest.from_mapping(
            json.loads(
                (
                    REPO_ROOT
                    / "packages/microcosm-build/src/microcosm/build/uk"
                    / "source_stages.json"
                ).read_text(encoding="utf-8")
            )
        )
        return next(stage for stage in manifest.stages if stage.stage == "age_tail")

    def test_the_committed_stage_runs_and_receipts(self) -> None:
        transform = UKAgeTailStageTransform(stage=self._stage())
        frame = _frame(n_piled=50)
        returned = transform(frame)
        assert returned is frame
        assert transform.last_result is not None
        assert transform.last_result["piled_persons"] == 50
        assert transform.checkpoint_metadata()["evidence"] is transform.last_result

    def test_the_stage_declares_no_new_columns(self) -> None:
        # It rewrites `age`; declaring an output would claim a column the
        # manifest already attributes to frs_spine.
        assert UKAgeTailStageTransform.output_columns() == ()

    def test_parameter_drift_is_refused(self) -> None:
        stage = self._stage()
        operation = stage.operations[0]
        object.__setattr__(
            operation, "parameters", {**operation.parameters, "top_code": 75}
        )
        transform = UKAgeTailStageTransform(stage=stage)
        with pytest.raises(ValueError, match="parameter drift"):
            transform(_frame())

    def test_an_undeclared_parameter_is_refused(self) -> None:
        stage = self._stage()
        operation = stage.operations[0]
        object.__setattr__(
            operation,
            "parameters",
            {**operation.parameters, "unimplemented_knob": 1},
        )
        transform = UKAgeTailStageTransform(stage=stage)
        with pytest.raises(ValueError, match="does not implement"):
            transform(_frame())
